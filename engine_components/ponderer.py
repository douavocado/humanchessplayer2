"""Opponent-time pondering, extracted verbatim from engine.Engine.

own_time_or_none / stockfish_ponder / ponder / ponder_moves /
recursive_ponder / re_evaluate together form the pondering subsystem:
while it isn't our turn, pre-compute responses to the opponent's
plausible replies so a recognised move can be answered instantly.
`ponder` is the human-filtered path (NN move probabilities narrow the
search width, then `re_evaluate`/`recursive_ponder` deepen a subset of
lines under a time/eval budget); `stockfish_ponder` is the faster,
filter-free alternative.

Ninth and final slice of the engine.py strangler-fig extraction (see
testing/engine_parity/ for the regression harness). This was the
hardest module to port faithfully in the old abandoned tree -- a real
transcription bug there (`_recursive_ponder` pushing both the candidate
move and the predicted response before recursing, doubling the
effective ply depth per recursion step) plus several structural
divergences (ponder width thresholds, missing noise-scale factor,
missing max-depth cap, a different time-budget split across branches)
never caught up with engine.py's evolution. Verbatim move
(self -> engine) here, no interface redesign -- the safety net for this
module is exactly the point of doing it this way instead of a rewrite.
"""
import random
import time

import numpy as np

import chess
import chess.engine

from common.board_information import phase_of_game, calculate_threatened_levels
from common.utils import extend_mate_score
from common.search_constants import (
    MAX_CALC_DEPTH_COEFF, REEVAL_ORDER, PONDER_MIN_ROOT_MOVES,
)


def own_time_or_none(engine):
    """ Our latest clock reading, or None before the first update. """
    times = engine.input_info.get("self_clock_times") if engine.input_info else None
    return times[-1] if times else None


def ponder_moves(engine, board, move_ucis, search_width, prev_board=None, log=True, use_ponder=False):
    """ We ponder on the given board position, and consider the moves given by the list
        of move_ucis. We again use human probabilities to narrow our search width.

        Returns a dictionary with:
            key: the move uci from move_ucis
            value: [move uci of response, eval of response]
        eval of response is given from the perspective of ourselves.
    """
    if log:
        engine.log += "Pondering the moves {} for the fen {} \n".format(move_ucis, board.fen())

    return_dic = {}
    for move_uci in move_ucis:
        dummy_board = board.copy()
        move_obj = chess.Move.from_uci(move_uci)
        dummy_board.push(move_obj)

        # First check that the game has not ended after this move
        outcome = dummy_board.outcome()
        if outcome is not None:
            # Then game has ended, return corresonding eval
            winner = outcome.winner
            if winner is None:
                return_dic[move_uci] = [None, 0]
            elif winner == engine.input_info["side"]:
                return_dic[move_uci] = [None, extend_mate_score(2500)]
            elif winner == (not engine.input_info["side"]):
                return_dic[move_uci] = [None, -2500]
            else:
                raise Exception("Unrecgonised outcome winner: {}".format(winner))
            continue

        # Now get human probabilities of this new position
        game_phase = phase_of_game(dummy_board)
        un_altered_move_dic = engine.get_human_probabilities(dummy_board, game_phase, log=False)
        # if we discover too few moves from getting human probabilities, then
        # instead the root moves will be just the legal moves instead
        if len(un_altered_move_dic) <= 2:
            root_moves = list(dummy_board.legal_moves)
        else:
            if prev_board is not None:
                prev_prev_board = prev_board.copy()
            else:
                prev_prev_board = None

            # altered_move_dic = engine._alter_move_probabilties(un_altered_move_dic, board=dummy_board, prev_board=board.copy(), prev_prev_board=prev_prev_board, log=False)
            altered_move_dic = engine._alter_move_prob_nn(un_altered_move_dic, board=dummy_board, prev_board=board.copy(), prev_prev_board=prev_prev_board, log=False)

            human_move_ucis = list(altered_move_dic.keys())

            root_moves = [chess.Move.from_uci(x) for x in human_move_ucis[:search_width]]

        if use_ponder == True:
            single_analysis = engine.ponder_stockfish_engine.analyse(dummy_board, chess.engine.Limit(time=0.05), root_moves=root_moves)
        else:
            single_analysis = engine.stockfish_engine.analyse(dummy_board, chess.engine.Limit(time=0.02), root_moves=root_moves)
        if "pv" in single_analysis:
            response = single_analysis["pv"][0].uci()
            eval_ = extend_mate_score(single_analysis['score'].pov(engine.input_info["side"]).score(mate_score=2500))
        else:
            engine.log += "ERROR: KeyError pv not in analysis object {}. Returning random response as well as retaining current_eval. \n".format(single_analysis)
            response = root_moves[0].uci() # pick first one
            eval_ = extend_mate_score(engine.stockfish_analysis[0]["score"].pov(engine.input_info["side"]).score(mate_score=2500))

        return_dic[move_uci] = [response, eval_]

    if log:
        engine.log += "Returning ponder results: {} \n".format(return_dic)
    return return_dic


def recursive_ponder(engine, board, move_uci, no_root_moves, depth, prev_board=None, limit=None, use_ponder=False):
    """ Recursive function for getting evaluations during pondering.

        Returns [move_uci eval, depth_considered]
        If limit is None, then depth_considered is None. If there are time limit
        constraints, then depth_considered is the depth that move_uci has been considered.
    """
    start = time.time()
    ponder_results = engine._ponder_moves(board, [move_uci], no_root_moves, prev_board=prev_board, log=False, use_ponder=use_ponder)
    end = time.time()
    # engine.log += "Ponder position fen {} with move {} at depth {} took {} seconds to calculate. \n".format(board.fen(), board.san(chess.Move.from_uci(move_uci)), depth, end-start)
    if limit is not None:
        re_evaluations_left, time_left, depth_considered, total_depth, comparison_eval = limit
        # engine.log += "We have {} time left to calculate {} variations. \n".format(time_left - (end-start), re_evaluations_left-1)
        if depth > 1:
            # then we have time constraint
            # we shall adaptively
            new_time_left = time_left - (end-start)
            if new_time_left <= 0.07: # no time left for one more
                # return result
                return [ponder_results[move_uci][1], depth_considered]

            # otherwise, forecast whether we are on track for finish or not
            new_re_evaluations_left = re_evaluations_left - 1
            # pretend that the rest of the evaluations will take end-start
            forecast_evaluations_left = new_time_left / (end-start)
            new_board = board.copy()
            new_board.push_uci(move_uci)
            consider_move = ponder_results[move_uci][0]
            if consider_move is not None:
                # we are running out of time, OR if the line does/doesn't seem that promising compared to comparison eval
                if ponder_results[move_uci][1] > comparison_eval + 100:
                    # proceed as usual, because line is quite promising
                    return engine._recursive_ponder(new_board, consider_move, no_root_moves, depth-1, prev_board=board.copy(), limit=[new_re_evaluations_left, new_time_left, depth_considered + 1, total_depth, comparison_eval], use_ponder=use_ponder)
                elif forecast_evaluations_left < new_re_evaluations_left - 1:
                    # then proceed onto next jump 1 depth less
                    if depth == 2:
                        return [ponder_results[move_uci][1], depth_considered]
                    else:
                        return engine._recursive_ponder(new_board, consider_move, no_root_moves, depth-2, prev_board=board.copy(), limit=[new_re_evaluations_left-1, new_time_left, depth_considered+1, total_depth, comparison_eval], use_ponder=use_ponder)
                elif ponder_results[move_uci][1] < comparison_eval - 250:
                    # variation not promising enough. Stop variation here
                    return [ponder_results[move_uci][1], depth_considered]
                else:
                    # continue as usual
                    return engine._recursive_ponder(new_board, consider_move, no_root_moves, depth-1, prev_board=board.copy(), limit=[new_re_evaluations_left, new_time_left, depth_considered + 1, total_depth, comparison_eval], use_ponder=use_ponder)
            else: # depth <=1 and limit is not None
                return [ponder_results[move_uci][1], depth_considered]
        else:
            return [ponder_results[move_uci][1], depth_considered]
    else: # not using limits
        if depth > 1:
            new_board = board.copy()
            new_board.push_uci(move_uci)
            consider_move = ponder_results[move_uci][0]
            if consider_move is not None:
                # if we actually have a valid move
                # sometimes the game has already ended at this point
                return engine._recursive_ponder(new_board, consider_move, no_root_moves, depth-1, prev_board=board.copy(), use_ponder=use_ponder)
            else:
                return [ponder_results[move_uci][1], None]
        else:
            return [ponder_results[move_uci][1], None]


def reeval_sequence(engine, moves):
    """ The order re_evaluate actually searches candidates in.

        This is the order the budget is spent in, and running out mid-list
        leaves the rest at depth_considered 0 and a ~60cp penalty, which
        REEVAL_ORDER's own notes call effective disqualification. So the
        order *is* the knob: callers hand the candidates over already sorted
        the way reeval_order asked for (human plausibility, or eval), and
        that sorting has to survive to here.

        It used to not. re_evaluate opened with an unconditional
        random.shuffle "to avoid bias", which put the disqualification draw
        back on a uniform lottery and left reeval_order controlling only
        which moves made the shortlist, never which of them got looked at
        first. Seen live 2026-08-22: Nxa1 won the position's human prior at
        p=0.888, was shortlisted first, shuffled to last, and ran out of
        budget - it went into the comparison at depth 0 against a rival's
        depth-4 eval and lost by the penalty, costing ~1.1 pawn.

        "random" is the one setting that wants the lottery, so it still
        shuffles; it is what that setting selects, not an accident.

        Returns a new list - the caller's own ordering is left intact, so a
        log line built before the call still describes what was searched.
    """
    ordered = list(moves)
    if getattr(engine, "reeval_order", REEVAL_ORDER) == "random":
        random.shuffle(ordered)
    return ordered


def re_evaluate(engine, board, re_evaluate_moves, no_root_moves, depth=0, prev_board=None, limit=None, use_ponder=False):
    """ Given a list of move_ucis, apply them to the current board and re_evaluate
        them using top human_moves only. This gives a non_accurate evaluation
        and simulates human foresight not being exhaustive.

        Returns a dictionary with key move_uci and value the evaluation (from our pov)
    """
    re_evaluate_moves = reeval_sequence(engine, re_evaluate_moves)
    return_dic = {}
    if limit is None:
        for move_uci in re_evaluate_moves:
            eval_, _ = engine._recursive_ponder(board, move_uci, no_root_moves, depth, prev_board=prev_board, use_ponder=use_ponder)
            return_dic[move_uci] = eval_
    else:
        comparison_eval = -9999
        evaluations_left, time_left = limit
        time_allowed = time_left
        start = time.time()
        for move_uci in re_evaluate_moves:
            if time_allowed <= 0.07:
                # that's it, no more time left
                return_dic[move_uci] = [None, 0]
                continue
            eval_, depth_considered = engine._recursive_ponder(board, move_uci, no_root_moves, depth, prev_board=prev_board, limit=[evaluations_left, time_allowed, 1, depth, comparison_eval], use_ponder=use_ponder)
            return_dic[move_uci] = [eval_, depth_considered]
            comparison_eval = max(eval_, comparison_eval)
            evaluations_left -= depth
            time_allowed = time_left - (time.time() - start)

    return return_dic


def stockfish_ponder(engine, board, time_allowed, ponder_width, use_ponder=False, root_moves=None):
    """ Given a board position that is not our side turn, we ponder moves using
        stockfish and return a dictionary with has key board_fen and value uci.
        This method is much faster than self.ponder()

    """
    engine.log += "Stockfish ponder position with fen {} \n".format(board.fen())
    engine.log += "Stockfish pondering time allowed: {} \n".format(time_allowed)

    # First double check our position has not ended. If it has no moves, return None
    if board.outcome() is not None:
        return None

    if root_moves is None:
        root_moves = list(board.legal_moves)

    if use_ponder:
        analysis_object = engine.ponder_stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=time_allowed), multipv=ponder_width, root_moves=root_moves)
    else:
        analysis_object = engine.stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=time_allowed), multipv=ponder_width, root_moves=root_moves)
    if isinstance(analysis_object, dict):
        analysis_object = [analysis_object]

    return_dic = {}
    for line in analysis_object:
        if "pv" in line:
            if len(line["pv"]) >=2:
                dummy_board = board.copy()
                opp_move = line["pv"][0]
                dummy_board.push(opp_move)
                board_fen = dummy_board.board_fen()
                response_move_uci = line["pv"][1].uci()
                return_dic[board_fen] = response_move_uci

    if len(return_dic) == 0:
        return None

    engine.log += "Stockfish ponder moves returned ponder dic: {} \n".format(return_dic)
    return return_dic


def ponder(engine, board, time_allowed, search_width, time_per_position=0.1, prev_board=None, ponder_width=None, use_ponder=False):
    """ Given a board position that is not our (side) turn, we ponder possible moves
        and return a dictionary which has key board_fen (so position only), and value uci in response.
        Time allowed represents how much we may ponder. Time per position is roughly
        the amount of time to form probabilies and alter them.

        Returns a dictionary with key board_fen and value move_uci response
    """
    engine.log += "Ponder position with fen {} \n".format(board.fen())
    engine.log += "Pondering time allowed: {} \n".format(time_allowed)
    variations_allowed = max(1, int(time_allowed/time_per_position))

    # First double check our position has not ended. If it has no moves, return None
    if board.outcome() is not None:
        return None

    max_depth = int(MAX_CALC_DEPTH_COEFF * search_width ** 0.5) # this is the maximum depth we will consider for ponder, as for too deep ponder we increase the chance of silly moves.

    if ponder_width is None:
        # As a maximum number of ponder moves (to prevent too quick responses),
        # per-game coverage draw when available (see _sample_game_character),
        # else by time control.
        initial_time = engine.input_info["self_initial_time"]
        if engine.game_ponder_width is not None:
            max_ponder_no = engine.game_ponder_width
            if max_ponder_no <= 0:
                # A no-ponder game (see GAME_PONDER_WIDTH_CLIP): this
                # game never prepares replies at all.
                engine.log += "Ponder width 0 this game: skipping ponder. \n"
                return None
        elif initial_time <= 180:
            max_ponder_no = 3
        else:
            max_ponder_no = 4

        # decide ponder depth and width
        ponder_width = 1 # min ponder width
        ponder_depth = 1  # always overwritten below (max_ponder_no >= 1 here); default keeps the static type checker happy
        for i in range(max_ponder_no):
            ponder_depth = round(variations_allowed / ((max_ponder_no-i) * search_width))
            if ponder_depth >= 2:
                ponder_width = max_ponder_no-i
                break
    else:
        # ponder width has been preset
        ponder_depth = round(variations_allowed / (ponder_width * search_width))

    ponder_depth = min(ponder_depth, max_depth)
    root_width = max(PONDER_MIN_ROOT_MOVES, search_width)

    if use_ponder:

        analysis_object = engine.ponder_stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=0.02), multipv=ponder_width)

    else:
        analysis_object = engine.stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=0.02), multipv=ponder_width)
    if isinstance(analysis_object, dict):
        analysis_object = [analysis_object]

    if len(analysis_object) == 0:
        engine.log += "ERROR: Couldn't fetch stockfish analysis object, returning None. \n"
        return None
    elif "pv" not in analysis_object[0]:
        engine.log += "ERROR: pv KeyError in stockfish analysis object: {} \n".format(analysis_object)
        return None
    opp_moves_considered = [entry["pv"][0].uci() for entry in analysis_object]
    san_opp_moves_considered = [board.san(chess.Move.from_uci(x)) for x in opp_moves_considered]
    engine.log += "Considering with ponder depth {}, moves in this position: {} \n".format(ponder_depth, san_opp_moves_considered)

    # For noise dictionary. We implement different levels of noise depending on the stage of the game
    noise_dic = {"opening": 0.8,
                 "midgame": 1.2,
                 "endgame": 0.3,}


    # Now execute or recursive function to find response dictionary
    return_dic = {}
    san_return_dic = {}
    for move_uci in opp_moves_considered:
        dummy_board = board.copy()
        dummy_board.push_uci(move_uci)
        # first check that the game is not over
        if dummy_board.outcome() is not None:
            # then game is over
            # we can't respond to this
            continue
        board_fen = dummy_board.board_fen()
        game_phase = phase_of_game(dummy_board)
        top_human_move_dic = engine.get_human_probabilities(dummy_board, game_phase, log=False)
        if len(top_human_move_dic) <= 2:
            # No usable plausibility ranking here, and this branch does not
            # narrow to search_width either, so the tail of the list is the
            # part the budget will disqualify. Legal-move generation order
            # would hand that verdict to the board's square ordering, so
            # draw for it instead - reeval_sequence can only preserve an
            # ordering, it cannot invent one.
            top_human_moves = [move.uci() for move in dummy_board.legal_moves]
            random.shuffle(top_human_moves)
        else:
            # top_human_move_dic = engine._alter_move_probabilties(top_human_move_dic, dummy_board, prev_board = board, prev_prev_board=prev_board, log= False)
            top_human_move_dic = engine._alter_move_prob_nn(top_human_move_dic, dummy_board, prev_board = board, prev_prev_board=prev_board, log= False)
            # Never shortlist a single move: with one candidate the argmax is
            # that candidate, so re_evaluate's Stockfish call can only hand it
            # straight back and the cached reply is the NN's top pick with no
            # second opinion at all. See PONDER_MIN_ROOT_MOVES for the two
            # logged losses this caused. The floor applies to OUR candidates
            # only -- search_width still goes to re_evaluate as no_root_moves,
            # which narrows the opponent's replies and is deliberate.
            top_human_moves = sorted(top_human_move_dic.keys(), key=lambda x: top_human_move_dic[x], reverse=True)[:root_width]

        # Re-evaluation budget counts our candidates, so it tracks root_width
        # rather than search_width or the extra candidate cannot be reached.
        re_evaluate_dic = engine._re_evaluate(dummy_board, top_human_moves, search_width, depth=ponder_depth, prev_board = board.copy(), limit=[ponder_depth*root_width, time_allowed/2], use_ponder=use_ponder)
        # adding noise
        noise_phase = noise_dic[game_phase]
        for move_uci in re_evaluate_dic.keys():
            eval_, depth_considered = re_evaluate_dic[move_uci]
            if eval_ is  None:
                # move never got considered
                # Get stockfish evaluation of move, but penalise heavily
                if use_ponder:
                    an_obj = engine.ponder_stockfish_engine.analyse(dummy_board, limit=chess.engine.Limit(depth=8, time=0.02), root_moves=[chess.Move.from_uci(move_uci)])
                    if "score" in an_obj:
                        new_eval = extend_mate_score(an_obj["score"].pov(engine.input_info["side"]).score(mate_score=2500))
                    else:
                        # something went wrong
                        engine.log += "Something went wrong with analysis object with use_ponder: {}. Returning no ponder dic. \n".format(an_obj)
                        return None
                else:
                    an_obj = engine.stockfish_engine.analyse(dummy_board, limit=chess.engine.Limit(depth=8, time=0.02), root_moves=[chess.Move.from_uci(move_uci)])
                    if "score" in an_obj:
                        new_eval = extend_mate_score(an_obj["score"].pov(engine.input_info["side"]).score(mate_score=2500))
                    else:
                        # something went wrong
                        engine.log += "Something went wrong with analysis object: {}. Returning no ponder dic. \n".format(an_obj)
                        return None
                eval_ = new_eval - 100
                re_evaluate_dic[move_uci][0] = new_eval - 100 # penalty
            base_noise_sd = 40*(np.tanh(eval_/(engine.playing_level*50)))**2 + 20
            noise_sd = engine.eval_noise_scale*4*base_noise_sd/(time_allowed*(depth_considered+4))

            noise = np.random.randn()*noise_sd*noise_phase
            re_evaluate_dic[move_uci][0] += noise
            # encourage capture moves, depending on how enpris the piece is
            move_obj = chess.Move.from_uci(move_uci)
            if dummy_board.is_capture(move_obj):
                capture_bonus = 40* calculate_threatened_levels(move_obj.to_square, dummy_board)
                re_evaluate_dic[move_uci][0] += capture_bonus

        best_response = max(re_evaluate_dic.keys(), key=lambda x : re_evaluate_dic[x][0])
        return_dic[board_fen] = best_response
        san_return_dic[board_fen] = dummy_board.san(chess.Move.from_uci(best_response))
    engine.log += "Computed responses for these moves: {} \n".format(san_return_dic)

    if len(return_dic) == 0:
        return None
    else:
        return return_dic
