"""Read-mostly decision helpers extracted from engine.Engine.

Each function is a verbatim move of the corresponding Engine method body
(self -> engine, no other changes): they read engine.input_info /
engine.current_board / engine.lucas_analytics / engine.stockfish_analysis
and return a decision, without mutating per-game state. This is the first
slice of the engine.py strangler-fig extraction (see
testing/engine_parity/ for the regression harness these are verified
against) -- the safest possible first cut, since none of the three touch
cross-cutting state like the per-game character draws.
"""
import numpy as np

import chess
import chess.engine

from common.board_information import is_takeback, calculate_threatened_levels, PIECE_VALS


def decide_resign(engine):
    """ The decision making function which decides whether to resign.

        Returns bool
    """
    engine.log += "Deciding whether to resign the current position. \n"
    # Can only decide to resign if analytics are updated
    if engine.analytics_updated == False:
        engine.log += "WARNING: _decide_resign() function has been called with outdated analytics. Please call .calculate_analytics() first to get accurate output. \n"

    own_time = max(engine.input_info["self_clock_times"][-1],1)
    opp_time = max(engine.input_info["opp_clock_times"][-1],1)
    starting_time = engine.input_info["self_initial_time"]

    # does not resign in the first 15 moves
    if engine.current_board.fullmove_number < 15:
        return False
    elif opp_time <= 10:
        return False
    elif own_time/(opp_time-10) > 3:
        return False
    win_pct = engine.lucas_analytics["win_prob"]

    # We shall only resign if our win percentage is sufficiently low, and the material
    # imbalance is obvious
    mat_dic = {1:1, 2:3.1, 3:3.5, 4:5.5, 5:9.9, 6:3}
    our_mat = sum([len(engine.current_board.pieces(x, engine.input_info["side"]))*mat_dic[x] for x in range(1,7)])
    opp_mat = sum([len(engine.current_board.pieces(x, not engine.input_info["side"]))*mat_dic[x] for x in range(1,7)])
    if win_pct < np.log(starting_time)* opp_time/(500*starting_time) and opp_mat - our_mat > 5:
        engine.log += "Win percentage too low, material imbalance too large and opponent time too high, resigning... \n"
        return True
    else:
        return False


def decide_human_filters(engine):
    """ Decide if given the position we want to use human filters or not. When
        using human filters we would need to use more processing time, which
        may not be suitable when we have low time. If possible however, we always
        want to use human filters.

        Returns True/False
    """

    # In time trouble, increasingly skip the human filters: the fast
    # stockfish path (get_stockfish_move) is both quicker and carries the
    # flag-race autopilot fallibility -- humans in a scramble play on
    # instinct, they don't run their full decision process. Denominator
    # sets the routing share (14: ~64% at 1s, ~36% at 5s).
    own_time = max(engine.input_info["self_clock_times"][-1],1)
    if own_time < 10:
        if np.random.random() < (10-own_time)/14:
            return False
        else:
            return True
    return True


def check_obvious_move(engine):
    """ Given input information, check whether there is an obvious move in the
        position that we may play immediately.

        Returns [obvious_move: uci_str, obvious_move_found : bool]
        In the case that no obvious is found, obvious_move is None"""
    # We shall define obvious moves as the following:
    # 1. Only moves
    # 2. Takebacks
    # 3. Forced mate in ones
    engine.opponent_just_blundered = False

    # Check for only legal moves
    if len(engine.stockfish_analysis) == 1:
        engine.log += "Found only legal move {}, so playing it as obvious move. \n".format(engine.stockfish_analysis[0]["pv"][0].uci())
        return engine.stockfish_analysis[0]["pv"][0].uci(), True

    # Check for takebacks. These moves are takebacks that are both the best move
    # and is also a takeback. If the takeback is not by far the best move (>100 cp)
    # then with some probability return no.
    if len(engine.input_info["last_moves"]) >= 1:
        prev_board = chess.Board(engine.input_info["fens"][-2])
        last_move_uci = engine.input_info["last_moves"][-1]
        best_move_uci = engine.stockfish_analysis[0]["pv"][0].uci()
        cp_diff = engine.stockfish_analysis[0]["score"].pov(engine.input_info["side"]).score(mate_score=2500) - engine.stockfish_analysis[1]["score"].pov(engine.input_info["side"]).score(mate_score=2500)
        if is_takeback(prev_board, last_move_uci, best_move_uci):
            engine.log += "Detected top move {} is a takeback. \n".format(best_move_uci)
            # check if the takeback is a blunder, if it is then capturing it is not an obvious moved
            last_move_obj = chess.Move.from_uci(last_move_uci)
            # to_square is the takeback's capture target, so both boards have
            # a real piece there (the invariant piece_type_at's Optional
            # return can't express).
            current_piece_type = engine.current_board.piece_type_at(last_move_obj.to_square)
            prev_piece_type = prev_board.piece_type_at(last_move_obj.to_square)
            assert current_piece_type is not None and prev_piece_type is not None
            if calculate_threatened_levels(last_move_obj.to_square, prev_board)  <= 0 and PIECE_VALS[current_piece_type] - PIECE_VALS[prev_piece_type] > 0.6:
                # then it is blunder
                engine.log += "Opponent has captured a piece that was not enpris: {}, not considering it as takeback. \n".format(prev_board.san(last_move_obj))
                engine.opponent_just_blundered = True
            else:
                if cp_diff > 100:
                    engine.log += "Top move is also by far best option, returning as obvious move. \n"
                    return best_move_uci, True
                else:
                    if np.random.random() < 0.7:
                        engine.log += "Top move is not by far best, but by chance we still return it as obvious move. \n"
                        return best_move_uci, True
                    else:
                        engine.log += "Top move is not by far the best, by chance we don't return it as obvious move. \n"

    # Check for forced mate in ones
    # We only return obvious move if previously it was mate in 2 and no matter what
    # enemy plays our obvious move would have been mate in one
    if len(engine.input_info["fens"]) >= 2 and engine.stockfish_analysis[0]["score"].pov(engine.input_info["side"]).mate() == 1:
        last_board = chess.Board(engine.input_info["fens"][-2])
        last_analysis = engine.stockfish_engine.analyse(last_board, limit=chess.engine.Limit(depth=6, time=0.02,mate=1))
        last_mate = last_analysis["score"].pov(engine.input_info["side"]).mate()
        if last_mate == 2:
            last_mate_move_uci = last_analysis["pv"][1].uci()
            if engine.stockfish_analysis[0]["pv"][0].uci() == last_mate_move_uci:
                engine.log += "Found obvious move that no matter what gives mate next move: {} \n". format(last_mate_move_uci)
                return last_mate_move_uci, True

    # Otherwise no obvious move
    engine.log += "No obvious move found. \n"
    return None, False
