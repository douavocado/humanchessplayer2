"""Fast, filter-free move selection extracted verbatim from engine.Engine.

get_stockfish_move is the "no human filters" path: strictly grades
Stockfish's candidate moves by eval and hand-distance, with no NN move-prob
step. Used when there isn't time to run the full human-filtered pipeline
(deep scrambles) -- includes the flag-race autopilot (eval cap + blind-move
chance keyed off the per-game game_scramble_skill draw) that reproduces
the human catastrophe tail of missed/thrown mates under a flagging clock.

Fourth slice of the engine.py strangler-fig extraction (see
testing/engine_parity/ for the regression harness). Verbatim move
(self -> engine), no interface redesign.
"""
import random

import numpy as np

import chess

from common.utils import extend_mate_score
from common.board_information import move_progress_score, seen_position_keys
from common.constants import (
    FLAG_RACE_TIME, FLAG_RACE_EVAL_CAP,
    FLAG_RACE_CAP_MIN, FLAG_RACE_CAP_MAX, FLAG_RACE_BLIND_P_MAX,
    PROGRESS_MIN_EVAL, PROGRESS_EVAL_TOLERANCE,
)


def get_stockfish_move(engine, board=None, analysis=None, last_move_uci=None, log=True):
    """ Uses board information to get a move strictly from stockfish with no human
        filters. Very fast, and only called for in necessary situations (when in
        super low time)

        Returns a move_uci string of move made
    """
    # If no kwargs given, then assume we are using the current position
    if board is None:
        board = engine.current_board.copy()

    if analysis is None :
        analysis = engine.stockfish_analysis

    if last_move_uci is None:
        if len(engine.input_info["last_moves"]) >= 2:
            last_move_uci = engine.input_info["last_moves"][-2]
    # take a random sample from the moves given by our stockfish analysis object
    # grade each move by their evaluation, and if previous moves are given (our
    # own moves) then grade them also based on proximity and distance moved by mouse
    total_moves = len(analysis)
    sample_n = max(int(total_moves*0.6),1)
    if log:
        engine.log += "Choosing to sample {} moves from analysis object. \n".format(sample_n)
    sampled_moves = random.sample(analysis, sample_n)
    move_eval_dic = {entry["pv"][0].uci(): extend_mate_score(entry["score"].pov(board.turn).score(mate_score=2500)) for entry in sampled_moves}
    # if we are winning by a big margin such that we have mate in less than 10 moves, and we have sufficient time, than with large probability we play the
    # zeroing moves. This helps us out in closing games
    own_time = max(engine.input_info["self_clock_times"][-1],1)
    opp_time = max(engine.input_info["opp_clock_times"][-1],1)
    top_engine_move = max(move_eval_dic.keys(), key= lambda x: move_eval_dic[x])
    if opp_time > own_time and max(move_eval_dic.values()) >= 2490 and engine.mood == "hurry": #mate in less than 15 and we are under time pressure
        # Humans dash out a spotted mate in a flag race -- but far from
        # always (see FLAG_RACE_* in constants: throwing the won endgame
        # is a signature human catastrophe the analyser looks for).
        if np.random.random() < 0.4:
            if log:
                engine.log += "We have sufficient time ({}) and we have spotted mate in {}. Playing top engine move to close the game. \n".format(own_time, 2500-move_eval_dic[top_engine_move])
            return top_engine_move
        else:
            if log:
                engine.log += "We spotted mate in {}, but by chance we don't go for top move. \n".format(2500-move_eval_dic[top_engine_move])
    else:
        no_pieces = len(chess.SquareSet(board.occupied))
        if no_pieces < 10 and engine.mood == "hurry": # less than 10 pieces including kings
            # with some probability play best move
            if np.random.random() < 0.15:
                if log:
                    engine.log += "Less than 10 pieces on the board, playing top engine move to close the game. \n"
                return top_engine_move
            else:
                if log:
                    engine.log += "Less than 10 pieces on the board, but by chance not playing top engine move. \n"
    if log:
        engine.log += "Sampled the following moves and their corresponding evals: {} \n".format(move_eval_dic)
    move_distance_dic = {}
    for move_uci in move_eval_dic.keys():
        distance = 0
        move_obj = chess.Move.from_uci(move_uci)
        # if we are given information about our own previous move, then include that distance too
        # we weight the move getting to our next piece than the distance travelled by that distance
        if last_move_uci is not None:
            own_last_move_obj = chess.Move.from_uci(last_move_uci)
            to_square = own_last_move_obj.to_square
            distance += chess.square_distance(to_square, move_obj.from_square)
        distance += 0.5*chess.square_distance(move_obj.from_square, move_obj.to_square)
        move_distance_dic[move_uci] = distance

    if log:
        engine.log += "Evaluated the moves square distance to move: {} \n".format(move_distance_dic)

    # Conversion progress. In a decided position every candidate's eval reads
    # "winning", so the eval term below goes flat and the ranking collapses
    # onto hand distance -- whose cheapest option is putting the piece we
    # just moved straight back. See PROGRESS_* in constants for the measured
    # numbers. Only when we are clearly winning: shuffling and repeating are
    # perfectly human when we are worse.
    progress_scores = {}
    if max(move_eval_dic.values()) >= PROGRESS_MIN_EVAL:
        # board.turn is our side at both call sites (live position, and the
        # premove path's board-after-our-move-and-their-reply), so the move
        # two plies back on the stack is our own last move -- unlike the
        # last_move_uci argument, which the premove path fills with the
        # *opponent's* predicted move for hand-distance purposes.
        own_last_move = board.move_stack[-2] if len(board.move_stack) >= 2 else None
        seen_keys = seen_position_keys(board)
        progress_scores = {
            move_uci: move_progress_score(board, chess.Move.from_uci(move_uci),
                                          seen_keys=seen_keys, own_last_move=own_last_move)
            for move_uci in move_eval_dic.keys()
        }

    own_time = max(engine.input_info["self_clock_times"][-1],1)
    # Flag-race autopilot: in a deep scramble a human reads "+mate" and
    # "+800" both as "winning" and picks on instinct (distance + noise),
    # occasionally missing the mate or throwing the win. Capping the eval
    # term reproduces that; without it a mate (2500) outweighs the noise
    # so completely that the bot never produces the human catastrophe tail.
    # The cap and the blind-move chance follow this game's scramble skill
    # (see _sample_game_character and FLAG_RACE_* in constants): most
    # games scramble cleanly, a minority blunder-prone, and a blind move
    # drops the eval term entirely -- pure hand-distance + noise -- which
    # is how a won ending actually gets thrown.
    if own_time < FLAG_RACE_TIME:
        skill = engine.game_scramble_skill
        if skill is None:
            eval_cap = FLAG_RACE_EVAL_CAP
            blind_p = 0.0
        else:
            eval_cap = FLAG_RACE_CAP_MIN + skill * (FLAG_RACE_CAP_MAX - FLAG_RACE_CAP_MIN)
            blind_p = FLAG_RACE_BLIND_P_MAX * (1 - skill) ** 2
        if np.random.random() < blind_p:
            appeal_eval_dic = {m: 0 for m in move_eval_dic}
            if log:
                engine.log += "Scramble blind move (skill {:.2f}, p {:.3f}): dropping evals, picking on hand distance and noise alone. \n".format(skill, blind_p)
        else:
            appeal_eval_dic = {m: min(v, eval_cap) for m, v in move_eval_dic.items()}
            if log:
                engine.log += "Scramble eval cap for this game: {:.0f} (skill {}). \n".format(
                    eval_cap, "unsampled" if skill is None else "{:.2f}".format(skill))
    else:
        appeal_eval_dic = move_eval_dic

    # Net the progress terms against the evals we can actually perceive.
    # The shuffle penalty is unconditional; the plan bonus is only offered
    # to moves that still look best under this scramble's own perception
    # (the capped evals above, or none at all on a blind move), so "push the
    # pawn" can break a tie between moves that all read as winning but can't
    # override a drop the player can see. Without that gate the plan bonus
    # cost real games -- an A/B over 30 endgame scrambles per arm doubled
    # the median eval loss in pawn endings, because the bot pushed pawns it
    # could see were losing. On a blind move every eval is 0, so every move
    # is "still best" and the plan runs unchecked -- which is exactly how a
    # human throws a won ending on autopilot.
    move_progress_dic = {move_uci: 0.0 for move_uci in move_eval_dic.keys()}
    if progress_scores:
        best_appeal_eval = max(appeal_eval_dic.values())
        for move_uci, (bonus, penalty) in progress_scores.items():
            if appeal_eval_dic[move_uci] < best_appeal_eval - PROGRESS_EVAL_TOLERANCE:
                bonus = 0.0
            move_progress_dic[move_uci] = bonus - penalty
        if log:
            engine.log += "Winning position, scoring conversion progress per move: {} \n".format(
                {m: round(v, 2) for m, v in move_progress_dic.items() if v != 0.0})

    move_appealing_dic = {move_uci : 10 + appeal_eval_dic[move_uci]*(own_time+5)/2000 - move_distance_dic[move_uci] + move_progress_dic[move_uci] for move_uci in move_eval_dic.keys()}
    if log:
        engine.log += "Combining both the dictionary preferences, we have their move preferences: {} \n".format(move_appealing_dic)
    # Add noise to introduce randomness. The lower our time, the more the noise
    # note own time has to be less than 15 seconds for valid calculation.
    noise_level = (15-own_time)/15
    move_appealing_dic = {move_uci: move_appealing_dic[move_uci] + noise_level*np.random.randn() for move_uci in move_appealing_dic.keys()}
    if log:
        engine.log += "Appealingness after adding noise: {} \n".format(move_appealing_dic)

    move_chosen = max(move_appealing_dic.keys(), key=lambda x: move_appealing_dic[x])
    if log:
        engine.log += "Chosen stockfish move under time pressure: {} \n".format(move_chosen)
    return move_chosen
