"""Mood/blunder-reaction heuristics, extracted verbatim from engine.Engine.

set_mood decides the human "mood" label (confident/cocky/cautious/tilted/
hurry/flagging) that other components (pacing, breadth) read off
engine.mood. check_opp_blunder is the separate "did the opponent just hang
something" detector that drives startled reactions. Deliberately kept
independent of engine analytics like lucas/sharpness where possible -- see
the set_mood docstring -- so it's mostly time/eval heuristics.

Third slice of the engine.py strangler-fig extraction (see
testing/engine_parity/ for the regression harness). Verbatim move
(self -> engine), no interface redesign.
"""
import numpy as np

import chess
import chess.engine

from common.board_information import is_takeback, calculate_threatened_levels
from common.utils import extend_mate_score


def set_mood(engine):
    """ Given input information, we determine our mood which influences the rest of our
        calculations. The calculation of these moods should ideally not depend on engine
        analytics such as lucas analytics and should rely on more human heuristics
        (Material up/down, time situation etc). Moods can be the following, and roughly
        correspond to the following behaviours:

        Confident - Default mood. Plays move at normal speeds relative to the game phase
        Cocky - Plays moves faster and considers less moves in general in a situation.
                Usually occurs when player is winning heavily in material and not under big threats.
        Cautious - Plays moves slower and considers more moves and ponders more.
                Usually occurs in complex even positions.
        Tilted - Plays moves either really fast or takes one big think and considers less
                moves. Happens when we have recently blundered some material,
                or are losing so badly and close to resigning.
        Hurry - Plays moves faster, consider less moves and does more pondering
                Occurs when we are in time trouble.
        Flagging - Plays moves faster, considers less sometimes or is cautious.
                Occurs when we have a lot more time than opponent and they are in time
                trouble.
    """

    engine.log += "Setting mood from given input information. \n"
    engine.just_blundered = False
    # First check our time situation
    # If we are low in time, we are in hurry mode.
    # We define low time to be normally distributed about (initial_time*0.1 + 15*0.7)
    # with standard deviation initial_time/30
    self_initial_time = engine.input_info["self_initial_time"]
    opp_initial_time = engine.input_info["opp_initial_time"]
    own_time = max(engine.input_info["self_clock_times"][-1],1)
    opp_time = max(engine.input_info["opp_clock_times"][-1],1)
    self_low_time_threshold = self_initial_time*0.1 + 15*0.7 + self_initial_time*np.random.randn()/30
    opp_low_time_threshold = opp_initial_time*0.1 + 15*0.7 + opp_initial_time*np.random.randn()/30

    if own_time < self_low_time_threshold:
        return "hurry"
    engine.log += "We have more than the threshold {} time, not in hurry mode. \n".format(self_low_time_threshold)

    # Next we consider whether we are tilted from a past blunder
    # this requires information of evaluations from previous positions
    # If we made the blunder one move ago (so eval from 3 positions ago was much higher than it is now)
    # Then we make a big pause and set just_blundered to be True
    # First determine whether we have made a mistake in the last few moves.
    current_eval = extend_mate_score(engine.stockfish_analysis[0]["score"].pov(engine.input_info["side"]).score(mate_score=2500))
    if len(engine.input_info["fens"]) >= 3:
        engine.log += "Checking to see if we have made a blunder recently. \n"
        last_avail_board = chess.Board(engine.input_info["fens"][0])
        last_eval = extend_mate_score(engine.stockfish_engine.analyse(last_avail_board, limit=chess.engine.Limit(time=0.02))["score"].pov(engine.input_info["side"]).score(mate_score=2500))
        if last_eval - current_eval > 300 and current_eval < 200: # if our previous position is much better than it is now, and we are not massively winning still
            # if the blunder happened exactly a move ago
            # then with some probability we recognise we have just made a blunder, and this
            # fact is then used in _get_time_taken()
            if np.random.random() < 0.8:
                engine.just_blundered = True
            else:
                engine.just_blundered = False
            return "tilted"
        engine.log += "Haven't made a blunder recently. Last eval: {}, Current eval: {} \n".format(last_eval, current_eval)

    # Next check opponent time. If opponent is in time pressure we would be in flagging mode
    if opp_time < opp_low_time_threshold:
        # we won't always be in flagging mode, particularly if we have enough time.
        if np.random.random() < 0.7:
            return "flagging"
        else:
            engine.log += "Opponent has less than the threshold {} time, but by chance not in flagging mode. \n".format(opp_low_time_threshold)
    else:
        engine.log += "Opponent has more than the threshold {} time, not in flagging mode. \n".format(opp_low_time_threshold)

    # If we are up on time lots and winning by alot on the position, then we are in cocky mode
    # Also in the initial stages of the game when we are blitzing out opening moves.
    # We define "up on time by a lot" by initial_time/6
    # We define initial stages as  time > initial_time - low_time_threshold/2
    if own_time > self_initial_time - self_low_time_threshold/2:
        return "cocky"
    engine.log += "Not in initial stages and blitzing opening moves. Not in cocky mode. \n"
    time_gap = self_initial_time/6
    if current_eval > 300 and own_time - opp_time > time_gap:
        return "cocky"
    engine.log += "Time gap {} and on current eval {}. We are not in cocky mode. \n".format(own_time - opp_time, current_eval)

    # If position is relatively even, and there are exactly a few good moves
    # (not 1 good move but between 2-4)
    complexity = engine.lucas_analytics["complexity"]
    eff_mob = engine.lucas_analytics["eff_mob"]
    if abs(current_eval) < 250 and eff_mob < 15 and eff_mob > 2:
        # top two moves cannot be takebacks. These cases are not considered complicated
        if len(engine.input_info["last_moves"]) >= 1 and len(engine.input_info["fens"]) >= 2:
            prev_move_uci = engine.input_info["last_moves"][-1]
            top_move_uci = engine.stockfish_analysis[0]["pv"][0].uci()
            second_move_uci = engine.stockfish_analysis[1]["pv"][0].uci()
            prev_board = chess.Board(engine.input_info["fens"][-2])
            if is_takeback(prev_board, prev_move_uci, top_move_uci) and is_takeback(prev_board, prev_move_uci, second_move_uci):
                # not cautious
                engine.log += "Top two engine moves are both takebacks. Cannot be cautious. \n"
                return "confident"
        if np.random.random() < (0.35 + complexity/(100*eff_mob + 100))**0.6:
            return "cautious"
        else:
            engine.log += "Postion is close to even (current eval {}) with complexity {} and eff_mob {}. But by chance not cautious. \n".format(current_eval, complexity, eff_mob)
    else:
        engine.log += "Position not even enough (current eval {}) or did not satisify eff_mob conditions (eff_mob {}) . Not in cautious mode. \n".format(current_eval, eff_mob)

    # If no previous criteria is satisfied, resort to default mood
    return "confident"


def check_opp_blunder(engine):
    """ Check from last couple of positions whether opponent has made a blunder on the
        last move, and did so by hanging a valuable piece.

        Returns None. We only update the self.opp_just_blundered variable
    """
    engine.opponent_just_blundered = False
    # We can only do this if we have the previous positions excluding our current position
    if len(engine.input_info["fens"]) < 2:
        engine.log += "Can't detect whether opponent has blundered as we don not have enough previous position info. \n"
        return
    # First check opponent just blundered based on eval
    curr_eval = extend_mate_score(engine.stockfish_analysis[0]["score"].pov(engine.input_info["side"]).score(mate_score=2500))
    prev_board = chess.Board(engine.input_info["fens"][-2])
    prev_analysis = engine.stockfish_engine.analyse(prev_board, limit=chess.engine.Limit(depth=8, time=0.02))
    prev_eval = extend_mate_score(prev_analysis["score"].pov(engine.input_info["side"]).score(mate_score=2500))
    if curr_eval - prev_eval > 150: # then opponent just blundered
        engine.log += "Opponent has blundered, checking to see if it is from hung piece. \n"
        # now check if opponent just hung piece they played
        last_move_obj = chess.Move.from_uci(engine.input_info["last_moves"][-1])
        if calculate_threatened_levels(last_move_obj.to_square, engine.current_board) >= 3:
            # then opponent has hung a piece
            engine.log += "Opponent has hung a piece, acting startled. \n"
            engine.opponent_just_blundered = True
        else:
            engine.log += "Opponent has not hung a piece. \n"
    else:
        engine.log += "Opponent has not blundered. Current eval {}, previous eval {} \n".format(curr_eval, prev_eval)
