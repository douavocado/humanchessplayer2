"""Mood/blunder-reaction heuristics, extracted verbatim from engine.Engine.

set_mood decides the human "mood" label (confident/cocky/cautious/tilted/
hurry/flagging) that other components (pacing, breadth) read off
engine.mood. check_opp_blunder is the separate "did the opponent just hang
something" detector that drives startled reactions -- a real piece left or
put en pris, by an unforced, unpredicted move -- rather than an instant,
mechanical snap-up of anything the eval briefly likes. Deliberately kept
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
from common.constants import OPP_BLUNDER_EVAL_SWING_MIN, OPP_BLUNDER_EN_PRIS_MIN_VALUE


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
    """ Check whether the opponent's last move was a clear, unexpected blunder:
        it left or put a piece en pris for real material, and it wasn't
        something we could already have predicted (their only move, or the
        engine's own top choice in that position). This is what should gate
        a startled, longer-than-usual think time -- an instant, no-reaction
        capture of a hung piece reads as mechanical, not human.

        Sets engine.opponent_just_blundered. Deliberately does NOT reset it
        to False itself: check_obvious_move (simple_decisions.py) does that
        reset once per make_move call, before this runs, and may already
        have set it True for its own narrower "recapture wasn't really a
        takeback" case -- this function only ever adds a True on top,
        never removes one.
    """
    # We can only do this if we have the previous positions excluding our current position
    if len(engine.input_info["fens"]) < 2:
        engine.log += "Can't detect whether opponent has blundered as we don not have enough previous position info. \n"
        return

    # Cheap pre-filter on the raw eval swing before doing the (relatively
    # expensive) per-square exchange scan below. Loose on purpose -- a fixed
    # cp threshold saturates in already-winning positions (hanging a whole
    # rook barely moves the eval when we were already crushing), so this
    # only rules out genuinely quiet moves. The en pris scan is the real
    # gate.
    curr_eval = extend_mate_score(engine.stockfish_analysis[0]["score"].pov(engine.input_info["side"]).score(mate_score=2500))
    prev_board = chess.Board(engine.input_info["fens"][-2])
    prev_analysis = engine.stockfish_engine.analyse(prev_board, limit=chess.engine.Limit(depth=8, time=0.02))
    prev_eval = extend_mate_score(prev_analysis["score"].pov(engine.input_info["side"]).score(mate_score=2500))
    if curr_eval - prev_eval <= OPP_BLUNDER_EVAL_SWING_MIN:
        engine.log += "Opponent has not blundered. Current eval {}, previous eval {} \n".format(curr_eval, prev_eval)
        return
    engine.log += "Eval swing large enough to check for a hung piece (previous eval {}, current eval {}). \n".format(prev_eval, curr_eval)

    # Degree of enprisness: the most material we could win back in an
    # exchange on any opponent piece still on the board. Scanning every
    # piece (not just the one that just moved) covers both putting a piece
    # en pris and leaving one en pris elsewhere on the board.
    opp_colour = not engine.input_info["side"]
    worst_sq, worst_val = None, 0.0
    for sq in chess.SQUARES:
        piece = engine.current_board.piece_at(sq)
        if piece is None or piece.color != opp_colour or piece.piece_type == chess.KING:
            continue
        val = calculate_threatened_levels(sq, engine.current_board)
        if val > worst_val:
            worst_val, worst_sq = val, sq

    if worst_val < OPP_BLUNDER_EN_PRIS_MIN_VALUE:
        engine.log += "Eval swung but nothing is clearly hanging (worst en pris value {}). Not acting startled. \n".format(worst_val)
        return
    engine.log += "Opponent has a piece en pris on {} worth {} in an exchange. \n".format(chess.square_name(worst_sq), worst_val)

    # Unexpectedness: only react startled when we couldn't already have
    # seen this coming -- not their only legal move, and not what a quick
    # engine scan already thought was their best try in that position (the
    # "blunder" may just be the least-bad option under a threat).
    if prev_board.legal_moves.count() == 1:
        engine.log += "Opponent's move was forced (only legal move). Not unexpected, not acting startled. \n"
        return
    last_move_obj = chess.Move.from_uci(engine.input_info["last_moves"][-1])
    prev_pv = prev_analysis.get("pv")
    if prev_pv and prev_pv[0] == last_move_obj:
        engine.log += "Opponent's move was also the engine's own top choice in that position, so we would have predicted it. Not acting startled. \n"
        return

    engine.log += "Opponent's blunder was unforced and unpredicted. Acting startled. \n"
    engine.opponent_just_blundered = True
