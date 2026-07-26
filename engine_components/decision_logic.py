"""Search-breadth and move-time pacing, extracted verbatim from engine.Engine.

decide_breadth picks how many root moves to consider; set_target_time and
get_time_taken decide how long to spend on the move (the sharpness-driven
"complicated position" envelope, the intuition gate, per-game pace/mood
scaling, and reflective/rating-based pacing); chosen_move_wc_loss and
adjust_time_for_move_loss are the hesitation-before-the-mistake /
decisive-snap pass applied after the move is chosen.

Sixth slice of the engine.py strangler-fig extraction (see
testing/engine_parity/ for the regression harness) -- the single biggest
behavioural gap in the old abandoned tree, which had none of the
sharpness/intuition-gate/per-game-character pacing added since. Verbatim
move (self -> engine), no interface redesign.
"""
import numpy as np

from common.board_information import phase_of_game, king_danger
from common.search_constants import (
    BREADTH_TIME_BONUS_TIERS, EFF_MOB_TACTICAL_CUTOFF, EFF_MOB_FORCED_CUTOFF,
    KING_DANGER_THRESHOLD, KING_DANGER_BREADTH_BONUS, TACTICAL_MIDGAME_BREADTH_DELTA,
    ENDGAME_LOW_MOB_BREADTH_FLOOR, ENDGAME_LOW_MOB_BREADTH_BONUS,
    ENDGAME_BREADTH_FLOOR, ENDGAME_BREADTH_BONUS, STANDARD_BREADTH_DELTA,
    MOOD_BREADTH_DELTAS, MODERATE_SHARPNESS_LO, MODERATE_SHARPNESS_HI,
    MODERATE_SHARPNESS_BREADTH_BONUS,
)
from common.constants import (
    MISTAKE_HESITATION_WC_LOSS, MISTAKE_HESITATION_PROB,
    MISTAKE_HESITATION_RANGE, MISTAKE_HESITATION_MIN_TIME,
    MISTAKE_SNAP_WC_LOSS, MISTAKE_SNAP_PROB, MISTAKE_SNAP_RANGE,
)


def decide_breadth(engine, total_time=None):
    """ Given our current board information and amount of time, decide how many of our human moves
        to consider and pass onto the engine.

        Returns integer
    """
    base_no = engine.playing_level
    if total_time is not None:
        for time_threshold, bonus in BREADTH_TIME_BONUS_TIERS:
            if total_time > time_threshold:
                base_no += bonus
                break

    game_phase = phase_of_game(engine.current_board)
    king_dang = king_danger(engine.current_board, engine.input_info["side"], game_phase)
    eff_mob = engine.lucas_analytics["eff_mob"]
    if eff_mob < EFF_MOB_TACTICAL_CUTOFF:
        # Either the best move is obvious (e.g. a takeback, mate in one) or
        # there is a tactic involved. Decrease search width to mimic human
        # behaviour in tactics under pressure
        if king_dang > KING_DANGER_THRESHOLD:
            # king is in danger, need to pay attention
            no_moves = base_no + KING_DANGER_BREADTH_BONUS
        elif eff_mob > EFF_MOB_FORCED_CUTOFF and game_phase == "midgame":
            no_moves = max(base_no + TACTICAL_MIDGAME_BREADTH_DELTA, 1)
        elif game_phase == "endgame":
            no_moves = max(ENDGAME_LOW_MOB_BREADTH_FLOOR, base_no + ENDGAME_LOW_MOB_BREADTH_BONUS)
        else:
            no_moves = max(base_no, 1)
    else:
        # There are plenty of good moves, search wide so close the game out effectively
        if game_phase == 'endgame' and king_dang < KING_DANGER_THRESHOLD:
            no_moves = max(ENDGAME_BREADTH_FLOOR, base_no + ENDGAME_BREADTH_BONUS)
        elif king_dang > KING_DANGER_THRESHOLD:
            # king is in danger, need to pay attention
            no_moves = base_no + KING_DANGER_BREADTH_BONUS
        else:
            # not in the endgame, lots of good moves
            no_moves = max(base_no + STANDARD_BREADTH_DELTA, 1)

    # Real eval-stakes but no single forced answer: neither the quiet-position
    # time cut nor the sharp-position king-danger/tactical/intuition-gate
    # treatment reaches this band, and it's the band the bucketed
    # human-likeness diagnostic showed diverging most from human play (see
    # search_constants.py). Bonus applied flat on top of whichever
    # eff_mob/king-danger branch fired above.
    sharpness = engine.sharpness
    if sharpness is not None and MODERATE_SHARPNESS_LO <= sharpness < MODERATE_SHARPNESS_HI:
        no_moves += MODERATE_SHARPNESS_BREADTH_BONUS
        engine.log += "Moderate sharpness ({:.3f}): breadth bonus +{} \n".format(
            sharpness, MODERATE_SHARPNESS_BREADTH_BONUS)

    # Now for mood dependent logic
    no_moves = max(no_moves + MOOD_BREADTH_DELTAS.get(engine.mood, 0), 1)

    engine.log += "Calculated number of root moves in current position: {} \n".format(no_moves)
    return no_moves


def set_target_time(engine, total_time):
    """ Given we are using human approaches to decide the move, we set and initial
        target time which we try to compute our human move. This is supposedly a
        reflection of how hurredly the player is before they've even thought
        about any moves. total_time is the time limit we cannot exceed.

        Returns target time, a non-negative float.
    """
    # self_initial_time = engine.input_info["self_initial_time"]
    # own_time = max(engine.input_info["self_clock_times"][-1],1)

    # base_time = max(0.6*QUICKNESS*self_initial_time**1.1/(100 + self_initial_time**0.7), 0.1)

    # # if we are in hurry mode (i.e. we are in low time), then we adjust our base
    # # time accordingly
    # mood_sf_dic = {"confident": 1,
    #                 "cocky": 0.6,
    #                 "cautious": 1.6,
    #                 "tilted": 0.4,
    #                 "hurry": (own_time/self_initial_time)**0.7,
    #                 "flagging": 0.8}

    # # we need to leverage information from lucas analytics
    # activity = engine.lucas_analytics["activity"]
    # activity_sf = ((activity+12)/25)**0.4

    # target_time = base_time * activity_sf * mood_sf_dic[engine.mood]
    # while target_time > total_time*0.9: # cannot exceed total time
    #     target_time /= 1.2

    # # likewise if the human consider time is too low
    # while target_time < total_time*0.5:
    #     target_time *= 1.2

    # Base entirely on max achievable time for greatest human calculation depth
    if total_time <= 1:
        # then we would never have enough time to ponder anyway, so human evaluation takes up majority.
        target_time = total_time*0.8
    elif total_time <= 2.5:
        # grey area where we have a bit of time but not too much time to ponder.
        ratio = 0.6*(total_time-1) + 0.8*(2-total_time)
        target_time = ratio*total_time
    else:
        # maximum amount of human time capped, use rest for ponder
        target_time = total_time*0.6

    engine.log += "Decided target time for human evaluation to be: {} \n".format(target_time)
    return max(target_time,0.1)


def get_time_taken(engine, obvious=False, human_filters=True):
    """ Calculates the amount of time in total we should spend on a move.
        obvious is whether we made a quick obvious move.
        human_filters is whether we used human filters

        Returns time_taken : float
    """
    engine.log += "Deciding time taken to make the move from receiving input. \n"
    self_initial_time = engine.input_info["self_initial_time"]
    base_time = max(engine.quickness*self_initial_time**1.1/(100 + self_initial_time**0.7),0.1)
    engine.log += "Initial base time without calculations: {} \n".format(base_time)
    # we move faster depending on whether we are proportionally behind on time
    # or move slower if we are ahead
    own_time = max(engine.input_info["self_clock_times"][-1],1)
    opp_time = max(engine.input_info["opp_clock_times"][-1],1)
    base_time *= (own_time/opp_time)**(10/self_initial_time)
    engine.log += "Base time after relative time vs opponent: {} \n".format(base_time)

    obvious_sf = 0.2
    # default value for time_taken
    time_taken = 1.0
    if obvious == True:
        # if we have made obvious move, we don't need to take much time
        time_taken = base_time * obvious_sf * (1 + np.random.uniform(-0.5, 0.5))
        if engine.game_pace_sf is not None:
            time_taken *= engine.game_pace_sf
        engine.log += "We have made obvious mode, so move quickly in {} seconds. \n".format(time_taken)
        return time_taken
    elif human_filters == True:
        # we have used human probabilities calculation.
        # our main analysis is for this case
        game_phase = phase_of_game(engine.current_board)
        # in the opening and endgame we spend less time on average than the mid game
        if game_phase == "opening":
            base_time = (base_time ** 0.2)/2
        elif game_phase == "midgame":
            if self_initial_time > 60:
                base_time *= 1.7
            else:
                base_time *= 1.4
        else:
            base_time *=0.7

        engine.log += "Base time after game phase analysis: {} \n".format(base_time)

        # The sharper (more critical) the position, the more we think.
        # "Complicated position" is defined by eval-stakes -- the spread in
        # win-probability across the plausible moves (engine.sharpness) --
        # rather than the structural Lucas activity/eff_mob it used to key
        # off. The scaling envelope is unchanged: sharpness is mapped onto
        # the same ((x+12)/25)**0.4 curve, with the "critical position"
        # threshold (0.25) sitting at the neutral point (multiplier 1.0),
        # so the range of delay outputs is the same -- only the trigger
        # changed.
        sharpness = engine.sharpness if engine.sharpness is not None else 0.25

        # Intuition gate: a human does NOT deep-think every critical
        # position -- often they trust intuition and snap the move out,
        # especially to avoid burning the clock. So in a sharp position we
        # only apply the slow-down some of the time; the rest of the time we
        # skip the boost and play quickly. The gating probability is a
        # per-game draw (see _sample_game_character, mean 0.65): a
        # trust-the-gut game snaps most sharp positions, a grinding game
        # genuinely thinks on most of them (which the mood branch then
        # thins further). Prevents the bot from reliably over-thinking
        # sharp positions and falling behind on time.
        snap_gate = engine.game_snap_gate if engine.game_snap_gate is not None else 0.65
        snap_in_sharp = (sharpness >= 0.25) and (np.random.random() < snap_gate)
        if snap_in_sharp:
            base_time *= 0.5
            engine.log += "Sharp position (sharpness={:.3f}) but snapping on intuition (gate {:.2f}); base_time *= 0.5 -> {} \n".format(
                sharpness, snap_gate, base_time)
            print(f"[ENGINE] Sharp position (sharpness={sharpness:.3f}): snapping on intuition (no long think).")
        else:
            # The sharper (more critical) the position, the more we think.
            # "Complicated position" is defined by eval-stakes -- the spread
            # in win-probability across the plausible moves (engine.sharpness)
            # -- rather than the structural Lucas activity/eff_mob it used to
            # key off. The scaling envelope is unchanged: sharpness is mapped
            # onto the same ((x+12)/25)**0.4 curve, with the "critical
            # position" threshold (0.25) sitting at the neutral point
            # (multiplier 1.0), so the range of delay outputs is the same --
            # only the trigger changed.
            sharp_scaled = min(sharpness * 52, 25)
            sharpness_sf = ((sharp_scaled + 12) / 25) ** 0.4
            base_time *= sharpness_sf

            engine.log += "Base time after sharpness analysis (sharpness={:.3f}, multiplier={:.3f}): {} \n".format(
                sharpness, sharpness_sf, base_time)

            # When the plausible moves all keep roughly the same
            # win-probability (a flat, non-critical position), little is at
            # stake, so we think less -- the sharpness analogue of the old
            # "many good moves" cut.
            if sharpness < 0.10:
                base_time *= 0.7
                engine.log += "Base time after low-sharpness (flat position) analysis: {} \n".format(base_time)

        # If opponent has just blundered, then act startled
        if engine.opponent_just_blundered == True:
            engine.log += "Opponent has just blundered, acting startled with longer think time. \n"
            base_time *= 2

        # based on the time control, we have a higher end multiplier
        high_range_multiplier = self_initial_time**0.35/(60**0.35)

        # Now sort according to moods
        if engine.mood == "confident":
            # low variation, solid move times.
            if np.random.random() < 0.5: # then we have a quick low variation move
                time_taken = base_time * (0.6 + np.random.uniform(-0.3, 0.3))
            elif np.random.random() < 0.85: # medium low variation move
                time_taken = base_time * (1.0 + np.random.uniform(-0.3, 0.3))
            else:
                # slightly longer think, larger variation
                time_taken = base_time * (3.2 + np.random.uniform(-0.7, 1.0)) * high_range_multiplier
        elif engine.mood == "cocky":
            # medium variation, quick move times
            if np.random.random() < 0.9: # then we have a quick low variation move
                time_taken = base_time * (0.5 + np.random.uniform(-0.2, 0.7))
            else:
                # slightly longer think
                time_taken = base_time * (3.1 + np.random.uniform(-0.8, 0.8)) * high_range_multiplier
        elif engine.mood == "cautious":
            # medium variation, slow moves
            if np.random.random() < 0.6: # then we have a quick low variation move
                time_taken = base_time * (1.4 + np.random.uniform(-0.3, 0.3))
            elif np.random.random() < 0.7: # medium low variation move
                time_taken = base_time * (2.1 + np.random.uniform(-0.4, 0.4))
            else:
                # slightly longer think, large variation
                time_taken = base_time * (4.9 + np.random.uniform(-0.9, 0.5)) * high_range_multiplier
        elif engine.mood == "tilted":
            # if we have just made the blunder, pause for long time to reflect on it
            # otherwise low variation, quick move times
            if engine.just_blundered == True:
                time_taken = base_time * (3.2 + np.random.uniform(-1.5, 1.5)) * high_range_multiplier
            else:
                time_taken = base_time * (0.4 + np.random.uniform(-0.2, 0.2))
        elif engine.mood == "hurry":
            # medium variation, quick move times
            if np.random.random() < 0.6:
                time_taken = base_time * (0.3 + np.random.uniform(-0.2, 0.3))
            elif np.random.random() < 0.8:
                time_taken = base_time * (0.9 + np.random.uniform(-0.3, 0.3))
            else:
                # slightly longer think
                time_taken = base_time * (2.0 + np.random.uniform(-0.4, 0.6))

            # if we are in hurry mode (i.e. we are in low time), then our time taken
            # depends on how much time we have left
            time_taken *= (3*own_time/self_initial_time)**0.9
        elif engine.mood == "flagging":
            # large variation, quick move times
            if np.random.random() < 0.6:
                time_taken = base_time * (0.5 + np.random.uniform(-0.3, 0.6))
            elif np.random.random() < 0.8:
                time_taken = base_time * (1.3 + np.random.uniform(-0.5, 0.8))
            else:
                # slightly longer think
                time_taken = base_time * (3.6 + np.random.uniform(-0.8, 1.0)) * high_range_multiplier

        engine.log += "Decided time taken after mood analysis: {} \n".format(time_taken)

    # This game's pace multiplier (see _sample_game_character). Applied after
    # the phase/mood envelopes -- the opening's **0.2 compression would
    # crush it upstream -- so the whole game runs uniformly faster or
    # slower. Sits before the reflective-pacing blend: matching the
    # opponent's tempo deliberately overrides our own pace.
    if engine.game_pace_sf is not None:
        time_taken *= engine.game_pace_sf
        engine.log += "Time taken after per-game pace multiplier ({:.3f}): {} \n".format(engine.game_pace_sf, time_taken)

    # we also use reflective moving, which is to play faster when opponent has been playing
    # fast over the last few moves (as we feel the pressure to keep up with the pace)
    # we do not move slower if opponents has beserked, because our mood function should take care of that
    if len(engine.input_info["opp_clock_times"]) >= 4:
        target_ratio = min(engine.input_info["self_clock_times"][-1] / (engine.input_info["opp_clock_times"][-1] + 1e-6), 1)
        recent_time = (engine.input_info["opp_clock_times"][-4] - engine.input_info["opp_clock_times"][-1])/3 # opponents average move time in last 3 moves
        non_recent_time = (engine.input_info["opp_clock_times"][0] - engine.input_info["opp_clock_times"][-1])/(len(engine.input_info["opp_clock_times"])-1)
        target_time_spent = 0.2*non_recent_time + 0.8*recent_time
        # normalise for what the opponent total time is (in the case where our starting time was different)
        # eg beserks
        berserk_ratio = max(engine.input_info["opp_initial_time"] / (engine.input_info["self_initial_time"] + 1e-6),1)
        target_time_spent /= berserk_ratio

        target_time_spent *= target_ratio

        # sometimes we don't use this
        if np.random.random() < 0.3:
            time_taken = 0.8*target_time_spent + time_taken * 0.2
    engine.log += "Decided time taken after opponent speed consideration: {} \n".format(time_taken)

    # we also change the time taken based on the opponent's rating.
    # generally we play faster against weaker opponents, and slower against stronger opponents
    opp_rating = engine.input_info["opp_rating"]
    self_rating = engine.input_info["self_rating"]
    if opp_rating is not None and self_rating is not None:
        rating_ratio = (self_rating - opp_rating)/self_rating
        rating_factor = 1 - 0.75*np.tanh(2*rating_ratio)
        time_taken *= rating_factor
        engine.log += "Decided time taken after opponent relative rating consideration: {} \n".format(time_taken)
    else:
        engine.log += "No rating information available. Not considering rating factor. \n"

    # make sure we are not taking time that we do not have
    if time_taken > own_time*0.7+1:
        time_taken = own_time*0.7 + 1
        engine.log += "We do not have enough time to take this much time. Reducing time taken to {} \n".format(time_taken)
    time_taken = max(time_taken, 0.1)
    engine.log += "Decided time taken for move: {} \n".format(time_taken)
    # Targeted console output
    print(f"[ENGINE] Decided time taken for move: {time_taken:.3f}s")
    return time_taken


def chosen_move_wc_loss(engine, move_uci):
    """ Win-probability given up by the chosen move vs the engine's best.

        Prefers the deep narrow sharpness scan (multipv 5, no time cap):
        if the chosen move is one of its lines the loss is read directly;
        if it fell outside the top 5, the scan's spread is a lower bound
        on the loss (the move is worse than every scanned line). The
        full-width analysis is only a fallback estimate -- it is capped
        at 20ms, so its per-move evals are very noisy. Returns None if
        nothing can be read.
    """
    scan = engine.sharpness_scan
    shallow = None
    if engine.stockfish_analysis:
        side = engine.input_info["side"]
        best_wc, move_wc = None, None
        for entry in engine.stockfish_analysis:
            cp = entry["score"].pov(side).score(mate_score=10000)
            wc = 1.0 / (1.0 + np.exp(-0.00368208 * cp))
            if best_wc is None or wc > best_wc:
                best_wc = wc
            if entry["pv"][0].uci() == move_uci:
                move_wc = wc
        if best_wc is not None and move_wc is not None:
            shallow = best_wc - move_wc
    if scan:
        scan_best = max(scan.values())
        if move_uci in scan:
            return scan_best - scan[move_uci]
        # Chosen move is outside the scan's top lines: at least as bad as
        # the worst scanned line. Floor the shallow estimate with that.
        lower_bound = scan_best - min(scan.values())
        if shallow is None:
            return lower_bound
        return max(shallow, lower_bound)
    return shallow


def adjust_time_for_move_loss(engine, move_uci, time_take):
    """ The hesitation before the mistake -- and its mirror, the decisive
        snap. Humans think longer in positions where they end up erring
        (difficulty consumes clock AND induces the error), giving a
        positive per-game correlation between move time and move loss.
        The engine's errors come from the human-probability sampling,
        independent of the decided think time, so the link is restored
        here: when the chosen move gives up real win probability, the
        already-decided think time is (sometimes) stretched; when it is
        clean, it is (sometimes) trimmed -- humans bang out moves they
        are sure of. The trim keeps the mean move time level despite the
        stretches. Not every mistake hesitates: snap blunders exist.

        Returns the adjusted time_take.
    """
    own_time = max(engine.input_info["self_clock_times"][-1], 1)
    if own_time < MISTAKE_HESITATION_MIN_TIME:
        # scramble errors are fast and must stay fast
        return time_take
    try:
        wc_loss = engine._chosen_move_wc_loss(move_uci)
    except Exception as e:
        engine.log += "WARNING: could not compute chosen-move loss ({}); skipping hesitation. \n".format(e)
        return time_take
    if wc_loss is None:
        return time_take
    if wc_loss >= MISTAKE_HESITATION_WC_LOSS:
        if np.random.random() >= MISTAKE_HESITATION_PROB:
            engine.log += "Chosen move loses {:.3f} win prob but snapping it anyway (no hesitation). \n".format(wc_loss)
            return time_take
        stretch = np.random.uniform(*MISTAKE_HESITATION_RANGE)
        new_time = min(time_take * stretch, own_time*0.7 + 1)
        engine.log += "Chosen move loses {:.3f} win prob; hesitating before the mistake (x{:.2f}: {:.2f}s -> {:.2f}s). \n".format(
            wc_loss, stretch, time_take, new_time)
        print(f"[ENGINE] Hesitating before a mistake (wc loss {wc_loss:.3f}): {time_take:.2f}s -> {new_time:.2f}s")
        return new_time
    if wc_loss <= MISTAKE_SNAP_WC_LOSS and np.random.random() < MISTAKE_SNAP_PROB:
        trim = np.random.uniform(*MISTAKE_SNAP_RANGE)
        new_time = time_take * trim
        engine.log += "Chosen move is clean (loss {:.3f}); playing it decisively (x{:.2f}: {:.2f}s -> {:.2f}s). \n".format(
            wc_loss, trim, time_take, new_time)
        return new_time
    return time_take
