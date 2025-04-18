import numpy as np
import chess
import chess.engine
import random

from .logger import Logger
from .state_manager import StateManager
from .analyzer import Analyzer, STOCKFISH # Import shared STOCKFISH instance
from .mood_manager import MoodManager
from common.constants import QUICKNESS
from common.board_information import (
    phase_of_game, PIECE_VALS, king_danger, is_takeback,
    calculate_threatened_levels, check_best_takeback_exists
)
from common.utils import extend_mate_score

def decide_resign(logger: Logger, state_manager: StateManager, analyzer: Analyzer) -> bool:
    """Decides whether the engine should resign based on the current position and game state."""
    logger.add_log("Deciding whether to resign the current position.\n")
    if not state_manager.is_analytics_updated():
        logger.add_log("WARNING: decide_resign() called with outdated analytics. Results may be inaccurate.\n")

    current_board = state_manager.get_board()
    side = state_manager.get_side()
    if side is None:
        logger.add_log("WARNING: Cannot decide resign, side not set.\n")
        return False

    self_clock_times = state_manager.get_info("self_clock_times", [])
    opp_clock_times = state_manager.get_info("opp_clock_times", [])
    own_time = max(self_clock_times[-1], 1) if self_clock_times else 60
    opp_time = max(opp_clock_times[-1], 1) if opp_clock_times else 60
    starting_time = state_manager.get_info("self_initial_time", 60)

    # Conditions to NOT resign
    if current_board.fullmove_number < 15:
        logger.add_log(f"Not resigning: Move number {current_board.fullmove_number} < 15.\n")
        return False
    if opp_time <= 10:
        logger.add_log(f"Not resigning: Opponent time {opp_time} <= 10s.\n")
        return False
    # Check if own_time/(opp_time-10) > 3, handle potential division by zero if opp_time <= 10 (already checked)
    if opp_time > 10 and (own_time / (opp_time - 10)) > 3:
         logger.add_log(f"Not resigning: Significant time advantage ({own_time}s vs {opp_time}s).\n")
         return False

    win_prob = analyzer.get_lucas_metric("win_prob", default=0.5) # Default to 50% if unavailable

    # Calculate material imbalance
    mat_dic = {chess.PAWN: 1, chess.KNIGHT: 3.1, chess.BISHOP: 3.5, chess.ROOK: 5.5, chess.QUEEN: 9.9, chess.KING: 0} # King value doesn't matter for imbalance
    our_mat = sum(len(current_board.pieces(pt, side)) * mat_dic[pt] for pt in mat_dic)
    opp_mat = sum(len(current_board.pieces(pt, not side)) * mat_dic[pt] for pt in mat_dic)
    material_diff = opp_mat - our_mat

    # Resign condition: Low win probability AND significant material disadvantage
    # Threshold adjusted based on opponent's time and starting time
    resign_threshold_win_prob = np.log(max(starting_time, 1)) * opp_time / (500 * max(starting_time, 1))
    resign_threshold_material = 5

    logger.add_log(f"Resign check: WinProb={win_prob:.3f} (Threshold={resign_threshold_win_prob:.3f}), MaterialDiff={material_diff:.1f} (Threshold={resign_threshold_material}).\n")

    if win_prob < resign_threshold_win_prob and material_diff > resign_threshold_material:
        logger.add_log("Conditions met: Win probability too low, material imbalance too large, and opponent time sufficient. Resigning.\n")
        return True
    else:
        logger.add_log("Conditions not met for resignation.\n")
        return False

def decide_human_filters(logger: Logger, state_manager: StateManager) -> bool:
    """Decides whether to use human-like move filters based on time pressure."""
    self_clock_times = state_manager.get_info("self_clock_times", [])
    own_time = max(self_clock_times[-1], 1) if self_clock_times else 60

    if own_time < 10:
        # Probability of *not* using filters increases as time decreases below 10s
        prob_no_filters = (10 - own_time) / 25
        if np.random.random() < prob_no_filters:
            logger.add_log(f"Low time ({own_time:.1f}s). Decided NOT to use human filters (Prob: {prob_no_filters:.2f}).\n")
            return False
        else:
            logger.add_log(f"Low time ({own_time:.1f}s). Decided TO use human filters (Prob: {1-prob_no_filters:.2f}).\n")
            return True
    else:
        logger.add_log(f"Sufficient time ({own_time:.1f}s). Using human filters.\n")
        return True

def decide_breadth(logger: Logger, state_manager: StateManager, analyzer: Analyzer, mood_manager: MoodManager, playing_level: int, target_time: float = None) -> int:
    """Determines the number of root moves (search breadth) to consider."""
    base_no = playing_level
    current_board = state_manager.get_board()
    side = state_manager.get_side()
    mood = mood_manager.get_mood()

    if side is None:
        logger.add_log("WARNING: Cannot decide breadth, side not set. Using default.\n")
        return base_no

    # Adjust base based on target thinking time (if provided)
    if target_time is not None:
        if target_time > 5:
            base_no += 3
        elif target_time > 2.5:
            base_no += 2
        elif target_time > 1.5:
            base_no += 1

    game_phase = phase_of_game(current_board)
    king_dang = king_danger(current_board, side, game_phase)
    eff_mob = analyzer.get_lucas_metric("eff_mob", default=20) # Default mobility

    # Adjust based on game state (mobility, king danger, phase)
    if eff_mob < 15: # Limited mobility - potentially tactical or forced
        if king_dang > 500:
            no_moves = base_no + 10 # King danger requires deeper look
        elif eff_mob > 5 and game_phase == "midgame":
            no_moves = max(base_no - 1, 1) # Tactical midgame, narrow search
        elif game_phase == "endgame":
            no_moves = max(10, base_no + 4) # Endgame requires precision
        else:
            no_moves = max(base_no, 1) # Default for low mobility
    else: # Plenty of moves - broader search often better
        if game_phase == 'endgame' and king_dang < 500:
            no_moves = max(9, base_no + 5) # Endgame conversion
        elif king_dang > 500:
            no_moves = base_no + 10 # King danger still priority
        else:
            no_moves = max(base_no - 1, 1) # Standard midgame/opening with options

    # Adjust based on mood
    if mood in ["cocky", "hurry"]:
        no_moves = max(no_moves - 1, 1)
    elif mood == "cautious":
        no_moves = no_moves + 1
    elif mood == "tilted":
        no_moves = max(no_moves - 2, 1)

    # Ensure at least 1 move is considered
    no_moves = max(no_moves, 1)

    logger.add_log(f"Calculated search breadth: {no_moves} (Base={playing_level}, TargetTime={target_time}, Phase={game_phase}, KingDanger={king_dang}, EffMob={eff_mob}, Mood={mood}).\n")
    return no_moves

def get_time_taken(logger: Logger, state_manager: StateManager, analyzer: Analyzer, mood_manager: MoodManager, obvious: bool = False, human_filters: bool = True) -> float:
    """Calculates the target time duration for the current move."""
    logger.add_log("Deciding time allocation for the move.\n")

    side = state_manager.get_side()
    if side is None:
        logger.add_log("WARNING: Cannot calculate time taken, side not set. Using default 1.0s.\n")
        return 1.0

    self_initial_time = state_manager.get_info("self_initial_time", 60)
    opp_initial_time = state_manager.get_info("opp_initial_time", 60)
    self_clock_times = state_manager.get_info("self_clock_times", [])
    opp_clock_times = state_manager.get_info("opp_clock_times", [])
    own_time = max(self_clock_times[-1], 1) if self_clock_times else self_initial_time
    opp_time = max(opp_clock_times[-1], 1) if opp_clock_times else opp_initial_time

    # Base time calculation (incorporates QUICKNESS constant)
    base_time = max(QUICKNESS * self_initial_time / (80 + self_initial_time**0.8), 0.1)
    logger.add_log(f"Initial base time: {base_time:.3f}s\n")

    # Adjust based on relative time vs opponent
    time_ratio_factor = (own_time / opp_time)**(10 / max(self_initial_time, 1))
    base_time *= time_ratio_factor
    logger.add_log(f"Base time after relative time factor ({time_ratio_factor:.2f}): {base_time:.3f}s\n")

    # Handle obvious moves
    if obvious:
        obvious_sf = 0.8
        time_taken = base_time * (obvious_sf + np.clip(0.2 * np.random.randn(), -0.5, 0.5))
        time_taken = max(time_taken, 0.05) # Minimum time for obvious move
        logger.add_log(f"Obvious move detected. Calculated time: {time_taken:.3f}s\n")
        return time_taken

    # Handle non-obvious moves (considering human filters or not)
    if not human_filters:
         # If not using human filters (e.g., pure Stockfish due to time), use a simpler, faster time allocation
         time_taken = base_time * 0.5 # Faster base time
         time_taken *= (own_time / self_initial_time)**0.5 # Scale down further with less time
         time_taken = max(time_taken, 0.05) # Minimum time
         logger.add_log(f"Not using human filters. Calculated time: {time_taken:.3f}s\n")
         return time_taken

    # --- Calculations for moves using human filters ---
    current_board = state_manager.get_board()
    game_phase = phase_of_game(current_board)
    mood = mood_manager.get_mood()

    # Adjust base time by game phase
    phase_sf = 1.0
    if game_phase == "opening":
        phase_sf = 0.3
    elif game_phase == "midgame":
        phase_sf = 1.7 if self_initial_time > 60 else 1.4
    else: # endgame
        phase_sf = 0.7
    base_time *= phase_sf
    logger.add_log(f"Base time after game phase '{game_phase}' factor ({phase_sf:.2f}): {base_time:.3f}s\n")

    # Adjust by position activity
    activity = analyzer.get_lucas_metric("activity", default=0)
    activity_sf = ((activity + 12) / 25)**0.4
    base_time *= activity_sf
    logger.add_log(f"Base time after activity ({activity}) factor ({activity_sf:.2f}): {base_time:.3f}s\n")

    # Adjust by effective mobility (more options -> slightly faster)
    eff_mob = analyzer.get_lucas_metric("eff_mob", default=20)
    eff_mob_sf = 1.0
    if eff_mob > 25:
        eff_mob_sf = 0.7
        base_time *= eff_mob_sf
        logger.add_log(f"Base time after high eff_mob ({eff_mob}) factor ({eff_mob_sf:.2f}): {base_time:.3f}s\n")

    # Adjust if opponent just blundered (act startled)
    if mood_manager.did_opponent_just_blunder():
        blunder_sf = 2.0
        base_time *= blunder_sf
        logger.add_log(f"Base time after opponent blunder factor ({blunder_sf:.2f}): {base_time:.3f}s\n")

    # Time allocation based on mood (with randomness)
    time_taken = base_time # Start with the adjusted base time
    high_range_multiplier = (self_initial_time**0.35) / (60**0.35) # Scales longer thinks for longer time controls

    mood_sf = 1.0
    rand_clip_low, rand_clip_high = -0.5, 0.5 # Default random range
    rand_scale = 0.2 # Default random scale

    if mood == "confident":
        p = np.random.random()
        if p < 0.4: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 0.8, 0.1, -0.3, 0.3
        elif p < 0.8: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 1.4, 0.2, -0.4, 0.7
        else: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 3.2 * high_range_multiplier, 0.7, -1.7, 2.0
    elif mood == "cocky":
        p = np.random.random()
        if p < 0.9: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 0.7, 0.3, -0.2, 0.7
        else: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 3.1 * high_range_multiplier, 0.4, -0.8, 0.8
    elif mood == "cautious":
        p = np.random.random()
        if p < 0.6: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 1.6, 0.2, -0.3, 0.5
        elif p < 0.7: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 2.5, 0.25, -0.4, 0.7
        else: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 5.9 * high_range_multiplier, 1.4, -3.9, 4.5
    elif mood == "tilted":
        if mood_manager.is_just_blundered():
            mood_sf, rand_scale, rand_clip_low, rand_clip_high = 4.2 * high_range_multiplier, 0.7, -1.5, 1.5
        else:
            mood_sf, rand_scale, rand_clip_low, rand_clip_high = 0.6, 0.08, -0.2, 0.2
    elif mood == "hurry":
        p = np.random.random()
        if p < 0.5: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 0.6, 0.1, -0.3, 0.3
        elif p < 0.8: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 1.3, 0.1, -0.3, 0.3
        else: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 2.0, 0.2, -0.4, 0.6
        # Further reduce time based on how little time is left
        hurry_time_factor = (3 * own_time / max(self_initial_time, 1))**0.9
        mood_sf *= hurry_time_factor
        logger.add_log(f"Hurry mood time factor: {hurry_time_factor:.2f}\n")
    elif mood == "flagging":
        p = np.random.random()
        if p < 0.5: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 1.0, 0.2, -0.3, 0.6
        elif p < 0.8: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 1.9, 0.3, -0.5, 0.8
        else: mood_sf, rand_scale, rand_clip_low, rand_clip_high = 3.6 * high_range_multiplier, 0.4, -0.8, 1.0

    time_taken = base_time * (mood_sf + np.clip(rand_scale * np.random.randn(), rand_clip_low, rand_clip_high))
    logger.add_log(f"Time after mood '{mood}' factor ({mood_sf:.2f} + random): {time_taken:.3f}s\n")

    # Reflective moving: Adjust time based on opponent's recent pace
    if len(opp_clock_times) >= 4:
        recent_opp_times = opp_clock_times[-4:]
        recent_time_spent = (recent_opp_times[0] - recent_opp_times[-1]) / 3
        if len(opp_clock_times) > 1:
             total_time_spent = (opp_clock_times[0] - opp_clock_times[-1]) / (len(opp_clock_times) - 1)
        else:
             total_time_spent = recent_time_spent # Fallback if only recent times available

        # Weighted average of recent and overall opponent move time
        target_opp_time_spent = 0.8 * recent_time_spent + 0.2 * total_time_spent
        # Normalize for potential berserk
        berserk_ratio = max(opp_initial_time / max(self_initial_time, 1), 1)
        target_opp_time_spent /= berserk_ratio

        # Blend calculated time with opponent's pace (with some probability)
        if np.random.random() < 0.4:
            blend_factor = 0.4 # How much to blend towards opponent's pace
            time_taken = (1 - blend_factor) * time_taken + blend_factor * target_opp_time_spent
            logger.add_log(f"Time after reflective pacing (TargetOppTime={target_opp_time_spent:.3f}s): {time_taken:.3f}s\n")

    # Ensure minimum time and cap based on remaining clock time
    time_taken = max(time_taken, 0.1) # Absolute minimum thinking time
    time_taken = min(time_taken, own_time * 0.8) # Don't use more than 80% of remaining time
    logger.add_log(f"Final calculated time: {time_taken:.3f}s (Clamped between 0.1s and {own_time * 0.8:.3f}s)\n")
    return time_taken


def check_obvious_move(logger: Logger, state_manager: StateManager, analyzer: Analyzer, mood_manager: MoodManager) -> tuple[str | None, bool]:
    """
    Checks for obvious moves like forced moves, simple takebacks, or mate-in-one.
    Returns (move_uci, is_obvious). Updates mood_manager's opponent_just_blundered flag.
    """
    logger.add_log("Checking for obvious moves...\n")
    side = state_manager.get_side()
    current_board = state_manager.get_board()
    analysis = analyzer.get_stockfish_analysis()
    mood_manager.set_opponent_just_blundered(False) # Reset flag

    if side is None or analysis is None:
        logger.add_log("WARNING: Cannot check obvious moves, side or analysis missing.\n")
        return None, False

    # 1. Only Legal Move
    if len(analysis) == 1 and len(list(current_board.legal_moves)) == 1:
        obvious_move = analysis[0]['pv'][0].uci()
        logger.add_log(f"Found only legal move {obvious_move}. Obvious.\n")
        return obvious_move, True

    # 2. Obvious Takebacks
    prev_board = state_manager.get_prev_board()
    last_moves = state_manager.get_move_history()
    if prev_board and last_moves:
        last_opp_move_uci = last_moves[-1]
        best_move_uci = analysis[0]['pv'][0].uci()
        best_move_eval = extend_mate_score(analysis[0]['score'].pov(side).score(mate_score=2500))
        second_best_eval = extend_mate_score(analysis[1]['score'].pov(side).score(mate_score=2500)) if len(analysis) > 1 else -float('inf')
        cp_diff = best_move_eval - second_best_eval

        if is_takeback(prev_board, last_opp_move_uci, best_move_uci):
            logger.add_log(f"Detected top move {best_move_uci} is a takeback.\n")
            # Check if the opponent's capture was a blunder (capturing undefended piece)
            last_opp_move = chess.Move.from_uci(last_opp_move_uci)
            captured_piece_type = prev_board.piece_type_at(last_opp_move.to_square)
            capturing_piece_type = prev_board.piece_type_at(last_opp_move.from_square)

            if captured_piece_type is not None and capturing_piece_type is not None:
                 # Check if the captured square was defended by opponent before the capture
                 # This is tricky, let's simplify: check if captured piece value > capturing piece value significantly
                 # OR if the captured piece was simply hanging.
                 captured_value = PIECE_VALS.get(captured_piece_type, 0)
                 capturing_value = PIECE_VALS.get(capturing_piece_type, 0)
                 # Check threat level on the captured square *before* the capture
                 threat_level_before = calculate_threatened_levels(last_opp_move.to_square, prev_board)

                 # If opponent captured a piece that wasn't hanging or was worth much less
                 if threat_level_before <= 0.6 and captured_value > capturing_value + 0.5:
                     logger.add_log(f"Opponent captured a non-hanging/more valuable piece ({prev_board.san(last_opp_move)}). Potential blunder.\n")
                     mood_manager.set_opponent_just_blundered(True)
                     # Don't consider this an "obvious" takeback if it follows a blunder capture
                     logger.add_log("Takeback follows opponent blunder, not considered 'obvious'.\n")
                 else:
                     # Takeback is obvious if it's much better than the alternative OR by chance
                     if cp_diff > 100:
                         logger.add_log(f"Takeback {best_move_uci} is significantly better (CP diff: {cp_diff}). Obvious.\n")
                         return best_move_uci, True
                     elif np.random.random() < 0.7:
                         logger.add_log(f"Takeback {best_move_uci} is not significantly better (CP diff: {cp_diff}), but returning as obvious by chance.\n")
                         return best_move_uci, True
                     else:
                         logger.add_log(f"Takeback {best_move_uci} is not significantly better (CP diff: {cp_diff}), not returning as obvious.\n")
            else:
                 logger.add_log("Could not determine piece types for takeback blunder check.\n")


    # 3. Forced Mate-in-One
    # Check if current top move is mate-in-one
    current_mate_score = analysis[0]['score'].pov(side).mate()
    if current_mate_score == 1:
        # Check if the position *before* opponent's last move was mate-in-two,
        # and our current mate-in-one move was the required response then.
        fen_history = state_manager.get_fen_history()
        if len(fen_history) >= 2 and STOCKFISH:
            try:
                prev_opp_board = chess.Board(fen_history[-2])
                if prev_opp_board.turn != side: # Ensure it was opponent's turn
                    prev_opp_analysis = STOCKFISH.analyse(prev_opp_board, limit=chess.engine.Limit(depth=10, mate=2)) # Search deeper for mate
                    prev_mate_score = prev_opp_analysis['score'].pov(side).mate()
                    if prev_mate_score == 2 and len(prev_opp_analysis['pv']) >= 2:
                        required_response_uci = prev_opp_analysis['pv'][1].uci() # Our required move after their best move
                        current_mate_move_uci = analysis[0]['pv'][0].uci()
                        if current_mate_move_uci == required_response_uci:
                            logger.add_log(f"Found forced mate-in-one ({current_mate_move_uci}) following opponent's move from mate-in-two. Obvious.\n")
                            return current_mate_move_uci, True
                        else:
                             logger.add_log(f"Mate-in-one available ({current_mate_move_uci}), but not the forced line from previous mate-in-two (required {required_response_uci}).\n")
                    else:
                         logger.add_log(f"Mate-in-one available, but previous position was not mate-in-two (Mate score: {prev_mate_score}).\n")
                else:
                     logger.add_log("History mismatch: Fen at index -2 was not opponent's turn for mate check.\n")
            except (IndexError, ValueError, chess.engine.EngineError, KeyError) as e:
                logger.add_log(f"WARNING: Error during mate-in-one check: {e}\n")
        else:
             logger.add_log("Not enough history or Stockfish unavailable for mate-in-one check.\n")


    # No obvious move found
    logger.add_log("No obvious move found.\n")
    return None, False