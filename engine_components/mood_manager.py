import numpy as np
import chess
import chess.engine
from .logger import Logger
from .state_manager import StateManager
from .analyzer import Analyzer, STOCKFISH # Import shared STOCKFISH instance
from common.utils import extend_mate_score
from common.board_information import is_takeback, calculate_threatened_levels, PIECE_VALS

class MoodManager:
    """Manages the engine's 'mood' based on game state and recent events."""

    POSSIBLE_MOODS = ["confident", "cocky", "cautious", "tilted", "hurry", "flagging"]

    def __init__(self, logger: Logger, state_manager: StateManager, analyzer: Analyzer):
        self.logger = logger
        self.state_manager = state_manager
        self.analyzer = analyzer
        self.current_mood = "confident" # Default mood
        self.just_blundered = False # Flag if the engine thinks it just blundered
        self.opponent_just_blundered = False # Flag if the opponent just blundered

    def get_mood(self) -> str:
        """Returns the current mood."""
        return self.current_mood

    def set_just_blundered(self, status: bool):
        """Sets the just_blundered flag."""
        self.just_blundered = status

    def is_just_blundered(self) -> bool:
        """Checks if the engine thinks it just blundered."""
        return self.just_blundered

    def set_opponent_just_blundered(self, status: bool):
        """Sets the opponent_just_blundered flag."""
        self.opponent_just_blundered = status

    def did_opponent_just_blunder(self) -> bool:
        """Checks if the opponent just blundered."""
        return self.opponent_just_blundered

    def determine_mood(self):
        """
        Determines and updates the engine's mood based on the current game state.
        This replaces the original _set_mood logic.
        """
        self.logger.add_log("Determining mood from current game information.\n")
        self.just_blundered = False # Reset blunder flag at the start of determination
        self.opponent_just_blundered = False # Reset opponent blunder flag

        side = self.state_manager.get_side()
        if side is None:
            self.logger.add_log("WARNING: Engine side not set. Cannot reliably determine mood. Defaulting to 'confident'.\n")
            self.current_mood = "confident"
            return self.current_mood

        # --- Time Check ---
        self_initial_time = self.state_manager.get_info("self_initial_time", default=60) # Default to 60s if not set
        opp_initial_time = self.state_manager.get_info("opp_initial_time", default=60)
        self_clock_times = self.state_manager.get_info("self_clock_times", default=[])
        opp_clock_times = self.state_manager.get_info("opp_clock_times", default=[])
        own_time = self_clock_times[-1] if self_clock_times else self_initial_time
        opp_time = opp_clock_times[-1] if opp_clock_times else opp_initial_time
        own_time = max(own_time, 1) # Avoid division by zero
        opp_time = max(opp_time, 1)

        # Calculate low time thresholds with randomness
        self_low_time_threshold = self_initial_time * 0.1 + 15 * 0.7 + self_initial_time * np.random.randn() / 30
        opp_low_time_threshold = opp_initial_time * 0.1 + 15 * 0.7 + opp_initial_time * np.random.randn() / 30
        self_low_time_threshold = max(self_low_time_threshold, 5) # Minimum threshold
        opp_low_time_threshold = max(opp_low_time_threshold, 5)

        if own_time < self_low_time_threshold:
            self.logger.add_log(f"Own time {own_time:.1f}s is below threshold {self_low_time_threshold:.1f}s. Mood: hurry.\n")
            self.current_mood = "hurry"
            return self.current_mood
        self.logger.add_log(f"Own time {own_time:.1f}s > threshold {self_low_time_threshold:.1f}s. Not in hurry mode.\n")

        # --- Blunder Check (Self) ---
        current_analysis = self.analyzer.get_stockfish_analysis()
        if not current_analysis:
             self.logger.add_log("WARNING: No current analysis available for mood determination (self blunder check).\n")
             current_eval = 0 # Default eval if analysis missing
        else:
             current_eval = extend_mate_score(current_analysis[0]['score'].pov(side).score(mate_score=2500))

        fen_history = self.state_manager.get_fen_history()
        if len(fen_history) >= 3 and STOCKFISH: # Need previous state and engine
            self.logger.add_log("Checking for recent self-blunder.\n")
            try:
                # Get evaluation from 2 moves ago (our previous position)
                prev_own_board = chess.Board(fen_history[-3])
                if prev_own_board.turn == side: # Ensure it was our turn
                    prev_own_analysis = STOCKFISH.analyse(prev_own_board, limit=chess.engine.Limit(time=0.02))
                    prev_own_eval = extend_mate_score(prev_own_analysis['score'].pov(side).score(mate_score=2500))

                    eval_drop = prev_own_eval - current_eval
                    # Blunder condition: significant eval drop AND not currently winning massively
                    if eval_drop > 300 and current_eval < 200:
                        self.logger.add_log(f"Significant eval drop detected ({prev_own_eval} -> {current_eval}). Potential self-blunder.\n")
                        if np.random.random() < 0.8: # Probability of recognizing the blunder
                            self.just_blundered = True
                            self.logger.add_log("Recognized self-blunder. Mood: tilted.\n")
                            self.current_mood = "tilted"
                            return self.current_mood
                        else:
                            self.logger.add_log("Potential self-blunder occurred, but not recognized this time.\n")
                    else:
                         self.logger.add_log(f"No significant eval drop ({prev_own_eval} -> {current_eval}). No self-blunder detected.\n")
                else:
                     self.logger.add_log("History mismatch: Fen at index -3 was not our turn.\n")

            except (IndexError, ValueError, chess.engine.EngineError, KeyError) as e:
                self.logger.add_log(f"WARNING: Error during self-blunder check: {e}\n")
        else:
             self.logger.add_log("Not enough history or Stockfish unavailable for self-blunder check.\n")


        # --- Flagging Check (Opponent Low Time) ---
        if opp_time < opp_low_time_threshold:
            # Probability of entering flagging mode
            if np.random.random() < 0.7:
                self.logger.add_log(f"Opponent time {opp_time:.1f}s is below threshold {opp_low_time_threshold:.1f}s. Mood: flagging.\n")
                self.current_mood = "flagging"
                return self.current_mood
            else:
                self.logger.add_log(f"Opponent time {opp_time:.1f}s < threshold {opp_low_time_threshold:.1f}s, but not entering flagging mode by chance.\n")
        else:
            self.logger.add_log(f"Opponent time {opp_time:.1f}s > threshold {opp_low_time_threshold:.1f}s. Not in flagging mode.\n")

        # --- Cocky Check ---
        # Conditions: Blitzing opening OR significantly ahead in eval and time
        time_spent = self_initial_time - own_time
        opening_blitz_threshold = self_low_time_threshold / 2 # Time spent threshold for opening blitz
        if time_spent < opening_blitz_threshold:
             self.logger.add_log(f"Still in opening blitz phase (time spent {time_spent:.1f}s < threshold {opening_blitz_threshold:.1f}s). Mood: cocky.\n")
             self.current_mood = "cocky"
             return self.current_mood

        significant_time_lead = self_initial_time / 6
        if current_eval > 300 and (own_time - opp_time) > significant_time_lead:
            self.logger.add_log(f"Significant lead in eval ({current_eval}) and time ({own_time - opp_time:.1f}s > {significant_time_lead:.1f}s). Mood: cocky.\n")
            self.current_mood = "cocky"
            return self.current_mood
        self.logger.add_log(f"Eval ({current_eval}) or time lead ({own_time - opp_time:.1f}s vs {significant_time_lead:.1f}s) not sufficient for cocky mode.\n")


        # --- Cautious Check ---
        # Conditions: Relatively even eval, complex position (low effective mobility but not forced)
        complexity = self.analyzer.get_lucas_metric("complexity", default=50)
        eff_mob = self.analyzer.get_lucas_metric("eff_mob", default=20)

        is_even = abs(current_eval) < 250
        is_complex_mobility = 2 < eff_mob < 15 # Not forced (eff_mob > 2), but limited options

        if is_even and is_complex_mobility:
            # Check if top moves are obvious takebacks (less likely to be cautious then)
            is_obvious_takeback_scenario = False
            move_history = self.state_manager.get_move_history()
            prev_board = self.state_manager.get_prev_board()
            if move_history and prev_board and len(current_analysis) >= 2:
                try:
                    prev_move_uci = move_history[-1]
                    top_move_uci = current_analysis[0]['pv'][0].uci()
                    second_move_uci = current_analysis[1]['pv'][0].uci()
                    if is_takeback(prev_board, prev_move_uci, top_move_uci) and \
                       is_takeback(prev_board, prev_move_uci, second_move_uci):
                        is_obvious_takeback_scenario = True
                        self.logger.add_log("Top two moves are takebacks, reducing chance of cautious mood.\n")
                except (KeyError, IndexError, ValueError):
                     self.logger.add_log("Warning: Could not check takeback status for cautious mood determination.\n")


            if not is_obvious_takeback_scenario:
                 # Probability of becoming cautious increases with complexity, decreases with mobility
                 cautious_prob = (0.35 + complexity / (100 * eff_mob + 100)) ** 0.6
                 if np.random.random() < cautious_prob:
                     self.logger.add_log(f"Even eval ({current_eval}), complex mobility (eff_mob={eff_mob}, complexity={complexity}). Mood: cautious (Prob: {cautious_prob:.2f}).\n")
                     self.current_mood = "cautious"
                     return self.current_mood
                 else:
                     self.logger.add_log(f"Even/complex position, but not entering cautious mode by chance (Prob: {cautious_prob:.2f}).\n")
            else:
                 self.logger.add_log("Even/complex position, but obvious takebacks present. Not entering cautious mode.\n")

        else:
             self.logger.add_log(f"Position not even ({current_eval}) or mobility ({eff_mob}) outside complex range. Not entering cautious mode.\n")


        # --- Default ---
        self.logger.add_log("No specific mood conditions met. Defaulting to 'confident'.\n")
        self.current_mood = "confident"
        return self.current_mood

    def check_opponent_blunder(self):
        """
        Checks if the opponent made a significant blunder on their last move.
        Updates self.opponent_just_blundered.
        """
        self.opponent_just_blundered = False # Reset flag
        side = self.state_manager.get_side()
        fen_history = self.state_manager.get_fen_history()
        move_history = self.state_manager.get_move_history()

        if side is None or len(fen_history) < 2 or len(move_history) < 1 or not STOCKFISH:
            self.logger.add_log("Insufficient info or Stockfish unavailable for opponent blunder check.\n")
            return

        try:
            current_board = self.state_manager.get_board()
            current_analysis = self.analyzer.get_stockfish_analysis()
            if not current_analysis:
                 self.logger.add_log("WARNING: No current analysis available for opponent blunder check.\n")
                 return

            current_eval = extend_mate_score(current_analysis[0]['score'].pov(side).score(mate_score=2500))

            # Get opponent's previous position (before their last move)
            prev_opp_board = chess.Board(fen_history[-2])
            if prev_opp_board.turn == side: # Should be opponent's turn
                 self.logger.add_log("History mismatch: Fen at index -2 was our turn during opponent blunder check.\n")
                 return

            prev_opp_analysis = STOCKFISH.analyse(prev_opp_board, limit=chess.engine.Limit(time=0.02))
            # Evaluation from OUR perspective before opponent's move
            prev_opp_eval = extend_mate_score(prev_opp_analysis['score'].pov(side).score(mate_score=2500))

            eval_gain = current_eval - prev_opp_eval
            # Opponent blunder condition: significant eval gain for us
            if eval_gain > 150:
                self.logger.add_log(f"Significant eval gain detected ({prev_opp_eval} -> {current_eval}). Potential opponent blunder.\n")
                # Check if it involved hanging the piece they just moved
                last_opp_move_uci = move_history[-1]
                last_opp_move = chess.Move.from_uci(last_opp_move_uci)
                piece_moved_value = PIECE_VALS.get(prev_opp_board.piece_type_at(last_opp_move.from_square), 0)

                # Check if the landing square is now significantly threatened
                # Use current_board state AFTER opponent's move
                threat_level = calculate_threatened_levels(last_opp_move.to_square, current_board)

                # Consider it a hung piece if threat level is high relative to piece value
                # (e.g., threat > 0.6 means at least a pawn threat, higher for more valuable pieces)
                # Or simply if threat level is high (e.g. >= 3 points)
                if threat_level >= 3 or (threat_level > 0.6 and threat_level >= piece_moved_value * 0.8):
                    self.logger.add_log(f"Opponent's last move {last_opp_move_uci} to {chess.square_name(last_opp_move.to_square)} seems to hang the piece (threat level: {threat_level}). Setting opponent_just_blundered.\n")
                    self.opponent_just_blundered = True
                else:
                     self.logger.add_log(f"Opponent blundered based on eval, but move {last_opp_move_uci} to {chess.square_name(last_opp_move.to_square)} doesn't appear to hang the piece (threat level: {threat_level}).\n")
            else:
                self.logger.add_log(f"No significant eval gain ({prev_opp_eval} -> {current_eval}). No opponent blunder detected.\n")

        except (IndexError, ValueError, chess.engine.EngineError, KeyError) as e:
            self.logger.add_log(f"WARNING: Error during opponent blunder check: {e}\n")