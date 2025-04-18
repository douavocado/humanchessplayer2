import time
import numpy as np
import chess
from .logger import Logger
from .state_manager import StateManager
from .scorers import Scorers
from .mood_manager import MoodManager
from .analyzer import Analyzer # Needed for eval access in adjust_human_prob
from common.board_information import (
    phase_of_game, PIECE_VALS, get_threatened_board, is_capturing_move,
    is_capturable, is_attacked_by_pinned, is_check_move, is_takeback,
    is_newly_attacked, is_offer_exchange, king_danger, is_open_file,
    calculate_threatened_levels, is_weird_move
)
from common.utils import flip_uci, patch_fens, extend_mate_score

class HumanMoveLogic:
    def __init__(self, logger: Logger, state_manager: StateManager, scorers: Scorers, mood_manager: MoodManager, analyzer: Analyzer):
        self.logger = logger
        self.state_manager = state_manager
        self.scorers = scorers
        self.mood_manager = mood_manager
        self.analyzer = analyzer # Store analyzer for eval access

    def adjust_human_prob(self, move_dic: dict, board: chess.Board) -> dict:
        """
        Normalizes human move probabilities based on time pressure, game stage,
        and evaluation advantage.
        """
        self.logger.add_log("Adjusting raw human probabilities...\n")
        power_factor = 1.0
        eps = 1e-10 # Small epsilon to avoid log(0) or division by zero issues
        return_move_dic = {k: np.clip(v, eps, 1 - eps) for k, v in move_dic.items()}

        mood = self.mood_manager.get_mood()
        own_time = max(self.state_manager.get_info("self_clock_times", [1])[-1], 1)
        initial_time = max(self.state_manager.get_info("self_initial_time", 1), 1)

        # Time pressure normalization
        if mood == "hurry":
            # The less time we have, the more probabilities are pushed towards uniform (less extreme)
            # Power factor > 1 flattens probabilities (p^(1/factor))
            time_norm_factor = np.sqrt(initial_time / own_time)
            power_factor *= time_norm_factor
            self.logger.add_log(f"Applying time pressure normalization (Factor: {time_norm_factor:.2f}).\n")

        # Endgame normalization (less extreme probabilities in simpler positions)
        pieces_left = len(list(chess.SquareSet(board.occupied)))
        if pieces_left <= 18:
            endgame_norm_factor = (19 - pieces_left) / 5.0
            power_factor *= max(endgame_norm_factor, 1.0) # Ensure factor >= 1
            self.logger.add_log(f"Applying endgame normalization (Pieces: {pieces_left}, Factor: {endgame_norm_factor:.2f}).\n")


        # Evaluation normalization (push towards better moves when winning significantly)
        analysis = self.analyzer.get_stockfish_analysis()
        side = self.state_manager.get_side()
        if analysis and side is not None:
            eval_ = extend_mate_score(analysis[0]['score'].pov(side).score(mate_score=2500))
            if eval_ > 500: # Significantly winning
                # Power factor > 1 flattens probabilities
                eval_norm_factor = (eval_ - 100) / 400.0
                power_factor *= max(eval_norm_factor, 1.0) # Ensure factor >= 1
                self.logger.add_log(f"Applying winning eval normalization (Eval: {eval_}, Factor: {eval_norm_factor:.2f}).\n")
        else:
             self.logger.add_log("WARNING: No analysis or side info for eval normalization.\n")


        self.logger.add_log(f"Total normalization power factor: {power_factor:.2f}\n")
        if power_factor <= 0: # Prevent invalid power
             self.logger.add_log(f"WARNING: Invalid power factor ({power_factor}). Skipping normalization power step.\n")
        else:
             # Apply the combined power factor (p^(1/factor))
             # Ensure probabilities don't become NaN or Inf
             try:
                 return_move_dic = {k: v**(1.0 / power_factor) for k, v in return_move_dic.items()}
             except (ValueError, OverflowError) as e:
                  self.logger.add_log(f"ERROR during power normalization: {e}. Returning un-powered probabilities.\n")
                  return_move_dic = {k: v for k, v in move_dic.items()} # Revert to original clipped


        # Re-normalize probabilities to sum to 1
        total = sum(return_move_dic.values())
        if total <= eps: # Avoid division by zero if all probabilities became tiny
             self.logger.add_log("WARNING: Sum of adjusted probabilities is near zero. Returning uniform distribution.\n")
             num_moves = len(return_move_dic)
             return {k: 1.0/num_moves for k in return_move_dic} if num_moves > 0 else {}

        return_move_dic = {k: v / total for k, v in return_move_dic.items()}
        return return_move_dic

    def get_human_probabilities(self, board: chess.Board, game_phase: str) -> dict:
        """
        Gets the raw move probabilities from the appropriate neural network scorer
        for the given board state and game phase.
        """
        self.logger.add_log(f"Getting raw human probabilities for phase: {game_phase}.\n")
        scorer = self.scorers.get_human_scorer(game_phase)
        if scorer is None:
            self.logger.add_log(f"ERROR: No scorer available for phase {game_phase}. Returning empty dict.\n")
            return {}

        # The model is trained from White's perspective, flip if Black's turn
        if board.turn == chess.WHITE:
            dummy_board = board.copy()
            needs_flip = False
        else: # board.turn == chess.BLACK
            dummy_board = board.mirror()
            needs_flip = True

        try:
            # Get probabilities from the model
            _, nn_top_move_dic = scorer.get_move_dic(dummy_board, san=False, top=100) # Get top 100 candidates
        except Exception as e:
             self.logger.add_log(f"ERROR: Exception during MoveScorer.get_move_dic: {e}\n")
             return {}


        # Flip UCIs back if the board was mirrored
        if needs_flip:
            flipped_dic = {}
            for k, v in nn_top_move_dic.items():
                try:
                    flipped_dic[flip_uci(k)] = v
                except ValueError:
                     self.logger.add_log(f"WARNING: Could not flip invalid UCI '{k}' from scorer.\n")
            nn_top_move_dic = flipped_dic


        # Filter out illegal moves that might have slipped through the model
        legal_moves_uci = {move.uci() for move in board.legal_moves}
        filtered_move_dic = {uci: prob for uci, prob in nn_top_move_dic.items() if uci in legal_moves_uci}


        # Normalize initial probabilities to sum to 1
        total = sum(filtered_move_dic.values())
        eps = 1e-10
        if total <= eps:
            self.logger.add_log("WARNING: Sum of raw probabilities is near zero. Cannot normalize.\n")
            # Return uniform probability for legal moves if possible
            num_legal = len(legal_moves_uci)
            return {uci: 1.0/num_legal for uci in legal_moves_uci} if num_legal > 0 else {}

        normalized_move_dic = {k: v / total for k, v in filtered_move_dic.items()}

        # Log the raw probabilities before time/eval adjustment
        log_move_dic_san = {}
        try:
            log_move_dic_san = {board.san(chess.Move.from_uci(k)): round(v, 5) for k, v in normalized_move_dic.items()}
        except Exception as e:
             self.logger.add_log(f"WARNING: Error generating SAN for logging raw probabilities: {e}\n")
             log_move_dic_san = {k: round(v, 5) for k, v in normalized_move_dic.items()} # Log UCIs instead
        self.logger.add_log(f"Raw normalized move probabilities: {log_move_dic_san}\n")

        # Adjust probabilities based on time, eval, etc.
        adjusted_move_dic = self.adjust_human_prob(normalized_move_dic, board)

        # Log the adjusted probabilities
        log_adj_move_dic_san = {}
        try:
            log_adj_move_dic_san = {board.san(chess.Move.from_uci(k)): round(v, 5) for k, v in adjusted_move_dic.items()}
        except Exception as e:
             self.logger.add_log(f"WARNING: Error generating SAN for logging adjusted probabilities: {e}\n")
             log_adj_move_dic_san = {k: round(v, 5) for k, v in adjusted_move_dic.items()} # Log UCIs instead
        self.logger.add_log(f"Adjusted normalized move probabilities: {log_adj_move_dic_san}\n")

        return adjusted_move_dic


    def alter_move_probabilties(self, move_dic: dict, board: chess.Board) -> dict:
        """
        Applies heuristic adjustments to move probabilities based on tactical
        and positional features (captures, checks, threats, king safety, etc.).
        """
        start_time = time.time()
        self.logger.add_log("Applying heuristic alterations to move probabilities...\n")
        if not move_dic:
             self.logger.add_log("No moves in dictionary to alter. Returning empty.\n")
             return {}

        # --- Configuration & Setup ---
        game_phase = phase_of_game(board)
        mood = self.mood_manager.get_mood()
        side = board.turn # Alterations are from the perspective of the current player

        # Get previous board states if available
        prev_board = self.state_manager.get_prev_board()
        prev_prev_board = self.state_manager.get_prev_prev_board()

        # Calculate king danger levels
        self_king_danger_lvl = king_danger(board, side, game_phase)
        opp_king_danger_lvl = king_danger(board, not side, game_phase)
        self.logger.add_log(f"King Danger Levels: Self={self_king_danger_lvl}, Opp={opp_king_danger_lvl}\n")

        # Threshold for boosting low-probability moves that become attractive
        lower_threshold_prob = sum(move_dic.values()) / len(move_dic) if move_dic else 0

        # --- Heuristic Scaling Factors (Mood Dependent) ---
        # Define base scaling factors
        sf = {
            "protect_king": 2.8,
            "capture_en_pris": 1.5,
            "break_pin": 3.0,
            "capture": 1.5,
            "capturable_move": 1.3,
            "takeback": 2.5,
            "passed_pawn_end": 3.0,
            "weird_move": {"opening": 0.1, "midgame": 0.3, "endgame": 1.0}, # Penalty factor
            # Mood-dependent factors
            "check": {"confident": 2.3, "cocky": 3.3, "cautious": 2.1, "tilted": 3.3, "hurry": 3.0, "flagging": 2.9},
            "new_threatened": {"confident": 3.5, "cocky": 2.7, "cautious": 3.9, "tilted": 2.1, "hurry": 2.4, "flagging": 3.2},
            "exchange_material": {"confident": 3.2, "cocky": 2.0, "cautious": 2.2, "tilted": 1.5, "hurry": 3.8, "flagging": 0.8}, # For when ahead/behind
            "exchange_king_danger": {"confident": 3.4, "cocky": 2.1, "cautious": 3.9, "tilted": 1.0, "hurry": 3.0, "flagging": 2.5}, # For when king is unsafe
            "repeat_move": {"confident": 0.3, "cocky": 0.5, "cautious": 0.3, "tilted": 0.6, "hurry": 0.5, "flagging": 0.4} # Penalty factor
        }

        # Adjust factors based on context (e.g., opponent king danger)
        if opp_king_danger_lvl > 500:
            sf["capture"] *= 1.5 # Boost captures more if opponent king is weak
        if opp_king_danger_lvl - self_king_danger_lvl > 500:
            sf["check"] = {k: v * 1.3 for k, v in sf["check"].items()} # Boost checks more

        # --- Apply Heuristics ---
        altered_dic = move_dic.copy() # Work on a copy
        logged_categories = {cat: [] for cat in [ # Track moves affected by each heuristic for logging
            "strengthening", "weakening", "weird", "protect_king", "capture_en_pris",
            "break_pin", "capture", "capturable_move", "check", "takeback",
            "new_threatened", "good_exchange_mat", "bad_exchange_mat",
            "good_exchange_king", "bad_exchange_king", "passed_pawn", "repeat_move"
        ]}

        # 1. Threat/Protection Analysis (Complex Heuristic)
        try:
            # Calculate current threat levels (ignore pawns for speed)
            piece_types_major = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
            curr_threatened_board_self = get_threatened_board(board, colour=side, piece_types=piece_types_major)
            self_curr_threatened_levels = sum(curr_threatened_board_self)
            opp_curr_threatened_levels = sum(get_threatened_board(board, colour=(not side), piece_types=piece_types_major))

            # Exaggerate if only one piece is threatened
            solo_factor = 2.0 if sum(np.array(curr_threatened_board_self) > 0.5) == 1 else 1.0

            for move_uci in altered_dic.keys():
                move_obj = chess.Move.from_uci(move_uci)
                dummy_board = board.copy()
                captured_piece_type = board.piece_type_at(move_obj.to_square)
                captured_value = PIECE_VALS.get(captured_piece_type, 0)

                dummy_board.push(move_obj)

                # Calculate new threat levels after the move
                new_threatened_board_self = get_threatened_board(dummy_board, colour=side, piece_types=piece_types_major)
                self_new_threatened_levels = sum(new_threatened_board_self)

                # Adjust opponent threat calculation if our move hangs the moved piece
                # This prevents rewarding attacks that sacrifice the attacker unnecessarily
                moving_piece_threat_after = new_threatened_board_self[move_obj.to_square]
                if moving_piece_threat_after > 0.6: # If the piece we moved is now attacked
                     # Calculate opponent threats as if our moved piece wasn't there
                     temp_board_no_attacker = dummy_board.copy()
                     temp_board_no_attacker.remove_piece_at(move_obj.to_square)
                     opp_new_threatened_levels = sum(get_threatened_board(temp_board_no_attacker, colour=(not side), piece_types=piece_types_major))
                else:
                     opp_new_threatened_levels = sum(get_threatened_board(dummy_board, colour=(not side), piece_types=piece_types_major))


                # Calculate change in threat levels
                self_lvl_diff = self_new_threatened_levels - self_curr_threatened_levels
                opp_lvl_diff = opp_new_threatened_levels - opp_curr_threatened_levels

                # Combine changes: protecting self is weighted slightly more than attacking opponent
                # Add back captured value if it was a capture
                total_lvl_diff = (opp_lvl_diff + captured_value) - (self_lvl_diff * 1.25 * solo_factor)
                if total_lvl_diff > 100:
                    print(f"Total level diff: {total_lvl_diff}")
                    print(f"Self level diff: {self_lvl_diff}")
                    print(f"Opp level diff: {opp_lvl_diff}")
                    print(f"Captured value: {captured_value}")
                    print(f"Solo factor: {solo_factor}")
                    print(f"Move: {move_uci}")
                    print(f"Move object: {move_obj}")
                    
                factor = np.clip(np.exp(total_lvl_diff / 2.0), 0.0, 1e10) # Exponential scaling

                # Boost probability floor for significantly improving moves
                if total_lvl_diff > 0.9 and altered_dic[move_uci] < lower_threshold_prob:
                    altered_dic[move_uci] = lower_threshold_prob

                altered_dic[move_uci] *= factor

                # Log categorization
                move_san = board.san(move_obj)
                if total_lvl_diff > 0.9: logged_categories["strengthening"].append(move_san)
                elif total_lvl_diff < -0.9: logged_categories["weakening"].append(move_san)
                else:
                    # Check for weird moves only if threat level didn't change much
                    if is_weird_move(board, game_phase, move_uci, self_king_danger_lvl):
                         weird_factor = sf["weird_move"].get(game_phase, 0.5) # Default penalty 0.5
                         altered_dic[move_uci] *= weird_factor
                         logged_categories["weird"].append(move_san)

        except Exception as e:
             self.logger.add_log(f"ERROR during threat/protection analysis: {e}\n")


        # --- Apply Simpler Heuristics ---
        for move_uci in altered_dic.keys():
            move_obj = chess.Move.from_uci(move_uci)
            move_san = board.san(move_obj) # Get SAN once for logging

            # 2. Break Pin
            to_square = move_obj.to_square
            pinned_attackers = is_attacked_by_pinned(board, to_square, not side)
            if pinned_attackers > 0:
                if altered_dic[move_uci] < lower_threshold_prob: altered_dic[move_uci] = lower_threshold_prob
                altered_dic[move_uci] *= (sf["break_pin"] ** pinned_attackers)
                logged_categories["break_pin"].append(move_san)

            # 3. Protect King (if in danger)
            if self_king_danger_lvl > 250:
                dummy_board = board.copy()
                dummy_board.push(move_obj)
                new_king_danger = king_danger(dummy_board, side, game_phase)
                danger_reduction = self_king_danger_lvl / max(new_king_danger, 50) # Avoid division by zero
                if new_king_danger <= 0 or danger_reduction > 1.5:
                    if altered_dic[move_uci] < lower_threshold_prob: altered_dic[move_uci] = lower_threshold_prob
                    king_factor = sf["protect_king"] * (danger_reduction**(1/4))
                    altered_dic[move_uci] *= king_factor
                    logged_categories["protect_king"].append(f"{move_san} (Factor: {king_factor:.2f})")


            # 4. Captures (General Boost)
            if board.is_capture(move_obj):
                captured_piece_type = board.piece_type_at(move_obj.to_square)
                piece_value = PIECE_VALS.get(captured_piece_type, 1) # Default to pawn value if unknown
                capture_factor = sf["capture"] * (piece_value**0.25)
                altered_dic[move_uci] *= capture_factor
                logged_categories["capture"].append(f"{move_san} (Factor: {capture_factor:.2f})")

                # 5. Capture En Pris Piece (Specific Boost)
                threatened_lvl_target = calculate_threatened_levels(move_obj.to_square, board)
                # Check if target is hanging AND capturing piece isn't hanging itself after capture
                dummy_board = board.copy(); dummy_board.push(move_obj)
                threat_level_attacker_after = calculate_threatened_levels(move_obj.to_square, dummy_board)

                if threatened_lvl_target > 0.6 and threat_level_attacker_after < captured_value + 0.1: # Target hanging, capture is safe enough
                    enpris_factor = sf["capture_en_pris"] * (threatened_lvl_target**0.25)
                    altered_dic[move_uci] *= enpris_factor
                    logged_categories["capture_en_pris"].append(f"{move_san} (Factor: {enpris_factor:.2f})")


            # 6. Move Capturable Piece
            from_square = move_obj.from_square
            if is_capturable(board, from_square):
                altered_dic[move_uci] *= sf["capturable_move"]
                logged_categories["capturable_move"].append(move_san)

            # 7. Checks
            if board.gives_check(move_obj):
                check_factor = sf["check"].get(mood, 2.0) # Default check factor
                altered_dic[move_uci] *= check_factor
                logged_categories["check"].append(f"{move_san} (Factor: {check_factor:.2f})")

            # 8. Takebacks (Requires previous board)
            if prev_board:
                res = patch_fens(prev_board.fen(), board.fen(), depth_lim=1)
                if res:
                    last_opp_move_uci = res[0][0]
                    if is_takeback(prev_board, last_opp_move_uci, move_uci):
                        altered_dic[move_uci] *= sf["takeback"]
                        logged_categories["takeback"].append(move_san)

            # 9. Respond to New Threat (Requires previous board)
            if prev_board:
                moving_piece_square = move_obj.from_square
                newly_attacked_value = is_newly_attacked(prev_board, board, moving_piece_square)
                if newly_attacked_value > 0.6: # If the piece we are moving was newly attacked
                    # Check if the move actually saves the piece
                    dummy_board = board.copy(); dummy_board.push(move_obj)
                    threat_after_move = calculate_threatened_levels(move_obj.to_square, dummy_board)
                    if threat_after_move < newly_attacked_value - 0.1: # If threat level decreased
                         threat_factor = sf["new_threatened"].get(mood, 2.5) * (1 + newly_attacked_value)**0.2
                         altered_dic[move_uci] *= threat_factor
                         logged_categories["new_threatened"].append(f"{move_san} (Factor: {threat_factor:.2f})")


            # 10. Exchanges (Material Advantage/Disadvantage)
            mat_dic = {chess.PAWN: 1, chess.KNIGHT: 3.1, chess.BISHOP: 3.5, chess.ROOK: 5.5, chess.QUEEN: 9.9, chess.KING: 0}
            own_mat = sum(len(board.pieces(pt, side)) * mat_dic[pt] for pt in mat_dic)
            opp_mat = sum(len(board.pieces(pt, not side)) * mat_dic[pt] for pt in mat_dic)
            material_diff = own_mat - opp_mat
            is_exchange = is_offer_exchange(board, move_uci)

            if is_exchange:
                exchange_mat_factor = sf["exchange_material"].get(mood, 2.0)
                if material_diff > 2.9: # Ahead -> Encourage trades
                    altered_dic[move_uci] *= exchange_mat_factor
                    logged_categories["good_exchange_mat"].append(f"{move_san} (Factor: {exchange_mat_factor:.2f})")
                elif material_diff < -2.9: # Behind -> Discourage trades
                    altered_dic[move_uci] /= exchange_mat_factor # Apply penalty
                    logged_categories["bad_exchange_mat"].append(f"{move_san} (Factor: {1/exchange_mat_factor:.2f})")


            # 11. Exchanges (King Danger)
            if is_exchange:
                exchange_king_factor = sf["exchange_king_danger"].get(mood, 2.0)
                king_danger_diff = self_king_danger_lvl - opp_king_danger_lvl
                if king_danger_diff >= 400 and self_king_danger_lvl > 500: # Our king much less safe -> Encourage trades
                    altered_dic[move_uci] *= exchange_king_factor
                    logged_categories["good_exchange_king"].append(f"{move_san} (Factor: {exchange_king_factor:.2f})")
                elif king_danger_diff <= -400 and opp_king_danger_lvl > 500: # Opponent king much less safe -> Discourage trades
                    altered_dic[move_uci] /= exchange_king_factor # Apply penalty
                    logged_categories["bad_exchange_king"].append(f"{move_san} (Factor: {1/exchange_king_factor:.2f})")


            # 12. Push Passed Pawn (Endgame)
            if game_phase == "endgame":
                move_piece_type = board.piece_type_at(move_obj.from_square)
                if move_piece_type == chess.PAWN:
                    passed_result = is_open_file(board, chess.square_file(move_obj.from_square))
                    if board.turn == chess.WHITE and passed_result == -2:                    
                        altered_dic[move_uci] *= sf["passed_pawn_end"]
                        logged_categories["passed_pawn"].append(board.san(move_obj))
                    elif board.turn == chess.BLACK and passed_result == 2:
                        altered_dic[move_uci] *= sf["passed_pawn_end"]
                        logged_categories["passed_pawn"].append(move_san)


            # 13. Repeat Moves (Penalty - Requires two previous boards)
            if prev_board and prev_prev_board:
                 res = patch_fens(prev_prev_board.fen(), prev_board.fen(), depth_lim=1)
                 if res:
                     last_own_move_uci = res[0][0]
                     last_own_move_obj = chess.Move.from_uci(last_own_move_uci)
                     # Check if the current move starts where the last one ended
                     if move_obj.from_square == last_own_move_obj.to_square:
                         # Check if the current move's destination was reachable from the piece's original square
                         squares_reachable_before = {
                             m.to_square for m in prev_prev_board.legal_moves
                             if m.from_square == last_own_move_obj.from_square
                         } | {last_own_move_obj.from_square} # Include original square

                         if move_obj.to_square in squares_reachable_before:
                             repeat_factor = sf["repeat_move"].get(mood, 0.4)
                             altered_dic[move_uci] *= repeat_factor
                             logged_categories["repeat_move"].append(f"{move_san} (Factor: {repeat_factor:.2f})")


        # --- Final Normalization & Logging ---
        # Log the effects of heuristics
        for category, moves in logged_categories.items():
            if moves:
                self.logger.add_log(f"Moves affected by '{category}': {', '.join(moves)}\n")

        # Normalize final probabilities
        total = sum(altered_dic.values())
        eps = 1e-10
        if total <= eps:
            self.logger.add_log("WARNING: Sum of altered probabilities is near zero. Returning uniform distribution.\n")
            num_moves = len(altered_dic)
            final_dic = {k: 1.0/num_moves for k in altered_dic} if num_moves > 0 else {}
        else:
            final_dic = {k: v / total for k, v in altered_dic.items()}

        # Sort by probability descending
        sorted_final_dic = dict(sorted(final_dic.items(), key=lambda item: item[1], reverse=True))

        end_time = time.time()
        # Log final altered probabilities
        log_final_dic_san = {}
        try:
            log_final_dic_san = {board.san(chess.Move.from_uci(k)): round(v, 5) for k, v in sorted_final_dic.items()}
        except Exception as e:
             self.logger.add_log(f"WARNING: Error generating SAN for logging final probabilities: {e}\n")
             log_final_dic_san = {k: round(v, 5) for k, v in sorted_final_dic.items()} # Log UCIs instead

        self.logger.add_log(f"Final altered move probabilities: {log_final_dic_san}\n")
        self.logger.add_log(f"Move probability alterations finished in {end_time - start_time:.4f} seconds.\n")

        return sorted_final_dic