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

    def adjust_human_prob(self, move_dic: dict, board: chess.Board, log: bool = True) -> dict:
        """
        Normalizes human move probabilities based on time pressure, game stage,
        and evaluation advantage.
        """
        if log:
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
            if log:
                self.logger.add_log(f"Applying time pressure normalization (Factor: {time_norm_factor:.2f}).\n")

        # Endgame normalization (less extreme probabilities in simpler positions)
        pieces_left = len(list(chess.SquareSet(board.occupied)))
        if pieces_left <= 18:
            endgame_norm_factor = (19 - pieces_left) / 5.0
            power_factor *= max(endgame_norm_factor, 1.0) # Ensure factor >= 1
            if log:
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
                if log:
                    self.logger.add_log(f"Applying winning eval normalization (Eval: {eval_}, Factor: {eval_norm_factor:.2f}).\n")
        else:
            if log:
                self.logger.add_log("WARNING: No analysis or side info for eval normalization.\n")


        if log:
            self.logger.add_log(f"Total normalization power factor: {power_factor:.2f}\n")
        if power_factor <= 0: # Prevent invalid power
            if log:
                self.logger.add_log(f"WARNING: Invalid power factor ({power_factor}). Skipping normalization power step.\n")
        else:
             # Apply the combined power factor (p^(1/factor))
             # Ensure probabilities don't become NaN or Inf
            try:
                return_move_dic = {k: v**(1.0 / power_factor) for k, v in return_move_dic.items()}
            except (ValueError, OverflowError) as e:
                if log:
                    self.logger.add_log(f"ERROR during power normalization: {e}. Returning un-powered probabilities.\n")
                return_move_dic = {k: v for k, v in move_dic.items()} # Revert to original clipped


        # Re-normalize probabilities to sum to 1
        total = sum(return_move_dic.values())
        if total <= eps: # Avoid division by zero if all probabilities became tiny
            if log:
                self.logger.add_log("WARNING: Sum of adjusted probabilities is near zero. Returning uniform distribution.\n")
            num_moves = len(return_move_dic)
            return {k: 1.0/num_moves for k in return_move_dic} if num_moves > 0 else {}

        return_move_dic = {k: v / total for k, v in return_move_dic.items()}
        return return_move_dic

    def get_human_probabilities(self, board: chess.Board, game_phase: str, log: bool = True) -> dict:
        """
        Gets the raw move probabilities from the appropriate neural network scorer
        for the given board state and game phase.
        """
        if log:
            self.logger.add_log(f"Getting raw human probabilities for phase: {game_phase}.\n")
        scorer = self.scorers.get_human_scorer(game_phase)
        if scorer is None:
            if log:
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
            if log:
                self.logger.add_log(f"ERROR: Exception during MoveScorer.get_move_dic: {e}\n")
            return {}


        # Flip UCIs back if the board was mirrored
        if needs_flip:
            flipped_dic = {}
            for k, v in nn_top_move_dic.items():
                try:
                    flipped_dic[flip_uci(k)] = v
                except ValueError:
                    if log:
                        self.logger.add_log(f"WARNING: Could not flip invalid UCI '{k}' from scorer.\n")
            nn_top_move_dic = flipped_dic


        # Filter out illegal moves that might have slipped through the model
        legal_moves_uci = {move.uci() for move in board.legal_moves}
        filtered_move_dic = {uci: prob for uci, prob in nn_top_move_dic.items() if uci in legal_moves_uci}


        # Normalize initial probabilities to sum to 1
        total = sum(filtered_move_dic.values())
        eps = 1e-10
        if total <= eps:
            if log:
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
            if log:
                self.logger.add_log(f"WARNING: Error generating SAN for logging raw probabilities: {e}\n")
            log_move_dic_san = {k: round(v, 5) for k, v in normalized_move_dic.items()} # Log UCIs instead
        if log:
            self.logger.add_log(f"Raw normalized move probabilities: {log_move_dic_san}\n")

        # Adjust probabilities based on time, eval, etc.
        adjusted_move_dic = self.adjust_human_prob(normalized_move_dic, board, log=log)

        # Log the adjusted probabilities
        log_adj_move_dic_san = {}
        try:
            log_adj_move_dic_san = {board.san(chess.Move.from_uci(k)): round(v, 5) for k, v in adjusted_move_dic.items()}
        except Exception as e:
             if log:
                self.logger.add_log(f"WARNING: Error generating SAN for logging adjusted probabilities: {e}\n")
             log_adj_move_dic_san = {k: round(v, 5) for k, v in adjusted_move_dic.items()} # Log UCIs instead
        if log:
            self.logger.add_log(f"Adjusted normalized move probabilities: {log_adj_move_dic_san}\n")

        return adjusted_move_dic


    def alter_move_probabilties(self, move_dic: dict, board: chess.Board, log: bool = True, prev_board: chess.Board = None, prev_prev_board: chess.Board = None) -> dict:
        """
        Applies heuristic adjustments to move probabilities based on tactical
        and positional features with performance optimisations.
        """
        start_time = time.time()
        if log:
            self.logger.add_log("Applying heuristic alterations to move probabilities...\n")
        if not move_dic:
            if log:
                self.logger.add_log("No moves in dictionary to alter. Returning empty.\n")
            return {}

        # --- Configuration & Setup ---
        game_phase = phase_of_game(board)
        mood = self.mood_manager.get_mood()
        side = board.turn
        
        # Get previous board states if available
        if prev_board is None:
            prev_board = self.state_manager.get_prev_board()
        if prev_prev_board is None:
            prev_prev_board = self.state_manager.get_prev_prev_board()
        
        # Calculate king danger levels once
        self_king_danger_lvl = king_danger(board, side, game_phase)
        opp_king_danger_lvl = king_danger(board, not side, game_phase)
        if log:
            self.logger.add_log(f"King Danger Levels: Self={self_king_danger_lvl}, Opp={opp_king_danger_lvl}\n")
        
        # Calculate threshold once
        lower_threshold_prob = sum(move_dic.values()) / len(move_dic) if move_dic else 0
        
        # Precompute scaling factors (avoid dictionary lookups in loops)
        sf_protect_king = 2.8
        sf_capture_en_pris = 1.5
        sf_break_pin = 3.0
        sf_capture = 1.5
        sf_capturable_move = 1.3
        sf_takeback = 2.5
        sf_passed_pawn_end = 3.0
        sf_weird_move = {"opening": 0.1, "midgame": 0.3, "endgame": 1.0}
        sf_check = {"confident": 2.3, "cocky": 3.3, "cautious": 2.1, "tilted": 3.3, "hurry": 3.0, "flagging": 2.9}.get(mood, 2.0)
        sf_new_threatened = {"confident": 3.5, "cocky": 2.7, "cautious": 3.9, "tilted": 2.1, "hurry": 2.4, "flagging": 3.2}.get(mood, 2.5)
        sf_exchange_material = {"confident": 3.2, "cocky": 2.0, "cautious": 2.2, "tilted": 1.5, "hurry": 3.8, "flagging": 0.8}.get(mood, 2.0)
        sf_exchange_king_danger = {"confident": 3.4, "cocky": 2.1, "cautious": 3.9, "tilted": 1.0, "hurry": 3.0, "flagging": 2.5}.get(mood, 2.0)
        sf_repeat_move = {"confident": 0.3, "cocky": 0.5, "cautious": 0.3, "tilted": 0.6, "hurry": 0.5, "flagging": 0.4}.get(mood, 0.4)
        
        # Adjust factors based on context (only once)
        if opp_king_danger_lvl > 500:
            sf_capture *= 1.5
        if opp_king_danger_lvl - self_king_danger_lvl > 500:
            sf_check *= 1.3
        
        # --- Apply Heuristics ---
        altered_dic = move_dic.copy()
        
        # Prepare move objects once
        move_objs = {uci: chess.Move.from_uci(uci) for uci in altered_dic.keys()}
        move_sans = {uci: board.san(move_obj) for uci, move_obj in move_objs.items()} if log else {}
        
        # Create logged categories if logging is enabled
        logged_categories = {}
        if log:
            logged_categories = {cat: [] for cat in [
                "strengthening", "weakening", "weird", "protect_king", "capture_en_pris",
                "break_pin", "capture", "capturable_move", "check", "takeback",
                "new_threatened", "good_exchange_mat", "bad_exchange_mat",
                "good_exchange_king", "bad_exchange_king", "passed_pawn", "repeat_move"
            ]}
        
        # --- Begin heuristics with timing ---
        
        # HEURISTIC 1: Threat/Protection Analysis
        h1_start = time.time()
        try:
            # Compute threatened boards once (outside the loop)
            piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
            curr_threatened_board_self = get_threatened_board(board, colour=side, piece_types=piece_types)
            self_curr_threatened_levels = sum(curr_threatened_board_self)
            opp_curr_threatened_levels = sum(get_threatened_board(board, colour=(not side), piece_types=piece_types))
            
            # Exaggerate if only one piece is threatened
            solo_factor = 2.0 if sum(np.array(curr_threatened_board_self) > 0.5) == 1 else 1.0
            
            for move_uci, move_obj in move_objs.items():
                # Create board copy just once per move
                dummy_board = board.copy()
                captured_piece_type = board.piece_type_at(move_obj.to_square)
                captured_value = PIECE_VALS.get(captured_piece_type, 0)
                
                dummy_board.push(move_obj)
                
                # Calculate new threat levels after the move
                new_threatened_board_self = get_threatened_board(dummy_board, colour=side, piece_types=piece_types)
                self_new_threatened_levels = sum(new_threatened_board_self)
                
                # Optimised opponent threat calculation
                moving_piece_threat_after = new_threatened_board_self[move_obj.to_square]
                if moving_piece_threat_after > 0.6:
                    # Only create the extra board if needed
                    temp_board_no_attacker = dummy_board.copy()
                    temp_board_no_attacker.remove_piece_at(move_obj.to_square)
                    opp_new_threatened_levels = sum(get_threatened_board(temp_board_no_attacker, colour=(not side), piece_types=piece_types))
                else:
                    opp_new_threatened_levels = sum(get_threatened_board(dummy_board, colour=(not side), piece_types=piece_types))
                
                # Calculate change in threat levels
                self_lvl_diff = self_new_threatened_levels - self_curr_threatened_levels
                opp_lvl_diff = opp_new_threatened_levels - opp_curr_threatened_levels
                
                # Combine changes
                total_lvl_diff = (opp_lvl_diff + captured_value) - (self_lvl_diff * 1.25 * solo_factor)
                
                factor = np.exp(total_lvl_diff / 2.0)
                
                # Boost probability floor for significantly improving moves
                if total_lvl_diff > 0.9 and altered_dic[move_uci] < lower_threshold_prob:
                    altered_dic[move_uci] = lower_threshold_prob
                
                altered_dic[move_uci] *= factor
                
                # Log categorization if needed
                if log:
                    move_san = move_sans[move_uci]
                    if total_lvl_diff > 0.9:
                        logged_categories["strengthening"].append(move_san)
                    elif total_lvl_diff < -0.9:
                        logged_categories["weakening"].append(move_san)
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during threat/protection analysis: {e}\n")
        
        h1_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 1 (Threat/Protection): {h1_end - h1_start:.4f} seconds\n")
        
        # HEURISTIC 1b: Weird Moves (separated for timing purposes)
        h1b_start = time.time()
        try:
            for move_uci in altered_dic.keys():
                # Only check for weird moves if not already categorized
                if log and move_sans[move_uci] not in logged_categories["strengthening"] and move_sans[move_uci] not in logged_categories["weakening"]:
                    if is_weird_move(board, game_phase, move_uci, self_king_danger_lvl):
                        weird_factor = sf_weird_move.get(game_phase, 0.5)
                        altered_dic[move_uci] *= weird_factor
                        logged_categories["weird"].append(move_sans[move_uci])
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during weird move analysis: {e}\n")
        
        h1b_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 1b (Weird Moves): {h1b_end - h1b_start:.4f} seconds\n")
        
        # HEURISTIC 2: Break Pins
        h2_start = time.time()
        try:
            for move_uci, move_obj in move_objs.items():
                to_square = move_obj.to_square
                pinned_attackers = is_attacked_by_pinned(board, to_square, not side)
                if pinned_attackers > 0:
                    if altered_dic[move_uci] < lower_threshold_prob:
                        altered_dic[move_uci] = lower_threshold_prob
                    altered_dic[move_uci] *= (sf_break_pin ** pinned_attackers)
                    if log:
                        logged_categories["break_pin"].append(move_sans[move_uci])
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during break pin analysis: {e}\n")
        
        h2_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 2 (Break Pins): {h2_end - h2_start:.4f} seconds\n")
        
        # HEURISTIC 3: Protect King
        h3_start = time.time()
        try:
            if self_king_danger_lvl > 250:  # Only process if king is in danger
                for move_uci, move_obj in move_objs.items():
                    dummy_board = board.copy()
                    dummy_board.push(move_obj)
                    
                    new_king_danger = king_danger(dummy_board, side, game_phase)
                    danger_reduction = self_king_danger_lvl / max(new_king_danger, 50)
                    if new_king_danger <= 0 or danger_reduction > 1.5:
                        if altered_dic[move_uci] < lower_threshold_prob:
                            altered_dic[move_uci] = lower_threshold_prob
                        king_factor = sf_protect_king * (danger_reduction**(1/4))
                        altered_dic[move_uci] *= king_factor
                        if log:
                            logged_categories["protect_king"].append(f"{move_sans[move_uci]} (Factor: {king_factor:.2f})")
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during king protection analysis: {e}\n")
        
        h3_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 3 (Protect King): {h3_end - h3_start:.4f} seconds\n")
        
        # HEURISTIC 4-5: Captures and Capture En Pris
        h4_start = time.time()
        try:
            for move_uci, move_obj in move_objs.items():
                is_capture = board.is_capture(move_obj)
                if is_capture:
                    captured_piece_type = board.piece_type_at(move_obj.to_square)
                    piece_value = PIECE_VALS.get(captured_piece_type, 1)
                    capture_factor = sf_capture * (piece_value**0.25)
                    altered_dic[move_uci] *= capture_factor
                    if log:
                        logged_categories["capture"].append(f"{move_sans[move_uci]} (Factor: {capture_factor:.2f})")
                    
                    # Capture En Pris (specific boost for capturing hanging pieces)
                    threatened_lvl_target = calculate_threatened_levels(move_obj.to_square, board)
                    dummy_board = board.copy()
                    dummy_board.push(move_obj)
                    
                    threat_level_attacker_after = calculate_threatened_levels(move_obj.to_square, dummy_board)
                    
                    if threatened_lvl_target > 0.6 and threat_level_attacker_after < piece_value + 0.1:
                        enpris_factor = sf_capture_en_pris * (threatened_lvl_target**0.25)
                        altered_dic[move_uci] *= enpris_factor
                        if log:
                            logged_categories["capture_en_pris"].append(f"{move_sans[move_uci]} (Factor: {enpris_factor:.2f})")
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during capture analysis: {e}\n")
        
        h4_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 4-5 (Captures): {h4_end - h4_start:.4f} seconds\n")
        
        # HEURISTIC 6: Move Capturable Piece
        h6_start = time.time()
        try:
            for move_uci, move_obj in move_objs.items():
                from_square = move_obj.from_square
                if is_capturable(board, from_square):
                    altered_dic[move_uci] *= sf_capturable_move
                    if log:
                        logged_categories["capturable_move"].append(move_sans[move_uci])
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during capturable piece analysis: {e}\n")
        
        h6_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 6 (Move Capturable): {h6_end - h6_start:.4f} seconds\n")
        
        # HEURISTIC 7: Checks
        h7_start = time.time()
        try:
            for move_uci, move_obj in move_objs.items():
                if board.gives_check(move_obj):
                    altered_dic[move_uci] *= sf_check
                    if log:
                        logged_categories["check"].append(f"{move_sans[move_uci]} (Factor: {sf_check:.2f})")
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during check analysis: {e}\n")
        
        h7_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 7 (Checks): {h7_end - h7_start:.4f} seconds\n")
        
        # HEURISTIC 8: Takebacks
        h8_start = time.time()
        try:
            if prev_board:  # Only process if we have a previous board
                res = patch_fens(prev_board.fen(), board.fen(), depth_lim=1)
                if res:
                    last_opp_move_uci = res[0][0]
                    for move_uci, move_obj in move_objs.items():
                        if is_takeback(prev_board, last_opp_move_uci, move_uci):
                            altered_dic[move_uci] *= sf_takeback
                            if log:
                                logged_categories["takeback"].append(move_sans[move_uci])
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during takeback analysis: {e}\n")
        
        h8_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 8 (Takebacks): {h8_end - h8_start:.4f} seconds\n")
        
        # HEURISTIC 9: Respond to New Threat
        h9_start = time.time()
        try:
            if prev_board:  # Only process if we have a previous board
                for move_uci, move_obj in move_objs.items():
                    moving_piece_square = move_obj.from_square
                    newly_attacked_value = is_newly_attacked(prev_board, board, moving_piece_square)
                    if newly_attacked_value > 0.6:
                        dummy_board = board.copy()
                        dummy_board.push(move_obj)
                        
                        threat_after_move = calculate_threatened_levels(move_obj.to_square, dummy_board)
                        if threat_after_move < newly_attacked_value - 0.1:
                            threat_factor = sf_new_threatened * (1 + newly_attacked_value)**0.2
                            altered_dic[move_uci] *= threat_factor
                            if log:
                                logged_categories["new_threatened"].append(f"{move_sans[move_uci]} (Factor: {threat_factor:.2f})")
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during new threat analysis: {e}\n")
        
        h9_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 9 (New Threats): {h9_end - h9_start:.4f} seconds\n")
        
        # Precompute material balance once (outside the loop)
        h10_start = time.time()
        mat_dic = {chess.PAWN: 1, chess.KNIGHT: 3.1, chess.BISHOP: 3.5, chess.ROOK: 5.5, chess.QUEEN: 9.9, chess.KING: 0}
        own_mat = sum(len(board.pieces(pt, side)) * mat_dic[pt] for pt in mat_dic)
        opp_mat = sum(len(board.pieces(pt, not side)) * mat_dic[pt] for pt in mat_dic)
        material_diff = own_mat - opp_mat
        
        # HEURISTIC 10-11: Exchanges (Material and King Danger)
        try:
            for move_uci, move_obj in move_objs.items():
                is_exchange = is_offer_exchange(board, move_uci)
                if is_exchange:
                    # Material-based exchanges
                    if material_diff > 2.9:  # Ahead -> Encourage trades
                        altered_dic[move_uci] *= sf_exchange_material
                        if log:
                            logged_categories["good_exchange_mat"].append(f"{move_sans[move_uci]} (Factor: {sf_exchange_material:.2f})")
                    elif material_diff < -2.9:  # Behind -> Discourage trades
                        altered_dic[move_uci] /= sf_exchange_material
                        if log:
                            logged_categories["bad_exchange_mat"].append(f"{move_sans[move_uci]} (Factor: {1/sf_exchange_material:.2f})")
                    
                    # King safety-based exchanges
                    king_danger_diff = self_king_danger_lvl - opp_king_danger_lvl
                    if king_danger_diff >= 400 and self_king_danger_lvl > 500:  # Our king much less safe -> Encourage trades
                        altered_dic[move_uci] *= sf_exchange_king_danger
                        if log:
                            logged_categories["good_exchange_king"].append(f"{move_sans[move_uci]} (Factor: {sf_exchange_king_danger:.2f})")
                    elif king_danger_diff <= -400 and opp_king_danger_lvl > 500:  # Opponent king much less safe -> Discourage trades
                        altered_dic[move_uci] /= sf_exchange_king_danger
                        if log:
                            logged_categories["bad_exchange_king"].append(f"{move_sans[move_uci]} (Factor: {1/sf_exchange_king_danger:.2f})")
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during exchanges analysis: {e}\n")
        
        h10_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 10-11 (Exchanges): {h10_end - h10_start:.4f} seconds\n")
        
        # HEURISTIC 12: Push Passed Pawn
        h12_start = time.time()
        try:
            if game_phase == "endgame":  # Only relevant in endgames
                for move_uci, move_obj in move_objs.items():
                    move_piece_type = board.piece_type_at(move_obj.from_square)
                    if move_piece_type == chess.PAWN:
                        passed_result = is_open_file(board, chess.square_file(move_obj.from_square))
                        if (board.turn == chess.WHITE and passed_result == -2) or (board.turn == chess.BLACK and passed_result == 2):
                            altered_dic[move_uci] *= sf_passed_pawn_end
                            if log:
                                logged_categories["passed_pawn"].append(move_sans[move_uci])
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during passed pawn analysis: {e}\n")
        
        h12_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 12 (Passed Pawns): {h12_end - h12_start:.4f} seconds\n")
        
        # HEURISTIC 13: Repeat Moves
        h13_start = time.time()
        try:
            if prev_board and prev_prev_board:  # Only process if we have two previous boards
                res = patch_fens(prev_prev_board.fen(), prev_board.fen(), depth_lim=1)
                if res:
                    last_own_move_uci = res[0][0]
                    last_own_move_obj = chess.Move.from_uci(last_own_move_uci)
                    
                    # Pre-calculate reachable squares for efficiency
                    squares_reachable_before = {
                        m.to_square for m in prev_prev_board.legal_moves
                        if m.from_square == last_own_move_obj.from_square
                    } | {last_own_move_obj.from_square}
                    
                    for move_uci, move_obj in move_objs.items():
                        # Check if the current move starts where the last one ended
                        if move_obj.from_square == last_own_move_obj.to_square:
                            # Check if the current move's destination was reachable from the piece's original square
                            if move_obj.to_square in squares_reachable_before:
                                altered_dic[move_uci] *= sf_repeat_move
                                if log:
                                    logged_categories["repeat_move"].append(f"{move_sans[move_uci]} (Factor: {sf_repeat_move:.2f})")
        except Exception as e:
            if log:
                self.logger.add_log(f"ERROR during repeat move analysis: {e}\n")
        
        h13_end = time.time()
        # if log:
        #     self.logger.add_log(f"HEURISTIC 13 (Repeat Moves): {h13_end - h13_start:.4f} seconds\n")
        
        # --- Final Normalization & Logging ---
        final_start = time.time()
        
        # Log the effects of heuristics
        if log:
            for category, moves in logged_categories.items():
                if moves:
                    self.logger.add_log(f"Moves affected by '{category}': {', '.join(moves)}\n")
        
        # Normalize final probabilities
        total = sum(altered_dic.values())
        eps = 1e-10
        if total <= eps:
            if log:
                self.logger.add_log("WARNING: Sum of altered probabilities is near zero. Returning uniform distribution.\n")
            num_moves = len(altered_dic)
            final_dic = {k: 1.0/num_moves for k in altered_dic} if num_moves > 0 else {}
        else:
            final_dic = {k: v / total for k, v in altered_dic.items()}
        
        # Sort by probability descending
        sorted_final_dic = dict(sorted(final_dic.items(), key=lambda item: item[1], reverse=True))
        
        final_end = time.time()
        if log:
            self.logger.add_log(f"Final normalization and sorting: {final_end - final_start:.4f} seconds\n")
        
        # Log final altered probabilities
        end_time = time.time()
        if log:
            try:
                log_final_dic_san = {board.san(move_objs[k]): round(v, 5) for k, v in sorted_final_dic.items()}
            except Exception as e:
                self.logger.add_log(f"WARNING: Error generating SAN for logging final probabilities: {e}\n")
                log_final_dic_san = {k: round(v, 5) for k, v in sorted_final_dic.items()}
            
            self.logger.add_log(f"Final altered move probabilities: {log_final_dic_san}\n")
            self.logger.add_log(f"Move probability alterations finished in {end_time - start_time:.4f} seconds\n")
        
        return sorted_final_dic