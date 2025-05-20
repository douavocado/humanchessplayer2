import chess
import numpy as np
import torch
import torch.nn as nn

from common.board_information import (
    phase_of_game, PIECE_VALS, king_danger, get_threatened_board, is_capturing_move, is_capturable,
    is_attacked_by_pinned, is_check_move, is_takeback, is_newly_attacked, is_offer_exchange,
    is_open_file, calculate_threatened_levels, is_weird_move
)
from common.utils import patch_fens


class AlterMoveProbNN(nn.Module):
    """
    A PyTorch module that replicates the alter_move_probabilities function with trainable parameters.
    """
    def __init__(self):
        super(AlterMoveProbNN, self).__init__()
        
        # Create trainable parameters for all factors used in alter_move_probabilities
        # Game phase dependent parameters, actually log-scale to ensure they are positive
        self.weird_move_sd_opening = nn.Parameter(torch.tensor(0.0, dtype=torch.float)) # actual sd is exp(self.weird_move_sd_opening)
        self.weird_move_sd_midgame = nn.Parameter(torch.tensor(0.0, dtype=torch.float)) # actual sd is exp(self.weird_move_sd_midgame)
        self.weird_move_sd_endgame = nn.Parameter(torch.tensor(0.0, dtype=torch.float)) # actual sd is exp(self.weird_move_sd_endgame)
        
        # Other scaling factors
        self.protect_king_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.capture_en_pris_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.break_pin_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.capture_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.capture_sf_king_danger = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.capturable_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.solo_factor_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.threatened_lvl_diff_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        # Simplified single-value parameters that were dictionaries based on mood
        self.check_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.takeback_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.new_threatened_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.exchange_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.exchange_k_danger_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.passed_pawn_end_sf = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.repeat_sf = nn.Parameter(torch.tensor(0.0, dtype=torch.float)) # actual repeat_sf is exp(self.repeat_sf)

        # threshold probability for any "interesting" move that moves defaults to if less than this
        self.interesting_move_threshold = nn.Parameter(torch.tensor(0.0, dtype=torch.float)) # actual threshold is actually exp(self.interesting_move_threshold)/len(move_dic)
        self.relu = nn.ReLU()
        self.params_dict = {}
        
    def forward(self, move_dic, board, prev_board=None, prev_prev_board=None):
        """
        Forward pass that replicates alter_move_probabilities but uses trainable parameters.
        
        Args:
            move_dic: Dictionary with move UCI as key and their unaltered probabilities
            board: Current chess board
            prev_board: Previous chess board (optional)
            prev_prev_board: Board from two moves ago (optional)
            
        Returns:
            altered_move_dic: Dictionary with altered move probabilities
            log: String log of changes made (for debugging)
        """
        log = ""
        game_phase = phase_of_game(board)
        self_king_danger_lvl = king_danger(board, board.turn, game_phase)
        opp_king_danger_lvl = king_danger(board, not board.turn, game_phase)
        
        # Get the appropriate weird_move_sd based on game phase
        weird_move_sd_dic = {
            "opening": torch.exp(self.weird_move_sd_opening),
            "midgame": torch.exp(self.weird_move_sd_midgame),
            "endgame": torch.exp(self.weird_move_sd_endgame),
        }

        # Setting true threshold based on number of possible moves from move_dic
        threshold = torch.exp(self.interesting_move_threshold) / len(move_dic)
        
        # Create a copy of the move dictionary to modify
        # This needs to maintain gradient information and avoid numerical instability
        altered_move_dic = {}
        for k, v in move_dic.items():
            if v <= 0:
                # Avoid zero or negative probabilities
                v = 1e-8
            altered_move_dic[k] = torch.tensor(v, dtype=torch.float, requires_grad=True)
        
        # Moves which protect/block/make our pieces less en pris and opponent pieces more en pris
        strenghening_moves = []
        weakening_moves = []     
        weird_moves = []
        
        # For time computation sake, we ignore the threatened levels of pawns
        curr_threatened_board = get_threatened_board(board, colour=board.turn, piece_types=[1,2,3,4,5])
        self_curr_threatened_levels = sum(curr_threatened_board)
        
        opp_curr_threatened_levels = sum(get_threatened_board(board, colour=(not board.turn), piece_types=[1,2,3,4,5]))
        
        for move_uci in move_dic.keys():
            move_obj = chess.Move.from_uci(move_uci)
            dummy_board = board.copy()
            dummy_board.push(move_obj)
            new_threatened_board = get_threatened_board(dummy_board, colour=board.turn, piece_types=[1,2,3,4,5])
            self_new_threatened_levels = sum(new_threatened_board)
            
            # If new move makes our piece en_pris, then calculate enemy threatened levels
            # as if there was no such piece
            piece_type = board.piece_type_at(move_obj.to_square)
            if piece_type is not None:
                to_value = PIECE_VALS[piece_type]
            else:
                to_value = 0
                
            if new_threatened_board[move_obj.to_square] - to_value > 0.6:
                dummy_board.remove_piece_at(move_obj.to_square)
            
            opp_new_threatened_levels = sum(get_threatened_board(dummy_board, colour=(not board.turn), piece_types=[1,2,3,4,5]))
            self_lvl_diff = self_new_threatened_levels - self_curr_threatened_levels
            
            opp_lvl_diff = opp_new_threatened_levels - opp_curr_threatened_levels
            
            # Psychologically, protecting pieces is more favorable than attacking pieces
            # If only one of our pieces is threatened, exaggerate the levels
            if sum(np.array(curr_threatened_board) > 0.5) == 1:
                lvl_diff = opp_lvl_diff - self_lvl_diff * 1.25 * self.solo_factor_sf
            else:
                lvl_diff = torch.tensor(opp_lvl_diff - self_lvl_diff * 1.25, dtype=torch.float)
            
            if piece_type is not None:  # if we captured something
                lvl_diff += to_value
            
            # Convert to tensor to ensure gradient flow
            factor = torch.exp(lvl_diff * self.threatened_lvl_diff_sf / 2)     
            
            # before we apply factor, if factor is greater than 1 (so our move is good), we need to make sure it is above the threshold
            if factor > 1:
                altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold

            # Use multiplication inplace to preserve gradients
            altered_move_dic[move_uci] = altered_move_dic[move_uci] * factor
            
            if lvl_diff > 0.9:
                strenghening_moves.append(board.san(chess.Move.from_uci(move_uci)))
            elif lvl_diff < -0.9:
                weakening_moves.append(board.san(chess.Move.from_uci(move_uci)))
            else:
                # If move doesn't seem to do anything that threatens or protects pieces, check
                # whether it is a "weird" move
                if is_weird_move(board, game_phase, move_uci, self_king_danger_lvl):
                    # Then incur penalty
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * weird_move_sd_dic[game_phase]
                    weird_moves.append(board.san(chess.Move.from_uci(move_uci)))

        log += f"Found moves that are weakening and make our pieces more enpris/opp pieces less enpris: {weakening_moves} \n"
        log += f"Found moves that protect our pieces more or apply more pressure to opponent: {strenghening_moves} \n"
        log += f"Found weird moves: {weird_moves} \n"
        
        # Squares that pinned pieces attack that break the pin are more desirable to move to
        adv_pinned_moves = []
        for move_uci in move_dic.keys():
            to_square = chess.Move.from_uci(move_uci).to_square
            no_pinned_atks = is_attacked_by_pinned(board, to_square, not board.turn)
            if no_pinned_atks > 0:
                # then this is interesting move, make it above threshold
                altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                altered_move_dic[move_uci] = altered_move_dic[move_uci] * (self.break_pin_sf ** no_pinned_atks)
                adv_pinned_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found moves that take advantage of pinned pieces: {adv_pinned_moves} \n"
        
        # If king danger high, then moves that defend our king are more attractive
        before_king_danger = king_danger(board, board.turn, game_phase)
        # If king is not in danger, pass
        if before_king_danger < 250:
            log += f"King danger {before_king_danger} not high to consider protecting king moves. Skipping... \n"
        else:
            protect_king_moves = []            
            for move_uci in move_dic.keys():
                dummy_board = board.copy()
                dummy_board.push_uci(move_uci)
                new_king_danger = king_danger(dummy_board, board.turn, game_phase)
                if new_king_danger <= 0:
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.protect_king_sf * (before_king_danger / 50) ** (1 / 4)
                    protect_king_moves.append(board.san(chess.Move.from_uci(move_uci)))
                elif before_king_danger / new_king_danger > 1.5:
                    denom = max(new_king_danger, 50)
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.protect_king_sf * (before_king_danger / denom) ** (1 / 4)
                    protect_king_moves.append(board.san(chess.Move.from_uci(move_uci)))
            log += f"Found moves that protect our vulnerable king: {protect_king_moves} \n"
            
        # Capturing moves are more appealing
        capturing_moves = []
        for move_uci in move_dic.keys():
            if is_capturing_move(board, move_uci):
                # It is a capturing move
                piece_value = PIECE_VALS[board.piece_type_at(chess.Move.from_uci(move_uci).to_square)]
                
                # Use different capture factor based on opponent king danger
                capture_factor = self.capture_sf_king_danger if opp_king_danger_lvl > 500 else self.capture_sf
                # then this is interesting move, make it above threshold
                altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold

                altered_move_dic[move_uci] = altered_move_dic[move_uci] * capture_factor * (piece_value ** 0.25)
                capturing_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found capturing moves from position: {capturing_moves} \n"
        
        # Capturing enpris pieces are more appealing
        capturing_enpris_moves = []
        for move_uci in move_dic.keys():
            if is_capturing_move(board, move_uci):
                # It is a capturing move
                move_obj = chess.Move.from_uci(move_uci)
                threatened_lvls = calculate_threatened_levels(move_obj.to_square, board)
                # Not only is the captured piece enpris, but we are capturing it with the correct piece
                piece_type = board.piece_type_at(move_obj.to_square)
                to_value = PIECE_VALS[piece_type]
                dummy_board = board.copy()
                dummy_board.push(move_obj)
                if threatened_lvls > 0.6 and calculate_threatened_levels(move_obj.to_square, dummy_board) - to_value < 0:  # captured piece is enpris
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.capture_en_pris_sf * (threatened_lvls ** 0.25)
                    capturing_enpris_moves.append(board.san(move_obj))
        log += f"Found capturing enpris moves from position: {capturing_enpris_moves} \n"
        
        # Capturable pieces are more appealling to move
        capturable_moves = []
        for move_uci in move_dic.keys():
            from_square = chess.Move.from_uci(move_uci).from_square
            if is_capturable(board, from_square):
                # It is a capturable piece
                # then this is interesting move, make it above threshold
                altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.capturable_sf
                capturable_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found moves that move capturable pieces: {capturable_moves} \n"
        
        # Checks are more appealing
        checking_moves = []
        for move_uci in move_dic.keys():
            if is_check_move(board, move_uci):
                # then this is interesting move, make it above threshold
                altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.check_sf
                checking_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found checking moves: {checking_moves} \n"
        
        # Takebacks are more appealing
        # We may only calculate this criterion if we have information of previous move
        if prev_board is not None:
            takeback_moves = []
            res = patch_fens(prev_board.fen(), board.fen(), depth_lim=1)
            last_move_uci = res[0][0]
            for move_uci in move_dic.keys():
                if is_takeback(prev_board, last_move_uci, move_uci):
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.takeback_sf
                    takeback_moves.append(board.san(chess.Move.from_uci(move_uci)))
            log += f"Found takeback moves: {takeback_moves} \n"
        
        # Newly threatened en-pris pieces are more appealing to move
        # We may only calculate this criterion if we have information of previous positions
        if prev_board is not None:            
            new_threatened_moves = []
            # First get all the squares our own pieces and work out whether they are
            # newly threatened or not
            from_squares = [sq for piece_type in range(1, 6) for sq in board.pieces(piece_type, board.turn)]
            from_sq_dic = {from_sq: is_newly_attacked(prev_board, board, from_sq) for from_sq in from_squares}
            newly_attacked_squares = [sq for sq in from_sq_dic if from_sq_dic[sq] > 0.6]
            for move_uci in move_dic.keys():
                dummy_board = board.copy()
                dummy_board.push_uci(move_uci)
                for square in newly_attacked_squares:
                    threatened_levels = calculate_threatened_levels(square, dummy_board)
                    difference_in_threatened = from_sq_dic[square] - threatened_levels
                    if difference_in_threatened > 0.6:
                        # then this is interesting move, make it above threshold
                        altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                        altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.new_threatened_sf * (1 + difference_in_threatened) ** 0.2
                        new_threatened_moves.append(board.san(chess.Move.from_uci(move_uci)))
                        break
            log += f"Found moves that respond to newly threatened pieces: {new_threatened_moves} \n"
        
        # Offering exchanges/exchanging when material up appealing
        # likewise offering exchanges when material down unappealing
        mat_dic = {1:1, 2:3.1, 3:3.5, 4:5.5, 5:9.9, 6:3}
        own_mat = sum([len(board.pieces(x, board.turn))*mat_dic[x] for x in range(1,6)])
        opp_mat = sum([len(board.pieces(x, not board.turn))*mat_dic[x] for x in range(1,6)])
        material_score = own_mat - opp_mat

        good_exchanges = []
        bad_exchanges = []
        good_king_exchanges = []
        bad_king_exchanges = []
        
        for move_uci in move_dic.keys():
            if is_offer_exchange(board, move_uci):
                if material_score > 2.9:  # If we are ahead by at least a pawn and a half, exchanges are good
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.exchange_sf
                    good_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
                elif material_score < -2.9:  # If we are behind by at least a pawn and a half, no exchanges                    
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * (1.0 / self.exchange_sf)
                    bad_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
                    
                # If King is in danger, then exchanges are good
                if self_king_danger_lvl > 500 and self_king_danger_lvl > opp_king_danger_lvl + 400:
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.exchange_k_danger_sf
                    good_king_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
                elif opp_king_danger_lvl > 500 and opp_king_danger_lvl > self_king_danger_lvl + 400:
                    # Keep pieces on board
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * (1.0 / self.exchange_k_danger_sf)
                    bad_king_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found moves that encourage exchanges as we are material up: {good_exchanges} \n"
        log += f"Found moves that trade when we are material down: {bad_exchanges} \n"
        log += f"Found moves that encourage exchanges when our king is in danger: {good_king_exchanges} \n"
        log += f"Found moves that trade when enemy king is in danger: {bad_king_exchanges} \n"

        # In endgame passed pawns need to be pushed
        if game_phase == "endgame":
            passed_pawn_moves = []
            for move_uci in move_dic.keys():
                move_obj = chess.Move.from_uci(move_uci)
                from_square = move_obj.from_square
                # check if the piece is a pawn, if so check if it is passed
                if board.piece_type_at(from_square) != 1:  # 1 = pawn
                    continue
                # Check if the pawn is passed, by checking if there are enemy pawns in any of the three files (including its own)
                passed = True
                pawn_file = chess.square_file(from_square)
                pawn_rank = chess.square_rank(from_square)
                for file in range(max(0, pawn_file - 1), min(7, pawn_file + 1) + 1):
                    for rank in range(pawn_rank, 8 if board.turn else -1, 1 if board.turn else -1):
                        if rank < 0 or rank > 7:
                            continue
                        if file == pawn_file and rank == pawn_rank:
                            continue
                        square = chess.square(file, rank)
                        opp_piece_at_sq = board.piece_at(square)
                        if opp_piece_at_sq is not None and opp_piece_at_sq.piece_type == 1 and opp_piece_at_sq.color != board.turn:
                            passed = False
                            break
                    if not passed:
                        break
                if passed:
                    # Adjust how far we are from promoting
                    to_promote = 7 - pawn_rank if board.turn else pawn_rank
                    scaling = ((8 - to_promote) ** 1.5) / (8 ** 1.5)
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = self.relu(altered_move_dic[move_uci] - threshold) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * self.passed_pawn_end_sf * scaling
                    passed_pawn_moves.append(board.san(chess.Move.from_uci(move_uci)))
            log += f"Found moves that move passed pawns forward: {passed_pawn_moves} \n"
        
        # Repeat moves are undesirable
        # These moves are defined to be moves which repeatedly move a piece, and goto a square 
        # which we could have gone to with the piece last move (including moving piece back and forth)
        if prev_board is not None and prev_prev_board is not None:
            repeat_moves = []
            res = patch_fens(prev_prev_board.fen(), prev_board.fen(), depth_lim=1)
            last_own_move_uci = res[0][0]
            last_own_move_obj = chess.Move.from_uci(last_own_move_uci)
            previous_reach = set([move.to_square for move in prev_prev_board.legal_moves if move.from_square == last_own_move_obj.from_square] + [last_own_move_obj.from_square])
            for move_uci in move_dic.keys():
                move_obj = chess.Move.from_uci(move_uci)
                if move_obj.from_square == last_own_move_obj.to_square: # if we are moving the same piece as before
                    to_square = move_obj.to_square
                    if to_square in previous_reach: # then we have found repeating move
                        altered_move_dic[move_uci] *= torch.exp(self.repeat_sf)
                        repeat_moves.append(board.san(move_obj))
            
            log += "Found moves that are repetitive, and waste time: {} \n".format(repeat_moves)
        
        # Now, normalize probabilities
        total_prob = sum(altered_move_dic.values())
        
        # Add stability check for total_prob
        if not torch.is_tensor(total_prob) or total_prob.item() <= 0:
            # If total_prob is invalid, reset move_dic to original with small random noise
            altered_move_dic = {k: torch.tensor(v + 1e-8, dtype=torch.float, requires_grad=True) 
                               for k, v in move_dic.items()}
            total_prob = sum(altered_move_dic.values())
        
        # Add a small epsilon to prevent division by zero
        epsilon = 1e-8
        normalized_total = total_prob + epsilon
        
        for move_uci in altered_move_dic:
            # Make sure we don't have extremely small values that could cause numerical issues
            prob_value = altered_move_dic[move_uci]
            if prob_value < epsilon:
                prob_value = torch.tensor(epsilon, dtype=torch.float, requires_grad=True)
            
            # Normalize the probability
            altered_move_dic[move_uci] = prob_value / normalized_total
        log_dic = {board.san(chess.Move.from_uci(k)) : round(v.item(), 3) for k,v in altered_move_dic.items()}
        log += f"Move_dic after alteration: {log_dic} \n"
        
        return altered_move_dic, log

    def get_parameters_dict(self):
        """
        Returns a dictionary of all trainable parameters.
        """
        return {
            "weird_move_sd_opening": float(self.weird_move_sd_opening),
            "weird_move_sd_midgame": float(self.weird_move_sd_midgame),
            "weird_move_sd_endgame": float(self.weird_move_sd_endgame),
            "protect_king_sf": float(self.protect_king_sf),
            "capture_en_pris_sf": float(self.capture_en_pris_sf),
            "break_pin_sf": float(self.break_pin_sf),
            "capture_sf": float(self.capture_sf),
            "capture_sf_king_danger": float(self.capture_sf_king_danger),
            "capturable_sf": float(self.capturable_sf),
            "check_sf": float(self.check_sf),
            "takeback_sf": float(self.takeback_sf),
            "new_threatened_sf": float(self.new_threatened_sf),
            "exchange_sf": float(self.exchange_sf),
            "exchange_k_danger_sf": float(self.exchange_k_danger_sf),
            "passed_pawn_end_sf": float(self.passed_pawn_end_sf),
            "repeat_sf": float(self.repeat_sf),
        }

    def load_params_dict(self, params_dict=None):
        """
        Loads a dictionary of parameters into the model.
        """
        if params_dict is None:
            params_dict = self.get_parameters_dict()
        self.params_dict.update(params_dict)

    def forward_numpy(self, move_dic, board, prev_board=None, prev_prev_board=None):
        """
        Numpy-based forward pass that replicates alter_move_probabilities without torch tensors.
        
        Args:
            move_dic: Dictionary with move UCI as key and their unaltered probabilities
            board: Current chess board
            params_dict: Dictionary containing parameter values loaded from the model
            prev_board: Previous chess board (optional)
            prev_prev_board: Board from two moves ago (optional)
            
        Returns:
            altered_move_dic: Dictionary with altered move probabilities
            log: String log of changes made (for debugging)
        """
        log = ""
        game_phase = phase_of_game(board)
        self_king_danger_lvl = king_danger(board, board.turn, game_phase)
        opp_king_danger_lvl = king_danger(board, not board.turn, game_phase)
        
        # Get the appropriate weird_move_sd based on game phase
        weird_move_sd_dic = {
            "opening": np.exp(self.params_dict.get("weird_move_sd_opening", 0.0)),
            "midgame": np.exp(self.params_dict.get("weird_move_sd_midgame", 0.0)),
            "endgame": np.exp(self.params_dict.get("weird_move_sd_endgame", 0.0)),
        }

        # Setting true threshold based on number of possible moves from move_dic
        threshold = np.exp(self.params_dict.get("interesting_move_threshold", 0.0)) / len(move_dic)
        
        # Create a copy of the move dictionary to modify
        altered_move_dic = {k: float(v) if v > 0 else 1e-8 for k, v in move_dic.items()}
        
        # Moves which protect/block/make our pieces less en pris and opponent pieces more en pris
        strenghening_moves = []
        weakening_moves = []     
        weird_moves = []
        
        # For time computation sake, we ignore the threatened levels of pawns
        curr_threatened_board = get_threatened_board(board, colour=board.turn, piece_types=[1,2,3,4,5])
        self_curr_threatened_levels = sum(curr_threatened_board)
        
        opp_curr_threatened_levels = sum(get_threatened_board(board, colour=(not board.turn), piece_types=[1,2,3,4,5]))
        
        for move_uci in move_dic.keys():
            move_obj = chess.Move.from_uci(move_uci)
            dummy_board = board.copy()
            dummy_board.push(move_obj)
            new_threatened_board = get_threatened_board(dummy_board, colour=board.turn, piece_types=[1,2,3,4,5])
            self_new_threatened_levels = sum(new_threatened_board)
            
            # If new move makes our piece en_pris, then calculate enemy threatened levels
            # as if there was no such piece
            piece_type = board.piece_type_at(move_obj.to_square)
            if piece_type is not None:
                to_value = PIECE_VALS[piece_type]
            else:
                to_value = 0
                
            if new_threatened_board[move_obj.to_square] - to_value > 0.6:
                dummy_board.remove_piece_at(move_obj.to_square)
            
            opp_new_threatened_levels = sum(get_threatened_board(dummy_board, colour=(not board.turn), piece_types=[1,2,3,4,5]))
            self_lvl_diff = self_new_threatened_levels - self_curr_threatened_levels
            
            opp_lvl_diff = opp_new_threatened_levels - opp_curr_threatened_levels
            
            # Psychologically, protecting pieces is more favorable than attacking pieces
            # If only one of our pieces is threatened, exaggerate the levels
            solo_factor_sf = self.params_dict.get("solo_factor_sf", 1.0)
            threatened_lvl_diff_sf = self.params_dict.get("threatened_lvl_diff_sf", 1.0)
            
            if sum(np.array(curr_threatened_board) > 0.5) == 1:
                lvl_diff = opp_lvl_diff - self_lvl_diff * 1.25 * solo_factor_sf
            else:
                lvl_diff = opp_lvl_diff - self_lvl_diff * 1.25
            
            if piece_type is not None:  # if we captured something
                lvl_diff += to_value
            
            # Calculate factor
            factor = np.exp(lvl_diff * threatened_lvl_diff_sf / 2)     
            
            # before we apply factor, if factor is greater than 1 (so our move is good), we need to make sure it is above the threshold
            if factor > 1:
                altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold

            # Apply factor
            altered_move_dic[move_uci] = altered_move_dic[move_uci] * factor
            
            if lvl_diff > 0.9:
                strenghening_moves.append(board.san(chess.Move.from_uci(move_uci)))
            elif lvl_diff < -0.9:
                weakening_moves.append(board.san(chess.Move.from_uci(move_uci)))
            else:
                # If move doesn't seem to do anything that threatens or protects pieces, check
                # whether it is a "weird" move
                if is_weird_move(board, game_phase, move_uci, self_king_danger_lvl):
                    # Then incur penalty
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * weird_move_sd_dic[game_phase]
                    weird_moves.append(board.san(chess.Move.from_uci(move_uci)))

        log += f"Found moves that are weakening and make our pieces more enpris/opp pieces less enpris: {weakening_moves} \n"
        log += f"Found moves that protect our pieces more or apply more pressure to opponent: {strenghening_moves} \n"
        log += f"Found weird moves: {weird_moves} \n"
        
        # Squares that pinned pieces attack that break the pin are more desirable to move to
        break_pin_sf = self.params_dict.get("break_pin_sf", 1.0)
        adv_pinned_moves = []
        for move_uci in move_dic.keys():
            to_square = chess.Move.from_uci(move_uci).to_square
            no_pinned_atks = is_attacked_by_pinned(board, to_square, not board.turn)
            if no_pinned_atks > 0:
                # then this is interesting move, make it above threshold
                altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                altered_move_dic[move_uci] = altered_move_dic[move_uci] * (break_pin_sf ** no_pinned_atks)
                adv_pinned_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found moves that take advantage of pinned pieces: {adv_pinned_moves} \n"
        
        # If king danger high, then moves that defend our king are more attractive
        before_king_danger = king_danger(board, board.turn, game_phase)
        protect_king_sf = self.params_dict.get("protect_king_sf", 1.0)
        
        # If king is not in danger, pass
        if before_king_danger < 250:
            log += f"King danger {before_king_danger} not high to consider protecting king moves. Skipping... \n"
        else:
            protect_king_moves = []            
            for move_uci in move_dic.keys():
                dummy_board = board.copy()
                dummy_board.push_uci(move_uci)
                new_king_danger = king_danger(dummy_board, board.turn, game_phase)
                if new_king_danger <= 0:
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * protect_king_sf * (before_king_danger / 50) ** (1 / 4)
                    protect_king_moves.append(board.san(chess.Move.from_uci(move_uci)))
                elif before_king_danger / new_king_danger > 1.5:
                    denom = max(new_king_danger, 50)
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * protect_king_sf * (before_king_danger / denom) ** (1 / 4)
                    protect_king_moves.append(board.san(chess.Move.from_uci(move_uci)))
            log += f"Found moves that protect our vulnerable king: {protect_king_moves} \n"
            
        # Capturing moves are more appealing
        capture_sf = self.params_dict.get("capture_sf", 1.0)
        capture_sf_king_danger = self.params_dict.get("capture_sf_king_danger", 1.0)
        
        capturing_moves = []
        for move_uci in move_dic.keys():
            if is_capturing_move(board, move_uci):
                # It is a capturing move
                piece_value = PIECE_VALS[board.piece_type_at(chess.Move.from_uci(move_uci).to_square)]
                
                # Use different capture factor based on opponent king danger
                capture_factor = capture_sf_king_danger if opp_king_danger_lvl > 500 else capture_sf
                # then this is interesting move, make it above threshold
                altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold

                altered_move_dic[move_uci] = altered_move_dic[move_uci] * capture_factor * (piece_value ** 0.25)
                capturing_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found capturing moves from position: {capturing_moves} \n"
        
        # Capturing enpris pieces are more appealing
        capture_en_pris_sf = self.params_dict.get("capture_en_pris_sf", 1.0)
        capturing_enpris_moves = []
        for move_uci in move_dic.keys():
            if is_capturing_move(board, move_uci):
                # It is a capturing move
                move_obj = chess.Move.from_uci(move_uci)
                threatened_lvls = calculate_threatened_levels(move_obj.to_square, board)
                # Not only is the captured piece enpris, but we are capturing it with the correct piece
                piece_type = board.piece_type_at(move_obj.to_square)
                to_value = PIECE_VALS[piece_type]
                dummy_board = board.copy()
                dummy_board.push(move_obj)
                if threatened_lvls > 0.6 and calculate_threatened_levels(move_obj.to_square, dummy_board) - to_value < 0:  # captured piece is enpris
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * capture_en_pris_sf * (threatened_lvls ** 0.25)
                    capturing_enpris_moves.append(board.san(move_obj))
        log += f"Found capturing enpris moves from position: {capturing_enpris_moves} \n"
        
        # Capturable pieces are more appealling to move
        capturable_sf = self.params_dict.get("capturable_sf", 1.0)
        capturable_moves = []
        for move_uci in move_dic.keys():
            from_square = chess.Move.from_uci(move_uci).from_square
            if is_capturable(board, from_square):
                # It is a capturable piece
                # then this is interesting move, make it above threshold
                altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                altered_move_dic[move_uci] = altered_move_dic[move_uci] * capturable_sf
                capturable_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found moves that move capturable pieces: {capturable_moves} \n"
        
        # Checks are more appealing
        check_sf = self.params_dict.get("check_sf", 1.0)
        checking_moves = []
        for move_uci in move_dic.keys():
            if is_check_move(board, move_uci):
                # then this is interesting move, make it above threshold
                altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                altered_move_dic[move_uci] = altered_move_dic[move_uci] * check_sf
                checking_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found checking moves: {checking_moves} \n"
        
        # Takebacks are more appealing
        # We may only calculate this criterion if we have information of previous move
        takeback_sf = self.params_dict.get("takeback_sf", 1.0)
        if prev_board is not None:
            takeback_moves = []
            res = patch_fens(prev_board.fen(), board.fen(), depth_lim=1)
            last_move_uci = res[0][0]
            for move_uci in move_dic.keys():
                if is_takeback(prev_board, last_move_uci, move_uci):
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * takeback_sf
                    takeback_moves.append(board.san(chess.Move.from_uci(move_uci)))
            log += f"Found takeback moves: {takeback_moves} \n"
        
        # Newly threatened en-pris pieces are more appealing to move
        # We may only calculate this criterion if we have information of previous positions
        new_threatened_sf = self.params_dict.get("new_threatened_sf", 1.0)
        if prev_board is not None:            
            new_threatened_moves = []
            # First get all the squares our own pieces and work out whether they are
            # newly threatened or not
            from_squares = [sq for piece_type in range(1, 6) for sq in board.pieces(piece_type, board.turn)]
            from_sq_dic = {from_sq: is_newly_attacked(prev_board, board, from_sq) for from_sq in from_squares}
            newly_attacked_squares = [sq for sq in from_sq_dic if from_sq_dic[sq] > 0.6]
            for move_uci in move_dic.keys():
                dummy_board = board.copy()
                dummy_board.push_uci(move_uci)
                for square in newly_attacked_squares:
                    threatened_levels = calculate_threatened_levels(square, dummy_board)
                    difference_in_threatened = from_sq_dic[square] - threatened_levels
                    if difference_in_threatened > 0.6:
                        # then this is interesting move, make it above threshold
                        altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                        altered_move_dic[move_uci] = altered_move_dic[move_uci] * new_threatened_sf * (1 + difference_in_threatened) ** 0.2
                        new_threatened_moves.append(board.san(chess.Move.from_uci(move_uci)))
                        break
            log += f"Found moves that respond to newly threatened pieces: {new_threatened_moves} \n"
        
        # Offering exchanges/exchanging when material up appealing
        # likewise offering exchanges when material down unappealing
        exchange_sf = self.params_dict.get("exchange_sf", 1.0)
        exchange_k_danger_sf = self.params_dict.get("exchange_k_danger_sf", 1.0)
        
        mat_dic = {1:1, 2:3.1, 3:3.5, 4:5.5, 5:9.9, 6:3}
        own_mat = sum([len(board.pieces(x, board.turn))*mat_dic[x] for x in range(1,6)])
        opp_mat = sum([len(board.pieces(x, not board.turn))*mat_dic[x] for x in range(1,6)])
        material_score = own_mat - opp_mat

        good_exchanges = []
        bad_exchanges = []
        good_king_exchanges = []
        bad_king_exchanges = []
        
        for move_uci in move_dic.keys():
            if is_offer_exchange(board, move_uci):
                if material_score > 2.9:  # If we are ahead by at least a pawn and a half, exchanges are good
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * exchange_sf
                    good_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
                elif material_score < -2.9:  # If we are behind by at least a pawn and a half, no exchanges                    
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * (1.0 / exchange_sf)
                    bad_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
                    
                # If King is in danger, then exchanges are good
                if self_king_danger_lvl > 500 and self_king_danger_lvl > opp_king_danger_lvl + 400:
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * exchange_k_danger_sf
                    good_king_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
                elif opp_king_danger_lvl > 500 and opp_king_danger_lvl > self_king_danger_lvl + 400:
                    # Keep pieces on board
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * (1.0 / exchange_k_danger_sf)
                    bad_king_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
        log += f"Found moves that encourage exchanges as we are material up: {good_exchanges} \n"
        log += f"Found moves that trade when we are material down: {bad_exchanges} \n"
        log += f"Found moves that encourage exchanges when our king is in danger: {good_king_exchanges} \n"
        log += f"Found moves that trade when enemy king is in danger: {bad_king_exchanges} \n"

        # In endgame passed pawns need to be pushed
        passed_pawn_end_sf = self.params_dict.get("passed_pawn_end_sf", 1.0)
        if game_phase == "endgame":
            passed_pawn_moves = []
            for move_uci in move_dic.keys():
                move_obj = chess.Move.from_uci(move_uci)
                from_square = move_obj.from_square
                # check if the piece is a pawn, if so check if it is passed
                if board.piece_type_at(from_square) != 1:  # 1 = pawn
                    continue
                # Check if the pawn is passed, by checking if there are enemy pawns in any of the three files (including its own)
                passed = True
                pawn_file = chess.square_file(from_square)
                pawn_rank = chess.square_rank(from_square)
                for file in range(max(0, pawn_file - 1), min(7, pawn_file + 1) + 1):
                    for rank in range(pawn_rank, 8 if board.turn else -1, 1 if board.turn else -1):
                        if rank < 0 or rank > 7:
                            continue
                        if file == pawn_file and rank == pawn_rank:
                            continue
                        square = chess.square(file, rank)
                        opp_piece_at_sq = board.piece_at(square)
                        if opp_piece_at_sq is not None and opp_piece_at_sq.piece_type == 1 and opp_piece_at_sq.color != board.turn:
                            passed = False
                            break
                    if not passed:
                        break
                if passed:
                    # Adjust how far we are from promoting
                    to_promote = 7 - pawn_rank if board.turn else pawn_rank
                    scaling = ((8 - to_promote) ** 1.5) / (8 ** 1.5)
                    # then this is interesting move, make it above threshold
                    altered_move_dic[move_uci] = max(altered_move_dic[move_uci] - threshold, 0) + threshold
                    altered_move_dic[move_uci] = altered_move_dic[move_uci] * passed_pawn_end_sf * scaling
                    passed_pawn_moves.append(board.san(chess.Move.from_uci(move_uci)))
            log += f"Found moves that move passed pawns forward: {passed_pawn_moves} \n"
        
        # Repeat moves are undesirable
        repeat_sf = torch.exp(self.params_dict.get("repeat_sf", 0.0))
        # These moves are defined to be moves which repeatedly move a piece, and goto a square 
        # which we could have gone to with the piece last move (including moving piece back and forth)
        if prev_board is not None and prev_prev_board is not None:
            repeat_moves = []
            res = patch_fens(prev_prev_board.fen(), prev_board.fen(), depth_lim=1)
            last_own_move_uci = res[0][0]
            last_own_move_obj = chess.Move.from_uci(last_own_move_uci)
            previous_reach = set([move.to_square for move in prev_prev_board.legal_moves if move.from_square == last_own_move_obj.from_square] + [last_own_move_obj.from_square])
            for move_uci in move_dic.keys():
                move_obj = chess.Move.from_uci(move_uci)
                if move_obj.from_square == last_own_move_obj.to_square: # if we are moving the same piece as before
                    to_square = move_obj.to_square
                    if to_square in previous_reach: # then we have found repeating move
                        altered_move_dic[move_uci] *= repeat_sf
                        repeat_moves.append(board.san(move_obj))
            
            log += "Found moves that are repetitive, and waste time: {} \n".format(repeat_moves)

        # Now, normalize probabilities
        total_prob = sum(altered_move_dic.values())
        
        # Add stability check for total_prob
        if total_prob <= 0:
            # If total_prob is invalid, reset move_dic to original with small random noise
            altered_move_dic = {k: float(v) + 1e-8 for k, v in move_dic.items()}
            total_prob = sum(altered_move_dic.values())
        
        # Add a small epsilon to prevent division by zero
        epsilon = 1e-8
        normalized_total = total_prob + epsilon
        
        for move_uci in altered_move_dic:
            # Make sure we don't have extremely small values that could cause numerical issues
            prob_value = altered_move_dic[move_uci]
            if prob_value < epsilon:
                prob_value = epsilon
            
            # Normalize the probability
            altered_move_dic[move_uci] = prob_value / normalized_total
        
        # sort move_dic by value, highest prob first
        altered_move_dic = dict(sorted(altered_move_dic.items(), key=lambda item: item[1], reverse=True))

        log_dic = {board.san(chess.Move.from_uci(k)) : round(v, 3) for k,v in altered_move_dic.items()}
        log += f"Move_dic after alteration: {log_dic} \n"
        
        return altered_move_dic, log 