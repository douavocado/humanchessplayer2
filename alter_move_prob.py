import chess
import numpy as np

from common.board_information import (
    phase_of_game, PIECE_VALS, king_danger, get_threatened_board, is_capturing_move, is_capturable,
    is_attacked_by_pinned, is_check_move, is_takeback, is_newly_attacked, is_offer_exchange,
    is_open_file, calculate_threatened_levels, is_weird_move
            )
from common.utils import patch_fens

def alter_move_probabilties(move_dic : dict, board:chess.Board, prev_board:chess.Board = None, prev_prev_board:chess.Board = None, mood:str = "confident"):
    """ Given a move dictionary with move uci as key and value as their unaltered
        probabilities, we alter the probabilties to make moves stick out more
        (for example hanging pieces more likely to be moved etc).
        
        Returns an altered move_dic. 
    """
    log = ""
    game_phase = phase_of_game(board) # useful for function
    self_king_danger_lvl = king_danger(board, board.turn, game_phase)
    opp_king_danger_lvl = king_danger(board, not board.turn, game_phase)
    
    # Moves which protect/block/any way that make our pieces less en pris are appealing
    # Factor is given by factor = np.exp(-lvl_diff/10)        
    # Moves which make opponent pieces more en pris (attacks/threatens) are appealing
    # Factor is given by factor = np.exp(lvl_diff/10)
    
    # Punish weird moves
    weird_move_sd_dic = {"opening" : 0.1,
                            "midgame" : 0.3,
                            "endgame" : 1.0,
        }
    
    lower_threshold_prob = sum(move_dic.values())/len(move_dic)
    
    # If king danger high, then moves that defend our king are more attractive
    protect_king_sf = 2.8
    # Capturing enpris pieces are more appealing, the more the piece is enpris the more appealing
    capture_en_pris_sf = 1.5
    # Squares that pinned pieces attack that break the pin are more desirable
    break_pin_sf = 3.0
    # Captures are just generally more appealing. The bigger the capture the more appealing
    # if opponent king is in danger, boost even more
    if opp_king_danger_lvl > 500:
        capture_sf = 2
    else:
        capture_sf = 1.5
    # Capturable pieces are more appealling to move
    capturable_sf = 1.3
    
    # Checks are more appealing (particularly under time pressure)
    check_sf_dic = {"confident": 2.3,
                    "cocky": 3.3,
                    "cautious": 2.1,
                    "tilted": 3.3,
                    "hurry": 3.0,
                    "flagging": 2.9}
    # if opponent king is in danger, checks are even more appealing
    if opp_king_danger_lvl - self_king_danger_lvl > 500:
        check_sf_dic = {k: v*1.3 for k, v in check_sf_dic.items()}
        
    # Takebacks are more appealing
    takeback_sf = 2.5
    # Newly threatened en-pris pieces are more appealling to move
    new_threatened_sf_dic = {"confident": 3.5,
                    "cocky": 2.7,
                    "cautious": 3.9,
                    "tilted": 2.1,
                    "hurry": 2.4,
                    "flagging": 3.2}
    
    # Offering exchanges/exchanging when material up appealing
    exchange_sf_dic = {"confident": 3.2,
                    "cocky": 2.0,
                    "cautious": 2.2,
                    "tilted": 1.5,
                    "hurry": 3.8,
                    "flagging": 0.8}
    # Offering exchanges/exchanging when king danger is high appealing
    exchange_k_danger_sf_dic = {"confident": 3.4,
                    "cocky": 2.1,
                    "cautious": 3.9,
                    "tilted": 1.0,
                    "hurry": 3.0,
                    "flagging": 2.5}
    # Pushing passed pawns in endgame more appealing
    passed_pawn_end_sf = 3.0
    
    # Repeat moves are undesirable
    repeat_sf_dic = {"confident": 0.3,
                    "cocky": 0.5,
                    "cautious": 0.3,
                    "tilted": 0.6,
                    "hurry": 0.5,
                    "flagging": 0.4}
    
    
###############################################################################
    
    
    # Moves which protect/block/any way that make our pieces less en pris are appealing        
    # Likewise moves that make our pieces more en pris are less appealing
    # Moves which make opponent pieces more en pris (attacks/threatens) are appealing
    strenghening_moves = []
    weakening_moves = []     
    weird_moves = []
    # For time computation sake, we ignore the threatened levels of pawns
    curr_threatened_board = get_threatened_board(board, colour=board.turn, piece_types=[1,2,3,4,5])
    self_curr_threatened_levels = sum(curr_threatened_board)
    
    # if out of the our threatened pieces there was only one thing that was threatened, then
    # exaggerate the levels.
    if sum(np.array(curr_threatened_board) > 0.5) == 1:
        solo_factor = 2
    else:
        solo_factor = 1
    
    opp_curr_threatened_levels = sum(get_threatened_board(board, colour=(not board.turn), piece_types=[1,2,3,4,5]))
    for move_uci in move_dic.keys():
        move_obj = chess.Move.from_uci(move_uci)
        dummy_board = board.copy()
        dummy_board.push(move_obj)
        new_threatened_board = get_threatened_board(dummy_board, colour=board.turn, piece_types=[1,2,3,4,5])
        self_new_threatened_levels = sum(new_threatened_board)
        # if new move makes our piece en_pris, then we calculate enemy threatened levels
        # as if there was no such piece. This prevents scenarios where we attack
        # opposition pieces of higher value at the cost of sacrificing our original
        # attacking piece.
        # if our move was a capture, we must take that into account that we gained some material
        piece_type = board.piece_type_at(move_obj.to_square)
        if piece_type is not None:
            to_value = PIECE_VALS[piece_type]
        else:
            to_value = 0
            
        if new_threatened_board[move_obj.to_square] - to_value  > 0.6:
            dummy_board.remove_piece_at(move_obj.to_square)
        
        
        opp_new_threatened_levels = sum(get_threatened_board(dummy_board, colour=(not board.turn), piece_types=[1,2,3,4,5]))
        self_lvl_diff = self_new_threatened_levels - self_curr_threatened_levels
        
        opp_lvl_diff = opp_new_threatened_levels - opp_curr_threatened_levels
        
        # psychologically, protecting pieces is more favorable than attacking pieces
        # therefore we weight being in a safer position more heavily than being in a 
        # more pressuring situation
        # however this happens only if our move is not a capture.
        
        
        lvl_diff = opp_lvl_diff - self_lvl_diff*1.25*solo_factor
        
        if piece_type is not None: # if we captured something
            lvl_diff += to_value
        factor = np.exp(lvl_diff/2)     
        
        # Some moves of the strengthening moves are weird and unlikely due to the
        # fault of the nn. We want these moves to be visible, so we set a threshold
        # for which if these moves have lower prob, we raise them to so that the factor
        # changes are significant.
        if lvl_diff > 0.9:
            move_dic[move_uci] = max(move_dic[move_uci], lower_threshold_prob)
        move_dic[move_uci] *= factor
        if lvl_diff > 0.9:
            strenghening_moves.append(board.san(chess.Move.from_uci(move_uci)))
        elif lvl_diff < -0.9:
            weakening_moves.append(board.san(chess.Move.from_uci(move_uci)))
        else:
            # if move doesn't seem to do anything that threatens or protects pieces, then we check
            # whether it is a "weird" move
            if is_weird_move(board, game_phase, move_uci, self_king_danger_lvl):
                # then incur penalty
                move_dic[move_uci] *= weird_move_sd_dic[game_phase]
                weird_moves.append(board.san(chess.Move.from_uci(move_uci)))

    log += "Found moves that are weakening and make our pieces more enpris/opp pieces less enpris: {} \n".format(weakening_moves)
    log += "Found moves that protect our pieces more or apply more pressure to opponent: {} \n".format(strenghening_moves)
    log += "Found weird moves: {} \n".format(weird_moves)
        
    
    # Squares that pinned pieces attack that break the pin are more desirable to move to
    adv_pinned_moves = []
    for move_uci in move_dic.keys():
        to_square = chess.Move.from_uci(move_uci).to_square
        no_pinned_atks = is_attacked_by_pinned(board, to_square, not board.turn)
        if no_pinned_atks > 0:
            # it is a capturing move
            if move_dic[move_uci] < lower_threshold_prob:
                move_dic[move_uci] = lower_threshold_prob
            move_dic[move_uci] *= (break_pin_sf**no_pinned_atks)
            adv_pinned_moves.append(board.san(chess.Move.from_uci(move_uci)))
    log += "Found moves that take advantage of pinned pieces: {} \n".format(adv_pinned_moves)
        
    # If king danger high, then moves that defend our king are more attractive
    before_king_danger = king_danger(board, board.turn, game_phase)
    # if king is not in danger, pass
    if before_king_danger < 250:
        log += "King danger {} not high to consider protecting king moves. Skipping... \n".format(before_king_danger)
    else:
        protect_king_moves = []            
        for move_uci in move_dic.keys():
            dummy_board= board.copy()
            dummy_board.push_uci(move_uci)
            new_king_danger = king_danger(dummy_board, board.turn, game_phase)
            if new_king_danger <= 0:
                # found move
                if move_dic[move_uci] < lower_threshold_prob:
                    move_dic[move_uci] = lower_threshold_prob
                move_dic[move_uci] *= protect_king_sf*(before_king_danger/50)**(1/4)
                protect_king_moves.append(board.san(chess.Move.from_uci(move_uci)))
            elif before_king_danger/new_king_danger > 1.5:
                # found move
                if move_dic[move_uci] < lower_threshold_prob:
                    move_dic[move_uci] = lower_threshold_prob
                denom = max(new_king_danger, 50)
                move_dic[move_uci] *= protect_king_sf*(before_king_danger/denom)**(1/4)
                protect_king_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += "Found moves that protect our vulnerable king: {} \n".format(protect_king_moves)
    # Capturing moves are more appealing
    capturing_moves = []
    for move_uci in move_dic.keys():
        if is_capturing_move(board, move_uci):
            # it is a capturing move
            piece_value = PIECE_VALS[board.piece_type_at(chess.Move.from_uci(move_uci).to_square)]
            move_dic[move_uci] *= capture_sf * (piece_value**0.25)
            capturing_moves.append(board.san(chess.Move.from_uci(move_uci)))
    log += "Found capturing moves from position: {} \n".format(capturing_moves)
    
    # Capturing enpris pieces are more appealing
    capturing_enpris_moves = []
    for move_uci in move_dic.keys():
        if is_capturing_move(board, move_uci):
            # it is a capturing move
            move_obj = chess.Move.from_uci(move_uci)
            threatened_lvls = calculate_threatened_levels(move_obj.to_square, board)
            # not only is the captured piece enpris, but we are capturing it with the correct piece
            piece_type = board.piece_type_at(move_obj.to_square)
            to_value = PIECE_VALS[piece_type]
            dummy_board = board.copy()
            dummy_board.push(move_obj)
            if threatened_lvls > 0.6 and calculate_threatened_levels(move_obj.to_square, dummy_board) - to_value < 0:  # captured piece is enpris
                move_dic[move_uci] *= capture_en_pris_sf * (threatened_lvls**0.25)
                capturing_enpris_moves.append(board.san(move_obj))
    log += "Found capturing enpris moves from position: {} \n".format(capturing_enpris_moves)
    
    # Capturable pieces are more appealling to move
    capturable_moves = []
    for move_uci in move_dic.keys():
        from_square = chess.Move.from_uci(move_uci).from_square
        if is_capturable(board, from_square):
            # it is a capturing move
            move_dic[move_uci] *= capturable_sf
            capturable_moves.append(board.san(chess.Move.from_uci(move_uci)))
    log += "Found moves that move capturable pieces: {} \n".format(capturable_moves)
    
    # Checks are more appealing (particularly under time pressure)        
    checking_moves = []
    for move_uci in move_dic.keys():
        if is_check_move(board, move_uci):
            # depending what mood we are, checks are more attractive
            move_dic[move_uci] *= check_sf_dic[mood]
            checking_moves.append(board.san(chess.Move.from_uci(move_uci)))
    log += "Found checking moves: {} \n".format(checking_moves)
    
    # Takebacks are more appealing
    # We may only calculate this criterion if we have information of previous move
    if prev_board is not None:
        takeback_moves = []
        res = patch_fens(prev_board.fen(), board.fen(), depth_lim=1)
        last_move_uci = res[0][0]
        for move_uci in move_dic.keys():
            if is_takeback(prev_board, last_move_uci, move_uci):
                move_dic[move_uci] *= takeback_sf
                takeback_moves.append(board.san(chess.Move.from_uci(move_uci)))
        log += "Found takeback moves: {} \n".format(takeback_moves)
    
    # calc threatened of prev board and board of from square
    # Newly threatened en-pris pieces are more appealling to move
    # We may only calculate this criterion if we have information of previous positions
    if prev_board is not None:            
        new_threatened_moves = []
        # first get all the squares our own pieces and work out whether they are
        # newly threatened or not
        from_squares = [sq for piece_type in range(1,6) for sq in board.pieces(piece_type, board.turn) ]
        from_sq_dic = {from_sq : is_newly_attacked(prev_board, board, from_sq) for from_sq in from_squares}
        newly_attacked_squares = [sq for sq in from_sq_dic if from_sq_dic[sq] > 0.6]
        for move_uci in move_dic.keys():
            dummy_board = board.copy()
            dummy_board.push_uci(move_uci)
            for square in newly_attacked_squares:
                threatened_levels = calculate_threatened_levels(square, dummy_board)
                difference_in_threatened = from_sq_dic[square] - threatened_levels
                if difference_in_threatened > 0.6:
                    # depending what mood we are, we may be more sensitive to new attacks
                    # also depends on the value of what is newly being attacked
                    move_dic[move_uci] *= new_threatened_sf_dic[mood] * (1 + difference_in_threatened)**0.2
                    new_threatened_moves.append(board.san(chess.Move.from_uci(move_uci)))
                    break
        log += "Found moves that respond to newly threatened pieces: {} \n".format(new_threatened_moves)
        
    # Offering exchanges/exchanging when material up appealing
    # likewise offering exchanges when material down unappealing
    # calculates threatened levels of every move after its moved at to_square when material imbalance
    good_exchanges = []
    bad_exchanges = []
    # first sum the material on our side compared with opponent side
    mat_dic = {1:1, 2:3.1, 3:3.5, 4:5.5, 5:9.9, 6:3}
    own_mat = sum([len(board.pieces(x, board.turn))*mat_dic[x] for x in range(1,6)])
    opp_mat = sum([len(board.pieces(x, not board.turn))*mat_dic[x] for x in range(1,6)])
    if own_mat - opp_mat > 2.9: # if we are more than a knight up, encourage trades
        for move_uci in move_dic.keys():
            if is_offer_exchange(board, move_uci) == True:
                move_dic[move_uci] *= exchange_sf_dic[mood]  
                good_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
    elif opp_mat - own_mat > 2.9: # if we are more than a knight down, discourage trades
        for move_uci in move_dic.keys():
            if is_offer_exchange(board, move_uci) == True:
                move_dic[move_uci] /= exchange_sf_dic[mood] 
                bad_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
    log += "Found moves that encourage exchanges as we are material up: {} \n".format(good_exchanges)
    log += "Found moves that trade when we are material down: {} \n".format(bad_exchanges)
    
    # Offering exchanges/exchanging when king danger is high appealing
    # However if opponent is in even more danger, than we pieces on board
    # calculates threatened levels of every move after its moved at to_square when king danger imbalance
    good_king_exchanges = []
    bad_king_exchanges = []
    
    
    if abs(opp_king_danger_lvl - self_king_danger_lvl) < 400:
        # indifferent, keep pieces on
        pass
    elif opp_king_danger_lvl - self_king_danger_lvl >= 400 and opp_king_danger_lvl > 500:
        # opponent is in much higher danger level than we are
        # discourage trades
        for move_uci in move_dic.keys():
            if is_offer_exchange(board, move_uci) == True:
                move_dic[move_uci] /= exchange_k_danger_sf_dic[mood]
                bad_king_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
    elif self_king_danger_lvl - opp_king_danger_lvl >= 400 and self_king_danger_lvl > 500:
        # we are in much more king danger than opponent
        # encourage trades
        for move_uci in move_dic.keys():
            if is_offer_exchange(board, move_uci) == True:
                move_dic[move_uci] *= exchange_k_danger_sf_dic[mood]
                good_king_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
    log += "Found moves that encourage exchanges when our king is in danger: {} \n".format(good_king_exchanges)
    log += "Found moves that trade when enemy king is in danger: {} \n".format(bad_king_exchanges)
        
    # Pushing passed pawns in endgame more appealing
    if game_phase == "endgame":
        passed_pawn_moves = []
        for move_uci in move_dic.keys():
            move_obj = chess.Move.from_uci(move_uci)
            move_piece_type = board.piece_type_at(move_obj.from_square)
            if move_piece_type == chess.PAWN:
                passed_result = is_open_file(board, chess.square_file(move_obj.from_square))
                if board.turn == chess.WHITE and passed_result == -2:                    
                    move_dic[move_uci] *= passed_pawn_end_sf
                    passed_pawn_moves.append(board.san(move_obj))
                elif board.turn == chess.BLACK and passed_result == 2:
                    move_dic[move_uci] *= passed_pawn_end_sf
                    passed_pawn_moves.append(board.san(move_obj))
        log += "Found moves that push passed pawns in the endgame: {} \n".format(passed_pawn_moves)
        
    # Normalising and sorting finalising probabilities
    total = sum(move_dic.values()) + 10**(-10)
    move_dic = {k : move_dic[k]/total for k in sorted(move_dic.keys(), reverse=True, key=lambda x: move_dic[x])}
    
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
                    move_dic[move_uci] *= repeat_sf_dic[mood]
                    repeat_moves.append(board.san(move_obj))
        
        log += "Found moves that are repetitive, and waste time: {} \n".format(repeat_moves)
    
    # make the move_dic and prob look nicely formatted in the log
    log_move_dic = {board.san(chess.Move.from_uci(k)) : round(v, 5) for k,v in move_dic.items()}
    log += "Move_dic after alteration: {} \n".format(log_move_dic)        
    return move_dic, log