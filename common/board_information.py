# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 19:10:11 2023

@author: xusem
"""
import chess
import numpy as np
import math

from .constants import PATH_TO_STOCKFISH

STOCKFISH = chess.engine.SimpleEngine.popen_uci(PATH_TO_STOCKFISH) # replace this line with the correct binary/executable
PIECE_VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.5, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 1000}

# value of each piece, used in engine.check_obvious for takebacks
points_dic = {chess.PAWN: 2,
              chess.KNIGHT: 20,
              chess.BISHOP: 20.5,
              chess.ROOK: 35.5,
              chess.QUEEN: 55,
              chess.KING: 0}
# danger value of each piece to the king
danger_dic = {chess.PAWN: 15,
              chess.KNIGHT: 10,
              chess.BISHOP: 10.5,
              chess.ROOK: 35.5,
              chess.QUEEN: 55,
              chess.KING: 0}
# protective value of each piece to the king
prot_dic = {chess.PAWN: 40,
              chess.KNIGHT: 20,
              chess.BISHOP: 20.5,
              chess.ROOK: 25.5,
              chess.QUEEN: 55,
              chess.KING: 0}
# a different value dictionary for checking en_pris
p_dic = {chess.PAWN: 1,
              chess.KNIGHT: 3,
              chess.BISHOP: 3,
              chess.ROOK: 5,
              chess.QUEEN: 9,
              chess.KING: 100}


def get_attackers_en_pris(board):
    ''' given a board, get the attackers (of current turn colour) that are attacking
        and en pris target such that the exchange is favourable. Return dictionary
        of squares and their attractiveness to move. '''
    opp_colour = not board.turn
    opp_threatened_board = get_threatened_board(board, opp_colour)
    opp_threatened_squares = [i for i in range(64) if opp_threatened_board[i] > 0]
    en_pris_attackers = {}
    for opp_sq in opp_threatened_squares:
        candidates = board.attackers(board.turn, opp_sq)
       
        for candidate in candidates:
            move = chess.Move(candidate,opp_sq)
            dummy_board = board.copy()
            dummy_board.push(move)
            if calculate_threatened_levels(opp_sq, dummy_board) - PIECE_VALS[board.piece_type_at(opp_sq)] <= 0:
                if candidate in en_pris_attackers:
                    en_pris_attackers[candidate] += opp_threatened_board[opp_sq]
                else:
                    en_pris_attackers[candidate] = opp_threatened_board[opp_sq]
    return en_pris_attackers

def get_threatened_board(board, colour=chess.WHITE, piece_types=[1,2,3,4,5,6]):
    ''' Given a position and a colour, create a board identifying how threatened
        each piece of that colour is. '''
    threatened_board = [0]*64
    for piece_type in piece_types:
        squares = board.pieces(piece_type,colour)
        for square in squares:
            levels = calculate_threatened_levels(square, board)
            threatened_board[square] = levels
    
    return threatened_board

def _threatened_recurse(square, board, vals, base_colour):
    opp_colour = not base_colour
    if len(vals) %2 == 1:
        # then we are looking for not base_colour attackers
        attackers = list(board.attackers(opp_colour, square))
        attackers = [a for a in attackers if chess.Move(a, square) in board.legal_moves]
        val_multiplier = 1
    else:
        # then we are looking for base_colour attackers
        attackers = list(board.attackers(base_colour, square))
        attackers = [a for a in attackers if chess.Move(a, square) in board.legal_moves]
        val_multiplier = -1
    if len(attackers) == 0:
        return vals
    else:
        least_val_attacker_sq = min(attackers, key= lambda x: PIECE_VALS[board.piece_type_at(x)])
        new_vals = vals + [vals[-1] + val_multiplier*PIECE_VALS[board.piece_type_at(square)]]

        new_board = board.copy()
        move = chess.Move(least_val_attacker_sq, square)
        # make sure that we arrange its the proper turn
        new_board.turn = new_board.piece_at(least_val_attacker_sq).color
        if move in new_board.legal_moves:
            new_board.push(move)
        else:
            return new_vals + [new_vals[-1] - val_multiplier*1000]
        return _threatened_recurse(square, new_board, new_vals, base_colour)
        
def _exchange_recurse(vals, start_index, current_score, depth):
    if len(vals) == 0:
        return current_score
    if start_index == 0: # then trying to maximise
        stopping_i = vals[start_index::2].index(max(vals[start_index::2]))*2
        if vals[stopping_i] <= current_score and depth != 1:
            # no longer in the interest of the even indices to stop earlier
            return current_score
        elif vals[stopping_i] <= current_score and depth == 1:
            # first iteration slightly different, we need to do an odd index check too
            return _exchange_recurse(vals[:vals.index(current_score)], not start_index, current_score, depth=depth+1)
    elif start_index == 1:
        if len(vals) == 1:
            # only possibility is vals = [0]
            return max(0, current_score)
        stopping_i = 1 + 2*vals[start_index::2].index(min(vals[start_index::2]))
        if vals[stopping_i] >= current_score:
            # no longer in the interest of the even indices to stop earlier
            return current_score
    new_current_score = vals[stopping_i]
    return _exchange_recurse(vals[:stopping_i], not start_index, new_current_score, depth=depth+1)

def _find_exchange_value(vals):
    ''' Given a set of exchanges, find the optimum value with both sides playing perfect '''
    # start with even indicies which we are trying to maximise
    # it actually doesn't matter whether we start with even or odd
    start_index=0
    score = vals[-1]
    score = _exchange_recurse(vals, start_index, score, depth=1)
    return score

def calculate_threatened_levels(square, board):
    if board.piece_at(square) is None:
        return 0
    dummy_board = board.copy()
    dummy_board.turn = not board.color_at(square)
    vals = [0]
    vals = _threatened_recurse(square, dummy_board, vals, board.piece_at(square).color)
    return _find_exchange_value(vals)

def get_potential_threatened(board, sq):
    ''' Given a board and a square (which is occupied), return a dictionary of
        the legal move to squares and the corresponding en pris values, taking
        into account of captures. Note that higher values are more attractive. '''
    # sanity check assertions
    assert board.piece_at(sq) is not None
    assert board.piece_at(sq).color == board.turn
    # now find all legal moves from that square
    relevant_moves = [move for move in board.legal_moves if move.from_square == sq]
    
    return_dic = {}
    for move in relevant_moves:
        if board.piece_at(move.to_square) is not None: # then the move captures something
            handicap = -1*PIECE_VALS[board.piece_type_at(move.to_square)]
        else:
            handicap = 0
        dummy_board = board.copy()
        dummy_board.push(move)
        return_dic[move.to_square] = (calculate_threatened_levels(move.to_square, dummy_board) + handicap)*-1
    
    return return_dic

def get_new_threatened(board, sq):
    ''' Given a board and a square (which is occupied), return a dictionary of
        the legal move to squares and the corresponding values which indicate if
        the new move created a new threat to the opposition (made opposition piece
        en pris). We take into account of captures (piece that disappear). Higher
        values are more attractive moves. '''
    # sanity check assertions
    assert board.piece_at(sq) is not None
    assert board.piece_at(sq).color == board.turn
    # first find oppositions threatened levels
    prev_threatened_board = np.array(get_threatened_board(board,colour=not board.turn))
    
    # now find all legal moves from that square
    relevant_moves = [move for move in board.legal_moves if move.from_square == sq]
    
    return_dic = {}
    for move in relevant_moves:
        if board.piece_at(move.to_square) is not None: # then the move captures something
            handicap = PIECE_VALS[board.piece_type_at(move.to_square)]
        else:
            handicap = 0
        
        dummy_board = board.copy()
        dummy_board.push(move)
        
        # now calculated new threatened board of opposition
        new_threatened_board = np.array(get_threatened_board(dummy_board,colour=dummy_board.turn))
        difference = np.sum(new_threatened_board - prev_threatened_board) + handicap
        
        return_dic[move.to_square] = difference
        # note difference can be negative if the move removes an attacker
    return return_dic

def get_new_hanging(board, sq):
    ''' Given a board and a square (which is occupied), return a dictionary of
        the legal move to squares and the corresponding values which indicate if
        the new move created a new hanging piece in own position. We do NOT take into 
        account of captures (piece that disappear). Note higher values mean
        bad thing (penalty). '''
    # sanity check assertions
    assert board.piece_at(sq) is not None
    assert board.piece_at(sq).color == board.turn
    # first find oppositions threatened levels
    prev_threatened_board = np.array(get_threatened_board(board,colour=board.turn))
    # note we want to ignore the entries where the move moved from/to
    # as we strictly only want to detect other squares left hanging
    prev_threatened_board[sq] = 0
    # now find all legal moves from that square
    relevant_moves = [move for move in board.legal_moves if move.from_square == sq]
    
    return_dic = {}
    for move in relevant_moves:        
        dummy_board = board.copy()
        dummy_board.push(move)
        
        # now calculated new threatened board of opposition
        new_threatened_board = np.array(get_threatened_board(dummy_board,colour=board.turn))
        # note we want to ignore the entries where the move moved from/to
        # as we strictly only want to detect other squares left hanging
        new_threatened_board[sq] = new_threatened_board[move.to_square] = 0
         
        difference = np.sum(new_threatened_board - prev_threatened_board)
        
        return_dic[move.to_square] = difference
        # note difference can be negative if the move removes an attacker
    return return_dic


def is_open_file(board, file):
    ''' Given a position, returns either +2: file is semi open for white, -2:
        file is semi open for black, 0: file is closed or True: file is fully open. '''
    
    bb_dic = {0:chess.BB_FILE_A,
              1:chess.BB_FILE_B,
              2:chess.BB_FILE_C,
              3:chess.BB_FILE_D,
              4:chess.BB_FILE_E,
              5:chess.BB_FILE_F,
              6:chess.BB_FILE_G,
              7:chess.BB_FILE_H}
    
    white_pawn_true = False
    black_pawn_true = False
    
    for square in chess.SquareSet(bb_dic[file]):
        if board.piece_type_at(square) == chess.PAWN:
            if board.color_at(square) == chess.WHITE:
                white_pawn_true = True
            elif board.color_at(square) == chess.BLACK:
                black_pawn_true = True
    
    if white_pawn_true:
        if black_pawn_true:
            return 0
        else:
            return -2
    else:
        if black_pawn_true:
            return 2
        else:
            return True

def is_locked_file(board, file):
    ''' Given a position, check if a given file is pawn locked. '''
    bb_dic = {0:chess.BB_FILE_A,
              1:chess.BB_FILE_B,
              2:chess.BB_FILE_C,
              3:chess.BB_FILE_D,
              4:chess.BB_FILE_E,
              5:chess.BB_FILE_F,
              6:chess.BB_FILE_G,
              7:chess.BB_FILE_H}
    dummy_board = board.copy()
    dummy_board.turn = chess.WHITE
    legal_moves_from_sq_white = [move.from_square for move in dummy_board.legal_moves]
    dummy_board.turn = chess.BLACK
    legal_moves_from_sq_black = [move.from_square for move in dummy_board.legal_moves]
    
    pawn_true = False
    white_pawn_true = False
    black_pawn_true = False
    for square in chess.SquareSet(bb_dic[file]):
        if board.piece_type_at(square) == chess.PAWN:
            pawn_true = True
            if board.color_at(square) == chess.WHITE:
                if square in legal_moves_from_sq_white:
                    white_pawn_true = True
            elif board.color_at(square) == chess.BLACK:
                if square in legal_moves_from_sq_black:
                    black_pawn_true = True
    
    if white_pawn_true == False and black_pawn_true == False and pawn_true == True:
        return True
    else:
        return False

def king_danger(board, side, phase):
    ''' Returns a score for the king danger for a side. '''
    king_danger = 0
    
    king_sq = list(board.pieces(chess.KING, side))[0]
    area_range_file_from = max(0, chess.square_file(king_sq)-2)
    area_range_file_to = min(7, chess.square_file(king_sq)+2)
    area_range_rank_from = max(0, chess.square_rank(king_sq)-3)
    area_range_rank_to = min(7, chess.square_rank(king_sq)+3)
    
    for file_i in range(area_range_file_from, area_range_file_to+1):
        for rank_i in range(area_range_rank_from, area_range_rank_to+1):
            square = chess.square(file_i, rank_i)
            squares_from_k = chess.square_distance(square, king_sq)
            for attacker_sq in board.attackers(not side, square):
                king_danger += (4 - squares_from_k)*danger_dic[board.piece_type_at(attacker_sq)]
            
            for defender_sq in board.attackers(side, square):
                king_danger -= (4 - squares_from_k)/1.5*prot_dic[board.piece_type_at(defender_sq)]
    
    # deal with open files in opening and midgame
    if phase != 'endgame':
        for file_i in range(max(0, chess.square_file(king_sq)-1), min(7, chess.square_file(king_sq)+1) + 1):
            open_ = is_open_file(board, file_i)
            if open_ == True:
                king_danger += 500
            elif side == chess.WHITE and open_ == +2:
                king_danger += 400
            elif side == chess.BLACK and open_ == -2:
                king_danger += 400
    
    # does the opposition have her queen?
    king_danger += len(board.pieces(chess.QUEEN, not side))*300
    
    return king_danger

def is_weird_move(board, phase, move, obvious_move, king_dang):
    ''' Given a chess.Move object, and the phase of the game, return whether
        a move looks 'weird' or computer like. Mainly concerns rook and queen moves. '''
    # if move is by far the obvious move it is not weird
    if move.uci() == obvious_move:
        return False
    # if the move is a rook move on the base rank, punish rook moves which
    # 'squash' each other
    
    square_from = move.from_square
    square_to = move.to_square
    side = board.color_at(square_from)
    if side is None:
        # not a valid move
        raise Exception('Not a valid move from square when calculating weirdness.')
    if board.piece_type_at(square_from) == chess.ROOK:
        if chess.square_rank(square_from) == 0 and side == chess.WHITE:
            if square_to == chess.E1 and square_from != chess.F1 and board.piece_type_at(chess.F1) == chess.ROOK and board.color_at(chess.F1) == side:
                return True
            elif square_to == chess.B1 and square_from != chess.A1 and board.piece_type_at(chess.A1) == chess.ROOK and board.color_at(chess.A1) == side:
                return True
        elif chess.square_rank(square_from) == 7 and side == chess.BLACK:
            if square_to == chess.E8 and square_from != chess.F8 and board.piece_type_at(chess.F8) == chess.ROOK and board.color_at(chess.F8) == side:
                return True
            elif square_to == chess.B8 and square_from != chess.A8 and board.piece_type_at(chess.A8) == chess.ROOK and board.color_at(chess.A8) == side:
                return True
        
        # if the move is a rook move to the 2nd or third rank and it's not the obvious move
        open_ = is_open_file(board, chess.square_file(square_to))
        if chess.square_rank(square_to) in [1,2] and side == chess.WHITE and phase != 'endgame' and open_ != True:
            return True
        elif chess.square_rank(square_to) in [5,6] and side == chess.BLACK and phase != 'endgame' and open_ != True:
            return True
        
        # if the move is a rook move to a completely closed file
        if is_locked_file(board, chess.square_file(square_to)) == True:
            return True
        
    # if the move is a gueen move onto the back rank in the opening
    elif board.piece_type_at(square_from) == chess.QUEEN and phase != 'endgame':
        if chess.square_rank(square_to) == 0 and side == chess.WHITE:
            return True
        elif chess.square_rank(square_to) == 7 and side == chess.BLACK:
            return True
    
    # if the move is a king move to a random square when king safety is good
    elif board.piece_type_at(square_from) == chess.KING and king_dang < 400:
        return True
    
    return False

def is_quiet_move(board, move):
    ''' Given a chess.Move object, check if is a quiet move or not i.e. a human
        would play quickly without thinking. '''
    # check the move does not capture anything
    if board.color_at(move.to_square) == (not board.turn):
        return False
    # check it does not attack anything
    prev_fen = board.fen()
    dummy_board = board.copy()
    dummy_board.push(move)
    fen = dummy_board.fen()
    if new_attacked(prev_fen, fen, (not board.turn))[0]:
        return False
    
    # next check that the move itself does not place of it's own pieces in en pris
    if new_attacked(prev_fen, fen, board.turn)[0]:
        return False
    
    return True

def is_capturing_move(board:chess.Board, move_uci:str):
    """ Given a move, determine whether the move captures a piece or not, regardless
        of whether it's favourable to do so.
        
        Returns bool
    """
    move = chess.Move.from_uci(move_uci)
    turn = board.color_at(move.from_square)
    if turn is None:
        raise Exception("There is not such move {} possibly made from board".format(move.uci()))
    
    if board.color_at(move.to_square) == (not turn):
        return True
    else:
        return False

def is_capturable(board:chess.Board, square:chess.Square):
    """ Returns whether a piece at a square is capturable or not. """
    piece_color = board.color_at(square)
    if piece_color is None:
        raise Exception("There is no piece on square {} on the board".format(square))
    return len(board.attackers(not piece_color, square)) > 0

def is_pinned_to_enpris(board:chess.Board, square:chess.Square):
    """ Checks whether a piece is pinned. To decide whether a piece is pinned
        or not, check whether removing it leaves a new piece en_pris if we remove it
        
        Returns a float, square. The float indicates the pinned value (the threatened levels of the pin)
        0 is equivalent to no pin. The square is the location of the square that the piece is pinned
        to, if any (None if no pin).
    """
    side = board.color_at(square)
    if side is None:
        return 0, None
    # In order for a square to be pinned, it must first be attacked by the pinning piece.
    # Furthermore the pinning piece must also attack the piece that our square 
    # is potentially pinned to.
    
    # First identify the pinning piece(s)
    # Only bishop, rook and queen can pin pieces
    attacker_sqs = board.attackers(not side, square)
    attacker_sqs = [sq for sq in attacker_sqs if board.piece_type_at(sq) in [chess.BISHOP, chess.QUEEN, chess.ROOK]]
    dummy_board = board.copy()
    # remove piece at square in dummy_board
    dummy_board.remove_piece_at(square)
    
    pinned_to_squares = []
    for atk_sq in attacker_sqs:
        before_attacks = board.attacks(atk_sq)
        after_attacks = dummy_board.attacks(atk_sq)
        difference_set = after_attacks - before_attacks # difference in attacking squares
        revealed_piece_sq = [sq for sq in difference_set if board.color_at(sq) == side]
        # maximum only one such piece
        if len(revealed_piece_sq) > 0:
            pinned_to_squares.append(revealed_piece_sq[0])
    
    # Now that we have identified the pinned-to pieces, only thing left is to see
    # whether they are en_pris or not
    if len(pinned_to_squares) == 0:
        return 0, None
    else:
        pinned_value = 0
        pinned_sq = None
        for sq in pinned_to_squares:
            _, threatened_levels = is_en_pris(dummy_board, sq)
            if threatened_levels > pinned_value:
                pinned_value = threatened_levels
                pinned_sq = sq
        return pinned_value, pinned_sq

def is_attacked_by_pinned(board:chess.Board, square: chess.Square, side: bool):
    """ Returns whether a square is attacked by a pinned piece from side. 
        
        Returns the number of pinned pieces that attack that square (int)
    """
    pinned_attackers = 0
    attacker_squares = board.attackers(side, square)
    for attack_sq in attacker_squares:
        # check if the attack_sq is pinned piece
        pinned_val, pinned_to_sq = is_pinned_to_enpris(board, attack_sq)
        if pinned_val > 0:
            # further check if we make the move, is the pinned to square still en pris?
            # this eliminates cases where say a pinned bishop may retreat to protect
            # an unprotected piece, making the piece no longer hanging
            dummy_board = board.copy()
            dummy_board.set_piece_at(square, chess.Piece(board.piece_type_at(attack_sq),side))
            dummy_board.remove_piece_at(attack_sq) 
            en_pris, _ = is_en_pris(dummy_board, pinned_to_sq)
            if en_pris == True:
                #print("Found pinned attacker at {}".format(chess.square_name(attack_sq)))
                pinned_attackers += 1
    return pinned_attackers

def is_takeback(prev_move_uci: str, move_uci: str):
    """ Given a move and the previously played move, return True if move_uci is
        a takeback. Otherwise return False.
    """
    prev_move_obj = chess.Move.from_uci(prev_move_uci)
    move_obj = chess.Move.from_uci(move_uci)
    return move_obj.to_square == prev_move_obj.to_square

def is_check_move(board:chess.Board, move_uci:str):
    """ Returns whether a move gives a check or not.
    
        Returns bool.
    """
    move_obj = chess.Move.from_uci(move_uci)
    side = board.color_at(move_obj.from_square)
    if side is None:
        raise Exception("Move {} is not a legal move in board {}. Cannot check for checks".format(move_uci, board.fen()))
    
    dummy_board = board.copy()
    dummy_board.turn = side
    if move_obj not in dummy_board.legal_moves:
        raise Exception("Move {} is not a legal move in board {}. Cannot check for checks".format(move_uci, board.fen()))
    else:
        dummy_board.push(move_obj)
        return dummy_board.is_check()

def is_newly_attacked(prev_board:chess.Board, curr_board:chess.Board, square: chess.Square):
    """ Given a previous board and current board, determine whether the piece at our current
        square has become more en_pris than before.
        
        Returns a float measuring how much more en_pris the piece has become
        (difference in threatened levels)
    """
    if prev_board.piece_at(square) is None or curr_board.piece_at(square) is None:
        raise Exception("Cannot calculated whether piece is newly attacked at square {} if it not present in both prev board and curr board.".format(chess.square_name(square)))
    before_levels = calculate_threatened_levels(square, prev_board)
    current_levels = calculate_threatened_levels(square, curr_board)
    return max(current_levels - before_levels, 0)
    
def is_offer_exchange(board:chess.Board, move_uci: str):
    """ Given a board and move_uci, return True if the move is offering an even
        exchange (knight/bishop exchanges count). Else return False
    """
    dummy_board = board.copy()
    move_obj = chess.Move.from_uci(move_uci)
    self_worth = PIECE_VALS[dummy_board.piece_type_at(move_obj.from_square)]
    to_square = move_obj.to_square
    dummy_board.push(move_obj)
    # first check newly moved piece is not en_pris
    if calculate_threatened_levels(to_square, dummy_board) < 0.6:
        # next check that the newly moved piece has attackers of the same piece value
        # where knight and bishop count the same
        attackers = dummy_board.attackers(dummy_board.turn, to_square)
        attacker_vals = [PIECE_VALS[dummy_board.piece_type_at(sq)] for sq in attackers]
        if len(attacker_vals) == 0: # no attackers anyway
            return False
        lowest = min(attacker_vals)
        if self_worth - lowest >  -0.1: # offering a knight for a bishop to take is not really an exchange
            return True
        else:
            return False
    else:
        return False

def check_best_takeback_exists(board:chess.Board, last_move_uci:str):
    """ Given a board and the last move played, check whether the current turn side
        has a takeback, and whether the takeback is (significantly) the best move. 
        
        Returns bool and the takeback move uci string. If the takeback move is not best
        then returns False, None
    """
    stockfish_analysis = STOCKFISH.analyse(board, limit=chess.engine.Limit(time=0.01), multipv=2)
    if isinstance(stockfish_analysis, dict):
        stockfish_analysis = [stockfish_analysis]
        cp_diff = 2500
    else:
        cp_diff = stockfish_analysis[0]["score"].pov(board.turn).score(mate_score=2500) - stockfish_analysis[1]["score"].pov(board.turn).score(mate_score=2500)
    best_move_uci = stockfish_analysis[0]["pv"][0].uci()
    
    if is_takeback(last_move_uci, best_move_uci):
        if cp_diff > 70:
            return True, best_move_uci                  
        else:
            return False , None # there is a takeback, but not the best move
    return False, None

def new_attacked(prev_fen, fen, color):
    ''' Determines whether with the last move the opposition has introduced a new threat, i.e.
        placed our piece in en pris when it wasn't previously. '''
    before_board = chess.Board(prev_fen)
    before_map = {}
    for sq in before_board.piece_map():
        if before_board.color_at(sq) == color:
            before_map[sq] = is_en_pris(before_board, sq)[0]
    
    new_threatened = {}
    after_board = chess.Board(fen)
    for sq in after_board.piece_map():
        if after_board.color_at(sq) == color:
            new_state = is_en_pris(after_board, sq)[0]
            try:
                if new_state == True and before_map[sq] == False:
                    new_threatened[p_dic[after_board.piece_type_at(sq)]] = sq
            except KeyError:
                # the previously en pris piece has moved away
                continue
    if len(new_threatened) > 0:
        # return the square with the highest piece threatened
        return True, new_threatened[sorted(new_threatened)[-1]]
    else:
        return False, None

def get_lucas_analytics(board, analysis=None):
    ''' Given a chess position, give the following stats about the position:
        Complexity: how complex the position is. from 0 to 100
        Win probability (of turn's side). from 0 to 100
        Efficient mobility: how easy it is to play without making mistakes. from 0 to 50
        Narrowness: how open the position is. from 0 to 50
        Piece activity (of turn's side). from 0 to 25'''
    
    mat_dic = {1:1, 2:3.1, 3:3.5, 4:5.5, 5:9.9, 6:3}
    #calculate constants
    xpie = len(board.piece_map()) # number of pieces alive on the board
    xpiec = sum([len(board.pieces(x, board.turn)) for x in range(1,7)]) # number of pieces alive of board turn colour
    xmov = board.legal_moves.count() # number of legal moves for turn player
    xpow = sum([len(board.pieces(x, board.turn))*mat_dic[x] for x in range(1,7)]) # sum of material on turn side
    xmat = xpow + sum([len(board.pieces(x, not board.turn))*mat_dic[x] for x in range(1,7)]) # sum of all material on board
    
    if analysis is None:
        analysis = STOCKFISH.analyse(board, chess.engine.Limit(time=0.02), multipv=18)
    evals = [entry['score'].pov(board.turn).score(mate_score=2500) for entry in analysis]
    xeval = max(evals) # evaluation of position (in centipawns)
    xgmo = len([x for x in evals if x + 100 >= xeval]) # number of good moves in position
    
    # complexity
    xcompl = xgmo * xmov * xpie * xmat / (400 * xpow)
    # win prob
    xmlr = (math.tanh(xeval/(2*xmat)) + 1) * 50
    # efficient mobility
    xemo = 100*(xgmo - 1)/xmov
    # narrowness
    xnar = 10 * (xpie**2) * xmat / (4 * (xgmo**0.5)* (xmov**0.5)* xpiec * xpow)
    # piece activity
    xact = 40 * (xgmo**0.5)* (xmov**0.5) * xpow / (xpie * xmat)
    
    
    #clipping
    xcompl = min(xcompl, 100)
    xemo = min(xemo, 50)
    xnar = min(xnar, 50)
    xact = min(xact, 25)
    return xcompl, xmlr, xemo, xnar, xact

def is_en_pris(board, square):
    ''' Takes in a chess.Board() instance and works out whether the attacked 
        piece is en pris. Also gives a score on how much the piece is en pris by,
        namely a higher score means a greater material loss/gain. 
        
        Returns bool, threatened levels
    '''
    threatened_levels = calculate_threatened_levels(square, board)
    if threatened_levels > 0:
        return True, threatened_levels
    else:
        return False, 0

def phase_of_game(board):
    ''' Takes in a chess.Board() instance and returns opening, midgame, endgame
        depending on what phase of the game the board position is. '''
    # count minor and major pieces on the board
    min_maj_pieces = 0
    for square in chess.SQUARES:
        if board.piece_type_at(square) is not None: # square is occupied
            if board.piece_type_at(square) != chess.PAWN and board.piece_type_at(square) != chess.KING:
                min_maj_pieces += 1
    if min_maj_pieces < 6:
        return 'endgame'
    elif min_maj_pieces < 11:
        return 'midgame'
    else:
        # see if back rank is sparse
        white_br = 0
        black_br = 0
        for square in chess.SquareSet(chess.BB_RANK_1):
            if board.color_at(square) == chess.WHITE:
                white_br += 1
        if white_br < 5:
            return 'midgame'
        
        for square in chess.SquareSet(chess.BB_RANK_8):
            if board.color_at(square) == chess.BLACK:
                black_br += 1
        if black_br < 5:
            return 'midgame'
        
        # otherwise, it is the opening
        return 'opening'



if __name__ == '__main__':
    b = chess.Board('8/1ppk1p2/p2bP3/6Q1/7p/2qB4/P4PPP/4R1K1 b - - 0 26')
    # a = get_threatened_board(b, chess.WHITE)
    #print(is_attacked_by_pinned(b, chess.E7, chess.BLACK))
    print(calculate_threatened_levels(chess.D3, b))

