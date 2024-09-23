# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:11:33 2024

@author: xusem
"""
import chess

def patch_fens(fen_before, fen_after):
    """ If get_move_made function is not able to find legal move to link the two fens
        we try to find in between fens to link the two fens.
        
        If no in between board are found, return None. Else return the fens and
        the moves made in between.
    """
    before_b = chess.Board(fen_before)
    after_b = chess.Board(fen_after)
    
    # first take note of whose turn it is, this tells us how many moves we want to simulate
    if before_b.turn == after_b.turn: # then we need to simulate one
        # first find which pieces have been moved
        changed_squares = chess.SquareSet(before_b.occupied ^ after_b.occupied)
        if len(changed_squares) == 0:
            return None
        # find which squares were empty and which squares were occupied in the before board
        white_moved_square_from = [sq for sq in changed_squares if before_b.color_at(sq) == chess.WHITE]
        black_moved_square_from = [sq for sq in changed_squares if before_b.color_at(sq) == chess.BLACK]
        
        white_moved_sq_to = [sq for sq in changed_squares if after_b.color_at(sq) == chess.WHITE] # these may be empty lists
        black_moved_sq_to = [sq for sq in changed_squares if after_b.color_at(sq) == chess.BLACK] # these may be empty lists if captures were involved
        
        if before_b.turn == chess.WHITE: # white to move first
            # if there's more than one moved from square of each color, then we know more than 2 moves have
            # been made (unless castling). this is beyond the scope of this function, so we return None
            if len(white_moved_square_from) > 1:
                # try castling
                white_possible_moves = ["e1g1", "e1c1"]
            else:
                if len(white_moved_sq_to) > 0:
                    white_possible_moves = [chess.Move(white_moved_square_from[0], sq).uci() for sq in white_moved_sq_to]
                else:
                    # look for captures
                    white_possible_moves = [move.uci() for move in before_b.legal_moves if move.from_square == white_moved_square_from[0]]
            
            white_possible_moves = [move_uci for move_uci in white_possible_moves if chess.Move.from_uci(move_uci) in before_b.legal_moves]
            if len(white_possible_moves) == 0:
                return None
            for move_uci in white_possible_moves:
                dummy_board = before_b.copy()
                dummy_board.push_uci(move_uci)
                next_move_made = get_move_made(dummy_board.fen(), fen_after)
                if next_move_made is not None: # successful
                    last_moves = [move_uci, next_move_made]
                    # correct move numbers
                    fen_middle = dummy_board.fen()
                    dummy_board.push_uci(next_move_made)
                    fen_after = dummy_board.fen()
                    fens = [fen_before, fen_middle, fen_after]
                    return last_moves, fens
            return None
        elif before_b.turn == chess.BLACK: # black to move first
            if len(black_moved_square_from) > 1:
                # try castling
                black_possible_moves = ["e8g8", "e8c8"]
            else:
                if len(black_moved_sq_to) > 0:
                    black_possible_moves = [chess.Move(black_moved_square_from[0], sq).uci() for sq in black_moved_sq_to]
                else:
                    # look for captures
                    black_possible_moves = [move.uci() for move in before_b.legal_moves if move.from_square == black_moved_square_from[0]]
            
            black_possible_moves = [move_uci for move_uci in black_possible_moves if chess.Move.from_uci(move_uci) in before_b.legal_moves]
            if len(black_possible_moves) == 0:
                return None
            for move_uci in black_possible_moves:
                dummy_board = before_b.copy()
                dummy_board.push_uci(move_uci)
                next_move_made = get_move_made(dummy_board.fen(), fen_after)
                if next_move_made is not None: # successful
                    last_moves = [move_uci, next_move_made]
                    # correct move numbers
                    fen_middle = dummy_board.fen()
                    dummy_board.push_uci(next_move_made)
                    fen_after = dummy_board.fen()
                    fens = [fen_before, fen_middle, fen_after]
                    return last_moves, fens
            return None
    else:
        # if there's more than one turn to simulate, skip for now
        return None

def flip_uci(move_uci):
    """ Given a move uci, return a uci which is the move flipped. For example
        g2g3 is flipped to g7g6. """
    move_obj = chess.Move.from_uci(move_uci)
    from_sq = move_obj.from_square
    to_sq = move_obj.from_uci(move_uci).to_square
    promotion  = move_obj.promotion
    flipped_move_obj = chess.Move(chess.square_mirror(from_sq), chess.square_mirror(to_sq), promotion=promotion)
    return flipped_move_obj.uci()

def get_move_made(fen_before, fen_after):
    """ Given two fens (before and after), find the move made. This ignores the move
        number information provided by fens. If we cannot link the board position between
        the two fens with a single move, then raise and error. 
        
        Returns move_uci of move made
    """
    
    before_b = chess.Board(fen_before)
    after_b = chess.Board(fen_after)
    changed_squares = chess.SquareSet(before_b.occupied ^ after_b.occupied)
    # find which squares were empty and which squares were occupied in the before board    
    if before_b.turn == chess.WHITE:
        moved_square_from = [sq for sq in changed_squares if before_b.color_at(sq) == chess.WHITE]
    else:
        moved_square_from = [sq for sq in changed_squares if before_b.color_at(sq) == chess.BLACK]
    
    test_moves = [move for move in before_b.legal_moves if move.from_square in moved_square_from]
    for move in test_moves:
        dummy_board = before_b.copy()
        dummy_board.push(move)
        if dummy_board.board_fen() == after_b.board_fen():
            return move.uci()
    
    # else, the fens cannot be conjoined
    return None

if __name__ == "__main__":
    before = chess.Board("r4rk1/pp3pbp/2bp1np1/q3p3/2P1PP2/2N1BB2/PP4PP/2RQ1RK1 b - - 1 14")
    after = chess.Board("r4rk1/pp3pbp/2bp1np1/q7/2P1PB2/2N2B2/PP4PP/2RQ1RK1 b - - 0 15")
    print(patch_fens(before.fen(), after.fen()))