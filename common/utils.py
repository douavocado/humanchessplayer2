# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:11:33 2024

@author: xusem
"""
import chess
import time

def patch_fens(fen_before, fen_after, depth_lim:int=3):
    """ If get_move_made function is not able to find legal move to link the two fens
        we try to find in between fens to link the two fens.
        
        If no in between board are found, return None. Else return the fens and
        the moves made in between.
        
        Note when looking to patch fens with 3 or more plies missing, no longer becomes accurate
    """    
    moves_found = _recurse_patch_fens(fen_before, fen_after, depth_lim, [])
    if moves_found is not None:
        return_fens = [fen_before]
        dummy_board = chess.Board(fen_before)
        for move_uci in moves_found:
            dummy_board.push_uci(move_uci)
            return_fens.append(dummy_board.fen())
        return moves_found, return_fens
    else:
        return None
    
def _recurse_patch_fens(fen_before, fen_after, depth_lim, move_stack):
    before_b = chess.Board(fen_before)
    after_b = chess.Board(fen_after)
    if before_b.board_fen() == after_b.board_fen() and before_b.turn == after_b.turn: # terminating condition
        return move_stack
    elif depth_lim <= 0: # second terminating condition to make sure search doesn't go on forever
        return None
    else:
        changed_squares = chess.SquareSet(before_b.occupied ^ after_b.occupied)
        # find which squares were empty and which squares were occupied in the before board    
        if before_b.turn == chess.WHITE:
            moved_square_from = [sq for sq in changed_squares if before_b.color_at(sq) == chess.WHITE]
        else:
            moved_square_from = [sq for sq in changed_squares if before_b.color_at(sq) == chess.BLACK]
        
        test_moves = [move for move in before_b.legal_moves if move.from_square in moved_square_from]
        for move in test_moves:
            new_move_stack = move_stack[:] + [move.uci()]
            dummy_board = before_b.copy()
            dummy_board.push(move)
            res = _recurse_patch_fens(dummy_board.fen(), fen_after, depth_lim -1, new_move_stack)
            if res is not None:
                return res
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


if __name__ == "__main__":
    before = chess.Board("r1bqrnk1/pp3pb1/6pp/8/1P1N4/P5P1/1B1QP1BP/1R2R1K1 w - - 2 20")
    after = chess.Board("r1b1rnk1/pp3pb1/1q4pp/8/1P1N4/P5P1/1B1QP1BP/3R1RK1 w - - 6 22")
    start = time.time()
    print(patch_fens(before.fen(), after.fen(), depth_lim=4))
    end = time.time()
    print(end-start)