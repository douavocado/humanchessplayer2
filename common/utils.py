# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:11:33 2024

@author: xusem
"""
import chess

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
    for move in before_b.legal_moves:
        dummy_board = before_b.copy()
        dummy_board.push(move)
        if dummy_board.board_fen() == after_b.board_fen():
            return move.uci()
    
    # else, the fens cannot be conjoined
    return None