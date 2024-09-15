# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:58:32 2024

@author: xusem
"""

import chess
import chess.pgn

from .board_information import calculate_threatened_levels, get_attackers_en_pris, get_potential_threatened, get_new_threatened, get_new_hanging

'''
each position is a 768-element list of numbers(each element can be from 1-12 for each of the pieces * 64 elements) indicating what piece is in the square
blank square:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'p':           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'n':           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'b':           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'r':           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
'q':           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
'k':           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
'P':           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
'N':           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
'B':           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
'R':           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
'Q':           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
'K':           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
'''
def position_list_one_hot(self):
    '''method added to the python-chess library for faster
    conversion of board to one hot encoding. Resulted in 100%
    increase in speed by bypassing conversion to fen() first.
    We include extra value concerning the threatened levels of the square.
    13th entry: threatened levels of squares
    14th entry: whether moving that piece will result to taking en pris opp piece
    
    WE MUST ENSURE THAT WHITE IS THE TURN TO MOVE. Otherwise we mirror the board
    '''
    assert self.turn == chess.WHITE
    en_pris_dic = get_attackers_en_pris(self)
    builder = []
    for square in chess.SQUARES_180:
        mask = chess.BB_SQUARES[square]

        if not self.occupied & mask:
            builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            if bool(self.occupied_co[chess.WHITE] & mask):            
                if self.pawns & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
                elif self.knights & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
                elif self.bishops & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                elif self.rooks & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                elif self.queens & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                elif self.kings & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
            elif self.pawns & mask:
                builder.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.knights & mask:
                builder.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.bishops & mask:
                builder.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.rooks & mask:
                builder.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.queens & mask:
                builder.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif self.kings & mask:
                builder.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            #print(self, chess.square_name(square), self.fen())
            if self.piece_at(square).color == self.turn:
                levels = calculate_threatened_levels(square, self)
            else:
                levels = 0
            builder.append(levels)
            
            if square in en_pris_dic:
                builder.append(en_pris_dic[square])
            else:
                builder.append(0)
    return builder

def moveto_position_list_one_hot(self, move_from_sq):
    '''method added to the python-chess library for faster
    conversion of board to one hot encoding. Resulted in 100%
    increase in speed by bypassing conversion to fen() first.
    We include extra value concerning the threatened levels of the square.
    13th entry: legal moves
    14th entry: en_pris levels of the square if new piece moves there
    15th entry: whether moving that piece to that square attacks and (new) en pris piece
    16th entry: whether moving there places an own piece en pris.
    
    WE MUST ENSURE THAT WHITE IS THE TURN TO MOVE. Otherwise we mirror the board
    '''
    assert self.turn == chess.WHITE
    # For extra dimensions
    # 13th entry
    assert self.turn == self.piece_at(move_from_sq).color
    legal_movetos = [move.to_square for move in self.legal_moves if move.from_square == move_from_sq]
    # 14th entry
    moveto_en_pris_dic = get_potential_threatened(self, move_from_sq)
    # 15th entry
    new_threat_dic = get_new_threatened(self, move_from_sq)
    # 16th entry
    new_hanging_dic = get_new_hanging(self, move_from_sq)
    
    builder = []
    for square in chess.SQUARES_180:
        mask = chess.BB_SQUARES[square]

        if not self.occupied & mask:
            builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            if bool(self.occupied_co[chess.WHITE] & mask):            
                if self.pawns & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
                elif self.knights & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
                elif self.bishops & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                elif self.rooks & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                elif self.queens & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                elif self.kings & mask:
                    builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
            elif self.pawns & mask:
                builder.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.knights & mask:
                builder.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.bishops & mask:
                builder.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.rooks & mask:
                builder.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.queens & mask:
                builder.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif self.kings & mask:
                builder.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            #print(self, chess.square_name(square), self.fen())
        #13th entry, legal moves
        if square in legal_movetos:
            builder.append(1)
        else:
            builder.append(0)
        
        #14th entry
        if square in moveto_en_pris_dic:
            builder.append(moveto_en_pris_dic[square])
        else:
            builder.append(0)
        
        #15th entry
        if square in new_threat_dic:
            builder.append(new_threat_dic[square])
        else:
            builder.append(0)
            
        #16th entry
        if square in new_hanging_dic:
            builder.append(new_hanging_dic[square])
        else:
            builder.append(0)
        
    return builder

def position_list(self):
    '''same as position_list_one_hot except this is converts pieces to
    numbers between 1 and 12. Used for piece_moved function'''
    builder = []
    builder_append = builder.append
    for square in chess.SQUARES_180:
        mask = chess.BB_SQUARES[square]

        if not self.occupied & mask:
            builder_append(0)
        elif bool(self.occupied_co[chess.WHITE] & mask):
            if self.pawns & mask:
                builder_append(7)
            elif self.knights & mask:
                builder_append(8)
            elif self.bishops & mask:
                builder_append(9)
            elif self.rooks & mask:
                builder_append(10)
            elif self.queens & mask:
                builder_append(11)
            elif self.kings & mask:
                builder_append(12)
        elif self.pawns & mask:
            builder_append(1)
        elif self.knights & mask:
            builder_append(2)
        elif self.bishops & mask:
            builder_append(3)
        elif self.rooks & mask:
            builder_append(4)
        elif self.queens & mask:
            builder_append(5)
        elif self.kings & mask:
            builder_append(6)

    return builder