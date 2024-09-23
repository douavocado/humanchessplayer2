# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:57:24 2023

@author: xusem
"""

import cv2
import numpy as np

w_rook = cv2.imread('w_rook.png', cv2.IMREAD_GRAYSCALE)
w_knight = cv2.imread('w_knight.png', cv2.IMREAD_GRAYSCALE)
w_bishop = cv2.imread('w_bishop.png', cv2.IMREAD_GRAYSCALE)
w_king = cv2.imread('w_king.png', cv2.IMREAD_GRAYSCALE)
w_queen = cv2.imread('w_queen.png', cv2.IMREAD_GRAYSCALE)
w_pawn = cv2.imread('w_pawn.png', cv2.IMREAD_GRAYSCALE)

b_rook = cv2.imread('b_rook.png', cv2.IMREAD_GRAYSCALE)
b_knight = cv2.imread('b_knight.png', cv2.IMREAD_GRAYSCALE)
b_bishop = cv2.imread('b_bishop.png', cv2.IMREAD_GRAYSCALE)
b_king = cv2.imread('b_king.png', cv2.IMREAD_GRAYSCALE)
b_queen = cv2.imread('b_queen.png', cv2.IMREAD_GRAYSCALE)
b_pawn = cv2.imread('b_pawn.png', cv2.IMREAD_GRAYSCALE)

ALL_PIECES = {'R': w_rook, 'N': w_knight, 'B': w_bishop, 'K': w_king, 'Q': w_queen, 'P': w_pawn,
              'r': b_rook, 'n': b_knight, 'b': b_bishop, 'k': b_king, 'q': b_queen, 'p': b_pawn,}


def get_move_change(image, bottom='w'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    board_width, board_height = image.shape[:2]
    tile_width = board_width/8
    tile_height = board_height/8
    epsilon = 5
    if bottom == 'w':
        row_dic = {0:'8',1:'7',2:'6',3:'5',4:'4',5:'3',6:'2',7:'1'}
        column_dic = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h'}
    else:
        row_dic = {0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'7',7:'8'}
        column_dic = {0:'h',1:'g',2:'f',3:'e',4:'d',5:'c',6:'b',7:'a'}
    
    detected = []
    
    for i in range(64):
        column_i = i%8
        row_i = i // 8
        pixel_x = int(tile_width*column_i + epsilon)
        pixel_y = int(tile_height*row_i + epsilon)
        rgb = image[pixel_y, pixel_x, :]
        if (rgb == [59,155,143]).all():
            detected.append(column_dic[column_i]+row_dic[row_i])
    if len(detected) == 0:
        print("Did not detect any move changes, returning None")
        return None
    elif len(detected) != 2:
        raise Exception("Unexpectedly found {} detected change squares: {}".format(len(detected), detected))
    else:
        return [detected[0]+detected[1], detected[1] + detected[0]]


def get_fen_from_image(image, bottom='w'):    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    board_width, board_height = img_gray.shape[::-1]
    tile_width = board_width/8
    tile_height = board_height/8
    
    
    board = ['']*64
    for key, piece in ALL_PIECES.items():
        res = cv2.matchTemplate(img_gray,piece,cv2.TM_CCOEFF_NORMED)
        threshold = 0.65
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            column_i = (pt[0]+tile_width/2)//tile_width
            row_i = (pt[1]+tile_height/2)//tile_height
            if bottom == 'w':
                index = int(row_i*8 + column_i)
            else:
                index = 63 - int(row_i*8 + column_i)
            board[index] = key
            
    fen = ''
    counter = 0
    for i, item in enumerate(board):
        
        if i%8 == 0 and  i != 0:
            if counter != 0:
                fen += str(counter)
                counter = 0
            fen += '/'
        
        if len(item) != 0 and counter != 0:
            fen+= str(counter)
            fen += item
            counter = 0
        elif len(item) != 0:
            fen += item
        elif len(item) == 0:
            counter += 1
    if counter != 0:
        fen += str(counter)
    
    return fen