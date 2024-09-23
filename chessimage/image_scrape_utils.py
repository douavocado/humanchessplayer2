#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:52:37 2024

@author: james
"""

from fastgrab import screenshot
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import chess

def remove_background_colours(img, thresh = 1.04):

    res = img*np.expand_dims((np.abs(img[:,:,0]/(img[:,:,1]+10**(-10))-1) < thresh-1),-1)
    res = res*np.expand_dims((np.abs(img[:,:,0]/(img[:,:,2]+10**(-10))-1) < thresh-1),-1)
    res = res*np.expand_dims((np.abs(img[:,:,1]/(img[:,:,2]+10**(-10))-1) < thresh-1),-1)
    # res = res*np.expand_dims((img[:,:,1] < thresh*img[:,:,0]),-1)
    # res = res*np.expand_dims((img[:,:,2] < thresh*img[:,:,0]),-1)
    # res = res*np.expand_dims((img[:,:,2] < thresh*img[:,:,1]),-1)
    res = res.astype(np.uint8)
    
    # now turn image grey scale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res

SCREEN_CAPTURE = screenshot.Screenshot()

BOTTOM_CLOCK_X = 1406
BOTTOM_CLOCK_Y = 762
BOTTOM_CLOCK_Y_START = 790
BOTTOM_CLOCK_Y_END = 831
BOTTOM_CLOCK_Y_END_2 = 767
BOTTOM_CLOCK_Y_END_3 = 795
TOP_CLOCK_X = 1406
TOP_CLOCK_Y = 443
TOP_CLOCK_Y_START = 413
TOP_CLOCK_Y_END = 372
TOP_CLOCK_Y_END_2 = 440
CLOCK_WIDTH = 147
CLOCK_HEIGHT = 44
W_NOTATION_X = 1458
W_NOTATION_Y = 591
W_NOTATION_WIDTH, W_NOTATION_HEIGHT = 166, 104
START_X = 554
START_Y = 213
STEP = 103

w_rook = remove_background_colours(cv2.imread('chessimage/w_rook.png',)).astype(np.uint8)
w_knight= remove_background_colours(cv2.imread('chessimage/w_knight.png', )).astype(np.uint8)
w_bishop = remove_background_colours(cv2.imread('chessimage/w_bishop.png')).astype(np.uint8)
w_king= remove_background_colours(cv2.imread('chessimage/w_king.png',)).astype(np.uint8)
w_queen = remove_background_colours(cv2.imread('chessimage/w_queen.png', )).astype(np.uint8)
w_pawn= remove_background_colours(cv2.imread('chessimage/w_pawn.png', )).astype(np.uint8)

b_rook = remove_background_colours(cv2.imread('chessimage/b_rook.png',)).astype(np.uint8)
b_knight= remove_background_colours(cv2.imread('chessimage/b_knight.png',)).astype(np.uint8)
b_bishop = remove_background_colours(cv2.imread('chessimage/b_bishop.png',)).astype(np.uint8)
b_king= remove_background_colours(cv2.imread('chessimage/b_king.png')).astype(np.uint8)
b_queen = remove_background_colours(cv2.imread('chessimage/b_queen.png',)).astype(np.uint8)
b_pawn= remove_background_colours(cv2.imread('chessimage/b_pawn.png')).astype(np.uint8)

ALL_PIECES = {'R': w_rook, 'N': w_knight, 'B': w_bishop, 'K': w_king, 'Q': w_queen, 'P': w_pawn,
              'r': b_rook, 'n': b_knight, 'b': b_bishop, 'k': b_king, 'q': b_queen, 'p': b_pawn,}
PIECE_TEMPLATES = np.stack([w_rook, w_knight, w_bishop, w_king, w_queen, w_pawn, b_rook, b_knight, b_bishop, b_king, b_queen, b_pawn],axis=0)

INDEX_MAPPER = {0: "R", 1: "N", 2: "B", 3: "K", 4: "Q", 5: "P",
              6: "r", 7: "n", 8: "b", 9: "k", 10: "q", 11: "p",}

one = remove_background_colours(cv2.imread('chessimage/1.png'), thresh=1.6).astype(np.uint8)
two = remove_background_colours(cv2.imread('chessimage/2.png'),thresh=1.6).astype(np.uint8)
three = remove_background_colours(cv2.imread('chessimage/3.png'),thresh=1.6).astype(np.uint8)
four = remove_background_colours(cv2.imread('chessimage/4.png'),thresh=1.6).astype(np.uint8)
five = remove_background_colours(cv2.imread('chessimage/5.png'),thresh=1.6).astype(np.uint8)
six = remove_background_colours(cv2.imread('chessimage/6.png'),thresh=1.6).astype(np.uint8)
seven = remove_background_colours(cv2.imread('chessimage/7.png'),thresh=1.6).astype(np.uint8)
eight = remove_background_colours(cv2.imread('chessimage/8.png'),thresh=1.6).astype(np.uint8)
nine = remove_background_colours(cv2.imread('chessimage/9.png'),thresh=1.6).astype(np.uint8)
zero = remove_background_colours(cv2.imread('chessimage/0.png'),thresh=1.6).astype(np.uint8)

ALL_NUMBERS = {1: one, 2: two, 3: three, 4: four, 5: five, 6: six,
              7: seven, 8: eight, 9: nine, 0: zero}

TEMPLATES = np.stack([zero,one,two,three,four,five,six,seven,eight,nine], axis=0)

def multitemplate_multimatch(imgs, templates):
    # assumes both imgs and templates are 3 dimensional NxWxH and MxWxH
    T = templates.astype(float) # M templates
    I = imgs.astype(float) # N images
    w, h = imgs.shape[-2:]

    T_primes = np.expand_dims(T- np.expand_dims(1/(w*h)*T.sum(axis=(1,2)), (-1,-2)),1) # so is Mx1xWxH
    I_primes = np.expand_dims(I - np.expand_dims(1/(w*h)*I.sum(axis=(1,2)), (-1,-2)), 0) # so is 1xNxWxH

    T_denoms = (T_primes**2).sum(axis=(-1,-2)) # so is Mx1 shape
    I_denoms = (I_primes**2).sum(axis=(-1,-2)) # so is 1xN shape
    denoms = np.sqrt(T_denoms*I_denoms) + 10**(-10) # MxN shape
    nums = (T_primes*I_primes).sum(axis=(-1,-2)) # MxN shape
    scores = nums/denoms

    # for ever square give its argmax prob over threshold
    threshold = 0.7
    arg_maxes = scores.argmax(axis=0) # shape N
    maxes = scores.max(axis=0)
    valid_squares = np.where(maxes > threshold)[0]
    return valid_squares, arg_maxes
    

def multitemplate_match_f(img, templates):
    # assumes img is 2 dimensional WxH and templates is 3 dimensional i.e. NxWxH where N is the number of templates
    # assumes that img and template are the same shape
    T = templates.astype(float)
    I = img.astype(float)
    w, h = img.shape
    T_primes = T- np.expand_dims(1/(w*h)*T.sum(axis=(1,2)), (-1,-2))
    I_prime = I - np.expand_dims(1/(w*h)*I.sum(), (-1))
    
    T_denom = (T_primes**2).sum(axis=(1,2))
    I_denom = (I_prime**2).sum()
    denoms = np.sqrt(T_denom*I_denom) + 10**(-10)
    nums = (T_primes*np.expand_dims(I_prime,0)).sum(axis=(1,2))
    
    scores =  nums/denoms
    # if scores are all low, return None
    if scores.max() < 0.5:
        return None
    return scores.argmax()

def template_match_f(img, template):
    # uses cv2.TM_CCOEFF_NORMED from https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html
    # assumes img and template are of same shape 
    # assumes img is 3 dimensional NxWxH and template is WxH, where N is the number of images
    
    T = template.astype(float)
    I = img.astype(float)
    w, h = template.shape
    T_prime = T- np.expand_dims(1/(w*h)*T.sum(), (-1,-2))
    I_prime = I - np.expand_dims(1/(w*h)*I.sum(axis=(1,2)), (-1,-2))
    
    T_denom = (T_prime**2).sum()
    I_denom = (I_prime**2).sum(axis=(1,2))
    denom = np.sqrt(T_denom*I_denom) + 10**(-10)
    num = (T_prime*I_prime).sum(axis=(1,2))
    
    return num/denom

def read_clock(clock_image):
    # assumes image is black and white
    if clock_image.ndim== 3:
        image = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
    else:
        image = clock_image.copy()
        
    d1 = image[:, :30]
    d2 = image[:, 34:64]
    d3 = image[:, 83:113]
    d4 = image[:, 117:147]

    digit_1 = multitemplate_match_f(d1, TEMPLATES)
    digit_2 = multitemplate_match_f(d2, TEMPLATES)
    digit_3 = multitemplate_match_f(d3, TEMPLATES)
    digit_4 = multitemplate_match_f(d4, TEMPLATES)
    if digit_1 is not None and digit_2 is not None and digit_3 is not None and digit_4 is not None:
        return digit_1 * 600 + digit_2*60 + digit_3*10 + digit_4
    else:
        # error
        return None

def detect_last_move_from_img(board_img):
    epsilon = 5
    detected = []
    for square in range(64):
        column_i = square%8
        row_i = square // 8
        pixel_x = int(STEP*column_i + epsilon)
        pixel_y = int(STEP*row_i + epsilon)
        rgb = board_img[pixel_y, pixel_x, :]
        if (rgb == [143,155,59]).all() or (rgb == [205, 211, 145]).all() or (rgb == [95, 92, 60]).all() or (rgb == [147, 140, 133]).all():
            detected.append(square)
    return detected

def capture_board(shift=False):
    if shift:
        im = SCREEN_CAPTURE.capture((int(START_X-7),int(START_Y), int(8*STEP), int(8*STEP))).copy()
    else:
        im = SCREEN_CAPTURE.capture((int(START_X),int(START_Y), int(8*STEP), int(8*STEP))).copy()
    img= im[:,:,:3]
    return img

def capture_bottom_clock(state="play"):
    # TODO: configurable
    if state == "play":
        im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    elif state == "start":
        im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_START, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    elif state == "end1":
        im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_END, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    elif state == "end2":
        im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_END_2, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    elif state == "end3":
        im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_END_3, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    img= im[:,:,:3]
    return img

def capture_top_clock(state="play"):
    # TODO: configurable
    if state == "play":
        im = SCREEN_CAPTURE.capture((TOP_CLOCK_X,TOP_CLOCK_Y, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    elif state == "start":
        im = SCREEN_CAPTURE.capture((TOP_CLOCK_X,TOP_CLOCK_Y_START, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    elif state == "end1":
        im = SCREEN_CAPTURE.capture((TOP_CLOCK_X,TOP_CLOCK_Y_END, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    elif state == "end2":
        im = SCREEN_CAPTURE.capture((TOP_CLOCK_X,TOP_CLOCK_Y_END_2, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    img= im[:,:,:3]
    return img

def capture_white_notation():
    im = SCREEN_CAPTURE.capture((W_NOTATION_X,W_NOTATION_Y, W_NOTATION_WIDTH, W_NOTATION_HEIGHT)).copy()
    img= im[:,:,:3]
    return img

def is_our_turn_from_clock(bottom_clock_img):
    colours = bottom_clock_img.reshape(-1,3)
    has_green = (colours[:,1] > 1.1*colours[:,0]).any()
    return has_green

def is_white_turn_from_notation(white_notation_img):    
    colours = white_notation_img.reshape(-1,3)
    has_red = (colours[:,0] > 1.1*colours[:,2]).any()
    return not has_red

def check_turn_from_last_moved(fen, board_img, bottom):
    detected_moved = detect_last_move_from_img(board_img)
    if len(detected_moved) == 0:
        return chess.Board(fen).turn == chess.WHITE # didn't detect any new moves. Can only assume it's white turn from opening position
    
    board = chess.Board(fen)
    test_turn = board.turn
    colour_count = 0
    for square in detected_moved:
        if bottom == "w":
            colour = board.color_at(chess.square_mirror(square))
        else:
            colour = board.color_at(chess.square_mirror(63-square))
        if colour is not None:
            colour_count += 2*((colour == test_turn)-0.5)
    if colour_count == 0:
        if len(detected_moved) > 0:
            # then it must have been a castling move
            if bottom == "w":
                real_square = chess.square_mirror(detected_moved[0])
            else:
                real_square = chess.square_mirror(63-detected_moved[0])
            if chess.square_rank(real_square) == 0: # first rank, must be white castles, so black move
                return test_turn == chess.BLACK
            elif chess.square_rank(real_square) == 7: # 8th rank, black castles, white to move
                return test_turn == chess.WHITE
            else:
                # there was an error, expected to be castle move but wasn't
                return None
        # no detected moves, there was error, return None
        return None
    elif colour_count < 0:
        return True # then current turn is correct
    else:
        return False # current turn is incorrect

def check_fen_last_move_bottom(fen, board_img, proposed_bottom):
    detected_moved = detect_last_move_from_img(board_img)
    if len(detected_moved) == 0:
        print("Didn't detect any new moves, returning True")
        return True
    test_board = chess.Board(fen)
    test_turn = test_board.turn # the turn we are testing whether true or not
    colour_count = 0
    for square in detected_moved:
        if proposed_bottom == "w":
            colour = test_board.color_at(chess.square_mirror(square))
        else:
            colour = test_board.color_at(chess.square_mirror(63-square))
        if colour is not None:
            colour_count += 2*((colour == test_turn)-0.5)
    if colour_count == 0:
        # there was error, return False
        return False
    elif colour_count < 0:
        # should happen
        return True
    else:
        return False

def find_initial_side():
    # check bottom left square for a white rook
    a1_img = SCREEN_CAPTURE.capture((int(START_X),int(START_Y + 7*STEP), 100, 100))
    a1_img = np.expand_dims(remove_background_colours(a1_img[:,:,:3]),0).astype(np.uint8)
    template = w_rook
    return (template_match_f(a1_img, template) > 0.7).item()

def get_fen_from_image(board_image, bottom:str='w', turn:bool=None):
    # Note this automatically sets castling rights to None, i.e. no castling
    # castling rights needs to be dealt with separately
    # if image is not black and white, process first:
    if board_image.ndim == 3:
        image = remove_background_colours(board_image).astype(np.uint8)
    else:
        image = board_image.copy()
        
    board_width, board_height = image.shape[:2]
    
    images = [image[x*103:x*103+100, y*103:y*103+100] for x in range(8) for y in range(8)]
    images = np.stack(images, axis=0)

    valid_squares, argmaxes = multitemplate_multimatch(images, PIECE_TEMPLATES)
    
    board = chess.Board(fen=None)
    for square in valid_squares:
        key = INDEX_MAPPER[argmaxes[square]]
        if bottom == "w":
            output_square = chess.square_mirror(square)
        else:
            output_square = chess.square_mirror(63-square)
        board.set_piece_at(output_square, chess.Piece.from_symbol(key))
    
    if turn is not None:
        board.turn = turn
    return board.fen()

    

if __name__ == "__main__":
    time.sleep(5)
    
    start = time.time()
    
    board_img = capture_board()
    top_clock_img = capture_top_clock()
    bot_clock_img = capture_bottom_clock()
    w_notation_img = capture_white_notation()

    our_turn = is_our_turn_from_clock(bot_clock_img)
    white_turn = is_white_turn_from_notation(w_notation_img)

    if our_turn == white_turn:
        bottom = "w"
    else:
        bottom = "b"

    our_time = read_clock(bot_clock_img)
    opp_time = read_clock(top_clock_img)
    
    fen = get_fen_from_image(board_img, bottom=bottom, turn=white_turn)    
    end = time.time()
    print(fen)
    print(chess.Board(fen))
    print(end-start)
    
    plt.imshow(board_img)