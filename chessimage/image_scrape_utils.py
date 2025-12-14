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
import pytesseract
import os
import sys
from datetime import datetime
from pathlib import Path

# Add auto_calibration to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from auto_calibration.config import get_config
    chess_config = get_config()
    USE_CONFIG = True
    if not chess_config.is_using_fallback():
        print("📍 Using auto-calibrated coordinates")
    else:
        print("⚠️  No calibration file found, using fallback coordinates")
except ImportError:
    USE_CONFIG = False
    chess_config = None
    print("⚠️  Auto-calibration not available, using hardcoded coordinates")

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

# Dynamic coordinate loading functions
def get_coordinates():
    """Get coordinates from config or fallback to hardcoded values."""
    if USE_CONFIG:
        return chess_config.get_coordinates()
    else:
        # Fallback hardcoded coordinates
        return {
            'board': {'x': 543, 'y': 179, 'width': 848, 'height': 848},
            'bottom_clock': {
                'play': {'x': 1420, 'y': 742, 'width': 147, 'height': 44},
                'start1': {'x': 1420, 'y': 756, 'width': 147, 'height': 44},
                'start2': {'x': 1420, 'y': 770, 'width': 147, 'height': 44},
                'end1': {'x': 1420, 'y': 811, 'width': 147, 'height': 44},
                'end2': {'x': 1420, 'y': 747, 'width': 147, 'height': 44},
                'end3': {'x': 1420, 'y': 776, 'width': 147, 'height': 44}
            },
            'top_clock': {
                'play': {'x': 1420, 'y': 424, 'width': 147, 'height': 44},
                'start1': {'x': 1420, 'y': 396, 'width': 147, 'height': 44},
                'start2': {'x': 1420, 'y': 410, 'width': 147, 'height': 44},
                'end1': {'x': 1420, 'y': 355, 'width': 147, 'height': 44},
                'end2': {'x': 1420, 'y': 420, 'width': 147, 'height': 44}
            },
            'notation': {'x': 1458, 'y': 591, 'width': 166, 'height': 104},
            'rating': {
                'opp_white': {'x': 1755, 'y': 458, 'width': 40, 'height': 24},
                'own_white': {'x': 1755, 'y': 706, 'width': 40, 'height': 24},
                'opp_black': {'x': 1755, 'y': 473, 'width': 40, 'height': 24},
                'own_black': {'x': 1755, 'y': 691, 'width': 40, 'height': 24}
            }
        }

def get_board_info():
    """Get board position and step size."""
    coords = get_coordinates()
    board = coords['board']
    step = board['width'] // 8  # Chess board is 8x8
    return board['x'], board['y'], step

def get_clock_info(clock_type, state="play"):
    """Get clock position info."""
    coords = get_coordinates()
    clock = coords[clock_type][state]
    return clock['x'], clock['y'], clock['width'], clock['height']

# Load coordinates into legacy variable names for backward compatibility
try:
    coords = get_coordinates()
    
    # Board coordinates
    START_X, START_Y, STEP = get_board_info()
    PIECE_STEP = STEP
    
    # Clock coordinates (using 'play' state as default)
    BOTTOM_CLOCK_X, BOTTOM_CLOCK_Y, CLOCK_WIDTH, CLOCK_HEIGHT = get_clock_info('bottom_clock', 'play')
    TOP_CLOCK_X, TOP_CLOCK_Y, _, _ = get_clock_info('top_clock', 'play')
    
    # State-specific Y coordinates for backward compatibility
    _, BOTTOM_CLOCK_Y_START, _, _ = get_clock_info('bottom_clock', 'start1')
    _, BOTTOM_CLOCK_Y_START_2, _, _ = get_clock_info('bottom_clock', 'start2')
    _, BOTTOM_CLOCK_Y_END, _, _ = get_clock_info('bottom_clock', 'end1')
    _, BOTTOM_CLOCK_Y_END_2, _, _ = get_clock_info('bottom_clock', 'end2')
    _, BOTTOM_CLOCK_Y_END_3, _, _ = get_clock_info('bottom_clock', 'end3')
    
    _, TOP_CLOCK_Y_START, _, _ = get_clock_info('top_clock', 'start1')
    _, TOP_CLOCK_Y_START_2, _, _ = get_clock_info('top_clock', 'start2')
    _, TOP_CLOCK_Y_END, _, _ = get_clock_info('top_clock', 'end1')
    _, TOP_CLOCK_Y_END_2, _, _ = get_clock_info('top_clock', 'end2')
    
    # Notation coordinates
    notation = coords['notation']
    W_NOTATION_X, W_NOTATION_Y = notation['x'], notation['y']
    W_NOTATION_WIDTH, W_NOTATION_HEIGHT = notation['width'], notation['height']
    
    # Rating coordinates
    rating = coords['rating']
    RATING_X = rating['opp_white']['x']
    RATING_WIDTH = rating['opp_white']['width']
    RATING_HEIGHT = rating['opp_white']['height']
    OPP_RATING_Y_WHITE = rating['opp_white']['y']
    OWN_RATING_Y_WHITE = rating['own_white']['y']
    OPP_RATING_Y_BLACK = rating['opp_black']['y']
    OWN_RATING_Y_BLACK = rating['own_black']['y']
    
except Exception as e:
    print(f"Warning: Error loading coordinates: {e}")
    print("Using fallback hardcoded coordinates")
    # Fallback values
    START_X, START_Y, STEP, PIECE_STEP = 543, 179, 106, 106
    BOTTOM_CLOCK_X, BOTTOM_CLOCK_Y = 1420, 742
    BOTTOM_CLOCK_Y_START, BOTTOM_CLOCK_Y_START_2 = 756, 770
    BOTTOM_CLOCK_Y_END, BOTTOM_CLOCK_Y_END_2, BOTTOM_CLOCK_Y_END_3 = 811, 747, 776
    TOP_CLOCK_X, TOP_CLOCK_Y = 1420, 424
    TOP_CLOCK_Y_START, TOP_CLOCK_Y_START_2 = 396, 410
    TOP_CLOCK_Y_END, TOP_CLOCK_Y_END_2 = 355, 420
    CLOCK_WIDTH, CLOCK_HEIGHT = 147, 44
    W_NOTATION_X, W_NOTATION_Y = 1458, 591
    W_NOTATION_WIDTH, W_NOTATION_HEIGHT = 166, 104
    RATING_X, RATING_WIDTH, RATING_HEIGHT = 1755, 40, 24
    OPP_RATING_Y_WHITE, OWN_RATING_Y_WHITE = 458, 706
    OPP_RATING_Y_BLACK, OWN_RATING_Y_BLACK = 473, 691

w_rook = remove_background_colours(cv2.resize(cv2.imread('chessimage/w_rook.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
w_knight= remove_background_colours(cv2.resize(cv2.imread('chessimage/w_knight.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
w_bishop = remove_background_colours(cv2.resize(cv2.imread('chessimage/w_bishop.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
w_king= remove_background_colours(cv2.resize(cv2.imread('chessimage/w_king.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
w_queen = remove_background_colours(cv2.resize(cv2.imread('chessimage/w_queen.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
w_pawn= remove_background_colours(cv2.resize(cv2.imread('chessimage/w_pawn.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)

b_rook = remove_background_colours(cv2.resize(cv2.imread('chessimage/b_rook.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
b_knight= remove_background_colours(cv2.resize(cv2.imread('chessimage/b_knight.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
b_bishop = remove_background_colours(cv2.resize(cv2.imread('chessimage/b_bishop.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
b_king= remove_background_colours(cv2.resize(cv2.imread('chessimage/b_king.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
b_queen = remove_background_colours(cv2.resize(cv2.imread('chessimage/b_queen.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)
b_pawn= remove_background_colours(cv2.resize(cv2.imread('chessimage/b_pawn.png'), ( PIECE_STEP, PIECE_STEP ), interpolation = cv2.INTER_CUBIC )).astype(np.uint8)

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
    threshold = 0.5
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

def compare_result_images(img1, img2, max_shift=3, threshold=0.95):
    """
    Compare two result images and return a confidence score of their similarity,
    accounting for slight shifts in position.
    
    Args:
        img1: First image (output from capture_result)
        img2: Second image (output from capture_result)
        max_shift: Maximum pixel shift to check in each direction
        threshold: Score threshold above which to stop searching (0 to 1)
        
    Returns:
        float: Confidence score between 0 and 1, where 1 indicates perfect match
    """
    # Convert to grayscale for better comparison
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()
        
    if img2.ndim == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2.copy()
    
    # Get image dimensions
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    
    # Ensure images have same dimensions
    min_h = min(h1, h2)
    min_w = min(w1, w2)
    gray1 = gray1[:min_h, :min_w]
    gray2 = gray2[:min_h, :min_w]
    
    best_score = 0
    
    # Try zero shift first
    result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    best_score = float(result[0][0])
    
    # If we already exceed threshold, return early
    if best_score >= threshold:
        return best_score
    
    # Try various small shifts to account for slight perturbations
    shifts = [(y, x) for y in range(-max_shift, max_shift + 1) 
              for x in range(-max_shift, max_shift + 1)
              if not (y == 0 and x == 0)]  # Skip (0,0) as we already checked it
    
    for y_shift, x_shift in shifts:
        # Apply shift to second image
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shifted = cv2.warpAffine(gray2, M, (min_w, min_h))
        
        # Calculate normalized cross-correlation coefficient
        result = cv2.matchTemplate(gray1, shifted, cv2.TM_CCOEFF_NORMED)
        score = float(result[0][0])
        
        # Update best score
        if score > best_score:
            best_score = score
            
            # If we exceed threshold, return early
            if best_score >= threshold:
                return best_score
    
    return best_score

def read_clock(clock_image):
    # assumes image is black and white
    if clock_image.ndim== 3:
        image = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
    else:
        image = clock_image.copy()
    
    # Get template dimensions for resizing
    template_h, template_w = TEMPLATES.shape[1:3]  # (height, width) from shape (10, H, W)
    
    d1 = image[:, :30]
    d2 = image[:, 34:64]
    d3 = image[:, 83:113]
    d4 = image[:, 117:147]
    
    # Resize each digit region to match template size
    d1 = cv2.resize(d1, (template_w, template_h), interpolation=cv2.INTER_AREA)
    d2 = cv2.resize(d2, (template_w, template_h), interpolation=cv2.INTER_AREA)
    d3 = cv2.resize(d3, (template_w, template_h), interpolation=cv2.INTER_AREA)
    d4 = cv2.resize(d4, (template_w, template_h), interpolation=cv2.INTER_AREA)

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

def capture_result(arena=False):
    if arena:
        im = SCREEN_CAPTURE.capture((1581,519,50,30)).copy()
    else:
        im = SCREEN_CAPTURE.capture((1581,522,50,30)).copy()
    img= im[:,:,:3]
    return img

def capture_board(shift=False):
    if shift:
        im = SCREEN_CAPTURE.capture((int(START_X-7),int(START_Y), int(8*STEP), int(8*STEP))).copy()
    else:
        im = SCREEN_CAPTURE.capture((int(START_X),int(START_Y), int(8*STEP), int(8*STEP))).copy()
    img= im[:,:,:3]
    return img

def capture_bottom_clock(state="play"):
    """Capture bottom clock using dynamic coordinates."""
    try:
        x, y, w, h = get_clock_info('bottom_clock', state)
        im = SCREEN_CAPTURE.capture((x, y, w, h)).copy()
    except:
        # Fallback to legacy variables if config fails
        if state == "play":
            im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
        elif state == "start1":
            im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_START, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
        elif state == "start2":
            im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_START_2, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
        elif state == "end1":
            im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_END, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
        elif state == "end2":
            im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_END_2, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
        elif state == "end3":
            im = SCREEN_CAPTURE.capture((BOTTOM_CLOCK_X,BOTTOM_CLOCK_Y_END_3, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
    img= im[:,:,:3]
    return img

def capture_top_clock(state="play"):
    """Capture top clock using dynamic coordinates."""
    try:
        x, y, w, h = get_clock_info('top_clock', state)
        im = SCREEN_CAPTURE.capture((x, y, w, h)).copy()
    except:
        # Fallback to legacy variables if config fails
        if state == "play":
            im = SCREEN_CAPTURE.capture((TOP_CLOCK_X,TOP_CLOCK_Y, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
        elif state == "start1":
            im = SCREEN_CAPTURE.capture((TOP_CLOCK_X,TOP_CLOCK_Y_START, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
        elif state == "start2":
            im = SCREEN_CAPTURE.capture((TOP_CLOCK_X,TOP_CLOCK_Y_START_2, CLOCK_WIDTH, CLOCK_HEIGHT)).copy()
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
            if len(detected_moved) % 2 == 0:
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
            else:
                # then it must have been a move followed by an immediate premove with the same piece.
                # in which case it may be impossible to work out whos turn it is.
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
    a1_img = SCREEN_CAPTURE.capture((int(START_X),int(START_Y + 7*STEP), PIECE_STEP, PIECE_STEP))
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
    
    images = [image[x*STEP:x*STEP+PIECE_STEP, y*STEP:y*STEP+PIECE_STEP] for x in range(8) for y in range(8)]
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

def capture_rating(side, position):
    """
    Capture and recognize a chess rating from the screen using pytesseract
    
    Args:
        side (str): Either 'own' or 'opp' 
        position (str): Either 'start' or 'playing'
    
    Returns:
        int: The recognized rating, or None if confidence is too low
    """
    # Determine Y coordinate based on arguments
    if side == 'own':
        if position == 'start':
            y_coord = OWN_RATING_Y_WHITE
        elif position == 'playing':
            y_coord = OWN_RATING_Y_BLACK
        else:
            raise ValueError("position must be 'start' or 'playing'")
    elif side == 'opp':
        if position == 'start':
            y_coord = OPP_RATING_Y_WHITE
        elif position == 'playing':
            y_coord = OPP_RATING_Y_BLACK
        else:
            raise ValueError("position must be 'start' or 'playing'")
    else:
        raise ValueError("side must be 'own' or 'opp'")
    
    # Capture screenshot
    rating_img = SCREEN_CAPTURE.capture((RATING_X, y_coord, RATING_WIDTH, RATING_HEIGHT)).copy()
    
    img = rating_img[:,:,:3]
    
    # Preprocess image using the same approach as the notebook's preprocess_image function
    # Convert to grayscale (similar to preprocess_image "default" type)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get image with only black and white
    _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use pytesseract to extract text with confidence
    custom_config = '--oem 3 --psm 7'
    
    try:
        # Get text and confidence data
        data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # Filter out low-confidence text and calculate average confidence
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        if not confidences:
            return None
            
        avg_confidence = sum(confidences) / len(confidences)
        
        # If confidence is too low, return None
        if avg_confidence < 75:
            return None
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
        
        if not text:
            return None
        
        # Remove question mark if present
        if text.endswith('?'):
            text = text[:-1]
        
        # Try to convert to integer
        try:
            rating = int(text)
            # Ensure rating is reasonable (less than 9999)
            if rating < 9999:
                return rating
            else:
                return None
        except ValueError:
            return None
            
    except Exception as e:
        # Handle any pytesseract errors
        print("Tesseract Error in capture_rating: {} \n".format(e))
        # Save the image to Error_files/ directory with timestamp for debugging
        try:
            # Create Error_files directory if it doesn't exist
            error_dir = "Error_files"
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_filename = os.path.join(error_dir, f"tesseract_error_{timestamp}.png")
            
            # Save the original and processed images
            cv2.imwrite(error_filename, img)
            cv2.imwrite(error_filename.replace(".png", "_processed.png"), processed_img)
            
            print(f"Saved error images to {error_filename}")
        except Exception as save_error:
            print(f"Could not save error image: {save_error}")
        return None

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