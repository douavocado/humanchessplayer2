#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:43:23 2024

@author: james
"""

import os
import pyautogui
import random
import chess
import ctypes
import subprocess
import time
import datetime
import cv2
import numpy as np

from common.custom_cursor import CustomCursor

from engine import Engine
from common.constants import QUICKNESS, MOUSE_QUICKNESS, DIFFICULTY
from common.utils import patch_fens, check_safe_premove

from chessimage.image_scrape_utils import (SCREEN_CAPTURE, START_X, START_Y, STEP, capture_board, capture_top_clock,
                                           capture_bottom_clock, get_fen_from_image, check_fen_last_move_bottom,
                                           read_clock, find_initial_side, detect_last_move_from_img, check_turn_from_last_moved,
                                           capture_result, compare_result_images, capture_rating)

# import threading
# from multiprocessing import Process, Manager

# global variables
FEN_NO_CAP = 8 # the max number of successive fens e store from the most recent position
SCRAPE_EVERY = 0.5 # the time gap between scraping
MOVE_DELAY = 0.25 # the amount of time we take per move minus the time the engine calc time from other aspects (time scrape, position updates, moving pieces etc.)
DRAG_MOVE_DELAY = 0.07 # the amount of time to allow for move to snap onto the board before taking a screenshot
CLICK_MOVE_DELAY = 0.03

CURSOR = CustomCursor() # mouse control object which simulates human-like movement with mouse

LOG_FILE = os.path.join(os.getcwd(),"Client_logs",str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.txt')
LOG = ""
     
ENGINE = Engine(playing_level=DIFFICULTY)

# TODO: incorporate settings for increment games and beserk games
GAME_INFO = {"playing_side": None,
                  "self_initial_time": None,
                  "opp_initial_time": None,
                  "opp_rating": None,
                  "self_rating": None} # these statistics don't change within a game

CASTLING_RIGHTS_FEN = "KQkq"

DYNAMIC_INFO = {"fens":[],
                     "self_clock_times":[],
                     "opp_clock_times":[],
                     "last_moves": []}


PONDER_DIC = {}

HOVER_SQUARE = None



#linux funciton to check capslock
def is_capslock_on():
    if subprocess.check_output('xset q | grep LED', shell=True)[65] == 48 :
        return False
    elif subprocess.check_output('xset q | grep LED', shell=True)[65] == 49 :
        return True
    

def drag_mouse(from_x, from_y, to_x, to_y, tolerance=0):
    """ Make human drag and drop move with human mouse speed and randomness in mind.
        
        Returns True if move was made successfully, else if mouse slip was made return False.
    """
    # 1 in 100 moves, we simulate a potential mouse slip
    successful = True
    if np.random.random() < 0.03:
        tolerance = tolerance * 2
        offset_x = np.clip(np.random.randn()*tolerance, - STEP/1.5, STEP/1.5)
        offset_y = np.clip(np.random.randn()*tolerance, - STEP/1.5, STEP/1.5)
        if np.abs(offset_x) > STEP/2 or np.abs(offset_y) > STEP/2:
            successful = False
    else:
        offset_x = np.clip(np.random.randn()*tolerance, - STEP/2.2, STEP/2.2)
        offset_y = np.clip(np.random.randn()*tolerance, - STEP/2.2, STEP/2.2)
    
    new_from_x = from_x + tolerance * (np.random.random() - 0.5)
    new_from_y = from_y + tolerance * (np.random.random() - 0.5)
    new_to_x = to_x + offset_x
    new_to_y = to_y + offset_y
    
    current_x, current_y = pyautogui.position()
    from_distance =np.sqrt( (new_from_x - current_x)**2 + (new_from_y - current_y)**2 )
    duration_from = MOUSE_QUICKNESS/10000 * (0.8 + 0.4*random.random()) * (from_distance)**0.3
    to_distance = np.sqrt( (new_from_x - new_to_x)**2 + (new_from_y - new_to_y)**2 )
    duration_to = MOUSE_QUICKNESS/10000 * (0.8 + 0.4*random.random()) * (to_distance)**0.5
    # duration_from =0.001
    # duration_to = 0.001    
    CURSOR.drag_and_drop([new_from_x, new_from_y], [new_to_x, new_to_y], duration=[duration_from, duration_to])

    return successful

def click_to_from_mouse(from_x, from_y, to_x, to_y, tolerance=0):
    """ Exactly the same as drag mouse, but sometimes we mix it up by clicking two squares
        rather than click and drag for variation. Tends to be a little faster than drag
        and drop.
        
        Returns True if move was made successfully, else False if mouse slip was made.    
    """
    # 1 in 100 moves, we simulate a potential mouse slip
    successful = True
    if np.random.random() < 0.03:
        tolerance = tolerance * 2
        offset_x = np.clip(np.random.randn()*tolerance, - STEP/1.5, STEP/1.5)
        offset_y = np.clip(np.random.randn()*tolerance, - STEP/1.5, STEP/1.5)
        if np.abs(offset_x) > STEP/2 or np.abs(offset_y) > STEP/2:
            successful = False
    else:
        offset_x = np.clip(np.random.randn()*tolerance, - STEP/2.2, STEP/2.2)
        offset_y = np.clip(np.random.randn()*tolerance, - STEP/2.2, STEP/2.2)
    
    new_from_x = from_x + tolerance * (np.random.random() - 0.5)
    new_from_y = from_y + tolerance * (np.random.random() - 0.5)
    new_to_x = to_x + offset_x
    new_to_y = to_y + offset_y
    
    current_x, current_y = pyautogui.position()
    from_distance =np.sqrt( (new_from_x - current_x)**2 + (new_from_y - current_y)**2 )
    duration_from = MOUSE_QUICKNESS/10000 * (0.8 + 0.4*random.random()) * np.sqrt(from_distance)
    to_distance = np.sqrt( (new_from_x - new_to_x)**2 + (new_from_y - new_to_y)**2 )
    duration_to = MOUSE_QUICKNESS/10000 * (0.8 + 0.4*random.random()) * np.sqrt(to_distance)
    
    # duration_from = 0.001
    # duration_to = 0.001
    
    CURSOR.move_to([new_from_x, new_from_y], duration=duration_from, steady=True)
    pyautogui.click(button="left")
    CURSOR.move_to([new_to_x, new_to_y], duration=duration_to, steady=True)
    pyautogui.click(button="left")
    
    return successful

def click_mouse(x, y, tolerance=0, clicks=1, duration=0.5):
    new_x = x + tolerance * (np.random.random() - 0.5)
    new_y = y + tolerance * (np.random.random() - 0.5)
    
    CURSOR.move_to([new_x, new_y], duration=duration, steady=True)
    pyautogui.click(button="left", clicks=clicks)


def scrape_move_change(side):
    im = SCREEN_CAPTURE.capture((int(START_X),int(START_Y), int(8*STEP), int(8*STEP)))
    return get_move_change(im[:,:,:3], bottom=side)

def get_move_change(image, bottom='w'):
    """ If there has been a move change detected (indicated by colours) on the screenshot,
        then returns the two squares in a list of two. Otherwise, returns None. """
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
    colours = set()
    for i in range(64):
        column_i = i%8
        row_i = i // 8
        pixel_x = int(tile_width*column_i + epsilon)
        pixel_y = int(tile_height*row_i + epsilon)
        rgb = image[pixel_y, pixel_x, :]
        colours.add(tuple(rgb))
        if (rgb == [143,155,59]).all() or (rgb == [205, 211, 145]).all() or (rgb == [95, 92, 60]).all():
            detected.append(column_dic[column_i]+row_dic[row_i])
            
    if len(detected) == 0:
        return None
    elif len(detected) != 2:
        #print("Unexpectedly found {} detected change squares: {}".format(len(detected), detected))
        # this tends to happen alot when premoving
        return None
    else:
        return [detected[0]+detected[1], detected[1] + detected[0]]

def new_game_found():
    """ Uses screenshot to detect whether we have started new game.
    
        Returns None if not, else returns our starting initial time in seconds.
    """
    # try to read bot clock for start position. if none is found, then haven't started the game
    res = read_clock(capture_bottom_clock(state="start1"))
    if res is not None:
        return res
    
    res2 = read_clock(capture_bottom_clock(state="start2"))
    if res2 is not None:
        return res2
    return None # either returns None, no clock found

def game_over_found():
    """ Uses screenshot to detect whether game has finished.
    
        Returns True or False
    """
    res = read_clock(capture_bottom_clock(state="end1"))
    if res is not None:
        return True
    res2 = read_clock(capture_bottom_clock(state="end2"))
    if res2 is not None:
        return True
    res3 = read_clock(capture_bottom_clock(state="end3"))
    if res3 is not None:
        return True
    return False

def await_new_game(timeout=60):
    time_start = time.time()
    while time.time() - time_start < timeout:
        res = new_game_found()
        if res is not None:
            sound_file = "assets/audio/new_game_found.mp3"
            os.system("mpg123 -q " + sound_file)
            return res
    
    sound_file = "assets/audio/alert.mp3"
    os.system("mpg123 -q " + sound_file)
    return None

def set_game(starting_time):
    ''' Once client has found game, sets up game parameters. '''
    global HOVER_SQUARE, GAME_INFO, LOG, CASTLING_RIGHTS_FEN, DYNAMIC_INFO, PONDER_DIC
    
    # resetting hover square
    HOVER_SQUARE = None
    # getting game information, including the side the player is playing and the initial time
    board_img = capture_board()

    # get ratings of both players
    opp_rating = capture_rating(side="opp", position="start")
    if opp_rating is None:
        # try again with playing position
        opp_rating = capture_rating(side="opp", position="playing")
    own_rating = capture_rating(side="own", position="start")
    if own_rating is None:
        # try again with start position
        own_rating = capture_rating(side="own", position="playing")
    GAME_INFO["opp_rating"] = opp_rating
    GAME_INFO["self_rating"] = own_rating
    LOG += "Detected ratings: Opponent: {}, Self: {} \n".format(opp_rating, own_rating)
    
    GAME_INFO["self_initial_time"] = starting_time
    GAME_INFO["opp_initial_time"] = starting_time
    
    # find out our side
    GAME_INFO["playing_side"] = find_initial_side()
    if GAME_INFO["playing_side"] == chess.WHITE:
        bottom = "w"
    else:
        bottom = "b"
    
    if bottom == "w":
        # assume it is our turn
        turn = chess.WHITE
    else:
        # check if move has been played
        move_res = detect_last_move_from_img(board_img)
        if len(move_res) == 0:
            turn = chess.WHITE
        else:
            turn = chess.BLACK

    starting_fen = get_fen_from_image(board_img, bottom=bottom, turn=turn)    
    
    # check turn is in fact our turn
    if check_fen_last_move_bottom(starting_fen, board_img, bottom) == False:
        LOG += "ERROR: Checking bottom unsuccessful, error. Trying again by switching the turn of starting fen. \n"
        dummy_board = chess.Board(starting_fen)
        dummy_board.turn = chess.BLACK
        new_fen = dummy_board.fen()
        if check_fen_last_move_bottom(new_fen, board_img, bottom) == True:
            LOG += "Corrected by switching turn. \n"
            starting_fen = new_fen
        else:
            error_filename = os.path.join("Error_files", "board_img_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')                
            LOG += "ERROR: Not corrected. Continuingly anyway. Saving board image to {}. \n".format(error_filename)
            cv2.imwrite(error_filename, board_img)
        
    else:
        LOG += "Checking bottom successfully matched. \n"
    
    # setting up castling rights
    CASTLING_RIGHTS_FEN = "KQkq"
    dummy_board = chess.Board(starting_fen)
    dummy_board.set_castling_fen(CASTLING_RIGHTS_FEN)
    starting_fen = dummy_board.fen()
    
    
    # Now update the dynamic_information
    if bottom == "w":
        DYNAMIC_INFO["fens"] = [starting_fen]
    elif chess.Board(starting_fen).board_fen() == chess.STARTING_BOARD_FEN:
        DYNAMIC_INFO["fens"] = [starting_fen]
    else:
        DYNAMIC_INFO["fens"] = [chess.STARTING_FEN, starting_fen]
    DYNAMIC_INFO["self_clock_times"]= [starting_time]
    DYNAMIC_INFO["opp_clock_times"] = [starting_time]
    
    # If we are black, then we can check the move made
    if bottom == "b":
        res = patch_fens(chess.STARTING_FEN, starting_fen, depth_lim=1)
        if res is not None:                
            DYNAMIC_INFO["last_moves"]= res[0]
        else:
            LOG += "ERROR: Couldn't find linking move between first fen and starting board fen {}. \n".format(starting_fen)
            DYNAMIC_INFO["last_moves"] = []
    else:
        DYNAMIC_INFO["last_moves"] = []
    
    LOG += "Finished setting up game. \n"
    LOG += "Game information updated to: {} \n".format(GAME_INFO)
    # reset ponder positions
    PONDER_DIC = {}

def write_log():
    """ Writes down thinking into a log file for debugging. """
    global LOG, LOG_FILE
    with open(LOG_FILE,'a') as log:
        log.write(LOG)
        log.close()
    LOG = ""

def update_castling_rights(new_moves: list):
    """ Given moves or new moves found, update castling rights based on these moves. """
    global LOG, CASTLING_RIGHTS_FEN, DYNAMIC_INFO
    LOG += "Updating castling rights from new move ucis {} with current castling rights {}. \n".format(new_moves, CASTLING_RIGHTS_FEN)
    for letters in [CASTLING_RIGHTS_FEN[i:i+2] for i in range(0, len(CASTLING_RIGHTS_FEN), 2)]:
        # make sure we have enough positions to evaluate whether the new_moves involved king moves or not
        if len(DYNAMIC_INFO["fens"]) < len(new_moves):
            LOG += "ERROR: Not enough fens (length {}) to update castling rights. Ignoring. \n".format(len(DYNAMIC_INFO["fens"]))
            break
        else:
            from_i = None
            move_objs = [chess.Move.from_uci(x) for x in new_moves]
            for i, move_obj in enumerate(move_objs): # earliest first
                if chess.Board(DYNAMIC_INFO["fens"][(i-len(new_moves)-1)]).piece_type_at(move_obj.from_square) == chess.KING:
                    colour = chess.Board(DYNAMIC_INFO["fens"][(i-len(new_moves)-1)]).color_at(move_obj.from_square)
                    if colour == chess.WHITE and letters == "KQ":
                        # white king moved and had castling rights
                        CASTLING_RIGHTS_FEN = CASTLING_RIGHTS_FEN.replace("KQ", "")
                        LOG += "Removed white castling rights based on move {} \n".format(move_obj.uci())
                    elif colour == chess.BLACK and letters == "kq":
                        CASTLING_RIGHTS_FEN = CASTLING_RIGHTS_FEN.replace("kq", "")
                        LOG += "Removed black castling rights based on move {} \n".format(move_obj.uci())
                    from_i = i
                    break
            if from_i is not None:
                # correct fens
                for i in range(from_i - len(new_moves), 0):
                    dummy_board = chess.Board(DYNAMIC_INFO["fens"][i])
                    dummy_board.set_castling_fen(CASTLING_RIGHTS_FEN)
                    DYNAMIC_INFO["fens"][i] = dummy_board.fen()
                LOG += "Corrected castling rights of last {} fens: {} \n".format(len(new_moves)-from_i, DYNAMIC_INFO["fens"][from_i - len(new_moves):])
                    

def update_dynamic_info_from_screenshot(move_obj: chess.Move):
    """ The second way we can update the dynamic information, from screenshots
        and change detection. 
    """
    global DYNAMIC_INFO, LOG, GAME_INFO
    # update fen list
    last_board = chess.Board(DYNAMIC_INFO["fens"][-1])
    last_board.push(move_obj)
    DYNAMIC_INFO["fens"].append(last_board.fen())
    DYNAMIC_INFO["fens"] = DYNAMIC_INFO["fens"][-FEN_NO_CAP:]
    
    # No need to update castling rights because this would do it automatically
    
    # Update last moves
    DYNAMIC_INFO["last_moves"].append(move_obj.uci())
    DYNAMIC_INFO["last_moves"] = DYNAMIC_INFO["last_moves"][-(FEN_NO_CAP-1):]
    
    # Update clock times
    # only update the clock times of the side that just moved
    if last_board.turn == GAME_INFO["playing_side"]:
        # then opponent has just moved
        top_clock_img = capture_top_clock(state="play")
        opp_clock_time = read_clock(top_clock_img)
        if opp_clock_time is None:
            # try the starting position
            opp_clock_time = read_clock(capture_top_clock(state="start1"))
            if opp_clock_time is None:
                opp_clock_time = read_clock(capture_top_clock(state="start2"))
            if opp_clock_time is None:
                error_filename = os.path.join("Error_files", "top_clock_play_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
                LOG += "ERROR: Could not find the opponent clock time from move change update. Saving image to {}. \n".format(error_filename)
                cv2.imwrite(error_filename, top_clock_img)
            else:
                DYNAMIC_INFO["opp_clock_times"].append(opp_clock_time)
                DYNAMIC_INFO["opp_clock_times"] = DYNAMIC_INFO["opp_clock_times"][-FEN_NO_CAP:]
        else:
            DYNAMIC_INFO["opp_clock_times"].append(opp_clock_time)
            DYNAMIC_INFO["opp_clock_times"] = DYNAMIC_INFO["opp_clock_times"][-FEN_NO_CAP:]
    else:
        # Then we have just moved
        bot_clock_img = capture_bottom_clock(state="play")
        self_clock_time = read_clock(bot_clock_img)
        if self_clock_time is None:
            # try the starting position
            self_clock_time = read_clock(capture_bottom_clock(state="start1"))
            if self_clock_time is None:
                self_clock_time = read_clock(capture_bottom_clock(state="start2"))
            if self_clock_time is None:
                error_filename = os.path.join("Error_files", "bot_clock_play_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
                LOG += "ERROR: Could not find own clock time from move change update. Saving image to {}. \n".format(error_filename)
                cv2.imwrite(error_filename, bot_clock_img)
            else:
                DYNAMIC_INFO["self_clock_times"].append(self_clock_time)
                DYNAMIC_INFO["self_clock_times"] = DYNAMIC_INFO["self_clock_times"][-FEN_NO_CAP:]
        else:
            DYNAMIC_INFO["self_clock_times"].append(self_clock_time)
            DYNAMIC_INFO["self_clock_times"] = DYNAMIC_INFO["self_clock_times"][-FEN_NO_CAP:]
    LOG += "Updated dynamic information from move-change screenshot prompt: \n"
    LOG += "{} \n".format(DYNAMIC_INFO)
    
def update_dynamic_info_from_fullimage():
    """ Scrape image information from screenshot and update info dic. """
    global LOG, DYNAMIC_INFO, GAME_INFO
    board_img = capture_board()
    top_clock_img = capture_top_clock()
    bot_clock_img = capture_bottom_clock()

    bottom = "w" if GAME_INFO["playing_side"] == chess.WHITE else "b"

    our_time = read_clock(bot_clock_img)        
    opp_time = read_clock(top_clock_img)
    
    fen = get_fen_from_image(board_img, bottom=bottom) # assumes white turn
    # now check the turn
    check_turn_res = check_turn_from_last_moved(fen, board_img, bottom)    
    
    if check_turn_res is None:
        # then there was error, save the error in error files
        error_filename = os.path.join("Error_files", "board_img_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
        LOG += "ERROR: Couldn't find turn from fen {} with bottom {}. Check {} for board_image. Trying to work out from last fen.\n".format(fen, bottom, error_filename)
        cv2.imwrite(error_filename, board_img.astype(np.uint8))
        fen_before = DYNAMIC_INFO["fens"][-1]
        last_board = chess.Board(fen_before)
        # trying 1 move ahead
        dummy_board = chess.Board(fen)
        dummy_board.turn = not last_board.turn
        fen_after = dummy_board.fen()
        
        res = patch_fens(fen_before, fen_after)
        if res is None:
            # try with different turn
            dummy_board = chess.Board(fen)
            dummy_board.turn = last_board.turn
            fen_after = dummy_board.fen()
            res = patch_fens(fen_before, fen_after)
            if res is None:
                LOG += "ERROR: Could not find turn using any method, resorting to the same turn as last turn. \n"
        fen = fen_after
    elif check_turn_res == False:
        # then need to switch the turn
        dummy_board = chess.Board(fen)
        dummy_board.turn = not dummy_board.turn
        fen = dummy_board.fen()
    
    # only update fens if new fen        
    dummy_board = chess.Board(fen)
    last_tracked = chess.Board(DYNAMIC_INFO["fens"][-1])
    if dummy_board.board_fen() == last_tracked.board_fen() and dummy_board.turn == last_tracked.turn:
        # also need the turn to be the same        
        # fen has not changed from last position, do nothing and return
        return 
    
    # Update board fen
    # need to do some adjustments with move numbers
    if len(DYNAMIC_INFO["fens"]) > 0:
        current_move_no = chess.Board(DYNAMIC_INFO["fens"][-1]).fullmove_number
    else:
        current_move_no = None
    
    if dummy_board.turn == chess.WHITE and current_move_no is not None:
        dummy_board.fullmove_number = current_move_no + 1
    elif current_move_no is not None:
        dummy_board.fullmove_number = current_move_no
    
    # set castling rights
    dummy_board.set_castling_fen(CASTLING_RIGHTS_FEN)
    fen = dummy_board.fen()
    
    DYNAMIC_INFO["fens"].append(fen)
    DYNAMIC_INFO["fens"] = DYNAMIC_INFO["fens"][-FEN_NO_CAP:]
    
    # Now update last move
    if len(DYNAMIC_INFO["fens"]) >= 2:
        prev_fen = DYNAMIC_INFO["fens"][-2]
        now_fen = DYNAMIC_INFO["fens"][-1]
        res = patch_fens(prev_fen, now_fen)
        if res is not None:
            LOG += "Able to find linking move(s) between {} and {}: {} \n".format(prev_fen, now_fen, res)
            last_moves, changed_fens = res
            del DYNAMIC_INFO["fens"][-2:]
            DYNAMIC_INFO["fens"].extend(changed_fens)
            DYNAMIC_INFO["fens"] = DYNAMIC_INFO["fens"][-FEN_NO_CAP:]
            DYNAMIC_INFO["last_moves"].extend(last_moves)
            DYNAMIC_INFO["last_moves"] = DYNAMIC_INFO["last_moves"][-(FEN_NO_CAP-1):]
        else:
            LOG += "ERROR: Couldn't find linking move between fens {} and {}. Defaulting to singular fen history and wiping last_move history. \n".format(prev_fen, now_fen)
            last_moves = []
            DYNAMIC_INFO["fens"] = DYNAMIC_INFO["fens"][-1:]
            DYNAMIC_INFO["last_moves"] = []
    
    # Now we have worked out the last move, we need to update castling rights
    if len(last_moves) > 0:
        update_castling_rights(last_moves)
    
    
    # Update clock times
    # Only update the side which has just moved
    if dummy_board.turn == GAME_INFO["playing_side"]:
        # then opponent just moved
        if opp_time is None:
            # try capture at start position
            opp_time = read_clock(capture_top_clock(state="start1"))
            if opp_time is None:
                opp_time = read_clock(capture_top_clock(state="start2"))
            if opp_time is None:
                error_filename = os.path.join("Error_files", "top_clock_play_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
                LOG += "ERROR: Couldn't read opponent time, defaulting to last known clock time. Saving image to {}. \n".format(error_filename)
                cv2.imwrite(error_filename, top_clock_img)
                opp_time = DYNAMIC_INFO["opp_clock_times"][-1]
        DYNAMIC_INFO["opp_clock_times"].append(opp_time)
        DYNAMIC_INFO["opp_clock_times"] = DYNAMIC_INFO["opp_clock_times"][-FEN_NO_CAP:]
        
        # check if opponent has beserked
        # we check this if opponent current time is under half the original initial time AND
        # current move of board is < 5
        curr_move_no = chess.Board(DYNAMIC_INFO["fens"][-1]).fullmove_number
        if curr_move_no < 5 and opp_time < GAME_INFO["opp_initial_time"]/2:
            # correct opp initial time
            LOG += "Opponent detected to have BESERKED, reducting opp initial time from {} to {} \n".format(GAME_INFO["opp_initial_time"], GAME_INFO["opp_initial_time"]/2)
            print("Opponent detected to have BESERKED, reducting opp initial time from {} to {} \n".format(GAME_INFO["opp_initial_time"], GAME_INFO["opp_initial_time"]/2))
            GAME_INFO["opp_initial_time"] /= 2
    else:
        # then we have just moved
        if our_time is None:
            # try capture at start position
            our_time = read_clock(capture_bottom_clock(state="start1"))
            if our_time is None:
                our_time = read_clock(capture_bottom_clock(state="start2"))
            if our_time is None:
                error_filename = os.path.join("Error_files", "bot_clock_play_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
                LOG += "ERROR: Couldn't read our own time, defaulting to last known clock time. Saving image to {}. \n".format(error_filename)
                cv2.imwrite(error_filename, bot_clock_img)
                our_time = DYNAMIC_INFO["self_clock_times"][-1]                
        DYNAMIC_INFO["self_clock_times"].append(our_time)
        DYNAMIC_INFO["self_clock_times"] = DYNAMIC_INFO["self_clock_times"][-FEN_NO_CAP:]
        
        # check if we have beserked
        # we check this if our current time is under half the original initial time AND
        # current move of board is < 5
        curr_move_no = chess.Board(DYNAMIC_INFO["fens"][-1]).fullmove_number
        if curr_move_no < 5 and our_time < GAME_INFO["self_initial_time"]/2:
            # correct opp initial time
            LOG += "Detected to have BESERKED, reducting self initial time from {} to {} \n".format(GAME_INFO["self_initial_time"], GAME_INFO["self_initial_time"]/2)
            print("Detected to have BESERKED, reducting self initial time from {} to {} \n".format(GAME_INFO["self_initial_time"], GAME_INFO["self_initial_time"]/2))
            GAME_INFO["self_initial_time"] /= 2
    
    LOG += "Updated dynamic information from full image scans: \n"
    LOG += "{} \n".format(DYNAMIC_INFO)
    


def check_our_turn():
    """ Check our dynamic information dictionary to see if it is currently our turn. """
    last_fen = DYNAMIC_INFO["fens"][-1]
    playing_side = GAME_INFO["playing_side"]
    board = chess.Board(last_fen)
    
    return board.turn == playing_side

def check_game_end(arena=False):
    """ Check whether the game has ended from the last scrape dic. """
    # check via last board info
    if len(DYNAMIC_INFO["fens"]) > 0:
        board = chess.Board(DYNAMIC_INFO["fens"][-1])
        if board.outcome() is not None:
            return True
    
    # check via clock position
    # return game_over_found()
    # check via result image
    result_img = capture_result(arena=arena)
    if compare_result_images(result_img, cv2.imread("chessimage/blackwin_result.png")) > 0.7:
        return True
    elif compare_result_images(result_img, cv2.imread("chessimage/whitewin_result.png")) > 0.7:
        return True
    elif compare_result_images(result_img, cv2.imread("chessimage/draw_result.png")) > 0.7:
        return True
    return False

def await_move(arena=False):
    ''' The main update step for the lichess client. We do not scrape any information
        at all, because this can be detected and banned quite quickly.
        
        We shall until it is our move, then we call the engine to decide what to do.
        Function returns True when it is our turn or False when the game has ended.
    '''
    global GAME_INFO, DYNAMIC_INFO, HOVER_SQUARE
    while True:
        # First check if game has ended
        if check_game_end(arena=arena):
            return  False# False
        # Check if manual mode is on
        if is_capslock_on():
            continue
        # Next try full body scan
        update_dynamic_info_from_fullimage()
        
        # See if it is our turn
        if check_our_turn() == True:
            return True
        
        # In the meantime check for updates via screenshot method. The amount of time we
        # shall spend doing this will be enough so we can scrape again after
        tries = 0
        tries_cap = 5 # some positive number to start with
        while tries < tries_cap:
            # start_time = time.time()
            if GAME_INFO["playing_side"] == chess.WHITE:
                bottom = "w"
            else:
                bottom = "b"
            # Check if manual mode is on
            if is_capslock_on():
                break
            move_change = scrape_move_change(bottom)
            # if there has been a move change detected,
            # we need to check whether it truly corresponds to a move we can play
            # on our last recorded board
            if move_change is not None:
                move1_uci, move2_uci = move_change
                last_board = chess.Board(DYNAMIC_INFO["fens"][-1])
                # for the case of castling, move change will give the two squares of king and rook rather than move squares of the king
                move1 = chess.Move.from_uci(move1_uci)
                if last_board.piece_type_at(move1.from_square) == chess.KING and last_board.color_at(move1.from_square) == chess.WHITE:
                    if move1.from_square == chess.E1 and move1.to_square == chess.H1:
                        move1 = chess.Move(chess.E1, chess.G1)
                    elif move1.from_square == chess.E1 and move1.to_square == chess.A1:
                        move1 = chess.Move(chess.E1, chess.C1)
                elif last_board.piece_type_at(move1.from_square) == chess.KING and last_board.color_at(move1.from_square) == chess.BLACK:
                    if move1.from_square == chess.E8 and move1.to_square == chess.H8:
                        move1 = chess.Move(chess.E1, chess.G1)
                    elif move1.from_square == chess.E8 and move1.to_square == chess.A8:
                        move1 = chess.Move(chess.E8, chess.C8)
                move2 = chess.Move.from_uci(move2_uci)
                if last_board.piece_type_at(move2.from_square) == chess.KING and last_board.color_at(move2.from_square) == chess.WHITE:
                    if move2.from_square == chess.E1 and move2.to_square == chess.H1:
                        move2 = chess.Move(chess.E1, chess.G1)
                    elif move2.from_square == chess.E1 and move2.to_square == chess.A1:
                        move2 = chess.Move(chess.E1, chess.C1)
                elif last_board.piece_type_at(move2.from_square) == chess.KING and last_board.color_at(move2.from_square) == chess.BLACK:
                    if move2.from_square == chess.E8 and move2.to_square == chess.H8:
                        move2 = chess.Move(chess.E1, chess.G1)
                    elif move2.from_square == chess.E8 and move2.to_square == chess.A8:
                        move2 = chess.Move(chess.E8, chess.C8)
                if move1 in last_board.legal_moves:
                    # then we have truly found a move update
                    update_dynamic_info_from_screenshot(move1)
                    return True
                elif move2 in last_board.legal_moves:
                    # then we have truly found a move update
                    update_dynamic_info_from_screenshot(move2)
                    return True
            
            # end_time = time.time()
            
            # Dynamically uppdate the max tries cap depending on how long it is
            # taking us to check ia screenshots
            # one_loop_time = end_time-start_time
            # tries_cap = SCRAPE_EVERY // one_loop_time
            tries += 1
        
        # hover mouse
        if DYNAMIC_INFO["self_clock_times"][-1] > 15:
            if HOVER_SQUARE is None:
                if np.random.random() < 0.9:
                    hover(duration=np.random.random()/5)
                else:
                    wander()
            elif np.random.random() < 0.06:
                hover(duration=np.random.random()/5)
            elif np.random.random() < 0.04:
                wander()
            
def wander(max_duration=0.15):
    """ Move the mouse randomly to a position on the board close to our side. """
    global LOG
    # Check if there is human interference
    if is_capslock_on():
        LOG += "Tried to hover, but failed as caps lock is on. \n "
        return False
    
    current_x, current_y = pyautogui.position()
    centre_x = START_X + 4*STEP
    centre_y = START_Y + 4*STEP
    
    m_x = 0.8*current_x + 0.2*centre_x
    m_y = 0.8*current_y + 0.2*centre_y
    
    chosen_x = np.clip(m_x + STEP*np.random.randn(), START_X, START_X + 8*STEP)
    chosen_y  = np.clip(m_y + STEP*np.random.randn(), START_Y, START_Y + 8*STEP)
    
    
    distance =np.sqrt( (chosen_x - current_x)**2 + (chosen_y - current_y)**2 )
    duration = max(min(MOUSE_QUICKNESS/5000 * (0.8 + 0.4*np.random.random()) * np.sqrt(distance), max_duration), 0.01)
    CURSOR.move_to([chosen_x, chosen_y], duration=duration)

# def hover_own_turn(duration=0.15, noise=STEP*2):
#     """ When it is our own turn, input random mouse movements which make us hover over our current pieces,
#         particularly if they are moves we are considering from ponder_dic.   Perhaps 
#         even pick up pieces before changing mind.
#     """
#     global PONDER_DIC, HOVER_SQUARE, LOG, DYNAMIC_INFO
#     while True:
#         stat_time = np.random.uniform(0.1, 0.2)
#         time.sleep(stat_time)
#         #print(is_hovering_dic)
#         if DYNAMIC_INFO['self_clock_times'][-1] <= 15:
#             # don't hover if we have too little time.
#             continue
#         elif np.random.random() < 0.7: # we hover
#             if HOVER_SQUARE is None:
#                 # set new hover square
#                 last_known_board = chess.Board(DYNAMIC_INFO["fens"][-1])
#                 relevant_move_objs = [chess.Move.from_uci(x) for x in PONDER_DIC.values() if last_known_board.color_at(chess.Move.from_uci(x).from_square) == GAME_INFO["playing_side"]]
#                 if len(relevant_move_objs) == 0:
#                     # then choose random own piece to hover
#                     own_piece_squares = list(chess.SquareSet(last_known_board.occupied_co[GAME_INFO["playing_side"]]))
#                     random_square = random.choice(own_piece_squares)
#                 else:
#                     # choose last ponder move relevant
#                     random_square = relevant_move_objs[-1].from_square
#                 HOVER_SQUARE = random_square
#             else:
#                 random_square = HOVER_SQUARE
            
#             if GAME_INFO["playing_side"] == chess.WHITE:
#                 # a1 square is bottom left
#                 rank_fr = chess.square_rank(random_square)
#                 file_fr = chess.square_file(random_square)
#                 to_x = np.clip(START_X + file_fr* STEP + STEP/2 + noise * (np.random.random()-0.5), START_X, START_X+8*STEP)
#                 to_y = np.clip(START_Y + (7-rank_fr)*STEP + STEP/2 + noise * (np.random.random()-0.5), START_Y, START_Y + 8*STEP)
                
#             else:
#                 # a1 square is top right
#                 rank_fr = chess.square_rank(random_square)
#                 file_fr = chess.square_file(random_square)
#                 to_x = np.clip(START_X + (7-file_fr)*STEP + STEP/2 + noise * (np.random.random()-0.5), START_X, START_X + 8*STEP)
#                 to_y = np.clip(START_Y + rank_fr*STEP + STEP/2 + noise * (np.random.random()-0.5), START_Y, START_Y + 8*STEP)
            
#             CURSOR.move_to([to_x, to_y], duration=duration,steady=True)
#         elif np.random.random() < 0.4:
#             # we fake drag
#             if HOVER_SQUARE is None or DYNAMIC_INFO["self_clock_times"][-1] < 15:
#                 # then skip
#                 pass
#             else:
#                 random_square = HOVER_SQUARE
#                 if GAME_INFO["playing_side"] == chess.WHITE:
#                     # a1 square is bottom left
#                     rank_fr = chess.square_rank(random_square)
#                     file_fr = chess.square_file(random_square)
#                     from_x = START_X + file_fr* STEP + STEP/2 + np.clip(noise * (np.random.random()-0.5)/4, STEP/2.2, STEP/2.2)
#                     from_y = START_Y + (7-rank_fr)*STEP + STEP/2 + np.clip(noise * (np.random.random()-0.5)/4, STEP/2.2, STEP/2.2)
                    
#                 else:
#                     # a1 square is top right
#                     rank_fr = chess.square_rank(random_square)
#                     file_fr = chess.square_file(random_square)
#                     from_x = START_X + (7-file_fr)*STEP + STEP/2 + np.clip(noise * (np.random.random()-0.5)/4, STEP/2.2, STEP/2.2)
#                     from_y = START_Y + rank_fr*STEP + STEP/2 + np.clip(noise * (np.random.random()-0.5)/4, STEP/2.2, STEP/2.2)
                
#                 to_x = np.clip(from_x + noise*np.random.randn(), START_X, START_X + 8*STEP)
#                 to_y = np.clip(from_y + noise*np.random.randn(), START_Y, START_Y + 8*STEP)
#                 if abs(to_y - from_y) < 5:
#                 	to_y += 5
#                 if abs(to_x - from_x) < 5:
#                 	to_x += 5
#                 CURSOR.fake_drag([from_x, from_y], [to_x, to_y], duration=duration,steady=True)
#         else:
#             #we wander
#             chosen_x = np.clip(START_X + 4*STEP + 2*STEP*np.random.randn(), START_X, START_X + 8*STEP)
#             chosen_y  = np.clip(START_Y + 4*STEP + 2*STEP*np.random.randn(), START_Y, START_Y + 8*STEP)
            
#             CURSOR.move_to([chosen_x, chosen_y], duration=duration)
#         # time.sleep(DRAG_MOVE_DELAY)

def hover(duration=0.1, noise=STEP*2):
    """ In between moves, input random mouse movements which make us hover over our current pieces,
        particularly if they are moves we are considering from ponder_dic.        
    """
    global PONDER_DIC, HOVER_SQUARE, LOG
    # Check if there is human interference
    if is_capslock_on():
        LOG += "Tried to hover, but failed as caps lock is on. \n "
        return False
    
    if HOVER_SQUARE is None:
        # set new hover square
        last_known_board = chess.Board(DYNAMIC_INFO["fens"][-1])
        relevant_move_objs = [chess.Move.from_uci(x["move"]) for x in PONDER_DIC.values() if last_known_board.color_at(chess.Move.from_uci(x["move"]).from_square) == GAME_INFO["playing_side"]]
        if len(relevant_move_objs) == 0:
            # then choose random own piece to hover
            own_piece_squares = list(chess.SquareSet(last_known_board.occupied_co[GAME_INFO["playing_side"]]))
            random_square = random.choice(own_piece_squares)
        else:
            # choose last ponder move relevant
            random_square = relevant_move_objs[-1].from_square
        HOVER_SQUARE = random_square
    else:
        random_square = HOVER_SQUARE
    
    if GAME_INFO["playing_side"] == chess.WHITE:
        # a1 square is bottom left
        rank_fr = chess.square_rank(random_square)
        file_fr = chess.square_file(random_square)
        to_x = np.clip(START_X + file_fr* STEP + STEP/2 + noise * (np.random.random()-0.5), START_X, START_X+8*STEP)
        to_y = np.clip(START_Y + (7-rank_fr)*STEP + STEP/2 + noise * (np.random.random()-0.5), START_Y, START_Y + 8*STEP)
        
    else:
        # a1 square is top right
        rank_fr = chess.square_rank(random_square)
        file_fr = chess.square_file(random_square)
        to_x = np.clip(START_X + (7-file_fr)*STEP + STEP/2 + noise * (np.random.random()-0.5), START_X, START_X + 8*STEP)
        to_y = np.clip(START_Y + rank_fr*STEP + STEP/2 + noise * (np.random.random()-0.5), START_Y, START_Y + 8*STEP)
    
    CURSOR.move_to([to_x, to_y], duration=duration,steady=True)
    
    
    return True

def make_move(move_uci:str, premove:str=None):
    """ Executes mouse clicks for the moves. 
        
        Returns True if clicks were made successfully, else returns False
    """
    global LOG, HOVER_SQUARE
    # Check if there is human interference
    if is_capslock_on():
        LOG += "Tried to make move {} and premove {}, but failed as caps lock is on. \n ".format(move_uci, premove)
        return False
    
    # First, reset previous clicks by right-clicking the centre of the board
    # centre_X, centre_Y = START_X + 3.5*STEP, START_Y + 3.5*STEP
    # pyautogui.click(centre_X, centre_Y, button='right')
    # Now make the move
    from_x, from_y, to_x, to_y = find_clicks(move_uci)
    # pyautogui.click(from_x, from_y, button='left')
    # pyautogui.click(to_x, to_y, button='left')
    # compute randomised offset from centre of the square
    # sometimes we drag and drop, other times we click two squares
    own_time = max(DYNAMIC_INFO["self_clock_times"][-1],1)
    if own_time < 20:
        prob = own_time/25
    else:
        prob = 0.8
    if np.random.random() < prob:
        LOG += "Dragging move {} \n".format(move_uci)
        successful = drag_mouse(from_x, from_y, to_x, to_y, tolerance= 0.2*STEP)
        dragged = True
    else:
        LOG += "Clicking move {} \n".format(move_uci)
        successful = click_to_from_mouse(from_x, from_y, to_x, to_y, tolerance= 0.2*STEP)
        dragged = False
    if successful:
        LOG += "Made clicks for the move {} \n".format(move_uci)
    else:
        LOG += "Tried to make clicks for move {}, but made mouse slip \n".format(move_uci)
        return False
    # If there is a premove
    if premove is not None:
        if dragged == True:
            # wait a bit for previous move to lock in
            time.sleep(DRAG_MOVE_DELAY)
            dragged = False
        else:
            time.sleep(CLICK_MOVE_DELAY)
        from_x, from_y, to_x, to_y = find_clicks(premove)
        if np.random.random() < prob:
            successful = drag_mouse(from_x, from_y, to_x, to_y, tolerance=0.2*STEP)
            dragged = True
        else:
            successful = click_to_from_mouse(from_x, from_y, to_x, to_y, tolerance=0.2*STEP)
        if successful:
            LOG += "Made clicks for the premove {} \n".format(premove)
        else:
            LOG += "Tried to make clicks for premove {}, but made mouse slip. \n".format(premove)
    
    # reset hover square
    HOVER_SQUARE = None
    
    if dragged == True:
        # wait a bit for board to update and snap move into place
        time.sleep(DRAG_MOVE_DELAY)
    else:
        time.sleep(CLICK_MOVE_DELAY)
    return True

def berserk():
    """ Click beserk button in tournaments """
    global LOG
    # can only execute if no human interference.
    if is_capslock_on():
        LOG += "Tried to berserk but failed as caps lock is on. \n "
        return False
    button_x, button_y =  START_X + 10.5*STEP, START_Y + 5.7*STEP
    
    click_mouse(button_x, button_y, tolerance = 10, clicks=1, duration=np.random.uniform(0.3,0.7))
    
    return True

def back_to_lobby():
    """ Click button to go back to lobby after tournament game has finished. """
    global LOG
    # can only execute if no human interference.
    if is_capslock_on():
        LOG += "Tried to go back to lobby but failed as caps lock is on. \n "
        return False
    button_x, button_y =  START_X + 10.5*STEP, START_Y + 4.1*STEP
    
    click_mouse(button_x, button_y, tolerance = 10, clicks=1, duration=np.random.uniform(0.3,0.7))
    
    return True

def resign():
    global LOG
    # can only execute if no human interference.
    if is_capslock_on():
        LOG += "Tried resign the game but failed as caps lock is on. \n "
        return False
    resign_button_x, resign_button_y =  START_X + 10.5*STEP, START_Y + 4.8*STEP
    # pyautogui.click(resign_button_x, resign_button_y, button='left')
    # time.sleep(0.2)
    # pyautogui.click(resign_button_x, resign_button_y, button='left')
    
    click_mouse(resign_button_x, resign_button_y, tolerance = 10, clicks=2, duration=np.random.uniform(0.3,0.7))
    
    return True

def new_game(time_control="1+0"):
    global LOG
    # can only execute if no human interference.
    if is_capslock_on():
        LOG += "Tried to start new game with time control {} but failed as caps lock is on. \n ".format(time_control)
        return False
    
    play_button_x, play_button_y = START_X - 1.9*STEP, START_Y - 0.4*STEP
    # pyautogui.click(play_button_x, play_button_y, button='left')
    click_mouse(play_button_x, play_button_y, tolerance = 10, clicks=1, duration=np.random.uniform(0.3,0.7))
    time.sleep(1.5)
    if time_control == "1+0":
        to_x, to_y = START_X + 1.7*STEP, START_Y + 0.7*STEP
        
        click_mouse(to_x, to_y, tolerance = 20, clicks=1, duration=np.random.uniform(0.3,0.7))
        # pyautogui.click(to_x, to_y, button='left')
    elif time_control == "3+0":
        to_x, to_y = START_X + 5.7*STEP, START_Y + 0.7*STEP
        
        click_mouse(to_x, to_y, tolerance = 20, clicks=1, duration=np.random.uniform(0.3,0.7))
    
    return True

def find_clicks(move_uci):
    ''' Given a move in uci form, find the click from and click to positions. '''
    start_x , start_y = START_X, START_Y # this represents top left square of chess board for calibration
    step = STEP
    move_obj = chess.Move.from_uci(move_uci)
    from_square = move_obj.from_square
    to_square = move_obj.to_square
    if GAME_INFO["playing_side"] == chess.WHITE:
        # a1 square is bottom left
        rank_fr = chess.square_rank(from_square)
        file_fr = chess.square_file(from_square)
        click_from_x = start_x + file_fr*step + step/2
        click_from_y = start_y + (7-rank_fr)*step + step/2
        
        rank_to = chess.square_rank(to_square)
        file_to = chess.square_file(to_square)
        click_to_x = start_x + file_to*step + step/2
        click_to_y = start_y + (7-rank_to)*step + step/2
    else:
        # a1 square is top right
        rank_fr = chess.square_rank(from_square)
        file_fr = chess.square_file(from_square)
        click_from_x = start_x + (7-file_fr)*step + step/2
        click_from_y = start_y + rank_fr*step + step/2
        
        rank_to = chess.square_rank(to_square)
        file_to = chess.square_file(to_square)
        click_to_x = start_x + (7-file_to)*step + step/2
        click_to_y = start_y + rank_to*step + step/2
    return click_from_x, click_from_y, click_to_x, click_to_y

def ponder_position():
    """ Given the position is not our turn, get our engine to ponder the position.
    
        Returns a ponder dic
    """
    global DYNAMIC_INFO, GAME_INFO, ENGINE, LOG, PONDER_DIC
    while True:
        time.sleep(0.1)
        # first assert the position we want to ponder is not our turn
        ponder_board = chess.Board(DYNAMIC_INFO["fens"][-1])
        if ponder_board.turn == GAME_INFO["playing_side"]:
            continue
        elif len(DYNAMIC_INFO["fens"]) < 4:
            # do not ponder if we don't have too much previous information
            continue
        
        # Now ponder position
        prev_ponder_board = chess.Board(DYNAMIC_INFO["fens"][-2])
        
        # if low on time, use stockfish ponder
        own_time = DYNAMIC_INFO["self_clock_times"][-1]
        if own_time < 10:
            time_allowed = 0.05
            ponder_width = 2
            LOG += "Pondering position with Stockfish due to time constraint: {}. \n".format(ponder_board.fen())
            # the number of root moves is less so our moves are not too computer like. we shall randomly sample
            no_legal_moves = len(list(ponder_board.legal_moves))
            sample_no = max(int(no_legal_moves/2),1)
            root_moves = random.sample(list(ponder_board.legal_moves), sample_no)
            LOG += "Randomly sample moves for opponent are: {} \n".format(root_moves)
            ponder_dic = ENGINE.stockfish_ponder(ponder_board, time_allowed, ponder_width, use_ponder=True, root_moves=root_moves)
        else:
            time_allowed = GAME_INFO["self_initial_time"]/60
            ponder_width = 1
            search_width = DIFFICULTY
            LOG += "Pondering position {}. \n".format(ponder_board.fen())
            ponder_dic = ENGINE.ponder(ponder_board, time_allowed, search_width, prev_board=prev_ponder_board, ponder_width=ponder_width, use_ponder=True)
        
        if ponder_dic is not None:
            PONDER_DIC.update(ponder_dic)
            LOG += "Engine outputted ponder_dic during ponder time, updating our ponder_dic. \n"
            LOG += "Current ponder dic is: \n {} \n".format(PONDER_DIC)
            write_log()
        

def run_game(arena=False):
    """ The main lopp for the client while playing the game. """
    global DYNAMIC_INFO, PONDER_DIC, GAME_INFO, ENGINE, LOG
    # ponder_proc = Process(target=ponder_position)
    # ponder_proc.start()
    while True:
        write_log()
            
        result = await_move(arena=arena)
        
        
        write_log()
        if result == False:
            # Then game has ended
            write_log()
            
            # ponder_proc.kill()
            # #own_hover_proc.kill()
            # ponder_proc.join()
            # #own_hover_proc.join()
            # print("Ponder process alive:", ponder_proc.is_alive())
            #print("Hover process alive:", own_hover_proc.is_alive())
            
            return
        
        # check if manual mode on
        if is_capslock_on():
            continue
        
        # start timing
        start = time.time()
        
        # Then it is our move
        # If we have sufficient time, he first thing we check if the current board 
        # position is in our ponder dic.
        # These are positions we have already considered in the past, and their
        # corresponding responses.
        own_time = DYNAMIC_INFO["self_clock_times"][-1]
        current_board_fen = chess.Board(DYNAMIC_INFO["fens"][-1]).board_fen()
        if own_time > 10:            
            if current_board_fen in PONDER_DIC:                    
                response_dic = PONDER_DIC[current_board_fen]
                response_uci = response_dic["move"]
                premove = response_dic["premove"]
                LOG += "Found current position in ponder dic. Responding with corresponding move: {} and premove: {} \n".format(response_uci, premove)
                
                # wait a certain amount of time that depends on the time control
                initial_time = GAME_INFO["self_initial_time"]
                base_time = 0.3*QUICKNESS*initial_time**1.1/(100 + initial_time**0.7)
                wait_time = base_time*(0.8+0.4*random.random())
                LOG += "Spending {} seconds wait for ponder dic response. \n".format(wait_time)
                time.sleep(wait_time)
                successful = make_move(response_uci, premove=premove)
                if successful == True:
                    # We made clicks for the move successfully
                    LOG += "Made pondered moves successfully. \n"
                else:
                    # We try one more time, in case it was mouse slip
                    LOG += "Did not make pondered move successfully, trying once more. \n"
                    successful = make_move(response_uci, premove=premove)
                write_log() 
                continue
            elif len(DYNAMIC_INFO["fens"]) >= 2 and len(PONDER_DIC) >= 1:
                # even if the position is not in ponder dic, if the last
                # pondered move was a safe premove in the previous position,
                # with some probability play it if it is legal move
                last_pondered_move_obj = chess.Move.from_uci(list(PONDER_DIC.values())[-1]["move"])
                last_board = chess.Board(DYNAMIC_INFO["fens"][-2])
                curr_board = chess.Board(current_board_fen)
                #switch the trun
                dummy_board = curr_board.copy()
                dummy_board.turn = GAME_INFO["playing_side"]
                if last_pondered_move_obj in dummy_board.legal_moves:                        
                    if check_safe_premove(last_board, last_pondered_move_obj.uci()):
                        # then with some probability play it
                        # the lower the time control the more likely we do this
                        initial_time = GAME_INFO["self_initial_time"]
                        prob = np.sqrt(1/initial_time)
                        if initial_time < 200 and np.random.random() < prob:                            
                            # then we do it
                            LOG += "Did not find position in ponder_dic, but the last ponder move {} was considered a safe premove in position {}. By chance making this pondered move anyway. \n".format(curr_board.san(last_pondered_move_obj), last_board.fen())
                            base_time = 0.3*QUICKNESS*initial_time**1.1/(100 + initial_time**0.7)
                            wait_time = base_time*(0.8+0.4*random.random())
                            LOG += "Spending {} seconds wait for ponder dic response. \n".format(wait_time)
                            time.sleep(wait_time)
                            successful = make_move(last_pondered_move_obj.uci())
                            if successful == True:
                                # We made clicks for the move successfully
                                LOG += "Made pondered moves successfully. \n"
                            else:
                                # We try one more time, in case it was mouse slip
                                LOG += "Did not make pondered move successfully, trying once more. \n"
                                successful = make_move(last_pondered_move_obj.uci())
                            write_log()
                            continue
                
        elif own_time < 10 and len(DYNAMIC_INFO["fens"]) >= 2 and len(PONDER_DIC) >= 1:
            last_board = chess.Board(DYNAMIC_INFO["fens"][-2])
            curr_board = chess.Board(current_board_fen)
            dummy_board = curr_board.copy()
            dummy_board.turn = GAME_INFO["playing_side"]
            # when we are super low on time, we are likely to premove even more.
            # even when it is not a safe pondered move, we may still do it if is legal
            # with some probability
            prob = (30 - own_time)/50
            # we consider last 10 pondered moves here instead
            candidate_moves = [chess.Move.from_uci(x["move"]) for x in list(PONDER_DIC.values())[-10:] if chess.Move.from_uci(x["move"]) in dummy_board.legal_moves]
            for move_obj in candidate_moves:
                if check_safe_premove(last_board, move_obj.uci()) or np.random.random() < prob:
                    # then we do it
                    LOG += "Did not find position in ponder_dic, but the last ponder move {} was considered a safe premove in position {}. By chance making this pondered move anyway. \n".format(curr_board.san(move_obj), curr_board.fen())
                    base_time = 0.1
                    wait_time = base_time*(0.8+0.4*random.random())
                    LOG += "Spending {} seconds wait for ponder dic response. \n".format(wait_time)
                    time.sleep(wait_time)
                    successful = make_move(move_obj.uci())
                    if successful == True:
                        # We made clicks for the move successfully
                        LOG += "Made pondered moves successfully. \n"
                    else:
                        # We try one more time, in case it was mouse slip
                        LOG += "Did not make pondered move successfully, trying once more. \n"
                        successful = make_move(last_pondered_move_obj.uci())
                    write_log()
                    break
                    
                
        write_log()
        
        # form engine_input_dic
        
        # First make sure the position is not over
        last_board = chess.Board(DYNAMIC_INFO["fens"][-1])
        if last_board.outcome() is not None:
            # then game has finished
            write_log()
            
            #ponder_proc.kill()
            #own_hover_proc.kill()
            #ponder_proc.join()
            #own_hover_proc.join()
            #print("Ponder process is alive:", ponder_proc.is_alive())
            #print("Hover process is alive:", own_hover_proc.is_alive())
            return
        input_dic = DYNAMIC_INFO.copy()
        input_dic["side"] = GAME_INFO["playing_side"]
        input_dic["self_initial_time"] = GAME_INFO["self_initial_time"]
        input_dic["opp_initial_time"] = GAME_INFO["opp_initial_time"]
        input_dic["opp_rating"] = GAME_INFO["opp_rating"]
        input_dic["self_rating"] = GAME_INFO["self_rating"]
        
        # check if manual mode on
        if is_capslock_on():
            continue
        
        ENGINE.update_info(input_dic)
        
        # Once we send the information to the engine, first check if 
        if ENGINE._decide_resign() == True:
            LOG += "Engine has decided to resign. Executing resign interaction. \n"
            time.sleep(2+3*random.random())
            successful = resign()
            if successful == True:
                return
            else:
                # was not able to resign, keep playing I guess
                pass
            
        # check if manual mode on
        if is_capslock_on():
            continue
        
        output_dic = ENGINE.make_move()
        
        LOG += "Received output_dic from engine: {} \n".format(output_dic)
        end = time.time()
        LOG += "Time taken to get move from engine: {} \n".format(end-start)
        write_log()
        # if there is time left over, then wait a bit
        intended_break = output_dic["time_take"]
        if end - start - intended_break < -1*MOVE_DELAY:
            time.sleep(intended_break - (end-start) - MOVE_DELAY)
            
        move_made_uci = output_dic["move_made"]
        premove = output_dic["premove"]
        ponder_dic = output_dic["ponder_dic"]
        
        if ponder_dic is not None:
            # update ponder_dic
            PONDER_DIC.update(ponder_dic)
            LOG += "Engine outputted ponder_dic, updating our ponder_dic. \n"
            LOG += "Current ponder dic is: \n {} \n".format(PONDER_DIC)
        write_log()
        
        
        successful = make_move(move_made_uci, premove=premove)
        if successful == True:
            # We made clicks for the move successfully
            LOG += "Made moves and/or premoves successfully. \n"
            time.sleep(0.1)
        else:
            LOG += "Didn't make move successfully, trying one more time. \n"
            time.sleep(0.1)
            successful = make_move(move_made_uci, premove=premove)
        write_log()
            
            
# if __name__ == "__main__":
#     # while True:
#     games = 6
#     for i in range(games):
#         time.sleep(0.3)
#         res = await_new_game()
#         if res is not None:
#             set_game(res)
#             run_game()
#             print("finished game")
#             if i < games-1:
#                 new_game("1+0")
#             print(i)
        
    
