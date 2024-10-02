#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:03:23 2024

@author: james
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:51:15 2024

@author: xusem
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

from humancursor import SystemCursor

from engine import Engine
from common.constants import QUICKNESS, MOUSE_QUICKNESS, DIFFICULTY
from common.utils import patch_fens

from chessimage.image_scrape_utils import (SCREEN_CAPTURE, START_X, START_Y, STEP, capture_board, capture_top_clock,
                                           capture_bottom_clock, get_fen_from_image, check_fen_last_move_bottom,
                                           read_clock, find_initial_side, detect_last_move_from_img, check_turn_from_last_moved)


# windows function to check capslock
# def is_capslock_on():
#     return True if ctypes.WinDLL("User32.dll").GetKeyState(0x14) else False

#linux funciton to check capslock
def is_capslock_on():
    if subprocess.check_output('xset q | grep LED', shell=True)[65] == 48 :
        return False
    elif subprocess.check_output('xset q | grep LED', shell=True)[65] == 49 :
        return True

FEN_NO_CAP = 8 # the max number of successive fens e store from the most recent position
SCRAPE_EVERY = 0.5 # the time gap between scraping
MOVE_DELAY = 0.25 # the amount of time we take per move minus the time the engine calc time from other aspects (time scrape, position updates, moving pieces etc.)
DRAG_MOVE_DELAY = 0.4 # the amount of time to allow for move to snap onto the board before taking a screenshot

CURSOR = SystemCursor() # mouse control object which simulates human-like movement with mouse

def drag_mouse(from_x, from_y, to_x, to_y, tolerance=0):
    """ Make human drag and drop move with human mouse speed and randomness in mind.
        
        Returns True if move was made successfully, else if mouse slip was made return False.
    """
    # 1 in 100 moves, we simulate a potential mouse slip
    successful = True
    if np.random.random() < 0.01:
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
    duration_from = MOUSE_QUICKNESS/5000 * (0.8 + 0.4*random.random()) * np.sqrt(from_distance)
    to_distance = np.sqrt( (new_from_x - new_to_x)**2 + (new_from_y - new_to_y)**2 )
    duration_to = MOUSE_QUICKNESS/5000 * (0.8 + 0.4*random.random()) * np.sqrt(to_distance)
    
    CURSOR.drag_and_drop([new_from_x, new_from_y], [new_to_x, new_to_y], duration=[duration_from, duration_to])
    
    return successful

def click_mouse(x, y, tolerance=0, clicks=1):
    new_x = x + tolerance * (np.random.random() - 0.5)
    new_y = y + tolerance * (np.random.random() - 0.5)
    
    CURSOR.click_on([new_x, new_y], clicks=clicks)


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

class GameFinder:
    ''' During idle phases, we are scanning a particuar username to see if any active
        games are present, hence avoiding starting/restarting the script. '''
    def __init__(self, shadow_mode=False, log=True):
        self.client = LichessClient()
    
    def run(self):
        while True:
            new_game_starting_time = new_game_found()
            if new_game_starting_time is not None:
                # then the user is in a game
                print ('Found user game!')
                sound_file = "new_game_found.mp3"
                os.system("mpg123 -q " + sound_file)
                self.client.set_game(new_game_starting_time)
                self.client.run_game()
                # new game
                time.sleep(5)
                self.client.new_game()
            else:
                time.sleep(1)

class LichessClient:
    ''' Main class which interacts with Lichess. Plays and recieves moves. Called
        every instance of a game. '''
    
    def __init__(self, log_file: str = os.path.join(os.getcwd(), 'Client_logs',str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.txt')):
        self.log_file = log_file
        self.log = ""
             
        self.engine = Engine(playing_level=DIFFICULTY)
        
        # TODO: incorporate settings for increment games and beserk games
        self.game_info = {"playing_side": None,
                          "initial_time": None,} # these statistics don't change within a game
        
        self.castling_rights_fen = "KQkq"
        
        self.dynamic_info = {"fens":[],
                             "self_clock_times":[],
                             "opp_clock_times":[],
                             "last_moves": []}
        
        
        self.ponder_dic = {}

    def _write_log(self):
        """ Writes down thinking into a log file for debugging. """
        with open(self.log_file,'a') as log:
            log.write(self.log)
            log.close()
        self.log = ""

    def _update_castling_rights(self, new_moves: list):
        """ Given moves or new moves found, update castling rights based on these moves. """
        self.log += "Updating castling rights from new move ucis {} with current castling rights {}. \n".format(new_moves, self.castling_rights_fen)
        for letters in [self.castling_rights_fen[i:i+2] for i in range(0, len(self.castling_rights_fen), 2)]:
            # make sure we have enough positions to evaluate whether the new_moves involved king moves or not
            if len(self.dynamic_info["fens"]) < len(new_moves):
                self.log += "ERROR: Not enough fens (length {}) to update castling rights. Ignoring. \n".format(len(self.dynamic_info["fens"]))
                break
            else:
                from_i = None
                move_objs = [chess.Move.from_uci(x) for x in new_moves]
                for i, move_obj in enumerate(move_objs): # earliest first
                    if chess.Board(self.dynamic_info["fens"][(i-len(new_moves)-1)]).piece_type_at(move_obj.from_square) == chess.KING:
                        colour = chess.Board(self.dynamic_info["fens"][(i-len(new_moves)-1)]).color_at(move_obj.from_square)
                        if colour == chess.WHITE and letters == "KQ":
                            # white king moved and had castling rights
                            self.castling_rights_fen = self.castling_rights_fen.replace("KQ", "")
                            self.log += "Removed white castling rights based on move {} \n".format(move_obj.uci())
                        elif colour == chess.BLACK and letters == "kq":
                            self.castling_rights_fen = self.castling_rights_fen.replace("kq", "")
                            self.log += "Removed black castling rights based on move {} \n".format(move_obj.uci())
                        from_i = i
                        break
                if from_i is not None:
                    # correct fens
                    for i in range(from_i - len(new_moves), 0):
                        dummy_board = chess.Board(self.dynamic_info["fens"][i])
                        dummy_board.set_castling_fen(self.castling_rights_fen)
                        self.dynamic_info["fens"][i] = dummy_board.fen()
                    self.log += "Corrected castling rights of last {} fens: {} \n".format(len(new_moves)-from_i, self.dynamic_info["fens"][from_i - len(new_moves):])
                        

    def _update_dynamic_info_from_screenshot(self, move_obj: chess.Move):
        """ The second way we can update the dynamic information, from screenshots
            and change detection. 
        """
        # update fen list
        last_board = chess.Board(self.dynamic_info["fens"][-1])
        last_board.push(move_obj)
        self.dynamic_info["fens"].append(last_board.fen())
        self.dynamic_info["fens"] = self.dynamic_info["fens"][-FEN_NO_CAP:]
        
        # No need to update castling rights because this would do it automatically
        
        # Update last moves
        self.dynamic_info["last_moves"].append(move_obj.uci())
        self.dynamic_info["last_moves"] = self.dynamic_info["last_moves"][-(FEN_NO_CAP-1):]
        
        # Update clock times
        # only update the clock times of the side that just moved
        if last_board.turn == self.game_info["playing_side"]:
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
                    self.log += "ERROR: Could not find the opponent clock time from move change update. Saving image to {}. \n".format(error_filename)
                    cv2.imwrite(error_filename, top_clock_img)
                else:
                    self.dynamic_info["opp_clock_times"].append(opp_clock_time)
                    self.dynamic_info["opp_clock_times"] = self.dynamic_info["opp_clock_times"][-FEN_NO_CAP:]
            else:
                self.dynamic_info["opp_clock_times"].append(opp_clock_time)
                self.dynamic_info["opp_clock_times"] = self.dynamic_info["opp_clock_times"][-FEN_NO_CAP:]
        else:
            # Then we have just moved
            bot_clock_img = capture_bottom_clock(state="play")
            self_clock_time = read_clock(bot_clock_img)
            if self_clock_time is None:
                # try the starting position
                self_clock_time = read_clock(capture_bottom_clock(state="start1"))
                if self_clock_time is None:
                    self_clock_time = read_clock(capture_board(state="start2"))
                if self_clock_time is None:
                    error_filename = os.path.join("Error_files", "bot_clock_play_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
                    self.log += "ERROR: Could not find own clock time from move change update. Saving image to {}. \n".format(error_filename)
                    cv2.imwrite(error_filename, bot_clock_img)
                else:
                    self.dynamic_info["self_clock_times"].append(self_clock_time)
                    self.dynamic_info["self_clock_times"] = self.dynamic_info["self_clock_times"][-FEN_NO_CAP:]
            else:
                self.dynamic_info["self_clock_times"].append(self_clock_time)
                self.dynamic_info["self_clock_times"] = self.dynamic_info["self_clock_times"][-FEN_NO_CAP:]
        self.log += "Updated dynamic information from move-change screenshot prompt: \n"
        self.log += "{} \n".format(self.dynamic_info)
        
    def _update_dynamic_info_from_fullimage(self):
        """ Scrape image information from screenshot and update info dic. """
        board_img = capture_board()
        top_clock_img = capture_top_clock()
        bot_clock_img = capture_bottom_clock()

        bottom = "w" if self.game_info["playing_side"] == chess.WHITE else "b"

        our_time = read_clock(bot_clock_img)        
        opp_time = read_clock(top_clock_img)
        
        fen = get_fen_from_image(board_img, bottom=bottom) # assumes white turn
        
        # only update fens if new fen        
        dummy_board = chess.Board(fen)
        if dummy_board.board_fen() == chess.Board(self.dynamic_info["fens"][-1]).board_fen():
            # fen has not changed from last position, do nothing and return
            return        
        
        # now check the turn
        check_turn_res = check_turn_from_last_moved(fen, board_img, bottom)
        if check_turn_res is None:
            # then there was error, save the error in error files
            error_filename = os.path.join("Error_files", "board_img_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
            self.log += "ERROR: Couldn't find turn from fen {} with bottom {}. Check {} for board_image. Defaulting to the next turn from the last fen. \n".format(fen, bottom, error_filename)
            cv2.imwrite(error_filename, board_img.astype(np.uint8))
            last_board = chess.Board(self.dynamic_info["fens"][-1])
            dummy_board = chess.Board(fen)
            dummy_board.turn = not last_board.turn
            fen = dummy_board.fen()
        elif check_turn_res == False:
            # then need to switch the turn
            dummy_board = chess.Board(fen)
            dummy_board.turn = not dummy_board.turn
            fen = dummy_board.fen()
            
        # Update board fen
        # need to do some adjustments with move numbers
        if len(self.dynamic_info["fens"]) > 0:
            current_move_no = chess.Board(self.dynamic_info["fens"][-1]).fullmove_number
        else:
            current_move_no = None
        
        if dummy_board.turn == chess.WHITE and current_move_no is not None:
            dummy_board.fullmove_number = current_move_no + 1
        elif current_move_no is not None:
            dummy_board.fullmove_number = current_move_no
        
        # set castling rights
        dummy_board.set_castling_fen(self.castling_rights_fen)
        fen = dummy_board.fen()
        
        self.dynamic_info["fens"].append(fen)
        self.dynamic_info["fens"] = self.dynamic_info["fens"][-FEN_NO_CAP:]
        
        # Now update last move
        if len(self.dynamic_info["fens"]) >= 2:
            prev_fen = self.dynamic_info["fens"][-2]
            now_fen = self.dynamic_info["fens"][-1]
            res = patch_fens(prev_fen, now_fen)
            if res is not None:
                self.log += "Able to find linking move(s) between {} and {}: {} \n".format(prev_fen, now_fen, res)
                last_moves, changed_fens = res
                del self.dynamic_info["fens"][-2:]
                self.dynamic_info["fens"].extend(changed_fens)
                self.dynamic_info["fens"] = self.dynamic_info["fens"][-FEN_NO_CAP:]
                self.dynamic_info["last_moves"].extend(last_moves)
                self.dynamic_info["last_moves"] = self.dynamic_info["last_moves"][-(FEN_NO_CAP-1):]
            else:
                self.log += "ERROR: Couldn't find linking move between fens {} and {}. Defaulting to singular fen history and wiping last_move history. \n".format(prev_fen, now_fen)
                last_moves = []
                self.dynamic_info["fens"] = self.dynamic_info["fens"][-1:]
                self.dynamic_info["last_moves"] = []
        
        # Now we have worked out the last move, we need to update castling rights
        if len(last_moves) > 0:
            self._update_castling_rights(last_moves)
        
        
        # Update clock times
        # Only update the side which has just moved
        if dummy_board.turn == self.game_info["playing_side"]:
            # then opponent just moved
            if opp_time is None:
                # try capture at start position
                opp_time = read_clock(capture_top_clock(state="start1"))
                if opp_time is None:
                    opp_time = read_clock(capture_top_clock(state="start2"))
                if opp_time is None:
                    error_filename = os.path.join("Error_files", "top_clock_play_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
                    self.log += "ERROR: Couldn't read opponent time, defaulting to last known clock time. Saving image to {}. \n".format(error_filename)
                    cv2.imwrite(error_filename, top_clock_img)
                    opp_time = self.dynamic_info["opp_clock_times"][-1]
            self.dynamic_info["opp_clock_times"].append(opp_time)
            self.dynamic_info["opp_clock_times"] = self.dynamic_info["opp_clock_times"][-FEN_NO_CAP:]
        else:
            # then we have just moved
            if our_time is None:
                # try capture at start position
                our_time = read_clock(capture_bottom_clock(state="start1"))
                if our_time is None:
                    our_time = read_clock(capture_bottom_clock(state="start2"))
                if our_time is None:
                    error_filename = os.path.join("Error_files", "bot_clock_play_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')
                    self.log += "ERROR: Couldn't read our own time, defaulting to last known clock time. Saving image to {}. \n".format(error_filename)
                    cv2.imwrite(error_filename, bot_clock_img)
                    our_time = self.dynamic_info["self_clock_times"][-1]                
            self.dynamic_info["self_clock_times"].append(our_time)
            self.dynamic_info["self_clock_times"] = self.dynamic_info["self_clock_times"][-FEN_NO_CAP:]
        
        self.log += "Updated dynamic information from full image scans: \n"
        self.log += "{} \n".format(self.dynamic_info)
        
                    
        
    
    def set_game(self, starting_time):
        ''' Once client has found game, sets up game parameters. '''
        # resetting hover square
        self.hover_square = None
        # getting game information, including the side the player is playing and the initial time
        board_img = capture_board()
        
        self.game_info["initial_time"] = starting_time
        
        # find out our side
        self.game_info["playing_side"] = find_initial_side()
        if self.game_info["playing_side"] == chess.WHITE:
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
            self.log += "ERROR: Checking bottom unsuccessful, error. Trying again by switching the turn of starting fen. \n"
            dummy_board = chess.Board(starting_fen)
            dummy_board.turn = chess.BLACK
            new_fen = dummy_board.fen()
            if check_fen_last_move_bottom(new_fen, board_img, bottom) == True:
                self.log += "Corrected by switching turn. \n"
                starting_fen = new_fen
            else:
                error_filename = os.path.join("Error_files", "board_img_" + str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.png')                
                self.log += "ERROR: Not corrected. Continuingly anyway. Saving board image to {}. \n".format(error_filename)
                cv2.imwrite(error_filename, board_img)
            
        else:
            self.log += "Checking bottom successfully matched. \n"
        
        # setting up castling rights
        self.castling_rights_fen = "KQkq"
        dummy_board = chess.Board(starting_fen)
        dummy_board.set_castling_fen(self.castling_rights_fen)
        starting_fen = dummy_board.fen()
        
        # TODO : include beserk options
        
        # Now update the dynamic_information
        if bottom == "w":
            self.dynamic_info["fens"] = [starting_fen]
        elif chess.Board(starting_fen).board_fen() == chess.STARTING_BOARD_FEN:
            self.dynamic_info["fens"] = [starting_fen]
        else:
            self.dynamic_info["fens"] = [chess.STARTING_FEN, starting_fen]
        self.dynamic_info["self_clock_times"]= [starting_time]
        self.dynamic_info["opp_clock_times"] = [starting_time]
        
        # If we are black, then we can check the move made
        if bottom == "b":
            res = patch_fens(chess.STARTING_FEN, starting_fen, depth_lim=1)
            if res is not None:                
                self.dynamic_info["last_moves"]= res[0]
            else:
                self.log += "ERROR: Couldn't find linking move between first fen and starting board fen {}. \n".format(starting_fen)
                self.dynamic_info["last_moves"] = []
        else:
            self.dynamic_info["last_moves"] = []
        
        self.log += "Finished setting up game. \n"
        self.log += "Game information updated to: {} \n".format(self.game_info)
        # reset ponder positions
        self.ponder_dic = {}
    
    def check_our_turn(self):
        """ Check our dynamic information dictionary to see if it is currently our turn. """
        last_fen = self.dynamic_info["fens"][-1]
        playing_side = self.game_info["playing_side"]
        board = chess.Board(last_fen)
        
        return board.turn == playing_side
    
    def _check_game_end(self):
        """ Check whether the game has ended from the last scrape dic. """
        # check via last board info
        if len(self.dynamic_info["fens"]) > 0:
            board = chess.Board(self.dynamic_info["fens"][-1])
            if board.outcome() is not None:
                return True
        
        # check via clock position
        return game_over_found()
    
    def run_game(self):
        """ The main lopp for the client while playing the game. """
        while True:
            self._write_log()
            result = self.await_move()
            self._write_log()
            if result == False:
                # Then game has ended
                self._write_log()
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
            own_time = self.dynamic_info["self_clock_times"][-1]
            if own_time > 10:
                current_board_fen = chess.Board(self.dynamic_info["fens"][-1]).board_fen()
                if current_board_fen in self.ponder_dic:                    
                    response_uci = self.ponder_dic[current_board_fen]
                    self.log += "Found current position in ponder dic. Responding with corresponding move: {} \n".format(response_uci)
                    
                    # wait a certain amount of time that depends on the time control
                    initial_time = self.game_info["initial_time"]
                    base_time = 0.4*QUICKNESS*initial_time/(85 + initial_time*0.25)
                    wait_time = base_time*(0.8+0.4*random.random())
                    self.log += "Spending {} seconds wait for ponder dic response. \n".format(wait_time)
                    time.sleep(wait_time)
                    successful = self.make_move(response_uci)
                    if successful == True:
                        # We made clicks for the move successfully
                        self.log += "Made pondered moves successfully. \n"
                    continue
            self._write_log()
            
            # form engine_input_dic
            
            # First make sure the position is not over
            last_board = chess.Board(self.dynamic_info["fens"][-1])
            if last_board.outcome() is not None:
                # then game has finished
                self._write_log()
                return
            input_dic = self.dynamic_info.copy()
            input_dic["side"] = self.game_info["playing_side"]
            input_dic["self_initial_time"] = self.game_info["initial_time"]
            input_dic["opp_initial_time"] = self.game_info["initial_time"] # TODO
            
            # check if manual mode on
            if is_capslock_on():
                continue
            
            self.engine.update_info(input_dic)
            
            # Once we send the information to the engine, first check if 
            if self.engine._decide_resign() == True:
                self.log += "Engine has decided to resign. Executing resign interaction. \n"
                time.sleep(2+3*random.random())
                successful = self.resign()
                if successful == True:
                    continue
                else:
                    # was not able to resign, keep playing I guess
                    pass
                
            # check if manual mode on
            if is_capslock_on():
                continue
            output_dic = self.engine.make_move()
            
            self.log += "Received output_dic from engine: {} \n".format(output_dic)
            end = time.time()
            self.log += "Time taken to get move from engine: {} \n".format(end-start)
            self._write_log()
            # if there is time left over, then wait a bit
            intended_break = output_dic["time_take"]
            if end - start - intended_break < -1*MOVE_DELAY:
                time.sleep(intended_break - (end-start) - MOVE_DELAY)
                
            move_made_uci = output_dic["move_made"]
            premove = output_dic["premove"]
            ponder_dic = output_dic["ponder_dic"]
            
            if ponder_dic is not None:
                # update ponder_dic
                self.ponder_dic.update(ponder_dic)
                self.log += "Engine outputted ponder_dic, updating our ponder_dic. \n"
                self.log += "Current ponder dic is: \n {} \n".format(self.ponder_dic)
                self._write_log()
            successful = self.make_move(move_made_uci, premove=premove)
            if successful == True:
                # We made clicks for the move successfully
                self.log += "Made moves and/or premoves successfully. \n"
                self._write_log()
    
    def await_move(self):
        ''' The main update step for the lichess client. We do not scrape any information
            at all, because this can be detected and banned quite quickly.
            
            We shall until it is our move, then we call the engine to decide what to do.
            Function returns True when it is our turn or False when the game has ended.
        '''
        while True:
            # First check if game has ended
            if self._check_game_end():
                return False
            # Check if manual mode is on
            if is_capslock_on():
                continue
            # Next try full body scan
            self._update_dynamic_info_from_fullimage()
            
            # See if it is our turn
            if self.check_our_turn() == True:
                return True
            
            # In the meantime check for updates via screenshot method. The amount of time we
            # shall spend doing this will be enough so we can scrape again after
            tries = 0
            tries_cap = 20 # some positive number to start with
            while tries < tries_cap:
                # start_time = time.time()
                if self.game_info["playing_side"] == chess.WHITE:
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
                    last_board = chess.Board(self.dynamic_info["fens"][-1])
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
                        self._update_dynamic_info_from_screenshot(move1)
                        return True
                    elif move2 in last_board.legal_moves:
                        # then we have truly found a move update
                        self._update_dynamic_info_from_screenshot(move2)
                        return True
                
                # end_time = time.time()
                
                # Dynamically uppdate the max tries cap depending on how long it is
                # taking us to check ia screenshots
                # one_loop_time = end_time-start_time
                # tries_cap = SCRAPE_EVERY // one_loop_time
                tries += 1
            
            # hover mouse
            if self.hover_square is None:
                if np.random.random() < 0.9:
                    self.hover()
                else:
                    self.wander
            elif np.random.random() < 0.06:
                self.hover()
            elif np.random.random() < 0.04:
                self.wander()
                
    def wander(self, max_duration=0.1):
        """ Move the mouse randomly to a position on the board close to our side. """
        # Check if there is human interference
        if is_capslock_on():
            self.log += "Tried to hover, but failed as caps lock is on. \n "
            return False
        
        chosen_x = np.clip(START_X + 4*STEP + 2*STEP*np.random.randn(), START_X, START_X + 8*STEP)
        chosen_y  = np.clip(START_Y + 4*STEP + 2*STEP*np.random.randn(), START_Y, START_Y + 8*STEP)
        
        current_x, current_y = pyautogui.position()
        distance =np.sqrt( (chosen_x - current_x)**2 + (chosen_y - current_y)**2 )
        duration = min(MOUSE_QUICKNESS/5000 * (0.8 + 0.4*np.random.random()) * np.sqrt(distance), max_duration)
        CURSOR.move_to([chosen_x, chosen_y], duration=duration)
    
    def hover(self, duration=0.1, noise=STEP*2):
        """ In between moves, input random mouse movements which make us hover over our current pieces,
            particularly if they are moves we are considering from ponder_dic.        
        """
        # Check if there is human interference
        if is_capslock_on():
            self.log += "Tried to hover, but failed as caps lock is on. \n "
            return False
        
        if self.hover_square is None:
            # set new hover square
            last_known_board = chess.Board(self.dynamic_info["fens"][-1])
            relevant_move_objs = [chess.Move.from_uci(x) for x in self.ponder_dic.values() if last_known_board.color_at(chess.Move.from_uci(x).from_square) == self.game_info["playing_side"]]
            if len(relevant_move_objs) == 0:
                # then choose random own piece to hover
                own_piece_squares = list(chess.SquareSet(last_known_board.occupied_co[self.game_info["playing_side"]]))
                random_square = random.choice(own_piece_squares)
            else:
                # choose last ponder move relevant
                random_square = relevant_move_objs[-1].from_square
            self.hover_square = random_square
        else:
            random_square = self.hover_square
        
        if self.game_info["playing_side"] == chess.WHITE:
            # a1 square is bottom left
            rank_fr = chess.square_rank(random_square)
            file_fr = chess.square_file(random_square)
            to_x = START_X + file_fr* STEP + STEP/2 + noise * (np.random.random()-0.5)
            to_y = START_Y + (7-rank_fr)*STEP + STEP/2 + noise * (np.random.random()-0.5)
            
        else:
            # a1 square is top right
            rank_fr = chess.square_rank(random_square)
            file_fr = chess.square_file(random_square)
            to_x = START_X + (7-file_fr)*STEP + STEP/2 + noise * (np.random.random()-0.5)
            to_y = START_Y + rank_fr*STEP + STEP/2 + noise * (np.random.random()-0.5)
        
        CURSOR.move_to([to_x, to_y], duration=duration)
        
        
        return True
    
    def make_move(self, move_uci:str, premove:str=None):
        """ Executes mouse clicks for the moves. 
            
            Returns True if clicks were made successfully, else returns False
        """
        # Check if there is human interference
        if is_capslock_on():
            self.log += "Tried to make move {} and premove {}, but failed as caps lock is on. \n ".format(move_uci, premove)
            return False
        
        # First, reset previous clicks by right-clicking the centre of the board
        # centre_X, centre_Y = START_X + 3.5*STEP, START_Y + 3.5*STEP
        # pyautogui.click(centre_X, centre_Y, button='right')
        
        # Now make the move
        from_x, from_y, to_x, to_y = self.find_clicks(move_uci)
        # pyautogui.click(from_x, from_y, button='left')
        # pyautogui.click(to_x, to_y, button='left')
        # compute randomised offset from centre of the square
        successful = drag_mouse(from_x, from_y, to_x, to_y, tolerance= 0.2*STEP)
        if successful:
            self.log += "Made clicks for the move {} \n".format(move_uci)
        else:
            self.log += "Tried to make clicks for move {}, but made mouse slip \n".format(move_uci)
            return False
        # If there is a premove
        if premove is not None:
            from_x, from_y, to_x, to_y = self.find_clicks(premove)
            
            successful = drag_mouse(from_x, from_y, to_x, to_y, tolerance=0.7*STEP)
            if successful:
                self.log += "Made clicks for the premove {} \n".format(premove)
            else:
                self.log += "Tried to make clicks for premove {}, but made mouse slip. \n".format(premove)
        
        # reset hover square
        self.hover_square = None
        
        # wait a bit for board to update and snap move into place
        time.sleep(DRAG_MOVE_DELAY)
        return True
    
    def resign(self):
        # can only execute if no human interference.
        if is_capslock_on():
            self.log += "Tried resign the game but failed as caps lock is on. \n "
            return False
        resign_button_x, resign_button_y =  START_X + 10.5*STEP, START_Y + 4.8*STEP
        # pyautogui.click(resign_button_x, resign_button_y, button='left')
        # time.sleep(0.2)
        # pyautogui.click(resign_button_x, resign_button_y, button='left')
        
        click_mouse(resign_button_x, resign_button_y, tolerance = 10, clicks=2)
        
        return True
    
    def new_game(self, time_control="1+0"):
        # can only execute if no human interference.
        if is_capslock_on():
            self.log += "Tried to start new game with time control {} but failed as caps lock is on. \n ".format(time_control)
            return False
        
        play_button_x, play_button_y = START_X - 1.9*STEP, START_Y - 0.4*STEP
        # pyautogui.click(play_button_x, play_button_y, button='left')
        click_mouse(play_button_x, play_button_y, tolerance = 10, clicks=1)
        time.sleep(1.5)
        if time_control == "1+0":
            to_x, to_y = START_X + 1.7*STEP, START_Y + 0.7*STEP
            
            click_mouse(to_x, to_y, tolerance = 20, clicks=1)
            # pyautogui.click(to_x, to_y, button='left')
        
        return True

    def find_clicks(self, move_uci):
        ''' Given a move in uci form, find the click from and click to positions. '''
        start_x , start_y = START_X, START_Y # this represents top left square of chess board for calibration
        step = STEP
        move_obj = chess.Move.from_uci(move_uci)
        from_square = move_obj.from_square
        to_square = move_obj.to_square
        if self.game_info["playing_side"] == chess.WHITE:
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
   

# URL = 'https://lichess.org/GlYCARuKJKOp'