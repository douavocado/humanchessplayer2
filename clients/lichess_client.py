# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:51:15 2024

@author: xusem
"""

import os
from fastgrab import screenshot
import pyautogui
import random
import chess
import requests
import re
from bs4 import BeautifulSoup
import configparser
from requests_html import HTMLSession
import ctypes
import subprocess
import time
import json
import datetime

from engine import Engine
from common.constants import QUICKNESS

# windows function to check capslock
# def is_capslock_on():
#     return True if ctypes.WinDLL("User32.dll").GetKeyState(0x14) else False

#linux funciton to check capslock
def is_capslock_on():
    if subprocess.check_output('xset q | grep LED', shell=True)[65] == 48 :
        return False
    elif subprocess.check_output('xset q | grep LED', shell=True)[65] == 49 :
        return True

config = configparser.ConfigParser()
config.read('config.ini')

STEP = float((config['DEFAULT']['step']))
START_X = int(config['DEFAULT']['start_x']) + STEP/2
START_Y = int(config['DEFAULT']['start_y']) + STEP/2

DIFFICULTY = int(config["DEFAULT"]["difficulty"])

FEN_NO_CAP = 8 # the max number of successive fens e store from the most recent position
SCRAPE_EVERY = 0.5 # the time gap between scraping
MOVE_DELAY = 0.25 # the amount of time we take per move minus the time the engine calc time from other aspects (time scrape, position updates, moving pieces etc.)

SCREEN_CAPTURE = screenshot.Screenshot()

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


class GameFinder:
    ''' During idle phases, we are scanning a particuar username to see if any active
        games are present, hence avoiding starting/restarting the script. '''
    def __init__(self, username, shadow_mode=False, log=True):
        self.username = username
        self.client = LichessClient(username)
    
    def run(self):
        profile_url = 'https://lichess.org/@/'+self.username+'/playing'
        # now see if username was valid or not
        r = requests.get(profile_url)
        valid = re.findall(r"<h1>404<\/h1><div><strong>Page not found!<\/strong>", r.text)
        if len(valid) > 0:
            raise Exception('Username not found: ' + self.username + '. Please make sure the spelling is correct (case sensitive)')
        print('Status connected.')
        while True:
            r = requests.get(profile_url)
            playing = re.findall('Playing right now', r.text)
            if len(playing) > 0:
                # then the user is in a game
                print ('Found user game!')
                sound_file = "assets/audio/new_game_found.mp3"
                os.system("mpg123 -q " + sound_file)
                time_start = time.time()
                # wait for new game
                while time.time() - time_start < 5:
                    if new_game_found():
                        sound_file = "assets/audio/new_game_found.mp3"
                        os.system("mpg123 -q " + sound_file)
                soup = BeautifulSoup(r.text, 'html.parser')
                game_url = 'https://lichess.org' + soup.findAll("a", {"class": "game-row__overlay"})[0]['href']
                
                self.client.set_game(game_url)
                self.client.run_game()
                # new game
                time.sleep(5)
                self.client.new_game()
            else:
                time.sleep(1)

class LichessClient:
    ''' Main class which interacts with Lichess. Plays and recieves moves. Called
        every instance of a game. '''
    
    def __init__(self, username, url=None, log_file: str = os.path.join(os.getcwd(), 'Client_logs',str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.txt')):
        self.log_file = log_file
        self.log = ""
        self.username = username
        self._url = url
        
        self.session = HTMLSession()        
        self.engine = Engine(playing_level=DIFFICULTY)
        
        # TODO: incorporate settings for increment games and beserk games
        self.game_info = {"playing_side": None,
                          "initial_time": None} # these statistics don't change within a game
        
        self.dynamic_info = {"fens":[],
                             "self_clock_times":[],
                             "opp_clock_times":[],
                             "last_moves": []}
        
        self.last_scrape_time = 0
        self.last_scrape_dic = None
        
        self.ponder_dic = {}

    def _write_log(self):
        """ Writes down thinking into a log file for debugging. """
        with open(self.log_file,'a') as log:
            log.write(self.log)
            log.close()
        self.log = ""
    
    def get_url(self):
        if self._url is not None:
            return self._url
        else:
            raise Exception("No url has been set.")

    def set_url(self, url):
        self._url = url
        
    def _update_scrape_dic(self):
        """ Form our url, scrape the information and parse it into last_scrape_dic.
        
            Returns True if successfully scraped. Otherwise we assume the game to have ended
            and return False
        """
        self.log += "Performing scraping from the current url {} \n".format(self.get_url())
        page = self.session.get(self.get_url())
        soup = BeautifulSoup(page.text, 'html.parser')
        json_parsed = json.loads(soup("script")[-1].text)
        if "data" in json_parsed:
            self.last_scrape_dic = json_parsed["data"]
            self.log += "Updated scrape dic. Position received is {} \n".format(self.last_scrape_dic["steps"][-1]["fen"])
        else:
            self.log += "data key not found in scraped dictionary. Assumed game to have terminated. \n"
            return False
        self.last_scrape_time = time.time()
        self.log += "Last scrape time set to: {} \n".format(self.last_scrape_time)
        return True
    
    def _update_dynamic_info_from_screenshot(self, move_obj: chess.Move):
        """ The second way we can update the dynamic information, from screenshots
            and change detection. 
        """
        # update fen list
        last_board = chess.Board(self.dynamic_info["fens"][-1])
        last_board.push(move_obj)
        self.dynamic_info["fens"].append(last_board.fen())
        self.dynamic_info["fens"] = self.dynamic_info["fens"][-FEN_NO_CAP:]
        
        # Update last moves
        self.dynamic_info["last_moves"].append(move_obj.uci())
        self.dynamic_info["last_moves"] = self.dynamic_info["last_moves"][-(FEN_NO_CAP-1):]
        
        # We cannot do anything to update the clock times, (although maybe TODO)
        self.log += "Updated dynamic information from screenshot prompt. \n"
    
    def _update_dynamic_info(self):
        """ From last_scrape_dic, update the dynamic info. Overwrite any previously edited information
            when it comes to board information. We shall not over-write clock history,
            as we have no way of verifying that information.
        """
        # We only care about the last FEN_NO_CAP number of fens
        relevant_step_list = self.last_scrape_dic["steps"][-FEN_NO_CAP:]
        
        # First update the fens
        fens = [entry["fen"] for entry in relevant_step_list]
        self.dynamic_info["fens"] = fens[:]
        # Next the last moves, there should be one less than the numbre of fens
        if len(fens) >= 2:
            last_moves = [entry["uci"] for entry in relevant_step_list[1:]]
        else:
            last_moves = []
        self.dynamic_info["last_moves"] = last_moves[:]
        
        # Now update the clock times. We can only guarantee these clock times to
        # be accurate to SCRAPE_EVERY seconds.
        color_dic = {chess.WHITE: "white", chess.BLACK: "black"}
        side = color_dic[self.game_info["playing_side"]]
        opp_side = color_dic[not self.game_info["playing_side"]]
        # TODO: add clock time histories
        self.dynamic_info["self_clock_times"] = [self.last_scrape_dic["clock"][side]]
        self.dynamic_info["opp_clock_times"] = [self.last_scrape_dic["clock"][opp_side]]
        
        self.log += "Updated dynamic information. \n"
        
    
    def set_game(self, url):
        ''' Once client has found game, sets up game parameters. '''
        self.set_url(url)        
        # getting game information, including the side the player is playing and the initial time
        self._update_scrape_dic()
        
        self.game_info["initial_time"] = self.last_scrape_dic["clock"]["initial"]
        
        color_dic = {"white": chess.WHITE, "black": chess.BLACK}
        self.game_info["playing_side"] = color_dic[self.last_scrape_dic["player"]["color"]]
        # just to make sure player identity
        assert self.username == self.last_scrape_dic["player"]["user"]["username"]
        
        # TODO : include beserk options
        # berserk_search = re.findall(rf'\"username\"\:\"{self.username}\".{{110,180}}\"berserk\"\:true', page.text)
        # if len(berserk_search) > 0:
        #     print('Detected berserk! New starting time: ', self.starting_time/2)
        #     self.berserk = True
        # else:
        #     self.berserk = False
        
        # Now update the dynamic_information
        self._update_dynamic_info()
        
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
        return self.last_scrape_dic["game"]["status"]["id"] != 20 # id 20 is the only id when a game is going
    
    def scrape(self):
        if self._update_scrape_dic() == False:
            return False
        self._update_dynamic_info()
    
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
                    base_time = 0.7*QUICKNESS*initial_time/(85 + initial_time*0.25)
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
        ''' The main update step for the lichess client. We have two ways of looking
            for updates, one through screenshots, other through scraping.
            
            We do not want to scrape too much, because it is both time consuming and
            arouses suspicion. However scraping at a rate of twice per second should
            be slow enough. twice per second is not fast enough for bullet segments,
            so we should resort to scraping only when screenshots fail us. 
            
            We shall until it is our move, then we call the engine to decide what to do.
            Function returns True when it is our turn or False when the game has ended.
        '''
        while True:
            # First check if game has ended
            if self._check_game_end():
                return False
            
            # Next check whether we are entitled to a scrape
            if time.time() - self.last_scrape_time > SCRAPE_EVERY:
                successful = self.scrape()
                if successful == False:
                    return False
            
            # See if it is our turn
            if self.check_our_turn() == True:
                return True
            
            # In the meantime check for updates via screenshot method. The amount of time we
            # shall spend doing this will be enough so we can scrape again after
            tries = 0
            tries_cap = 10 # some positive number to start with
            while tries < tries_cap:
                # start_time = time.time()
                if self.game_info["playing_side"] == chess.WHITE:
                    bottom = "w"
                else:
                    bottom = "b"
                
                move_change = scrape_move_change(bottom)
                # if there has been a move change detected,
                # we need to check whether it truly corresponds to a move we can play
                # on our last recorded board
                if move_change is not None:
                    move1_uci, move2_uci = move_change
                    last_board = chess.Board(self.dynamic_info["fens"][-1])
                    move1 = chess.Move.from_uci(move1_uci)
                    move2 = chess.Move.from_uci(move2_uci)
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
    
    def make_move(self, move_uci:str, premove:str=None):
        """ Executes mouse clicks for the moves. 
            
            Returns True if clicks were made successfully, else returns False
        """
        # Check if there is human interference
        if is_capslock_on():
            self.log += "Tried to make move {} and premove {}, but failed as caps lock is on. \n ".format(move_uci, premove)
            return False
        
        # First, reset previous clicks by right-clicking the centre of the board
        centre_X, centre_Y = START_X + 3.5*STEP, START_Y + 3.5*STEP
        pyautogui.click(centre_X, centre_Y, button='right')
        
        # Now make the move
        from_x, from_y, to_x, to_y = self.find_clicks(move_uci)
        pyautogui.click(from_x, from_y, button='left')
        pyautogui.click(to_x, to_y, button='left')
        
        self.log += "Made clicks for the move {} \n".format(move_uci)
        # If there is a premove
        if premove is not None:
            from_x, from_y, to_x, to_y = self.find_clicks(premove)
            pyautogui.click(from_x, from_y, button='left')
            pyautogui.click(to_x, to_y, button='left')
            self.log += "Made clicks for the premove {} \n".format(premove)
        
        return True
    
    def resign(self):
        # can only execute if no human interference.
        if is_capslock_on():
            self.log += "Tried resign the game but failed as caps lock is on. \n "
            return False
        resign_button_x, resign_button_y =  START_X + 10.2*STEP, START_Y + 4.2*STEP
        pyautogui.click(resign_button_x, resign_button_y, button='left')
        time.sleep(0.2)
        pyautogui.click(resign_button_x, resign_button_y, button='left')
        
        return True
    
    def new_game(self, time_control="1+0"):
        # can only execute if no human interference.
        if is_capslock_on():
            self.log += "Tried to start new game with time control {} but failed as caps lock is on. \n ".format(time_control)
            return False
        
        play_button_x, play_button_y = START_X - 2.2*STEP, START_Y - 0.8*STEP
        pyautogui.click(play_button_x, play_button_y, button='left')
        time.sleep(1)
        if time_control == "1+0":
            to_x, to_y = START_X + 1.5*STEP, START_Y + 0.5*STEP
            pyautogui.click(to_x, to_y, button='left')
        
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
            click_from_x = start_x + file_fr*step
            click_from_y = start_y + (7-rank_fr)*step
            
            rank_to = chess.square_rank(to_square)
            file_to = chess.square_file(to_square)
            click_to_x = start_x + file_to*step
            click_to_y = start_y + (7-rank_to)*step
        else:
            # a1 square is top right
            rank_fr = chess.square_rank(from_square)
            file_fr = chess.square_file(from_square)
            click_from_x = start_x + (7-file_fr)*step
            click_from_y = start_y + rank_fr*step
            
            rank_to = chess.square_rank(to_square)
            file_to = chess.square_file(to_square)
            click_to_x = start_x + (7-file_to)*step
            click_to_y = start_y + rank_to*step
        return click_from_x, click_from_y, click_to_x, click_to_y
   

# URL = 'https://lichess.org/GlYCARuKJKOp'