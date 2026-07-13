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
from common.constants import QUICKNESS, MOUSE_QUICKNESS, DIFFICULTY, RESOLUTION_SCALE
from common.move_timing import (MOVE_DELAY, DRAG_MOVE_DELAY, CLICK_MOVE_DELAY,
                                movement_duration, drag_settle_sleep,
                                click_settle_sleep, drag_probability,
                                ponder_response_wait, scramble_response_wait,
                                resign_pause)
from common.utils import patch_fens, check_safe_premove, scramble_fire_veto, scraped_fen_sanity_issues, InvalidPositionError
from common.logging import get_logger, LogLevel, LegacyLoggerAdapter

from chessimage.image_scrape_utils import (SCREEN_CAPTURE, START_X, START_Y, STEP, capture_board, capture_top_clock,
                                           capture_bottom_clock, capture_all_regions, get_fen_from_image, check_fen_last_move_bottom,
                                           read_clock, find_initial_side, detect_last_move_from_img, check_turn_from_last_moved,
                                           capture_result, compare_result_images, capture_rating,
                                           capture_white_notation)

# Import dynamic button detection
try:
    from auto_calibration.button_detector import (
        find_play_button, find_time_control_button, find_new_opponent_button,
        ButtonDetector, QuickPairingDetector
    )
    DYNAMIC_BUTTON_DETECTION = True
except ImportError:
    DYNAMIC_BUTTON_DETECTION = False
    print("⚠️  Dynamic button detection not available, using hardcoded positions")

# Import calibration config for resign button position
try:
    from auto_calibration.config import get_config
    CALIBRATION_CONFIG_AVAILABLE = True
except ImportError:
    CALIBRATION_CONFIG_AVAILABLE = False
    print("⚠️  Calibration config not available, using hardcoded resign button position")

# import threading
# from multiprocessing import Process, Manager

# pyautogui sleeps PAUSE after every mouseDown/mouseUp/click - at the default
# 0.1s that adds ~0.2s of dead time per move on top of the actual gestures
pyautogui.PAUSE = 0.02


def save_debug_screenshot(prefix: str, board_img=None, clock_imgs=None, extra_info=None):
    """
    Save comprehensive debug screenshots when errors occur.
    
    Uses the unified logging system's error directory when available,
    falls back to Error_files/ for backwards compatibility.
    
    Args:
        prefix: Error type prefix (e.g. "linking_move_error", "turn_detection_error")
        board_img: The board image if available
        clock_imgs: Dict with 'top' and 'bottom' clock images if available
        extra_info: Dict with additional info to save (fens, positions, etc.)
    
    Returns:
        List of saved filenames for logging
    """
    logger = get_logger()
    saved_files = []
    
    # Use new logging system if available, otherwise fallback to legacy
    if logger is not None:
        # Save using unified logging system
        try:
            full_screen = SCREEN_CAPTURE.capture()
            if full_screen is not None:
                path = logger.save_error_image(f"{prefix}_fullscreen", full_screen[:,:,:3])
                if path:
                    saved_files.append(str(path))
        except Exception:
            pass
        
        if board_img is not None:
            path = logger.save_error_image(f"{prefix}_board", board_img.astype(np.uint8))
            if path:
                saved_files.append(str(path))
        
        if clock_imgs is not None:
            if 'top' in clock_imgs and clock_imgs['top'] is not None:
                path = logger.save_error_image(f"{prefix}_top_clock", clock_imgs['top'])
                if path:
                    saved_files.append(str(path))
            if 'bottom' in clock_imgs and clock_imgs['bottom'] is not None:
                path = logger.save_error_image(f"{prefix}_bottom_clock", clock_imgs['bottom'])
                if path:
                    saved_files.append(str(path))
        
        if extra_info is not None:
            path = logger.save_error_context(prefix, extra_info)
            saved_files.append(str(path))
    else:
        # Fallback to legacy Error_files directory
        timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
        base_name = f"{prefix}_{timestamp}"
        
        os.makedirs("Error_files", exist_ok=True)
        
        try:
            full_screen = SCREEN_CAPTURE.capture()
            if full_screen is not None:
                full_path = os.path.join("Error_files", f"{base_name}_fullscreen.png")
                cv2.imwrite(full_path, full_screen[:,:,:3])
                saved_files.append(full_path)
        except Exception:
            pass
        
        if board_img is not None:
            board_path = os.path.join("Error_files", f"{base_name}_board.png")
            cv2.imwrite(board_path, board_img.astype(np.uint8))
            saved_files.append(board_path)
        
        if clock_imgs is not None:
            if 'top' in clock_imgs and clock_imgs['top'] is not None:
                top_path = os.path.join("Error_files", f"{base_name}_top_clock.png")
                cv2.imwrite(top_path, clock_imgs['top'])
                saved_files.append(top_path)
            if 'bottom' in clock_imgs and clock_imgs['bottom'] is not None:
                bot_path = os.path.join("Error_files", f"{base_name}_bottom_clock.png")
                cv2.imwrite(bot_path, clock_imgs['bottom'])
                saved_files.append(bot_path)
        
        if extra_info is not None:
            info_path = os.path.join("Error_files", f"{base_name}_info.txt")
            with open(info_path, 'w') as f:
                for key, value in extra_info.items():
                    f.write(f"{key}: {value}\n")
            saved_files.append(info_path)
    
    return saved_files


# global variables
FEN_NO_CAP = 8 # the max number of successive fens e store from the most recent position
SCRAPE_EVERY = 0.5 # the time gap between scraping
# MOVE_DELAY / DRAG_MOVE_DELAY / CLICK_MOVE_DELAY now live in
# common/move_timing.py, shared with the offline simulator.

CURSOR = CustomCursor() # mouse control object which simulates human-like movement with mouse

# Logging - uses unified logging system via legacy adapter for backwards compatibility
# The actual log file is managed by SessionLogger (initialised in main.py)
LOG = LegacyLoggerAdapter(channel="client")
     
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

# Performance tracking
SCAN_TIMES = []  # Track recent scan times for performance monitoring
SCAN_LOG_INTERVAL = 10  # Log performance summary every N scans

# Move timing tracking - for measuring realised move time vs executed move time
MOVE_TIMING = {
    "move_decision_time": None,      # When we decided to make the move
    "move_execution_end_time": None, # When mouse execution finished
    "clock_before_move": None,       # Our clock time before the move
    "move_uci": None,                # The move we made
    "mouse_time_ms": None,           # Time spent on mouse movement
    "waiting_for_clock": False       # Whether we're waiting to detect clock change
}

# Guard against acting twice off stale vision. Set when a move's clicks are
# successfully executed, cleared whenever a scan is adopted into
# DYNAMIC_INFO (even one showing an unchanged board - that is positive
# evidence the move never registered, so a retry is then correct). While
# set, check_our_turn() may not trigger another move: every duplicate
# move/premove observed in the wild came from acting on a fen frozen by
# discarded mid-animation scans.
AWAITING_FRESH_SCAN = False

# Below this much time on our own clock, skip the confirmation re-capture
# for unlinkable scans and act on the first reading - a human under time
# pressure is also prone to misreading the board.
RESYNC_CONFIRM_MIN_TIME = 15



#linux funciton to check capslock
def is_capslock_on():
    if subprocess.check_output('xset q | grep LED', shell=True)[65] == 48 :
        return False
    elif subprocess.check_output('xset q | grep LED', shell=True)[65] == 49 :
        return True
    

def _movement_duration(distance):
    """One mouse-leg duration — formula shared with the simulator via
    common/move_timing.py."""
    return movement_duration(distance, MOUSE_QUICKNESS, RESOLUTION_SCALE)


def drag_mouse(from_x, from_y, to_x, to_y, tolerance=0):
    """ Make human drag and drop move with human mouse speed and randomness in mind.
        Uses optimised Bezier curves that are faster at higher resolutions.
        
        Returns True if move was made successfully, else if mouse slip was made return False.
    """
    global LOG
    
    # 1 in 100 moves, we simulate a potential mouse slip
    successful = True
    if np.random.random() < 0.03:
        # Mouse slip mode: allow larger offset but still keep within safe zone
        slip_tolerance = tolerance * 1.5
        offset_x = np.clip(np.random.randn()*slip_tolerance, - STEP/3, STEP/3)
        offset_y = np.clip(np.random.randn()*slip_tolerance, - STEP/3, STEP/3)
        if np.abs(offset_x) > STEP/3.5 or np.abs(offset_y) > STEP/3.5:
            successful = False
    else:
        # Normal mode: keep clicks well within centre of square
        # Use tighter clip to ensure we stay in safe zone (~30% of square from centre)
        offset_x = np.clip(np.random.randn()*tolerance, - STEP/4, STEP/4)
        offset_y = np.clip(np.random.randn()*tolerance, - STEP/4, STEP/4)
    
    # From position: small random offset to appear human
    new_from_x = from_x + tolerance * 0.5 * (np.random.random() - 0.5)
    new_from_y = from_y + tolerance * 0.5 * (np.random.random() - 0.5)
    new_to_x = to_x + offset_x
    new_to_y = to_y + offset_y
    
    current_x, current_y = pyautogui.position()
    from_distance = np.sqrt((new_from_x - current_x)**2 + (new_from_y - current_y)**2)
    to_distance = np.sqrt((new_from_x - new_to_x)**2 + (new_from_y - new_to_y)**2)
    
    duration_from = _movement_duration(from_distance)
    duration_to = _movement_duration(to_distance)
    
    drag_start = time.time()
    
    # Use quick_move_to for faster human-like curves
    CURSOR.quick_move_to([new_from_x, new_from_y], duration=duration_from, resolution_scale=RESOLUTION_SCALE)
    
    # Tiny pause to ensure cursor has settled before picking up piece
    time.sleep(drag_settle_sleep())
    
    pyautogui.mouseDown()
    CURSOR.quick_move_to([new_to_x, new_to_y], duration=duration_to, resolution_scale=RESOLUTION_SCALE)
    pyautogui.mouseUp()
    
    actual_duration = (time.time() - drag_start) * 1000
    
    planned_ms = (duration_from + duration_to) * 1000
    LOG += f"[PERF] Drag: planned={planned_ms:.0f}ms, actual={actual_duration:.0f}ms, dist={from_distance:.0f}+{to_distance:.0f}px\n"

    return successful

def click_to_from_mouse(from_x, from_y, to_x, to_y, tolerance=0):
    """ Exactly the same as drag mouse, but sometimes we mix it up by clicking two squares
        rather than click and drag for variation. Uses optimised human-like Bezier curves.
        
        Returns True if move was made successfully, else False if mouse slip was made.    
    """
    global LOG
    
    # 1 in 100 moves, we simulate a potential mouse slip
    successful = True
    if np.random.random() < 0.03:
        # Mouse slip mode: allow larger offset but still keep within safe zone
        slip_tolerance = tolerance * 1.5
        offset_x = np.clip(np.random.randn()*slip_tolerance, - STEP/3, STEP/3)
        offset_y = np.clip(np.random.randn()*slip_tolerance, - STEP/3, STEP/3)
        if np.abs(offset_x) > STEP/3.5 or np.abs(offset_y) > STEP/3.5:
            successful = False
    else:
        # Normal mode: keep clicks well within centre of square
        # Use tighter clip to ensure we stay in safe zone (~25% of square from centre)
        offset_x = np.clip(np.random.randn()*tolerance, - STEP/4, STEP/4)
        offset_y = np.clip(np.random.randn()*tolerance, - STEP/4, STEP/4)
    
    # From position: small random offset to appear human
    new_from_x = from_x + tolerance * 0.5 * (np.random.random() - 0.5)
    new_from_y = from_y + tolerance * 0.5 * (np.random.random() - 0.5)
    new_to_x = to_x + offset_x
    new_to_y = to_y + offset_y
    
    current_x, current_y = pyautogui.position()
    from_distance = np.sqrt((new_from_x - current_x)**2 + (new_from_y - current_y)**2)
    to_distance = np.sqrt((new_from_x - new_to_x)**2 + (new_from_y - new_to_y)**2)
    
    duration_from = _movement_duration(from_distance)
    duration_to = _movement_duration(to_distance)
    
    click_start = time.time()
    
    # Use quick_move_to for faster human-like curves
    CURSOR.quick_move_to([new_from_x, new_from_y], duration=duration_from, resolution_scale=RESOLUTION_SCALE)
    pyautogui.click(button="left")
    
    # Small delay after first click to let Lichess register piece selection
    time.sleep(click_settle_sleep())
    
    CURSOR.quick_move_to([new_to_x, new_to_y], duration=duration_to, resolution_scale=RESOLUTION_SCALE)
    pyautogui.click(button="left")
    
    actual_duration = (time.time() - click_start) * 1000
    
    planned_ms = (duration_from + duration_to) * 1000
    LOG += f"[PERF] Click: planned={planned_ms:.0f}ms, actual={actual_duration:.0f}ms, dist={from_distance:.0f}+{to_distance:.0f}px\n"
    
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

_START_LIKE_BOARD_FENS = None

def _start_like_board_fens():
    """
    Board placements that can be on screen when a new game is found: the
    starting position, or one white move into it - as black the opponent
    often moves (or premoves) before our first scan.
    """
    global _START_LIKE_BOARD_FENS
    if _START_LIKE_BOARD_FENS is None:
        fens = {chess.STARTING_BOARD_FEN}
        base = chess.Board()
        for move in list(base.legal_moves):
            base.push(move)
            fens.add(base.board_fen())
            base.pop()
        _START_LIKE_BOARD_FENS = fens
    return _START_LIKE_BOARD_FENS


def new_game_found(expected_time=None):
    """ Uses screenshot to detect whether we have started new game.

    Args:
        expected_time: Optional expected initial time in seconds to validate against.

    Returns None if not, else returns our starting initial time in seconds.
    """
    # try to read bot clock for start position. if none is found, then haven't started the game
    for state in ["start1", "start2"]:
        res, details = read_clock(capture_bottom_clock(state=state), return_details=True)
        if res is not None:
            # Fix 4: Vertical offset awareness
            # In 'start' states, the digits should be roughly vertically centered in the crop.
            # If they are significantly shifted, it might be the 'play' clock being seen.
            v_center = details.get('v_center', 0)
            orig_h = details.get('original_height', 66)
            v_error = abs(v_center - orig_h / 2)
            
            if v_error > orig_h * 0.1: # Tightened to 10% off-center
                continue
                
            # Fix 1: Validate against expected time
            if expected_time is not None:
                # Accept if within 10% of expected time, and NOT 0
                if res == 0 or abs(res - expected_time) > max(10, expected_time * 0.1):
                    continue
            elif res == 0:
                # Always ignore 0 as a starting time
                continue
            
            # Fix 2: Starting Board Verification (The "Strict" Check)
            # If we found a valid clock, verify the board is at (or one white
            # move into) the starting position: as black the opponent may
            # already have moved before we scan
            board_img = capture_board()
            # Side is not known yet; try 'w' first as it's most common
            # (find_initial_side will be called properly in set_game)
            try:
                start_like = _start_like_board_fens()
                test_fen = get_fen_from_image(board_img, bottom="w", fast_mode=True)
                if chess.Board(test_fen).board_fen() not in start_like:
                    # Could be we are playing as black, try other orientation
                    test_fen_b = get_fen_from_image(board_img, bottom="b", fast_mode=True)
                    if chess.Board(test_fen_b).board_fen() not in start_like:
                        continue # Not a start-like board in either orientation
            except:
                continue
                
            return res
            
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

def await_new_game(timeout=60, expected_time=None):
    global LOG
    time_start = time.time()
    while time.time() - time_start < timeout:
        res = new_game_found(expected_time=expected_time)
        if res is not None:
            sound_file = "assets/audio/new_game_found.mp3"
            os.system("mpg123 -q " + sound_file)
            return res

    debug_files = save_debug_screenshot(
        "new_game_timeout",
        extra_info={'timeout': timeout, 'expected_time': expected_time})
    LOG += "ERROR: No new game found within {}s (expected_time={}). Debug files: {}. \n".format(
        timeout, expected_time, debug_files)

    sound_file = "assets/audio/alert.mp3"
    os.system("mpg123 -q " + sound_file)
    return None

def set_game(starting_time):
    ''' Once client has found game, sets up game parameters. '''
    global HOVER_SQUARE, GAME_INFO, LOG, CASTLING_RIGHTS_FEN, DYNAMIC_INFO, PONDER_DIC, AWAITING_FRESH_SCAN

    # resetting hover square
    HOVER_SQUARE = None
    AWAITING_FRESH_SCAN = False
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
    # The board can still be rendering right after the game is found, so
    # retry for a few seconds before giving up
    side = find_initial_side()
    side_detect_deadline = time.time() + 3.0
    while side is None and time.time() < side_detect_deadline:
        time.sleep(0.25)
        side = find_initial_side()
    if side is None:
        debug_files = save_debug_screenshot(
            "side_detection_failure", board_img=board_img,
            extra_info={'starting_time': starting_time})
        LOG += "ERROR: Could not detect playing side after retries. Board might be obscured or setup failed. Aborting game setup. Debug files: {}. \n".format(debug_files)
        write_log()
        return False

    # Re-capture the board: the first capture may predate the board render
    # (and as black the opponent may already have moved)
    board_img = capture_board()

    GAME_INFO["playing_side"] = side
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
            debug_files = save_debug_screenshot(
                "setup_bottom_check_error", board_img=board_img,
                extra_info={'starting_fen': starting_fen, 'bottom': bottom})
            LOG += "ERROR: Not corrected. Continuingly anyway. Debug files: {}. \n".format(debug_files)
        
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
    return True

def write_log():
    """ Writes buffered log messages to the log file. """
    global LOG
    LOG.write()

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
    global DYNAMIC_INFO, LOG, GAME_INFO, AWAITING_FRESH_SCAN
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
                debug_files = save_debug_screenshot(
                    "opp_clock_move_change_error", clock_imgs={'top': top_clock_img})
                LOG += "ERROR: Could not find the opponent clock time from move change update. Debug files: {}. \n".format(debug_files)
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
                debug_files = save_debug_screenshot(
                    "own_clock_move_change_error", clock_imgs={'bottom': bot_clock_img})
                LOG += "ERROR: Could not find own clock time from move change update. Debug files: {}. \n".format(debug_files)
            else:
                DYNAMIC_INFO["self_clock_times"].append(self_clock_time)
                DYNAMIC_INFO["self_clock_times"] = DYNAMIC_INFO["self_clock_times"][-FEN_NO_CAP:]
        else:
            DYNAMIC_INFO["self_clock_times"].append(self_clock_time)
            DYNAMIC_INFO["self_clock_times"] = DYNAMIC_INFO["self_clock_times"][-FEN_NO_CAP:]
    LOG += "Updated dynamic information from move-change screenshot prompt: \n"
    LOG += "{} \n".format(DYNAMIC_INFO)
    # A move was adopted - the move loop may act on the position again.
    AWAITING_FRESH_SCAN = False
    
def _under_time_pressure():
    """ Whether our own clock is low enough that we accept first readings
        rather than spend time double-checking the board. """
    times = DYNAMIC_INFO["self_clock_times"]
    return bool(times) and times[-1] < RESYNC_CONFIRM_MIN_TIME


def _confirm_board_stable(board_fen, bottom, delay=0.15):
    """ Re-capture the board after a short delay and check the same piece
        placement is still on screen.

        A frame that fails move-linking or turn detection is either a
        transient mid-animation misread or a genuine resync after missed
        moves. A misread never survives two captures this far apart; a
        settled board does. Returns True if the placement reproduced.
    """
    time.sleep(delay)
    try:
        board_img = capture_board()
        fen_now = get_fen_from_image(board_img, bottom=bottom)
    except Exception:
        return False
    if scraped_fen_sanity_issues(fen_now):
        return False
    return chess.Board(fen_now).board_fen() == board_fen


def _link_candidates_for_unreadable_turn(fen_before, scraped_fen):
    """ Rank both turn readings of an unreadable-turn scrape by link length.

        The scraped fen's turn field is meaningless (the scraper always
        claims white to move), so try linking the last tracked position to
        the scraped placement under both possible turns and return the
        successful (ply_count, fen) candidates shortest-first. Fewest plies
        wins: patch_fens can "confirm" the wrong turn with a piece-shuffle
        line, and accepting the longer link hands the move to the wrong
        side (seen live: f8g7,e2e4 misread as f8h6,e2e4,h6g7, adopting the
        opponent's turn and flagging a won game). Parity means the two
        candidates can never tie.
    """
    candidates = []
    for turn in (chess.WHITE, chess.BLACK):
        dummy_board = chess.Board(scraped_fen)
        dummy_board.turn = turn
        candidate_fen = dummy_board.fen()
        res = patch_fens(fen_before, candidate_fen)
        if res is not None:
            candidates.append((len(res[0]), candidate_fen))
    candidates.sort(key=lambda c: c[0])
    return candidates


def update_dynamic_info_from_fullimage():
    """ Scrape image information from screenshot and update info dic. """
    global LOG, DYNAMIC_INFO, GAME_INFO, MOVE_TIMING, AWAITING_FRESH_SCAN
    
    scan_start = time.time()
    
    # Use single-capture optimization: one screenshot, crop all regions
    board_img, top_clock_img, bot_clock_img = capture_all_regions()
    capture_time = time.time()

    bottom = "w" if GAME_INFO["playing_side"] == chess.WHITE else "b"

    our_time = read_clock(bot_clock_img)        
    opp_time = read_clock(top_clock_img)
    clock_time = time.time()
    
    fen = get_fen_from_image(board_img, bottom=bottom) # assumes white turn
    fen_time = time.time()

    # Reject structurally impossible scrapes (e.g. a king hidden behind a
    # capture animation): adopting one poisons the fen history and a
    # king-less position segfaults stockfish downstream. Skip this scan
    # and let the next one see the settled board.
    sanity_issues = scraped_fen_sanity_issues(fen)
    if sanity_issues:
        debug_files = save_debug_screenshot(
            "insane_scraped_fen",
            board_img=board_img,
            extra_info={
                'detected_fen': fen,
                'sanity_issues': sanity_issues,
                'previous_fens': DYNAMIC_INFO["fens"][-3:],
            }
        )
        LOG += "ERROR: Scraped fen {} is structurally impossible ({}), discarding this scan. Debug files: {}. \n".format(
            fen, sanity_issues, debug_files)
        return

    # now check the turn
    check_turn_res = check_turn_from_last_moved(fen, board_img, bottom)
    turn_time = time.time()
    
    # Track and log timing stats
    global SCAN_TIMES
    total_scan_ms = (turn_time - scan_start) * 1000
    SCAN_TIMES.append(total_scan_ms)
    
    # Keep only last 50 scan times
    if len(SCAN_TIMES) > 50:
        SCAN_TIMES = SCAN_TIMES[-50:]
    
    # Log performance summary periodically
    if len(SCAN_TIMES) % SCAN_LOG_INTERVAL == 0:
        avg_ms = sum(SCAN_TIMES[-SCAN_LOG_INTERVAL:]) / SCAN_LOG_INTERVAL
        LOG += f"[PERF] Avg scan time (last {SCAN_LOG_INTERVAL}): {avg_ms:.0f}ms, scans/sec: {1000/avg_ms:.1f}\n"
    
    # Log individual slow scans
    if total_scan_ms > 80:  # Log if scan took longer than expected
        LOG += f"[PERF] Slow scan: {total_scan_ms:.0f}ms (capture:{(capture_time-scan_start)*1000:.0f}, clock:{(clock_time-capture_time)*1000:.0f}, fen:{(fen_time-clock_time)*1000:.0f}, turn:{(turn_time-fen_time)*1000:.0f})\n"    
    
    if check_turn_res is None:
        # then there was error, save comprehensive debug files
        debug_files = save_debug_screenshot(
            "turn_detection_error",
            board_img=board_img,
            clock_imgs={'top': top_clock_img, 'bottom': bot_clock_img},
            extra_info={
                'detected_fen': fen,
                'bottom_side': bottom,
                'previous_fens': DYNAMIC_INFO["fens"][-3:] if len(DYNAMIC_INFO["fens"]) >= 3 else DYNAMIC_INFO["fens"],
                'last_moves': DYNAMIC_INFO["last_moves"][-3:] if len(DYNAMIC_INFO["last_moves"]) >= 3 else DYNAMIC_INFO["last_moves"],
                'opp_time_read': opp_time,
                'our_time_read': our_time,
            }
        )
        LOG += "ERROR: Couldn't find turn from fen {} with bottom {}. Debug files saved: {}. Trying to work out from last fen.\n".format(fen, bottom, debug_files)
        
        fen_before = DYNAMIC_INFO["fens"][-1]
        last_board = chess.Board(fen_before)
        candidates = _link_candidates_for_unreadable_turn(fen_before, fen)
        if candidates:
            n_plies, fen_after = candidates[0]
            if len(candidates) > 1:
                LOG += "Both turn readings link from the last fen; adopting the shorter ({} plies): {}. \n".format(n_plies, fen_after)
        else:
            dummy_board = chess.Board(fen)
            dummy_board.turn = last_board.turn
            fen_after = dummy_board.fen()
            # An unlinkable board with an unreadable turn is either a
            # transient animation frame or a genuine resync; only a
            # settled board survives a second capture. Guessing the
            # turn off a transient frame is how moves get played out
            # of turn, so confirm before adopting (unless low on time,
            # where we accept the misread risk to move quickly).
            if not _under_time_pressure() and not _confirm_board_stable(dummy_board.board_fen(), bottom):
                LOG += "Unlinkable board with unreadable turn did not survive a confirmation re-capture, discarding this scan as a transient frame. \n"
                return
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
        # This is still an adopted scan: an unchanged board after we made
        # clicks is positive evidence the move never registered, so allow
        # the move loop to act (retry) again.
        AWAITING_FRESH_SCAN = False
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
            # Unlinkable frames are usually mid-animation misreads, not
            # genuine missed-move resyncs; wiping history off one poisons
            # the fen state (and downstream, the engine input). Adopt the
            # resync only if the board reproduces on a second capture -
            # except under time pressure, where we act on first readings.
            if not _under_time_pressure() and not _confirm_board_stable(chess.Board(now_fen).board_fen(), bottom):
                del DYNAMIC_INFO["fens"][-1]
                LOG += "ERROR: Couldn't find linking move between fens {} and {}, and the new board did not survive a confirmation re-capture. Discarding this scan as a transient frame. \n".format(prev_fen, now_fen)
                return
            # Save comprehensive debug info for linking move errors
            debug_files = save_debug_screenshot(
                "linking_move_error",
                board_img=board_img,
                clock_imgs={'top': top_clock_img, 'bottom': bot_clock_img},
                extra_info={
                    'prev_fen': prev_fen,
                    'now_fen': now_fen,
                    'all_tracked_fens': DYNAMIC_INFO["fens"],
                    'last_moves_before_error': DYNAMIC_INFO["last_moves"],
                    'playing_side': str(GAME_INFO.get("playing_side", "unknown")),
                    'castling_rights': CASTLING_RIGHTS_FEN,
                }
            )
            LOG += "ERROR: Couldn't find linking move between fens {} and {}. Debug files saved: {}. Defaulting to singular fen history and wiping last_move history. \n".format(prev_fen, now_fen, debug_files)
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
                # Save comprehensive debug files for clock reading error
                debug_files = save_debug_screenshot(
                    "opp_clock_read_error",
                    board_img=board_img,
                    clock_imgs={'top': top_clock_img, 'bottom': bot_clock_img},
                    extra_info={
                        'current_fen': fen if 'fen' in dir() else 'not available',
                        'last_known_opp_time': DYNAMIC_INFO["opp_clock_times"][-1] if DYNAMIC_INFO["opp_clock_times"] else None,
                    }
                )
                LOG += "ERROR: Couldn't read opponent time, defaulting to last known clock time. Debug files: {}. \n".format(debug_files)
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
                # Save comprehensive debug files for clock reading error
                debug_files = save_debug_screenshot(
                    "own_clock_read_error",
                    board_img=board_img,
                    clock_imgs={'top': top_clock_img, 'bottom': bot_clock_img},
                    extra_info={
                        'current_fen': fen if 'fen' in dir() else 'not available',
                        'last_known_own_time': DYNAMIC_INFO["self_clock_times"][-1] if DYNAMIC_INFO["self_clock_times"] else None,
                    }
                )
                LOG += "ERROR: Couldn't read our own time, defaulting to last known clock time. Debug files: {}. \n".format(debug_files)
                our_time = DYNAMIC_INFO["self_clock_times"][-1]                
        DYNAMIC_INFO["self_clock_times"].append(our_time)
        DYNAMIC_INFO["self_clock_times"] = DYNAMIC_INFO["self_clock_times"][-FEN_NO_CAP:]
        
        # Check for realised move time (clock changed after our move)
        if MOVE_TIMING["waiting_for_clock"] and MOVE_TIMING["clock_before_move"] is not None:
            if our_time != MOVE_TIMING["clock_before_move"]:
                # Clock has changed - calculate realised move time
                realised_time_ms = (time.time() - MOVE_TIMING["move_decision_time"]) * 1000
                mouse_time_ms = MOVE_TIMING["mouse_time_ms"] or 0
                scan_overhead_ms = realised_time_ms - mouse_time_ms
                clock_diff = MOVE_TIMING["clock_before_move"] - our_time
                
                LOG += f"[PERF] REALISED MOVE TIME for {MOVE_TIMING['move_uci']}: {realised_time_ms:.0f}ms total\n"
                LOG += f"[PERF]   - Mouse execution: {mouse_time_ms:.0f}ms\n"
                LOG += f"[PERF]   - Detection/overhead: {scan_overhead_ms:.0f}ms\n"
                LOG += f"[PERF]   - Clock changed by: {clock_diff}s (server latency included)\n"
                
                # Reset timing state
                MOVE_TIMING["waiting_for_clock"] = False
                MOVE_TIMING["move_decision_time"] = None
                MOVE_TIMING["clock_before_move"] = None
        
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
    # A scan was adopted - the move loop may act on the position again.
    AWAITING_FRESH_SCAN = False
    


def check_our_turn():
    """ Check our dynamic information dictionary to see if it is currently our turn. """
    last_fen = DYNAMIC_INFO["fens"][-1]
    playing_side = GAME_INFO["playing_side"]
    board = chess.Board(last_fen)
    
    return board.turn == playing_side

_RESULT_REFERENCES = None

def _get_result_references():
    """
    Result reference images for game-end detection, as (image, threshold) pairs.

    Prefers templates extracted for the active calibration profile, matched
    strictly. Falls back to the legacy chessimage/ references at the historic
    looser threshold - those may come from a different screen layout.
    """
    global _RESULT_REFERENCES
    if _RESULT_REFERENCES is not None:
        return _RESULT_REFERENCES
    refs = []
    try:
        from auto_calibration.template_extractor import TemplateExtractor
        template_dir = get_config().get_template_dir()
        profile_templates = TemplateExtractor(template_dir=str(template_dir)).load_result_templates()
        for ref in (profile_templates or {}).values():
            if ref is not None:
                refs.append((ref, 0.8))
    except Exception:
        pass
    if not refs:
        for name in ("blackwin_result", "whitewin_result", "draw_result"):
            ref = cv2.imread(f"chessimage/{name}.png")
            if ref is not None:
                refs.append((ref, 0.70))
    _RESULT_REFERENCES = refs
    return refs


_GAME_OVER_MESSAGE_REFERENCES = None
GAME_OVER_MESSAGE_MATCH_THRESHOLD = 0.75

def _get_game_over_message_references():
    """
    Templates of game-over messages that appear WITHOUT a result box, as a
    list of (name, image) pairs: "... aborted the game", "... didn't move".
    """
    global _GAME_OVER_MESSAGE_REFERENCES
    if _GAME_OVER_MESSAGE_REFERENCES is None:
        try:
            from auto_calibration.template_extractor import TemplateExtractor
            template_dir = get_config().get_template_dir()
            templates = TemplateExtractor(template_dir=str(template_dir)).load_game_over_message_templates()
            _GAME_OVER_MESSAGE_REFERENCES = list(templates.items())
        except Exception:
            _GAME_OVER_MESSAGE_REFERENCES = []
    return _GAME_OVER_MESSAGE_REFERENCES


def game_over_message_found():
    """
    Detect a game that ended without a result box, via its notation-panel
    message: "White/Black aborted the game" or "White/Black didn't move".

    These endings leave none of the usual game-end signals: the board is
    still start-like (so no board outcome, and the clock fallback is
    guarded off) and there is no result box for the result templates to
    match - only an italic message. The panel is more compact than after
    a normal game (zero or one move in the list) and shifts with layout,
    so the message is template-searched anywhere within the notation
    region rather than compared at a fixed spot. Templates deliberately
    exclude the leading colour word.

    Returns the matched message name (truthy) or None.
    """
    refs = _get_game_over_message_references()
    if not refs:
        return None
    try:
        region = capture_white_notation()
        if region is None or region.size == 0:
            return None
        region_gray = cv2.cvtColor(np.ascontiguousarray(region), cv2.COLOR_BGR2GRAY)
        for name, ref in refs:
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            if (region_gray.shape[0] < ref_gray.shape[0]
                    or region_gray.shape[1] < ref_gray.shape[1]):
                continue
            score = cv2.matchTemplate(region_gray, ref_gray, cv2.TM_CCOEFF_NORMED).max()
            if score > GAME_OVER_MESSAGE_MATCH_THRESHOLD:
                return name
        return None
    except Exception:
        return None


def check_game_end(arena=False):
    """
    Check whether the game has ended.
    
    Uses multiple signals for robustness:
    1. Board outcome (checkmate, stalemate, etc.) - most reliable
    2. Result image comparison - uses calibrated coordinates
    3. Clock position check - fallback using end-state clock positions
    4. Clock unreadable at play position - indicates UI state change
    """
    global LOG

    # Method 1: Check via board outcome (checkmate/stalemate)
    if len(DYNAMIC_INFO["fens"]) > 0:
        board = chess.Board(DYNAMIC_INFO["fens"][-1])
        if board.outcome() is not None:
            LOG += "Game end detected via board outcome {} on fen {}. \n".format(
                board.outcome().termination, DYNAMIC_INFO["fens"][-1])
            return True

    # Method 2: Check via result image comparison
    # Uses calibrated coordinates from auto-calibration config
    try:
        result_img = capture_result(arena=arena)
        if result_img is not None and result_img.size > 0:
            for ref, threshold in _get_result_references():
                score = compare_result_images(result_img, ref)
                if score > threshold:
                    debug_files = save_debug_screenshot(
                        "game_end_result_match",
                        extra_info={'score': score, 'threshold': threshold,
                                    'last_fen': DYNAMIC_INFO["fens"][-1] if DYNAMIC_INFO["fens"] else None})
                    LOG += "Game end detected via result image match (score {:.2f} > {}). Debug files: {}. \n".format(
                        score, threshold, debug_files)
                    return True
    except Exception as e:
        # Don't fail the game check if result image comparison fails
        pass

    # Method 2b: Aborted / didn't-move endings show no result box at all -
    # match their message in the notation panel instead
    message = game_over_message_found()
    if message:
        debug_files = save_debug_screenshot(
            "game_end_message_{}".format(message),
            extra_info={'message': message,
                        'last_fen': DYNAMIC_INFO["fens"][-1] if DYNAMIC_INFO["fens"] else None})
        LOG += "Game end detected via game-over message match ({}). Debug files: {}. \n".format(
            message, debug_files)
        return True

    # Method 3: Fallback - check if clock is readable at end positions but NOT at play position
    # This catches cases where the UI has changed due to game ending
    # Guard: at game start ("play the first move" state) the clock also sits
    # away from the play position and can bleed into an end-state region, so
    # a start-position board is never treated as a game end here.
    try:
        start_placement = chess.STARTING_FEN.split()[0]
        board_start_like = (len(DYNAMIC_INFO["fens"]) > 0 and
                            DYNAMIC_INFO["fens"][-1].split()[0] == start_placement)
        if not board_start_like:
            # First check if we can read the clock at play position
            play_clock = read_clock(capture_bottom_clock(state="play"))

            # If we CAN'T read the clock at play position, the game might have ended
            # (UI overlay blocking the clock area)
            if play_clock is None:
                # Try reading at end positions to confirm
                if game_over_found():
                    debug_files = save_debug_screenshot(
                        "game_end_clock_fallback",
                        extra_info={'last_fen': DYNAMIC_INFO["fens"][-1] if DYNAMIC_INFO["fens"] else None})
                    LOG += "Game end detected via clock fallback: play clock unreadable, end-state clock readable. Debug files: {}. \n".format(
                        debug_files)
                    return True
    except Exception as e:
        pass

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
        
        # See if it is our turn. While AWAITING_FRESH_SCAN is set we have
        # already clicked a move off this position and no scan has been
        # adopted since - the fen is stale (frozen by discarded frames),
        # and returning True here would re-issue the same clicks. Keep
        # scanning: a valid scan clears the flag whether it shows the
        # advanced position (not our turn) or an unchanged one (retry).
        if check_our_turn() == True and not AWAITING_FRESH_SCAN:
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

def _verify_move_registered(move_uci: str, timeout: float = 0.5):
    """
    Poll the board until the moved piece has left its origin square.

    Scanning immediately after the clicks races the board render: a stale
    scan makes the client re-issue the same move, which grabs an
    already-empty square and cancels any queued premove. Returns True once
    the origin square no longer holds our piece; False if that never
    happens within the timeout (the move likely did not register, e.g. the
    game is already over or the drop was rejected).
    """
    bottom = "w" if GAME_INFO["playing_side"] == chess.WHITE else "b"
    from_square = chess.parse_square(move_uci[:2])
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            board_img = capture_board()
            fen_now = get_fen_from_image(board_img, bottom=bottom, fast_mode=True)
            piece = chess.Board(fen_now).piece_at(from_square)
            if piece is None or piece.color != GAME_INFO["playing_side"]:
                return True
        except Exception:
            pass
        time.sleep(0.04)
    return False


def make_move(move_uci:str, premove:str=None):
    """ Executes mouse clicks for the moves.

        Returns True if clicks were made successfully, else returns False
    """
    global LOG, HOVER_SQUARE, MOVE_TIMING, AWAITING_FRESH_SCAN

    move_start_time = time.time()
    
    # Record timing for realised move comparison
    MOVE_TIMING["move_decision_time"] = move_start_time
    MOVE_TIMING["move_uci"] = move_uci
    if len(DYNAMIC_INFO["self_clock_times"]) > 0:
        MOVE_TIMING["clock_before_move"] = DYNAMIC_INFO["self_clock_times"][-1]
    else:
        MOVE_TIMING["clock_before_move"] = None
    
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
    prob = drag_probability(own_time)

    main_move_start = time.time()
    if np.random.random() < prob:
        LOG += "Dragging move {} \n".format(move_uci)
        successful = drag_mouse(from_x, from_y, to_x, to_y, tolerance=0.12*STEP)
        dragged = True
    else:
        LOG += "Clicking move {} \n".format(move_uci)
        successful = click_to_from_mouse(from_x, from_y, to_x, to_y, tolerance=0.12*STEP)
        dragged = False
    main_move_time = (time.time() - main_move_start) * 1000
    
    if successful:
        LOG += "Made clicks for the move {} \n".format(move_uci)
    else:
        LOG += "Tried to make clicks for move {}, but made mouse slip \n".format(move_uci)
        return False

    # Confirm the move actually landed before queueing the premove or
    # handing control back to the scan loop; stray premove clicks on an
    # unregistered move select the wrong squares and cancel premoves
    if not _verify_move_registered(move_uci):
        debug_files = save_debug_screenshot(
            "move_not_registered",
            extra_info={'move_uci': move_uci, 'premove': premove,
                        'last_fen': DYNAMIC_INFO["fens"][-1] if DYNAMIC_INFO["fens"] else None})
        LOG += "WARNING: Clicks made for move {} but the piece never left {} - move did not register, skipping premove. Debug files: {}. \n".format(
            move_uci, move_uci[:2], debug_files)
        HOVER_SQUARE = None
        return False

    # If there is a premove
    premove_time = 0
    if premove is not None:
        if dragged == True:
            # wait a bit for previous move to lock in
            time.sleep(DRAG_MOVE_DELAY)
            dragged = False
        else:
            time.sleep(CLICK_MOVE_DELAY)
        from_x, from_y, to_x, to_y = find_clicks(premove)
        premove_start = time.time()
        if np.random.random() < prob:
            successful = drag_mouse(from_x, from_y, to_x, to_y, tolerance=0.12*STEP)
            dragged = True
        else:
            successful = click_to_from_mouse(from_x, from_y, to_x, to_y, tolerance=0.12*STEP)
        premove_time = (time.time() - premove_start) * 1000
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
    
    # Log move execution timing
    total_move_time = (time.time() - move_start_time) * 1000
    if premove is not None:
        LOG += f"[PERF] Move execution: {total_move_time:.0f}ms total (main:{main_move_time:.0f}ms, premove:{premove_time:.0f}ms)\n"
    else:
        LOG += f"[PERF] Move execution: {total_move_time:.0f}ms (mouse:{main_move_time:.0f}ms)\n"
    
    # Record timing for realised move time comparison
    MOVE_TIMING["move_execution_end_time"] = time.time()
    MOVE_TIMING["mouse_time_ms"] = total_move_time
    MOVE_TIMING["waiting_for_clock"] = True

    # Block further moves off this position until a scan is adopted
    AWAITING_FRESH_SCAN = True

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
    
    # Use calibrated resign button position if available
    if CALIBRATION_CONFIG_AVAILABLE:
        config = get_config()
        resign_button_x, resign_button_y = config.get_resign_button_position()
        LOG += f"Using calibrated resign button position: ({resign_button_x}, {resign_button_y})\n"
    else:
        # Fallback to hardcoded position
        resign_button_x, resign_button_y = START_X + 10.5*STEP, START_Y + 4.8*STEP
        LOG += f"Using hardcoded resign button position: ({resign_button_x}, {resign_button_y})\n"
    
    click_mouse(resign_button_x, resign_button_y, tolerance = 10, clicks=2, duration=np.random.uniform(0.3,0.7))
    
    return True

def new_game(time_control="1+0"):
    """
    Click through the UI to start a new game with the specified time control.
    
    Uses dynamic button detection when available, with fallback to hardcoded positions.
    
    Args:
        time_control: Time control string like "1+0", "3+0", etc.
    
    Returns:
        True if successful, False if caps lock prevents action.
    """
    global LOG
    # can only execute if no human interference.
    if is_capslock_on():
        LOG += "Tried to start new game with time control {} but failed as caps lock is on. \n ".format(time_control)
        return False

    # Never seek while a game is visibly live: the play-button clicks would
    # land on the running game (this happens when game setup fails during
    # the lobby-to-game page transition)
    for state in ["play", "start1", "start2"]:
        if read_clock(capture_bottom_clock(state=state)) is not None:
            # A readable clock could also be a FINISHED game whose end-state
            # clock bleeds into this region; a visible result box (or an
            # aborted / didn't-move message, which has no result box) means
            # the game is over and seeking is safe
            game_over = game_over_message_found() is not None
            try:
                if not game_over:
                    result_img = capture_result(arena=False)
                    if result_img is not None and result_img.size > 0:
                        for ref, threshold in _get_result_references():
                            if compare_result_images(result_img, ref) > threshold:
                                game_over = True
                                break
            except Exception:
                pass
            if game_over:
                break
            debug_files = save_debug_screenshot(
                "new_game_blocked_live_game", extra_info={'clock_state': state})
            LOG += "ERROR: Tried to seek a new game but a live game appears to be on screen (clock readable in state {}). Not clicking. Debug files: {}. \n".format(
                state, debug_files)
            write_log()
            return False

    # Try dynamic button detection first
    if DYNAMIC_BUTTON_DETECTION:
        LOG += "Using dynamic button detection for new game with time control {}. \n".format(time_control)
        
        # Step 1: Click the PLAY button
        play_pos = find_play_button()
        if play_pos is not None:
            play_button_x, play_button_y = play_pos
            LOG += "Found PLAY button at ({}, {}). \n".format(play_button_x, play_button_y)
        else:
            # Fallback to hardcoded position
            LOG += "PLAY button not found dynamically, using fallback position. \n"
            play_button_x, play_button_y = START_X - 1.9*STEP, START_Y - 0.4*STEP
        
        click_mouse(play_button_x, play_button_y, tolerance=10, clicks=1, duration=np.random.uniform(0.3, 0.7))
        time.sleep(1.5)  # Wait for menu to appear
        
        # Step 2: Click the time control button
        tc_pos = find_time_control_button(time_control)
        if tc_pos is not None:
            to_x, to_y = tc_pos
            LOG += "Found time control {} button at ({}, {}). \n".format(time_control, to_x, to_y)
            click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
        else:
            # Fallback to hardcoded position for known time controls
            LOG += "Time control {} button not found dynamically, trying fallback. \n".format(time_control)
            _new_game_fallback_click(time_control)
    else:
        # Use hardcoded positions (original behaviour)
        LOG += "Using hardcoded positions for new game with time control {}. \n".format(time_control)
        play_button_x, play_button_y = START_X - 1.9*STEP, START_Y - 0.4*STEP
        click_mouse(play_button_x, play_button_y, tolerance=10, clicks=1, duration=np.random.uniform(0.3, 0.7))
        time.sleep(1.5)
        _new_game_fallback_click(time_control)
    
    return True


def _new_game_fallback_click(time_control):
    """
    Fallback click positions for time controls when dynamic detection fails.
    
    Uses hardcoded positions relative to board coordinates.
    """
    if time_control == "1+0":
        to_x, to_y = START_X + 1.7*STEP, START_Y + 0.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    elif time_control == "2+1":
        to_x, to_y = START_X + 3.7*STEP, START_Y + 0.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    elif time_control == "3+0":
        to_x, to_y = START_X + 5.7*STEP, START_Y + 0.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    elif time_control == "3+2":
        to_x, to_y = START_X + 1.7*STEP, START_Y + 1.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    elif time_control == "5+0":
        to_x, to_y = START_X + 3.7*STEP, START_Y + 1.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    elif time_control == "5+3":
        to_x, to_y = START_X + 5.7*STEP, START_Y + 1.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    elif time_control == "10+0":
        to_x, to_y = START_X + 1.7*STEP, START_Y + 2.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    elif time_control == "10+5":
        to_x, to_y = START_X + 3.7*STEP, START_Y + 2.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    elif time_control == "15+10":
        to_x, to_y = START_X + 5.7*STEP, START_Y + 2.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    else:
        # Default to 1+0 position
        to_x, to_y = START_X + 1.7*STEP, START_Y + 0.7*STEP
        click_mouse(to_x, to_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))

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
                wait_time = ponder_response_wait(GAME_INFO["self_initial_time"], QUICKNESS,
                                                 pace_sf=ENGINE.ponder_pace_sf)
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
                            wait_time = ponder_response_wait(initial_time, QUICKNESS,
                                                             pace_sf=ENGINE.ponder_pace_sf)
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
            # Fire eagerness and hang-blindness both follow this game's
            # character: a snappy game fires more stale moves, a low-skill
            # game doesn't check where they land (skill-gated veto -- an
            # unconditional veto collapsed the blunder tail to 0.37x human).
            prob = (30 - own_time)/50 * (ENGINE.game_scramble_fire_sf or 1.0)
            veto_p = ENGINE.scramble_veto_p
            # we consider last 10 pondered moves here instead
            candidate_moves = [chess.Move.from_uci(x["move"]) for x in list(PONDER_DIC.values())[-10:] if chess.Move.from_uci(x["move"]) in dummy_board.legal_moves]
            for move_obj in candidate_moves:
                if check_safe_premove(last_board, move_obj.uci()) or \
                        (np.random.random() < prob and not (np.random.random() < veto_p and scramble_fire_veto(dummy_board, move_obj.uci()))):
                    # then we do it
                    LOG += "Did not find position in ponder_dic, but the last ponder move {} was considered a safe premove in position {}. By chance making this pondered move anyway. \n".format(curr_board.san(move_obj), curr_board.fen())
                    wait_time = scramble_response_wait()
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
        
        try:
            ENGINE.update_info(input_dic)
        except InvalidPositionError as e:
            # A structurally impossible position slipped into the fen
            # history (corrupt scrape). Don't let it kill the client:
            # go back to scanning, the next clean scrape replaces it.
            debug_files = save_debug_screenshot(
                "engine_rejected_position",
                extra_info={'error': str(e), 'fens': DYNAMIC_INFO["fens"]})
            LOG += "ERROR: Engine rejected position as structurally impossible ({}). Rescanning. Debug files: {}. \n".format(
                e, debug_files)
            write_log()
            continue

        # Once we send the information to the engine, first check if
        if ENGINE._decide_resign() == True:
            if not DYNAMIC_INFO["last_moves"]:
                # A resign-worthy position with no linked move history means
                # the fen history was just wiped by an unlinkable scan - far
                # more likely a stable misread (e.g. a piece on a grey
                # premove-highlight square reading as empty) than a real
                # collapse. A genuinely lost position survives to the next
                # linked scan, so resigning waits for one.
                LOG += "Engine wants to resign but the position has no linked move history (possible misread). Not resigning on this scan. \n"
                write_log()
            else:
                LOG += "Engine has decided to resign. Executing resign interaction. \n"
                time.sleep(resign_pause())
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
        
    
