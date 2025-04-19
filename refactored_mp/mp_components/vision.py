#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handles computer vision tasks for the chess client.

Includes screen capturing, FEN generation from images, clock reading,
move detection, and game start/end detection via image analysis.
"""

import cv2
import numpy as np
import chess
import datetime
import os
import time # For potential delays if needed

# Assuming original image scrape utils are accessible
# Adjust path if necessary based on project structure
from chessimage.image_scrape_utils import (
    SCREEN_CAPTURE, START_X, START_Y, STEP, capture_board,
    capture_top_clock, capture_bottom_clock, get_fen_from_image,
    check_fen_last_move_bottom, read_clock, find_initial_side,
    detect_last_move_from_img, check_turn_from_last_moved
)

# Import from refactored components
from .constants import ERROR_FILE_DIRECTORY
from .state import GAME_INFO, DYNAMIC_INFO, CASTLING_RIGHTS_FEN, update_dynamic_info
from .utils import append_log

# --- Move Detection ---

def get_move_change_from_image(image: np.ndarray, bottom: str = 'w') -> list[str] | None:
    """
    Analyzes a board image to detect highlighted squares indicating the last move.

    Compares pixel colors at specific points in each square to known highlight colors.

    Args:
        image: The cropped image of the chessboard (RGB format).
        bottom: The color ('w' or 'b') at the bottom of the board from the player's perspective.

    Returns:
        A list containing two UCI strings [from_sq+to_sq, to_sq+from_sq] if a
        two-square highlight is detected, otherwise None. Returns None also if
        an unexpected number of squares are highlighted.
    """
    board_height, board_width = image.shape[:2]
    if board_height == 0 or board_width == 0:
        append_log("ERROR: Received empty image in get_move_change_from_image.\n")
        return None

    tile_width = board_width / 8
    tile_height = board_height / 8
    epsilon = 5 # Offset from corner to check pixel color

    # Define square naming based on orientation
    if bottom == 'w':
        row_dic = {0: '8', 1: '7', 2: '6', 3: '5', 4: '4', 5: '3', 6: '2', 7: '1'}
        column_dic = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    else: # bottom == 'b'
        row_dic = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8'}
        column_dic = {0: 'h', 1: 'g', 2: 'f', 3: 'e', 4: 'd', 5: 'c', 6: 'b', 7: 'a'}

    detected_squares = []
    # Define known highlight colors (adjust these based on the specific website/UI)
    # Format: [R, G, B]
    highlight_colors = [
        [143, 155, 59],  # Example color 1
        [205, 211, 145], # Example color 2
        [95, 92, 60],   # Example color 3 (darker theme?)
        # Add more potential highlight colors if needed
    ]

    for i in range(64):
        column_i = i % 8
        row_i = i // 8
        # Calculate pixel coordinates, ensuring they are within bounds
        pixel_x = int(tile_width * column_i + epsilon)
        pixel_y = int(tile_height * row_i + epsilon)
        pixel_x = min(pixel_x, board_width - 1)
        pixel_y = min(pixel_y, board_height - 1)

        rgb = image[pixel_y, pixel_x, :]

        # Check if the pixel color matches any known highlight color
        for color in highlight_colors:
            if np.array_equal(rgb, color):
                try:
                    square_name = column_dic[column_i] + row_dic[row_i]
                    detected_squares.append(square_name)
                    break # Move to the next square once a match is found
                except KeyError:
                    append_log(f"ERROR: Invalid column/row index {column_i}/{row_i} in get_move_change.\n")
                    # This shouldn't happen with i in range(64)
                    break


    if len(detected_squares) == 2:
        # Return both possible UCI combinations as the order isn't guaranteed
        return [detected_squares[0] + detected_squares[1], detected_squares[1] + detected_squares[0]]
    elif len(detected_squares) == 0:
        return None # No move detected
    else:
        # This often happens during premoves or animations. Log it but return None.
        # append_log(f"Warning: Unexpectedly found {len(detected_squares)} highlighted squares: {detected_squares}. Likely animation/premove.\n")
        return None


def scrape_board_for_move_change() -> list[str] | None:
    """
    Captures the board and checks for highlighted squares indicating the last move.

    Returns:
        A list containing two UCI strings [from+to, to+from] if a move is detected,
        otherwise None.
    """
    try:
        # Capture the board region
        im = SCREEN_CAPTURE.capture((int(START_X), int(START_Y), int(8 * STEP), int(8 * STEP)))
        if im is None or im.size == 0:
             append_log("ERROR: Failed to capture board image in scrape_board_for_move_change.\n")
             return None
        # Process the image (ensure it's RGB)
        board_image_rgb = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB) if im.shape[2] == 4 else im[:, :, :3]

        # Determine orientation
        bottom = 'w' if GAME_INFO.get("playing_side", chess.WHITE) == chess.WHITE else 'b'

        return get_move_change_from_image(board_image_rgb, bottom=bottom)
    except Exception as e:
        append_log(f"ERROR during scrape_board_for_move_change: {e}\n")
        return None


# --- Game Start/End Detection ---

def check_new_game_start() -> float | None:
    """
    Checks screenshots of the clock area to detect if a new game has started.

    Looks for non-zero time values in specific known "start game" clock appearances.

    Returns:
        The detected initial time in seconds if a new game is found, otherwise None.
    """
    # Try reading the bottom clock in various potential "start" states
    # These states likely correspond to different UI elements/layouts
    start_states = ["start1", "start2"] # Add more if needed based on original scrape utils
    for state in start_states:
        try:
            clock_img = capture_bottom_clock(state=state)
            if clock_img is not None and clock_img.size > 0:
                time_val = read_clock(clock_img)
                if time_val is not None and time_val > 0: # Check for non-zero time
                    append_log(f"New game detected via bottom clock (state '{state}'). Initial time: {time_val}\n")
                    return time_val
            # else: append_log(f"Debug: capture_bottom_clock returned None/empty for state {state}\n")
        except Exception as e:
            append_log(f"ERROR checking new game start (state '{state}'): {e}\n")

    return None # No new game detected


def check_game_over_vision() -> bool:
    """
    Checks screenshots of the clock area to detect if the game has ended.

    Looks for specific clock appearances or values indicating game over (e.g., 0:00).

    Returns:
        True if a game over state is detected visually, False otherwise.
    """
    # Try reading the bottom clock in various potential "end" states
    end_states = ["end1", "end2", "end3"] # Add more if needed
    for state in end_states:
        try:
            clock_img = capture_bottom_clock(state=state)
            if clock_img is not None and clock_img.size > 0:
                time_val = read_clock(clock_img)
                # Game over might be indicated by finding *any* readable clock in an end state,
                # or specifically by finding a zero time value. Adjust logic as needed.
                # Original logic seemed to return True if *any* clock was read in these states.
                if time_val is not None:
                    append_log(f"Game over detected visually via bottom clock (state '{state}').\n")
                    return True
            # else: append_log(f"Debug: capture_bottom_clock returned None/empty for state {state}\n")
        except Exception as e:
            append_log(f"ERROR checking game over vision (state '{state}'): {e}\n")

    return False # No game over state detected visually


# --- Full State Update from Vision ---

def update_state_from_full_image_scan():
    """
    Performs a full scan by capturing board and clocks, then updates the game state.

    This is a more comprehensive update than just checking for move highlights.
    It reads FEN, clocks, checks turn, and updates DYNAMIC_INFO accordingly.
    """
    try:
        board_img = capture_board()
        top_clock_img = capture_top_clock() # Capture in default 'play' state
        bot_clock_img = capture_bottom_clock() # Capture in default 'play' state

        if board_img is None or top_clock_img is None or bot_clock_img is None:
            append_log("ERROR: Failed to capture necessary images for full scan.\n")
            return

        # --- Read Clocks ---
        our_time = read_clock(bot_clock_img)
        opp_time = read_clock(top_clock_img)

        # Handle cases where clock reading fails (e.g., during initial setup)
        if our_time is None:
            our_time_start1 = read_clock(capture_bottom_clock(state="start1"))
            our_time_start2 = read_clock(capture_bottom_clock(state="start2"))
            our_time = our_time_start1 if our_time_start1 is not None else our_time_start2
            if our_time is None and DYNAMIC_INFO["self_clock_times"]:
                 append_log("Warning: Couldn't read self time from 'play' or 'start' states. Using last known time.\n")
                 our_time = DYNAMIC_INFO["self_clock_times"][-1]
            elif our_time is None:
                 append_log("Warning: Couldn't read self time and no history exists. Setting to initial time.\n")
                 our_time = GAME_INFO.get("self_initial_time", 60) # Default if initial not set

        if opp_time is None:
            opp_time_start1 = read_clock(capture_top_clock(state="start1"))
            opp_time_start2 = read_clock(capture_top_clock(state="start2"))
            opp_time = opp_time_start1 if opp_time_start1 is not None else opp_time_start2
            if opp_time is None and DYNAMIC_INFO["opp_clock_times"]:
                 append_log("Warning: Couldn't read opponent time from 'play' or 'start' states. Using last known time.\n")
                 opp_time = DYNAMIC_INFO["opp_clock_times"][-1]
            elif opp_time is None:
                 append_log("Warning: Couldn't read opponent time and no history exists. Setting to initial time.\n")
                 opp_time = GAME_INFO.get("opp_initial_time", 60) # Default if initial not set


        # --- Get FEN and Determine Turn ---
        bottom = 'w' if GAME_INFO.get("playing_side", chess.WHITE) == chess.WHITE else 'b'
        # Get FEN assuming white's turn initially, then correct it
        try:
            fen_from_img = get_fen_from_image(board_img, bottom=bottom, turn=chess.WHITE) # Assume white turn first
            if fen_from_img is None:
                append_log("ERROR: get_fen_from_image returned None.\n")
                return

            # Check whose turn it *actually* is based on last move highlight
            turn_check_result = check_turn_from_last_moved(fen_from_img, board_img, bottom)

            current_turn = None
            if turn_check_result is True:
                current_turn = chess.WHITE # The assumed turn was correct
            elif turn_check_result is False:
                current_turn = chess.BLACK # The assumed turn was wrong, flip it
            else: # turn_check_result is None (error or no highlight)
                append_log("Warning: check_turn_from_last_moved failed. Attempting to infer turn from previous state.\n")
                if DYNAMIC_INFO["fens"]:
                    try:
                        last_board = chess.Board(DYNAMIC_INFO["fens"][-1])
                        current_turn = not last_board.turn # Assume turn flipped
                        append_log(f"Inferred turn as {current_turn} based on previous FEN.\n")
                    except (ValueError, IndexError):
                         append_log("ERROR: Could not parse previous FEN to infer turn. Defaulting to White.\n")
                         current_turn = chess.WHITE
                else:
                    append_log("Warning: No previous FEN available to infer turn. Defaulting to White.\n")
                    current_turn = chess.WHITE # Default if no other info

            # --- Construct Final FEN ---
            final_board = chess.Board(fen_from_img) # Start with board structure
            final_board.turn = current_turn
            final_board.castling_rights = chess.Board(f"{final_board.board_fen()} w {CASTLING_RIGHTS_FEN} - 0 1").castling_rights # Preserve known rights
            # Update move counters based on previous state if possible
            if DYNAMIC_INFO["fens"]:
                 try:
                     last_board = chess.Board(DYNAMIC_INFO["fens"][-1])
                     # Increment fullmove number if Black just moved (i.e., it's now White's turn)
                     final_board.fullmove_number = last_board.fullmove_number + (1 if current_turn == chess.WHITE else 0)
                     # Reset halfmove clock - assuming it's not easily readable from image
                     # A capture or pawn move would reset it, otherwise increment. Hard to tell from image alone.
                     # Safest might be to just increment or leave as FEN parser default (0). Let's increment.
                     final_board.halfmove_clock = last_board.halfmove_clock + 1 # Simplistic update
                 except (ValueError, IndexError):
                     append_log("Warning: Could not update move counters from previous FEN.\n")
                     # Use defaults from FEN parser
            else:
                 # If no history, set fullmove based on turn
                 final_board.fullmove_number = 1
                 final_board.halfmove_clock = 0


            final_fen = final_board.fen()

            # --- Update State ---
            # update_dynamic_info handles checking for actual changes before adding
            update_dynamic_info(fen=final_fen, self_time=our_time, opp_time=opp_time)

        except Exception as e:
            append_log(f"ERROR during FEN generation/turn check in full scan: {e}\n")
            # Save error image
            timestamp = str(datetime.datetime.now()).replace(" ", "").replace(":", "_")
            error_filename = os.path.join(ERROR_FILE_DIRECTORY, f"board_img_fullscan_error_{timestamp}.png")
            try:
                cv2.imwrite(error_filename, board_img)
                append_log(f"Saved error board image to {error_filename}\n")
            except Exception as write_e:
                append_log(f"ERROR: Failed to save error board image: {write_e}\n")

    except Exception as e:
        append_log(f"ERROR during capture phase of full image scan: {e}\n")


def verify_initial_fen(starting_fen: str) -> str:
    """
    Verifies the initial FEN obtained from image analysis against the board image.
    Attempts to correct the turn if the initial check fails.

    Args:
        starting_fen: The initially determined FEN string.

    Returns:
        The verified (potentially corrected) FEN string.
    """
    try:
        board_img = capture_board()
        if board_img is None:
            append_log("ERROR: Failed to capture board image for initial FEN verification.\n")
            return starting_fen # Return original if capture fails

        bottom = 'w' if GAME_INFO.get("playing_side", chess.WHITE) == chess.WHITE else 'b'

        if check_fen_last_move_bottom(starting_fen, board_img, bottom):
            append_log("Initial FEN check successful.\n")
            return starting_fen
        else:
            append_log("Warning: Initial FEN check failed. Attempting to flip turn.\n")
            try:
                dummy_board = chess.Board(starting_fen)
                dummy_board.turn = not dummy_board.turn
                corrected_fen = dummy_board.fen()
                if check_fen_last_move_bottom(corrected_fen, board_img, bottom):
                    append_log("Corrected initial FEN by flipping turn.\n")
                    return corrected_fen
                else:
                    append_log("ERROR: Could not verify initial FEN even after flipping turn.\n")
                    # Save error image
                    timestamp = str(datetime.datetime.now()).replace(" ", "").replace(":", "_")
                    error_filename = os.path.join(ERROR_FILE_DIRECTORY, f"board_img_fen_verify_error_{timestamp}.png")
                    try:
                        cv2.imwrite(error_filename, board_img)
                        append_log(f"Saved error board image to {error_filename}\n")
                    except Exception as write_e:
                        append_log(f"ERROR: Failed to save error board image: {write_e}\n")
                    return starting_fen # Return original if correction fails
            except ValueError:
                 append_log(f"ERROR: Invalid FEN '{starting_fen}' during verification correction.\n")
                 return starting_fen

    except Exception as e:
        append_log(f"ERROR during initial FEN verification: {e}\n")
        return starting_fen # Return original on error