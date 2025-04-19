#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handles all mouse interactions for the chess client, simulating human-like movements.
"""

import pyautogui
import random
import time
import numpy as np
import chess

# Assuming CustomCursor is in the original project structure accessible via PYTHONPATH
# If not, adjust the import path accordingly.
from common.custom_cursor import CustomCursor
from chessimage.image_scrape_utils import START_X, START_Y, STEP # Assuming these are defined here

# Import constants and utilities from the refactored structure
from .constants import MOUSE_QUICKNESS, DRAG_MOVE_DELAY, CLICK_MOVE_DELAY
from .utils import is_capslock_on, append_log
# Import state variables (will be defined in state.py later)
from .state import GAME_INFO, DYNAMIC_INFO, PONDER_DIC, HOVER_SQUARE, update_hover_square

# Initialize the custom cursor globally within this module
CURSOR = CustomCursor()

def drag_mouse(from_x: float, from_y: float, to_x: float, to_y: float, tolerance: float = 0.2 * STEP) -> bool:
    """
    Simulates a human-like drag-and-drop mouse movement.

    Introduces randomness and a small chance of a 'mouse slip'.

    Args:
        from_x: Starting X coordinate.
        from_y: Starting Y coordinate.
        to_x: Ending X coordinate.
        to_y: Ending Y coordinate.
        tolerance: Controls the randomness/spread around the target points.

    Returns:
        True if the drag was likely successful, False if a simulated mouse slip occurred.
    """
    successful = True
    # Simulate potential mouse slip (increased offset, higher chance of failure)
    if np.random.random() < 0.03: # 3% chance of potential slip logic
        tolerance = tolerance * 2
        offset_x = np.clip(np.random.randn() * tolerance, -STEP / 1.5, STEP / 1.5)
        offset_y = np.clip(np.random.randn() * tolerance, -STEP / 1.5, STEP / 1.5)
        # Check if offset is large enough to likely cause a slip (missing the target square)
        if np.abs(offset_x) > STEP / 2 or np.abs(offset_y) > STEP / 2:
            successful = False
    else: # Normal move randomness
        offset_x = np.clip(np.random.randn() * tolerance, -STEP / 2.2, STEP / 2.2)
        offset_y = np.clip(np.random.randn() * tolerance, -STEP / 2.2, STEP / 2.2)

    # Apply small random offsets to start and end points for realism
    new_from_x = from_x + tolerance * (np.random.random() - 0.5)
    new_from_y = from_y + tolerance * (np.random.random() - 0.5)
    new_to_x = to_x + offset_x
    new_to_y = to_y + offset_y

    # Calculate dynamic durations based on distance for more human-like speed
    current_x, current_y = pyautogui.position()
    from_distance = np.sqrt((new_from_x - current_x)**2 + (new_from_y - current_y)**2)
    # Duration calculation adjusted slightly from original for potentially smoother feel
    duration_from = MOUSE_QUICKNESS / 10000 * (0.8 + 0.4 * random.random()) * (from_distance + 1)**0.3 # Added +1 to avoid log(0) issues
    to_distance = np.sqrt((new_from_x - new_to_x)**2 + (new_from_y - new_to_y)**2)
    duration_to = MOUSE_QUICKNESS / 10000 * (0.8 + 0.4 * random.random()) * (to_distance + 1)**0.5

    # Ensure minimum duration to prevent instant moves
    duration_from = max(duration_from, 0.01)
    duration_to = max(duration_to, 0.01)

    CURSOR.drag_and_drop([new_from_x, new_from_y], [new_to_x, new_to_y], duration=[duration_from, duration_to])

    return successful

def click_to_from_mouse(from_x: float, from_y: float, to_x: float, to_y: float, tolerance: float = 0.2 * STEP) -> bool:
    """
    Simulates making a move by clicking the 'from' square then the 'to' square.

    Includes randomness and potential for mouse slips, similar to drag_mouse.

    Args:
        from_x: 'From' square center X coordinate.
        from_y: 'From' square center Y coordinate.
        to_x: 'To' square center X coordinate.
        to_y: 'To' square center Y coordinate.
        tolerance: Controls the randomness/spread around the target points.

    Returns:
        True if the clicks were likely successful, False if a simulated mouse slip occurred.
    """
    successful = True
    # Simulate potential mouse slip
    if np.random.random() < 0.03: # 3% chance
        tolerance = tolerance * 2
        offset_x = np.clip(np.random.randn() * tolerance, -STEP / 1.5, STEP / 1.5)
        offset_y = np.clip(np.random.randn() * tolerance, -STEP / 1.5, STEP / 1.5)
        if np.abs(offset_x) > STEP / 2 or np.abs(offset_y) > STEP / 2:
            successful = False
    else: # Normal move randomness
        offset_x = np.clip(np.random.randn() * tolerance, -STEP / 2.2, STEP / 2.2)
        offset_y = np.clip(np.random.randn() * tolerance, -STEP / 2.2, STEP / 2.2)

    # Apply random offsets
    new_from_x = from_x + tolerance * (np.random.random() - 0.5)
    new_from_y = from_y + tolerance * (np.random.random() - 0.5)
    new_to_x = to_x + offset_x
    new_to_y = to_y + offset_y

    # Calculate dynamic durations
    current_x, current_y = pyautogui.position()
    from_distance = np.sqrt((new_from_x - current_x)**2 + (new_from_y - current_y)**2)
    duration_from = MOUSE_QUICKNESS / 10000 * (0.8 + 0.4 * random.random()) * np.sqrt(from_distance + 1)
    to_distance = np.sqrt((new_from_x - new_to_x)**2 + (new_from_y - new_to_y)**2)
    duration_to = MOUSE_QUICKNESS / 10000 * (0.8 + 0.4 * random.random()) * np.sqrt(to_distance + 1)

    # Ensure minimum duration
    duration_from = max(duration_from, 0.01)
    duration_to = max(duration_to, 0.01)

    # Execute moves
    CURSOR.move_to([new_from_x, new_from_y], duration=duration_from, steady=True)
    pyautogui.click(button="left")
    CURSOR.move_to([new_to_x, new_to_y], duration=duration_to, steady=True)
    pyautogui.click(button="left")

    return successful

def click_mouse(x: float, y: float, tolerance: float = 10, clicks: int = 1, duration: float = 0.5):
    """
    Moves the mouse to a target coordinate with randomness and clicks.

    Args:
        x: Target X coordinate.
        y: Target Y coordinate.
        tolerance: Pixel range for random offset around the target.
        clicks: Number of clicks to perform.
        duration: Time taken for the mouse movement.
    """
    # Apply random offset
    new_x = x + tolerance * (np.random.random() - 0.5)
    new_y = y + tolerance * (np.random.random() - 0.5)

    # Ensure minimum duration
    duration = max(duration, 0.01)

    CURSOR.move_to([new_x, new_y], duration=duration, steady=True)
    pyautogui.click(button="left", clicks=clicks)


def find_clicks(move_uci: str) -> tuple[float, float, float, float]:
    """
    Calculates the screen coordinates for the center of the 'from' and 'to' squares of a UCI move.

    Args:
        move_uci: The move in UCI format (e.g., "e2e4").

    Returns:
        A tuple containing (from_x, from_y, to_x, to_y).
    """
    move_obj = chess.Move.from_uci(move_uci)
    from_square = move_obj.from_square
    to_square = move_obj.to_square

    # Calculate coordinates based on playing side
    if GAME_INFO["playing_side"] == chess.WHITE:
        # a1 is bottom left (rank 0, file 0) -> screen (y=7*STEP, x=0*STEP)
        rank_fr = chess.square_rank(from_square)
        file_fr = chess.square_file(from_square)
        click_from_x = START_X + file_fr * STEP + STEP / 2
        click_from_y = START_Y + (7 - rank_fr) * STEP + STEP / 2

        rank_to = chess.square_rank(to_square)
        file_to = chess.square_file(to_square)
        click_to_x = START_X + file_to * STEP + STEP / 2
        click_to_y = START_Y + (7 - rank_to) * STEP + STEP / 2
    else: # Playing as Black
        # a1 is top right (rank 0, file 0) -> screen (y=0*STEP, x=7*STEP)
        rank_fr = chess.square_rank(from_square)
        file_fr = chess.square_file(from_square)
        click_from_x = START_X + (7 - file_fr) * STEP + STEP / 2
        click_from_y = START_Y + rank_fr * STEP + STEP / 2

        rank_to = chess.square_rank(to_square)
        file_to = chess.square_file(to_square)
        click_to_x = START_X + (7 - file_to) * STEP + STEP / 2
        click_to_y = START_Y + rank_to * STEP + STEP / 2

    return click_from_x, click_from_y, click_to_x, click_to_y

def make_move(move_uci: str, premove: str | None = None) -> bool:
    """
    Executes the mouse clicks/drags for a given move and optional premove.

    Randomly chooses between dragging and clicking for variation. Handles potential
    mouse slips by returning False.

    Args:
        move_uci: The main move to make in UCI format.
        premove: An optional premove in UCI format.

    Returns:
        True if the actions (move and optional premove) were likely completed without slips,
        False otherwise.
    """
    if is_capslock_on():
        append_log(f"Tried to make move {move_uci} and premove {premove}, but failed as caps lock is on.\n")
        return False

    # Decide whether to drag or click based on remaining time (more likely to click when low)
    own_time = max(DYNAMIC_INFO["self_clock_times"][-1], 1) if DYNAMIC_INFO["self_clock_times"] else 60 # Default if no time info
    # Probability of dragging decreases as time gets lower
    drag_probability = 0.8 if own_time > 20 else own_time / 25.0
    drag_probability = np.clip(drag_probability, 0.1, 0.9) # Bounds

    # --- Execute Main Move ---
    from_x, from_y, to_x, to_y = find_clicks(move_uci)
    dragged_main_move = False
    if np.random.random() < drag_probability:
        append_log(f"Dragging move {move_uci}\n")
        successful = drag_mouse(from_x, from_y, to_x, to_y)
        dragged_main_move = True
    else:
        append_log(f"Clicking move {move_uci}\n")
        successful = click_to_from_mouse(from_x, from_y, to_x, to_y)

    if not successful:
        append_log(f"Tried to make clicks for move {move_uci}, but made mouse slip\n")
        return False
    else:
        append_log(f"Made clicks for the move {move_uci}\n")

    # Apply appropriate delay after main move
    time.sleep(DRAG_MOVE_DELAY if dragged_main_move else CLICK_MOVE_DELAY)

    # --- Execute Premove (if any) ---
    if premove:
        from_x_pre, from_y_pre, to_x_pre, to_y_pre = find_clicks(premove)
        dragged_premove = False
        # Decide drag/click for premove (can be different from main move)
        if np.random.random() < drag_probability:
             append_log(f"Dragging premove {premove}\n")
             successful_pre = drag_mouse(from_x_pre, from_y_pre, to_x_pre, to_y_pre)
             dragged_premove = True
        else:
             append_log(f"Clicking premove {premove}\n")
             successful_pre = click_to_from_mouse(from_x_pre, from_y_pre, to_x_pre, to_y_pre)

        if not successful_pre:
            append_log(f"Tried to make clicks for premove {premove}, but made mouse slip.\n")
            # Even if premove slips, the main move might have registered.
            # The original code didn't explicitly return False here, so we maintain that.
            # The game state update logic should handle the consequences.
        else:
            append_log(f"Made clicks for the premove {premove}\n")

        # Apply delay after premove
        time.sleep(DRAG_MOVE_DELAY if dragged_premove else CLICK_MOVE_DELAY)

    # Reset hover square after making a move
    update_hover_square(None)

    return True # Return True if main move was successful, regardless of premove slip


def wander(max_duration: float = 0.15):
    """
    Moves the mouse randomly to a position on the board, biased towards the center.

    Args:
        max_duration: The maximum time the mouse movement should take.
    """
    if is_capslock_on():
        append_log("Tried to wander, but failed as caps lock is on.\n")
        return

    current_x, current_y = pyautogui.position()
    centre_x = START_X + 4 * STEP
    centre_y = START_Y + 4 * STEP

    # Bias towards center, but allow randomness
    m_x = 0.8 * current_x + 0.2 * centre_x
    m_y = 0.8 * current_y + 0.2 * centre_y

    # Choose a target point with random offset, clamped to board boundaries
    chosen_x = np.clip(m_x + STEP * np.random.randn(), START_X, START_X + 8 * STEP)
    chosen_y = np.clip(m_y + STEP * np.random.randn(), START_Y, START_Y + 8 * STEP)

    # Calculate dynamic duration based on distance
    distance = np.sqrt((chosen_x - current_x)**2 + (chosen_y - current_y)**2)
    duration = max(min(MOUSE_QUICKNESS / 5000 * (0.8 + 0.4 * np.random.random()) * np.sqrt(distance + 1), max_duration), 0.01)

    CURSOR.move_to([chosen_x, chosen_y], duration=duration)


def hover(duration: float = 0.1, noise: float = STEP * 2):
    """
    Moves the mouse to hover over a relevant piece, often one considered in pondering.

    Args:
        duration: The time the mouse movement should take.
        noise: Controls the random offset around the target square center.
    """
    global HOVER_SQUARE # Use the global HOVER_SQUARE from state.py

    if is_capslock_on():
        append_log("Tried to hover, but failed as caps lock is on.\n")
        return

    # Determine the square to hover over
    target_square = HOVER_SQUARE
    if target_square is None:
        # If no specific square is being hovered, choose one
        try:
            last_known_board = chess.Board(DYNAMIC_INFO["fens"][-1])
            # Prefer pieces involved in recent ponder moves
            relevant_ponder_moves = [
                chess.Move.from_uci(uci) for uci in PONDER_DIC.values()
                if chess.Move.from_uci(uci).from_square < 64 # Basic validation
                   and last_known_board.color_at(chess.Move.from_uci(uci).from_square) == GAME_INFO["playing_side"]
            ]

            if relevant_ponder_moves:
                target_square = relevant_ponder_moves[-1].from_square # Hover over last pondered piece
            else:
                # Otherwise, choose a random piece of ours
                own_piece_squares = list(chess.SquareSet(last_known_board.occupied_co[GAME_INFO["playing_side"]]))
                if own_piece_squares:
                    target_square = random.choice(own_piece_squares)
                else:
                    append_log("Tried to hover, but no own pieces found on board.\n")
                    return # No pieces to hover over
            update_hover_square(target_square) # Update the global state

        except (IndexError, ValueError, KeyError):
             # Handle cases where DYNAMIC_INFO or GAME_INFO might be temporarily empty/invalid
             append_log("Error determining hover square: Invalid state information.\n")
             wander() # Wander randomly instead
             return

    # Calculate target coordinates based on the chosen square and playing side
    if GAME_INFO["playing_side"] == chess.WHITE:
        rank_fr = chess.square_rank(target_square)
        file_fr = chess.square_file(target_square)
        target_x = START_X + file_fr * STEP + STEP / 2
        target_y = START_Y + (7 - rank_fr) * STEP + STEP / 2
    else: # Black
        rank_fr = chess.square_rank(target_square)
        file_fr = chess.square_file(target_square)
        target_x = START_X + (7 - file_fr) * STEP + STEP / 2
        target_y = START_Y + rank_fr * STEP + STEP / 2

    # Apply noise
    final_x = np.clip(target_x + noise * (np.random.random() - 0.5), START_X, START_X + 8 * STEP)
    final_y = np.clip(target_y + noise * (np.random.random() - 0.5), START_Y, START_Y + 8 * STEP)

    # Ensure minimum duration
    duration = max(duration, 0.01)

    CURSOR.move_to([final_x, final_y], duration=duration, steady=True)


# --- Button Click Functions ---

def berserk() -> bool:
    """ Clicks the berserk button (coordinates hardcoded relative to board). """
    if is_capslock_on():
        append_log("Tried to berserk but failed as caps lock is on.\n")
        return False
    # Coordinates likely need calibration based on screen resolution/layout
    button_x, button_y = START_X + 10.5 * STEP, START_Y + 5.7 * STEP
    click_mouse(button_x, button_y, tolerance=10, clicks=1, duration=np.random.uniform(0.3, 0.7))
    append_log("Clicked berserk button.\n")
    return True

def back_to_lobby() -> bool:
    """ Clicks the 'back to lobby' or similar button after a game. """
    if is_capslock_on():
        append_log("Tried to go back to lobby but failed as caps lock is on.\n")
        return False
    # Coordinates likely need calibration
    button_x, button_y = START_X + 10.5 * STEP, START_Y + 4.1 * STEP
    click_mouse(button_x, button_y, tolerance=10, clicks=1, duration=np.random.uniform(0.3, 0.7))
    append_log("Clicked back to lobby button.\n")
    return True

def resign() -> bool:
    """ Clicks the resign button (usually requires two clicks). """
    if is_capslock_on():
        append_log("Tried to resign the game but failed as caps lock is on.\n")
        return False
    # Coordinates likely need calibration
    resign_button_x, resign_button_y = START_X + 10.5 * STEP, START_Y + 4.8 * STEP
    # Double click to confirm resignation
    click_mouse(resign_button_x, resign_button_y, tolerance=10, clicks=2, duration=np.random.uniform(0.3, 0.7))
    append_log("Clicked resign button twice.\n")
    return True

def new_game(time_control: str = "1+0") -> bool:
    """ Clicks buttons to start a new game with a specific time control. """
    if is_capslock_on():
        append_log(f"Tried to start new game ({time_control}) but failed as caps lock is on.\n")
        return False

    # Click "Play" button (coordinates need calibration)
    play_button_x, play_button_y = START_X - 1.9 * STEP, START_Y - 0.4 * STEP
    click_mouse(play_button_x, play_button_y, tolerance=10, clicks=1, duration=np.random.uniform(0.3, 0.7))
    append_log("Clicked Play button.\n")
    time.sleep(1.5) # Wait for time control options to appear

    # Click specific time control button (coordinates need calibration)
    if time_control == "1+0":
        tc_x, tc_y = START_X + 1.7 * STEP, START_Y + 0.7 * STEP
    elif time_control == "3+0":
        tc_x, tc_y = START_X + 5.7 * STEP, START_Y + 0.7 * STEP
    # Add other time controls as needed
    # elif time_control == "5+0":
    #     tc_x, tc_y = ...
    else:
        append_log(f"Unsupported time control '{time_control}' for new game click.\n")
        return False

    click_mouse(tc_x, tc_y, tolerance=20, clicks=1, duration=np.random.uniform(0.3, 0.7))
    append_log(f"Clicked {time_control} time control button.\n")
    return True