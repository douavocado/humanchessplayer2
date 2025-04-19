#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main execution script for the refactored multiprocessing chess client.

Orchestrates screen scraping, state management, engine interaction,
and mouse control to play chess games automatically.
"""

import time
import random
import os
import chess
import numpy as np
import datetime

# Import components from the mp_components package
from mp_components import constants
from mp_components import state
from mp_components import utils
from mp_components import vision
from mp_components import mouse
from mp_components import engine_adapter
from mp_components import threading_manager  # Import the new threading manager

# --- Game Setup ---

def await_new_game(timeout: int = 60) -> float | None:
    """
    Waits for a new game to start by monitoring the screen.

    Args:
        timeout: Maximum time to wait in seconds.

    Returns:
        The initial clock time in seconds if a game is found, otherwise None.
    """
    utils.append_log(f"Awaiting new game (timeout={timeout}s)...\n")
    time_start = time.time()
    while time.time() - time_start < timeout:
        initial_time = vision.check_new_game_start()
        if initial_time is not None:
            utils.append_log(f"New game found! Initial time: {initial_time}\n")
            try:
                # Play sound notification
                os.system(f"mpg123 -q {constants.NEW_GAME_SOUND}")
            except Exception as e:
                utils.append_log(f"Warning: Could not play new game sound: {e}\n")
            return initial_time
        time.sleep(0.5) # Check every half second

    utils.append_log("Timeout reached while waiting for new game.\n")
    try:
        # Play alert sound on timeout
        os.system(f"mpg123 -q {constants.ALERT_SOUND}")
    except Exception as e:
        utils.append_log(f"Warning: Could not play alert sound: {e}\n")
    return None


def set_up_game(starting_time: float):
    """
    Initializes the game state once a new game is detected.

    Determines playing side, initial FEN, and resets state variables.

    Args:
        starting_time: The initial clock time detected.
    """
    utils.append_log("Setting up new game...\n")
    engine_adapter.initialize_engine() # Ensure engine is ready

    # Determine playing side
    playing_side = vision.find_initial_side()
    utils.append_log(f"Determined playing side: {'White' if playing_side == chess.WHITE else 'Black'}\n")
    bottom = 'w' if playing_side == chess.WHITE else 'b'

    # Get initial board state
    board_img = vision.capture_board()
    if board_img is None:
        utils.append_log("ERROR: Failed to capture board image during game setup. Aborting setup.\n")
        # TODO: How to handle this failure? Maybe retry?
        return False # Indicate setup failure

    # Determine initial FEN and turn
    # Assume white turn first, vision components will verify/correct
    initial_fen_guess = vision.get_fen_from_image(board_img, bottom=bottom, turn=chess.WHITE)
    if initial_fen_guess is None:
         utils.append_log("ERROR: Failed to get initial FEN from image. Aborting setup.\n")
         return False

    # Verify FEN and turn using vision checks
    verified_fen = vision.verify_initial_fen(initial_fen_guess)

    # Reset the game state using the determined info
    state.reset_game_state(playing_side, starting_time, verified_fen)

    utils.append_log("Game setup complete.\n")
    utils.write_log() # Write setup logs
    return True # Indicate setup success

# --- Main Game Loop Logic ---

def await_opponent_move():
    """
    Monitors the screen and updates state until it's our turn or the game ends.

    Uses a combination of full scans and faster move highlight checks.

    Returns:
        True if it's our turn, False if the game has ended.
    """
    utils.append_log("Waiting for opponent's move...\n")
    last_full_scan_time = 0
    while True:
        # --- Check for Game End ---
        # Check via board state first (fastest)
        if state.is_game_over():
            utils.append_log("Game over detected by board state.\n")
            return False
        # Check visually as a fallback
        if vision.check_game_over_vision():
            utils.append_log("Game over detected visually.\n")
            return False

        # --- Check for Manual Override ---
        if utils.is_capslock_on():
            utils.append_log("Manual mode (Caps Lock) detected. Pausing automation.\n")
            time.sleep(1) # Wait while caps lock is on
            continue

        # --- Perform Full Scan Periodically ---
        current_time = time.time()
        if current_time - last_full_scan_time >= constants.SCRAPE_EVERY:
            utils.append_log("Performing full image scan...\n")
            vision.update_state_from_full_image_scan()
            last_full_scan_time = current_time
            # Check turn immediately after full scan
            if state.is_our_turn():
                utils.append_log("Our turn detected after full scan.\n")
                return True
            # Check game end again after scan
            if state.is_game_over() or vision.check_game_over_vision():
                 utils.append_log("Game over detected after full scan.\n")
                 return False

        # --- Perform Faster Move Highlight Checks ---
        # Check for move highlights between full scans for responsiveness
        move_uci_pair = vision.scrape_board_for_move_change()
        if move_uci_pair:
            move1_uci, move2_uci = move_uci_pair
            current_board = state.get_current_board()
            if current_board:
                move1 = chess.Move.from_uci(move1_uci)
                move2 = chess.Move.from_uci(move2_uci)
                # TODO: Handle castling detection from highlights more robustly if needed
                # The original code had complex logic (lines 633-655) to map king/rook highlights to castling moves.
                # This might need reimplementation if simple UCI check isn't enough.
                # For now, assume standard moves or rely on full scan for castling.

                valid_move_found = None
                if move1 in current_board.legal_moves:
                    valid_move_found = move1
                elif move2 in current_board.legal_moves:
                     valid_move_found = move2

                if valid_move_found:
                    utils.append_log(f"Move highlight detected and validated: {valid_move_found.uci()}\n")
                    # Update state based *only* on this move for speed
                    # Need opponent's clock time after their move
                    opp_clock_img = vision.capture_top_clock(state="play") # Or other relevant states
                    opp_time = vision.read_clock(opp_clock_img)
                    if opp_time is None: # Fallback if reading fails
                        opp_time = state.DYNAMIC_INFO["opp_clock_times"][-1] if state.DYNAMIC_INFO["opp_clock_times"] else state.GAME_INFO["opp_initial_time"]

                    new_board = current_board.copy()
                    new_board.push(valid_move_found)
                    state.update_dynamic_info(fen=new_board.fen(), opp_time=opp_time, move_uci=valid_move_found.uci())

                    # It should now be our turn
                    if state.is_our_turn():
                        utils.append_log("Our turn detected after move highlight update.\n")
                        return True
                    else:
                         utils.append_log("Warning: Updated state after move highlight, but it's still not our turn?\n")
                         # Might need a full scan to resolve discrepancy
                         vision.update_state_from_full_image_scan()
                         if state.is_our_turn(): return True


        # --- Random mouse hovering/wandering (for human-like appearance) ---
        own_time = state.DYNAMIC_INFO["self_clock_times"][-1] if state.DYNAMIC_INFO["self_clock_times"] else 60
        if not state.is_our_turn() and own_time > 15: # Only hover if sufficient time
            # Logic from original script for mouse hovering/wandering
            if state.HOVER_SQUARE is None:
                if np.random.random() < 0.9: mouse.hover(duration=np.random.random()/5)
                else: mouse.wander()
            elif np.random.random() < 0.06: mouse.hover(duration=np.random.random()/5)
            elif np.random.random() < 0.04: mouse.wander()

        # Small delay to prevent excessive CPU usage in the loop
        time.sleep(0.05)


def play_our_turn():
    """
    Handles the logic when it's our turn to move.

    Checks ponder hits, asks engine for move, makes the move, handles resignation.
    """
    utils.append_log("--- Our Turn ---")
    start_turn_time = time.time()

    # --- Check Ponder Hit ---
    ponder_move_uci = engine_adapter.get_ponder_move()
    if ponder_move_uci:
        utils.append_log(f"Ponder hit! Move: {ponder_move_uci}")
        # Calculate wait time based on original logic (lines 1084-1087)
        initial_time = state.GAME_INFO.get("self_initial_time", 60)
        base_time = 0.4 * constants.QUICKNESS * initial_time / (85 + initial_time * 0.25)
        wait_time = base_time * (0.8 + 0.4 * random.random())
        wait_time = max(0, wait_time) # Ensure non-negative
        utils.append_log(f"Calculated ponder wait time: {wait_time:.3f}s")
        time.sleep(wait_time)

        made_move = mouse.make_move(ponder_move_uci) # Ponder moves usually don't have premoves
        if not made_move:
            utils.append_log("Ponder move failed (mouse slip?). Trying again.")
            time.sleep(0.1)
            mouse.make_move(ponder_move_uci) # Try once more
        utils.write_log()
        return # End turn after ponder move

    # --- Check Safe Ponder Premoves (especially if low time) ---
    safe_premove_uci = engine_adapter.get_safe_ponder_premove()
    if safe_premove_uci:
         utils.append_log(f"Playing safe/low-time ponder premove: {safe_premove_uci}")
         # Use shorter wait time for these less certain moves
         wait_time = 0.1 * (0.8 + 0.4 * random.random())
         time.sleep(wait_time)
         made_move = mouse.make_move(safe_premove_uci)
         if not made_move:
             utils.append_log("Safe ponder premove failed (mouse slip?). Trying again.")
             time.sleep(0.1)
             mouse.make_move(safe_premove_uci)
         utils.write_log()
         return # End turn

    # --- Update Engine State ---
    if not engine_adapter.update_engine_state():
        utils.append_log("ERROR: Failed to update engine state. Cannot request move.")
        utils.write_log()
        return # Skip turn if engine update fails

    # --- Check Resignation ---
    if engine_adapter.check_engine_resign():
        utils.append_log("Engine recommends resignation. Waiting a bit...")
        time.sleep(2 + 3 * random.random()) # Human-like delay before resigning
        if mouse.resign():
            utils.append_log("Resignation successful.")
            utils.write_log()
            # Game should end, await_opponent_move will detect it next iteration
            return
        else:
            utils.append_log("Resignation failed (Caps Lock?). Continuing game.")

    # --- Get Move from Engine ---
    engine_output = engine_adapter.ask_engine_for_move()
    if engine_output is None:
        utils.append_log("ERROR: Did not receive valid move from engine. Skipping turn.")
        # TODO: What should happen here? Maybe play a random legal move?
        utils.write_log()
        return

    move_to_make = engine_output["move_made"]
    premove_to_make = engine_output["premove"]
    time_taken_by_engine = engine_output["time_take"]
    utils.append_log(f"Engine decided move: {move_to_make}, premove: {premove_to_make}, time: {time_taken_by_engine:.3f}s")

    # --- Calculate Delay ---
    time_spent_so_far = time.time() - start_turn_time
    remaining_delay = time_taken_by_engine - time_spent_so_far - constants.MOVE_DELAY
    if remaining_delay > 0:
        utils.append_log(f"Waiting additional {remaining_delay:.3f}s before making move.")
        time.sleep(remaining_delay)

    # --- Make the Move ---
    made_move = mouse.make_move(move_to_make, premove=premove_to_make)
    if not made_move:
        utils.append_log("Main move failed (mouse slip?). Trying again.")
        time.sleep(0.1)
        mouse.make_move(move_to_make, premove=premove_to_make) # Try once more

    utils.append_log("--- Turn End ---")
    utils.write_log()


def run_game():
    """
    The main loop for a single game.

    Alternates between waiting for the opponent and playing our turn.
    """
    utils.append_log("Starting game loop...")
    engine_adapter.initialize_engine() # Ensure engine is ready
    
    # Start the pondering thread at the beginning of the game
    threading_manager.start_ponder_thread()
    utils.append_log("Pondering thread started for this game.\n")

    try:
        while True:
            # Check for manual override first
            if utils.is_capslock_on():
                utils.append_log("Manual mode (Caps Lock) detected. Pausing game loop.")
                time.sleep(1)
                utils.write_log() # Write logs periodically even when paused
                continue

            # Wait for opponent / Update state
            is_our_turn_now = await_opponent_move()
            utils.write_log() # Write logs accumulated during waiting

            if not is_our_turn_now:
                # Game has ended
                utils.append_log("Game has ended. Exiting run_game loop.")
                break

            # It's our turn
            play_our_turn()
            utils.write_log() # Write logs accumulated during our turn

            # Check if game ended immediately after our move (e.g., checkmate)
            if state.is_game_over() or vision.check_game_over_vision():
                 utils.append_log("Game ended immediately after our move.\n")
                 break

        utils.append_log("Game loop finished.")
    finally:
        # Make sure to stop the pondering thread when the game ends
        threading_manager.stop_ponder_thread()
        utils.append_log("Pondering thread stopped at end of game.\n")
        utils.write_log()


# --- Main Execution ---

if __name__ == "__main__":
    print("Refactored Chess Client Starting...")
    print(f"Log file: {constants.LOG_FILE}")
    utils.append_log("--- Client Start ---\n")
    utils.append_log(f"Timestamp: {datetime.datetime.now()}\n")
    utils.append_log(f"Difficulty: {constants.DIFFICULTY}, Quickness: {constants.QUICKNESS}, Mouse Quickness: {constants.MOUSE_QUICKNESS}\n")
    utils.write_log()

    # Replicate original main loop structure (lines 1242-1254)
    num_games_to_play = 6 # Example: Play 6 games
    time_control = "1+0" # Example time control

    for i in range(num_games_to_play):
        print(f"\n--- Starting Game {i+1} of {num_games_to_play} ---")
        utils.append_log(f"--- Starting Game {i+1} of {num_games_to_play} ---\n")
        utils.write_log()

        # Wait for a new game to be available
        initial_time = await_new_game(timeout=120) # Increased timeout for finding games
        utils.write_log()

        if initial_time is not None:
            # Setup game state
            setup_successful = set_up_game(initial_time)
            utils.write_log()

            if setup_successful:
                # Run the main game loop
                run_game()
                print(f"--- Game {i+1} Finished ---")
                utils.append_log(f"--- Game {i+1} Finished ---\n")
                utils.write_log()
            else:
                 print(f"Game {i+1} setup failed. Skipping game.")
                 utils.append_log(f"Game {i+1} setup failed. Skipping game.\n")
                 utils.write_log()

        else:
            print("Failed to find a new game within the timeout period.")
            utils.append_log("Failed to find a new game within the timeout period. Stopping.\n")
            utils.write_log()
            break # Stop if no game is found

        # If not the last game, try to start a new one
        if i < num_games_to_play - 1:
            print(f"Attempting to start next game ({time_control})...")
            utils.append_log(f"Attempting to start next game ({time_control})...\n")
            if not mouse.new_game(time_control=time_control):
                 utils.append_log("Failed to click 'new game' button (Caps Lock?). Stopping.\n")
                 print("Failed to click 'new game' button (Caps Lock?). Stopping.")
                 utils.write_log()
                 break
            time.sleep(2) # Give time for the new game screen to load
            utils.write_log()
        else:
            print("\nAll requested games played.")
            utils.append_log("--- All Games Played ---\n")
            utils.write_log()

    print("Chess Client Exiting.")
    utils.append_log("--- Client End ---\n")
    utils.write_log()