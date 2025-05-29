#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manages the state of the chess client application.

Includes static game information, dynamic updates during the game,
castling rights, pondering results, and UI state like hover square.
"""

import chess
from .constants import FEN_NO_CAP
from .utils import append_log
from common.utils import patch_fens # Assuming this utility is accessible

# --- Static Game Information (Set once per game) ---
GAME_INFO = {
    "playing_side": None,       # chess.WHITE or chess.BLACK
    "self_initial_time": None,  # Initial time in seconds
    "opp_initial_time": None,   # Initial time in seconds
}

# --- Dynamic Game Information (Updated during the game) ---
DYNAMIC_INFO = {
    "fens": [],                 # List of recent FEN strings (up to FEN_NO_CAP)
    "self_clock_times": [],     # List of recent self clock times (seconds)
    "opp_clock_times": [],      # List of recent opponent clock times (seconds)
    "last_moves": [],           # List of recent moves in UCI format
}

# --- Castling Rights ---
# Stores the current castling rights FEN string (e.g., "KQkq", "Kq", "-")
CASTLING_RIGHTS_FEN = "KQkq"

# --- Pondering ---
# Dictionary storing pondered moves: {board_fen: best_response_uci}
PONDER_DIC = {}

# --- UI State ---
# The square the mouse is currently hovering over (chess.Square or None)
HOVER_SQUARE = None

# --- State Management Functions ---

def reset_game_state(playing_side: chess.Color, initial_time: float, starting_fen: str):
    """
    Resets the game state variables for a new game.

    Args:
        playing_side: The color the bot is playing (chess.WHITE or chess.BLACK).
        initial_time: The starting time control in seconds.
        starting_fen: The FEN string of the initial position.
    """
    global GAME_INFO, DYNAMIC_INFO, CASTLING_RIGHTS_FEN, PONDER_DIC, HOVER_SQUARE

    append_log("Resetting game state for new game.\n")

    # Reset static info
    GAME_INFO["playing_side"] = playing_side
    GAME_INFO["self_initial_time"] = initial_time
    GAME_INFO["opp_initial_time"] = initial_time # Assume opponent has same time initially

    # Reset dynamic info
    # Handle starting FEN logic similar to original script
    if playing_side == chess.WHITE or chess.Board(starting_fen).board_fen() == chess.STARTING_BOARD_FEN:
         DYNAMIC_INFO["fens"] = [starting_fen]
         DYNAMIC_INFO["last_moves"] = []
    else: # Black and not starting position - assume opponent moved first
         DYNAMIC_INFO["fens"] = [chess.STARTING_FEN, starting_fen]
         # Try to find the move linking STARTING_FEN and starting_fen
         res = patch_fens(chess.STARTING_FEN, starting_fen, depth_lim=1)
         if res is not None:
             DYNAMIC_INFO["last_moves"] = res[0] # res[0] should be the list of moves
             # Update fens if patch_fens corrected them (though depth_lim=1 shouldn't)
             if len(res) > 1 and res[1]:
                 DYNAMIC_INFO["fens"] = [chess.STARTING_FEN] + res[1]
         else:
             append_log(f"ERROR: Couldn't find linking move between {chess.STARTING_FEN} and {starting_fen}.\n")
             DYNAMIC_INFO["last_moves"] = [] # Clear moves if patching failed

    DYNAMIC_INFO["self_clock_times"] = [initial_time]
    DYNAMIC_INFO["opp_clock_times"] = [initial_time]

    # Reset castling rights (will be updated by FEN parsing later if needed)
    # The initial FEN should contain the correct rights, but we start assuming all.
    CASTLING_RIGHTS_FEN = "KQkq"
    # Apply initial castling rights from the starting FEN
    try:
        initial_board = chess.Board(starting_fen)
        CASTLING_RIGHTS_FEN = initial_board.castling_fen() if initial_board.castling_fen() else "-"
        append_log(f"Initial castling rights set from FEN: {CASTLING_RIGHTS_FEN}\n")
    except ValueError:
        append_log(f"Warning: Could not parse initial FEN '{starting_fen}' for castling rights. Assuming 'KQkq'.\n")
        CASTLING_RIGHTS_FEN = "KQkq"


    # Reset pondering and hover state
    PONDER_DIC = {}
    HOVER_SQUARE = None

    # Trim lists to capacity just in case
    trim_dynamic_info()

    append_log("Game state reset complete.\n")
    append_log(f"GAME_INFO: {GAME_INFO}\n")
    append_log(f"DYNAMIC_INFO (initial): {DYNAMIC_INFO}\n")


def update_dynamic_info(fen: str | None = None,
                        self_time: float | None = None,
                        opp_time: float | None = None,
                        move_uci: str | None = None):
    """
    Updates the dynamic game state information.

    Adds new FEN, clock times, and move, maintaining list capacities.
    Handles potential inconsistencies and logs errors.

    Args:
        fen: The new FEN string to add.
        self_time: The new clock time for the bot.
        opp_time: The new clock time for the opponent.
        move_uci: The UCI string of the move that led to the new state.
    """
    global DYNAMIC_INFO, CASTLING_RIGHTS_FEN

    updated = False
    log_msg = "Updating dynamic info:"

    if fen:
        # Basic validation: check if it's different from the last known FEN
        if not DYNAMIC_INFO["fens"] or fen != DYNAMIC_INFO["fens"][-1]:
            # More robust check: compare board state and turn, ignore move counters etc.
            try:
                new_board = chess.Board(fen)
                last_board = chess.Board(DYNAMIC_INFO["fens"][-1]) if DYNAMIC_INFO["fens"] else None
                if last_board is None or \
                   new_board.board_fen() != last_board.board_fen() or \
                   new_board.turn != last_board.turn:

                    # Update castling rights from the new FEN
                    new_castling = new_board.castling_fen() if new_board.castling_fen() else "-"
                    if new_castling != CASTLING_RIGHTS_FEN:
                         log_msg += f" Castling rights changed from '{CASTLING_RIGHTS_FEN}' to '{new_castling}'."
                         CASTLING_RIGHTS_FEN = new_castling

                    DYNAMIC_INFO["fens"].append(fen)
                    log_msg += f" Added FEN {fen}."
                    updated = True
                else:
                    log_msg += " New FEN is functionally identical to last, not adding."
            except ValueError:
                append_log(f"ERROR: Invalid FEN received for update: {fen}\n")
        else:
             log_msg += " New FEN is identical to last, not adding."


    # Determine whose clock time to update based on the *new* FEN's turn
    # If it's our turn in the new FEN, the opponent just moved.
    # If it's opponent's turn, we just moved.
    if fen:
        try:
            board = chess.Board(fen)
            if board.turn == GAME_INFO["playing_side"]: # Opponent just moved
                if opp_time is not None and (not DYNAMIC_INFO["opp_clock_times"] or opp_time != DYNAMIC_INFO["opp_clock_times"][-1]):
                    DYNAMIC_INFO["opp_clock_times"].append(opp_time)
                    log_msg += f" Added Opponent Time {opp_time}."
                    updated = True
                    # Check for opponent berserk
                    check_berserk(opponent=True)
            else: # We just moved
                if self_time is not None and (not DYNAMIC_INFO["self_clock_times"] or self_time != DYNAMIC_INFO["self_clock_times"][-1]):
                    DYNAMIC_INFO["self_clock_times"].append(self_time)
                    log_msg += f" Added Self Time {self_time}."
                    updated = True
                    # Check for self berserk
                    check_berserk(opponent=False)
        except ValueError:
             append_log(f"ERROR: Could not parse FEN {fen} to determine turn for clock update.\n")
        except KeyError:
             append_log("ERROR: GAME_INFO['playing_side'] not set, cannot update clocks correctly.\n")

    if move_uci:
        # Basic validation
        try:
            chess.Move.from_uci(move_uci)
            DYNAMIC_INFO["last_moves"].append(move_uci)
            log_msg += f" Added Move {move_uci}."
            updated = True
            # Update castling rights based on the move made *before* this state
            # This requires the previous FEN.
            if len(DYNAMIC_INFO["fens"]) >= 2:
                 update_castling_from_move(move_uci, DYNAMIC_INFO["fens"][-2])

        except ValueError:
            append_log(f"ERROR: Invalid move UCI received for update: {move_uci}\n")


    if updated:
        trim_dynamic_info()
        append_log(log_msg + "\n")
        # append_log(f"Current DYNAMIC_INFO: {DYNAMIC_INFO}\n") # Optional: Log full state
    # else:
    #     append_log("No changes detected in update_dynamic_info call.\n")


def trim_dynamic_info():
    """Ensures the lists in DYNAMIC_INFO do not exceed FEN_NO_CAP."""
    global DYNAMIC_INFO
    DYNAMIC_INFO["fens"] = DYNAMIC_INFO["fens"][-FEN_NO_CAP:]
    DYNAMIC_INFO["self_clock_times"] = DYNAMIC_INFO["self_clock_times"][-FEN_NO_CAP:]
    DYNAMIC_INFO["opp_clock_times"] = DYNAMIC_INFO["opp_clock_times"][-FEN_NO_CAP:]
    # last_moves should have one less element than fens
    DYNAMIC_INFO["last_moves"] = DYNAMIC_INFO["last_moves"][-(FEN_NO_CAP - 1):]


def update_castling_from_move(move_uci: str, fen_before_move: str):
    """
    Updates the global CASTLING_RIGHTS_FEN based on a move that just occurred.

    Args:
        move_uci: The UCI string of the move that was just made.
        fen_before_move: The FEN string *before* the move was made.
    """
    global CASTLING_RIGHTS_FEN
    if CASTLING_RIGHTS_FEN == "-": # No rights left to lose
        return

    try:
        board = chess.Board(fen_before_move)
        move = chess.Move.from_uci(move_uci)

        original_rights = CASTLING_RIGHTS_FEN

        # Check for king moves
        if board.piece_type_at(move.from_square) == chess.KING:
            if board.color_at(move.from_square) == chess.WHITE:
                CASTLING_RIGHTS_FEN = CASTLING_RIGHTS_FEN.replace("K", "").replace("Q", "")
            else:
                CASTLING_RIGHTS_FEN = CASTLING_RIGHTS_FEN.replace("k", "").replace("q", "")

        # Check for rook moves or captures on rook squares
        # White Kingside (H1)
        if move.from_square == chess.H1 or move.to_square == chess.H1:
            CASTLING_RIGHTS_FEN = CASTLING_RIGHTS_FEN.replace("K", "")
        # White Queenside (A1)
        if move.from_square == chess.A1 or move.to_square == chess.A1:
            CASTLING_RIGHTS_FEN = CASTLING_RIGHTS_FEN.replace("Q", "")
        # Black Kingside (H8)
        if move.from_square == chess.H8 or move.to_square == chess.H8:
            CASTLING_RIGHTS_FEN = CASTLING_RIGHTS_FEN.replace("k", "")
        # Black Queenside (A8)
        if move.from_square == chess.A8 or move.to_square == chess.A8:
            CASTLING_RIGHTS_FEN = CASTLING_RIGHTS_FEN.replace("q", "")

        if not CASTLING_RIGHTS_FEN:
            CASTLING_RIGHTS_FEN = "-"

        if original_rights != CASTLING_RIGHTS_FEN:
             append_log(f"Castling rights updated due to move {move_uci}: {original_rights} -> {CASTLING_RIGHTS_FEN}\n")

    except (ValueError, TypeError): # Handle invalid FEN or UCI
        append_log(f"ERROR: Could not update castling rights from move {move_uci} and FEN {fen_before_move}\n")


def check_berserk(opponent: bool):
    """
    Checks if a player likely berserked based on time and move number.
    Updates GAME_INFO if berserk is detected.

    Args:
        opponent: True to check opponent, False to check self.
    """
    global GAME_INFO
    try:
        if not DYNAMIC_INFO["fens"]: return # Need FEN for move number

        current_board = chess.Board(DYNAMIC_INFO["fens"][-1])
        move_number = current_board.fullmove_number

        if move_number < 5: # Only check in the first few moves
            if opponent:
                if not DYNAMIC_INFO["opp_clock_times"] or GAME_INFO["opp_initial_time"] is None: return
                current_time = DYNAMIC_INFO["opp_clock_times"][-1]
                initial_time = GAME_INFO["opp_initial_time"]
                if current_time < initial_time / 2:
                    new_initial_time = initial_time / 2
                    if GAME_INFO["opp_initial_time"] != new_initial_time: # Avoid repeated logs
                        append_log(f"Opponent detected to have BESERKED, reducing initial time from {initial_time} to {new_initial_time}\n")
                        print(f"Opponent detected to have BESERKED, reducing initial time from {initial_time} to {new_initial_time}")
                        GAME_INFO["opp_initial_time"] = new_initial_time
            else: # Check self
                if not DYNAMIC_INFO["self_clock_times"] or GAME_INFO["self_initial_time"] is None: return
                current_time = DYNAMIC_INFO["self_clock_times"][-1]
                initial_time = GAME_INFO["self_initial_time"]
                if current_time < initial_time / 2:
                    new_initial_time = initial_time / 2
                    if GAME_INFO["self_initial_time"] != new_initial_time:
                        append_log(f"Detected to have BESERKED, reducing self initial time from {initial_time} to {new_initial_time}\n")
                        print(f"Detected to have BESERKED, reducing self initial time from {initial_time} to {new_initial_time}")
                        GAME_INFO["self_initial_time"] = new_initial_time
    except (ValueError, IndexError, KeyError, TypeError) as e:
        append_log(f"Warning: Error during berserk check: {e}\n")


def update_ponder_dic(new_ponder_data: dict):
    """Updates the global PONDER_DIC with new entries."""
    global PONDER_DIC
    if new_ponder_data:
        PONDER_DIC.update(new_ponder_data)
        append_log(f"Ponder dictionary updated with {len(new_ponder_data)} new entries.\n")
        # append_log(f"Current PONDER_DIC size: {len(PONDER_DIC)}\n") # Optional: Log size

def clear_ponder_dic():
    """Clears the global PONDER_DIC."""
    global PONDER_DIC
    PONDER_DIC = {}
    append_log("Ponder dictionary cleared.\n")

def update_hover_square(square: chess.Square | None):
    """Updates the global HOVER_SQUARE."""
    global HOVER_SQUARE
    if HOVER_SQUARE != square:
        HOVER_SQUARE = square
        # append_log(f"Hover square updated to: {chess.square_name(square) if square is not None else 'None'}\n")


def get_current_board() -> chess.Board | None:
    """Returns the latest board object from the dynamic info, or None if unavailable."""
    if DYNAMIC_INFO["fens"]:
        try:
            # Ensure castling rights are applied from our tracked state, not just the FEN
            board = chess.Board(DYNAMIC_INFO["fens"][-1])
            # board.set_castling_fen(CASTLING_RIGHTS_FEN) # Apply potentially corrected rights
            return board
        except ValueError:
            append_log(f"ERROR: Could not parse latest FEN: {DYNAMIC_INFO['fens'][-1]}\n")
            return None
    return None

def is_our_turn() -> bool | None:
    """Checks if it's currently our turn based on the latest FEN. Returns None on error."""
    board = get_current_board()
    if board is None:
        return None # Error case
    try:
        return board.turn == GAME_INFO["playing_side"]
    except KeyError:
        append_log("ERROR: GAME_INFO['playing_side'] not set, cannot determine turn.\n")
        return None # Error case

def is_game_over() -> bool:
    """Checks if the game has ended based on the latest board state."""
    board = get_current_board()
    if board and board.outcome() is not None:
        return True
    return False