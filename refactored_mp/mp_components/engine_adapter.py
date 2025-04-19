#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter module for interacting with the chess engine (engine.py).

Handles initializing the engine, sending game state updates, requesting moves,
managing pondering, and checking for resignation conditions.
"""

import chess
import random
import time

# Assuming the original Engine class is accessible
# Adjust the import path if engine.py is located elsewhere
try:
    from engine import Engine
except ImportError:
    print("ERROR: Could not import Engine class. Ensure engine.py is in the Python path.")
    # Provide a dummy class to avoid crashing, but functionality will be broken
    class Engine:
        def __init__(self, *args, **kwargs): print("Dummy Engine Initialized")
        def update_info(self, *args, **kwargs): print("Dummy Engine Update Info")
        def make_move(self, *args, **kwargs): return {"move_made": "a1a2", "premove": None, "ponder_dic": {}, "time_take": 0.1} # Dummy move
        def decide_resign(self, *args, **kwargs): return False
        def ponder(self, *args, **kwargs): return {}
        def stockfish_ponder(self, *args, **kwargs): return {}

# Import from refactored components
from .constants import DIFFICULTY, QUICKNESS, MOVE_DELAY
from .state import DYNAMIC_INFO, GAME_INFO, PONDER_DIC, update_ponder_dic, get_current_board
from .utils import append_log, write_log
from common.utils import check_safe_premove # Assuming accessible

# Global engine instance
ENGINE_INSTANCE: Engine | None = None

def initialize_engine():
    """Initializes the global chess engine instance."""
    global ENGINE_INSTANCE
    if ENGINE_INSTANCE is None:
        append_log(f"Initializing chess engine with difficulty level: {DIFFICULTY}\n")
        ENGINE_INSTANCE = Engine(playing_level=DIFFICULTY)
        append_log("Engine initialized.\n")
    else:
        append_log("Engine already initialized.\n")

def get_engine() -> Engine:
    """Returns the initialized engine instance, initializing if necessary."""
    if ENGINE_INSTANCE is None:
        initialize_engine()
    # Add a type check just in case initialization failed silently
    if not isinstance(ENGINE_INSTANCE, Engine):
         raise RuntimeError("Engine instance is not correctly initialized.")
    return ENGINE_INSTANCE

def update_engine_state():
    """Formats and sends the current game state to the engine."""
    engine = get_engine()
    try:
        # Prepare the input dictionary for the engine
        input_dic = DYNAMIC_INFO.copy() # Use a copy to avoid modifying the original state dict
        input_dic["side"] = GAME_INFO.get("playing_side")
        input_dic["self_initial_time"] = GAME_INFO.get("self_initial_time")
        input_dic["opp_initial_time"] = GAME_INFO.get("opp_initial_time")

        # Basic validation before sending
        if input_dic["side"] is None or not DYNAMIC_INFO["fens"]:
            append_log("Warning: Cannot update engine state - insufficient game info (side or fens missing).\n")
            return False

        append_log("Sending updated state to engine.\n")
        engine.update_info(input_dic)
        return True
    except Exception as e:
        append_log(f"ERROR updating engine state: {e}\n")
        return False


def ask_engine_for_move() -> dict | None:
    """
    Asks the engine to calculate and return the best move for the current position.

    Returns:
        A dictionary containing engine output (move_made, premove, ponder_dic, time_take)
        or None if an error occurs or engine cannot provide a move.
    """
    engine = get_engine()
    try:
        append_log("Requesting move from engine...\n")
        start_time = time.time()
        output_dic = engine.make_move()
        end_time = time.time()
        append_log(f"Engine returned move: {output_dic}\n")
        append_log(f"Time taken by engine.make_move(): {end_time - start_time:.3f}s\n")

        # Validate output_dic structure (basic check)
        if not all(k in output_dic for k in ["move_made", "premove", "ponder_dic", "time_take"]):
             append_log(f"ERROR: Engine output dictionary is missing expected keys: {output_dic}\n")
             return None # Or return a default dict if preferred

        # Update global ponder dictionary if engine provided one
        if output_dic.get("ponder_dic"):
            update_ponder_dic(output_dic["ponder_dic"])

        return output_dic

    except Exception as e:
        append_log(f"ERROR requesting move from engine: {e}\n")
        return None


def check_engine_resign() -> bool:
    """Asks the engine if resignation is appropriate for the current state."""
    engine = get_engine()
    try:
        resign_decision = engine.decide_resign()
        if resign_decision:
            append_log("Engine recommends resignation.\n")
        return resign_decision
    except Exception as e:
        append_log(f"ERROR checking engine resignation: {e}\n")
        return False # Default to not resigning on error


def run_engine_ponder():
    """
    Performs pondering on the current position if it's the opponent's turn.
    Updates the global PONDER_DIC with results.
    """
    engine = get_engine()
    try:
        current_board = get_current_board()
        if current_board is None or current_board.turn == GAME_INFO.get("playing_side"):
            # Not opponent's turn, or board state unavailable
            return

        if len(DYNAMIC_INFO["fens"]) < 2: # Need previous board for context sometimes
             # append_log("Pondering skipped: Not enough FEN history.\n")
             return

        prev_board = chess.Board(DYNAMIC_INFO["fens"][-2]) if len(DYNAMIC_INFO["fens"]) >= 2 else None

        own_time = DYNAMIC_INFO["self_clock_times"][-1] if DYNAMIC_INFO["self_clock_times"] else 60

        ponder_result = None
        if own_time < 10: # Use faster Stockfish ponder if low on time
            time_allowed = 0.05
            ponder_width = 2 # Consider fewer opponent responses
            append_log(f"Pondering (Stockfish, {time_allowed}s): {current_board.fen()}\n")
            # Randomly sample opponent moves to limit search space (as in original)
            legal_moves = list(current_board.legal_moves)
            sample_no = max(int(len(legal_moves) / 2), 1)
            root_moves = random.sample(legal_moves, sample_no) if legal_moves else []
            append_log(f"Stockfish ponder considering opponent moves: {root_moves}\n")
            ponder_result = engine.stockfish_ponder(current_board, time_allowed, ponder_width, use_ponder=True, root_moves=root_moves)
        else: # Use default engine ponder
            # Adjust time allowed based on initial time? Original used initial_time/60
            initial_time = GAME_INFO.get("self_initial_time", 60)
            time_allowed = max(initial_time / 60, 0.1) # Ensure some minimum time
            ponder_width = 1 # Consider only the top opponent move for our response
            search_width = DIFFICULTY # Use main difficulty setting
            append_log(f"Pondering (Default, {time_allowed:.2f}s): {current_board.fen()}\n")
            ponder_result = engine.ponder(current_board, time_allowed, search_width, prev_board=prev_board, ponder_width=ponder_width, use_ponder=True)

        if ponder_result:
            update_ponder_dic(ponder_result)
            # write_log() # Write log immediately after ponder update?

    except Exception as e:
        append_log(f"ERROR during engine pondering: {e}\n")
        # Optionally write log here on error too
        # write_log()


def get_ponder_move() -> str | None:
    """
    Checks if the current board state exists in the PONDER_DIC.

    Returns:
        The corresponding pondered move UCI string if found, otherwise None.
    """
    current_board = get_current_board()
    if current_board is None:
        return None

    board_fen_key = current_board.board_fen() # Use only board part as key
    response_uci = PONDER_DIC.get(board_fen_key)

    if response_uci:
        append_log(f"Found current position ({board_fen_key}) in ponder dic. Response: {response_uci}\n")
        # Optional: Validate if the move is legal in the current full board state
        try:
            move = chess.Move.from_uci(response_uci)
            if move not in current_board.legal_moves:
                append_log(f"Warning: Pondered move {response_uci} is illegal in current state {current_board.fen()}. Ignoring.\n")
                # Optionally remove the invalid entry from PONDER_DIC
                # del PONDER_DIC[board_fen_key]
                return None
        except ValueError:
             append_log(f"Warning: Invalid UCI '{response_uci}' found in PONDER_DIC. Ignoring.\n")
             return None
        return response_uci
    else:
        return None


def get_safe_ponder_premove() -> str | None:
    """
    Checks if the last pondered move (from the previous state) is a safe premove
    in the current state and returns it if conditions are met (e.g., low time).
    This replicates the logic for playing pondered moves even if the exact position wasn't hit.

    Returns:
        The UCI of the safe premove if applicable, otherwise None.
    """
    try:
        if len(DYNAMIC_INFO["fens"]) < 2 or not PONDER_DIC:
            return None # Need previous state and ponder history

        last_board = chess.Board(DYNAMIC_INFO["fens"][-2])
        current_board = get_current_board()
        if current_board is None: return None

        # Get the most recently added ponder move
        # Note: dicts are ordered in Python 3.7+, but relying on insertion order might be fragile.
        # A more robust approach might store ponder moves with timestamps or sequence numbers.
        # For now, sticking to original logic's apparent intent:
        last_pondered_uci = list(PONDER_DIC.values())[-1]
        last_pondered_move_obj = chess.Move.from_uci(last_pondered_uci)

        # Check if the move is legal *now*
        if last_pondered_move_obj in current_board.legal_moves:
            # Check if it was considered safe in the *previous* state
            if check_safe_premove(last_board, last_pondered_uci):
                # Check time conditions (more likely in faster time controls)
                initial_time = GAME_INFO.get("self_initial_time", 60)
                prob = np.sqrt(10 / max(initial_time, 1)) # Avoid division by zero
                if initial_time < 200 and np.random.random() < prob:
                    append_log(f"Last ponder move {last_pondered_uci} was safe premove. Playing by chance.\n")
                    return last_pondered_uci

            # Additional check for very low time (original logic lines 1133-1163)
            own_time = DYNAMIC_INFO["self_clock_times"][-1] if DYNAMIC_INFO["self_clock_times"] else 60
            if own_time < 10:
                 prob_low_time = (30 - own_time) / 50
                 # Consider last few ponder moves if very low on time
                 candidate_ucis = list(PONDER_DIC.values())[-10:]
                 for uci in candidate_ucis:
                     try:
                         move_obj = chess.Move.from_uci(uci)
                         if move_obj in current_board.legal_moves:
                             # Play if it was safe OR by random chance due to low time
                             if check_safe_premove(last_board, uci) or np.random.random() < prob_low_time:
                                 append_log(f"Playing potentially unsafe ponder move {uci} due to low time ({own_time}s).\n")
                                 return uci
                     except ValueError: continue # Ignore invalid UCIs

    except (ValueError, IndexError, KeyError, TypeError) as e:
        append_log(f"Warning: Error during safe ponder premove check: {e}\n")

    return None # No applicable safe ponder premove found