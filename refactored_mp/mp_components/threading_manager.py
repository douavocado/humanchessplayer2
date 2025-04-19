#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threading manager for the refactored multiprocessing chess client.

Handles running engine pondering in a separate thread to improve responsiveness.
"""

import threading
import time
from queue import Queue, Empty
import logging

# Import from mp_components
from . import engine_adapter
from . import state
from . import utils

# Global thread control variables
ponder_thread = None
ponder_command_queue = Queue()
ponder_active = False
ponder_thread_lock = threading.Lock()

def start_ponder_thread():
    """
    Starts the engine pondering thread if it's not already running.
    Returns True if started, False if already running.
    """
    global ponder_thread, ponder_active
    
    with ponder_thread_lock:
        if ponder_thread is not None and ponder_thread.is_alive():
            utils.append_log("Ponder thread already running.\n")
            return False
            
        utils.append_log("Starting engine ponder thread...\n")
        ponder_active = True
        ponder_thread = threading.Thread(target=_ponder_thread_func, daemon=True)
        ponder_thread.start()
        return True

def stop_ponder_thread():
    """
    Signals the pondering thread to stop.
    Returns True if signal was sent, False if thread was not running.
    """
    global ponder_active, ponder_command_queue
    
    with ponder_thread_lock:
        if ponder_thread is None or not ponder_thread.is_alive():
            utils.append_log("No ponder thread running to stop.\n")
            return False
            
        utils.append_log("Stopping engine ponder thread...\n")
        ponder_active = False
        ponder_command_queue.put("STOP")
        return True

def ponder_now():
    """
    Signals the pondering thread to run a pondering operation now.
    Returns True if signal was sent, False if thread was not running.
    """
    global ponder_command_queue
    
    with ponder_thread_lock:
        if ponder_thread is None or not ponder_thread.is_alive():
            utils.append_log("No ponder thread running to signal.\n")
            return False
            
        ponder_command_queue.put("PONDER")
        return True

def is_pondering():
    """Returns True if the ponder thread is active."""
    with ponder_thread_lock:
        return ponder_thread is not None and ponder_thread.is_alive() and ponder_active

def _ponder_thread_func():
    """
    The main function for the pondering thread.
    Runs engine pondering in a loop, checking for stop signals.
    """
    global ponder_active, ponder_command_queue
    
    utils.append_log("Ponder thread started.\n")
    last_ponder_time = 0
    
    while ponder_active:
        try:
            # Check for commands with a short timeout
            try:
                cmd = ponder_command_queue.get(block=True, timeout=0.1)
                if cmd == "STOP":
                    utils.append_log("Ponder thread received stop command.\n")
                    break
                elif cmd == "PONDER":
                    utils.append_log("Ponder thread received immediate ponder command.\n")
                    # Force pondering now by resetting last_ponder_time
                    last_ponder_time = 0
            except Empty:
                pass  # No command, continue normal operation
            
            # Only ponder if it's the opponent's turn
            if not state.is_our_turn() and not state.is_game_over():
                current_time = time.time()
                # Don't ponder too frequently
                if current_time - last_ponder_time >= 0.5:  # Minimum 0.5s between ponders
                    engine_adapter.run_engine_ponder()
                    last_ponder_time = current_time
            
            # Short sleep to prevent excessive CPU usage
            time.sleep(0.05)
            
        except Exception as e:
            utils.append_log(f"ERROR in ponder thread: {e}\n")
            time.sleep(1)  # Sleep longer on error to avoid spinning
    
    utils.append_log("Ponder thread stopped.\n") 