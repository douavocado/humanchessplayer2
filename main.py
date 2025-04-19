#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:22:09 2025

@author: james
"""
import time
import sys
import random
import signal
import argparse

# from mp_original import run_game, await_new_game, set_game, new_game, back_to_lobby, berserk
from refactored_mp.run_mp import run_game, await_new_game, set_up_game # Renamed set_game to set_up_game
from refactored_mp.mp_components import mouse # Import the mouse module for button clicks
from refactored_mp.mp_components import threading_manager # Import threading manager for clean shutdown
from common.constants import DIFFICULTY, QUICKNESS, MOUSE_QUICKNESS

# Set up signal handlers for clean shutdown
def signal_handler(sig, frame):
    print("\nShutting down chess client...")
    # Stop any running ponder threads
    threading_manager.stop_ponder_thread()
    print("Threads stopped. Exiting.")
    sys.exit(0)

# Register the signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # Handle termination signal

parser = argparse.ArgumentParser(description="Setting game mode")
parser.add_argument("-t", "--time", type=int, default=60,
                    help="The time control in seconds")
parser.add_argument("-i", "--increment", type=int, default=0,
                    help="The time control increcrement in seconds")
parser.add_argument("-g", "--games", type=int, default=5,
                    help="The number of games to play of the specified time control. Only used when not in tournament mode.")
parser.add_argument("-a", "--arena", help="Tournament arena mode",
                        action="store_true")
parser.add_argument("-b", "--berserk", help="Always beserk in tournament arena mode",
                        action="store_true")
args = parser.parse_args()

print("Engine Difficulty: {}, Quickness: {}, Mouse Quickness: {}".format(DIFFICULTY, QUICKNESS, MOUSE_QUICKNESS))

try:
    if args.arena == True:
        # in tournament mode
        while True:
            time.sleep(0.5)
            res = await_new_game(timeout=300)
            if res is not None:
                setup_successful = set_up_game(res) # Use the new setup function
                if setup_successful:
                    if args.berserk:
                        # beserk mode
                        mouse.berserk() # Use function from mouse module
                        time.sleep(0.5)
                    run_game() # This function remains the same from run_mp
                    print("Finished tournament game.")
                    # go back to lobby. This can be done by clicking where the resign button is once
                    time.sleep(random.randint(1,3))
                    mouse.back_to_lobby() # Use function from mouse module
                else:
                    print("Game setup failed, skipping tournament game.")
    else:
        if args.time == 60:
            tc_str = "1+0"
        elif args.time == 180:
            tc_str = "3+0"
        else:
            raise Exception("Time control not recognised: {}".format(args.time))
        games = args.games
        for i in range(games):
            time.sleep(0.5)
            res = await_new_game()
            if res is not None:
                setup_successful = set_up_game(res) # Use the new setup function
                if setup_successful:
                    run_game() # This function remains the same from run_mp
                    print("Finished game {}".format(i+1))
                    if i < games-1:
                        mouse.new_game(tc_str) # Use function from mouse module
                elif i < games-1:
                    print("Game setup failed, trying to seek again.")
                    mouse.new_game(tc_str) # Use function from mouse module
                else:
                    print("Game setup failed for the last game.")
finally:
    # Make sure threading resources are cleaned up when the program exits
    threading_manager.stop_ponder_thread()
    print("Chess client shut down cleanly.")
    sys.exit()