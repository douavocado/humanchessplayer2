#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:22:09 2025

@author: james
"""
import time
import sys
import random

import argparse

# Import only the constants module first, delay other imports
import common.constants as constants

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
parser.add_argument("-d", "--difficulty", type=int, default=None,
                    help="Engine difficulty level (overrides default from constants)")
parser.add_argument("-q", "--quickness", type=float, default=None,
                    help="Engine quickness (overrides default from constants)")
parser.add_argument("-m", "--mouse-quickness", type=float, default=None,
                    help="Mouse quickness (overrides default from constants)")
args = parser.parse_args()

# Set the values, using command line arguments if provided, otherwise use defaults from constants
DIFFICULTY = args.difficulty if args.difficulty is not None else constants.DIFFICULTY
QUICKNESS = args.quickness if args.quickness is not None else constants.QUICKNESS
MOUSE_QUICKNESS = args.mouse_quickness if args.mouse_quickness is not None else constants.MOUSE_QUICKNESS

# Update the constants module with the final values so other modules can access them
constants.DIFFICULTY = DIFFICULTY
constants.QUICKNESS = QUICKNESS
constants.MOUSE_QUICKNESS = MOUSE_QUICKNESS

# Now import mp_original after constants are set
from clients.mp_original import run_game, await_new_game, set_game, new_game, back_to_lobby, berserk

def verify_and_patch_constants():
    """Verify that other modules can see the overridden constants and patch them if needed"""
    import clients.mp_original as mp_original
    
    # Patch the constants in mp_original module
    mp_original.DIFFICULTY = DIFFICULTY
    mp_original.QUICKNESS = QUICKNESS  
    mp_original.MOUSE_QUICKNESS = MOUSE_QUICKNESS
    
    # Re-initialize the engine with the correct difficulty if it was overridden
    if args.difficulty is not None:
        from engine import Engine
        mp_original.ENGINE = Engine(playing_level=DIFFICULTY)
        print(f"Re-initialized engine with difficulty: {DIFFICULTY}")
    
    print("\n--- Constants Verification ---")
    print(f"main.py - DIFFICULTY: {DIFFICULTY}, QUICKNESS: {QUICKNESS}, MOUSE_QUICKNESS: {MOUSE_QUICKNESS}")
    print(f"constants module - DIFFICULTY: {constants.DIFFICULTY}, QUICKNESS: {constants.QUICKNESS}, MOUSE_QUICKNESS: {constants.MOUSE_QUICKNESS}")
    print(f"mp_original module - DIFFICULTY: {mp_original.DIFFICULTY}, QUICKNESS: {mp_original.QUICKNESS}, MOUSE_QUICKNESS: {mp_original.MOUSE_QUICKNESS}")
    print(f"mp_original.ENGINE.playing_level: {mp_original.ENGINE.playing_level}")
    print("--- End Verification ---\n")

# Run verification and patching if any constants were overridden  
if args.difficulty is not None or args.quickness is not None or args.mouse_quickness is not None:
    verify_and_patch_constants()

print("Engine Difficulty: {}, Quickness: {}, Mouse Quickness: {}".format(DIFFICULTY, QUICKNESS, MOUSE_QUICKNESS))

if args.arena == True:
    # in tournament mode
    while True:
        time.sleep(0.5)
        res = await_new_game(timeout=300)
        if res is not None:
            set_game(res)
            if args.berserk:
                # beserk mode
                berserk()
                time.sleep(0.5)
            run_game(arena=True)
            print("Finished tournament game.")
            # go back to lobby. This can be done by clicking where the resign button is once
            time.sleep(random.randint(1,3))
            back_to_lobby()
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
        res = await_new_game(timeout=5)
        if res is not None:
            set_game(res)
            run_game(arena=False)
            print("Finished game {}".format(i+1))
            if i < games-1:
                new_game(tc_str)
        elif i < games-1:
            print("Skipped game, trying to seek again.")
            new_game(tc_str)
sys.exit()