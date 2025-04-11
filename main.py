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

from mp_original import run_game, await_new_game, set_game, new_game, back_to_lobby, berserk
from common.constants import DIFFICULTY, QUICKNESS, MOUSE_QUICKNESS


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
            run_game()
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
        res = await_new_game()
        if res is not None:
            set_game(res)
            run_game()
            print("Finished game {}".format(i+1))
            if i < games-1:
                new_game(tc_str)
        elif i < games-1:
            print("Skipped game, trying to seek again.")
            new_game(tc_str)
sys.exit()