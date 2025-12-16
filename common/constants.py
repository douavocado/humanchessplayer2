# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:33:26 2024

@author: xusem
"""
import math

PATH_TO_STOCKFISH = "Engines/stockfish17-ubuntu"
PATH_TO_PONDER_STOCKFISH = "Engines/stockfish16-ubuntu"

MOVE_FROM_WEIGHTS_OP_PTH = 'models/model_weights/piece_selector_opening_weights.pth'
MOVE_FROM_WEIGHTS_MID_PTH = 'models/model_weights/piece_selector_midgame_weights.pth'
MOVE_FROM_WEIGHTS_END_PTH = 'models/model_weights/piece_selector_endgame_weights.pth'
MOVE_FROM_WEIGHTS_TACTICS_PTH = 'models/model_weights/piece_selector_defensive_tactics_weights.pth'
MOVE_TO_WEIGHTS_MID_PTH = 'models/model_weights/piece_to_midgame_weights.pth'
MOVE_TO_WEIGHTS_END_PTH = 'models/model_weights/piece_to_endgame_weights.pth'
MOVE_TO_WEIGHTS_OP_PTH = 'models/model_weights/piece_to_opening_weights.pth'
MOVE_TO_WEIGHTS_TACTICS_PTH = 'models/model_weights/piece_to_defensive_tactics_weights.pth'

# Alter move probability model constants
WEIRD_MOVE_SD_DIC = {"opening" : 0.01,
                     "midgame" : 0.01,
                     "endgame" : 0.01,
                     }
LOWER_THRESH_SF = math.exp(-1.0406)
PROTECT_KING_SF = 0.7614
CAPTURE_EN_PRIS_SF = 0.7933
BREAK_PIN_SF = 1.2306
CAPTURE_SF = 1.1687
CAPTURE_SF_KING_DANGER = 1.1475
CAPTURABLE_SF = 0.8171
CHECK_SF_DIC = {"confident": 1.2022,
                "cocky": 1.4,
                "cautious": 1.1,
                "tilted": 1.6,
                "hurry": 1.9,
                "flagging": 1.6}
TAKEBACK_SF = 2.6469
NEW_THREATENED_SF_DIC = {"confident":1.6287,
                        "cocky": 1.3,
                        "cautious": 1.8,
                        "tilted": 1.1,
                        "hurry": 1.3,
                        "flagging": 1.5}
EXCHANGE_SF_DIC = {"confident": 1.4049,
                        "cocky": 1.1,
                        "cautious": 1.2,
                        "tilted": 1.0,
                        "hurry": 1.7,
                        "flagging": 0.8}
EXCHANGE_K_DANGER_SF_DIC = {"confident": 1.1235,
                        "cocky": 0.9,
                        "cautious": 1.3,
                        "tilted": 1.0,
                        "hurry": 1.1,
                        "flagging": 1.0}
PASSED_PAWN_END_SF = 3.3780
SOLO_FACTOR_SF = 1.3324
THREATENED_LVL_DIFF_SF = 0.4485

# Noise penalties
DEPTH_PENALTY = 15
ZERO_DEPTH_PENALTY = 30
CAPTURE_BONUS = 10


QUICKNESS = 2.1 # adjust depending on computer fastness. The bigger the number the slower the moves made
DIFFICULTY = 4 # engine difficulty
MOUSE_QUICKNESS = 3 # number between 0 and 10. Bigger the number the slower we are with mouse movements
RESOLUTION_SCALE = 2.0  # Set to 2.0 for 4K, 1.0 for 1080p - adjusts mouse curve point density

# BENCHMARKS
""" 
DIFFICULTY   |   QUICKNESS   |    ELO
    3        |     2.2      |  ~2500


"""
