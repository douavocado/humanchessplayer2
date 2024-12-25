# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:33:26 2024

@author: xusem
"""
PATH_TO_STOCKFISH = "Engines/stockfish-ubuntu-x86-64-avx2"
PATH_TO_PONDER_STOCKFISH = "Engines/stockfish16-ubuntu"

MOVE_FROM_WEIGHTS_OP_PTH = 'models/model_weights/piece_selector_opening_weights.pth'
MOVE_FROM_WEIGHTS_MID_PTH = 'models/model_weights/piece_selector_midgame_weights.pth'
MOVE_FROM_WEIGHTS_END_PTH = 'models/model_weights/piece_selector_endgame_weights.pth'
MOVE_TO_WEIGHTS_MID_PTH = 'models/model_weights/piece_to_weights_midgame.pth'
MOVE_TO_WEIGHTS_END_PTH = 'models/model_weights/piece_to_weights_endgame.pth'
MOVE_TO_WEIGHTS_OP_PTH = 'models/model_weights/piece_to_weights_opening.pth'

QUICKNESS = 1.8 # adjust depending on computer fastness. The bigger the number the slower the moves made
DIFFICULTY = 4 # engine difficulty
MOUSE_QUICKNESS = 20 # number between 0 and 10. Bigger the number the slower we are with mouse movements

# BENCHMARKS
""" 
DIFFICULTY   |   QUICKNESS   |    ELO
    5        |     1.85      |  ~2250


"""
