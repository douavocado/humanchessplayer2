# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 00:39:52 2024

@author: xusem
"""

import chess
import chess.pgn
import os
import random
import json

from engine import Engine
from board_information import phase_of_game

engine = Engine()

pgn_file = random.choice(os.listdir("test_PGNs"))
pgn_path = os.path.join("test_PGNs", pgn_file)

games_used = 1
pgn = open(pgn_path, encoding="utf-8")
games_processed = 0
games_tried = 0

notable_positions = []

while games_processed < games_used and games_tried < 100:
    games_tried += 1
    print("Games processed:", games_processed)
    try:
        game = chess.pgn.read_game(pgn)
        if game is None:
            print('game type None, continuing...')
            continue
    except UnicodeDecodeError:
        print('error in parsing game')
        continue
    
    if game.headers["TimeControl"] != "60+0" or game.next().clock() != 60.0 or game.next().next().clock() != 60.0:
        print("Time control is not 60+0, skipping")
        continue
    fens = []
    white_clock_times = [60.0]
    black_clock_times = [60.0]
    last_moves = []
    while game is not None and game.next() is not None:
        
        engine.log = ""
        fens.append(game.board().fen())
        if game.turn() == chess.WHITE:
            if game.clock() is None:
                pass
            else:
                white_clock_times.append(game.clock())
            if len(white_clock_times) > 5:
                del white_clock_times[0]
            self_clock_times = white_clock_times[:]
            opp_clock_times = black_clock_times[:]
        elif game.turn() == chess.BLACK:
            if game.clock() is None:
                pass
            else:
                black_clock_times.append(game.clock())
            if len(black_clock_times) > 5:
                del black_clock_times[0]
            self_clock_times = black_clock_times[:]
            opp_clock_times = white_clock_times[:]
        if len(fens)> 5:
            del fens[0]
        
        if game.move is not None:
            last_moves.append(game.move.uci())
            if len(last_moves) > 5:
                del last_moves[0]
        input_dic = {"fens":fens[:],
                     "side":game.turn(),
                     "self_clock_times": self_clock_times[:],
                     "opp_clock_times": opp_clock_times[:],
                     "self_initial_time": 60.0,
                     "opp_initial_time": 60.0,
                     "last_moves": last_moves[:],
                     }
        engine.update_info(input_dic)
        game_phase = phase_of_game(game.board())
        move_dic  = engine.get_human_probabilities(game.board(), game_phase)
        altered_move_dic = engine._alter_move_probabilties(move_dic)
        
        no_moves = engine._decide_breadth()
        sorted_moves = sorted(altered_move_dic.keys(), reverse=True, key=lambda x: altered_move_dic[x])
        
        move_made = game.next().move.uci()
        if move_made in sorted_moves:
            index = sorted_moves.index(move_made)
            if index > no_moves:
                entry = {}
                entry["input_dic"] = input_dic.copy()
                entry["engine_log"] = engine.log
                entry["move_made"] = move_made
                entry["index"] = index
                notable_positions.append(entry.copy())
        else:
            entry = {}
            entry["input_dic"] = input_dic.copy()
            entry["engine_log"] = engine.log
            entry["move_made"] = move_made
            entry["index"] = None
            notable_positions.append(entry.copy())
        
        game = game.next()
        games_processed += 1

with open(os.path.join("Debugging", "notable_positions.json"), "w") as f:
    json.dump(notable_positions, f)