# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 12:53:53 2023

https://lichess.org/api/games/user/{username}?max=300?&rated=true&perfType=bullet&clocks=true&evals=true
@author: xusem
"""

import os
import chess
import chess.pgn
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime

from common.board_information import phase_of_game
from common.board_encodings import position_list_one_hot, position_list

chess.BaseBoard.position_list_one_hot = position_list_one_hot
chess.BaseBoard.position_list = position_list

def piece_moved(position1, position2):
    '''Main data conversion function.
    step 1: checks the difference between two positions and returns a list
            of the affected squares.
    step 2: checks whether it is a normal move (only two squares affected), or
            en passant (3 squares affected) or castling (4 squares affected)
            step 2a: If castling, the square moved from is where the king was
                     in the beginning of the turn. Square moved to is where
                     the king is at the end of the turn.
            step 2b: If en passant, square moved from is where the pawn was
                     at the beginning of the turn. Moved to is where the pawn
                     is at the end of the turn.
    step 3: Returns two ints with the square moved from, and square moved to
    '''
    affected_squares = []
    for i in range(64):  # Step 1
        if position1[i] != position2[i]:
            affected_squares.append(i)
    if len(affected_squares) > 2:  # Step 2
        for square in affected_squares:
            if position1[square] == 12 or position1[square] == 6:  # Step 2a
                moved_from = square
            if position2[square] == 12 or position2[square] == 6:
                moved_to = square
            if position1[square] == 0:  # Step 2b
                if position2[square] == 1:
                    moved_to = square
                    for square in affected_squares:
                        if position1[square] == 1:
                            moved_from = square
                elif position2[square] == 7:
                    moved_to = square
                    for square in affected_squares:
                        if position1[square] == 7:
                            moved_from = square
    else:
        if position2[affected_squares[0]] == 0:
            moved_from, moved_to = affected_squares[0], affected_squares[1]
        else:
            moved_from, moved_to = affected_squares[1], affected_squares[0]
    return moved_from , moved_to


    
    
PGN_DIR = 'assets/data/train_PGNs'

SAVE_DST_ROOT = os.path.join("models", "data", "piece_selector")
test_prob = 0.2
datetime_str = datetime.today().strftime('%Y_%m_%d')

if __name__ == '__main__':
    game_count = 0
    for pgn_batch in tqdm(os.listdir(PGN_DIR)):
        pgn = open(os.path.join(PGN_DIR,pgn_batch), encoding="utf-8")
        train_opening_input, train_opening_moved_from = [], []
        test_opening_input, test_opening_moved_from = [], []
        
        train_mid_input, train_mid_moved_from = [], []
        test_mid_input, test_mid_moved_from = [], []
        
        train_end_input, train_end_moved_from = [], []
        test_end_input, test_end_moved_from = [], []
        for j in range(200):
            if random.random() < test_prob:
                fold = 'test'
            else:
                fold = 'train'
            try:
                game = chess.pgn.read_game(pgn)
            except UnicodeDecodeError:
                print('error in parsing game')
                continue
            
            if game is None:
                break
            elif game.next() is None:
                continue
            
            if game.headers["TimeControl"] != "60+0" or game.next().clock() != 60.0 or game.next().next().clock() != 60.0:
                print("Time control is not 60+0, skipping")
                continue
            try:
                board = game.board()  # set the game board
            except ValueError as e:
                print('variant error', e)
                # some sort of variant issue
                continue
            
            for move in list(game.mainline_moves()):
                phase = phase_of_game(board)
                if board.turn: #if it's white's turn
                    dummy_board = board.copy()
                    dummy_move = move
                else:
                    # board.push(move)
                    # continue
                    dummy_board = board.mirror()
                    dummy_move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))
                position1 = dummy_board.position_list()
                one_hot_position = dummy_board.position_list_one_hot()
                
                if fold == 'train':
                    if phase == "opening":
                        train_opening_input.append(one_hot_position)
                    elif phase == "midgame":
                        train_mid_input.append(one_hot_position)
                    elif phase == "endgame":
                        train_end_input.append(one_hot_position)
                    else:
                        raise Exception("Do not recognise the game phase {}".format(phase))
                else:
                    if phase == "opening":
                        test_opening_input.append(one_hot_position)
                    elif phase == "midgame":
                        test_mid_input.append(one_hot_position)
                    elif phase == "endgame":
                        test_end_input.append(one_hot_position)
                    else:
                        raise Exception("Do not recognise the game phase {}".format(phase))
                dummy_board.push(dummy_move)
    
                position2 = dummy_board.position_list()
                piece_from, _ = piece_moved(position1, position2)
                if fold == 'train':
                    if phase == "opening":
                        train_opening_moved_from.append(piece_from)
                    elif phase == "midgame":
                        train_mid_moved_from.append(piece_from)
                    elif phase == "endgame":
                        train_end_moved_from.append(piece_from)
                    else:
                        raise Exception("Do not recognise the game phase {}".format(phase))
                else:
                    if phase == "opening":
                        test_opening_moved_from.append(piece_from)
                    elif phase == "midgame":
                        test_mid_moved_from.append(piece_from)
                    elif phase == "endgame":
                        test_end_moved_from.append(piece_from)
                    else:
                        raise Exception("Do not recognise the game phase {}".format(phase))
                # position1 = position2
                
                board.push(move)
            
            game_count += 1
            
        train_phase_dic = {"opening": [train_opening_input, train_opening_moved_from],
                                "midgame": [train_mid_input, train_mid_moved_from],
                                "endgame": [train_end_input, train_end_moved_from]}
        
        test_phase_dic = {"opening": [test_opening_input, test_opening_moved_from],
                                "midgame": [test_mid_input, test_mid_moved_from],
                                "endgame": [test_end_input, test_end_moved_from]}
        
        for phase, data in train_phase_dic.items():
            train_input, train_moved_from = data
            if len(train_input) == 0 or len(train_moved_from) == 0:
                continue
            
            train_positions = np.array(train_input)
            train_moved_from = np.array(train_moved_from)
            train_moved_from_one_hot = np.zeros((train_moved_from.size, 64))
            train_moved_from_one_hot[np.arange(train_moved_from.size), train_moved_from] = 1            
            
            TRAIN_SAVE_FILE = os.path.join(SAVE_DST_ROOT, "train", phase, "data"+datetime_str + ".h5")
            try:
                existing_train_df = pd.read_hdf(TRAIN_SAVE_FILE, key='data')
                print('length of df so far', len(existing_train_df))
                existing_train_label_df = pd.read_hdf(TRAIN_SAVE_FILE, key='label')
            except:
                existing_train_df = pd.DataFrame()
                existing_train_label_df = pd.DataFrame()
            appended_train_df = pd.DataFrame(train_positions)
            appended_train_label_df = pd.DataFrame(train_moved_from_one_hot)
            new_train_df = pd.concat([existing_train_df, appended_train_df])
            new_train_label_df = pd.concat([existing_train_label_df, appended_train_label_df])
            new_train_df.to_hdf(TRAIN_SAVE_FILE, key='data')
            new_train_label_df.to_hdf(TRAIN_SAVE_FILE, key='label')
        
        for phase, data in test_phase_dic.items():
            test_input, test_moved_from = data
            if len(test_input) == 0 or len(test_moved_from) == 0:
                continue
            test_positions = np.array(test_input)
            test_moved_from = np.array(test_moved_from)
            test_moved_from_one_hot = np.zeros((test_moved_from.size, 64))
            test_moved_from_one_hot[np.arange(test_moved_from.size), test_moved_from] = 1
            
            TEST_SAVE_FILE = os.path.join(SAVE_DST_ROOT, "test", phase, "data"+datetime_str + ".h5")
            try:
                existing_test_df = pd.read_hdf(TEST_SAVE_FILE, key='data')
                print('length of df so far', len(existing_test_df))
                existing_test_label_df = pd.read_hdf(TEST_SAVE_FILE, key='label')
            except:
                existing_test_df = pd.DataFrame()
                existing_test_label_df = pd.DataFrame()
            appended_test_df = pd.DataFrame(test_positions)
            appended_test_label_df = pd.DataFrame(test_moved_from_one_hot)
            new_test_df = pd.concat([existing_test_df, appended_test_df])
            new_test_label_df = pd.concat([existing_test_label_df, appended_test_label_df])
            new_test_df.to_hdf(TEST_SAVE_FILE, key='data')
            new_test_label_df.to_hdf(TEST_SAVE_FILE, key='label')