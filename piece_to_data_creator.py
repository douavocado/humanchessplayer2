# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:25:27 2023

@author: xusem
"""

import os
import chess
import chess.pgn
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random

from common.board_information import phase_of_game
from common.board_encodings import moveto_position_list_one_hot, position_list


chess.BaseBoard.position_list_one_hot = moveto_position_list_one_hot
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

PGN_DIR = 'train_PGNs'

SAVE_DST_ROOT = os.path.join("models", "data", "piece_to")
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

                piece_from, piece_to = chess.square_mirror(dummy_move.from_square), chess.square_mirror(dummy_move.to_square)
                
                one_hot_position = dummy_board.position_list_one_hot(chess.square_mirror(piece_from))
                
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
                        
                if fold == 'train':
                    if phase == "opening":
                        train_opening_moved_from.append(piece_to)
                    elif phase == "midgame":
                        train_mid_moved_from.append(piece_to)
                    elif phase == "endgame":
                        train_end_moved_from.append(piece_to)
                    else:
                        raise Exception("Do not recognise the game phase {}".format(phase))
                else:
                    if phase == "opening":
                        test_opening_moved_from.append(piece_to)
                    elif phase == "midgame":
                        test_mid_moved_from.append(piece_to)
                    elif phase == "endgame":
                        test_end_moved_from.append(piece_to)
                    else:
                        raise Exception("Do not recognise the game phase {}".format(phase))
                        
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
    
# PGN_DIR = 'PGNs/'
# TRAIN_SAVE_FILE = 'Data/piece_to/train/endgame/training_data_combined_endgame.h5'
# TEST_SAVE_FILE = 'Data/piece_to/test/endgame/testing_data_combined_endgmae.h5'
# PHASE_OF_GAME = 'endgame'

# if __name__ == '__main__':
#     game_count = 0
#     for pgn_batch in os.listdir(PGN_DIR):
#         print ('new file', pgn_batch)
#         pgn = open(PGN_DIR + pgn_batch, encoding="utf-8")
#         train_input, train_moved_from = [], []
#         test_input, test_moved_from = [], []
#         for j in range(1000):
#             if (j+1) %100 == 0:
#                 print(j+1, 'games processed')
#             if j < 800:
#                 fold = 'train'
#             else:
#                 fold = 'test'
#             try:
#                 game = chess.pgn.read_game(pgn)
#                 if game is None:
#                     print('game type None, continuing...', pgn_batch)
#                     continue
#             except UnicodeDecodeError:
#                 print('error in parsing game')
#                 continue
            
#             try:
#                 board = game.board()  # set the game board
#             except ValueError as e:
#                 print('variant error', e)
#                 # some sort of variant issue
#                 continue
            
#             ## only record mid game moves/endgame moves
            
#             for move in list(game.mainline_moves()): 
#                 if phase_of_game(board) != PHASE_OF_GAME:
#                     board.push(move)
#                     continue
                
#                 if board.turn: #if it's white's turn
#                     dummy_board_before = board.copy()
#                 else:
#                     board.push(move)
#                     continue
#                     #dummy_board_before = board.mirror()
#                 position1 = dummy_board_before.position_list()
#                 board.push(move)
    
#                 if board.turn: #if it's white's turn
#                     dummy_board_after = board.mirror()
#                 else:
#                     dummy_board_after = board.copy()
#                 position2 = dummy_board_after.position_list()
#                 piece_from, piece_to = piece_moved(position1, position2)
                
#                 one_hot_position = dummy_board_before.position_list_one_hot(chess.square_mirror(piece_from))
    
                
#                 if fold == 'train':
#                     train_input.append(one_hot_position)
#                 else:
#                     test_input.append(one_hot_position)
                
                
#                 if fold == 'train':
#                     train_moved_from.append(piece_to)
#                 else:
#                     test_moved_from.append(piece_to)
#                 # position1 = position2
            
#             game_count += 1
            
        
#         # try:
#         train_positions = np.array(train_input)
#         train_moved_from = np.array(train_moved_from)
#         train_moved_from_one_hot = np.zeros((train_moved_from.size, 64))
#         train_moved_from_one_hot[np.arange(train_moved_from.size), train_moved_from] = 1
        
#         test_positions = np.array(test_input)
#         test_moved_from = np.array(test_moved_from)
#         print(test_moved_from.size)
#         test_moved_from_one_hot = np.zeros((test_moved_from.size, 64))
#         test_moved_from_one_hot[np.arange(test_moved_from.size), test_moved_from] = 1
        
        
#         try:
#             existing_train_df = pd.read_hdf(TRAIN_SAVE_FILE, key='data')
#             print('length of df so far', len(existing_train_df))
#             existing_train_label_df = pd.read_hdf(TRAIN_SAVE_FILE, key='label')
#         except:
#             existing_train_df = pd.DataFrame()
#             existing_train_label_df = pd.DataFrame()
#         appended_train_df = pd.DataFrame(train_positions)
#         appended_train_label_df = pd.DataFrame(train_moved_from_one_hot)
#         new_train_df = pd.concat([existing_train_df, appended_train_df])
#         new_train_label_df = pd.concat([existing_train_label_df, appended_train_label_df])
#         new_train_df.to_hdf(TRAIN_SAVE_FILE, key='data')
#         new_train_label_df.to_hdf(TRAIN_SAVE_FILE, key='label')
        
#         try:
#             existing_test_df = pd.read_hdf(TEST_SAVE_FILE, key='data')
#             print('length of df so far', len(existing_test_df))
#             existing_test_label_df = pd.read_hdf(TEST_SAVE_FILE, key='label')
#         except:
#             existing_test_df = pd.DataFrame()
#             existing_test_label_df = pd.DataFrame()
#         appended_test_df = pd.DataFrame(test_positions)
#         appended_test_label_df = pd.DataFrame(test_moved_from_one_hot)
#         new_test_df = pd.concat([existing_test_df, appended_test_df])
#         new_test_label_df = pd.concat([existing_test_label_df, appended_test_label_df])
#         new_test_df.to_hdf(TEST_SAVE_FILE, key='data')
#         new_test_label_df.to_hdf(TEST_SAVE_FILE, key='label')