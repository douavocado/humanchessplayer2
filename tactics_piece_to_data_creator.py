# -*- coding: utf-8 -*-

import os
import sys
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

PGN_DIR = 'train_PGNs/defensive_tactics'

SAVE_DST_ROOT = os.path.join("models", "data", "defensive_tactics", "piece_to")
SAVE_EVERY = 1000
MAX_GAMES = 300000
MAX_DF_SIZE = 150000
test_prob = 0.2
# create folders if they don't exist
os.makedirs(os.path.join(SAVE_DST_ROOT, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DST_ROOT, "test"), exist_ok=True)

def create_data(train_input, train_moved_to, test_input, test_moved_to, save_root=SAVE_DST_ROOT):
    datetime_str = datetime.today().strftime('%Y_%m_%d')
    if len(train_input) == 0 or len(train_moved_to) == 0:
        print('no train data to save')
        return

    train_positions = np.array(train_input)
    train_moved_to = np.array(train_moved_to)
    train_moved_to_one_hot = np.zeros((train_moved_to.size, 64))
    train_moved_to_one_hot[np.arange(train_moved_to.size), train_moved_to] = 1            
    
    TRAIN_SAVE_FILE = os.path.join(SAVE_DST_ROOT, "train", "data"+datetime_str + ".h5")
    try:
        existing_train_df = pd.read_hdf(TRAIN_SAVE_FILE, key='data')
        print('length of df so far', len(existing_train_df))
        existing_train_label_df = pd.read_hdf(TRAIN_SAVE_FILE, key='label')
    except:
        existing_train_df = pd.DataFrame()
        existing_train_label_df = pd.DataFrame()
    appended_train_df = pd.DataFrame(train_positions)
    appended_train_label_df = pd.DataFrame(train_moved_to_one_hot)
    new_train_df = pd.concat([existing_train_df, appended_train_df])
    new_train_label_df = pd.concat([existing_train_label_df, appended_train_label_df])
    new_train_df.to_hdf(TRAIN_SAVE_FILE, key='data')
    new_train_label_df.to_hdf(TRAIN_SAVE_FILE, key='label')
    
    if len(test_input) == 0 or len(test_moved_to) == 0:
        print('no test data to save')
        return
    test_positions = np.array(test_input)
    test_moved_to = np.array(test_moved_to)
    test_moved_to_one_hot = np.zeros((test_moved_to.size, 64))
    test_moved_to_one_hot[np.arange(test_moved_to.size), test_moved_to] = 1
    
    TEST_SAVE_FILE = os.path.join(SAVE_DST_ROOT, "test", "data"+datetime_str + ".h5")
    try:
        existing_test_df = pd.read_hdf(TEST_SAVE_FILE, key='data')
        print('length of df so far', len(existing_test_df))
        existing_test_label_df = pd.read_hdf(TEST_SAVE_FILE, key='label')
    except:
        existing_test_df = pd.DataFrame()
        existing_test_label_df = pd.DataFrame()
    appended_test_df = pd.DataFrame(test_positions)
    appended_test_label_df = pd.DataFrame(test_moved_to_one_hot)
    new_test_df = pd.concat([existing_test_df, appended_test_df])
    new_test_label_df = pd.concat([existing_test_label_df, appended_test_label_df])
    new_test_df.to_hdf(TEST_SAVE_FILE, key='data')
    new_test_label_df.to_hdf(TEST_SAVE_FILE, key='label')
    
    return len(existing_train_df) + len(existing_test_df)

if __name__ == '__main__':
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process a specific PGN file for tactics training data')
    parser.add_argument('pgn_file', type=str, help='Path to the specific PGN file to process')
    args = parser.parse_args()
    
    # Validate that the file exists
    if not os.path.isfile(args.pgn_file):
        print(f"Error: The specified PGN file '{args.pgn_file}' does not exist.")
        sys.exit(1)
    
    game_count = 0
    pgn = open(args.pgn_file, encoding="utf-8")
    train_input, train_moved_to = [], []
    test_input, test_moved_to = [], []
    
    print(f"Processing games from: {args.pgn_file}")
    
    while True:
        if random.random() < test_prob:
            fold = 'test'
        else:
            fold = 'train'
        try:
            game = chess.pgn.read_game(pgn)
        except UnicodeDecodeError:
            print('Error in parsing game')
            continue
        
        if game is None:
            break
        elif game.next() is None:
            continue
        try:
            board = game.board()  # set the game board
        except ValueError as e:
            print('Variant error', e)
            # some sort of variant issue
            continue
        
        # solver_turn = not board.turn
        solver_turn = board.turn
        for move in list(game.mainline_moves()):
            # only consider moves when it's the solver's turn
            if solver_turn != board.turn:
                board.push(move)
                continue

            if board.turn: #if it's white's turn
                dummy_board = board.copy()
                dummy_move = move
            else:
                dummy_board = board.mirror()
                dummy_move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))

            piece_from, piece_to = chess.square_mirror(dummy_move.from_square), chess.square_mirror(dummy_move.to_square)
            
            one_hot_position = dummy_board.position_list_one_hot(chess.square_mirror(piece_from))
            
            if fold == 'train':
                train_input.append(one_hot_position)
            else:
                test_input.append(one_hot_position)

            if fold == 'train':
                train_moved_to.append(piece_to)
            else:
                test_moved_to.append(piece_to)

            board.push(move)
        
        game_count += 1
        
        if game_count % SAVE_EVERY == 0:
            num_entries = create_data(train_input, train_moved_to, test_input, test_moved_to)
            train_input, train_moved_to = [], []
            test_input, test_moved_to = [], []
            print(f"Games processed: {game_count}")
            if num_entries > MAX_DF_SIZE:
                print(f"Max DF size reached: {num_entries}")
                break
        
        if game_count > MAX_GAMES:
            break
