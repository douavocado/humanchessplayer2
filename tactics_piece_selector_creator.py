# -*- coding: utf-8 -*-

import os
import sys
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
   


SAVE_DST_ROOT = os.path.join("models", "data", "defensive_tactics", "piece_selector")
# create folders if they don't exist
os.makedirs(os.path.join(SAVE_DST_ROOT, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DST_ROOT, "test"), exist_ok=True)

test_prob = 0.2

SAVE_EVERY = 1000
MAX_GAMES = 1000000
MAX_DF_SIZE = 150000

def create_data(train_input, train_moved_from, test_input, test_moved_from, save_root=SAVE_DST_ROOT):
    datetime_str = datetime.today().strftime('%Y_%m_%d')
    if len(train_input) == 0 or len(train_moved_from) == 0:
        print('no train data to save')
        return
        
    train_positions = np.array(train_input)
    train_moved_from = np.array(train_moved_from)
    train_moved_from_one_hot = np.zeros((train_moved_from.size, 64))
    train_moved_from_one_hot[np.arange(train_moved_from.size), train_moved_from] = 1            
    
    train_save_file = os.path.join(save_root, "train", "data"+datetime_str + ".h5")
    try:
        existing_train_df = pd.read_hdf(train_save_file, key='data')
        print('length of df so far', len(existing_train_df))
        existing_train_label_df = pd.read_hdf(train_save_file, key='label')
    except:
        existing_train_df = pd.DataFrame()
        existing_train_label_df = pd.DataFrame()
    appended_train_df = pd.DataFrame(train_positions)
    appended_train_label_df = pd.DataFrame(train_moved_from_one_hot)
    new_train_df = pd.concat([existing_train_df, appended_train_df])
    new_train_label_df = pd.concat([existing_train_label_df, appended_train_label_df])
    new_train_df.to_hdf(train_save_file, key='data')
    new_train_label_df.to_hdf(train_save_file, key='label')
    


    if len(test_input) == 0 or len(test_moved_from) == 0:
        print('no test data to save')
        return
    test_positions = np.array(test_input)
    test_moved_from = np.array(test_moved_from)
    test_moved_from_one_hot = np.zeros((test_moved_from.size, 64))
    test_moved_from_one_hot[np.arange(test_moved_from.size), test_moved_from] = 1
    
    test_save_file = os.path.join(save_root, "test",  "data"+datetime_str + ".h5")
    try:
        existing_test_df = pd.read_hdf(test_save_file, key='data')
        print('length of df so far', len(existing_test_df))
        existing_test_label_df = pd.read_hdf(test_save_file, key='label')
    except:
        existing_test_df = pd.DataFrame()
        existing_test_label_df = pd.DataFrame()
    appended_test_df = pd.DataFrame(test_positions)
    appended_test_label_df = pd.DataFrame(test_moved_from_one_hot)
    new_test_df = pd.concat([existing_test_df, appended_test_df])
    new_test_label_df = pd.concat([existing_test_label_df, appended_test_label_df])
    new_test_df.to_hdf(test_save_file, key='data')
    new_test_label_df.to_hdf(test_save_file, key='label')

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
    train_input, train_moved_from = [], []
    test_input, test_moved_from = [], []
    
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
            if board.turn != solver_turn: # we don't care about the opponent's moves
                board.push(move)
                continue

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
                train_input.append(one_hot_position)
            else:
                test_input.append(one_hot_position)
            dummy_board.push(dummy_move)

            position2 = dummy_board.position_list()
            piece_from, _ = piece_moved(position1, position2)
            if fold == 'train':
                train_moved_from.append(piece_from)
            else:
                test_moved_from.append(piece_from)
            
            board.push(move)
        
        game_count += 1
        
        if game_count % SAVE_EVERY == 0:
            print(f"Games processed: {game_count}")
            num_entries = create_data(train_input, train_moved_from, test_input, test_moved_from)
            train_input, train_moved_from = [], []
            test_input, test_moved_from = [], []
            if num_entries > MAX_DF_SIZE:
                break
        if game_count > MAX_GAMES:
            break