#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate training data for the AlterMoveProbNN module.

This script processes PGN files containing chess games, extracts positions, and
creates a CSV file with the necessary data for training the AlterMoveProbNN module.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import glob
import chess
import chess.pgn
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from common.board_information import (
    phase_of_game, king_danger, get_lucas_analytics, is_under_mate_threat
)
from models.models import MoveScorer
from common.utils import flip_uci

def load_decision_tree_classifier():
    """
    Train a decision tree classifier based on analysis_results.csv to decide
    which move scorer to use for a given position.
    
    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained decision tree classifier
    """
    # Load the analysis results data
    df = pd.read_csv('model_research/results/analysis_results.csv')
    
    # Prepare the features and target
    X = df[['complexity', 'win_prob', 'efficient_mobility', 'narrowness', 
            'piece_activity', 'game_phase', 'self_king_danger', 'opp_king_danger']].copy()
    
    # Convert game_phase to numeric
    phase_map = {'opening': 0, 'midgame': 1, 'endgame': 2}
    X['game_phase'] = X['game_phase'].map(phase_map)
    
    # Get the best model for each position (minimum rank)
    rank_columns = ['opening_rank', 'midgame_rank', 'endgame_rank', 
                   'tactics_rank', 'defensive_tactics_rank']
    df['best_model'] = df[rank_columns].idxmin(axis=1)
    df['best_model'] = df['best_model'].apply(lambda x: x.replace('_rank', ''))
    y = df['best_model']
    
    # Train the decision tree
    clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42)
    clf.fit(X, y)
    
    return clf


def predict_best_model(board, clf=None):
    """
    Predict the best move scorer model to use for a given board position.
    
    Args:
        clf: Trained decision tree classifier
        board: Chess board position
        
    Returns:
        str: Name of the best model to use
    """
    # Get game phase
    game_phase = phase_of_game(board)
    # phase_map = {'opening': 0, 'midgame': 1, 'endgame': 2}
    # game_phase_numeric = phase_map[game_phase]
    
    # # Get king danger values
    # self_king_danger = king_danger(board, board.turn, game_phase)
    # opp_king_danger = king_danger(board, not board.turn, game_phase)
    
    # # Get Lucas analytics
    # complexity, win_prob, efficient_mobility, narrowness, piece_activity = get_lucas_analytics(board)
    
    # # Create a feature vector for prediction
    # features = pd.DataFrame([[
    #     complexity, win_prob, efficient_mobility, narrowness, 
    #     piece_activity, game_phase_numeric, self_king_danger, opp_king_danger
    # ]], columns=['complexity', 'win_prob', 'efficient_mobility', 'narrowness', 
    #              'piece_activity', 'game_phase', 'self_king_danger', 'opp_king_danger'])
    
    # # Predict the best model
    # model_name = clf.predict(features)[0]
    
    # return model_name
    return game_phase


def load_move_scorer(model_name):
    """
    Load the appropriate move scorer model based on the model name.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        MoveScorer: Loaded move scorer model
    """
    # Mapping from model names to file paths
    model_paths = {
        'opening': ('models/model_weights/piece_selector_opening_weights.pth', 
                  'models/model_weights/piece_to_opening_weights.pth'),
        'midgame': ('models/model_weights/piece_selector_midgame_weights.pth', 
                   'models/model_weights/piece_to_midgame_weights.pth'),
        'endgame': ('models/model_weights/piece_selector_endgame_weights.pth', 
                   'models/model_weights/piece_to_endgame_weights.pth'),
        'tactics': ('models/model_weights/piece_selector_tactics_weights.pth', 
                   'models/model_weights/piece_to_tactics_weights.pth'),
        'defensive_tactics': ('models/model_weights/piece_selector_defensive_tactics_weights.pth', 
                             'models/model_weights/piece_to_defensive_tactics_weights.pth')
    }
    
    move_from_weights_path, move_to_weights_path = model_paths[model_name]
    
    return MoveScorer(move_from_weights_path, move_to_weights_path)


def process_game(game, output_writer, clf=None):
    """
    Process a single chess game and extract data for all positions.
    
    Args:
        game: Chess game to process
        clf: Decision tree classifier for model selection
        output_writer: CSV writer for output
    """
    board = chess.Board()
    moves = list(game.mainline_moves())
    
    prev_prev_board = None
    prev_board = None
    
    # For each position in the game
    for i, move in tqdm(enumerate(moves), total=len(moves), desc="Processing moves"):        
        # Get the true move (the move that was actually played)
        # only start recording after 20 moves in the game
        if i < 20:
            pass
        else:
            true_move = move.uci()
            
            # Get the best model to use for this position
            model_name = predict_best_model(board, clf=clf)
            
            # Load the appropriate move scorer
            move_scorer = load_move_scorer(model_name)
            
            # The board must be mirrored for the move scorer since it expects white's turn
            # (this is only necessary if the board position is black's turn)
            board_for_scorer = board.copy()
            
            if board.turn == chess.BLACK:
                board_for_scorer = board.mirror()
            
            # Get move probabilities from the move scorer
            _, move_dic = move_scorer.get_move_dic(board_for_scorer, san=False, top=100)

            # if we were black, we need to convert all the ucis to be flipped
            if board.turn == chess.BLACK:
                move_dic = {flip_uci(k): v for k,v in move_dic.items()}
            
            # Store the data for this position
            output_writer.writerow({
                'fen': board.fen(),
                'prev_fen': prev_board.fen() if prev_board else '',
                'prev_prev_fen': prev_prev_board.fen() if prev_prev_board else '',
                'move_dic': str(move_dic),
                'true_move': true_move,
                'game_phase': phase_of_game(board),
                'model_used': model_name
            })
        
        # Update the board history
        if prev_board is not None:
            prev_prev_board = prev_board.copy()
        else:
            prev_prev_board = None
        prev_board = board.copy()
        
        # Make the move on the board
        try:
            board.push(move)
        except Exception as e:
            print(f"Error making move {move} on board {board.fen()}: {e}")
            print("Skipping this game")
            break


def process_pgn_file(pgn_path, output_writer, max_games=None, clf=None):
    """
    Process all games in a PGN file and extract data for all positions.
    
    Args:
        pgn_path: Path to the PGN file
        clf: Decision tree classifier for model selection
        output_writer: CSV writer for output
        max_games: Maximum number of games to process (None for all)
    """
    print(f"Processing PGN file: {pgn_path}")
    
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        game_count = 0
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            if max_games is not None and game_count >= max_games:
                break
            
            process_game(game, output_writer, clf=clf)
            game_count += 1


def main(max_games_per_file=None, existing_file=None, start_from_pgn=None):
    """
    Main function to generate training data for the AlterMoveProbNN module.
    
    Args:
        max_games_per_file: Maximum number of games to process per PGN file
        existing_file: Path to an existing CSV file to append data to
        start_from_pgn: Name of the PGN file to start processing from
    """
    # Create output directory if it doesn't exist
    os.makedirs('development/alter_move_prob_train/data', exist_ok=True)
    
    # Load the decision tree classifier
    # print("Training decision tree classifier...")
    # clf = load_decision_tree_classifier()
    
    # Save the classifier for future use
    # joblib.dump(clf, 'development/alter_move_prob_train/data/model_selector_clf.joblib')
    
    # Determine output file path
    if existing_file:
        output_file = existing_file
        file_mode = 'a'  # Append mode
        write_header = False
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'development/alter_move_prob_train/data/training_data_{timestamp}.csv'
        file_mode = 'w'  # Write mode
        write_header = True
    
    fieldnames = ['fen', 'prev_fen', 'prev_prev_fen', 'move_dic', 'true_move', 
                 'game_phase', 'model_used']
    
    with open(output_file, file_mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        # Process all PGN files in the train_PGNs directory
        pgn_files = glob.glob('assets/data/train_PGNs/*.pgn')
        
        # If starting from a specific PGN file, find its index
        start_index = 0
        if start_from_pgn:
            for i, pgn_path in enumerate(pgn_files):
                if os.path.basename(pgn_path) == start_from_pgn:
                    start_index = i
                    break
            print(f"Starting from PGN file: {start_from_pgn} (index {start_index})")
        
        # Process PGN files from the starting index
        for i, pgn_path in enumerate(pgn_files[start_index:], start=start_index):
            # process_pgn_file(pgn_path, writer, max_games=max_games_per_file, clf=clf)
            process_pgn_file(pgn_path, writer, max_games=max_games_per_file, clf=None)
            print(f"Processed {i+1}/{len(pgn_files)} PGN files")
    
    print(f"Training data generation complete. Output saved to {output_file}")


if __name__ == "__main__":
    max_games_per_file = 30  # Set to None to process all games
    main(max_games_per_file) 