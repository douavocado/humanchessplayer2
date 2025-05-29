#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to evaluate the trained AlterMoveProbNN model.

This script evaluates the performance of the AlterMoveProbNN model by comparing
how well it predicts human moves compared to the original move scorer.
"""

import os
import sys
import ast
import glob
import chess
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add the main directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.alter_move_prob_train.alter_move_prob_nn import AlterMoveProbNN
from common.board_information import phase_of_game


def load_model(model_path):
    """
    Load a trained AlterMoveProbNN model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        AlterMoveProbNN: Loaded model
    """
    model = AlterMoveProbNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, test_data):
    """
    Evaluate the performance of the AlterMoveProbNN model.
    
    Args:
        model: AlterMoveProbNN model
        test_data: Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    total_samples = len(test_data)
    correct_original = 0  # Number of times the original top move is correct
    correct_altered = 0   # Number of times the altered top move is correct
    top3_original = 0     # Number of times the true move is in top 3 original moves
    top3_altered = 0      # Number of times the true move is in top 3 altered moves
    top5_original = 0     # Number of times the true move is in top 5 original moves
    top5_altered = 0      # Number of times the true move is in top 5 altered moves
    
    # Group by game phase for more detailed analysis
    phase_metrics = {
        'opening': {'total': 0, 'orig_top1': 0, 'alt_top1': 0, 'orig_top3': 0, 'alt_top3': 0, 'orig_top5': 0, 'alt_top5': 0},
        'midgame': {'total': 0, 'orig_top1': 0, 'alt_top1': 0, 'orig_top3': 0, 'alt_top3': 0, 'orig_top5': 0, 'alt_top5': 0},
        'endgame': {'total': 0, 'orig_top1': 0, 'alt_top1': 0, 'orig_top3': 0, 'alt_top3': 0, 'orig_top5': 0, 'alt_top5': 0}
    }
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating model"):
        # Parse the move dictionary
        move_dic = ast.literal_eval(row['move_dic'])
        
        # Create board objects from FEN strings
        board = chess.Board(row['fen'])
        
        prev_board = None
        if row['prev_fen']:
            prev_board = chess.Board(row['prev_fen'])
        
        prev_prev_board = None
        if row['prev_prev_fen']:
            prev_prev_board = chess.Board(row['prev_prev_fen'])
        
        # Get the true move (the move that was actually played)
        true_move = row['true_move']
        
        # Get the game phase
        game_phase = row['game_phase']
        
        # Apply the AlterMoveProbNN model
        altered_move_dic, _ = model(move_dic, board, prev_board, prev_prev_board)
        
        # Original top move
        original_top_moves = sorted(move_dic.items(), key=lambda x: x[1], reverse=True)
        original_top1_move = original_top_moves[0][0] if original_top_moves else None
        
        # Altered top move
        altered_top_moves = sorted(altered_move_dic.items(), key=lambda x: x[1], reverse=True)
        altered_top1_move = altered_top_moves[0][0] if altered_top_moves else None
        
        # Check if the original top move is correct
        if original_top1_move == true_move:
            correct_original += 1
            phase_metrics[game_phase]['orig_top1'] += 1
        
        # Check if the altered top move is correct
        if altered_top1_move == true_move:
            correct_altered += 1
            phase_metrics[game_phase]['alt_top1'] += 1
        
        # Check if the true move is in the top 3 original moves
        original_top3_moves = [move[0] for move in original_top_moves[:3]]
        if true_move in original_top3_moves:
            top3_original += 1
            phase_metrics[game_phase]['orig_top3'] += 1
        
        # Check if the true move is in the top 3 altered moves
        altered_top3_moves = [move[0] for move in altered_top_moves[:3]]
        if true_move in altered_top3_moves:
            top3_altered += 1
            phase_metrics[game_phase]['alt_top3'] += 1
        
        # Check if the true move is in the top 5 original moves
        original_top5_moves = [move[0] for move in original_top_moves[:5]]
        if true_move in original_top5_moves:
            top5_original += 1
            phase_metrics[game_phase]['orig_top5'] += 1
        
        # Check if the true move is in the top 5 altered moves
        altered_top5_moves = [move[0] for move in altered_top_moves[:5]]
        if true_move in altered_top5_moves:
            top5_altered += 1
            phase_metrics[game_phase]['alt_top5'] += 1
        
        # Update phase metrics total
        phase_metrics[game_phase]['total'] += 1
    
    # Calculate overall metrics
    metrics = {
        'total_samples': total_samples,
        'top1_accuracy_original': correct_original / total_samples,
        'top1_accuracy_altered': correct_altered / total_samples,
        'top3_accuracy_original': top3_original / total_samples,
        'top3_accuracy_altered': top3_altered / total_samples,
        'top5_accuracy_original': top5_original / total_samples,
        'top5_accuracy_altered': top5_altered / total_samples,
        'top1_improvement': (correct_altered - correct_original) / total_samples,
        'top3_improvement': (top3_altered - top3_original) / total_samples,
        'top5_improvement': (top5_altered - top5_original) / total_samples,
        'phase_metrics': {}
    }
    
    # Calculate phase-specific metrics
    for phase, data in phase_metrics.items():
        if data['total'] > 0:
            metrics['phase_metrics'][phase] = {
                'total_samples': data['total'],
                'top1_accuracy_original': data['orig_top1'] / data['total'],
                'top1_accuracy_altered': data['alt_top1'] / data['total'],
                'top3_accuracy_original': data['orig_top3'] / data['total'],
                'top3_accuracy_altered': data['alt_top3'] / data['total'],
                'top5_accuracy_original': data['orig_top5'] / data['total'],
                'top5_accuracy_altered': data['alt_top5'] / data['total'],
                'top1_improvement': (data['alt_top1'] - data['orig_top1']) / data['total'],
                'top3_improvement': (data['alt_top3'] - data['orig_top3']) / data['total'],
                'top5_improvement': (data['alt_top5'] - data['orig_top5']) / data['total']
            }
    
    return metrics


def print_metrics(metrics):
    """
    Print evaluation metrics in a readable format.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n===== Overall Metrics =====")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Top-1 accuracy (original): {metrics['top1_accuracy_original']:.4f}")
    print(f"Top-1 accuracy (altered): {metrics['top1_accuracy_altered']:.4f}")
    print(f"Top-1 improvement: {metrics['top1_improvement']:.4f}")
    print(f"Top-3 accuracy (original): {metrics['top3_accuracy_original']:.4f}")
    print(f"Top-3 accuracy (altered): {metrics['top3_accuracy_altered']:.4f}")
    print(f"Top-3 improvement: {metrics['top3_improvement']:.4f}")
    print(f"Top-5 accuracy (original): {metrics['top5_accuracy_original']:.4f}")
    print(f"Top-5 accuracy (altered): {metrics['top5_accuracy_altered']:.4f}")
    print(f"Top-5 improvement: {metrics['top5_improvement']:.4f}")
    
    for phase, phase_metrics in metrics['phase_metrics'].items():
        print(f"\n===== {phase.capitalize()} Phase Metrics =====")
        print(f"Total samples: {phase_metrics['total_samples']}")
        print(f"Top-1 accuracy (original): {phase_metrics['top1_accuracy_original']:.4f}")
        print(f"Top-1 accuracy (altered): {phase_metrics['top1_accuracy_altered']:.4f}")
        print(f"Top-1 improvement: {phase_metrics['top1_improvement']:.4f}")
        print(f"Top-3 accuracy (original): {phase_metrics['top3_accuracy_original']:.4f}")
        print(f"Top-3 accuracy (altered): {phase_metrics['top3_accuracy_altered']:.4f}")
        print(f"Top-3 improvement: {phase_metrics['top3_improvement']:.4f}")
        print(f"Top-5 accuracy (original): {phase_metrics['top5_accuracy_original']:.4f}")
        print(f"Top-5 accuracy (altered): {phase_metrics['top5_accuracy_altered']:.4f}")
        print(f"Top-5 improvement: {phase_metrics['top5_improvement']:.4f}")


def main():
    """
    Main function to evaluate the AlterMoveProbNN model.
    """
    # Look for training data files
    data_files = glob.glob('development/alter_move_prob_train/data/training_data_*.csv')
    if not data_files:
        print("No training data found. Please run create_data.py first.")
        return
    
    latest_data_file = max(data_files, key=os.path.getctime)
    print(f"Using data from {latest_data_file}")
    
    # Load the data
    data = pd.read_csv(latest_data_file)
    
    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Test set size: {len(test_data)} samples")
    
    # Find model files to evaluate
    model_files = glob.glob('development/alter_move_prob_train/data/alter_move_prob_nn*.pth')
    if not model_files:
        print("No model file found. Please run train_model.py first.")
        return
    
    latest_model_file = max(model_files, key=os.path.getctime)
    print(f"Using model from {latest_model_file}")
    
    # Load the model
    model = load_model(latest_model_file)
    
    # Evaluate the model
    metrics = evaluate_model(model, test_data)
    
    # Print the metrics
    print_metrics(metrics)
    
    # Save metrics to a file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = f'development/alter_move_prob_train/data/evaluation_metrics_{timestamp}.txt'
    
    with open(metrics_file, 'w') as f:
        f.write("===== Overall Metrics =====\n")
        f.write(f"Total samples: {metrics['total_samples']}\n")
        f.write(f"Top-1 accuracy (original): {metrics['top1_accuracy_original']:.4f}\n")
        f.write(f"Top-1 accuracy (altered): {metrics['top1_accuracy_altered']:.4f}\n")
        f.write(f"Top-1 improvement: {metrics['top1_improvement']:.4f}\n")
        f.write(f"Top-3 accuracy (original): {metrics['top3_accuracy_original']:.4f}\n")
        f.write(f"Top-3 accuracy (altered): {metrics['top3_accuracy_altered']:.4f}\n")
        f.write(f"Top-3 improvement: {metrics['top3_improvement']:.4f}\n")
        f.write(f"Top-5 accuracy (original): {metrics['top5_accuracy_original']:.4f}\n")
        f.write(f"Top-5 accuracy (altered): {metrics['top5_accuracy_altered']:.4f}\n")
        f.write(f"Top-5 improvement: {metrics['top5_improvement']:.4f}\n")
        
        for phase, phase_metrics in metrics['phase_metrics'].items():
            f.write(f"\n===== {phase.capitalize()} Phase Metrics =====\n")
            f.write(f"Total samples: {phase_metrics['total_samples']}\n")
            f.write(f"Top-1 accuracy (original): {phase_metrics['top1_accuracy_original']:.4f}\n")
            f.write(f"Top-1 accuracy (altered): {phase_metrics['top1_accuracy_altered']:.4f}\n")
            f.write(f"Top-1 improvement: {phase_metrics['top1_improvement']:.4f}\n")
            f.write(f"Top-3 accuracy (original): {phase_metrics['top3_accuracy_original']:.4f}\n")
            f.write(f"Top-3 accuracy (altered): {phase_metrics['top3_accuracy_altered']:.4f}\n")
            f.write(f"Top-3 improvement: {phase_metrics['top3_improvement']:.4f}\n")
            f.write(f"Top-5 accuracy (original): {phase_metrics['top5_accuracy_original']:.4f}\n")
            f.write(f"Top-5 accuracy (altered): {phase_metrics['top5_accuracy_altered']:.4f}\n")
            f.write(f"Top-5 improvement: {phase_metrics['top5_improvement']:.4f}\n")
    
    print(f"Evaluation metrics saved to {metrics_file}")


if __name__ == "__main__":
    main() 