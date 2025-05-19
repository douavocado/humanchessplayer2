#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive MoveScorer Explorer

This script provides an interactive interface to explore MoveScorer model performance
on specific chess positions. It allows loading previously analyzed results and examining
individual positions to understand why certain models perform better in different situations.

Usage:
    python model_research/interactive_explorer.py [results_file]

Arguments:
    results_file    Path to the CSV results file from run_analysis.py (optional)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chess
import chess.svg
import random
from IPython.display import SVG, display

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import MoveScorer
from common.board_information import get_lucas_analytics, phase_of_game

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Interactive MoveScorer Explorer")
    
    parser.add_argument("results_file", nargs="?", type=str, 
                       default="model_research/results/analysis_results.csv",
                       help="Path to the CSV results file from run_analysis.py")
    
    return parser.parse_args()

class InteractiveExplorer:
    """
    Interactive tool to explore MoveScorer model performance on specific positions.
    """
    
    def __init__(self, results_file):
        """
        Initialize the explorer with results from previous analysis.
        
        Args:
            results_file (str): Path to the CSV results file from run_analysis.py
        """
        self.results_df = pd.read_csv(results_file)
        self.model_names = [col.replace('_rank', '') for col in self.results_df.columns if col.endswith('_rank')]
        self.current_position = None
        
        # Load models
        self.move_scorers = {}
        self._load_models()
    
    def _load_models(self):
        """Load all MoveScorer models"""
        print(f"Loading {len(self.model_names)} models...")
        
        for model_name in self.model_names:
            from_weights_path = f"models/model_weights/piece_selector_{model_name}_weights.pth"
            to_weights_path = f"models/model_weights/piece_to_{model_name}_weights.pth"
            
            try:
                self.move_scorers[model_name] = MoveScorer(from_weights_path, to_weights_path)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
    
    def find_interesting_positions(self, criteria="divergent", n=5):
        """
        Find interesting positions based on specified criteria.
        
        Args:
            criteria (str): Criteria for finding interesting positions
                - "divergent": Positions where models have very different ranks
                - "unanimous": Positions where all models agree on the top move
                - "surprising": Positions where the actual move is ranked low by most models
            n (int): Number of positions to return
        
        Returns:
            list: List of position indices
        """
        if criteria == "divergent":
            # Calculate standard deviation of ranks across models
            rank_columns = [f"{model}_rank" for model in self.model_names]
            self.results_df["rank_std"] = self.results_df[rank_columns].std(axis=1)
            interesting_positions = self.results_df.sort_values("rank_std", ascending=False).head(n)
        
        elif criteria == "unanimous":
            # Find positions where all models rank the move the same
            rank_columns = [f"{model}_rank" for model in self.model_names]
            self.results_df["rank_std"] = self.results_df[rank_columns].std(axis=1)
            interesting_positions = self.results_df.sort_values("rank_std").head(n)
        
        elif criteria == "surprising":
            # Find positions where the actual move is ranked low by most models
            rank_columns = [f"{model}_rank" for model in self.model_names]
            self.results_df["avg_rank"] = self.results_df[rank_columns].mean(axis=1)
            interesting_positions = self.results_df.sort_values("avg_rank", ascending=False).head(n)
        
        else:
            # Random positions
            interesting_positions = self.results_df.sample(n)
        
        return interesting_positions.index.tolist()
    
    def explore_position(self, index):
        """
        Explore a specific position from the results.
        
        Args:
            index (int): Index of the position in the results DataFrame
        """
        if index < 0 or index >= len(self.results_df):
            print(f"Invalid index: {index}")
            return
        
        position_data = self.results_df.iloc[index]
        self.current_position = position_data
        
        # Create a chess board from FEN
        board = chess.Board(position_data["position_fen"])
        
        # Print position details
        print("\n" + "="*50)
        print(f"Position #{index} (Move {position_data['move_num']})")
        print("="*50)
        
        # Print board as text
        print("\nBoard:")
        print(board)
        
        # Print Lucas analytics
        print("\nLucas Analytics:")
        print(f"  Complexity:         {position_data['complexity']:.2f}")
        print(f"  Win Probability:    {position_data['win_prob']:.2f}")
        print(f"  Efficient Mobility: {position_data['efficient_mobility']:.2f}")
        print(f"  Narrowness:         {position_data['narrowness']:.2f}")
        print(f"  Piece Activity:     {position_data['piece_activity']:.2f}")
        print(f"  Game Phase:         {position_data['game_phase']}")
        
        # Print actual move and model rankings
        print("\nActual Move:", position_data["actual_move"])
        
        print("\nModel Rankings:")
        model_ranks = [(model, position_data[f"{model}_rank"]) for model in self.model_names]
        for model, rank in sorted(model_ranks, key=lambda x: x[1]):
            print(f"  {model.ljust(18)}: {rank}")
        
        # Calculate best model
        rank_columns = [f"{model}_rank" for model in self.model_names]
        best_model = min([(model, position_data[f"{model}_rank"]) for model in self.model_names], key=lambda x: x[1])[0]
        print(f"\nBest Model: {best_model}")
        
        # Show top moves from each model
        self._show_model_predictions(board, position_data)
        
        return board
    
    def _show_model_predictions(self, board, position_data):
        """
        Show and compare top move predictions from each model.
        
        Args:
            board (chess.Board): Chess board
            position_data (pd.Series): Data for the current position
        """
        print("\nTop 5 Moves by Model:")
        
        all_predictions = {}
        
        for model_name, model in self.move_scorers.items():
            try:
                # Get top moves from the model
                top_moves = model.get_moves(board, san=True, top=5)
                all_predictions[model_name] = top_moves
                
                # Print top moves
                print(f"\n{model_name.capitalize()}:")
                for i, move in enumerate(top_moves):
                    marker = "→" if move == position_data["actual_move"] else " "
                    print(f"  {i+1}. {move.ljust(8)} {marker}")
            except Exception as e:
                print(f"Error getting predictions for {model_name}: {e}")
        
        # Create a comparison table of all unique predicted moves
        all_unique_moves = set()
        for moves in all_predictions.values():
            all_unique_moves.update(moves)
        
        # Print comparison table
        print("\nMove Comparison Table:")
        headers = ["Move"] + [model.capitalize() for model in self.model_names]
        print("  " + "  ".join([h.ljust(15) for h in headers]))
        print("  " + "-"*((15+2)*len(headers)))
        
        for move in all_unique_moves:
            row = [move]
            for model in self.model_names:
                try:
                    rank = all_predictions[model].index(move) + 1 if move in all_predictions[model] else "-"
                    row.append(str(rank))
                except:
                    row.append("-")
            
            marker = "→" if move == position_data["actual_move"] else " "
            print(f"{marker} " + "  ".join([r.ljust(15) for r in row]))
    
    def show_model_statistics(self):
        """
        Show overall statistics for each model.
        """
        print("\n" + "="*50)
        print("Model Performance Statistics")
        print("="*50)
        
        # Calculate statistics
        stats = {}
        for model in self.model_names:
            rank_col = f"{model}_rank"
            stats[model] = {
                "avg_rank": self.results_df[rank_col].mean(),
                "median_rank": self.results_df[rank_col].median(),
                "std_rank": self.results_df[rank_col].std(),
                "rank_1_pct": (self.results_df[rank_col] == 1).mean() * 100,
                "rank_3_pct": (self.results_df[rank_col] <= 3).mean() * 100,
                "rank_5_pct": (self.results_df[rank_col] <= 5).mean() * 100,
            }
        
        # Print statistics
        print("\nAverage Rank:")
        for model, data in sorted(stats.items(), key=lambda x: x[1]["avg_rank"]):
            print(f"  {model.ljust(18)}: {data['avg_rank']:.2f}")
        
        print("\nMedian Rank:")
        for model, data in sorted(stats.items(), key=lambda x: x[1]["median_rank"]):
            print(f"  {model.ljust(18)}: {data['median_rank']:.1f}")
        
        print("\nPercentage of Rank 1:")
        for model, data in sorted(stats.items(), key=lambda x: x[1]["rank_1_pct"], reverse=True):
            print(f"  {model.ljust(18)}: {data['rank_1_pct']:.1f}%")
        
        print("\nPercentage of Rank ≤ 3:")
        for model, data in sorted(stats.items(), key=lambda x: x[1]["rank_3_pct"], reverse=True):
            print(f"  {model.ljust(18)}: {data['rank_3_pct']:.1f}%")
        
        print("\nPercentage of Rank ≤ 5:")
        for model, data in sorted(stats.items(), key=lambda x: x[1]["rank_5_pct"], reverse=True):
            print(f"  {model.ljust(18)}: {data['rank_5_pct']:.1f}%")
    
    def show_model_strengths(self):
        """
        Show the strengths of each model based on Lucas analytics.
        """
        print("\n" + "="*50)
        print("Model Strengths by Lucas Analytics")
        print("="*50)
        
        metrics = ["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity"]
        
        # Find the best model for each position
        rank_columns = [f"{model}_rank" for model in self.model_names]
        self.results_df["best_model"] = self.results_df[rank_columns].idxmin(axis=1)
        self.results_df["best_model"] = self.results_df["best_model"].apply(lambda x: x.replace("_rank", ""))
        
        # Calculate average metric values for positions where each model is best
        strengths = {}
        for model in self.model_names:
            model_positions = self.results_df[self.results_df["best_model"] == model]
            if len(model_positions) > 0:
                strengths[model] = {metric: model_positions[metric].mean() for metric in metrics}
        
        # Print strengths
        for metric in metrics:
            print(f"\nBest at {metric.capitalize()}:")
            sorted_models = sorted(strengths.items(), key=lambda x: x[1][metric], reverse=True)
            for model, values in sorted_models:
                print(f"  {model.ljust(18)}: {values[metric]:.2f}")
        
        # Create dataframe for clearer comparison
        print("\nStrengths Summary:")
        strengths_df = pd.DataFrame(strengths).T
        print(strengths_df.round(2))
    
    def interactive_mode(self):
        """
        Enter interactive exploration mode.
        """
        print("\nWelcome to Interactive MoveScorer Explorer!")
        print("="*50)
        print("Explore MoveScorer model performance on chess positions.")
        print("="*50)
        
        self.show_model_statistics()
        self.show_model_strengths()
        
        while True:
            print("\nOptions:")
            print("  1. Explore a random position")
            print("  2. Find and explore divergent positions (where models disagree)")
            print("  3. Find and explore unanimous positions (where models agree)")
            print("  4. Find and explore surprising positions (low-ranked actual moves)")
            print("  5. Show model statistics")
            print("  6. Show model strengths")
            print("  0. Exit")
            
            choice = input("\nEnter your choice: ")
            
            if choice == "1":
                index = random.randint(0, len(self.results_df) - 1)
                self.explore_position(index)
            elif choice == "2":
                positions = self.find_interesting_positions("divergent")
                for index in positions:
                    self.explore_position(index)
                    if input("\nPress Enter to continue, or 'q' to return to menu: ").lower() == 'q':
                        break
            elif choice == "3":
                positions = self.find_interesting_positions("unanimous")
                for index in positions:
                    self.explore_position(index)
                    if input("\nPress Enter to continue, or 'q' to return to menu: ").lower() == 'q':
                        break
            elif choice == "4":
                positions = self.find_interesting_positions("surprising")
                for index in positions:
                    self.explore_position(index)
                    if input("\nPress Enter to continue, or 'q' to return to menu: ").lower() == 'q':
                        break
            elif choice == "5":
                self.show_model_statistics()
            elif choice == "6":
                self.show_model_strengths()
            elif choice == "0":
                print("\nExiting Interactive Explorer.")
                break
            else:
                print("\nInvalid choice. Please try again.")

def main():
    """Main function to run the interactive explorer"""
    args = parse_arguments()
    
    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Results file not found: {args.results_file}")
        print("Please run model_research/run_analysis.py first to generate results.")
        return
    
    # Create and run interactive explorer
    explorer = InteractiveExplorer(args.results_file)
    explorer.interactive_mode()

if __name__ == "__main__":
    main() 