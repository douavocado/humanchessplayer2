#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MoveScorer Analysis Tool

This script analyzes different MoveScorer models across various chess situations
characterized by Lucas analytics. It evaluates their performance based on the rank
they assign to the actual move played in real games.

Usage:
    python model_research/run_analysis.py [options]

Options:
    --max-games N        Maximum number of games to analyze per PGN file
    --max-positions N    Maximum number of positions to analyze per game
    --pgn-pattern STR    Pattern to match PGN files (default: all in train_PGNs/)
    --output-dir STR     Directory to save results (default: model_research/results)
"""

import os
import sys
import glob
import argparse
import time
import pandas as pd
from tqdm import tqdm

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_research.movescorer_analyzer import MovesScorerAnalyzer
from model_research.visualizer import MovesScorerVisualizer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze MoveScorer models across different chess situations")
    
    parser.add_argument(
        "--pgn_dir", 
        type=str, 
        default="data/pgn",
        help="Directory containing PGN files to analyze"
    )
    parser.add_argument(
        "--max_games", 
        type=int, 
        default=10,
        help="Maximum number of games to analyze per PGN file"
    )
    parser.add_argument(
        "--max_positions", 
        type=int, 
        default=5,
        help="Maximum number of positions to analyze per game"
    )
    parser.add_argument(
        "--model_names", 
        type=str, 
        nargs="+",
        default=["opening", "midgame", "endgame", "tactics", "defensive_tactics"],
        help="Names of models to analyze"
    )
    parser.add_argument(
        "--split_by_phase", 
        action="store_true",
        help="Whether to split visualizations by game phase"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the analysis"""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs("model_research/results", exist_ok=True)
    os.makedirs("model_research/plots", exist_ok=True)
    os.makedirs("model_research/reports", exist_ok=True)
    
    # Get PGN files
    pgn_files = glob.glob(os.path.join(args.pgn_dir, "*.pgn"))
    if not pgn_files:
        print(f"Error: No PGN files found in {args.pgn_dir}")
        sys.exit(1)
        
    print(f"Found {len(pgn_files)} PGN files.")
    
    # Create analyzer
    analyzer = MovesScorerAnalyzer(pgn_files, model_names=args.model_names)
    
    # Analyze PGN files
    print(f"Analyzing up to {args.max_games} games per file, {args.max_positions} positions per game...")
    results_df = analyzer.analyze_pgn_files(max_games=args.max_games, max_positions_per_game=args.max_positions)
    
    # Save results to CSV
    results_path = "model_research/results/analysis_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Analysis results saved to {results_path}")
    
    # Plot results
    print("Generating plots...")
    
    # Basic analyzer plots
    analyzer.plot_best_model_distribution(split_by_phase=args.split_by_phase)
    analyzer.plot_all_metrics(split_by_phase=args.split_by_phase)
    analyzer.plot_performance_by_game_phase()
    
    # Generate decision boundary report
    report = analyzer.generate_decision_boundary_report(split_by_phase=args.split_by_phase)
    print(f"Decision boundary report generated.")
    
    # Additional visualizations
    visualizer = MovesScorerVisualizer(results_df)
    visualizer.generate_visualization_report(split_by_phase=args.split_by_phase)
    print("Visualization report generated.")
    
    print("Analysis complete!")
    print(f"- Results saved to: {results_path}")
    print(f"- Plots saved to: model_research/plots/")
    print(f"- Reports saved to: model_research/reports/")

if __name__ == "__main__":
    main() 