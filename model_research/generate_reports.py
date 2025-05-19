import os
import pandas as pd
import argparse
from visualizer import MovesScorerVisualizer
from movescorer_analyzer import MovesScorerAnalyzer

def main():
    """
    Generate all reports and visualizations from previously saved analysis results.
    """
    parser = argparse.ArgumentParser(description="Generate reports and plots from saved analysis results")
    parser.add_argument(
        "--results_file", 
        type=str, 
        default="model_research/results/analysis_results.csv",
        help="Path to the CSV file containing analysis results"
    )
    parser.add_argument(
        "--split_by_phase", 
        action="store_true",
        help="Whether to split visualizations by game phase"
    )
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found at {args.results_file}")
        print("Please run run_analysis.py first or specify the correct path with --results_file")
        return
    
    # Load the results
    print(f"Loading analysis results from {args.results_file}...")
    results_df = pd.read_csv(args.results_file)
    print(f"Loaded {len(results_df)} positions.")
    
    # Create directories for output
    os.makedirs("model_research/plots", exist_ok=True)
    os.makedirs("model_research/reports", exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    visualizer = MovesScorerVisualizer(results_df)
    visualizer.generate_visualization_report(split_by_phase=args.split_by_phase)
    
    # Generate decision boundary report
    print("Generating decision boundary report...")
    # Create a minimal analyzer that can use the loaded results
    analyzer = create_minimal_analyzer(results_df)
    analyzer.generate_decision_boundary_report(split_by_phase=args.split_by_phase)
    analyzer.train_minimum_rank_decision_tree(split_by_phase=args.split_by_phase)
    
    # Generate additional plots
    print("Generating additional plots...")
    analyzer.plot_best_model_distribution(split_by_phase=args.split_by_phase)
    analyzer.plot_all_metrics(split_by_phase=args.split_by_phase)
    analyzer.plot_performance_by_game_phase()
    
    print("All reports and visualizations have been generated!")
    print(f"- Plots saved to: model_research/plots/")
    print(f"- Reports saved to: model_research/reports/")

def create_minimal_analyzer(results_df):
    """
    Create a minimal MovesScorerAnalyzer that can use pre-loaded results.
    
    Args:
        results_df (pd.DataFrame): DataFrame with analysis results
        
    Returns:
        MovesScorerAnalyzer: Analyzer with pre-loaded results
    """
    # Create analyzer with empty PGN files list (we won't use it)
    analyzer = MovesScorerAnalyzer([])
    
    # Extract model names from results columns
    model_names = [col.replace('_rank', '') for col in results_df.columns if col.endswith('_rank')]
    analyzer.model_names = model_names
    
    # Set the results dataframe
    analyzer.results_df = results_df
    
    return analyzer

if __name__ == "__main__":
    main() 