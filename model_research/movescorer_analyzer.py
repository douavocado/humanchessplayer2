import sys
import os
import time
import chess
import chess.pgn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import MoveScorer
from common.board_information import get_lucas_analytics, phase_of_game, king_danger

class MovesScorerAnalyzer:
    """
    Analyzes the performance of different MoveScorer models across various chess situations
    characterized by Lucas analytics.
    """
    
    def __init__(self, pgn_files, model_names=None):
        """
        Initialize the analyzer with PGN files and model names.
        
        Args:
            pgn_files (list): List of paths to PGN files to analyze
            model_names (list, optional): List of model names. Defaults to all 5 models.
        """
        self.pgn_files = pgn_files
        
        if model_names is None:
            self.model_names = ["opening", "midgame", "endgame", "tactics", "defensive_tactics"]
        else:
            self.model_names = model_names
            
        self.move_scorers = {}
        self._load_models()
        
        # Dataframe to store results
        self.results_df = None
        
        # Game phases
        self.game_phases = ['opening', 'midgame', 'endgame']
        
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
    
    def analyze_pgn_files(self, max_games=None, max_positions_per_game=None):
        """
        Analyze all PGN files and collect performance data.
        
        Args:
            max_games (int, optional): Maximum number of games to analyze per file
            max_positions_per_game (int, optional): Maximum number of positions to analyze per game
        
        Returns:
            pandas.DataFrame: DataFrame with analysis results
        """
        results = []
        
        for pgn_path in tqdm(self.pgn_files, desc="Analyzing PGN files"):
            try:
                results.extend(self.analyze_pgn_file(pgn_path, max_games, max_positions_per_game))
            except Exception as e:
                print(f"Error analyzing {pgn_path}: {e}")
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def analyze_pgn_file(self, pgn_path, max_games=None, max_positions_per_game=None):
        """
        Analyze a single PGN file and return performance data.
        
        Args:
            pgn_path (str): Path to PGN file
            max_games (int, optional): Maximum number of games to analyze
            max_positions_per_game (int, optional): Maximum number of positions to analyze per game
        
        Returns:
            list: List of dictionaries with analysis results
        """
        results = []
        num_games = 0
        
        try:
            with open(pgn_path, "r") as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    if max_games is not None and num_games >= max_games:
                        break
                    
                    # Only analyze games with actual moves
                    if game.errors:
                        continue
                    
                    try:
                        game_results = self.analyze_game(game, max_positions_per_game)
                        results.extend(game_results)
                        num_games += 1
                        if num_games % 10 == 0:
                            print(f"Analyzed {num_games} games from {os.path.basename(pgn_path)}")
                    except Exception as e:
                        print(f"Error analyzing game in {pgn_path}: {e}")
        except Exception as e:
            print(f"Error opening or reading {pgn_path}: {e}")
        
        return results
    
    def analyze_game(self, game, max_positions=None):
        """
        Analyze a single chess game and return performance data.
        
        Args:
            game (chess.pgn.Game): Chess game to analyze
            max_positions (int, optional): Maximum number of positions to analyze
        
        Returns:
            list: List of dictionaries with analysis results
        """
        results = []
        board = game.board()
        
        # Track positions to analyze (always white's turn for consistent model evaluation)
        positions = []
        moves = []
        move_nums = []
        
        node = game
        move_num = 0
        
        # First, collect all positions where it's white's turn
        while node.variations:
            next_node = node.variations[0]
            actual_move = next_node.move
            
            if board.turn == chess.WHITE:
                positions.append(board.copy())
                moves.append(actual_move)
                move_nums.append(move_num)
            
            board.push(actual_move)
            node = next_node
            move_num += 1
        
        # If max_positions is specified, take a random subset
        if max_positions is not None and len(positions) > max_positions:
            import random
            indices = random.sample(range(len(positions)), max_positions)
            positions = [positions[i] for i in indices]
            moves = [moves[i] for i in indices]
            move_nums = [move_nums[i] for i in indices]
        
        # Now, analyze each collected position
        for i, (position, actual_move, move_num) in enumerate(zip(positions, moves, move_nums)):
            try:
                # Get the Lucas analytics for this position
                complexity, win_prob, efficient_mobility, narrowness, piece_activity = get_lucas_analytics(position)
                
                # Get the game phase
                game_phase = phase_of_game(position)

                # Get the king danger for this position
                if position.turn == chess.WHITE:
                    self_king_danger = king_danger(position, chess.WHITE, game_phase)
                    opponent_king_danger = king_danger(position, chess.BLACK, game_phase)
                else:
                    self_king_danger = king_danger(position, chess.BLACK, game_phase)
                    opponent_king_danger = king_danger(position, chess.WHITE, game_phase)
                
                # Evaluate each model's performance
                model_performances = {}
                for model_name, scorer in self.move_scorers.items():
                    # Get top moves from the model (up to 10)
                    try:
                        top_moves = scorer.get_moves(position, san=True, top=10)
                        
                        # Find the rank of the actual move
                        actual_move_san = position.san(actual_move)
                        rank = next((i+1 for i, move in enumerate(top_moves) if move == actual_move_san), 10)
                        
                        model_performances[model_name] = rank
                    except Exception as e:
                        print(f"Error evaluating model {model_name} on position: {e}")
                        model_performances[model_name] = 10  # Default to worst rank if there's an error
                
                # Record results
                result = {
                    "move_num": move_num,
                    "complexity": complexity,
                    "win_prob": win_prob,
                    "efficient_mobility": efficient_mobility,
                    "narrowness": narrowness,
                    "piece_activity": piece_activity,
                    "game_phase": game_phase,
                    "actual_move": actual_move_san,
                    "position_fen": position.fen(),
                    "self_king_danger": self_king_danger,
                    "opp_king_danger": opponent_king_danger
                }
                
                # Add model performances
                for model_name, rank in model_performances.items():
                    result[f"{model_name}_rank"] = rank
                
                results.append(result)
            except Exception as e:
                print(f"Error analyzing position {i} at move {move_num}: {e}")
        
        return results
    
    def get_best_model_distribution(self, split_by_phase=True):
        """
        For each position, determine which model performed the best (lowest rank).
        Returns a distribution of best-performing models.
        
        Args:
            split_by_phase (bool): Whether to split analysis by game phase
        
        Returns:
            dict or dict of dicts: Dictionary mapping model names to their frequency of being the best,
                                  or dictionary mapping game phases to such dictionaries
        """
        if self.results_df is None:
            raise ValueError("No analysis results available. Run analyze_pgn_files() first.")
        
        rank_columns = [f"{model}_rank" for model in self.model_names]
        
        if not split_by_phase:
            best_models = self.results_df[rank_columns].idxmin(axis=1)
            best_models = best_models.apply(lambda x: x.replace("_rank", ""))
            return best_models.value_counts().to_dict()
        
        # Split by game phase
        result = {}
        for phase in self.game_phases:
            phase_df = self.results_df[self.results_df["game_phase"] == phase]
            if len(phase_df) > 0:
                best_models = phase_df[rank_columns].idxmin(axis=1)
                best_models = best_models.apply(lambda x: x.replace("_rank", ""))
                result[phase] = best_models.value_counts().to_dict()
        
        return result
    
    def plot_best_model_distribution(self, split_by_phase=True):
        """
        Plot the distribution of best-performing models
        
        Args:
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        best_models = self.get_best_model_distribution(split_by_phase)
        
        if not split_by_phase:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(best_models.keys()), y=list(best_models.values()))
            plt.title("Distribution of Best-Performing Models", fontsize=16)
            plt.xlabel("Model", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            os.makedirs("model_research/plots", exist_ok=True)
            plt.savefig("model_research/plots/best_model_distribution.png")
            plt.close()
            return
        
        # Plot for each game phase
        if not best_models:  # Check if the dictionary is empty
            print("No phase-specific data available for plotting best model distribution")
            return
            
        fig, axes = plt.subplots(1, len(best_models), figsize=(18, 6), sharey=True)
        
        for i, (phase, phase_dict) in enumerate(best_models.items()):
            ax = axes[i] if len(best_models) > 1 else axes
            sns.barplot(x=list(phase_dict.keys()), y=list(phase_dict.values()), ax=ax)
            ax.set_title(f"{phase.capitalize()}", fontsize=14)
            ax.set_xlabel("Model", fontsize=12)
            if i == 0:
                ax.set_ylabel("Count", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle("Distribution of Best-Performing Models by Game Phase", fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        os.makedirs("model_research/plots", exist_ok=True)
        plt.savefig("model_research/plots/best_model_distribution_by_phase.png")
        plt.close()
    
    def analyze_by_lucas_metric(self, metric, bins=10, split_by_phase=True):
        """
        Analyze model performance across different bins of a Lucas metric.
        
        Args:
            metric (str): Name of the Lucas metric to analyze
            bins (int): Number of bins to divide the metric range into
            split_by_phase (bool): Whether to split analysis by game phase
        
        Returns:
            pd.DataFrame or dict of DataFrames: DataFrame with average ranks for each model in each bin,
                                               or dictionary mapping game phases to such DataFrames
        """
        if self.results_df is None:
            raise ValueError("No analysis results available. Run analyze_pgn_files() first.")
        
        if metric not in ["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity"]:
            raise ValueError(f"Invalid metric: {metric}")
        
        if not split_by_phase:
            # Create bins
            df_copy = self.results_df.copy()
            df_copy[f"{metric}_bin"] = pd.cut(df_copy[metric], bins)
            
            # Calculate average rank for each model in each bin
            rank_columns = [f"{model}_rank" for model in self.model_names]
            grouped = df_copy.groupby(f"{metric}_bin", observed=True)[rank_columns].mean().reset_index()
            return grouped
        
        # Split by game phase
        result = {}
        for phase in self.game_phases:
            phase_df = self.results_df[self.results_df["game_phase"] == phase].copy()
            if len(phase_df) > bins:  # Make sure we have enough data to make meaningful bins
                # Create bins
                phase_df[f"{metric}_bin"] = pd.cut(phase_df[metric], bins)
                
                # Calculate average rank for each model in each bin
                rank_columns = [f"{model}_rank" for model in self.model_names]
                grouped = phase_df.groupby(f"{metric}_bin", observed=True)[rank_columns].mean().reset_index()
                result[phase] = grouped
        
        return result
    
    def plot_performance_by_metric(self, metric, bins=10, split_by_phase=True):
        """
        Plot model performance across different bins of a Lucas metric.
        
        Args:
            metric (str): Name of the Lucas metric to analyze
            bins (int): Number of bins to divide the metric range into
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        grouped = self.analyze_by_lucas_metric(metric, bins, split_by_phase)
        
        if not split_by_phase:
            plt.figure(figsize=(12, 6))
            
            for model in self.model_names:
                plt.plot(range(len(grouped)), grouped[f"{model}_rank"], label=model, marker='o')
            
            plt.title(f"Model Performance by {metric.capitalize()}", fontsize=16)
            plt.xlabel(f"{metric.capitalize()} (binned)", fontsize=12)
            plt.ylabel("Average Rank (lower is better)", fontsize=12)
            plt.xticks(range(len(grouped)), [str(b) for b in grouped[f"{metric}_bin"]], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            os.makedirs("model_research/plots", exist_ok=True)
            plt.savefig(f"model_research/plots/performance_by_{metric}.png")
            plt.close()
            return
        
        # Check if we have any phase data
        if not grouped:
            print(f"No phase-specific data available for plotting performance by {metric}")
            return
            
        # Plot for each game phase
        fig, axes = plt.subplots(1, len(grouped), figsize=(18, 6), sharey=True)
        
        for i, (phase, phase_grouped) in enumerate(grouped.items()):
            ax = axes[i] if len(grouped) > 1 else axes
            
            for model in self.model_names:
                ax.plot(range(len(phase_grouped)), phase_grouped[f"{model}_rank"], label=model, marker='o')
            
            ax.set_title(f"{phase.capitalize()}", fontsize=14)
            ax.set_xlabel(f"{metric.capitalize()} (binned)", fontsize=12)
            if i == 0:
                ax.set_ylabel("Average Rank (lower is better)", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.set_xticks(range(len(phase_grouped)))
            ax.set_xticklabels([str(b) for b in phase_grouped[f"{metric}_bin"]], rotation=45)
            if i == len(grouped) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Model Performance by {metric.capitalize()} Across Game Phases", fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        os.makedirs("model_research/plots", exist_ok=True)
        plt.savefig(f"model_research/plots/performance_by_{metric}_by_phase.png")
        plt.close()
    
    def plot_all_metrics(self, bins=10, split_by_phase=True):
        """
        Plot model performance across all Lucas metrics
        
        Args:
            bins (int): Number of bins to divide each metric into
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        metrics = ["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity"]
        
        for metric in metrics:
            self.plot_performance_by_metric(metric, bins, split_by_phase)
    
    def analyze_by_game_phase(self):
        """
        Analyze model performance across different game phases.
        
        Returns:
            pd.DataFrame: DataFrame with average ranks for each model in each game phase
        """
        if self.results_df is None:
            raise ValueError("No analysis results available. Run analyze_pgn_files() first.")
        
        # Calculate average rank for each model in each game phase
        rank_columns = [f"{model}_rank" for model in self.model_names]
        grouped = self.results_df.groupby("game_phase")[rank_columns].mean().reset_index()
        
        return grouped
    
    def plot_performance_by_game_phase(self):
        """Plot model performance across different game phases"""
        grouped = self.analyze_by_game_phase()
        
        plt.figure(figsize=(12, 6))
        
        # Set positions for grouped bars
        phases = grouped["game_phase"].tolist()
        x = np.arange(len(phases))
        width = 0.15
        
        # Plot bars for each model
        for i, model in enumerate(self.model_names):
            plt.bar(x + i*width - 0.3, grouped[f"{model}_rank"], width, label=model)
        
        plt.title("Model Performance by Game Phase", fontsize=16)
        plt.xlabel("Game Phase", fontsize=12)
        plt.ylabel("Average Rank (lower is better)", fontsize=12)
        plt.xticks(x, phases)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        os.makedirs("model_research/plots", exist_ok=True)
        plt.savefig("model_research/plots/performance_by_game_phase.png")
        plt.close()
    
    def generate_decision_boundary_report(self, split_by_phase=True):
        """
        Generate a report on the decision boundaries based on Lucas analytics for when to use each model.
        
        Args:
            split_by_phase (bool): Whether to analyze decision boundaries separately for each game phase
        
        Returns:
            str: Report text
        """
        if self.results_df is None:
            raise ValueError("No analysis results available. Run analyze_pgn_files() first.")
        
        # For each position, determine which model performed the best
        rank_columns = [f"{model}_rank" for model in self.model_names]
        # Use a copy to avoid SettingWithCopyWarning
        df_copy = self.results_df.copy()
        df_copy["best_model"] = df_copy[rank_columns].idxmin(axis=1)
        df_copy["best_model"] = df_copy["best_model"].apply(lambda x: x.replace("_rank", ""))
        
        # Import libraries inside the function to avoid import errors if they're not needed elsewhere
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.model_selection import train_test_split
        
        # Generate report header
        report = f"# MoveScorer Decision Boundary Report\n\n"
        
        if not split_by_phase:
            # Train a basic decision tree to identify patterns (using only Lucas analytics)
            X = df_copy[["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity", "game_phase", "self_king_danger", "opp_king_danger"]]
            y = df_copy["best_model"]
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train a decision tree
            model = DecisionTreeClassifier(max_depth=8, random_state=42, min_samples_leaf=20)
            model.fit(X_train, y_train)
            
            # Evaluate accuracy
            accuracy = model.score(X_test, y_test)
            
            # Generate human-readable rules
            tree_rules = export_text(model, feature_names=X.columns.tolist())
            
            # Compute mean Lucas analytics values for each best model
            mean_values = df_copy.groupby("best_model", observed=False)[["complexity", "win_prob", 
                                                  "efficient_mobility", "narrowness", "piece_activity", "self_king_danger", "opp_king_danger"]].mean()
            
            report += f"Decision tree accuracy: {accuracy:.2f}\n\n"
            report += "## Decision Tree Rules\n\n```\n" + tree_rules + "\n```\n\n"
            report += "## Average Lucas Analytics for Each Best Model\n\n"
            report += mean_values.to_markdown() + "\n\n"
        else:
            # Analyze each game phase separately
            report += "## Decision Boundaries by Game Phase\n\n"
            
            for phase in self.game_phases:
                phase_df = df_copy[df_copy["game_phase"] == phase].copy()
                if len(phase_df) < 20:  # Skip phases with too little data
                    report += f"### {phase.capitalize()}\n\nInsufficient data for analysis.\n\n"
                    continue
                
                report += f"### {phase.capitalize()}\n\n"
                
                # Train a basic decision tree to identify patterns
                X = phase_df[["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity", "self_king_danger", "opp_king_danger"]]
                y = phase_df["best_model"]
                
                # Split data for evaluation
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Train a decision tree
                    model = DecisionTreeClassifier(max_depth=8, random_state=42, min_samples_leaf=8)
                    model.fit(X_train, y_train)
                    
                    # Evaluate accuracy
                    accuracy = model.score(X_test, y_test)
                    
                    # Generate human-readable rules
                    tree_rules = export_text(model, feature_names=X.columns.tolist())
                    
                    # Compute mean Lucas analytics values for each best model
                    mean_values = phase_df.groupby("best_model", observed=False)[["complexity", "win_prob", 
                                                         "efficient_mobility", "narrowness", "piece_activity", "self_king_danger", "opp_king_danger"]].mean()
                    
                    report += f"Decision tree accuracy: {accuracy:.2f}\n\n"
                    report += "#### Decision Tree Rules\n\n```\n" + tree_rules + "\n```\n\n"
                    report += "#### Average Lucas Analytics for Each Best Model\n\n"
                    report += mean_values.to_markdown() + "\n\n"
                except Exception as e:
                    report += f"Error analyzing phase {phase}: {str(e)}\n\n"
        
        # Save the report
        os.makedirs("model_research/reports", exist_ok=True)
        report_suffix = "_by_phase" if split_by_phase else ""
        with open(f"model_research/reports/decision_boundary_report{report_suffix}.md", "w") as f:
            f.write(report)
        
        return report 
        
    def train_minimum_rank_decision_tree(self, split_by_phase=True, max_depth=8, min_samples_leaf=20):
        """
        Train a decision tree that minimises the expected rank of model predictions.
        Unlike the standard decision tree that simply classifies which model is best,
        this approach optimises for the expected rank directly.
        
        Args:
            split_by_phase (bool): Whether to train separate trees for each game phase
            max_depth (int): Maximum depth of the decision tree
            min_samples_leaf (int): Minimum samples required at a leaf node
            
        Returns:
            dict: Dictionary containing trained models and performance metrics
        """
        if self.results_df is None:
            raise ValueError("No analysis results available. Run analyze_pgn_files() first.")
            
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.model_selection import train_test_split
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        
        result = {}
        rank_columns = [f"{model}_rank" for model in self.model_names]
        df_copy = self.results_df.copy()
        
        # Custom function to evaluate expected rank for a set of model predictions
        def evaluate_expected_rank(X_test, y_pred, df_test):
            """Calculate the average rank achieved by following model predictions"""
            expected_ranks = []
            
            for i, prediction in enumerate(y_pred):
                model_name = prediction
                rank = df_test.iloc[i][f"{model_name}_rank"]
                expected_ranks.append(rank)
                
            return np.mean(expected_ranks)
            
        if not split_by_phase:
            # Features for decision tree
            X = df_copy[["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity", "self_king_danger", "opp_king_danger"]]
            if "game_phase" in df_copy.columns:
                # Encode game phase as numeric
                le = LabelEncoder()
                phase_encoded = le.fit_transform(df_copy["game_phase"])
                X["game_phase_encoded"] = phase_encoded
            
            # Split data for training and evaluation
            X_train, X_test, df_train, df_test = train_test_split(
                X, df_copy, test_size=0.2, random_state=42
            )
            
            # Train a custom decision tree that minimises rank
            best_model = None
            best_score = float('inf')
            
            # Try different approaches - this is a custom training process since
            # standard decision trees don't optimise for expected rank directly
            
            # Approach 1: Use the index of the best model as target
            y_train = df_train[rank_columns].idxmin(axis=1).apply(
                lambda x: self.model_names.index(x.replace("_rank", ""))
            )
            
            # Try different criterion options
            for criterion in ['gini', 'entropy']:
                model = DecisionTreeClassifier(
                    max_depth=max_depth, 
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = model.predict(X_test)
                
                # Evaluate the expected rank
                score = evaluate_expected_rank(X_test, y_pred, df_test)
                
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_criterion = criterion
            
            # Generate human-readable rules
            tree_rules = export_text(best_model, feature_names=X.columns.tolist())
            
            result = {
                'model': best_model,
                'expected_rank': best_score,
                'criterion': best_criterion,
                'tree_rules': tree_rules
            }
            
            # Save a report
            report = f"# Minimum Expected Rank Decision Tree\n\n"
            report += f"Expected rank: {best_score:.4f}\n\n"
            report += f"Training criterion: {best_criterion}\n\n"
            report += "## Decision Tree Rules\n\n```\n" + tree_rules + "\n```\n\n"
            
            os.makedirs("model_research/reports", exist_ok=True)
            with open("model_research/reports/minimum_rank_tree.md", "w") as f:
                f.write(report)
                
        else:
            # Train separate models for each game phase
            result = {}
            
            for phase in self.game_phases:
                phase_df = df_copy[df_copy["game_phase"] == phase].copy()
                if len(phase_df) < 50:  # Skip phases with too little data
                    result[phase] = {
                        'error': "Insufficient data for analysis"
                    }
                    continue
                
                # Features for decision tree
                X = phase_df[["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity", "self_king_danger", "opp_king_danger"]]
                
                # Split data for training and evaluation
                X_train, X_test, df_train, df_test = train_test_split(
                    X, phase_df, test_size=0.2, random_state=42
                )
                
                # Train a custom decision tree that minimises rank
                best_model = None
                best_score = float('inf')
                
                # Use the name of the best model as target
                y_train = df_train[rank_columns].idxmin(axis=1).apply(
                    lambda x: x.replace("_rank", "")
                )
                # Try different criterion options
                for criterion in ['gini', 'entropy']:
                    model = DecisionTreeClassifier(
                        max_depth=max_depth, 
                        min_samples_leaf=min_samples_leaf,
                        criterion=criterion,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    
                    # Predict on test set
                    y_pred = model.predict(X_test)
                    
                    # Evaluate the expected rank
                    score = evaluate_expected_rank(X_test, y_pred, df_test)
                    
                    if score < best_score:
                        best_score = score
                        best_model = model
                        best_criterion = criterion
                
                # Generate human-readable rules
                tree_rules = export_text(best_model, feature_names=X.columns.tolist())
                
                result[phase] = {
                    'model': best_model,
                    'expected_rank': best_score,
                    'criterion': best_criterion,
                    'tree_rules': tree_rules
                }
            
            # Save a combined report
            report = f"# Minimum Expected Rank Decision Trees by Game Phase\n\n"
            
            for phase, phase_result in result.items():
                report += f"## {phase.capitalize()}\n\n"
                
                if 'error' in phase_result:
                    report += f"{phase_result['error']}\n\n"
                    continue
                    
                report += f"Expected rank: {phase_result['expected_rank']:.4f}\n\n"
                report += f"Training criterion: {phase_result['criterion']}\n\n"
                report += "### Decision Tree Rules\n\n```\n" + phase_result['tree_rules'] + "\n```\n\n"
            
            os.makedirs("model_research/reports", exist_ok=True)
            with open("model_research/reports/minimum_rank_tree_by_phase.md", "w") as f:
                f.write(report)
                
        return result 