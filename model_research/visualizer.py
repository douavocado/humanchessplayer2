import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import chess
import chess.svg
from io import BytesIO
from PIL import Image
from cairosvg import svg2png

class MovesScorerVisualizer:
    """
    Provides additional visualizations for MoveScorer analysis results.
    """
    
    def __init__(self, results_df):
        """
        Initialize the visualizer with results DataFrame.
        
        Args:
            results_df (pd.DataFrame): DataFrame with MoveScorer analysis results
        """
        self.results_df = results_df
        self.model_names = [col.replace('_rank', '') for col in results_df.columns if col.endswith('_rank')]
        self.game_phases = ['opening', 'midgame', 'endgame']
        
        # Create plots directory if it doesn't exist
        os.makedirs("model_research/plots", exist_ok=True)
    
    def heatmap_model_performance_by_metrics(self, bins=5, split_by_phase=True):
        """
        Create heatmaps showing model performance across bins of Lucas metrics.
        
        Args:
            bins (int): Number of bins to divide each metric into
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        metrics = ["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity"]
        
        # If splitting by phase, process each phase separately
        phase_data = [self.results_df]
        phase_names = ['overall']
        
        if split_by_phase:
            phase_data = []
            phase_names = []
            for phase in self.game_phases:
                # Make a proper copy to avoid SettingWithCopyWarning
                phase_df = self.results_df[self.results_df['game_phase'] == phase].copy()
                if len(phase_df) > 0:
                    phase_data.append(phase_df)
                    phase_names.append(phase)
        
        for phase_df, phase_name in zip(phase_data, phase_names):
            for model_name in self.model_names:
                plt.figure(figsize=(15, 12))
                
                # Create a 2x3 grid of heatmaps for different pairs of metrics
                plot_idx = 1
                for i, metric1 in enumerate(metrics):
                    for metric2 in metrics[i+1:]:
                        # Ensure we don't exceed the 2x3 grid (6 total plots)
                        if plot_idx > 6:
                            break
                            
                        plt.subplot(2, 3, plot_idx)
                        
                        # Make sure we have enough data points for binning
                        if len(phase_df) >= bins:
                            # Create bins for both metrics
                            phase_df[f"{metric1}_bin"] = pd.cut(phase_df[metric1], bins, labels=False)
                            phase_df[f"{metric2}_bin"] = pd.cut(phase_df[metric2], bins, labels=False)
                            
                            # Calculate average rank for each bin combination
                            pivot = phase_df.pivot_table(
                                values=f"{model_name}_rank",
                                index=f"{metric1}_bin",
                                columns=f"{metric2}_bin",
                                aggfunc='mean'
                            )
                            
                            # Create heatmap
                            sns.heatmap(pivot, cmap="YlGnBu_r", annot=True, fmt=".2f", 
                                      cbar_kws={'label': 'Average Rank (lower is better)'})
                        else:
                            # Not enough data for this phase
                            plt.text(0.5, 0.5, f"Insufficient data for {phase_name}", 
                                    ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
                        
                        plt.title(f"{metric1.capitalize()} vs {metric2.capitalize()}")
                        plt.xlabel(metric2.capitalize())
                        plt.ylabel(metric1.capitalize())
                        
                        plot_idx += 1
                    
                    # If we've already plotted 6 subplots, break from the outer loop too
                    if plot_idx > 6:
                        break
                
                phase_suffix = f"_{phase_name}" if split_by_phase else ""
                plt.suptitle(f"{model_name.capitalize()} Performance Across Lucas Metrics ({phase_name.capitalize()})", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
                
                # Save the plot
                plt.savefig(f"model_research/plots/{model_name}_heatmap_metrics{phase_suffix}.png")
                plt.close()
    
    def plot_rank_distributions(self, split_by_phase=True):
        """
        Plot distributions of ranks for each model.
        
        Args:
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        # If splitting by phase, process each phase separately
        phase_data = [self.results_df]
        phase_names = ['overall']
        
        if split_by_phase:
            phase_data = []
            phase_names = []
            for phase in self.game_phases:
                # Make a proper copy to avoid SettingWithCopyWarning
                phase_df = self.results_df[self.results_df['game_phase'] == phase].copy()
                if len(phase_df) > 0:
                    phase_data.append(phase_df)
                    phase_names.append(phase)
        
        for phase_df, phase_name in zip(phase_data, phase_names):
            # Get rank columns
            rank_columns = [f"{model}_rank" for model in self.model_names]
            
            # Only proceed if we have data for this phase
            if len(phase_df) == 0:
                continue
                
            # Violin plots
            plt.figure(figsize=(14, 8))
            sns.violinplot(data=phase_df[rank_columns])
            
            phase_suffix = f" ({phase_name.capitalize()})" if split_by_phase else ""
            plt.title(f"Rank Distribution by Model{phase_suffix}", fontsize=16)
            plt.xlabel("Model", fontsize=12)
            plt.ylabel("Rank (lower is better)", fontsize=12)
            plt.xticks(range(len(self.model_names)), [model.capitalize() for model in self.model_names], rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            file_suffix = f"_{phase_name}" if split_by_phase else ""
            plt.savefig(f"model_research/plots/rank_distributions{file_suffix}.png")
            plt.close()
            
            # Also create a boxplot version
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=phase_df[rank_columns])
            plt.title(f"Rank Distribution by Model{phase_suffix} (Boxplot)", fontsize=16)
            plt.xlabel("Model", fontsize=12)
            plt.ylabel("Rank (lower is better)", fontsize=12)
            plt.xticks(range(len(self.model_names)), [model.capitalize() for model in self.model_names], rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f"model_research/plots/rank_distributions_boxplot{file_suffix}.png")
            plt.close()
    
    def plot_metric_correlation_with_performance(self, split_by_phase=True):
        """
        Plot correlation between each Lucas metric and model performance.
        
        Args:
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        metrics = ["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity"]
        rank_columns = [f"{model}_rank" for model in self.model_names]
        
        # If splitting by phase, process each phase separately
        phase_data = [self.results_df]
        phase_names = ['overall']
        
        if split_by_phase:
            phase_data = []
            phase_names = []
            for phase in self.game_phases:
                # Make a proper copy to avoid SettingWithCopyWarning
                phase_df = self.results_df[self.results_df['game_phase'] == phase].copy()
                if len(phase_df) > 0:
                    phase_data.append(phase_df)
                    phase_names.append(phase)
        
        for phase_df, phase_name in zip(phase_data, phase_names):
            # Only proceed if we have data for this phase
            if len(phase_df) == 0:
                continue
                
            # Calculate correlation matrix
            corr_df = phase_df[metrics + rank_columns].corr()
            
            # Extract only the correlation between metrics and ranks
            corr_metrics_ranks = corr_df.loc[metrics, rank_columns]
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_metrics_ranks, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                       cbar_kws={'label': 'Correlation Coefficient'})
            
            phase_suffix = f" ({phase_name.capitalize()})" if split_by_phase else ""
            plt.title(f"Correlation: Lucas Metrics vs Model Performance{phase_suffix}", fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            file_suffix = f"_{phase_name}" if split_by_phase else ""
            plt.savefig(f"model_research/plots/metrics_performance_correlation{file_suffix}.png")
            plt.close()
    
    def plot_pca_metrics_and_performance(self, split_by_phase=True):
        """
        Plot PCA of Lucas metrics with model performance highlighted.
        
        Args:
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        metrics = ["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity"]
        
        # If splitting by phase, process each phase separately
        phase_data = [self.results_df]
        phase_names = ['overall']
        
        if split_by_phase:
            phase_data = []
            phase_names = []
            for phase in self.game_phases:
                # Make a proper copy to avoid SettingWithCopyWarning
                phase_df = self.results_df[self.results_df['game_phase'] == phase].copy()
                if len(phase_df) > 0:
                    phase_data.append(phase_df)
                    phase_names.append(phase)
        
        for phase_df, phase_name in zip(phase_data, phase_names):
            # Only proceed if we have data for this phase
            if len(phase_df) < 10:  # Need at least some data for PCA
                continue
                
            # For each position, determine which model performed the best
            rank_columns = [f"{model}_rank" for model in self.model_names]
            phase_df["best_model"] = phase_df[rank_columns].idxmin(axis=1)
            phase_df["best_model"] = phase_df["best_model"].apply(lambda x: x.replace("_rank", ""))
            
            # Standardize metrics for PCA
            X = phase_df[metrics]
            X_scaled = StandardScaler().fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df["best_model"] = phase_df["best_model"]
            
            # Plot PCA results
            plt.figure(figsize=(12, 10))
            
            # Create scatter plot
            for model in self.model_names:
                subset = pca_df[pca_df["best_model"] == model]
                if not subset.empty:
                    plt.scatter(subset['PC1'], subset['PC2'], label=model.capitalize(), alpha=0.7)
            
            # Add feature vectors
            feature_vectors = pca.components_.T
            feature_names = metrics
            
            # Scale vectors for visualization
            scale = 5
            for i, (feature, vec) in enumerate(zip(feature_names, feature_vectors)):
                plt.arrow(0, 0, scale * vec[0], scale * vec[1], color='k', alpha=0.5)
                plt.text(scale * vec[0] * 1.1, scale * vec[1] * 1.1, feature, color='k')
            
            phase_suffix = f" ({phase_name.capitalize()})" if split_by_phase else ""
            plt.title(f"PCA of Lucas Metrics with Best Model Performance{phase_suffix}", fontsize=16)
            plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
            plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            file_suffix = f"_{phase_name}" if split_by_phase else ""
            plt.savefig(f"model_research/plots/pca_metrics_performance{file_suffix}.png")
            plt.close()
    
    def plot_model_performance_radar(self, split_by_phase=True):
        """
        Create radar charts (spider plots) of each model's performance across metrics.
        
        Args:
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        metrics = ["complexity", "win_prob", "efficient_mobility", "narrowness", "piece_activity"]
        
        # If splitting by phase, process each phase separately
        phase_data = [self.results_df]
        phase_names = ['overall']
        
        if split_by_phase:
            phase_data = []
            phase_names = []
            for phase in self.game_phases:
                # Make a proper copy to avoid SettingWithCopyWarning
                phase_df = self.results_df[self.results_df['game_phase'] == phase].copy()
                if len(phase_df) > 0:
                    phase_data.append(phase_df)
                    phase_names.append(phase)
        
        for phase_df, phase_name in zip(phase_data, phase_names):
            # Only proceed if we have data for this phase
            if len(phase_df) == 0:
                continue
                
            # Create 5 bins for each metric
            for metric in metrics:
                phase_df[f"{metric}_bin"] = pd.cut(phase_df[metric], 5, labels=[1, 2, 3, 4, 5])
            
            # For each bin of each metric, calculate average rank for each model
            radar_data = {}
            for model in self.model_names:
                radar_data[model] = {}
                for metric in metrics:
                    bin_avgs = phase_df.groupby(f"{metric}_bin", observed=True)[f"{model}_rank"].mean()
                    # Invert the values so that higher is better for radar chart
                    if len(bin_avgs) > 0:  # Make sure we have data for all bins
                        radar_data[model][metric] = 10 - bin_avgs.values
                    else:
                        # Handle empty bins gracefully
                        radar_data[model][metric] = np.zeros(5)
            
            # Plot radar charts
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Close the loop
            
            fig, axes = plt.subplots(1, len(self.model_names), figsize=(20, 6), subplot_kw={'polar': True})
            
            for i, model in enumerate(self.model_names):
                ax = axes[i] if len(self.model_names) > 1 else axes
                
                # Add each bin as a separate line on the radar
                for bin_val in range(1, 6):
                    # Make sure all metrics have data for this bin
                    if all(metric in radar_data[model] and len(radar_data[model][metric]) >= bin_val for metric in metrics):
                        values = [radar_data[model][metric][bin_val-1] for metric in metrics]
                        values = np.concatenate((values, [values[0]]))  # Close the loop
                        
                        ax.plot(angles, values, linewidth=2, label=f"Bin {bin_val}")
                        ax.fill(angles, values, alpha=0.1)
                
                # Add labels and customize chart
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.set_title(f"{model.capitalize()}", fontsize=12)
                
                # Set y-limits for consistency
                ax.set_ylim(0, 10)
                
                # Only add legend to the first chart to save space
                if i == 0:
                    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            phase_suffix = f" ({phase_name.capitalize()})" if split_by_phase else ""
            plt.suptitle(f"Model Performance by Metric Bins{phase_suffix} (Higher is Better)", fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            file_suffix = f"_{phase_name}" if split_by_phase else ""
            plt.savefig(f"model_research/plots/model_performance_radar{file_suffix}.png")
            plt.close()
    
    def plot_rank_comparison_scatter(self, split_by_phase=True):
        """
        Create scatter plots comparing ranks of different model pairs.
        
        Args:
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        # Create model pairs for comparison
        model_pairs = []
        for i, model1 in enumerate(self.model_names):
            for model2 in self.model_names[i+1:]:
                model_pairs.append((model1, model2))
        
        # If splitting by phase, process each phase separately
        phase_data = [self.results_df]
        phase_names = ['overall']
        
        if split_by_phase:
            phase_data = []
            phase_names = []
            for phase in self.game_phases:
                # Make a proper copy to avoid SettingWithCopyWarning
                phase_df = self.results_df[self.results_df['game_phase'] == phase].copy()
                if len(phase_df) > 0:
                    phase_data.append(phase_df)
                    phase_names.append(phase)
        
        for phase_df, phase_name in zip(phase_data, phase_names):
            # Only proceed if we have data for this phase
            if len(phase_df) == 0:
                continue
                
            rows = (len(model_pairs) + 1) // 2
            plt.figure(figsize=(16, 4 * rows))
            
            for i, (model1, model2) in enumerate(model_pairs):
                plt.subplot(rows, 2, i+1)
                
                # Add jitter to ranks for better visualization
                jitter = np.random.normal(0, 0.1, len(phase_df))
                x = phase_df[f"{model1}_rank"] + jitter
                y = phase_df[f"{model2}_rank"] + jitter
                
                # Create color based on which model was better
                colors = np.where(x < y, 'blue', np.where(x > y, 'red', 'purple'))
                
                plt.scatter(x, y, alpha=0.5, c=colors)
                
                # Add diagonal line
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                # Add frequency counts as text
                better_count = np.sum(x < y)
                worse_count = np.sum(x > y)
                equal_count = np.sum(x == y)
                
                plt.annotate(f"{model1} better: {better_count} ({better_count/len(x):.1%})",
                            xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, ha='left', va='top')
                plt.annotate(f"{model2} better: {worse_count} ({worse_count/len(x):.1%})",
                            xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10, ha='left', va='top')
                plt.annotate(f"Equal: {equal_count} ({equal_count/len(x):.1%})",
                            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, ha='left', va='top')
                
                plt.title(f"{model1.capitalize()} vs {model2.capitalize()}")
                plt.xlabel(f"{model1.capitalize()} Rank")
                plt.ylabel(f"{model2.capitalize()} Rank")
                plt.xlim(0.5, 10.5)
                plt.ylim(0.5, 10.5)
                plt.grid(True, alpha=0.3)
            
            phase_suffix = f" ({phase_name.capitalize()})" if split_by_phase else ""
            plt.suptitle(f"Rank Comparison Between Models{phase_suffix}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
            
            # Save the plot
            file_suffix = f"_{phase_name}" if split_by_phase else ""
            plt.savefig(f"model_research/plots/rank_comparison_scatter{file_suffix}.png")
            plt.close()
    
    def visualize_top_model_examples(self, target_models=None, min_rank_difference=3, num_examples=9, save_fig=True):
        """
        Visualize example positions where specific models were uncontestably the best choice.
        
        Args:
            target_models (list): Models to visualize examples for. Default is ['tactics', 'defensive_tactics']
            min_rank_difference (int): Minimum rank difference to consider a model uncontestably best
            num_examples (int): Number of examples to show for each model
            save_fig (bool): Whether to save the figure
            
        Returns:
            None: Displays and optionally saves visualizations of board positions
        """
        if target_models is None:
            target_models = ['tactics', 'defensive_tactics']
        
        # Ensure all specified models exist in the results
        for model in target_models:
            if model not in self.model_names:
                raise ValueError(f"Model '{model}' not found in results. Available models: {self.model_names}")
        
        # Filter positions for each target model
        for model in target_models:
            print(f"Finding positions where {model} was uncontestably the best model...")
            
            # Filter positions where this model had rank 1
            model_best = self.results_df[self.results_df[f"{model}_rank"] == 1].copy()
            
            if len(model_best) == 0:
                print(f"No positions found where {model} was ranked #1")
                continue
            
            # For each position, calculate rank difference between this model and second-best model
            for idx, row in model_best.iterrows():
                all_ranks = [row[f"{other}_rank"] for other in self.model_names if other != model]
                model_best.loc[idx, 'rank_diff'] = min(all_ranks) - 1  # Difference between 1 and second best
            
            # Filter positions where the rank difference exceeds the minimum
            uncontested = model_best[model_best['rank_diff'] >= min_rank_difference]
            
            if len(uncontested) == 0:
                print(f"No positions found where {model} was uncontestably better (min diff {min_rank_difference})")
                continue
            
            # Randomly select examples rather than taking the top ones
            if len(uncontested) > num_examples:
                examples = uncontested.sample(num_examples).copy()
            else:
                examples = uncontested.copy()
            
            # Create a figure with num_examples subplots arranged in a grid
            rows = int(np.ceil(np.sqrt(len(examples))))
            cols = int(np.ceil(len(examples) / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            axes = axes.flatten() if len(examples) > 1 else [axes]  # Flatten for easier iteration
            if len(examples) == 1:
                axes = [axes]  # Make axes always iterable
            
            # For each example position
            for i, (_, position) in enumerate(examples.iterrows()):
                # Create a chess board from the FEN
                try:
                    board = chess.Board(position['position_fen'])
                    
                    # Generate SVG of the board
                    svg_data = chess.svg.board(
                        board=board,
                        size=400,
                        coordinates=True
                    )
                    
                    # Convert SVG to PNG
                    png_data = svg2png(bytestring=svg_data)
                    img_data = BytesIO(png_data)
                    
                    # Create an image from the PNG data
                    img = Image.open(img_data)
                    
                    # Display the image
                    axes[i].imshow(img)
                    axes[i].axis('off')
                    
                    # Add metrics and other information
                    metrics_text = (
                        f"Complexity: {position['complexity']:.2f}\n"
                        f"Win Prob: {position['win_prob']:.2f}\n"
                        f"Eff Mobility: {position['efficient_mobility']:.2f}\n"
                        f"Narrowness: {position['narrowness']:.2f}\n"
                        f"Piece Activity: {position['piece_activity']:.2f}\n"
                        f"Game Phase: {position['game_phase']}\n"
                        f"Move: {position['actual_move']}\n"
                        f"Move Number: {position['move_num']}\n"
                        f"Self King Danger: {position['self_king_danger']:.2f}\n"
                        f"Opp King Danger: {position['opp_king_danger']:.2f}\n"
                    )
                    
                    # Add model ranks information
                    ranks_text = "Model Rankings:\n"
                    for other_model in self.model_names:
                        ranks_text += f"{other_model}: {position[f'{other_model}_rank']:.1f}\n"
                    
                    # Add title with rank difference
                    axes[i].set_title(
                        f"{model.capitalize()} wins by {position['rank_diff']:.1f} ranks",
                        fontsize=12
                    )
                    
                    # Add text boxes for metrics and ranks
                    axes[i].text(
                        1.25, 0.75, metrics_text,
                        transform=axes[i].transAxes,
                        fontsize=8, verticalalignment='top',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
                    )
                    
                except Exception as e:
                    print(f"Error creating board visualization: {e}")
                    axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.suptitle(f"Example Positions Where {model.capitalize()} Model Excels", fontsize=16, y=1.05)
            
            # Save the figure
            if save_fig:
                plt.savefig(f"model_research/plots/{model}_best_examples.png", bbox_inches='tight', dpi=150)
                print(f"Saved figure: model_research/plots/{model}_best_examples.png")
            
            plt.show()
    
    def generate_visualization_report(self, split_by_phase=True):
        """
        Generate a comprehensive visualization report with all plots.
        
        Args:
            split_by_phase (bool): Whether to split visualizations by game phase
        """
        self.plot_rank_distributions(split_by_phase)
        self.plot_metric_correlation_with_performance(split_by_phase)
        self.plot_pca_metrics_and_performance(split_by_phase)
        self.plot_model_performance_radar(split_by_phase)
        self.plot_rank_comparison_scatter(split_by_phase)
        self.heatmap_model_performance_by_metrics(split_by_phase=split_by_phase)
        
        # Generate example positions for tactics and defensive_tactics models
        try:
            self.visualize_top_model_examples(target_models=['tactics', 'defensive_tactics'], 
                                             min_rank_difference=4, 
                                             num_examples=9)
        except Exception as e:
            print(f"Warning: Could not generate model example visualizations: {e}")
        
        # Return a markdown summary for the report
        phase_text = "each game phase (opening, midgame, endgame)" if split_by_phase else "overall dataset"
        
        report = f"""# MoveScorer Visualization Report

## Overview
This report presents various visualizations of MoveScorer model performance across different chess situations as characterized by Lucas analytics. Visualizations are provided for {phase_text}.

## Visualizations Included

1. **Rank Distributions** - Shows the distribution of ranks for each model
2. **Metrics-Performance Correlation** - Heatmap showing correlation between Lucas metrics and model performance
3. **PCA of Metrics and Performance** - Principal Component Analysis of Lucas metrics with best model performance highlighted
4. **Model Performance Radar Charts** - Radar charts showing model performance across different metric bins
5. **Rank Comparison Scatter Plots** - Scatter plots comparing ranks between pairs of models
6. **Heatmaps of Model Performance** - Heatmaps showing model performance across bins of Lucas metrics
7. **Example Positions for Specialist Models** - Visualizations of chess positions where tactics and defensive tactics models excel

## Interpretation

The visualizations in this report help identify which models perform best in different chess situations. Key observations:

- Models tend to specialize in certain types of positions as characterized by Lucas analytics
- There are clear patterns in how model performance relates to game complexity, win probability, efficiency, narrowness, and piece activity
- The PCA visualization shows the clustering of positions where each model performs best
- Performance patterns differ across game phases, highlighting the importance of phase-specific model selection
- Example positions where tactics and defensive tactics models excel help illustrate the specific board patterns where these specialist models should be preferred

These insights can be used to develop a model selection strategy based on board characteristics and game phase.
"""
        
        # Save the report
        os.makedirs("model_research/reports", exist_ok=True)
        report_suffix = "_by_phase" if split_by_phase else ""
        with open(f"model_research/reports/visualization_report{report_suffix}.md", "w") as f:
            f.write(report)
        
        return report 