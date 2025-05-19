#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chess
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

def load_data(data_path, subset='train'):
    """Load data from H5 files."""
    # Get the latest file in the directory
    files = [f for f in os.listdir(os.path.join(data_path, subset)) if f.startswith('data')]
    if not files:
        raise FileNotFoundError(f"No data files found in {os.path.join(data_path, subset)}")
    
    latest_file = sorted(files)[-1]
    file_path = os.path.join(data_path, subset, latest_file)
    
    # Load both input data and labels
    print(f"Loading data from {file_path}...")
    input_data = pd.read_hdf(file_path, key='data')
    labels = pd.read_hdf(file_path, key='label')
    
    return input_data, labels

def plot_move_heatmap(labels, title='Distribution of Source Squares'):
    """Plot a heatmap showing the frequency of source squares."""
    # Sum the one-hot encoded labels to get counts
    move_counts = labels.sum(axis=0).values
    
    # Reshape to 8x8 board
    board_counts = move_counts.reshape(8, 8)
    
    # Create a chess-style heatmap
    plt.figure(figsize=(10, 10))
    
    # Customise colours for a chess board (light squares, dark squares with data overlay)
    cmap = LinearSegmentedColormap.from_list(
        'chessboard', 
        [(0.9, 0.9, 0.8), (0.2, 0.5, 0.7)], 
        N=100
    )
    
    # Plot the heatmap
    ax = sns.heatmap(board_counts, cmap=cmap, annot=True, fmt='.0f', cbar=True)
    
    # Add chess board coordinates
    ranks = '87654321'
    files = 'abcdefgh'
    
    plt.xticks(np.arange(8) + 0.5, files)
    plt.yticks(np.arange(8) + 0.5, ranks)
    
    # Overlay checkerboard pattern for visual clarity
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='gray', lw=1))
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    return plt.gcf()

def plot_piece_type_distribution(input_data, labels):
    """Plot the distribution of piece types that are moved."""
    # Convert one-hot labels to move indices
    move_indices = np.argmax(labels.values, axis=1)
    
    # Get the pieces at those positions from the input data
    # The piece encoding is in one-hot format, 12 channels per square
    piece_types = []
    
    for i, move_idx in enumerate(move_indices):
        # Extract the one-hot piece representation at the move square
        piece_encoding = input_data.iloc[i, move_idx*14:(move_idx+1)*14-2]  # -2 to exclude the threat level and en pris
        if np.any(piece_encoding > 0):
            piece_idx = np.argmax(piece_encoding)
            piece_types.append(piece_idx)
        else:
            piece_types.append(-1)  # No piece (should not happen)
    
    # Map piece indices to names
    piece_names = ['Black Pawn', 'Black Knight', 'Black Bishop', 'Black Rook', 'Black Queen', 'Black King',
                   'White Pawn', 'White Knight', 'White Bishop', 'White Rook', 'White Queen', 'White King']
    
    # Count occurrences
    unique_pieces, counts = np.unique(piece_types, return_counts=True)
    
    # Create a bar chart
    plt.figure(figsize=(12, 8))
    piece_labels = [piece_names[idx] for idx in unique_pieces if idx >= 0]
    valid_counts = [counts[i] for i, idx in enumerate(unique_pieces) if idx >= 0]
    
    colors = ['#999999', '#999999', '#999999', '#999999', '#999999', '#999999',
              '#dddddd', '#dddddd', '#dddddd', '#dddddd', '#dddddd', '#dddddd']
    piece_colors = [colors[idx] for idx in unique_pieces if idx >= 0]
    
    bars = plt.bar(piece_labels, valid_counts, color=piece_colors)
    plt.title('Distribution of Piece Types That Are Moved', fontsize=16)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.0f}',
                 ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    return plt.gcf()

def visualise_random_positions(input_data, labels, num_examples=5):
    """Visualise a few random positions and highlight the moved piece."""
    # Select random examples
    indices = np.random.choice(len(input_data), size=min(num_examples, len(input_data)), replace=False)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(5*len(indices), 5))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get the board position and the move
        position_data = input_data.iloc[idx]
        move_label = labels.iloc[idx]
        
        # Find the moved piece
        moved_from = np.argmax(move_label)
        
        # Create a chess board from the data
        board = chess.Board(chess.STARTING_FEN)
        board.clear_board()
        
        # Populate the board with pieces
        for square in range(64):
            piece_data = position_data[square*14:(square+1)*14-2]  # Exclude threat level
            if np.any(piece_data > 0):
                piece_idx = np.argmax(piece_data)
                
                # Map piece index to chess.Piece
                if piece_idx == 0:  # Black pawn
                    piece = chess.Piece(chess.PAWN, chess.BLACK)
                elif piece_idx == 1:  # Black knight
                    piece = chess.Piece(chess.KNIGHT, chess.BLACK)
                elif piece_idx == 2:  # Black bishop
                    piece = chess.Piece(chess.BISHOP, chess.BLACK)
                elif piece_idx == 3:  # Black rook
                    piece = chess.Piece(chess.ROOK, chess.BLACK)
                elif piece_idx == 4:  # Black queen
                    piece = chess.Piece(chess.QUEEN, chess.BLACK)
                elif piece_idx == 5:  # Black king
                    piece = chess.Piece(chess.KING, chess.BLACK)
                elif piece_idx == 6:  # White pawn
                    piece = chess.Piece(chess.PAWN, chess.WHITE)
                elif piece_idx == 7:  # White knight
                    piece = chess.Piece(chess.KNIGHT, chess.WHITE)
                elif piece_idx == 8:  # White bishop
                    piece = chess.Piece(chess.BISHOP, chess.WHITE)
                elif piece_idx == 9:  # White rook
                    piece = chess.Piece(chess.ROOK, chess.WHITE)
                elif piece_idx == 10:  # White queen
                    piece = chess.Piece(chess.QUEEN, chess.WHITE)
                elif piece_idx == 11:  # White king
                    piece = chess.Piece(chess.KING, chess.WHITE)
                
                board.set_piece_at(square, piece)
        
        # Create a SVG diagram
        axes[i].axis('off')
        
        # Draw the board
        for sq in range(64):
            file_idx = sq % 8
            rank_idx = 7 - (sq // 8)  # Flip to match chess notation
            
            # Draw the square
            square_color = '#FFCE9E' if (file_idx + rank_idx) % 2 == 0 else '#D18B47'
            rect = plt.Rectangle((file_idx, rank_idx), 1, 1, color=square_color)
            axes[i].add_patch(rect)
            
            # Highlight the moved piece with a border
            if sq == moved_from:
                highlight = plt.Rectangle((file_idx, rank_idx), 1, 1, fill=False, edgecolor='red', linewidth=3)
                axes[i].add_patch(highlight)
            
            # Add the piece
            piece = board.piece_at(sq)
            if piece:
                color = 'w' if piece.color == chess.WHITE else 'b'
                piece_symbol = piece.symbol().lower()
                
                # Map piece symbol to Unicode
                if piece_symbol == 'p':
                    unicode_symbol = '♟' if color == 'b' else '♙'
                elif piece_symbol == 'n':
                    unicode_symbol = '♞' if color == 'b' else '♘'
                elif piece_symbol == 'b':
                    unicode_symbol = '♝' if color == 'b' else '♗'
                elif piece_symbol == 'r':
                    unicode_symbol = '♜' if color == 'b' else '♖'
                elif piece_symbol == 'q':
                    unicode_symbol = '♛' if color == 'b' else '♕'
                elif piece_symbol == 'k':
                    unicode_symbol = '♚' if color == 'b' else '♔'
                
                # Add the piece symbol
                axes[i].text(file_idx + 0.5, rank_idx + 0.5, unicode_symbol, 
                            fontsize=24, ha='center', va='center',
                            color='black' if color == 'b' else 'white')
        
        # Add coordinates
        files = 'abcdefgh'
        ranks = '12345678'
        for j in range(8):
            axes[i].text(j + 0.5, -0.3, files[j], ha='center', va='center')
            axes[i].text(-0.3, j + 0.5, ranks[7-j], ha='center', va='center')
        
        axes[i].set_xlim(-0.5, 8.5)
        axes[i].set_ylim(-0.5, 8.5)
        axes[i].set_title(f"Position {idx+1} - Move from {chess.square_name(moved_from)}")
    
    plt.tight_layout()
    return fig

def threat_level_analysis(input_data, labels):
    """Analyse the threat levels of moved pieces."""
    # Convert one-hot labels to move indices
    move_indices = np.argmax(labels.values, axis=1)
    
    # Extract threat levels for moved pieces
    threat_levels = []
    
    for i, move_idx in enumerate(move_indices):
        # The threat level is the 13th value in each 14-value block per square
        threat_level = input_data.iloc[i, move_idx*14 + 12]
        threat_levels.append(threat_level)
    
    # Plot distribution of threat levels
    plt.figure(figsize=(10, 6))
    
    # Count occurrences
    unique_levels, counts = np.unique(threat_levels, return_counts=True)
    
    # Create a bar chart
    plt.bar([str(int(level)) for level in unique_levels], counts)
    plt.title('Distribution of Threat Levels for Moved Pieces', fontsize=16)
    plt.xlabel('Threat Level', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    plt.tight_layout()
    return plt.gcf()

def generate_visualisations(data_path='models/data/tactics/piece_selector'):
    """Generate all visualisations and save them to a directory."""
    # Create output directory
    output_dir = 'visualisations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_data, train_labels = load_data(data_path, 'train')
    test_data, test_labels = load_data(data_path, 'test')
    
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")
    
    # Generate visualisations
    print("Generating visualisations...")
    
    # 1. Move heatmaps
    train_heatmap = plot_move_heatmap(train_labels, "Training Data: Distribution of Source Squares")
    train_heatmap.savefig(os.path.join(output_dir, 'train_move_heatmap.png'), dpi=300)
    
    test_heatmap = plot_move_heatmap(test_labels, "Test Data: Distribution of Source Squares")
    test_heatmap.savefig(os.path.join(output_dir, 'test_move_heatmap.png'), dpi=300)
    
    # 2. Piece type distribution
    train_piece_dist = plot_piece_type_distribution(train_data, train_labels)
    train_piece_dist.savefig(os.path.join(output_dir, 'train_piece_distribution.png'), dpi=300)
    
    test_piece_dist = plot_piece_type_distribution(test_data, test_labels)
    test_piece_dist.savefig(os.path.join(output_dir, 'test_piece_distribution.png'), dpi=300)
    
    # 3. Random position visualisations
    train_positions = visualise_random_positions(train_data, train_labels, num_examples=3)
    train_positions.savefig(os.path.join(output_dir, 'train_random_positions.png'), dpi=300)
    
    test_positions = visualise_random_positions(test_data, test_labels, num_examples=3)
    test_positions.savefig(os.path.join(output_dir, 'test_random_positions.png'), dpi=300)
    
    # 4. Threat level analysis
    train_threat = threat_level_analysis(train_data, train_labels)
    train_threat.savefig(os.path.join(output_dir, 'train_threat_levels.png'), dpi=300)
    
    test_threat = threat_level_analysis(test_data, test_labels)
    test_threat.savefig(os.path.join(output_dir, 'test_threat_levels.png'), dpi=300)
    
    print(f"Visualisations saved to {output_dir}")
    
if __name__ == '__main__':
    generate_visualisations() 