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

def load_data(data_path, subset='train', sample_size=5000):
    """Load data from H5 files with optional sampling for large files."""
    # Get the latest file in the directory
    files = [f for f in os.listdir(os.path.join(data_path, subset)) if f.startswith('data')]
    if not files:
        raise FileNotFoundError(f"No data files found in {os.path.join(data_path, subset)}")
    
    latest_file = sorted(files)[-1]
    file_path = os.path.join(data_path, subset, latest_file)
    
    # Load both input data and labels with sampling for large files
    print(f"Loading data from {file_path}...")
    
    # For large files, we'll sample directly without trying to get total row count
    try:
        # Try to get the number of rows to sample efficiently
        with pd.HDFStore(file_path, mode='r') as store:
            try:
                nrows = store.get_storer('data').nrows
                print(f"File contains {nrows} examples")
                
                if nrows > sample_size:
                    print(f"Sampling {sample_size} examples...")
                    # Generate random row indices for sampling
                    rows = np.random.choice(nrows, size=sample_size, replace=False)
                    # Load sampled data
                    input_data = pd.read_hdf(file_path, key='data', where=f"index in {list(rows)}")
                    labels = pd.read_hdf(file_path, key='label', where=f"index in {list(rows)}")
                else:
                    input_data = pd.read_hdf(file_path, key='data')
                    labels = pd.read_hdf(file_path, key='label')
            except:
                # If we can't get the row count, use a different sampling approach
                print("Unable to determine row count, using alternative sampling method")
                # Get a sample of the first rows to determine structure
                input_data = pd.read_hdf(file_path, key='data', start=0, stop=sample_size)
                labels = pd.read_hdf(file_path, key='label', start=0, stop=sample_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try direct loading with limits as a fallback
        print("Attempting direct loading with limits...")
        input_data = pd.read_hdf(file_path, key='data', start=0, stop=sample_size)
        labels = pd.read_hdf(file_path, key='label', start=0, stop=sample_size)
    
    print(f"Successfully loaded {len(input_data)} examples")
    return input_data, labels

def plot_move_heatmap(labels, title='Distribution of Destination Squares'):
    """Plot a heatmap showing the frequency of destination squares."""
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

def plot_legal_moves_analysis(input_data):
    """Analyse the pattern of legal moves in positions."""
    # Count the number of legal moves per position
    legal_move_counts = []
    
    for i in range(len(input_data)):
        # For each square, check the 13th value (index 12) in each 16-value block
        legal_moves = 0
        for j in range(64):
            if input_data.iloc[i, j*16+12] > 0:  # Legal move flag
                legal_moves += 1
        legal_move_counts.append(legal_moves)
    
    # Plot distribution of legal move counts
    plt.figure(figsize=(12, 8))
    sns.histplot(legal_move_counts, bins=30, kde=True)
    plt.title('Distribution of Number of Legal Moves', fontsize=16)
    plt.xlabel('Number of Legal Moves', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_en_pris_analysis(input_data):
    """Analyse the en pris patterns in the data."""
    # Count the number of en pris squares per position (14th value)
    en_pris_counts = []
    
    for i in range(len(input_data)):
        en_pris = 0
        for j in range(64):
            if input_data.iloc[i, j*16+13] > 0:  # En pris flag
                en_pris += 1
        en_pris_counts.append(en_pris)
    
    # Plot distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(en_pris_counts, bins=20, kde=True)
    plt.title('Distribution of En Pris Squares', fontsize=16)
    plt.xlabel('Number of En Pris Squares', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_new_threats_analysis(input_data):
    """Analyse the pattern of new threats in positions (15th value)."""
    new_threat_counts = []
    
    for i in range(len(input_data)):
        threats = 0
        for j in range(64):
            if input_data.iloc[i, j*16+14] > 0:  # New threat flag
                threats += 1
        new_threat_counts.append(threats)
    
    # Plot distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(new_threat_counts, bins=20, kde=True)
    plt.title('Distribution of New Threat Possibilities', fontsize=16)
    plt.xlabel('Number of Potential New Threats', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_new_hanging_analysis(input_data):
    """Analyse the pattern of new hanging pieces in positions (16th value)."""
    new_hanging_counts = []
    
    for i in range(len(input_data)):
        hanging = 0
        for j in range(64):
            if input_data.iloc[i, j*16+15] > 0:  # New hanging flag
                hanging += 1
        new_hanging_counts.append(hanging)
    
    # Plot distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(new_hanging_counts, bins=20, kde=True)
    plt.title('Distribution of New Hanging Piece Possibilities', fontsize=16)
    plt.xlabel('Number of Potential New Hanging Pieces', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def visualise_random_positions(input_data, labels, num_examples=3):
    """Visualise a few random positions and highlight the source and target squares."""
    # Select random examples
    indices = np.random.choice(len(input_data), size=min(num_examples, len(input_data)), replace=False)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(5*len(indices), 5))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get the board position and the move
        position_data = input_data.iloc[idx]
        move_label = labels.iloc[idx]
        
        # Find the destination square
        move_to = np.argmax(move_label)
        
        # Find legal moves in this position
        legal_moves = []
        for sq in range(64):
            if position_data[sq*16+12] > 0:  # This is a legal move
                legal_moves.append(sq)
                
        # Create a chess board from the data
        board = chess.Board(chess.STARTING_FEN)
        board.clear_board()
        
        # Populate the board with pieces
        for square in range(64):
            piece_data = position_data[square*16:square*16+12]  # Extract piece encoding (12 values)
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
        
        # Draw the board
        axes[i].axis('off')
        
        for sq in range(64):
            file_idx = sq % 8
            rank_idx = 7 - (sq // 8)  # Flip to match chess notation
            
            # Draw the square
            square_color = '#FFCE9E' if (file_idx + rank_idx) % 2 == 0 else '#D18B47'
            rect = plt.Rectangle((file_idx, rank_idx), 1, 1, color=square_color)
            axes[i].add_patch(rect)
            
            # Highlight legal moves with a subtle border
            if sq in legal_moves:  # Legal move
                highlight = plt.Rectangle((file_idx, rank_idx), 1, 1, fill=False, edgecolor='blue', linewidth=1)
                axes[i].add_patch(highlight)
            
            # Highlight the destination square with a border
            if sq == move_to:
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
        
        # Title with the move information
        axes[i].set_title(f"Position {idx+1} - Move to {chess.square_name(move_to)}")
    
    plt.tight_layout()
    return fig

def heatmap_of_legal_moves(input_data):
    """Create a heatmap showing the distribution of legal move squares."""
    legal_moves_count = np.zeros((8, 8))
    
    for i in range(len(input_data)):
        for j in range(64):
            if input_data.iloc[i, j*16+12] > 0:  # Legal move flag
                rank = j // 8
                file = j % 8
                legal_moves_count[rank][file] += 1
    
    # Create heatmap
    plt.figure(figsize=(10, 10))
    
    # Customise colours
    cmap = LinearSegmentedColormap.from_list(
        'chessboard', 
        [(0.9, 0.9, 0.8), (0.2, 0.5, 0.7)], 
        N=100
    )
    
    # Plot the heatmap
    ax = sns.heatmap(legal_moves_count, cmap=cmap, annot=True, fmt='.0f', cbar=True)
    
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
    
    plt.title("Heatmap of Legal Destination Squares", fontsize=16)
    plt.tight_layout()
    return plt.gcf()

def generate_visualisations(data_path='models/data/tactics/piece_to', sample_size=1000):
    """Generate all visualisations and save them to a directory."""
    # Create output directory
    output_dir = 'visualisations_piece_to'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data with sampling due to large file size
    train_data, train_labels = load_data(data_path, 'train', sample_size)
    test_data, test_labels = load_data(data_path, 'test', sample_size)
    
    print(f"Analysing {len(train_data)} training examples and {len(test_data)} test examples")
    
    # Generate visualisations
    print("Generating visualisations...")
    
    # 1. Move heatmaps for destination squares
    train_heatmap = plot_move_heatmap(train_labels, "Training Data: Distribution of Destination Squares")
    train_heatmap.savefig(os.path.join(output_dir, 'train_move_to_heatmap.png'), dpi=300)
    
    test_heatmap = plot_move_heatmap(test_labels, "Test Data: Distribution of Destination Squares")
    test_heatmap.savefig(os.path.join(output_dir, 'test_move_to_heatmap.png'), dpi=300)
    
    # 2. Legal moves analysis
    train_legal = plot_legal_moves_analysis(train_data)
    train_legal.savefig(os.path.join(output_dir, 'train_legal_moves.png'), dpi=300)
    
    test_legal = plot_legal_moves_analysis(test_data)
    test_legal.savefig(os.path.join(output_dir, 'test_legal_moves.png'), dpi=300)
    
    # 3. En pris analysis
    train_en_pris = plot_en_pris_analysis(train_data)
    train_en_pris.savefig(os.path.join(output_dir, 'train_en_pris.png'), dpi=300)
    
    test_en_pris = plot_en_pris_analysis(test_data)
    test_en_pris.savefig(os.path.join(output_dir, 'test_en_pris.png'), dpi=300)
    
    # 4. New threats analysis
    train_threats = plot_new_threats_analysis(train_data)
    train_threats.savefig(os.path.join(output_dir, 'train_new_threats.png'), dpi=300)
    
    test_threats = plot_new_threats_analysis(test_data)
    test_threats.savefig(os.path.join(output_dir, 'test_new_threats.png'), dpi=300)
    
    # 5. New hanging analysis
    train_hanging = plot_new_hanging_analysis(train_data)
    train_hanging.savefig(os.path.join(output_dir, 'train_new_hanging.png'), dpi=300)
    
    test_hanging = plot_new_hanging_analysis(test_data)
    test_hanging.savefig(os.path.join(output_dir, 'test_new_hanging.png'), dpi=300)
    
    # 6. Random position visualisations
    train_positions = visualise_random_positions(train_data, train_labels, num_examples=3)
    train_positions.savefig(os.path.join(output_dir, 'train_random_positions.png'), dpi=300)
    
    test_positions = visualise_random_positions(test_data, test_labels, num_examples=3)
    test_positions.savefig(os.path.join(output_dir, 'test_random_positions.png'), dpi=300)
    
    # 7. Heatmap of legal moves
    train_legal_heatmap = heatmap_of_legal_moves(train_data)
    train_legal_heatmap.savefig(os.path.join(output_dir, 'train_legal_moves_heatmap.png'), dpi=300)
    
    test_legal_heatmap = heatmap_of_legal_moves(test_data)
    test_legal_heatmap.savefig(os.path.join(output_dir, 'test_legal_moves_heatmap.png'), dpi=300)
    
    print(f"Visualisations saved to {output_dir}")
    
if __name__ == '__main__':
    # Use a smaller sample size since the files are very large
    generate_visualisations(sample_size=100) 