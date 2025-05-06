import chess
import chess.svg
import os
import sys
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_move_arrows(board, move_probs, max_arrows=5):
    """
    Generate arrow data for chess.svg.board() from move probabilities
    
    Args:
        board: A chess.Board object
        move_probs: Dictionary of moves and their probabilities
        max_arrows: Maximum number of arrows to display
    
    Returns:
        List of chess.svg.Arrow objects for chess.svg.board()
    """
    # Sort moves by probability (highest first)
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:max_arrows]
    
    # Skip if no moves
    if not sorted_moves:
        return []
    
    # Get the maximum probability for normalization
    max_prob = max(sorted_moves[0][1], 0.5)
    
    # Create a colourmap - 'hot_r' provides red->yellow->white
    # Other good choices: 'viridis', 'plasma', 'inferno'
    colormap = cm.get_cmap('plasma')
    
    # Create arrows with colours based on probability
    arrows = []
    for move_san, prob in sorted_moves:
        try:
            # Parse the move from SAN notation
            move = board.parse_san(move_san)
            from_square = move.from_square
            to_square = move.to_square
            
            # Normalize probability relative to the max probability
            # Use a non-linear scaling to enhance the visual distinction
            norm_prob = (prob / max_prob) ** 0.7  # Power < 1 enhances lower values
            
            # Get colour from colourmap
            rgba = colormap(norm_prob)
            
            # Convert to hex colour string
            hex_color = mcolors.rgb2hex(rgba[:3])  # Exclude alpha
            
            # Add the arrow using the Arrow class
            arrows.append(chess.svg.Arrow(from_square, to_square, color=hex_color))
        except ValueError:
            continue
    
    return arrows

def render_board_with_moves(board, move_probs, size=400, max_arrows=5):
    """
    Render a chess board with move arrows
    
    Args:
        board: A chess.Board object
        move_probs: Dictionary of moves and their probabilities
        size: Size of the SVG board
        max_arrows: Maximum number of arrows to display
    
    Returns:
        SVG string of the chess board with arrows
    """
    # Get arrows for the most probable moves
    arrows = get_move_arrows(board, move_probs, max_arrows)
    
    # Generate the SVG
    svg = chess.svg.board(
        board=board,
        size=size,
        arrows=arrows,
        coordinates=True
    )
    
    return svg

if __name__ == "__main__":
    # Example usage
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    move_scorer = MoveScorer(
        move_from_weights_path="../models/model_weights/piece_selector_midgame_weights.pth",
        move_to_weights_path="../models/model_weights/piece_to_weights_midgame.pth"
    )
    
    _, move_probs = move_scorer.get_move_dic(board, san=True, top=10)
    svg = render_board_with_moves(board, move_probs)
    
    # Save to file for testing
    with open("test_board.svg", "w") as f:
        f.write(svg)
