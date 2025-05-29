import os
import csv
import random
import chess
import chess.pgn
import chess.engine
from pathlib import Path
import pandas as pd
import argparse

# Define file paths
CSV_PATH = "models/data/defensive_tactics/lichess_db_puzzle.csv"
ENGINE_PATH = "Engines/stockfish17-ubuntu"
OUTPUT_PGN = "assets/data/train_PGNs/defensive_tactics/synthetic.pgn"

# Create directories if they don't exist
os.makedirs(os.path.dirname(OUTPUT_PGN), exist_ok=True)

def is_mate_in_n_theme(themes_str, n_range=range(1, 6)):
    """Check if the puzzle contains a mate in n theme where n is in the specified range."""
    themes = themes_str.split()
    for n in n_range:
        if f"mateIn{n}" in themes:
            return True
    return False

def get_viable_moves_with_pvs(engine, board, depth=8, multipv=5):
    """
    Analyze the position and return all viable moves according to criteria:
    - If best povscore is positive, return all moves with positive povscore
    - If best povscore is negative, return only the best move if the best move is >= -250cp
    
    For each viable move, return the full principal variation.
    """
    limit = chess.engine.Limit(depth=depth)
    
    # Analyze with multipv=5
    analysis = engine.analyse(
        board, 
        limit=limit, 
        multipv=multipv,
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV
    )
    
    # Extract all moves and their scores
    moves_with_scores_and_pvs = []
    for pv_info in analysis:
        moves = pv_info["pv"]
        if moves:
            first_move = moves[0]
            score = pv_info["score"].pov(board.turn)
            moves_with_scores_and_pvs.append((first_move, score, moves))
    
    # Check if the best score is positive or negative
    best_score = moves_with_scores_and_pvs[0][1]
    
    viable_moves = []
    if best_score.is_mate():
        # If best move is a mate, include only that move
        if best_score.mate() > 0:
            viable_moves.append(moves_with_scores_and_pvs[0])
    elif not best_score.is_mate() and best_score.cp >= 0:
        # If best score is positive centipawns, include all moves with positive scores
        for move, score, pv in moves_with_scores_and_pvs:
            if not score.is_mate() and score.cp >= 0:
                viable_moves.append((move, score, pv))
    else:
        # If best score is negative, include only the best move if the best move is >= - 250cp
        if moves_with_scores_and_pvs[0][1].cp >= -250:
            viable_moves.append(moves_with_scores_and_pvs[0])
    
    return viable_moves

def create_games_from_position(engine, fen, game_id):
    """Create game entries for each viable move from the given position."""
    games = []
    
    # Create a board from the FEN
    board = chess.Board(fen)
    
    # Get viable moves for this position with their PVs
    viable_moves = get_viable_moves_with_pvs(engine, board)
    
    for i, (move, score, pv) in enumerate(viable_moves):
        # Create a new game for each viable move
        game = chess.pgn.Game()
        
        # Set headers
        game.headers["Event"] = f"Defensive Tactics {game_id}"
        game.headers["Site"] = "Synthetic"
        game.headers["Date"] = "????.??.??"
        game.headers["Round"] = str(i+1)
        game.headers["White"] = "?" if board.turn == chess.WHITE else "Stockfish"
        game.headers["Black"] = "Stockfish" if board.turn == chess.WHITE else "?"
        game.headers["Result"] = "*"
        game.headers["FEN"] = fen
        game.headers["SetUp"] = "1"
        
        # Determine how many plies to include (0, 2, or 4 additional plies after the first move)
        additional_plies = random.choice([0, 2, 4])
        
        # Calculate total moves to use from PV (first move + additional plies)
        total_moves = min(1 + additional_plies, len(pv))
        
        # Add PV moves to the game
        node = game
        current_board = board.copy()
        
        for j, pv_move in enumerate(pv[:total_moves]):
            node = node.add_variation(pv_move)
            current_board.push(pv_move)
            
            # Check if the game is over after this move
            if current_board.is_game_over():
                break
        
        games.append(game)
    
    return games

def main():
    # Check if files exist
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    
    if not os.path.exists(ENGINE_PATH):
        print(f"Error: Stockfish engine not found at {ENGINE_PATH}")
        return
    
    # Start the engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        print(f"Engine loaded: {ENGINE_PATH}")
    except Exception as e:
        print(f"Error loading the engine: {e}")
        return
    
    try:
        # Read the CSV file using pandas for better handling of large files
        print(f"Reading CSV file: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        print(f"Total rows in CSV: {len(df)}")
        
        # Filter rows with mateIn1-5 themes
        mate_puzzles = df[df['Themes'].apply(lambda x: is_mate_in_n_theme(x))]
        print(f"Filtered to {len(mate_puzzles)} puzzles with mate in 1-5")
        
        # Open output PGN file
        with open(OUTPUT_PGN, "w") as pgn_file:
            # Process each puzzle
            # Use enumerate to track position in the DataFrame
            for i, (idx, row) in enumerate(mate_puzzles.iterrows()):
                if i % 1000 == 0:
                    print(f"Processing puzzle {i}/{len(mate_puzzles)}")
                
                # Get FEN
                fen = row['FEN']
                
                # Create games from this position
                games = create_games_from_position(engine, fen, idx)
                
                # Write games to PGN file
                for game in games:
                    print(game, file=pgn_file, end="\n\n")
        
        print(f"Successfully created defensive tactics in {OUTPUT_PGN}")
    
    except Exception as e:
        print(f"Error processing the CSV: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Make sure to quit the engine
        if 'engine' in locals():
            engine.quit()

if __name__ == "__main__":
    main()
