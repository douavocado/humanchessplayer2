import os
import sys
import torch
import random
import torch.optim as optim
import chess
from models.models import MoveScorer
import ast
from alter_move_prob_train.alter_move_prob_nn import AlterMoveProbNN
from common.board_information import phase_of_game

# Initialise the AlterMoveProbNN model
model = AlterMoveProbNN()

# Print the model parameters
print("Model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")

# Load a random game from a random PGN file in train_PGNs
def load_random_game_position():
    # Get list of PGN files in train_PGNs directory
    pgn_dir = "train_PGNs"
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    
    if not pgn_files:
        raise FileNotFoundError("No PGN files found in train_PGNs directory")
    
    # Select a random PGN file
    random_pgn_file = random.choice(pgn_files)
    pgn_path = os.path.join(pgn_dir, random_pgn_file)
    
    # Open the PGN file and read a random game
    with open(pgn_path) as pgn:
        games = []
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    
    if not games:
        raise ValueError(f"No games found in {random_pgn_file}")
    
    # Select a random game
    random_game = random.choice(games)
    
    # Play through to a random position
    board = random_game.board()
    moves = list(random_game.mainline_moves())
    
    # Choose a random move index (not the last move)
    if len(moves) <= 1:
        return board, None
    
    move_index = int(2*random.randint(0, int((len(moves) - 2)/2)))
    
    # Play until that position
    for i in range(move_index):
        board.push(moves[i])
    
    # Return the board and the true next move
    true_move = moves[move_index]
    
    return board, true_move

# Set a random seed for reproducibility
random.seed(42)

# Load a random position and the true next move
board, true_move = load_random_game_position()

print("\nExample board position:")
print(board)
print(f"Game phase: {phase_of_game(board)}")
print(f"True move: {board.san(true_move)}")

# generate move dic using middlegame scorer
# Create a simple move dictionary for testing
# We'll use a midgame move scorer to generate realistic probabilities


# Load the midgame move scorer
midgame_scorer = MoveScorer("models/model_weights/piece_selector_midgame_weights.pth", "models/model_weights/piece_to_midgame_weights.pth")

# Get move dictionary from the scorer
_, move_dic = midgame_scorer.get_move_dic(board, san=False, top=100)

# Convert to regular dictionary if it's not already
if not isinstance(move_dic, dict):
    move_dic = {k: float(v) for k, v in move_dic.items()}


# No previous boards for this example
prev_board = None
prev_prev_board = None

# Run the model on the example
altered_move_dic, log = model(move_dic, board, prev_board, prev_prev_board)

# Print the results
print("\nOriginal move dictionary:")
for move, prob in move_dic.items():
    print(f"{board.san(chess.Move.from_uci(move))}: {prob:.4f}")

print("\nAltered move dictionary:")
for move, prob in altered_move_dic.items():
    if isinstance(prob, torch.Tensor):
        prob_value = prob.item()
    else:
        prob_value = float(prob)
    print(f"{board.san(chess.Move.from_uci(move))}: {prob_value:.4f}")

print("\nLog of alterations:")
print(log)

# Create an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Zero the gradients
optimizer.zero_grad()

# Calculate loss (using a simple example)
# In a real training scenario, this would be based on the true move probability
true_move = random.choice(list(move_dic.keys()))
if true_move in altered_move_dic:
    true_move_prob = altered_move_dic[true_move]
    # No need to create a new tensor, as the model should already return tensors with grad tracking
    # Simply ensure it's connected to the computation graph
    if not isinstance(true_move_prob, torch.Tensor):
        true_move_prob = torch.tensor(true_move_prob, dtype=torch.float)
    # We want negative log likelihood loss
    loss = -torch.log(true_move_prob)
else:
    # If the true move is not in the altered move dictionary, assign a high loss
    loss = torch.tensor(10.0, dtype=torch.float)

print(f"\nLoss for this example: {loss.item():.4f}")

# Backward pass
loss.backward()

# Print gradients before update
print("\nGradients before update:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.item():.6f}")

# Update parameters
optimizer.step()

# Print parameters after update
print("\nParameters after optimisation step:")
for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")

