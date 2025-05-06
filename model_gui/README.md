# Chess Move Scorer GUI

A graphical user interface for visualising the outputs of the MoveScorer model, which predicts human-like chess moves.

## Features

- Input chess positions using FEN notation
- Visualise the chess board
- Display top human-like moves with their probabilities
- Support for both white and black to move positions

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the model weights are in the correct location:
   - The weights should be in `models/model_weights/` folder.
   - If your weights are in a different location, update the paths in `app.py`

## Usage

1. Run the application:

```bash
python model_gui/run.py
```

2. Enter a valid FEN string in the input box
3. Click "Analyse Position" to visualise the board and see the move probabilities
4. The moves are displayed in SAN (Standard Algebraic Notation) with their associated probabilities

## Examples

Try these FEN positions:

- Starting position: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`
- Middle game example: `r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4`
- Complex position: `3q1rk1/pr2bpp1/1p2pn1B/n3N3/8/2NP4/PPQ1PP1P/R3K1R1 b Q - 0 17`

## Note

For positions where black is to move, the board will be mirrored internally for analysis as the model was trained on white-to-move positions. This is handled automatically by the application. 