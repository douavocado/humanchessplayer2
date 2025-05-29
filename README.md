# HumanChessPlayer

A sophisticated chess bot that emulates human-like play on online chess platforms.

## Overview

HumanChessPlayer is an AI system designed to play chess online while simulating human-like behaviour. It combines neural network models trained on human gameplay data with chess engine analysis to produce moves that appear natural and human-like rather than purely engine-driven.

The system controls the mouse cursor to interact with web-based chess interfaces (primarily Lichess), making moves with realistic timing, occasional mouse slips, and varying "moods" that affect play style.

## Features

- **Human-like Move Selection**: Uses neural networks trained on human gameplay data to select moves that mimic human play patterns
- **Realistic Mouse Movement**: Simulates natural mouse movements with appropriate timing and occasional inaccuracies
- **Variable Playing Strength**: Configurable difficulty level to match desired ELO rating
- **Adaptive Time Management**: Adjusts thinking time based on game situation and remaining clock time
- **Multiple Game Support**: Can play multiple games and participate in arena tournaments
- **Configurable "Moods"**: Changes playing style based on game situation (confident, cocky, cautious, tilted, etc.)
- **Position Analysis**: Uses Stockfish for position evaluation while applying human-like filters to the engine output

## Requirements

- Python 3.x
- PyTorch
- python-chess
- PyAutoGUI
- OpenCV
- Stockfish chess engine

A complete list of dependencies is available in the `requirements.txt` file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/humanchessplayer.git
   cd humanchessplayer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and install Stockfish:
   - Download the latest version from [Stockfish website](https://stockfishchess.org/download/)
   - Extract to the Engines directory
   - Make sure the path in `common/constants.py` points to your Stockfish executable

## Usage

### Basic Usage

```bash
python main.py
```

### Command Line Arguments

- `-t, --time`: Time control in seconds (default: 60)
- `-i, --increment`: Time control increment in seconds (default: 0)
- `-g, --games`: Number of games to play (default: 5, only used when not in tournament mode)
- `-a, --arena`: Tournament arena mode
- `-b, --berserk`: Always berserk in tournament arena mode
- `-d, --difficulty`: Engine difficulty level (overrides default from constants)
- `-q, --quickness`: Engine quickness (overrides default from constants)
- `-m, --mouse-quickness`: Mouse quickness (overrides default from constants)

Example for playing in an arena tournament with berserk mode:
```bash
python main.py -a -b
```

Example for setting specific difficulty and quickness:
```bash
python main.py -d 5 -q 2.5 -m 2
```

## Components

- **Engine**: Core logic for move selection and evaluation
- **Clients**: Interface with chess websites (primarily Lichess)
- **Models**: Neural network models for human-like move selection
- **Common**: Shared utilities and constants
- **Chessimage**: Screen capture and image processing for board state recognition

## Customisation

The system's behaviour can be customised by modifying values in the `common/constants.py` file:

- `DIFFICULTY`: Overall playing strength (higher values = stronger play)
- `QUICKNESS`: Speed of move calculation (higher values = slower moves)
- `MOUSE_QUICKNESS`: Speed of mouse movement (higher values = slower mouse)

## Disclaimer

This project is for educational and research purposes only. Please use responsibly and in accordance with the terms of service of any chess platforms.

## License

[Your license information here] 