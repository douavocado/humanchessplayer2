# HumanChessPlayer

A chess bot that plays on Lichess and chess.com by looking at the screen and
moving the mouse, with human-like move choice and timing.

## Overview

The bot reads the board from screen captures (OpenCV template matching), picks
moves by blending neural networks trained on human games with Stockfish
analysis, and plays them by moving the physical cursor. There is no site API
and no browser automation — from the site's perspective it is a person at a
computer.

"Human-like" is the whole point, and it is treated as a measurable property
rather than a vibe: the `cheat_detection/` package scores the bot against a
baseline of real 2300+ bullet games and flags any feature where it looks
un-human — including being *too consistent*, which is its own tell.

## Layout

| Directory | What lives there |
|---|---|
| `engine.py`, `engine_components/` | Move selection. `engine.py` is the composition root; the logic sits in `engine_components/` as plain functions taking the `Engine` as their first argument. |
| `clients/mp_original.py` | The client: game loop, mouse automation, scan-reliability guards. |
| `sites/` | *How a site behaves* — how a new game, a game end, and a result are recognised; resign and lobby flows. Lichess and chess.com. |
| `auto_calibration/` | *Where things are on this screen* — fits board and UI coordinates from screenshots into a reusable profile. |
| `chessimage/` | Screen capture, FEN extraction, clock OCR, rating OCR. |
| `models/` | The neural networks and their weights. |
| `common/` | Shared constants, board analysis, timing formulas, utilities. |
| `simulation/` | Offline bot-vs-bot self-play with simulated clocks — no display or mouse needed. |
| `cheat_detection/` | Human-likeness analysis (CLI + Tkinter GUI) against a real-game baseline. |
| `testing/` | Client regression tests and the engine parity harness. |

`sites/` versus `auto_calibration/` is the axis that matters: behaviour over
time versus pixel geometry. A calibration profile binds the two by recording
which site it was fitted against. See `docs/site-abstraction.md`.

## Requirements

- Python 3.12 (the repo expects a virtualenv at `venv/`)
- Stockfish, plus the Python dependencies in `requirements.txt`
- Tesseract for clock and rating OCR (`sudo apt install tesseract-ocr`; on
  Windows, the UB-Mannheim build, with `pytesseract.tesseract_cmd` pointed at it)
- A desktop session — the bot drives a real cursor. Linux/X11 and Windows are
  both supported; everything that differs between them lives in
  `common/platform_compat.py`.

On Windows, run with `PYTHONUTF8=1` (or `python -X utf8`): several modules print
non-ASCII status characters and read PGN corpora that contain them, and the
default cp1252 console encoding raises `UnicodeEncodeError` on both.

## Installation

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

Then supply the pieces that are deliberately not in git:

- **Stockfish binaries** in `Engines/` — paths set in `common/constants.py`
  (`PATH_TO_STOCKFISH`, `PATH_TO_PONDER_STOCKFISH`)
- **Model weights** (9 `.pth` files) in `models/model_weights/`
- **Opening books** at `assets/data/Opening_books/bullet.bin` and
  `repertoire.bin` (build the latter with
  `scripts/utilities/build_repertoire.py`)
- **A calibration profile** — fit one for your screen:

  ```bash
  venv/bin/python -m auto_calibration.offline_fitter \
      --dir <screenshots/> --profile desktop --site lichess --extract-all --visualise
  ```

  Select it at runtime with `--calibration-profile desktop`, or the
  `HCP_CALIBRATION_PROFILE` environment variable. Without one the bot falls
  back to hardcoded 1920x1080 coordinates.

## Usage

```bash
venv/bin/python main.py                      # 5 bullet games at 60+0
venv/bin/python main.py -t 180 -i 2 -g 10    # ten 3+2 games
venv/bin/python main.py -a -b                # arena, always berserk
venv/bin/python main.py -d 5 -q 2.5 -m 2     # override strength and pacing
```

Run `venv/bin/python main.py --help` for the full flag list. The ones worth
knowing beyond the basics:

| Flag | Effect |
|---|---|
| `--calibration-profile NAME` / `--calibration-file PATH` | Which screen profile to use |
| `--debug` | Dry run: only test new-game detection, with visualisations |
| `--offline` | Replay saved screenshots instead of capturing live |
| `--log-level` | Defaults to `PERF`, so live games record timing data the simulator calibrates against |

## Development

The bot cannot be exercised end-to-end without a display and a live game, so
most verification is offline:

```bash
venv/bin/python -m unittest discover -s testing/client         # must stay green
venv/bin/python -m unittest discover -s testing/engine_parity  # real-Stockfish golden master
venv/bin/python -m auto_calibration.calibration_readback_test \
    --screenshots auto_calibration/offline_screenshots/desktop --profile desktop
```

Every client test is a regression from a real logged game; each file's
docstring names the incident it came from. For behavioural work, `simulation/`
plays the bot against itself with simulated clocks and writes PGNs that feed
straight into `cheat_detection/`.

Tuning knobs (`DIFFICULTY`, `QUICKNESS`, `MOUSE_QUICKNESS`, `RESOLUTION_SCALE`)
live in `common/constants.py`. `CLAUDE.md` documents the reasoning behind the
timing and human-likeness parameters in far more depth than this file.

## Disclaimer

For educational and research purposes. Using it on a live site will breach
that site's terms of service.
