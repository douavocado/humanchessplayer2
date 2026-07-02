# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A bot that plays chess on Lichess with human-like behaviour: neural-network move selection blended with Stockfish analysis, screen-capture board vision (OpenCV + template matching), and mouse automation. Entry point is `main.py`, which delegates to `clients/mp_original.py`.

## Python environment

Use the repo virtualenv: `venv/bin/python` (Python 3.12). There is no bare `python` on PATH, and the system `python3` lacks the dependencies.

## Lint

`venv/bin/ruff check <changed files>` (config in `ruff.toml`; stylistic rules that clash with existing code idioms are disabled). The repo has ~170 pre-existing violations (mostly unused imports/variables), so lint only the files you touched, not the whole tree.

## Verifying changes

The bot cannot be run end-to-end here — it needs an X11 display and a live Lichess page. Never assume a live run is possible; use the offline checks below and leave live verification to the user.

- **Engine unit tests**: `venv/bin/python -m unittest discover -s testing/engine` (run from repo root). ⚠️ Currently all 19 tests error: they patch `engine.STOCKFISH`, a module-level attribute removed in a refactor. Don't treat these failures as caused by your change.
- **Vision / calibration changes**: run the readback test against saved screenshots:
  `venv/bin/python -m auto_calibration.calibration_readback_test --screenshots auto_calibration/offline_screenshots/desktop --profile desktop`
  It reports FEN extraction, clock OCR, turn/result detection, and per-call timing using production-identical functions.
- **Calibration fitting**: `venv/bin/python -m auto_calibration.offline_fitter --dir <screenshots/> --profile <name> --extract-all --visualise` writes `auto_calibration/calibrations/<name>.json`.
- `main.py --debug` (game-detection dry-run) and `--offline` (replay saved screenshots instead of live capture) exist for when a display is available.

## Runtime prerequisites (not in git)

- Stockfish binaries in `Engines/` — paths configured in `common/constants.py` (`PATH_TO_STOCKFISH`, `PATH_TO_PONDER_STOCKFISH`).
- Model weights (`*.pth`) in `models/model_weights/`.
- Opening book at `assets/data/Opening_books/bullet.bin`.
- Calibration profile: env vars `HCP_CALIBRATION_FILE` (explicit JSON path) or `HCP_CALIBRATION_PROFILE` (resolves to `auto_calibration/calibrations/<profile>.json`); falls back to hardcoded 1920x1080 coordinates if unset.

## Key tuning knobs

Behaviour is tuned via `common/constants.py`: `DIFFICULTY`, `QUICKNESS` (move-time pacing), `MOUSE_QUICKNESS` (cursor speed), `RESOLUTION_SCALE` (1.0 for 1080p, 2.0 for 4K). `main.py` flags override some of these per run.

Client-side scan reliability lives in `clients/mp_original.py`: after making a move, `AWAITING_FRESH_SCAN` blocks further moves until a scan is adopted (prevents duplicate/out-of-turn clicks off stale vision), and board scans that fail move-linking or turn detection must survive a confirmation re-capture before being adopted — skipped when our clock is under `RESYNC_CONFIRM_MIN_TIME` seconds (deliberate: humans misread boards under time pressure).

## Architecture pointers

- `engine.py` — move selection core; modular logic lives in `engine_components/` (decision_logic, human_move_logic, premover, ponderer, mood_manager, state_manager).
- `clients/mp_original.py` — the active Lichess client (mouse automation, game loop, scan-reliability guards); other clients in `clients/` are older variants.
- `common/utils.py` — `scraped_fen_sanity_issues()` / `InvalidPositionError`: reject structurally impossible scraped positions (e.g. king hidden by a capture animation); the engine refuses to analyse them (Stockfish segfaults) and the client discards the scan.
- `chessimage/image_scrape_utils.py` — screen capture, FEN extraction, clock OCR.
- `auto_calibration/` — fits board/UI coordinates from screenshots; profiles stored in `auto_calibration/calibrations/`.
