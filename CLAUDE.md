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

## Move-time pacing (`engine.py:_get_time_taken`)

How long the bot "thinks" per move is decided in `Engine._get_time_taken`. A "complicated position" — one worth spending more time on — is keyed off **sharpness** (`self.sharpness`, computed in `_compute_sharpness`): the win-probability spread across the engine's top candidates from a narrow, slightly-deeper scan (multipv 5, depth 12), using the same logistic and definition as the `cheat_detection/` human-likeness analyser. Sharpness ≥ 0.25 is a "critical" position. This replaced the older structural trigger (Lucas `activity`/`eff_mob`); those Lucas metrics are still computed and used elsewhere (mood, resign logic), just no longer for time-scaling. Sharpness maps onto the same `((x+12)/25)**0.4` envelope, so the *magnitude* of the slow-down is unchanged — only the trigger.

An **intuition gate** stops the bot over-thinking (and flagging on time): in a sharp position it applies the slow-down only ~35% of the time and otherwise snaps the move quickly (gate probability 0.65, `base_time *= 0.5`), mirroring that humans trust intuition on most critical moves. Very flat positions (sharpness < 0.10) get a `*0.7` "little at stake" cut. All three log to `engine.log` (sharpness value, multiplier, gate outcome).

## Architecture pointers

- `engine.py` — move selection core; modular logic lives in `engine_components/` (decision_logic, human_move_logic, premover, ponderer, mood_manager, state_manager).
- `clients/mp_original.py` — the active Lichess client (mouse automation, game loop, scan-reliability guards); other clients in `clients/` are older variants.
- `common/utils.py` — `scraped_fen_sanity_issues()` / `InvalidPositionError`: reject structurally impossible scraped positions (e.g. king hidden by a capture animation); the engine refuses to analyse them (Stockfish segfaults) and the client discards the scan.
- `chessimage/image_scrape_utils.py` — screen capture, FEN extraction, clock OCR.
- `auto_calibration/` — fits board/UI coordinates from screenshots; profiles stored in `auto_calibration/calibrations/`.
- `common/move_timing.py` — the timing formulas (mouse-leg duration, settle sleeps, drag-vs-click probability, ponder/scramble waits, MOVE_DELAY constants) shared between the live client and the simulator. Tune durations here, not inline in the client, or the two drift apart.
- `simulation/` — offline bot self-play with **simulated** clocks (no display/mouse; wall time ≈ engine compute only). `python -m simulation.run --games N --tc 60+0` writes PGNs with integer-second `[%clk]` that feed straight into cheat_detection (`--player SimBotWhite SimBotBlack`). Per-move clock charge = detection + max(time_take − MOVE_DELAY, compute) + mouse gesture, with the live client's fast paths (premove fire, ponder-dic hits, scrambles) reproduced in `client_model.py`. Machine-specific latency distributions come from `python -m simulation.calibrate {detection,compute}` (JSONs in `simulation/calibration/`, gitignored). Drives monolithic `engine.py`, not the unwired `engine_components/` refactor. See its README for the charge model and Phase 2/3 roadmap (sim-vs-real Welch validation). Live sessions default to `--log-level PERF` (`main.py`) so real games accumulate `[PERF]` realised-move/scan/mouse timings for refining the model.
- `cheat_detection/` — offline human-likeness diagnostic (CLI `analyze.py`, Tkinter `gui.py` with per-feature distribution drill-down; see its README). Analysis depth defaults to 10 (benchmarked ~24× faster than the original 18). Bot analysis and the baseline JSON must use the same depth — mismatched depths skew every engine-derived feature; baselines store per-game feature values (`values`) for the GUI histograms. Baselines/runs/cache are gitignored. Two flagging modes (`--test` / GUI "Test" dropdown): effect-size `|z| >= 2` vs human spread (default, sample-size independent) or a Welch two-sample t-test at `--alpha` (sensitivity grows with game count); both statistics are always shown. Fetches should pin one exact clock with `--tc 60+0` (GUI "exact clock" field) — mixing e.g. 30+0 with 60+0 muddies every timing feature; `fetch_corpus --tc` does the same for the human corpus, and the shipped baseline is pure 60+0.
