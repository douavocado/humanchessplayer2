---
name: verify-offline
description: Run the repo's offline verification checks — client unit tests, the engine parity harness, and the auto-calibration readback test against saved screenshots. Use after changing engine logic, client logic, vision code, or calibration, or when asked to verify changes without a live game session.
---

Run these offline checks from the repo root and report results together. The bot
itself needs a live game page and an X11 display, so these are the only
automated checks available.

1. **Client unit tests** — must be green, no exceptions
   ```
   venv/bin/python -m unittest discover -s testing/client
   ```
   Every test here is a regression from a real logged game. A failure is a real
   break, never a baseline.

2. **Engine parity harness** (real Stockfish + real NN weights, no mocks)
   ```
   venv/bin/python -m unittest discover -s testing/engine_parity
   ```
   The golden-master gate for `engine.py` / `engine_components/`. Run it after
   touching either. Slow — it runs real searches. Note the harness deliberately
   tolerates jitter on eval-derived floats and does not assert `premove`; see
   its module docstring before treating a diff as a regression.

   (`testing/engine/`, a mocked-Stockfish suite, was deleted on 2026-07-25
   after months of being uniformly red. Don't go looking for it.)

3. **Calibration readback test** (vision pipeline, production-identical functions)
   ```
   venv/bin/python -m auto_calibration.calibration_readback_test --screenshots auto_calibration/offline_screenshots/desktop --profile desktop
   ```
   Report the detection rates (FEN, clock, turn, result, false-start) and
   per-call timings. Compare against the numbers from before the change when
   available; flag any detection-rate drop or large timing regression.

Scope the run to what changed — client-only edits don't need the parity
harness, engine-only edits don't need the readback test. Summarize in one short
report: what passed, what failed, and whether each failure is pre-existing or new.
