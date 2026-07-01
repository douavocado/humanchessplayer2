---
name: verify-offline
description: Run the repo's offline verification checks — the engine unittest suite and the auto-calibration readback test against saved screenshots. Use after changing engine logic, vision code, or calibration, or when asked to verify changes without a live Lichess session.
---

Run both offline checks from the repo root and report results together. The bot itself needs a live Lichess page and X11 display, so these are the only automated checks available.

1. **Engine unit tests**
   ```
   venv/bin/python -m unittest discover -s testing/engine
   ```
   Known issue: as of July 2026 all 19 tests error because they patch `engine.STOCKFISH`, which was removed in a refactor. If the failure output is exactly this pattern, report it as the pre-existing baseline, not a regression. If the failures differ from that pattern, investigate — the change may have broken something.

2. **Calibration readback test** (vision pipeline, production-identical functions)
   ```
   venv/bin/python -m auto_calibration.calibration_readback_test --screenshots auto_calibration/offline_screenshots/desktop --profile desktop
   ```
   Report the detection rates (FEN, clock, turn, result, false-start) and per-call timings. Compare against the numbers from before the change when available; flag any detection-rate drop or large timing regression.

Summarize both results in one short report: what passed, what failed, and whether failures are pre-existing or new.
