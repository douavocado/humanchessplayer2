# simulation — offline bot self-play with simulated clocks

Plays the bot against itself locally (no display, no mouse, no Lichess) and
writes PGNs with `[%clk]` tags that `cheat_detection/` can analyse. Replaces
"testing in production": instead of playing hundreds of live games to get
timing data, simulate thousands overnight.

**No real time passes.** Every duration a live game would put on the clock —
think time, mouse gestures, vision latency, ponder waits — is *computed* and
charged to a `SimClock`, never slept. The only wall-clock cost is the
engines' genuine computation (Stockfish + torch), so a 60+0 game simulates in
roughly 1–2 minutes of compute instead of 2+ minutes of real time, and runs
parallelise.

## Quick start

```bash
# simulate 20 games of 60+0
venv/bin/python -m simulation.run --games 20 --tc 60+0 --seed 1 \
    --out simulation/games/selfplay_60plus0.pgn

# analyse them exactly like real bot games
venv/bin/python -m cheat_detection.analyze report \
    --pgn simulation/games/selfplay_60plus0.pgn \
    --player SimBotWhite SimBotBlack \
    --baseline cheat_detection/baselines/bullet_1plus0_2300_2600.json
```

Games are reproducible: game *i* uses `--seed + i`, which seeds both the
engine's move RNG (`make_move(seed=...)`) and every latency draw. A manifest
JSON (results, terminations, per-game move-kind counts) is written beside the
PGN. Outputs under `simulation/games/` and `simulation/calibration/` are
gitignored.

## How a move's clock charge is built

Server-side, your clock runs from the opponent's move registering until your
move registers. The simulator charges the mover:

| component | source |
|---|---|
| detection (capture + FEN + turn + clock OCR, occasional 0.15s confirm re-capture, skipped under 15s) | `latency_model.py`, calibrated |
| think = `max(time_take − MOVE_DELAY, engine compute)` | `time_take` from the real engine; compute resampled from calibration |
| gesture (two mouse legs + settle, drag-vs-click by clock, 3% slip retry) | exact live formulas via `common/move_timing.py` |

Fast paths mirror `clients/mp_original.py` and bypass the engine entirely,
just like live: queued **premoves** fire for ~0.1s when legal; **ponder-dic
hits** and time-scramble moves use the live wait formulas. The manifest's
`move_kinds` shows the mix per game.

Clocks are written as **integer seconds** — the same granularity as the
bot's real Lichess exports; `cheat_detection` derives move times from clock
diffs, so matching granularity matters.

## Calibration (Phase 0)

Machine-specific measurements live in `simulation/calibration/*.json`:

```bash
# vision latency, from saved screenshots (no display needed)
venv/bin/python -m simulation.calibrate detection

# engine compute time, replaying real bot positions headless
venv/bin/python -m simulation.calibrate compute \
    --pgn cheat_detection/runs/bot_JXu2019_60plus0.pgn --player JXu2019
```

The simulator runs without these (documented fallbacks), but calibrated
distributions are what make the timing tail realistic. Live sessions now log
at `PERF` level by default (`main.py --log-level`), so real games accumulate
ground truth (`[PERF] REALISED MOVE TIME`, scan and mouse breakdowns) for
refining the model.

## Module map

- `clock.py` — `SimClock`: remaining time per side, increment, flag detection.
- `latency_model.py` — samples detection / compute / gesture durations.
- `client_model.py` — `SimClient`: the live client's decision branches
  (premove fire → ponder fast paths → full engine path → resign).
- `game_runner.py` — one game loop: two engines, SimClock, history feeding
  `Engine.update_info`.
- `pgn_writer.py` — PGN with integer-second `[%clk]`.
- `calibrate.py` / `run.py` — CLIs.

The simulator drives the **monolithic `engine.py`** (what the live client
uses), not the unwired `engine_components/` refactor.

## Status / roadmap

- **Phase 0 (done)**: calibration CLIs; measured on this machine
  (scan ≈ 13ms + ~8ms capture; compute mean 0.76s, p90 1.42s, p99 3.75s).
- **Phase 1 (done)**: core simulator, this package.
- **Phase 2 (partial)**: fast paths, premove semantics, compute-overrun and
  slip retries are in. Not yet modelled: berserk, opponent-blunder "startle"
  double-take interactions with fast paths, hover/wander cursor drift,
  sub-second server lag-compensation.
- **Phase 3 (next)**: validation loop — run `cheat_detection` comparing
  simulated games vs the real fetched 60+0 bot games (Welch t-test mode);
  every significant timing feature points at the latency component to fix.
  Only after sim-vs-real converges should simulated games substitute for
  production data.

## Known caveats

- Self-play opponent: the bot's timing reacts to its opponent (reflective
  pacing, blunder reactions); a bot opponent is self-consistent but not a
  human. Check feature drift in Phase 3 rather than assuming.
- Engines are reused across games in a batch; per-game client state
  (ponder dic, premove queue, cursor) is reset, but any residual engine
  internals carry over — same as consecutive live games in one session.
- Both sides share one weights set; diversity comes from the opening book
  and per-move RNG. Audit for duplicate games when generating large batches.
