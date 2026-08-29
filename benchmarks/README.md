# benchmarks/

Measures what the bot's per-move work actually costs **on this machine**, so
you can tell whether a device is fast enough to play the way it is tuned to
play — and, when it is not, whether that is something a pacing knob can fix.

```bash
venv/bin/python -m benchmarks.run all --label thinkpad
venv/bin/python -m benchmarks.run all --label desktop --mouse-live
venv/bin/python -m benchmarks.run compare benchmarks/results/*.json
```

Results are written to `benchmarks/results/*.json`. They are **not**
gitignored, deliberately: collecting several devices side by side is the
whole point, and committing them is the simplest way to get another
machine's numbers next to yours. Nothing reads them at runtime — unlike
`simulation/calibration/*.json`, which the simulator *does* read and which
stays gitignored and machine-local.

Everything here runs on a fresh clone: the position corpus is hardcoded and
the board images are synthesised from the piece templates tracked in
`chessimage/`. No calibration profile, no PGN corpus, no saved screenshots.
That is the difference from `python -m simulation.calibrate compute`, which
replays a real bot PGN to feed the simulator's latency model — excellent for
that job, useless for comparing two machines that share no PGN.

## The one number that matters: the compute floor

`make_move` deliberately spends whatever is left of the think budget on
pondering (`engine.py`: `time_left = time_take - time_spent`, then
`ponder(time_left / 1.15)`). So `make_move_total` tracks `time_take` by
construction, and comparing the two tells you about the design, not the
device.

The device-limited quantity is `make_move_total` minus that elective ponder:
**the fastest a move can be produced here at all**. It matters because
`clients/mp_original.py` sleeps only the *remainder* of the intended think
time and skips the sleep entirely when compute already overran it:

```python
intended_break = output_dic["time_take"]
if end - start - intended_break < -1 * MOVE_DELAY:
    time.sleep(intended_break - (end - start) - MOVE_DELAY)
```

So an intended move time below the floor is silently unachievable. If the
floor sits above the 1.0s `instant_move_secs` threshold that
`cheat_detection` scores instant moves against, this device cannot produce
instant moves **at any `QUICKNESS`** — and instant-move rate is a tracked
human-likeness feature that rises with rating. That is a compute problem
(ponder width, `SHARPNESS_SCAN_DEPTH`, re-eval budget), not a pacing one.

`QUICKNESS` owns the opposite case: two devices with comparable floors that
differ only in how long they *choose* to think.

## Suites

### compute

Per-move engine and neural-network cost. **Stages are nested, not a
partition** — each is timed by calling it directly, so the column does not
sum to `make_move_total`:

```
make_move_total              end-to-end, the window the client brackets
  analytics_total            runs inside update_info in production
    sf_scan_fullwidth          multipv full-width, Limit(depth=10, time=0.02)
    lucas_analytics            pure numpy/python over that scan
    sf_sharpness               multipv 5, Limit(depth=12), no time cap
    set_mood                   a further short Stockfish probe
  nn_probabilities           the production NN probability path
    nn_move_scorer             the raw MoveScorer torch forward passes
  nn_alter                   AlterMoveProbNN.forward_numpy
  ponder                     elective; excluded from compute_floor
```

The two Stockfish scans are limited differently, and this is the point of
the split:

- **`sf_scan_fullwidth` is time-capped.** It reaches depth 4–8 of the 10 it
  asks for, so the depth limit never binds — the time limit is what is
  operative. A slower CPU therefore buys a *shallower* scan rather than a
  longer one: it degrades eval quality rather than adding latency, and no
  pacing knob compensates for that. Depth reached is recorded next to wall
  time so the difference shows up instead of hiding.
- **`sf_sharpness` is depth-capped with no time limit,** so a slower CPU
  pays here in wall time, straightforwardly. On the dev machine it is the
  single largest stage — larger than the full-width scan and the NN
  together — which makes `SHARPNESS_SCAN_DEPTH` the first knob to reach for
  if a device needs its floor lowered.

`analytics_total` is derived (the sum of its components) rather than timed
by calling `calculate_analytics`. Timing the composite as well would re-run
both scans against a transposition table the components just warmed, and
CLAUDE.md flags exactly this: the quick scan "leaves load-dependent
transposition-table state that perturbs later same-process scans". Measured
that way the composite came out *below* its own sharpness component.
Running each component once, in production order, is the reading that
transfers between machines.

A raw `stockfish_nps_probe` (fixed position, fixed depth 18, no time cap)
gives a clean hardware number independent of the bot's own scan settings.

### vision

Two halves, because only one is portable.

- **capture** — real grabs off the real screen at the region sizes the
  client uses. Needs a display, measures the whole stack, varies most
  between devices, and cannot be faked. Degrades to a recorded reason when
  unavailable (headless, locked session) rather than failing.
- **recognition** — `get_fen_from_image` and `remove_background_colours` on
  board images synthesised in-process from the tracked piece templates.
  Runs headless and gives every device *identical input*, which is what
  makes it comparable; `auto_calibration/offline_screenshots` is gitignored,
  so two machines would otherwise be timing different pictures. A
  self-check reports how many synthesised boards read back to the exact
  input placement, guarding against timing an early-out instead of the real
  matching path.

Two caveats: `remove_background_colours` is timed at full board resolution
while production's `get_fen_from_image` downscales first, so the standalone
figure is an upper bound; and piece-template size depends on whether a
calibration profile is present, so the result records
`using_profile_templates` and `compare` refuses to read a recognition
difference as hardware when the two sides disagree on it.

Clock OCR is not covered — the digit templates alone do not reconstruct a
realistic clock crop, and `simulation.calibrate detection` already measures
it wherever real screenshots exist.

### mouse

The planned half is free of device variance by construction: every gesture
duration is *decided* by a formula in `common/move_timing.py`, not measured
from hardware. It is reported anyway, as the target the realised half is
judged against.

The realised half (`--mouse-live`, moves the real cursor) is where a device
can betray the bot. `CustomCursor.quick_move_to` walks a Bezier curve at
~100Hz — `steps = max(3, int(duration / 0.01))`, one
`pyautogui.moveTo(_pause=False)` per step — so the loop only holds its
schedule if a single `moveTo` costs appreciably less than its 10ms per
step. Where it does not (remote desktop, heavy DPI virtualisation, a busy
compositor), gestures silently overrun and that lands directly on the game
clock. The report states plainly whether `moveTo` fits the step budget.

The cursor is returned to where it started. PyAutoGUI's fail-safe is left
on throughout the measurement loop; it is suspended only for the single
move that re-centres the cursor, because a cursor idling in a screen corner
(very common) makes even the move that would rescue it raise.

## Comparing devices

`compare` lines results up against the first file, which is treated as the
baseline, and tries hard not to blame hardware for a configuration
difference: a machine on a different Stockfish build, thread count, or
uncalibrated templates gets a **comparability warning** rather than a
silently misleading ratio. The Windows and Linux clones ship *different
Stockfish versions* (16 vs 17), so this fires on the most obvious
cross-device comparison there is.

## Limitations

- **`compute_floor` is noisy run to run.** Two back-to-back runs on the
  same machine (`--repeat 2` then `--repeat 3`) gave floors of 510ms and
  699ms — a ×1.37 spread, as large as a genuine device difference. The
  stages underneath it moved far less over the same pair (≤16%, and
  `sf_scan_fullwidth` not at all), so the extra variance is in
  `make_move_total`, whose branches — premove search, ponder width, mood —
  are sampled per move rather than fixed. The per-game character draw is
  also resampled whenever the ply count goes backwards, which a corpus of
  unrelated positions triggers constantly. Do not call a device slow on one
  run: use `--repeat 5` or more, and compare several runs per machine
  before believing a floor difference.
- Single process, single run: no attempt to control for thermal throttling,
  background load, or power profile. Run it twice on a laptop, once on
  battery and once plugged in, before trusting a cross-device delta.
- The corpus is 12 positions with minimal move history, so history-dependent
  paths (opponent-blunder startle, `patch_fens` linking) stay quiet. That is
  deliberate — they fire unpredictably and would add variance unrelated to
  the machine — but it means the floor is a clean-path floor, not a
  worst-case one.
- `compute_floor` is derived by parsing the engine's own
  `"Took N seconds for pondering"` log line. If that string changes, the
  floor silently becomes equal to `make_move_total`.
