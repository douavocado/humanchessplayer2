# Ambiguity-conditioned intuition snap gate

Date: 2026-07-27
Status: **design approved, not implemented.**

## Problem

`_get_time_taken`'s intuition gate (`engine_components/decision_logic.py`)
currently fires on sharpness alone:

```python
snap_gate = engine.game_snap_gate if engine.game_snap_gate is not None else 0.65
snap_in_sharp = (sharpness >= 0.25) and (np.random.random() < snap_gate)
```

Every sharp position is treated identically, but humans do not treat them
identically. From `docs/position-conditioned-human-likeness.md`, the human
speed-up with rating concentrates in positions with **one right answer**:

| bucket | instant rate 2100-2299 → 2800+ | Δ |
|---|---|---|
| Sharp, forced (ambiguity == 1) | .332 → .464 | **+13.2pp** |
| Sharp, messy (ambiguity >= 2) | .293 → .407 | +11.4pp |
| Quiet | .329 → .390 | +6.1pp |

At 2100-2299 the instant rates for sharp-forced (.332) and quiet (.329) are
identical; by 2800+ they have separated to .464 vs .390. Strong players stop
*calculating* positions they can *recognise*. The current gate cannot express
that, because it does not know whether the position has one good move or
several.

The bot's measured symptoms:

| | bot | human 2300+ |
|---|---|---|
| instant_in_sharp_rate | 0.3201 | 0.3500 |
| instant_move_rate | 0.2253 | 0.2965 |
| corr_time_sharpness | **+0.0558** | +0.0374 |

`corr_time_sharpness` is the diagnostic one: the bot's think time tracks
sharpness *more* strongly than humans'. It slows down where strong humans
speed up.

A secondary benefit: in sharp-forced positions the bot is simultaneously too
slow **and too accurate** (t1 0.857 vs the 2800+ human 0.801). Snapping there
shortens `target_time`, which raises `noise_sd` (`engine.py:388`,
`noise_sd ∝ 1/target_time`) and pulls sharp-forced t1 down toward human. One
change, two divergences.

## Scope

**In:** an ambiguity split on the snap-gate probability, shipped inert, with
per-instance overrides so it can be swept.

**Out, deliberately:**

- Re-fitting the sharpness→time envelope. Human mean emt *across* buckets
  (quiet 0.96, moderate 1.27, sharp-forced 0.87 at 2800+) suggests the bot's
  curve is close to inverted, but that comparison is confounded by clock state
  — sharp positions may cluster in scrambles where everything is fast. Only
  the within-bucket, band-to-band contrasts are trustworthy, and they support
  the split and nothing more.
- Any change to move *selection*. That is workstream 1 (quiet-position t1/t2),
  deferred so the two do not confound each other — both move t1, and they
  interact through `target_time`.
- A continuous strength dial (see Follow-ups).

## Design

### Ambiguity from the existing scan

`state.compute_sharpness` already stores `engine.sharpness_scan` as
`{uci: win_chance}` from a multipv-5, depth-12 scan. Ambiguity is a count over
that dict — **no new Stockfish work, no new RNG draw**:

```python
ambiguity = sum(1 for wc in scan.values() if best_wc - wc <= AMBIGUITY_WC_WINDOW)
```

`AMBIGUITY_WC_WINDOW = 0.05` deliberately mirrors
`cheat_detection/config.py:ambiguity_wc_window`. The engine must gate on the
same quantity the analyser measures; if the two drift, every sweep is tuned
against something we cannot observe.

`engine.ambiguity` is set in `update_info` beside `engine.sharpness`
(`engine_components/state.py:163`), reset to `None` alongside
`sharpness_scan`, and left `None` when the scan fails — the existing failure
path already falls back to a neutral 0.25 sharpness, and `None` ambiguity
means "apply no split", so a failed scan degrades to exactly today's
behaviour.

### The gate

```python
gate = engine.game_snap_gate if engine.game_snap_gate is not None else 0.65
if engine.ambiguity == 1:
    gate = min(1.0, gate + engine.ambiguity_forced_snap_delta)
elif engine.ambiguity is not None and engine.ambiguity >= 2:
    gate = max(0.0, gate - engine.ambiguity_messy_snap_delta)
snap_in_sharp = (sharpness >= 0.25) and (np.random.random() < gate)
```

Additive offsets on the per-game draw, rather than replacement values, so the
per-game character (`game_snap_gate ~ U(0.55, 0.95)`, the "trust-the-gut game
vs grinding game" axis) survives the split instead of being flattened by it.

**Inertness is a hard property, not an aspiration.** With both deltas at 0,
`gate == game_snap_gate` exactly, the `np.random.random()` call sits in the
same place consuming the same value, and computing ambiguity consumes no
randomness. The change is bit-identical to today. This is what the parity
harness will assert.

The existing log line gains the ambiguity value and the effective gate, so
`engine.log` shows why a given position snapped or did not.

### Constants and sweep wiring

| constant | file | default |
|---|---|---|
| `AMBIGUITY_WC_WINDOW` | `common/search_constants.py`, beside `SHARPNESS_SCAN_*` | 0.05 |
| `AMBIGUITY_FORCED_SNAP_DELTA` | `common/constants.py`, beside `GAME_SNAP_GATE_RANGE` | **0.0** |
| `AMBIGUITY_MESSY_SNAP_DELTA` | `common/constants.py` | **0.0** |

The window is a scan-derived threshold (hence `search_constants.py`); the
deltas are behavioural tuning (hence `constants.py`).

Both deltas thread through as per-instance overrides following the
`midgame_breadth_strength_bonus` precedent exactly:

```
Engine.__init__(ambiguity_forced_snap_delta=None, ambiguity_messy_snap_delta=None)
  → simulation/game_runner.py BotSpec fields
  → simulation/run.py flags: --ambiguity-forced-snap-delta / --ambiguity-messy-snap-delta
                             plus --a-* / --b-* variants
```

That precedent is what made the breadth sweep a config change rather than a
code change, and this lever needs the same treatment: the map from gate
probability to realised instant rate is not analytically predictable, so it
must be measured.

## Testing

Landing gate, all cheap:

1. **Engine parity harness passes unchanged, without `--record`.**
   (`venv/bin/python -m unittest discover -s testing/engine_parity`) This is
   the load-bearing test — a failure means the change is not inert.
2. **Client unit tests** green (`testing/client`).
3. **Lint** clean on changed files (`venv/bin/ruff check <files>`).
4. **New unit test for the ambiguity computation.** The only genuinely new
   logic, and it is pure. Cases: single candidate → 1; all candidates equal →
   n; one clear best → 1; empty dict → `None`; `None` scan → `None`; a
   candidate exactly `AMBIGUITY_WC_WINDOW` below best → counted (boundary is
   inclusive, matching `<=` in the analyser). Worth testing because the whole
   lever is tuned against this number agreeing with `cheat_detection`, and an
   off-by-one in the window would silently invalidate every future sweep.

**No simulation run in this change.** With inert defaults there is nothing
behavioural to measure; parity proves identity more cheaply and more exactly
than a 40-minute sim could.

## Follow-up sweep (separate work)

Self-play, not head-to-head: arms with different score rates are not
comparable on win-probability error metrics (the saturation trap), and equal-
score self-play removes that. 150 games/arm, 60+0, ~30 min each.

| arm | forced Δ | messy Δ |
|---|---|---|
| control | 0 | 0 |
| A | 0.15 | 0 |
| B | 0.30 | 0 |
| C | 0.15 | 0.15 |
| D | 0.30 | 0.15 |

`MESSY_DELTA = 0` is in the grid on purpose. The human evidence for the forced
side is strong (+13.2pp band-to-band); for the messy side it is a modest
4-6pp forced-vs-messy gap, and the bot already under-fires instants overall
(0.2253 vs 0.2965), so lowering the messy gate pushes an already-low number
lower. The data should decide whether the split is symmetric.

Judged on:

- `instant_in_sharp_rate` 0.3201 → target ~0.3500
- `corr_time_sharpness` +0.056 → target ~+0.037
- sharp-forced t1 via `bucket_diagnostic`, 0.857 → expected to fall toward the
  human 0.801
- `mean|player_z|` as the guard that nothing else regressed (target ~0.8, and
  **not** 0 — too-average is its own tell)

Two notes for whoever runs it. Results must be read against the
**truncation-matched** baseline (`bullet_1plus0_2300_plus__trunc15.json`), not
the plain one. And the run should report the ambiguity distribution the bot
actually encounters: if sharp positions with `ambiguity == 1` are rarer than
the corpus's ~7% of moves, the lever has less to bite on than expected, and a
null result would mean "no headroom" rather than "no effect".

## Follow-ups (not in scope)

- **Workstream 1: quiet-position t1/t2.** The larger divergence (~74% of
  moves, t1 0.262 vs 0.300). The likely lever is a sharpness-conditioned
  multiplier on `noise_sd`, analogous to the existing `noise_phase` dict
  (opening 0.8 / midgame 1.2 / endgame 0.3), exploiting the fact that in quiet
  positions all candidates sit within a hair of each other in win probability
  — so raising engine agreement there barely moves wc_loss or blunder rate,
  which are already human.
- **A continuous strength dial** ("play at 2700 vs 2750"). Does not exist
  today: `--rating` is a PGN label plus a *relative* confidence input
  (`rating_factor` on think time, `premove_sf` boost), not a strength setting,
  and the only absolute anchor in the repo is a one-line `~2500` benchmark
  comment. `eval_noise_scale` is the natural continuous knob. **Blocked on
  shape, not effort:** the bot's strength is currently not scalar — it plays
  above 2800+ in sharp-forced positions and below 2100-2299 in quiet ones, so
  a result-Elo fit and a behavioural-profile fit would disagree, and neither
  would be wrong. Flatten the bucket profile first (this spec and workstream
  1), then fit the dial.
