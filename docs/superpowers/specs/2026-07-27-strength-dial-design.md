# A continuous strength dial calibrated on human rating bands

Date: 2026-07-27
Status: **design approved; Phase A (measurement) running, no code written.**
Phase A needs no engine changes, so the fit is being measured before anything
is built.

## Problem

There is no way to ask this bot to play at a chosen level. `--rating` is a PGN
label plus a *relative* confidence input (it scales think time and the premove
probability against the opponent's rating); it is not a strength setting. The
only absolute anchor anywhere in the repo is a one-line benchmark comment in
`common/constants.py`:

```
DIFFICULTY   |   QUICKNESS   |    ELO
    3        |     2.2       |  ~2500
```

Levers that *do* change strength exist -- `eval_noise_scale`, the breadth
bonuses, `DIFFICULTY`, `quickness` -- but each is a raw mechanism with no known
map onto a rating, and several move human-likeness and strength in opposite
directions. Setting them by hand means guessing.

What makes a principled dial possible now is that the target already exists.
`cheat_detection/runs/elo_progression/report_timing.md` resolves 2.28M human
moves into 7 rating bands x 8 features. That table *is* the definition of what
a rating plays like; the work is inverting the bot's knobs onto it, not
measuring humans again.

## What "2700" will mean

**Behavioural, with a results offset applied later.** The dial is fit offline
so the bot's move statistics match the target band's measured profile. Once the
bot has played enough live rated games, wherever its actual rating settles
becomes a one-off offset on the labels.

The alternative -- calibrating directly on result Elo -- was rejected as
unachievable offline rather than undesirable. Adjudicated self-play Elo is
purely *relative*: it can say arm B beats arm A by 115 Elo, never that either
is 2700. An absolute anchor needs hundreds of live rated games per anchor
point, and it would tune strength blind to *how* the bot gets there.

## Resolution limit: ~200-300 Elo, not 50

This constrains what the dial may honestly offer, so it is settled up front.

Human features move remarkably little across 700 Elo (2100-2299 -> 2800+,
aggregate, complete games):

| feature | 2100-2299 | 2800+ | span over 700 Elo |
|---|---|---|---|
| Top-1 match | 0.3466 | 0.3970 | +5.0pp |
| Blunder rate | 0.0657 | 0.0444 | -2.1pp |
| Mean emt | 1.26 | 1.00 | -0.26s |
| Instant rate | 0.3087 | 0.3791 | +7.0pp |

All four are monotone and near-linear in rating, which is what makes knob
interpolation plausible. But a 150-game complete self-play arm yields ~10-13k
mover-moves, so the standard error on top-1 match is ~0.004-0.005 -- and the
gap between *adjacent 100-Elo bands* is 0.005. One arm can barely separate
neighbouring bands, and separating them reliably would need roughly 16x the
compute per point.

Top-1 match is the binding constraint; mean emt (se ~0.015s against band gaps
of 0.04-0.10s) and instant rate (se ~0.0045 against gaps of ~0.01) both resolve
more comfortably.

**Consequence for the interface:** the dial should expose coarse steps rather
than accept any integer, so it cannot promise precision that was never
measured. A caller asking for 2700 vs 2750 should get the same bot, and should
be able to see that from the API rather than by reading this document.

## Scope

**In:** a `target_rating` parameter resolving to a knob vector, fit against the
*aggregate* four features above, for 1+0 bullet.

**Out, deliberately:**

- **Flattening the per-bucket profile.** The bot's strength profile is inverted
  relative to humans -- quiet-position t1 0.262 against a human 0.300,
  sharp-forced 0.857 against 0.790 (`docs/position-conditioned-human-
  likeness.md`). An aggregate fit hits the overall number by *averaging over*
  that inversion, not by removing it. A bot set to 2900 will show aggregate t1
  ~0.397 as intended while `bucket_diagnostic` still shows it playing sub-2100
  in quiet positions and above-2800 in tactical ones. This is a known,
  documented limitation, not an oversight; it is a separate problem and does
  not need to block the dial.
- **Other time controls.** The band table is 60+0 and most of it is expected to
  differ elsewhere -- see the Time-control scope section of the findings doc.
  Another control needs its own table, not a rescaling of this one.
- **Result-Elo calibration.** See above.

## Design

### Target table

Lift the 7 bands x 4 aggregate features into a tracked file. The run output
under `cheat_detection/runs/` is gitignored, and the dial's definition cannot
live in an untracked artefact. A rating resolves to a target vector by linear
interpolation between band midpoints (2200, 2350, 2450, 2550, 2650, 2750,
2850), clamped outside that range.

### Knobs: start with two

The instinct is one knob per target feature, but neither the human features nor
the bot's move independently. Minimal hypothesis:

| axis | knob | default | direction |
|---|---|---|---|
| accuracy | `eval_noise_scale` | 0.75 | lower -> t1 up, blunder down |
| speed | `quickness` | 2.5 | lower -> emt down, instant rate up |

Two knobs against four targets is deliberately overdetermined. If two knobs can
hit all four, the bot's coupling between accuracy and speed already matches the
human coupling and there is nothing further to fit; if they cannot, the residual
says precisely which extra knob is needed.

The off-diagonal terms are strong and *adverse*, which is the whole reason this
is a joint fit rather than four independent assignments: humans get faster and
more accurate together, but in the bot `noise_sd` is proportional to
`1/target_time` (`engine.py:403`), so speeding the bot up mechanically makes it
worse. Inverting a 2x2 Jacobian handles exactly this.

Both knobs are already per-instance constructor arguments and already exposed
as `simulation.run` flags, which is why Phase A needs no code.

Two extra knobs are identified in advance, to be added only if residuals demand
them:

- `midgame_breadth_strength_bonus` for blunder rate, if `eval_noise_scale`
  saturates. `common/constants.py` already warns of diminishing returns below
  0.75, and breadth is known to drive blunder rate hard (0.0380 -> 0.0169 over
  0..+5). It is an **integer**, so it can only ever be a coarse step with
  `eval_noise_scale` interpolating within it.
- The snap-gate means (`GAME_PREMOVE_MEAN`, `GAME_PONDER_SNAP_MEAN`, and the
  ambiguity deltas from the 2026-07-27 snap-gate spec) for instant rate.

`GAME_PACE_MEAN` is held in reserve behind `quickness`: both scale think time,
so they are near-redundant, and `quickness` is already threaded end to end
while `GAME_PACE_MEAN` is read straight from the module constant in
`game_character.py:60`.

### Fitting

All arms: pure self-play, complete games (`--simulate-full`), 60+0, 150
games/arm. Driver: `cheat_detection/runs/strength_dial/run_phase_a.py`.

Self-play because arms with different score rates are not comparable on
win-probability features -- `blunder_rate` and `mean_wc_loss` saturate in
decided positions.

Complete games for two reasons, the second the stronger. First, the calibration
table is built from complete human games, and the truncation matching used in
the earlier bucket work was a device for that specific A/B -- reusing it here
would reintroduce the mismatch it fixed. Second, and more fundamentally, **two
of the four target features are `mean emt` and `instant rate`, which are time-
economy measurements, and the scramble is where clock economy shows itself.**
Truncating at 15s would delete the phase the dial most needs to see: it is
precisely where a strong player's clock handling separates from a weak one's.

Adjudication exists to stop clock-race variance diluting an *Elo* signal. We are
measuring per-move features rather than results, so it buys nothing here while
costing the entire scramble phase.

- **Phase A -- baseline + Jacobian.** 5 arms: baseline, `eval_noise_scale`
  0.55/0.95, `quickness` 2.0/3.0. Its first output answers a question we
  currently cannot answer at all: **where does the bot sit on this table today,
  on complete games?** Every later phase depends on that number.
- **Phase B -- solve and verify.** Invert the Jacobian, solve for three anchors
  (2400 / 2650 / 2900), run them, measure residuals against target.
- **Phase C -- refine.** One Newton step on any anchor outside 1 se. If a
  feature is systematically unreachable, add its identified knob and repeat
  locally rather than globally.

Interpolation between fitted anchors is linear, and must be **validated at one
off-anchor rating** rather than assumed -- breadth's effect is known to be
strongly non-linear (it saturates at +3), so linearity is a hypothesis about
these two knobs, not a given.

### Code shape (lands only after the mapping is known)

New `common/strength_profiles.py`: the fitted anchor table plus
`resolve(rating) -> dict`.

`Engine.__init__` gains `target_rating`. Precedence is explicit and tested:

```
explicit knob argument  >  target_rating  >  module constant
```

Explicit knob arguments must keep winning, or every existing sweep breaks.

Default `target_rating=None` resolves to today's constants exactly, so the
change lands **inert** and the parity harness passes unchanged without
`--record` -- the same discipline as the ambiguity snap gate, and for the same
reason: inertness is a property to be proven, not asserted.

`simulation/run.py` gains `--target-rating` plus `--a-*`/`--b-*` variants,
following the `midgame_breadth_strength_bonus` precedent.

## Testing

1. **Engine parity harness passes unchanged, without `--record`.** Load-bearing:
   a failure means the default is not inert.
2. **Unit tests on `resolve()`** -- pure, fast, no Stockfish. Anchor ratings
   reproduce their fitted vectors exactly; output is monotone in rating for
   each knob; ratings outside the table clamp rather than extrapolate; the
   precedence order above holds.
3. **Client unit tests** green (7 of 34 error on `fastgrab._linux_x11` without
   X11 -- environment, not regression; check against a clean tree).
4. **Lint** clean on changed files.
5. **Fit quality reported honestly**: a table of target vs achieved for each
   anchor, with standard errors, published in this directory. An anchor that
   misses is recorded as missing rather than quietly re-fit.

## Limitations to carry into the docs

- ~200-300 Elo granularity; the interface should expose coarse steps.
- Aggregate only -- the bucket profile stays inverted, and a conditioned
  analysis will still identify the bot.
- 1+0 bullet only.
- Labels are behavioural until live-validated.

## Follow-ups (not in scope)

- **Live result validation** to convert behavioural labels into a rating offset.
- **Per-bucket dial (v2).** Would need a working sharpness-conditioned selection
  lever. The obvious candidate -- reweighting selection toward the NN human
  prior in flat positions -- was measured on 172 quiet positions and **ruled
  out**: the prior agrees with the engine's top move only 22% of the time
  there, below the bot's current 0.262 and well below the human 0.300, so
  leaning on it would push quiet t1 the wrong way. (Caveat: that probe used the
  depth-12 multipv-5 scan as its reference rather than cheat_detection's
  depth-10 measurement, so it is directional, not an exact comparison.)
- **A second time control**, starting from its own `elo_progression` table.
