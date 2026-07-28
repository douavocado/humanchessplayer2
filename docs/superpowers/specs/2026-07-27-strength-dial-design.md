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

## Feature priority: speed first

Two knobs cannot in general hit four targets, and Phase A showed the bot's
starting profile is spread across the entire width of the table (see Phase A
findings). So the objective is **weighted, not equal**: the timing features
(`movetime_mean`, `instant_move_rate`) are primary, and `t1_rate` /
`blunder_rate` are fit as far as the speed choice allows, with the residual
reported rather than hidden.

This was a judgement call by the repo owner, and the measurements support it.
Per-feature *resolvable* rating signal -- the 2100-2299 -> 2800+ span divided
by the standard error a 150-game arm achieves -- ranks the timing features
above the accuracy ones:

| feature | span over 700 Elo | se (150-game arm) | span/se |
|---|---|---|---|
| Mean emt | 0.26s | ~0.015 | **17.3** |
| Instant rate | 7.0pp | ~0.0045 | **15.6** |
| Blunder rate | 2.1pp | ~0.0017 | 12.5 |
| Top-1 match | 5.0pp | ~0.0045 | 11.2 |

Timing carries more rating information per unit of measurement noise, so a dial
that gets timing right and move-agreement approximately right is better
calibrated than one that splits the difference.

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

## Phase A findings (measured 2026-07-27)

5 arms x 150 complete self-play games, 60+0, analysed at depth 10 against
`baselines/bullet_1plus0_2300_plus.json`.

| arm | noise | quick | t1 | blunder | emt | instant |
|---|---|---|---|---|---|---|
| noise_low | 0.55 | 2.5 | 0.3514 | 0.0458 | 1.109 | 0.2440 |
| baseline | 0.75 | 2.5 | 0.3338 | 0.0418 | 1.114 | 0.2519 |
| noise_high | 0.95 | 2.5 | 0.3331 | 0.0406 | 1.100 | 0.2469 |
| quick_fast | 0.75 | 2.0 | 0.3377 | 0.0370 | 1.016 | 0.2526 |
| quick_slow | 0.75 | 3.0 | 0.3492 | 0.0450 | 1.213 | 0.2439 |

**1. Mean emt is fully controllable and linear in `quickness`.**

```
emt = 0.1964 * quickness + 0.6234        (R^2 ~ 1 on three points)
```

That spans the entire band table: quickness 1.92 (2850) to 3.24 (2200), with
the shipped default of 2.5 landing at emt 1.114, i.e. **~2560**. This is the
strongest and best-measured relationship in Phase A (~20 se across the probed
range) and it is the dial's working axis.

| band | 2200 | 2350 | 2450 | 2550 | 2650 | 2750 | 2850 |
|---|---|---|---|---|---|---|---|
| target emt | 1.26 | 1.23 | 1.19 | 1.15 | 1.11 | 1.02 | 1.00 |
| quickness | 3.24 | 3.09 | 2.88 | 2.68 | 2.48 | 2.02 | 1.92 |

**2. Instant rate is immovable by either knob -- the key negative result.**
Across all five arms it sits in 0.2439-0.2526, a span of 0.0087 against a
per-arm se of ~0.0045: flat. The human table runs 0.3087 (2100-2299) to 0.3791
(2800+), so the bot is **12.6 se below the table floor** and neither knob
closes any of it.

This is mechanism, not tuning. Sub-1s moves cannot be produced by shortening
think time, because the engine-compute floor sets a hard minimum -- which is
exactly why `common/constants.py` routes confidence through the premove channel
rather than through think time. Instant rate is a *channel* feature: it is
produced by premove fires, ponder-dic hits, and snap-gate decisions, none of
which `quickness` or `eval_noise_scale` touch.

**Consequence for the knob set:** the two timing features have different
mechanisms and only one currently has a working knob. `quickness` owns mean
emt; instant rate needs the instant-channel knobs (`GAME_PREMOVE_MEAN`,
`GAME_PONDER_SNAP_MEAN`, the snap gate, and the ambiguity deltas landed inert
in a4d2c7e). Phase B must add them.

**3. `eval_noise_scale` moves t1 weakly and saturates above 0.75.** 0.55 ->
0.75 costs 0.0176 t1 (2.8 se, real); 0.75 -> 0.95 costs 0.0007 (0.1 se, noise).
Extrapolating the live region's slope, reaching the 2850 t1 of 0.3970 would
need a noise scale near 0.03. **The top of the table is not reachable on t1**,
and the pre-identified fallback does not help: the breadth sweep already showed
breadth leaves t1 flat while buying blunder rate, which is the one feature
already past the top of the table.

**4. Do not tune on `blunder_rate` or `acpl` at this sample size.** They move
non-monotonically across arms (acpl 75.0 / 88.8 / 95.2 / 105.6 / 85.8 with no
clean ordering in either knob), consistent with the documented ACPL-variance
warning in CLAUDE.md. Treat them as guards, not targets, until a replicate
exists.

**5. The bot has no single rating.** Baseline placement per feature: t1 below
2100, blunder above 2800, emt ~2560, instant below 2100. The per-feature spread
is the full width of the table, which is what makes the weighted objective
above necessary rather than merely convenient.

## Control surface: one owning knob per feature

This resolves the knob-set question the Design section left open, and replaces
`DIFFICULTY` as the way playing strength is set.

**`DIFFICULTY` (`Engine.playing_level`) is frozen as a structural constant, not
a tuning knob.** It fails on three measured counts: it is an integer so it
cannot interpolate; it drives *both* base search breadth
(`decision_logic.py:39`) and the eval-noise floor (`engine.py:423`,
`ponderer.py:375`) simultaneously, so it cannot be aimed at one feature; and
widening breadth raises the per-move compute floor, which is precisely what
gates the instant-move rate. `common/constants.py:166` already records that
`HUMAN_EVAL_NOISE_SCALE` was introduced as the free alternative to a
`DIFFICULTY` bump, and `cheat_detection/elo_progression.py:13` that
`DIFFICULTY=4` improved ACPL and blunder rate but got flagged.

Pin it at 3. Its constructor default is currently 6, which matches nothing
shipped -- a bare `Engine()` (parity harness, unit tests) runs at twice the
live bot's base breadth. Change the default to 3 or drop it so callers must be
explicit.

**The freeze is scoped to 1+0 and should be revisited at longer controls.**
Two of the three objections above are bullet-specific rather than intrinsic:

- *The compute cost* only bites because the per-move floor is most of a 60+0
  move. At ten minutes it is a small fraction, so a breadth bump no longer gates the
  instant-move rate -- the mechanism that makes it disqualifying here simply
  is not present there.
- *Integer granularity* only bites because the band-to-band feature gaps at
  this control are tiny (5pp of top-1 across 700 Elo). Where the gaps are
  wider, a coarse knob is proportionally less coarse, and the continuous knobs
  can interpolate within its steps.

Only the third objection -- that it moves breadth and noise together, so it
cannot be aimed at one feature -- is structural and survives at any control.

There is also a positive argument for it at longer controls: bullet rewards
pattern recognition and clock economy, longer controls reward calculation
depth, and search breadth is the thing in this engine that most directly
models calculation depth. `eval_noise_scale`'s saturation above 0.75 is what
caps the dial's top end at 1+0; breadth may well be the knob that lifts that
cap at ten minutes, where `DIFFICULTY` could reasonably become the *primary*
strength axis rather than a frozen constant.

| feature | owning knob | evidence | usable range |
|---|---|---|---|
| Mean emt | `quickness` | `emt = 0.1964*q + 0.6234`, ~20 se (Phase A) | **full table**, q 1.92-3.24 |
| Instant rate | `opening_book_fast_path`, then the ponder knobs | opening 0.389 -> 0.567 vs human 0.565 | opening closed; endgame unproven |
| Blunder rate | `midgame_breadth_strength_bonus` | 0.0380 -> 0.0169 over +0..+5 | full table, integer steps |
| Top-1 match | `eval_noise_scale` | -0.088 t1 per unit over 0.55-0.75 | **partial -- saturates above 0.75** |

The rule that makes this a firewall rather than a pile of knobs: **each feature
has exactly one owner, and a second knob is never reached for to fix a feature
that already has one.** That is exactly what stopped working with `DIFFICULTY`,
which owned two features at once.

Precedence stays as specced above: explicit knob argument > `target_rating` >
module constant.

### What this surface cannot do, and must say on the tin

- **Top-1 match cannot reach the top of the table.** `eval_noise_scale`
  saturates above 0.75, and the live region's slope implies a value near 0.03
  to hit 2850's t1. Breadth does not help -- it leaves t1 flat while buying
  blunder rate. So above roughly 2650 the bot plays at the right *speed* with
  the right *error rate* while agreeing with the engine like a 2500. State
  this at the interface rather than letting it be discovered.
- **The per-bucket profile stays inverted** (quiet below 2100, sharp-forced
  above 2800). Untouched by any of this.
- **Endgame instant rate is the open gap** (0.270 vs a human 0.496) and is the
  one feature whose owner is not yet established. `PONDER_TIME_PER_POSITION` is
  the candidate and is now a per-instance knob, but nothing has measured it.

Only two mappings are actually fitted today: `quickness -> emt`, and the book
fast path. Before any `target_rating` ships, breadth -> blunder and noise -> t1
need the same treatment and the endgame lever needs to be established or ruled
out.

## Phase B findings (measured 2026-07-28): both accuracy axes fail

5 arms x 150 complete self-play games, 60+0, same conventions as Phase A, plus
Phase A's three noise arms reused.

**1. `eval_noise_scale` -> t1 is non-monotone and has no range.**

| noise | 0.25 | 0.40 | 0.55 | 0.75 | 0.95 |
|---|---|---|---|---|---|
| t1 | 0.3556 | 0.3429 | 0.3514 | 0.3338 | 0.3331 |

Down, up, down, flat; every step is 0.1-2 se, so the 0.25-0.55 region is flat
within noise. The 0.55 -> 0.75 slope Phase A extrapolated from was one wobble
among several, and the "saturates above 0.75" conclusion recorded there is
superseded: it saturates in **both** directions. Total span across the whole
usable range is 0.0225 against a band table needing 0.0504, and the best t1
reached anywhere (0.3556, at noise 0.25) sits between the 2200 and 2450 bands.
Even at near-zero noise the bot cannot reach 2450's t1.

Mechanism: in `get_human_move`, un-re-evaluated moves take a 60cp penalty
(`DEPTH_PENALTY` x2 plus `ZERO_DEPTH_PENALTY`) and *which* moves get
re-evaluated is `random.sample`. In quiet positions -- 74% of moves -- the true
eval spread between candidates is far below that penalty, so the choice is
decided by the re-evaluation lottery, not by evals or by the noise term on top
of them. The t1 lever was never noise.

**2. `midgame_breadth_strength_bonus` -> blunder rate points the wrong way.**

| breadth | +0 | +1 | +2 | +3 |
|---|---|---|---|---|
| blunder | 0.0418 | 0.0448 | 0.0380 | 0.0343 |
| t1 | 0.3338 | 0.3433 | 0.3495 | 0.3554 |

The blunder trend is real (-0.0075 over +0 -> +3, 3.1 se) but the achievable
range, 0.0343-0.0448, sits at or below the **2850** band (0.0444). The bot is
already better than the best humans at every setting and breadth only improves
it further; reaching 2200's 0.0657 would need to go the other way, past +0.

Breadth *does* move t1 (0.3338 -> 0.3554, ~3.4 se), which **contradicts the
earlier "breadth leaves t1 flat" finding** -- that was measured on adjudicated
games. Breadth is a better t1 lever than noise, though +3 only reaches ~2350.

**3. The structural obstruction.** The two results share one cause. The bot is
**superhuman on error avoidance and subhuman on move agreement**, and every
strength knob moves those together: more breadth raises t1 *and* lowers blunder
rate. To look human it needs t1 up **and** blunder rate up. No knob does that,
because strength in this engine is a single direction while the bot's profile
is off the human manifold in two directions at once.

This is why the accuracy half of the dial is not merely weakly calibrated but
uncalibratable with the current knob set. It is not a tuning problem, and no
amount of extra sweep compute addresses it -- it needs a lever that makes the
bot *choose differently* without making it *choose better*, i.e. the
re-evaluation-lottery structure above, not a strength knob.

**Consequence for `CALIBRATED_KNOBS`:** it stays `("quickness",)`. The dial is
a pace dial, and the spec should keep saying so.

## Limitations to carry into the docs

- ~200-300 Elo granularity; the interface should expose coarse steps.
- Aggregate only -- the bucket profile stays inverted, and a conditioned
  analysis will still identify the bot.
- **Notation warning.** `--tc 60+0` is parsed in **seconds**
  (`simulation/run.py` sets `initial_time = 60.0`), so "60+0" throughout this
  document means one minute of bullet, i.e. 1+0 in standard chess notation.
  Prose elsewhere in the repo (and in CLAUDE.md) uses standard notation, where
  "3+2" and "10+0" mean *minutes*. The same token therefore means different
  things in a CLI argument and in prose -- this document spells out longer
  controls in words to avoid it.
- **Every number here is calibrated on 60+0 (one minute), and the timing half of the
  control surface is the part least likely to transfer.** Two of the four
  owning knobs are aimed at timing features whose absolute values are
  artefacts of a 60-second budget: the `quickness -> emt` fit
  (`0.1964*q + 0.6234`) is a 60+0 regression and its intercept has no meaning
  at another control, and the instant-move threshold (emt < 1s) is a far
  larger share of a bullet move than of a three- or ten-minute one. The
  compute floor
  that makes instant rate a bypass-only feature is *absolute*, so it occupies
  proportionally less of a longer move -- at ten minutes the engine path may
  reach sub-1s moves on its own and the book fast path may be unnecessary or
  even wrong. Expect to redo the strength-timing calibration per time control:
  a fresh `elo_progression` band table on a corpus pinned to that exact clock,
  then a fresh Phase A. What carries over is the *rule* -- one owning knob per
  feature -- not the roster: which knob owns which feature is itself a
  per-control finding, and **`DIFFICULTY` is expected to come back off the
  freeze at longer controls** (see the Control surface section). None of the
  coefficients should be assumed to transfer.
- Likewise the accuracy half: the finding that human improvement is *smallest*
  in quiet positions is plausibly bullet-specific, so which feature is hardest
  to move may itself change at longer controls.
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
