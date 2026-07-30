# Pacing calibration for longer time controls

Date: 2026-07-30
Status: **design approved, unimplemented.**
Scope: 180+0 only. Two orthogonal corrections — a global pace level and the
opening envelope shape. Everything about 300+0 and 600+0 is explicitly out.

Follows `2026-07-29-longer-time-control-calibration-design.md`, which
parameterised the analyser and produced the 180+0 human band table this fits
against (`cheat_detection/runs/elo_progression/report_timing_180.md`, 40,000
games / 3.14M moves).

Time controls are `seconds+increment`: **`180+0` means three minutes.**

## The measurement this rests on

100-game self-play arm, shipped defaults, complete games (`--simulate-full`),
seed 910000, `cheat_detection/runs/tc_envelope/sim_180_control.pgn`. Human
figures are the pooled 2300+ band table. Both sides use `cheat_detection`'s
phase definition (`features._phase`: ply < 16 is opening, non-pawn material
≤ 13 is endgame).

| phase | bot | human | ratio | bot mix | human mix |
|---|---|---|---|---|---|
| opening | 1.124s | 1.519s | **0.74×** | 16.2% | 19.8% |
| middlegame | 3.528s | 4.028s | 0.88× | 64.0% | 65.4% |
| endgame | 1.304s | 1.474s | 0.88× | 19.7% | 14.8% |
| **mean emt** | **2.699s** | **3.283s** | **0.82×** | | |

Games complete normally: 98.6 plies mean, 48/38/14 W/L/D, no flagging
pathology.

⚠️ **Use `features._phase`, not `common/board_information.phase_of_game`.**
They are unrelated rules — the engine's counts minor/major pieces and tests
back-rank sparseness. Mixing them produced a bot endgame share of 35% against
the human 14.8% and an apparent phase-mix crisis that does not exist. The two
definitions are both legitimate for their own callers; only one is comparable
to the band table.

### Why the analytic prediction was wrong, and what that costs

From the requested-time formulas alone, the bot's opening/midgame ratio looked
like it collapsed **0.212 → 0.075** between 60+0 and 180+0, against a human
0.374 → 0.377. That is a ~4× defect. Measured, the realised ratio is **0.319 vs
0.377** — a 15% shortfall.

The gap between the two is the compute floor plus the pacing machinery
(`game_pace_sf`, the mood envelopes, the intuition gate), which together absorb
most of the requested-time distortion. A predicted 9.3s midgame realises as
3.528s.

The lesson, and the reason this spec fits against measurements only: **the
requested-time algebra is not evidence about behaviour.** Specifying against it
would have designed a 4× correction for a 15% problem.

## The two defects, and why they are separable

**1. Level.** The bot is ~18% too fast overall, and evenly so outside the
opening — midgame and endgame are both 0.88×. `quickness` owns this. Its fitted
mapping `mean_emt = 0.1964*quickness + 0.6234` is 60+0-only and has no meaning
at 180s; the intercept in particular is a bullet artefact.

**2. Shape.** The opening is 0.74× where the other phases are 0.88×, giving
open/mid 0.319 against the human 0.377.

These are orthogonal, which is what makes the design tractable: **`quickness`
is a global scale, so the phase ratio is invariant under it.** Raising the
level lifts the opening from 1.124s to roughly 1.37s but leaves the ratio at
0.319. Only the envelope shape moves the ratio. Neither knob can do the other's
job, so neither fit can absorb the other's error.

Humans hold open/mid at **0.374 (60+0) → 0.377 (180+0)** across a 3× clock
change. That near-invariance over 3.14M moves is the target for defect 2, and
it is a measured constant rather than a hyperparameter to search.

## Design

### Part 1 — refit `quickness` at 180+0

Same method as the 60+0 Phase A fit: sweep `quickness` across arms, regress
realised mean emt on it, invert onto the per-band targets. Three or four arms
spanning the range, 100 games each, complete games, pure self-play.

Output: a 180+0 row for `common/strength_profiles.py`, whose `_PROFILES` becomes
keyed by `(initial_time, rating)` and whose `resolve` gains an `initial_time`
argument. The 60+0 rows are unchanged.

Per-band targets come from the 180+0 band table's Mean emt column, over the
bands the fit actually uses: 3.23s (2100-2299), 3.17, 3.16, 3.09, 3.05, and
2.99s (2700-2799). The 2800+ figure of 2.42s is deliberately **not** a target —
see the exclusion below.

⚠️ **Exclude the 2800+ band from the fit.** It has 33,265 moves against the
bottom band's 485,633 and is anomalous on every column simultaneously (instant
rate 0.216→0.315, ACPL 79.8→48.5, t1 0.455→0.499, emt 2.99→2.42 in a single
band step where the preceding six move by 0.24s in total). It is a handful of
individuals, not a population trend. Anchoring the top of the dial on it would
be wrong in a way that looks fine. Use 2100-2799, and say so in the profile
table's docstring.

### Part 2 — a TC-keyed opening envelope

`engine_components/decision_logic.py:195` currently hardcodes the phase
envelopes inline, including an ad-hoc clock branch:

```python
if game_phase == "opening":
    base_time = (base_time ** 0.2)/2
elif game_phase == "midgame":
    base_time *= 1.7 if self_initial_time > 60 else 1.4
else:
    base_time *= 0.7
```

Lift these into `common/tc_profiles.py`, a table keyed by initial time and
resolved by `resolve_tc(initial_time)`, sitting beside `strength_profiles.py`.
The existing `1.7`/`1.4` branch folds into it as data.

**The 60+0 row is pinned to today's behaviour exactly** — the legacy
`(base_time ** 0.2)/2` opening form, `*1.4` midgame, `*0.7` endgame. Not a
re-derivation, not an approximation: the same arithmetic. A unit test asserts
the pinned row reproduces the current formula across a range of `quickness` and
clock values, and the engine parity harness must pass unchanged.

**The 180+0 row is fitted** so the realised open/mid ratio reaches 0.377, with
the midgame and endgame multipliers left at their current values (`*1.7`,
`*0.7`) since both phases already measure 0.88× — the same as each other, so
their relative shape is right and only the level is off, which Part 1 owns.

A single global constant cannot serve both rows. To reproduce today's 60+0
opening the coefficient is 0.297 of base. The 180+0 coefficient is whatever the
fit lands on — an analytic starting point is ~0.151, but that number is derived
from requested times and this spec has just shown those do not predict realised
behaviour, so treat it as where the first arm starts, not as the answer. The two
rows differ because the midgame multiplier differs *and* because the compute
floor binds differently at each control. Per-TC rows are therefore not a
convenience, they are required by the data.

**Resolution between and beyond keyed rows:** `resolve_tc` returns the nearest
keyed row and the row records which clock it was fitted at. Only 60 and 180 are
fitted; anything else resolves to the nearer of the two and is **not** a claim
about that control. The engine logs the resolved row so a run at, say, 300+0 is
visibly using an unfitted profile rather than silently doing so.

### Fitting is iterative, not analytic

The requested→realised mapping is nonlinear (floor, moods, gate, pace draw), so
neither part can be solved in closed form. Both are measure-fit-remeasure loops.
Budget 2-3 arms per part, and **fit Part 1 before Part 2** — the level fit lifts
the opening too, and the envelope correction must be fitted against the residual
rather than against today's gap.

## Out of scope

- **The endgame envelope.** 0.88×, identical to the midgame. There is no
  phase-specific endgame defect; Part 1 covers it.
- **300+0 and 600+0.** Unmeasured. The analytic ratio continues to fall
  (0.053, 0.033) but this spec has just demonstrated that the analytic ratio is
  not evidence about realised behaviour.
- **`high_range_multiplier`** (`decision_logic.py:276`, `T**0.35/60**0.35`) —
  another TC-dependent term, applied inside the mood branches rather than the
  phase envelopes. Untouched here; flagged so a later fit knows it exists.
- **The accuracy levers.** Read 2 of the band-table comparison showed the t1
  spread doubles at 180+0 (+25.0% vs bullet's +14.5%), which makes `DIFFICULTY`
  and `eval_noise_scale` worth sweeping at this control despite both failing at
  1+0. That is a separate spec; this one is pacing only.

## Testing

1. **Engine parity harness passes unchanged, without `--record`.** The 60+0 path
   must be byte-identical. If parity moves, the pinned row is wrong.
2. **Unit test: the pinned 60+0 row reproduces the legacy formula** across a
   grid of `quickness` and clock values — this is the regression guarantee for
   every existing 1+0 calibration, and the same shape of guarantee that Phase 0
   used (0 of 990,068 leaves).
3. **Unit test: `resolve_tc` returns the nearest keyed row** and reports which
   clock it was fitted at.
4. **Unit test: `strength_profiles.resolve` honours precedence** — explicit
   knob argument > `target_rating` > module constant — now with `initial_time`
   in the key, and returns the effective rating *and* the effective clock.
5. **Validation arm**: 100 games at 180+0, complete games, confirming realised
   open/mid ≈ 0.377 and mean emt ≈ 3.283s.
6. **Guard, on `--simulate-full` arms only**: `blunder_rate_timepressure`
   against the *complete* 180+0 baseline (`blitz_3plus0_2300_plus.json`, human
   0.0674). ⚠️ It cannot be measured against the truncation-matched baseline —
   the simulator's cutoff is `0.25*T` and `time_pressure_secs` is `T/6`, so a
   truncated corpus has `n=0` time-pressure moves at every time control. An
   `n/a` there is structural absence, not a clean result.

## Open questions

- Does raising the opening time interact with the opening-book fast path
  (`OPENING_BOOK_FAST_PATH`, ships off)? The fast path prices a book hit as an
  obvious move, so a longer opening envelope may change what it saves. Both
  levers target the opening; check them together rather than assuming
  independence.
- The 180+0 human opening is 1.519s against a bullet 0.561s — a 2.71× scaling
  where the clock scales 3×. Close to proportional but not exactly, and two
  points cannot distinguish "proportional" from "slightly sublinear". A third
  control would settle the functional form, which is the main argument for
  eventually measuring 600+0.
