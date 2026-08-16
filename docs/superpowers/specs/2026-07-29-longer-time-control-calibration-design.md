# Extending strength calibration to longer time controls: measurement phase

Date: 2026-07-29
Status: **Phase 0 shipped 2026-07-30** (merges `449c194`, `8acf8b6`); Phase 1
in progress.
Scope: Phase 0 (TC-parameterise the analyser) and Phase 1 (the 3+0 human band
table) only. Bot-side work is deliberately out of scope — see *Why this stops
at measurement*.

## Implementation notes (added 2026-07-30)

Phase 0 landed as planned, with the exactness guarantee verified end-to-end:
every numeric leaf of a report generated at the branch point matched one
generated at head — **0 of 990,068 differ**, including `long_think_rate` and
`blunder_rate_timepressure`. Plan:
`docs/superpowers/plans/2026-07-29-tc-parameterised-calibration.md`.

Two defects that review caught, both worth remembering because both were
reported as harmless:

- The shared `--tc` default silently switched `analyze.py run`'s *fetch* from
  all-clocks to 60+0-only, because `run`'s `--tc` also feeds
  `fetch_lichess.fetch_user_games`, where `None` means unfiltered.
- The corpus-clock guard shipped **inert on the parallel path** — wired into
  `pipeline.iter_units` and `elo_progression._collect_sequential` but not
  `parallel._worker`, so it enforced nothing at `--workers > 1`, the documented
  normal usage. Behaviour that depends on `--workers` is worse than no guard.

Three prerequisites the final review surfaced were done before Phase 1
(`8acf8b6`): baselines now record `initial_time` and `cmd_report` warns on
disagreement; `--tc` added to `player_dispersion`, `mistake_impact`,
`bucket_diagnostic`, `emt_buckets`; `parse_tc_seconds` moved to `pgn_loader`.

### Corpus feasibility: the top bands are the binding constraint

The dump does not need downloading in full — `database.lichess.org` serves byte
ranges, so a prefix can be streamed through `fetch_corpus` with nothing large
landing on disk. On the 2026-05 dump, **0.88% of games qualify** for
180+0/2300+ (40,000 kept from 4.57M scanned), and the full 40,000 quota filled
in roughly 90 minutes from a ~5 GB range — a 119 MB corpus against bullet's
78 MB / 30,000 games.

⚠️ **Do not size the fetch from its opening minutes.** Throughput ramps by more
than an order of magnitude after the first few minutes; an early reading of
~64 KB/s projected 20 hours for what finished in 90. An intermediate plan to
stop at 8,000 games was made on that bad extrapolation and was unnecessary.
Also note `fetch_corpus` exits on quota and closes the pipe, so the pipeline
reports `SIGPIPE` (141) on success.

More importantly, the *band distribution* is far more scarce at the top at 3+0
than at bullet. A 40-game smoke slice gave 649 moves in 2100-2299 but only 41
in 2700-2799 and 25 in 2800+ (below the `min_n` gate). Top-rated players play
much more bullet than 3+0, so **the 2800+ band, not the corpus as a whole, sets
the precision of read 1** — the mean-emt span is an endpoint comparison and its
error bar is dominated by its thinnest end. Judge the span using the trend
across all seven bands rather than the two endpoints alone, and if the result
is marginal, say so and fetch longer rather than reading a 3-se difference as
confirmation. This project has already retracted two conclusions built on
error bars that were assumed rather than measured.

Follows the 60+0 work in `2026-07-27-strength-dial-design.md` and
`2026-07-28-instant-move-channel-design.md`. Everything those specs measured is
1+0-specific; this begins the second calibration point.

Throughout, time controls are written as `seconds+increment`, the units
`simulation.run --tc` and `fetch_corpus --tc` take. **`180+0` means three
minutes, not three hours.** Where prose uses the conventional chess notation
(`3+0`) it means the same control.

## Motivating hypothesis

The repo owner's framing: *the longer the time control, the less `quickness`
becomes a strength factor, and the more strength shifts into other levers.*

This is very likely right, but the mechanism needs stating precisely, because
the obvious reading of it is wrong in a way that would misdirect the work.

**`quickness` does not weaken as a lever.** `engine_components/decision_logic.py:168`
already scales pacing with the clock:

```python
base_time = max(engine.quickness * self_initial_time**1.1
                / (100 + self_initial_time**0.7), 0.1)
```

At `quickness=2.5` that gives 1.92s at 60+0, 5.48s at 180+0, 15.1s at 600+0 —
roughly 3% of the clock throughout. The knob's *mechanical* range in fact
improves at longer controls: at 60+0 it is clipped from below by the ~0.85s
per-move compute floor across the opening and endgame, and at 180+0 the floor
binds in fewer places.

**What weakens is the correlation between pace and rating.** At one minute the
clock is the scarce resource, so playing fast *is* playing strong — which is
why the fitted `mean_emt = 0.1964*quickness + 0.6234` spans the entire
2200–2850 band table and became the dial's only calibrated mapping. At three
minutes, both a 2200 and a 2800 have time to spare; what separates them should
move out of the clock and into move quality.

That is an empirical claim about *humans*, not about the bot, and it is
measurable from a corpus alone. Phase 1 measures it.

## What the code already reveals

Two findings fell out of reading the pacing path while scoping this, both of
which shape the later (unspecced) work:

### The opening envelope is bullet-shaped at every time control

`decision_logic.py:195` compresses opening think time with `(base_time ** 0.2)/2`:

| clock | midgame base | opening think |
|---|---|---|
| 60s | 1.92s | **0.57s** |
| 180s | 5.48s | **0.70s** |
| 600s | 15.1s | **0.86s** |

The clock grows 10× and the opening think time grows 1.5×. That envelope was
fitted to bullet, where it is correct — CLAUDE.md records the opening as
structurally suppressed and the requested time already below the compute floor
— but it hardcodes "the opening is memorised" into every time control. A 2600
at three minutes spends real time on the first move out of book. The endgame's
flat `*0.7` has the same shape of problem.

So the phase envelopes, not `quickness`, are the pacing code that is most
plausibly TC-wrong. They currently have no owner in the control surface.

### `DIFFICULTY` plausibly unfreezes

CLAUDE.md freezes `DIFFICULTY` at 3 for 1+0 on two grounds: compute cost, and
that it drives base breadth *and* the noise floor together so it cannot be
aimed at a single feature. The compute objection weakens as the control
lengthens. The coupling objection may even invert into an advantage if Phase 1
shows several accuracy features moving together across bands.

Likewise `eval_noise_scale`, measured non-monotone and declared a dead lever
for t1 at 1+0 (span 0.0225 against a needed 0.0504), deserves re-testing rather
than inheriting its bullet verdict — its non-monotonicity may itself be a
compute-floor artefact.

Neither is acted on in this spec. Both are recorded so the later design round
starts from them.

## Why this stops at measurement

The band table is the experiment the whole hypothesis rests on, and its result
changes what the later phases should be. If the emt span across bands does
*not* flatten at 180+0, then an envelope refit and a `DIFFICULTY` sweep are
aimed at a decoupling that does not exist, and `quickness` remains the primary
axis at three minutes exactly as at one.

The 1+0 work produced two retractions from reasoning ahead of the error bars
(the Phase B breadth trend, and Phase C's "t1 and blunder both rose"). Both
came from committing to an interpretation before the measurement could carry
it. Stopping here is the same lesson applied earlier.

## Phase 0 — TC-parameterise the analyser

`cheat_detection/` currently hardcodes 60+0 in three places. Nothing else can
run until they are parameterised.

### Derived thresholds

Add `initial_time: float = 60.0` to `AnalysisConfig`, settable from a `--tc`
flag, and derive the clock-dependent thresholds from it:

| threshold | today | derivation | at 180+0 |
|---|---|---|---|
| `long_think_secs` | 2.0 | `initial_time / 30` | 6.0 |
| `time_pressure_secs` | 10.0 | `initial_time / 6` | 30.0 |
| `instant_move_secs` | 1.0 | **stays absolute** | 1.0 |

The `/30` is already documented as the intent at `config.py:40` ("Set at
initial_time/30 … a different time control needs this rescaled"). The `/6` was
derived backwards from the shipped constant and lands on 10.0 exactly, which is
reasonable evidence that a fraction is the right reading of "time pressure"
rather than a coincidence.

**Both derivations reproduce the shipped constants precisely at
`initial_time=60`.** That exactness is the regression guarantee: Phase 0 is
provably inert for every existing bullet baseline, report and run.

`instant_move_secs` stays absolute *by design*. It is a human motor-and-decision
floor — the time to see, decide and move a mouse — not a share of the clock. A
one-second move means the same thing at any control. Scaling it would make
"instant" mean 6 seconds at 3+0, which is not an instant move by any reading.

`blunder_wc_loss`, `ambiguity_wc_window`, `opening_plies` and `endgame_npm` are
TC-invariant and are not touched. (`ambiguity_wc_window` additionally must stay
equal to `compute_ambiguity`'s window in `engine_components/`, asserted by
`testing/engine_components/test_ambiguity.py`.)

### Other changes

- `elo_progression.py:184` snapshots `LONG_THINK_SECS = AnalysisConfig().long_think_secs`
  at module level, which would silently ignore a configured `initial_time`.
  Read it from the cfg that is already threaded through `_stats`. Same for
  `INSTANT_SECS` at line 183, for consistency, even though its value does not
  change.
- Add a guard that errors when PGN `TimeControl` headers disagree with the
  configured `initial_time`. The corpus policy in CLAUDE.md already says to pin
  one exact clock ("mixing e.g. 30+0 with 60+0 muddies every timing feature");
  this turns a convention into something enforced, and it is the specific
  mistake this whole workstream is most exposed to.
- Baselines get a per-TC filename: `baselines/blitz_3plus0_2300_plus.json`.

The SQLite eval cache needs no change. It is FEN-and-depth keyed and therefore
TC-independent, so the 180+0 corpus reuses bullet's cached evaluations wherever
positions coincide — most opening positions, and a fair share of common endings.

### Testing

1. A test asserting the derived thresholds equal `(2.0, 10.0)` exactly at
   `initial_time=60`. This is the inertness claim above, and it is the only
   thing standing between this refactor and every bullet result in the repo.
2. A test that the header guard fires on a TC-mixed PGN and passes on a pinned one.
3. Re-run an existing bullet report and confirm the numbers are unchanged.
4. `venv/bin/ruff check` clean on changed files.

Engine parity is not affected — nothing in `engine.py` or `engine_components/`
is touched by this phase.

## Phase 1 — the 3+0 human band table

### Prerequisite: the corpus

There is no 180+0 corpus and no local Lichess dump. `fetch_corpus` reads a PGN
stream (typically `zstd -dc` of a monthly dump), so a dump download is a
prerequisite and is the long pole of this phase — larger than the analysis it
feeds.

```
zstd -dc <dump>.pgn.zst | venv/bin/python -m cheat_detection.fetch_corpus \
    --tc 180+0 --any-player --band 2300 + <count> \
    --out cheat_detection/corpora/blitz_3plus0.pgn
```

`--band` takes `MIN MAX COUNT` with `+` for unbounded, and in band mode `--out`
is a *stem* the per-band filename derives from — so the above writes
`blitz_3plus0_2300_+.pgn`. Corpus policy matches bullet: 2300+ meaning *at
least one* player in band (`--any-player`), with the off-band opponent excluded
at analysis time via `--rating 2300 9999`.

### The run

`elo_progression` at `initial_time=180`, output to
`cheat_detection/runs/elo_progression/report_timing_180.md`. Add it to the
gitignore negations beside `report_timing.md` and `report_longthink.md` — same
justification, which the existing `cheat_detection/.gitignore` comment already
states: the band tables are the calibration target, they are small, and they
are not reliably reproducible without a gitignored corpus.

Cost is roughly the bullet run's ~65 min plus whatever the dump download takes,
less the fraction the shared eval cache absorbs.

### Pre-committed reads

These are written before the data exists so that a marginal result cannot be
talked into agreeing with the hypothesis afterwards. Bullet reference values
are from `report_timing.md`, overall table.

**1. Mean emt span across bands.** Bullet runs 1.26s (2100-2299) to 1.00s
(2800+): a 21% proportional drop. *Predicted: materially smaller at 180+0.*

> If the proportional span at 180+0 is greater than or equal to bullet's, **the
> hypothesis is wrong.** Pace remains a rating signal at three minutes,
> `quickness` stays the primary axis, and the follow-up work is a re-fit of the
> existing dial at a second control rather than a search for new levers.

**2. Top-1 match span across bands.** Bullet's adjacent bands differ by ~0.005,
which is precisely what caps the dial at ~200-300 Elo resolution and what made
every accuracy lever unusable at 1+0. *Predicted: larger at 180+0.*

> This is the load-bearing one for the follow-up. A wider t1 span is what would
> make `DIFFICULTY` and `eval_noise_scale` worth sweeping at 3+0 despite both
> failing at 1+0.

**3. Per-phase mean emt.** The opening-to-midgame emt ratio per band. The bot's
`(base_time**0.2)/2` makes its opening near-invariant across controls; humans
are predicted not to be. This measurement is what prices an envelope refit, and
it is the only one of the three that speaks directly to the phase-envelope
finding above.

Report all three as a table against the bullet figures, so the comparison is
the artefact rather than a claim in prose.

### Explicitly not in Phase 1

No simulation arms, no bot-side sweeps, no envelope changes, no `DIFFICULTY`
work, and no changes to `common/strength_profiles.py`. The dial keeps its
current docstring, which already scopes itself to 60+0 and says longer controls
"need their own band table and their own fit" — Phase 1 produces exactly that
band table and nothing more.

## The TC profile table, designed but not built

The agreed destination for per-time-control pacing constants is a keyed table
(`common/tc_profiles.py`), resolving from `initial_time`, sitting beside
`strength_profiles.py`. The existing ad-hoc branch at `decision_logic.py:195`
(`if self_initial_time > 60: base_time *= 1.7 else: 1.4`) folds into it, and
the 60+0 row is pinned as data so that later-TC work cannot move the bullet
calibration by accident — with the engine parity harness guarding that pin.

```python
TC_PROFILES = {
    60:  {"opening": ("pow", 0.2, 0.5), "midgame": 1.4, "endgame": 0.7},
    180: {...},   # populated from Phase 2, unmeasured today
}
resolve_tc(initial_time) -> nearest keyed row
```

It is **not built in this spec**, because there is nothing measured to put in
the 180 row. Building it now would be a refactor carrying no data, and it would
have to be redesigned once Phase 1 says which constants actually differ.

## Open questions for the next design round

- Does `time_pressure_secs = initial_time/6` survive contact with the data? The
  scramble may be partly *absolute* — 10 seconds is roughly where humans stop
  calculating regardless of what the clock started at — in which case the right
  form is a floor, e.g. `max(10.0, initial_time/6)`.
- Do the phase envelopes need per-TC constants, or a different functional form?
  The `**0.2` compression is not a constant that can be re-fitted into
  correctness if the shape itself is bullet-specific.
- Does `simulation`'s `CLOCK_THRESHOLD_FRACTION = 0.25` truncation convention
  transfer? It is already a fraction, so it should, but the truncation-matched
  baseline trap documented in `docs/position-conditioned-human-likeness.md`
  applies with full force at any new control.
- Sim cost at 180+0 is expected to be close to bullet's, since every Stockfish
  call in the engine is depth-bound rather than time-bound (`state.py:179`
  depth 10 capped at 20ms; `state.py:254` fixed `SHARPNESS_SCAN_DEPTH`) and
  `client_model.py` charges simulated clock from a phase-keyed latency
  distribution. Only the extra plies per game cost more. This should be
  confirmed on a small arm before any Phase 2 sweep is sized.
