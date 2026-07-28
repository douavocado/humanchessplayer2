# Position-conditioned human-likeness: where the bot actually diverges

Status: **diagnostic complete. Superseded in two places by later measurement
(2026-07-28) -- read those corrections before acting on anything here.**

1. **"This rules out every strength lever"** (Finding 1) overstated the case.
   It is true of the *strength* knobs it was about -- `eval_noise_scale` was
   later measured non-monotone with no usable range, and breadth's blunder
   effect is not measurable at 150 games/arm. But a lever outside that class
   does move quiet-position agreement: the ordering of the re-evaluation draw
   in `get_human_move`, shipped as `REEVAL_ORDER = "human"` (12b322d), worth
   ~+0.013 aggregate t1 and replicated across three seeds. The draw randomly
   disqualifies ~2.3 of 8.46 candidates in quiet positions (active in 63.2% of
   them), which is why the deficit was concentrated there.
2. **The timing analysis here covers only the fast tail.** `long_think_rate`
   (emt > 2s) did not exist when this was written. It moves in the *opposite*
   direction with rating -- 0.122 at 2100-2299 down to 0.088 at 2800+ -- so
   instant rate and long-think rate are independent axes, and the bot is low on
   both. See `cheat_detection/runs/elo_progression/report_longthink.md`.

Also note every human comparison below is against the **pooled** 2300+ mean.
That is the right reference for the bucket contrasts it draws, but it is not a
target: judging the bot's mean emt against the pooled 1.217 made a value that
sits at ~2700 on the band table look like a shortfall.

Two follow-up workstreams are identified at the end; both have since been
acted on -- see `docs/superpowers/specs/`.

Scope warning, up front: **every number here is 1+0 bullet.** See "Time-control
scope" below before generalising any of it.

## The problem

Aggregate human-likeness features collapse a whole game into one vector, which
answers "is the bot un-human?" but not "*where*?". A bot can look wrong on
`t1_rate` for opposite reasons — playing too well in critical positions, or
playing oddly in dull ones — and the aggregate cannot tell those apart. They
need opposite fixes.

Conditioning the comparison on position character (`cheat_detection/
bucket_diagnostic.py`, `elo_progression.py`) separates them, and the answer for
this bot turned out to be the second one, in a way that rules out every
strength lever we have.

## Two methodological traps found along the way

Both produced confident, wrong conclusions before being caught. Anyone
repeating this analysis will hit them.

### 1. Truncated bot games vs complete human games

The simulator stops games at `CLOCK_THRESHOLD_FRACTION * initial_time` (15s for
60+0) and adjudicates from a Stockfish eval, so simulated PGNs contain **no
sub-15s moves**. Human corpus games run to their natural end. The scramble is
where humans do most of their damage — human `acpl_timepressure` is ~544 — so
comparing the two directly deletes the worst human moves and none of the bot's
equivalent phase.

Measured effect, identical bot code, only the reference changing:

| feature | vs complete human games | vs truncation-matched |
|---|---|---|
| acpl | 51.0 vs human 144.2 | 51.0 vs human 90.0 |
| acpl_endgame | 21.7 vs human 294.4 | — |
| blunder_rate | 0.031 vs human 0.054 | 0.031 vs human 0.054→matched |
| mean\|player_z\| | 1.72 (8 outliers) | 1.28 (4 outliers) |

The fix is to apply the simulator's own rule to the human corpus, importing
`_find_cutoff_ply` from `simulation/adjudicate_result.py` rather than
reimplementing it. Validation that the rule is the right one: applied to bot
PGNs it reproduces their exact length on 150/150 games, and it splits the human
corpus into 70% crossed / 30% ended-early against the bot's 71% / 29%.

Truncation costs almost nothing in sample size (51,286 units vs 51,300) because
it removes plies, not games. It does make `acpl_timepressure` and
`blunder_rate_timepressure` undefined on **both** sides — those two features are
simply unmeasurable in a truncated comparison, which means **the flag-race /
scramble behaviour is not tested by any of this.**

### 2. Win-probability saturation

`blunder_rate` and `mean_wc_loss` are defined as win-*probability* drops. At
±600cp the win probability is already saturated, so an additional 300cp error
costs almost nothing and cannot register. Any bot spending most of its time in
decided positions therefore looks like it blunders less — even when it is
playing worse.

Caught by an inverted result: against a ~115 Elo *stronger* opponent the bot's
ACPL went **up** (49.7 → 51.0, i.e. worse) while its blunder rate went **down**
(0.0380 → 0.0306). ACPL has no ceiling; the win-probability metrics do.

Consequence for any A/B where the variant is meaningfully stronger: **the score
rate must be equalised or the error metrics are not comparable.** In the breadth
sweep, score ran 0.337 → 0.827 across arms, so its blunder-rate ladder was
partly artefact. Re-running as pure self-play (both sides same config, score 0.5
by construction) removes it, and the finding changed materially — see below.

## Finding 1: the match-rate deficit is entirely in quiet positions

Current default (both breadth bonuses 0), self-play, vs 2500–2800 humans,
truncation-matched:

| bucket | share of moves | bot t1 | human t1 | Δ | bot t2 | human t2 | Δ |
|---|---|---|---|---|---|---|---|
| **Quiet** (sharpness <0.10) | ~74% | 0.262 | 0.300 | **−0.038** | 0.416 | 0.485 | **−0.069** |
| Moderate (0.10–0.25) | ~16% | 0.536 | 0.513 | +0.023 | 0.744 | 0.699 | +0.045 |
| Sharp, forced (amb==1) | ~7% | 0.857 | 0.790 | +0.067 | 0.925 | 0.896 | +0.029 |
| Sharp, messy (amb≥2) | ~3% | 0.555 | 0.517 | +0.038 | 0.855 | 0.867 | −0.012 |

Weighted, quiet contributes −0.028 and the other three +0.010, which fully
accounts for the aggregate gap (bot 0.352 vs ~0.370). Replicated across two bot
configurations and two independent game sets.

**The bot's profile is inverted relative to humans.** Its quiet t1 of 0.262 sits
*below the 2100–2299 band* (0.2786); its sharp-forced 0.857 sits *above 2800+*
(0.8011). Superhuman where tactics decide, sub-2100 where nothing is at stake.

The decisive detail is what does *not* differ. In the quiet bucket the bot's
blunder rate (0.023 vs 0.024, p=0.51) and win-chance loss (0.029 vs 0.029,
p=0.54) are already human. So in quiet positions the bot is not playing *worse*,
it is playing *differently* — choosing non-engine moves that lose nothing.

**This rules out every strength lever.** Human t1 improvement from 2100–2299 to
2800+ is smallest in exactly this bucket: quiet **+3.7pp**, versus moderate
+7.5pp and sharp-forced +7.2pp. Seven hundred Elo of human improvement buys 3.7
points there. A lever that makes the bot stronger cannot close a gap that
strength does not close in humans, and every such lever also lowers error rates
that are already correct. This is a move-*choice distribution* problem in the NN
at low sharpness, not a search problem.

## Finding 2: human speed-up concentrates in forced tactical positions

Mean elapsed move time and instant rate (emt < 1s), 2100–2299 → 2800+, from
`elo_progression` over 2.28M moves:

| context | mean emt | Δ | instant rate | Δ |
|---|---|---|---|---|
| **Sharp, forced** | 1.22 → 0.87 | **−29%** | .332 → **.464** | **+13.2pp** |
| Sharp, messy | 1.24 → 0.91 | −27% | .293 → .407 | +11.4pp |
| Forced/tactical (eff_mob<15) | 1.51 → 1.17 | −23% | .239 → .333 | +9.4pp |
| Moderate | 1.60 → 1.27 | −21% | .209 → .284 | +7.6pp |
| Quiet | 1.19 → 0.96 | −19% | .329 → .390 | +6.1pp |
| Open (eff_mob≥15) | 1.12 → 0.91 | −19% | .350 → .406 | +5.6pp |
| endgame | 0.65 → 0.51 | −22% | .496 → .578 | +8.3pp |
| middlegame | 1.61 → 1.28 | −20% | .188 → .265 | +7.7pp |
| opening | 0.55 → 0.48 | −13% | .565 → .599 | +3.4pp |

**The signature is instant recognition of forced shots.** At 2100–2299 the
instant rate in sharp-forced (.332) and quiet (.329) positions is identical. By
2800+ they have diverged: **.464 vs .390**. Strong players are not uniformly
faster — they have stopped *calculating* positions with one right answer. A
uniform `QUICKNESS` change cannot reproduce that shape.

These are within-bucket, band-to-band contrasts, so the obvious confound (sharp
positions clustering in scrambles, where everything is fast) is controlled: it
would shift all bands equally.

Against this the bot shows:

| | bot | human 2300+ |
|---|---|---|
| instant_move_rate | 0.2253 | 0.2965 |
| instant_in_sharp_rate | 0.3201 | 0.3500 |
| corr_time_sharpness | **+0.0558** | +0.0374 |

`corr_time_sharpness` is the informative one: the bot's think time tracks
sharpness *more* strongly than humans'. It slows down in critical positions
where strong humans speed up. The intuition gate (`game_snap_gate`) pulls the
right way but is keyed on sharpness alone, so it treats forced and messy sharp
positions identically — while the human data says those move in *opposite*
directions.

Note also that in sharp-forced positions the bot is simultaneously **too slow
and too accurate** (t1 0.857 vs 2800+ humans' 0.801). One change addresses both:
snapping there spends accuracy the bot does not need and buys the instant rate
it lacks.

## Breadth levers (settled, no follow-up)

Head-to-head adjudicated Elo, 150 games/arm, and self-play human-likeness:

| lever | Elo | mean\|player_z\| (self-play, vs 2300+) |
|---|---|---|
| default (0/0) | — | 1.00 |
| midgame +1 | +115 | 1.12 |
| midgame +2 | +158 | 1.34 |
| midgame +3 | +271 | 1.32 |
| midgame +5 | +271 | 1.39 |
| opening +2 | −0 | — |
| opening +5 | −40 | — |

Midgame breadth saturates at 3; opening breadth is worthless or harmful. What
breadth actually controls is the **blunder rate** (0.0380 → 0.0169 from 0 to
+5), and human-likeness degrades monotonically with it. Match rates are *not*
improved by it — the apparent improvement in the sweep data was the
weaker-opponent confound, and it vanishes under self-play (t1 flat at
0.352/0.353/0.355/0.344/0.363 across the five levels).

Conclusion: **the shipped default is right for 1+0.** `midgame +1` is arguable
at +115 Elo for a human-likeness cost of +0.12, which is at the documented ±0.1
noise floor — but there is no upside beyond raw strength.

## Time-control scope

**All of the above is 1+0 bullet**, and most of it should be expected to differ
elsewhere:

- **The truncation rule is time-control relative** (`0.25 × initial_time`), so a
  3+2 or 10+0 corpus needs its own truncated baseline. Reusing the 60+0 one is
  exactly the mismatch documented above.
- **Timing patterns are the most time-control-specific result here.** The
  absolute values (mean emt ~1.0–1.6s, instant rate 0.19–0.46) are artefacts of
  a 60-second budget. Whether the *shape* — speed-up concentrated in forced
  tactical positions — survives at longer controls is an open question. At 10+0
  there is time to verify a tactic rather than trust recognition, so the elite
  signature may weaken or invert.
- **Rating progressions may differ in kind, not just degree.** Bullet rewards
  pattern recognition and clock economy; longer controls reward calculation
  depth. The finding that human improvement is *smallest* in quiet positions is
  plausibly a bullet-specific result — with more time, quiet-position accuracy
  may be exactly where stronger players separate.
- **Phase weighting shifts.** Bullet games are truncated by the clock far more
  often than by the board; endgame samples here are thin and scramble-dominated.
- The **breadth** conclusions are Elo measurements at 60+0 with the engine's
  compute floor in play; at longer controls the search has room to use breadth
  differently.

Before applying any of this to another time control, rebuild the corpus, the
truncated baseline and the dispersion file at that exact clock — mixing clocks
muddies every timing feature (see `cheat_detection/README.md`).

## Two workstreams this opens

1. **Raise quiet-position t1/t2** (0.262→~0.300, 0.416→~0.485) *without*
   lowering error rates, which are already human. Quiet is ~74% of moves, so any
   change dominates every aggregate — go gently and re-measure. Target is the NN
   candidate distribution at low sharpness, not search.
2. **Condition the snap gate on ambiguity, not just sharpness** — snap harder
   when `ambiguity == 1`, less when `ambiguity >= 2`. Validate on Elo as well as
   human-likeness: in bullet, speed converts to strength via *clock economy*
   (time banked for genuinely hard positions), not via better moves, so the two
   may disagree.

## Reproducing

```bash
# truncate the human corpus with the simulator's own rule, then build both refs
venv/bin/python -m cheat_detection.analyze baseline \
    --pgn cheat_detection/corpora/bullet_1plus0_2300_plus__trunc15.pgn \
    --rating 2300 9999 --workers 8 --threads 2 \
    --out cheat_detection/baselines/bullet_1plus0_2300_plus__trunc15.json
venv/bin/python -m cheat_detection.player_dispersion \
    --pgn cheat_detection/corpora/bullet_1plus0_2300_plus__trunc15.pgn \
    --rating 2500 2800 \
    --out cheat_detection/baselines/dispersion_2500_2800__trunc15.json

# pure self-play at a given lever setting (score 0.5 => no saturation skew)
venv/bin/python -m simulation.run --games 150 --tc 60+0 --workers 6 \
    --rating 2450 --midgame-breadth-bonus 0 --seed 600000 --out <pgn>

# conditioned bot-vs-human comparison
venv/bin/python -m cheat_detection.bucket_diagnostic \
    --bot-pgn <pgn> --bot-player SimBotWhite SimBotBlack \
    --human-pgn cheat_detection/corpora/bullet_1plus0_2300_plus__trunc15.pgn \
    --human-rating 2500 2800 --out-md <md>

# human rating progression, incl. timing columns (~3.5h, Lucas pass is uncached)
venv/bin/python -u -m cheat_detection.elo_progression \
    --pgn cheat_detection/corpora/bullet_1plus0_2300_plus.pgn \
    --workers 6 --out-md cheat_detection/runs/elo_progression/report_timing.md
```

`elo_progression` must run on the **untruncated** corpus: it is a human-only
analysis, and the interesting timing behaviour lives in the phase truncation
deletes. Use `-u`, or progress output sits in the stdout buffer for hours.

Supporting artifacts (PGNs, baselines, reports) live under `cheat_detection/
runs/`, `baselines/`, `corpora/` and are **all gitignored** — they do not
survive a machine change. This note is the durable record.
