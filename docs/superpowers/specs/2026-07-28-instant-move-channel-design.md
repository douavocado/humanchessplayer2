# Raising the instant-move rate through the compute-bypass channel

Date: 2026-07-28
Status: **design; not implemented.** Phase B of the strength dial
(`2026-07-27-strength-dial-design.md`).

## Problem

The bot fires 0.252 of its moves in under a second. The human table runs 0.3087
(2100-2299) to 0.3791 (2800+), so it sits **12.6 se below the floor of the
entire band range** -- the single worst feature in the Phase A profile, and the
repo owner has set timing as the dial's primary axis.

Phase A established that **no pacing knob can fix this.** Across five arms
varying `eval_noise_scale` (0.55-0.95) and `quickness` (2.0-3.0), instant rate
stayed within 0.0087 -- flat against a per-arm se of 0.0045 -- while mean move
time moved cleanly across the whole band table. The two timing features have
different mechanisms.

The reason is a floor, and the floor is real. Reconstructing an opening move's
charge:

```
engine asks for think time                         0.513s
think = max(time_take - MOVE_DELAY, compute)
      = max(0.263, 0.556)                        = 0.556s   <- compute wins
+ detection 0.170  + gesture 0.122
TOTAL                                              0.848s
irreducible floor (think set to zero)              0.848s   <- identical
```

The engine's requested think time in the opening is *already below* the compute
floor, so lowering it further changes nothing. `compute_time` covers the live
loop -- screen detection, board recognition, mouse automation -- not bare
engine wall-clock, and the simulator's defaults reflect the machine that
actually plays. (A headless measurement on a dev box gives 0.082s for a book
move and is **not** the comparable number; do not recalibrate against a machine
that is not running the client. Deleting a wrongly-calibrated
`simulation/calibration/compute_time.json` restores the realistic defaults.)

So the only way to produce a sub-1s move is to **bypass engine compute
entirely**. `client_model.py:173` charges the fast paths `wait + gesture` with
no compute term; those paths are premove fires and ponder hits. Everything
below targets that channel.

Phase breakdown of the deficit, which decides where each lever is aimed:

| phase | moves | bot instant | human 2100-2299 | gap |
|---|---|---|---|---|
| **Opening** | 2,627 | 0.362 | 0.565 | **-0.203** |
| Midgame | 6,455 | 0.193 | 0.188 | +0.005 |
| **Endgame** | 4,986 | 0.301 | 0.496 | **-0.194** |

The midgame is already human. It is also the only phase where requested think
time exceeds the compute floor -- the same fact, seen twice.

## Scope

**In:** three levers on the compute-bypass channel, all sweepable, all shipped
inert or at current behaviour.

**Out, deliberately:**

- **Raising premove queue volume.** Documented in CLAUDE.md as twice-proven
  poison: a mean bias above ~1.25, or a midgame full-premove channel even when
  `check_safe_premove`-vetted, re-breaks the time-pressure blunder rate and the
  match rates. Queued premoves fire *after* the opponent deviates and *after*
  the clock has dropped, which is beyond the reach of any queue-time vet. This
  spec adds no premove volume.
- **Re-fitting `quickness` or the phase envelopes.** Phase A showed pacing
  cannot move instant rate, and the opening/endgame envelopes already sit below
  the floor.
- **Recalibrating the latency model.** See above.

## Lever 1: opening-book fast path

The largest single gap, and the only case where compute is genuinely avoidable
rather than merely reallocated: a memorised book move needs no analysis.

Today `consult_book` is called at `engine.py:286`, inside `get_human_move` --
so a book hit already skips the NN move-probability work, but only *after*
`update_info` has unconditionally run `calculate_analytics` (a full-width
multipv scan plus an uncapped depth-12 sharpness scan). The move that needs the
least thought pays for the most.

**Design:** consult the book *before* the expensive scans. On a hit, return the
book move with a short think time and skip analytics for that move.

**Measured coverage and safety** (60 games, 1,176 opening positions):

| | |
|---|---|
| book hit rate in opening positions | **69.9%** |
| book-hit positions with a hanging queen or rook | **0 of 822 (0.00%)** |

The safety number is what makes this acceptable. Taking the fast path skips
`check_obvious_move`, so the bot would not spot a free queen -- but a book hit
is strong evidence the position is ordinary theory, because a real blunder
takes the game *out* of book. Zero counterexamples in 822 hits.

**Expected size:** book hits are ~13% of all moves (69.9% of an opening that is
18.7% of the game). Converting them to instants moves the opening rate from
0.362 toward ~0.70 and the aggregate from 0.252 to roughly **0.315** -- across
the 2100-2299 floor on this lever alone.

**Risks and guards.** The fast path must not apply when the position is
unusual: keep it to genuine book hits, and retain the existing sanity check on
the scraped position. Skipping analytics also means no mood update and no
opponent-blunder check for that move; both are per-move state that the next
move recomputes, but the ponder/premove preparation that normally rides along
with analytics is also skipped, so the sweep must confirm the opening fast path
does not *reduce* downstream ponder hits.

## Lever 2: ponder expansion

`common/search_constants.py:240-248` already names this the structural lever:

> "Lowering the per-position cost is the structural lever on the ponder-hit
> rate, the dominant instant-move channel: 2500-2800 humans fire 60% of sub-10s
> moves instantly off preparation."

`PONDER_TIME_PER_POSITION` was cut 0.1 -> 0.05 once already, because at 0.1 a
typical bullet move's leftover budget only ever covered one reply and the width
cap never bound. `game_ponder_width` clips at 0-5 with base 3.0, so there is
headroom above the currently realised width.

**Design:** sweep `PONDER_TIME_PER_POSITION` downward (0.05 -> 0.03) and the
width base upward, both as per-instance overrides. Cost is shallower
per-position evals for the pondered replies -- a real quality trade, which is
why the sweep must watch the error guards and not only the instant rate.

Aimed at the midgame and scramble. The bot fires 0.389 of its sub-10s moves
instantly against the ~0.60 CLAUDE.md cites for 2500-2800 humans, so the
scramble is where this should show.

## Lever 3: ambiguity snap-gate deltas

Already built and landed inert in `a4d2c7e`
(`2026-07-27-ambiguity-snap-gate-design.md`); the deltas are still 0.0 and its
sweep was never run. It targets the same channel and belongs in the same batch
rather than a separate run.

Note Phase A reframes its expected size: it was specced against
`instant_in_sharp_rate` (bot 0.3201 vs human 0.3500), but the aggregate deficit
is far larger and sits in the opening and endgame, which this lever does not
touch. Judge it on `instant_in_sharp_rate` and `corr_time_sharpness`, not on
the aggregate.

## Sweep

Implement first (levers 1 and 2 need code), then one batch. Pure self-play,
complete games, 60+0, 150 games/arm -- same conventions as Phase A, and for the
same reasons (score-rate saturation; timing features need the scramble).

| arm | book fast path | ponder | snap deltas |
|---|---|---|---|
| control | off | 0.05 / base 3.0 | 0 / 0 |
| A | **on** | 0.05 / base 3.0 | 0 / 0 |
| B | off | **0.03** / base 3.0 | 0 / 0 |
| C | off | 0.05 / **base 4.0** | 0 / 0 |
| D | off | 0.05 / base 3.0 | **0.15 / 0** |
| E | **on** | **0.03** / base 3.0 | **0.15 / 0** |

One lever at a time (A-D) plus a combined arm (E), because the three act on the
same feature through different phases and may not add linearly. ~7 arms at ~40
min each.

**Judged on:**

- `instant_move_rate` 0.252 -> target 0.309 (floor) to 0.379 (2800+), **primary**
- per-phase instant rate, which is how each lever's aim is verified
- `instant_in_sharp_rate` and `corr_time_sharpness` for lever 3
- **Guards, not targets:** `blunder_rate_timepressure` and the match rates.
  Phase A showed `blunder_rate`/`acpl` swing non-monotonically at n=150, so
  treat a single-run move as noise unless it is large. The premove-poison
  history means a TP-blunder regression is the specific thing to watch.

## Testing

1. **Engine parity harness passes unchanged, without `--record`**, for levers
   shipped inert. The book fast path *changes behaviour by construction*, so it
   must be gated behind a default-off flag for parity to hold; if parity moves
   with the flag off, the gating is wrong.
2. **Unit tests**: the book fast path fires on a hit and not on a miss; it is
   skipped when the position fails the sanity check; ponder width/budget
   resolution honours the per-instance overrides.
3. **Client unit tests** green (7 of 34 error on `fastgrab._linux_x11` without
   X11 -- environment, not regression).
4. **Lint** clean on changed files.

## Open question for the sweep

Whether the opening fast path suppresses downstream ponder preparation enough
to cancel its own gain. Analytics currently runs on every move and feeds the
ponder/premove setup; skipping it in the opening may cost ponder hits later.
Arm E versus arms A and B is what separates that -- if E is materially below
A+B combined, the interaction is real.
