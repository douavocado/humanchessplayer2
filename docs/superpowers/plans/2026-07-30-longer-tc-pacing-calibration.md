# Longer-TC Pacing Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the bot pace 180+0 like a human — correct overall level via a refitted `quickness`, and a correct opening/midgame shape via a time-control-keyed envelope.

**Architecture:** Two orthogonal corrections. A new `common/tc_profiles.py` holds
the phase envelopes as data keyed by initial clock, with a `LEGACY` profile that
reproduces today's arithmetic exactly for every control; `decision_logic` reads
it instead of hardcoding. `common/strength_profiles.py` becomes keyed by
`(initial_time, rating)`. Both fits are measure-fit-remeasure loops against
self-play arms, because the requested→realised mapping is nonlinear.

**Tech Stack:** Python 3.12 via `venv/bin/python`, `unittest`, `dataclasses`,
`simulation.run`, `cheat_detection`, `ruff`.

Spec: `docs/superpowers/specs/2026-07-30-longer-tc-pacing-calibration-design.md`

## Global Constraints

- **Python is `venv/bin/python`.** No bare `python` on PATH; the system `python3` lacks the dependencies.
- **Lint with `venv/bin/ruff check <changed files>` only** — never the whole tree (~170 pre-existing violations elsewhere).
- **The 60+0 path must not move.** The engine parity harness (`venv/bin/python -m unittest discover -s testing/engine_parity`) must pass **without `--record`**. A parity move means the pinning is wrong. ⚠️ Do not trust parity under load: a normal run is ~16s; 25-36s means something else is competing (check `pgrep -f multiprocessing`).
- **Use `cheat_detection.features._phase`, never `common.board_information.phase_of_game`, when comparing against the human band table.** They are unrelated rules. Mixing them invents a phase-mix crisis that does not exist.
- **Fit Part 1 (`quickness`) before Part 2 (envelope).** The level fit lifts the opening too (1.124s → ~1.37s); the envelope must be fitted against the residual.
- **Exclude the 2800+ band from every fit.** 33,265 moves vs the bottom band's 485,633, anomalous on every column at once. Use bands 2100-2799.
- Time controls are `seconds+increment`: `180+0` means three minutes.
- Human 180+0 targets: mean emt **3.283s**, opening **1.519s**, midgame **4.028s**, endgame **1.474s**, open/mid **0.377**.
- Measured bot baseline (seed 910000, 100 games): mean emt **2.699s**, opening **1.124s**, midgame **3.528s**, endgame **1.304s**, open/mid **0.319**.

## Two corrections to the spec, made here deliberately

**1. Resolution is exact-match-else-legacy, not "nearest keyed row."** The spec
says `resolve_tc` returns the nearest keyed row. That would change behaviour at
unkeyed controls: with rows at 60 and 180, a 90-second game resolves to the 60
row and gets midgame `*1.4`, where today's code gives `*1.7` (its branch is
`> 60`). Nearest-row silently alters every control we have not fitted. Instead:
an exactly-keyed clock gets its fitted row, everything else gets `LEGACY`, which
reproduces today's arithmetic for any `T`. Zero behaviour change except where we
have measured.

**2. `Engine.__init__` cannot resolve a TC-keyed dial without being told the
clock.** `self.quickness` is set in `__init__`, but the clock only arrives later
via `update_info`'s `self_initial_time`. Task 4 therefore adds an explicit
`initial_time` argument to `Engine.__init__` used *only* for dial resolution,
defaulting to 60.0, plus a warning when it disagrees with the clock that later
arrives. The per-move pacing formula keeps reading `input_info` as it does now —
this does not become a second source of truth for the live clock.

## File Structure

| File | Responsibility |
|---|---|
| `common/tc_profiles.py` (create) | Phase envelopes as data, keyed by initial clock. Owns `LEGACY` (today's arithmetic) and any fitted rows. |
| `engine_components/decision_logic.py` (modify) | Reads the resolved profile instead of hardcoding the envelopes. |
| `common/strength_profiles.py` (modify) | `_PROFILES` keyed by `(initial_time, rating)`; `resolve` gains `initial_time`. |
| `engine.py` (modify) | Passes `initial_time` into the dial; warns on disagreement. |
| `testing/engine_components/test_tc_profiles.py` (create) | The pinning guarantee and resolution rules. |
| `cheat_detection/runs/tc_envelope/` (add drivers) | Sweep and validation drivers, resumable, matching the `run_phase_*.py` convention. |

---

## Task 1: `common/tc_profiles.py` with the legacy row pinned

**Files:**
- Create: `common/tc_profiles.py`
- Test: `testing/engine_components/test_tc_profiles.py`

**Interfaces:**
- Produces:
  - `TCProfile` dataclass with fields `opening: Callable[[float], float]`, `midgame: Callable[[float, float], float]`, `endgame: Callable[[float], float]`, `fitted_at: Optional[float]`, `label: str`
  - `LEGACY: TCProfile`
  - `TC_PROFILES: dict[float, TCProfile]` (empty at this task; Task 5 adds 180.0)
  - `resolve_tc(initial_time: float) -> TCProfile`
  - `apply_envelope(profile: TCProfile, base_time: float, phase: str, initial_time: float) -> float`

**Why callables rather than plain multipliers:** the legacy opening is
`(base ** 0.2)/2`, not a multiplier, and the legacy midgame branches on the
clock. Storing them as small functions lets `LEGACY` reproduce today's
arithmetic *exactly* for every control, which is what makes Task 2 provably
inert. Fitted rows may use simple multipliers.

- [ ] **Step 1: Write the failing test**

Create `testing/engine_components/test_tc_profiles.py`:

```python
"""Guards the time-control phase-envelope table.

The load-bearing test is `test_legacy_reproduces_current_formula`. The 60+0
calibration is the only one this repo has validated, and Task 2 replaces the
inline envelope arithmetic with a table lookup. If LEGACY is not bit-for-bit
the old formula, every 1+0 measurement silently changes meaning.
"""
import unittest

from common.tc_profiles import LEGACY, TC_PROFILES, apply_envelope, resolve_tc


def _legacy_expected(base, phase, initial_time):
    """The arithmetic decision_logic.py used before the table existed."""
    if phase == "opening":
        return (base ** 0.2) / 2
    if phase == "midgame":
        return base * (1.7 if initial_time > 60 else 1.4)
    return base * 0.7


class TestLegacyPinning(unittest.TestCase):

    GRID = [(b, t) for b in (0.1, 0.5, 1.92, 5.48, 15.12, 40.0)
            for t in (30.0, 60.0, 61.0, 90.0, 180.0, 600.0)]

    def test_legacy_reproduces_current_formula(self):
        for base, t in self.GRID:
            for phase in ("opening", "midgame", "endgame"):
                with self.subTest(base=base, t=t, phase=phase):
                    self.assertEqual(
                        apply_envelope(LEGACY, base, phase, t),
                        _legacy_expected(base, phase, t))

    def test_midgame_branch_is_strictly_above_60(self):
        """The legacy branch is `> 60`, so 60 itself takes the 1.4 path."""
        self.assertAlmostEqual(apply_envelope(LEGACY, 1.0, "midgame", 60.0), 1.4)
        self.assertAlmostEqual(apply_envelope(LEGACY, 1.0, "midgame", 60.1), 1.7)


class TestResolution(unittest.TestCase):

    def test_unkeyed_control_resolves_to_legacy(self):
        """Exact-match-else-legacy: an unfitted control must keep today's
        behaviour rather than borrowing a neighbouring row's fit."""
        for t in (30.0, 60.0, 90.0, 300.0, 600.0):
            if t not in TC_PROFILES:
                self.assertIs(resolve_tc(t), LEGACY)

    def test_keyed_control_resolves_to_its_row(self):
        for t, prof in TC_PROFILES.items():
            self.assertIs(resolve_tc(t), prof)

    def test_legacy_reports_no_fitted_clock(self):
        self.assertIsNone(LEGACY.fitted_at)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m unittest testing.engine_components.test_tc_profiles -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'common.tc_profiles'`

- [ ] **Step 3: Implement the module**

Create `common/tc_profiles.py`:

```python
"""Per-time-control phase envelopes for move-time pacing.

`decision_logic._get_time_taken` scales a base think time by the phase of the
game. Those multipliers were fitted at 60+0 -- 60 *seconds*, one minute of
bullet -- and one of them is wrong at longer controls: the opening form
`(base ** 0.2)/2` compresses toward a constant while the midgame scales
linearly, so the opening/midgame ratio falls as the clock grows. Humans hold
that ratio at 0.374 (60+0) -> 0.377 (180+0) across a 3x clock change, measured
over 3.14M moves.

**Resolution is exact-match-else-LEGACY, deliberately.** A "nearest row" rule
would change behaviour at controls nobody has measured: with rows at 60 and
180, a 90-second game would take the 60 row's midgame *1.4 where today's code
gives *1.7 (its branch is `> 60`). Only a clock with a fitted row behaves
differently from before.

Design and measurements:
`docs/superpowers/specs/2026-07-30-longer-tc-pacing-calibration-design.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class TCProfile:
    """Phase envelopes for one time control.

    Each field maps a base think time to a phase-adjusted one. They are
    callables rather than plain multipliers because the legacy opening is a
    power-law compression, not a multiplier, and the legacy midgame branches
    on the clock -- storing them as functions lets LEGACY reproduce the
    pre-table arithmetic exactly for every control.
    """
    opening: Callable[[float], float]
    midgame: Callable[[float, float], float]   # (base, initial_time)
    endgame: Callable[[float], float]
    fitted_at: Optional[float]  # clock this row was fitted at; None = unfitted
    label: str


LEGACY = TCProfile(
    opening=lambda base: (base ** 0.2) / 2,
    midgame=lambda base, t: base * (1.7 if t > 60 else 1.4),
    endgame=lambda base: base * 0.7,
    fitted_at=None,
    label="legacy (pre-2026-07-30 inline envelopes)",
)

# Fitted rows, keyed by exact initial clock in seconds. Task 5 adds 180.0.
TC_PROFILES: dict[float, TCProfile] = {}


def resolve_tc(initial_time: float) -> TCProfile:
    """The profile for this clock: its fitted row, or LEGACY if unfitted."""
    return TC_PROFILES.get(float(initial_time), LEGACY)


def apply_envelope(profile: TCProfile, base_time: float, phase: str,
                   initial_time: float) -> float:
    """Base think time scaled by `phase`'s envelope under `profile`.

    `phase` uses the engine's own vocabulary from
    `common.board_information.phase_of_game`: "opening", "midgame", or
    anything else (treated as endgame, matching the pre-table `else` branch).
    NOTE this is NOT cheat_detection's phase rule -- see the spec.
    """
    if phase == "opening":
        return profile.opening(base_time)
    if phase == "midgame":
        return profile.midgame(base_time, initial_time)
    return profile.endgame(base_time)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `venv/bin/python -m unittest testing.engine_components.test_tc_profiles -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Lint and commit**

```bash
venv/bin/ruff check common/tc_profiles.py testing/engine_components/test_tc_profiles.py
git add common/tc_profiles.py testing/engine_components/test_tc_profiles.py
git commit -m "feat(pacing): time-control phase-envelope table, legacy row pinned

Inert: TC_PROFILES is empty, so every control resolves to LEGACY, which
reproduces the inline arithmetic exactly. Resolution is exact-match-else-legacy
rather than nearest-row, so unfitted controls cannot drift."
```

---

## Task 2: Read the table from `decision_logic`

**Files:**
- Modify: `engine_components/decision_logic.py:191-200`
- Test: existing `testing/engine_parity/` (the gate), plus Task 1's suite

**Interfaces:**
- Consumes: `resolve_tc`, `apply_envelope` from Task 1.
- Produces: no new API. Behaviour must be **identical** — `TC_PROFILES` is still empty.

**This is the risky task.** It replaces working, calibrated arithmetic with a
lookup. The parity harness is the proof it did nothing.

- [ ] **Step 1: Record the pre-change parity baseline**

```bash
venv/bin/python -m unittest discover -s testing/engine_parity 2>&1 | tail -5
```
Expected: PASS in roughly 16s. If it fails or takes 25s+ on a clean tree,
something else is competing — check `pgrep -f multiprocessing`, wait, retry.
Do not proceed until you have seen it green.

- [ ] **Step 2: Replace the inline envelopes**

In `engine_components/decision_logic.py`, add to the imports:

```python
from common.tc_profiles import apply_envelope, resolve_tc
```

Replace this block (currently at lines 191-200):

```python
        game_phase = phase_of_game(engine.current_board)
        # in the opening and endgame we spend less time on average than the mid game
        if game_phase == "opening":
            base_time = (base_time ** 0.2)/2
        elif game_phase == "midgame":
            if self_initial_time > 60:
                base_time *= 1.7
            else:
                base_time *= 1.4
        else:
            base_time *=0.7
```

with:

```python
        game_phase = phase_of_game(engine.current_board)
        # Phase envelopes live in common/tc_profiles.py, keyed by initial
        # clock: the 60+0 shape is wrong at longer controls because the
        # opening form compresses toward a constant while the midgame scales.
        # An unfitted control resolves to LEGACY, i.e. exactly this file's
        # previous arithmetic.
        _tc = resolve_tc(self_initial_time)
        base_time = apply_envelope(_tc, base_time, game_phase, self_initial_time)
        engine.log += "Phase envelope: {} profile ({}) \n".format(
            game_phase, _tc.label)
```

- [ ] **Step 3: Run the parity harness**

Run: `venv/bin/python -m unittest discover -s testing/engine_parity`
Expected: PASS, unchanged, **without `--record`**.

If it fails: check the runtime first. 25-36s means load, not a regression —
wait and re-run before concluding anything. Two false failures were nearly
attributed to unrelated changes on 2026-07-28.

- [ ] **Step 4: Run the component suites**

```bash
venv/bin/python -m unittest discover -s testing/engine_components
venv/bin/python -m unittest discover -s testing/cheat_detection
```
Expected: both green.

- [ ] **Step 5: Lint and commit**

```bash
venv/bin/ruff check engine_components/decision_logic.py
git add engine_components/decision_logic.py
git commit -m "refactor(pacing): read phase envelopes from tc_profiles

Behaviour-identical: TC_PROFILES is empty so every control resolves to LEGACY.
Verified by the engine parity harness passing unchanged."
```

---

## Task 3: Sweep `quickness` at 180+0

**Files:**
- Create: `cheat_detection/runs/tc_envelope/run_quickness_sweep_180.py`
- Modify: `cheat_detection/.gitignore` (negation for the driver)

**Interfaces:**
- Consumes: nothing from earlier tasks (measurement only; the code under test is unchanged).
- Produces: `cheat_detection/runs/tc_envelope/quickness_sweep_180.json` — `{"arms": [{"quickness": float, "mean_emt": float, "open_mid": float, "n_moves": int}, ...]}`, consumed by Task 4.

**Why four arms:** the 60+0 fit `mean_emt = 0.1964*q + 0.6234` used a spread of
`quickness` values and regressed. Two points would give a line with no residual
to judge it by. Shipped `QUICKNESS` produces mean emt 2.699s against a 3.283s
target, so the fit must extrapolate upward — bracket it rather than
extrapolating off the end.

- [ ] **Step 1: Read the shipped quickness so the sweep brackets it**

```bash
grep -n "^QUICKNESS" common/constants.py
```
Record the value. The sweep must span it and reach above it, since the bot is
too fast.

- [ ] **Step 2: Write the driver**

Create `cheat_detection/runs/tc_envelope/run_quickness_sweep_180.py`:

```python
"""Part 1: refit `quickness` at 180+0.

The dial's only calibrated mapping, `mean_emt = 0.1964*quickness + 0.6234`, is
a 60+0 fit; its intercept is a bullet artefact and it has no meaning at three
minutes. This sweeps quickness and regresses realised mean emt on it, so
Task 4 can invert onto the per-band targets.

Measured starting point (seed 910000, shipped defaults): mean emt 2.699s
against a human 3.283s, so the fit extrapolates upward -- the arms bracket
shipped QUICKNESS rather than sitting below it.

Conventions match the spec's measurement arm or the numbers are not
comparable: pure self-play, complete games (`--simulate-full`), 180+0, 100
games per arm. Phase means use cheat_detection's `_phase`, NOT the engine's
`phase_of_game` -- they are unrelated rules and mixing them invents a
phase-mix crisis that does not exist.

~4 arms. Run from repo root:
    venv/bin/python cheat_detection/runs/tc_envelope/run_quickness_sweep_180.py
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RESULT = OUT_DIR / "quickness_sweep_180.json"

GAMES = 100
TC = "180+0"
WORKERS = 6
ARMS = [(2.0, 920000), (2.5, 930000), (3.2, 940000), (4.0, 950000)]

HUMAN_MEAN_EMT = 3.283
HUMAN_OPEN_MID = 0.377


def profile(pgn: Path) -> dict:
    """Realised mean emt and opening/midgame ratio, cheat_detection phases."""
    sys.path.insert(0, str(ROOT))
    from cheat_detection.config import AnalysisConfig
    from cheat_detection.features import _phase
    from cheat_detection.pgn_loader import iter_games

    cfg = AnalysisConfig(initial_time=180.0)
    byphase = collections.defaultdict(list)
    for g in iter_games(str(pgn)):
        b = chess.Board()
        for m in g.moves:
            ph = _phase(b, m.ply, cfg)
            if m.emt is not None:
                byphase[ph].append(m.emt)
            b.push(chess.Move.from_uci(m.move_uci))
    allm = [x for v in byphase.values() for x in v]
    op, mid = byphase.get("opening", []), byphase.get("middlegame", [])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "mean_emt": mean(allm),
        "opening": mean(op), "middlegame": mean(mid),
        "endgame": mean(byphase.get("endgame", [])),
        "open_mid": mean(op) / mean(mid) if op and mid else float("nan"),
        "n_moves": len(allm),
    }


def main() -> None:
    arms = []
    for q, seed in ARMS:
        pgn = OUT_DIR / f"sweepq_{q}_seed{seed}.pgn"
        if not pgn.exists():
            print(f"[{time.strftime('%H:%M:%S')}] quickness={q} seed={seed} ...",
                  flush=True)
            start = time.time()
            proc = subprocess.run(
                [sys.executable, "-m", "simulation.run",
                 "--games", str(GAMES), "--tc", TC, "--seed", str(seed),
                 "--workers", str(WORKERS), "--simulate-full", "--plain",
                 "--a-quickness", str(q), "--b-quickness", str(q),
                 "--out", str(pgn)],
                cwd=ROOT, check=False,
                stdout=open(OUT_DIR / f"sweepq_{q}.progress", "w"),
                stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise SystemExit(f"arm q={q} failed rc={proc.returncode}")
            print(f"  done in {(time.time()-start)/60:.1f} min", flush=True)
        p = profile(pgn)
        p["quickness"] = q
        arms.append(p)
        print(f"  q={q}: mean_emt={p['mean_emt']:.3f} "
              f"open_mid={p['open_mid']:.3f} n={p['n_moves']}", flush=True)

    RESULT.write_text(json.dumps(
        {"tc": TC, "games_per_arm": GAMES,
         "human_mean_emt": HUMAN_MEAN_EMT, "human_open_mid": HUMAN_OPEN_MID,
         "arms": arms}, indent=2), encoding="utf-8")
    print(f"Wrote {RESULT}")
    print("Read: is mean_emt monotone in quickness, and is open_mid flat "
          "across arms? open_mid MUST be roughly flat -- quickness is a global "
          "scale, so if the ratio moves with it, the model in the spec is wrong.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify `--a-quickness` is the real flag name**

```bash
venv/bin/python -m simulation.run --help 2>&1 | grep -i quickness
```
Expected: `--a-quickness` and `--b-quickness` listed. If the names differ, fix
the driver to match — do not rename the CLI.

- [ ] **Step 4: Track the driver and commit before the long run**

```bash
printf '!runs/tc_envelope/\nruns/tc_envelope/*\n!runs/tc_envelope/*.py\n' >> cheat_detection/.gitignore
venv/bin/ruff check cheat_detection/runs/tc_envelope/run_quickness_sweep_180.py
git add cheat_detection/.gitignore cheat_detection/runs/tc_envelope/run_quickness_sweep_180.py
git commit -m "feat(pacing): 180+0 quickness sweep driver"
```

- [ ] **Step 5: Run the sweep**

```bash
venv/bin/python cheat_detection/runs/tc_envelope/run_quickness_sweep_180.py
```
Expected: ~4 arms. Report each arm's `mean_emt` and `open_mid`.

**The falsification check:** `open_mid` must be roughly flat across arms. The
spec's whole structure rests on `quickness` being a global scale under which
the phase ratio is invariant. If `open_mid` moves materially with `quickness`,
stop and report — Part 2's premise is wrong and the plan needs revisiting.

- [ ] **Step 6: Commit the sweep result**

```bash
git add -f cheat_detection/runs/tc_envelope/quickness_sweep_180.json
git commit -m "feat(pacing): 180+0 quickness sweep results"
```

---

## Task 4: TC-keyed `strength_profiles` and the fitted 180+0 quickness

**Files:**
- Modify: `common/strength_profiles.py`
- Modify: `engine.py:62-81`
- Test: `testing/engine_components/test_strength_profiles.py` (extend)

**Interfaces:**
- Consumes: `quickness_sweep_180.json` from Task 3.
- Produces:
  - `STRENGTH_CLOCKS: tuple[float, ...]` — clocks with fitted rows, `(60.0, 180.0)`
  - `snap_rating(rating, initial_time=60.0) -> int`
  - `resolve(rating, initial_time=60.0) -> dict` — adds `"effective_clock": float` alongside the existing `"effective_rating"`
  - `Engine.__init__(..., initial_time: Optional[float] = None)`

- [ ] **Step 1: Fit the regression from the sweep**

```bash
venv/bin/python - <<'EOF'
import json
d = json.load(open("cheat_detection/runs/tc_envelope/quickness_sweep_180.json"))
pts = [(a["quickness"], a["mean_emt"]) for a in d["arms"]]
n = len(pts)
mx = sum(x for x, _ in pts)/n; my = sum(y for _, y in pts)/n
sxx = sum((x-mx)**2 for x, _ in pts); sxy = sum((x-mx)*(y-my) for x, y in pts)
m = sxy/sxx; c = my - m*mx
print(f"mean_emt = {m:.4f}*q + {c:.4f}")
print("residuals:", [(x, round(y-(m*x+c), 4)) for x, y in pts])
print("open_mid across arms:", [round(a["open_mid"], 4) for a in d["arms"]])
for band, tgt in [(2200,3.23),(2350,3.17),(2450,3.16),(2550,3.09),(2650,3.05),(2750,2.99)]:
    print(f"  band {band}: target {tgt}s -> quickness {(tgt-c)/m:.3f}")
EOF
```

Record the fitted `m`, `c`, the residuals, and the six inverted quickness
values. **Sanity-check the residuals**: if any exceeds ~0.1s the relationship
is not linear over this range and a wider sweep is needed — report rather than
forcing a line through it.

- [ ] **Step 2: Write the failing test**

Append to `testing/engine_components/test_strength_profiles.py`:

```python
class TestTimeControlKeying(unittest.TestCase):
    """The dial is per-time-control. A 60+0 quickness at 180+0 would pace the
    bot to a clock it is not playing."""

    def test_60_and_180_are_both_fitted(self):
        from common.strength_profiles import STRENGTH_CLOCKS
        self.assertIn(60.0, STRENGTH_CLOCKS)
        self.assertIn(180.0, STRENGTH_CLOCKS)

    def test_same_rating_differs_by_clock(self):
        from common.strength_profiles import resolve
        a = resolve(2700, initial_time=60.0)
        b = resolve(2700, initial_time=180.0)
        self.assertNotAlmostEqual(a["quickness"], b["quickness"], places=2)

    def test_default_clock_is_60_and_unchanged(self):
        """Callers that predate the clock argument must be unaffected."""
        from common.strength_profiles import resolve
        self.assertEqual(resolve(2700), resolve(2700, initial_time=60.0))
        self.assertAlmostEqual(resolve(2700)["quickness"], 2.248, places=3)

    def test_reports_effective_clock(self):
        from common.strength_profiles import resolve
        self.assertEqual(resolve(2700, initial_time=180.0)["effective_clock"],
                         180.0)

    def test_unfitted_clock_snaps_and_says_so(self):
        """A 300+0 request must not silently receive a 180+0 fit."""
        from common.strength_profiles import resolve
        r = resolve(2700, initial_time=300.0)
        self.assertIn(r["effective_clock"], (60.0, 180.0))
        self.assertNotEqual(r["effective_clock"], 300.0)
```

- [ ] **Step 3: Run to verify it fails**

Run: `venv/bin/python -m unittest testing.engine_components.test_strength_profiles -v`
Expected: FAIL — `ImportError: cannot import name 'STRENGTH_CLOCKS'`

- [ ] **Step 4: Implement the TC keying**

In `common/strength_profiles.py`, rekey `_PROFILES` from `{rating: {...}}` to
`{(clock, rating): {...}}`, keeping the four existing 60.0 rows at their
current values (2200: 3.241, 2450: 2.885, 2700: 2.248, 2850: 1.918) and adding
180.0 rows from Step 1's inverted quickness values, snapped to the same
`STRENGTH_LEVELS`. Add:

```python
STRENGTH_CLOCKS = (60.0, 180.0)


def snap_clock(initial_time):
    """Nearest fitted clock. Unfitted controls get the nearest fit, which the
    caller sees via resolve()'s "effective_clock"."""
    return min(STRENGTH_CLOCKS, key=lambda c: (abs(c - initial_time), c))


def resolve(rating, initial_time=60.0):
    clock = snap_clock(initial_time)
    level = snap_rating(rating)
    out = dict(_PROFILES[(clock, level)])
    out["effective_rating"] = level
    out["effective_clock"] = clock
    return out
```

Update the module docstring: the scope line currently says "60+0 only"; it now
covers 60+0 and 180+0, and must state that the 180+0 fit **excludes the 2800+
band** (33,265 moves, anomalous on every column) and so the 2850 level is
extrapolated from the 2100-2799 trend rather than measured.

- [ ] **Step 5: Plumb `initial_time` into the Engine**

In `engine.py`, add `initial_time: Optional[float] = None` to `__init__`'s
signature and change the dial resolution (currently line 71):

```python
        self.dial_initial_time = 60.0 if initial_time is None else float(initial_time)
        _dial = strength_profiles.resolve(
            target_rating, initial_time=self.dial_initial_time
        ) if target_rating is not None else {}
```

The clock only arrives per-move via `update_info`, so the dial cannot read it
at construction — this argument is the declared control, used **only** for dial
resolution. The pacing formula keeps reading `input_info["self_initial_time"]`;
this must not become a second source of truth. Add a one-shot warning in
`engine_components/state.py:update_info`, after `self_initial_time` is read:

```python
    if (engine.target_rating is not None
            and not getattr(engine, "_warned_dial_clock", False)
            and abs(engine.input_info["self_initial_time"] - engine.dial_initial_time) > 1.0):
        engine.log += (
            "WARNING: target_rating was resolved for a {}s clock but this game "
            "is {}s; the dial's quickness is fitted to the wrong control.\n"
            .format(engine.dial_initial_time,
                    engine.input_info["self_initial_time"]))
        engine._warned_dial_clock = True
```

Initialise `self._warned_dial_clock = False` in `__init__` beside
`self.dial_initial_time`.

- [ ] **Step 6: Run the tests**

```bash
venv/bin/python -m unittest discover -s testing/engine_components
venv/bin/python -m unittest discover -s testing/engine_parity
```
Expected: both green. Parity must be unchanged — `target_rating` defaults to
`None`, so the dial resolves nothing and no scenario touches this path.

- [ ] **Step 7: Lint and commit**

```bash
venv/bin/ruff check common/strength_profiles.py engine.py engine_components/state.py testing/engine_components/test_strength_profiles.py
git add common/strength_profiles.py engine.py engine_components/state.py testing/engine_components/test_strength_profiles.py
git commit -m "feat(pacing): key the strength dial by time control

Adds fitted 180+0 quickness rows and an initial_time argument. The 2850 level
at 180+0 is extrapolated, not measured -- the 2800+ human band is excluded
from the fit as anomalous."
```

---

## Task 5: Fit the 180+0 opening envelope and validate

**Files:**
- Modify: `common/tc_profiles.py` (add the 180.0 row)
- Create: `cheat_detection/runs/tc_envelope/run_envelope_fit_180.py`
- Test: `testing/engine_components/test_tc_profiles.py` (extend)

**Interfaces:**
- Consumes: `resolve_tc`/`apply_envelope`/`TC_PROFILES` (Task 1), the fitted quickness (Task 4).
- Produces: `TC_PROFILES[180.0]`, a `TCProfile` with `fitted_at=180.0`.

**Run this only after Task 4 is merged.** The level fit lifts the opening from
1.124s to roughly 1.37s; the envelope must be fitted against that residual, not
against today's gap. Fitting them in the other order double-counts.

- [ ] **Step 1: Measure the residual at the fitted quickness**

```bash
venv/bin/python -m simulation.run --games 100 --tc 180+0 --seed 960000 \
  --workers 6 --simulate-full --plain \
  --a-quickness <FITTED> --b-quickness <FITTED> \
  --out cheat_detection/runs/tc_envelope/sim_180_level_fitted.pgn
```

Then profile it. `cheat_detection/runs/tc_envelope/` is not a package, so import
the driver's `profile()` helper by path:

```bash
venv/bin/python - <<'EOF'
import importlib.util, json
spec = importlib.util.spec_from_file_location(
    "sweep", "cheat_detection/runs/tc_envelope/run_quickness_sweep_180.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from pathlib import Path
p = mod.profile(Path("cheat_detection/runs/tc_envelope/sim_180_level_fitted.pgn"))
print(json.dumps(p, indent=2))
print(f"open_mid {p['open_mid']:.4f} vs target 0.377")
print(f"mean_emt {p['mean_emt']:.4f} vs target 3.283")
EOF
```

Record `opening`, `middlegame`, and `open_mid`. Target: **0.377**.

- [ ] **Step 2: Write the failing test**

Append to `testing/engine_components/test_tc_profiles.py`:

```python
class TestFitted180(unittest.TestCase):

    def test_180_has_a_fitted_row(self):
        self.assertIn(180.0, TC_PROFILES)
        self.assertEqual(TC_PROFILES[180.0].fitted_at, 180.0)

    def test_180_opening_is_proportional_not_compressed(self):
        """The legacy form compresses the opening toward a constant as the
        clock grows, which is the defect. The fitted row must scale with base."""
        p = TC_PROFILES[180.0]
        lo = apply_envelope(p, 2.0, "opening", 180.0)
        hi = apply_envelope(p, 4.0, "opening", 180.0)
        self.assertAlmostEqual(hi / lo, 2.0, places=6)

    def test_60_is_untouched_by_the_180_row(self):
        self.assertIs(resolve_tc(60.0), LEGACY)
        self.assertNotIn(60.0, TC_PROFILES)
```

- [ ] **Step 3: Run to verify it fails**

Run: `venv/bin/python -m unittest testing.engine_components.test_tc_profiles.TestFitted180 -v`
Expected: FAIL — `180.0 not found in {}`

- [ ] **Step 4: Add the fitted row**

In `common/tc_profiles.py`, replace the empty `TC_PROFILES` with:

```python
# Fitted rows, keyed by exact initial clock in seconds.
#
# 180.0: the opening becomes proportional to base rather than the legacy
# power-law compression, because humans hold opening/midgame at 0.374 (60+0)
# -> 0.377 (180+0) while the legacy form drives the bot's ratio down as the
# clock grows. OPENING_K_180 is fitted, not derived: the requested-time
# algebra predicted a ratio of 0.075 where the measured realised ratio was
# 0.319, so only a measured fit is meaningful here. Midgame and endgame keep
# their legacy multipliers -- both phases measured 0.88x of human, i.e. the
# same as each other, so their relative shape is right and the level is
# `quickness`'s job.
OPENING_K_180 = <FITTED>

TC_PROFILES: dict[float, TCProfile] = {
    180.0: TCProfile(
        opening=lambda base: base * OPENING_K_180,
        midgame=lambda base, t: base * 1.7,
        endgame=lambda base: base * 0.7,
        fitted_at=180.0,
        label="fitted 2026-07-30 at 180+0",
    ),
}
```

Fit `OPENING_K_180` by iteration: start from `0.377 * 1.7 = 0.641`, run the
validation arm, and adjust by the ratio of measured to target `open_mid`. The
requested→realised mapping is nonlinear (compute floor, moods, intuition gate),
so expect 2-3 iterations. Do **not** solve it analytically.

- [ ] **Step 5: Write the validation driver and iterate**

Create `cheat_detection/runs/tc_envelope/run_envelope_fit_180.py` — same
structure as Task 3's driver (subprocess `simulation.run`, then the `profile()`
helper), but a single arm at the fitted quickness, writing
`envelope_fit_180.json` with `opening`, `middlegame`, `open_mid`, `mean_emt`.
Re-run after each `OPENING_K_180` adjustment.

**Accept when:** `open_mid` is within 0.02 of 0.377 and `mean_emt` is within
0.15s of 3.283. Report both, plus how many iterations it took.

- [ ] **Step 6: Guard against regression**

```bash
venv/bin/python -m cheat_detection.analyze report \
  --pgn cheat_detection/runs/tc_envelope/sim_180_envelope_fitted.pgn \
  --player SimBotWhite SimBotBlack --tc 180+0 \
  --baseline cheat_detection/baselines/blitz_3plus0_2300_plus.json \
  --workers 3 --threads 2 \
  --out-md cheat_detection/runs/tc_envelope/rep_180_fitted.md \
  --out-json cheat_detection/runs/tc_envelope/rep_180_fitted.report.json
```

Check `blunder_rate_timepressure` against the human 0.0674. ⚠️ Use the
**complete** baseline as above, never a truncation-matched one — the
simulator's cutoff is `0.25*T` and `time_pressure_secs` is `T/6`, so a
truncated corpus has `n=0` time-pressure moves at every control and an `n/a`
there is structural absence, not a clean result.

- [ ] **Step 7: Full test sweep, lint, commit**

```bash
venv/bin/python -m unittest discover -s testing/engine_components
venv/bin/python -m unittest discover -s testing/engine_parity
venv/bin/ruff check common/tc_profiles.py cheat_detection/runs/tc_envelope/run_envelope_fit_180.py
git add common/tc_profiles.py testing/engine_components/test_tc_profiles.py cheat_detection/runs/tc_envelope/run_envelope_fit_180.py
git commit -m "feat(pacing): fitted 180+0 opening envelope

Opening becomes proportional to base rather than power-law compressed, fitted
so realised open/mid reaches the human 0.377. 60+0 still resolves to LEGACY
and the parity harness is unchanged."
```

---

## Self-Review

**Spec coverage.** Part 1 is Tasks 3-4; Part 2 is Tasks 1, 2 and 5. The pinned
60+0 row is Task 1 step 1 and Task 2 step 3 (parity). The 2800+ exclusion is
Task 4 steps 1 and 4. The ordering constraint is Task 5's preamble. The
iterative-fit requirement is Task 5 step 4. Testing items 1-4 map to Tasks 1,
2, 4, 5; item 5 (validation arm) is Task 5 step 5; item 6 (the TP guard with
its truncation warning) is Task 5 step 6. Out-of-scope items are not
implemented: no endgame envelope change, no 300/600+0 rows, no
`high_range_multiplier` change, no accuracy levers.

**Known deviations from the spec, both stated at the top:** exact-match-else-
legacy resolution instead of nearest-row, and the `Engine.__init__` clock
argument the spec did not anticipate.

**Placeholder scan.** Two intentional `<FITTED>` markers exist (Task 5 step 1's
command and step 4's constant) because their values are produced by Task 3's
sweep and cannot be known when the plan is written. Every other step carries
real content. `OPENING_K_180`'s starting value and adjustment rule are given
explicitly so the marker is actionable.

**Type consistency.** `resolve_tc(initial_time: float) -> TCProfile` and
`apply_envelope(profile, base_time, phase, initial_time) -> float` are used
with those signatures in Tasks 1, 2 and 5. `resolve(rating, initial_time=60.0)`
returns `effective_rating` and `effective_clock` in Task 4's tests and
implementation alike. `profile(pgn) -> dict` returns the keys Task 5 step 1
reads. `TCProfile.midgame` takes `(base, initial_time)` everywhere, including
the fitted row, which ignores its second argument.
