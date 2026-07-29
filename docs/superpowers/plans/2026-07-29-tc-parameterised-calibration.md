# TC-Parameterised Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `cheat_detection/` time-control-aware so the human rating-band table can be regenerated at 180+0, without moving any existing 60+0 result.

**Architecture:** Two clock-dependent thresholds in `AnalysisConfig` become
properties derived from a new `initial_time` field (`long_think_secs =
initial_time/30`, `time_pressure_secs = initial_time/6`), both of which
reproduce their shipped constants exactly at `initial_time=60`. A `--tc` flag
sets it, a PGN-header guard enforces that the corpus actually is that control,
and `elo_progression.py`'s module-level threshold snapshots are threaded
through the config instead. Then a runbook regenerates the band table at 180+0.

**Tech Stack:** Python 3.12 via `venv/bin/python`, `unittest`, `dataclasses`,
`python-chess`, `ruff`.

Spec: `docs/superpowers/specs/2026-07-29-longer-time-control-calibration-design.md`

## Global Constraints

- **Python is `venv/bin/python`.** There is no bare `python` on PATH and the
  system `python3` lacks the dependencies. Every command in this plan uses the
  venv interpreter.
- **Lint with `venv/bin/ruff check <changed files>` only** — never the whole
  tree. The repo has ~170 pre-existing violations, so a whole-tree run is noise.
- **`instant_move_secs` stays absolute at 1.0.** It is a human motor-and-decision
  floor, not a share of the clock. Do not derive it from `initial_time`.
- **Exact derivations, non-negotiable:** `long_think_secs = initial_time / 30`
  and `time_pressure_secs = initial_time / 6`. At `initial_time = 60.0` these
  must equal `2.0` and `10.0` exactly. That exactness is the only thing
  protecting every existing bullet baseline and report in the repo.
- **`ambiguity_wc_window` must stay at 0.05**, equal to `compute_ambiguity`'s
  window in `engine_components/`. It is TC-invariant. Asserted by
  `testing/engine_components/test_ambiguity.py`; do not touch it.
- **Do not touch `engine.py` or `engine_components/`.** This work is confined to
  `cheat_detection/`. The engine parity harness must remain untouched and
  unrun-against.
- **Time controls are `seconds+increment`** — `180+0` means three minutes.
- **No bot-side sweeps, no simulation arms, no changes to
  `common/strength_profiles.py`.** Out of scope by design; see the spec's *Why
  this stops at measurement*.

## File Structure

| File | Responsibility |
|---|---|
| `cheat_detection/config.py` (modify) | Owns `initial_time` and the two derived thresholds. The single source of truth for what "long think" and "time pressure" mean at a given control. |
| `cheat_detection/pgn_loader.py` (modify) | Gains `TimeControlMismatchError` and `check_time_control()`. Lives here because `GameRecord.base_secs` — the thing being checked — is parsed here. |
| `cheat_detection/analyze.py` (modify) | Wires `--tc` and `--allow-tc-mismatch` into the config, and applies the guard at the two game-consumption sites it owns. |
| `cheat_detection/pipeline.py` (modify) | Applies the guard inside `iter_units`. |
| `cheat_detection/elo_progression.py` (modify) | Drops its module-level threshold snapshots; threads `cfg` through `_stats`/`render`; gains `--tc`. |
| `testing/cheat_detection/` (create) | New test home for the analyser, matching the existing per-area convention (`testing/client/`, `testing/engine_components/`). |
| `cheat_detection/runs/elo_progression/run_band_table_180.py` (create) | Resumable driver for the Phase 1 run, matching the `run_phase_*.py` convention. |

---

## Task 1: TC-derived thresholds in AnalysisConfig

**Files:**
- Modify: `cheat_detection/config.py:20-51`
- Create: `testing/cheat_detection/__init__.py` (empty)
- Test: `testing/cheat_detection/test_tc_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `cheat_detection.config.LONG_THINK_FRACTION: float` = `1/30`
  - `cheat_detection.config.TIME_PRESSURE_FRACTION: float` = `1/6`
  - `AnalysisConfig.initial_time: float` (dataclass field, default `60.0`)
  - `AnalysisConfig.long_think_secs_override: Optional[float]` (field, default `None`)
  - `AnalysisConfig.time_pressure_secs_override: Optional[float]` (field, default `None`)
  - `AnalysisConfig.long_think_secs -> float` (**read-only property**)
  - `AnalysisConfig.time_pressure_secs -> float` (**read-only property**)

**Design note for the implementer:** these become *properties*, not fields,
specifically so that the existing `setattr`-after-construction pattern in
`analyze.py:_config_from_args` keeps working. If they were fields filled in
`__post_init__`, then setting `cfg.initial_time = 180` afterwards would leave
the thresholds stale at their 60+0 values — a silent wrong answer, which is the
exact failure mode this whole task exists to prevent. No current caller passes
`long_think_secs=` or `time_pressure_secs=` to the constructor (verified across
all nine `AnalysisConfig(` sites), so removing them as fields breaks nothing.

- [ ] **Step 1: Create the test package**

```bash
mkdir -p testing/cheat_detection
touch testing/cheat_detection/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `testing/cheat_detection/test_tc_config.py`:

```python
"""Guards the time-control derivation of the analyser's clock thresholds.

The load-bearing test here is `test_bullet_values_are_unchanged`. Both
derivations were chosen to reproduce the previously-hardcoded 60+0 constants
exactly (2.0s and 10.0s); if either drifts, every baseline, report and tracked
band table in the repo silently changes meaning, because they were all built
against those two numbers.
"""
import unittest

from cheat_detection.config import (
    LONG_THINK_FRACTION,
    TIME_PRESSURE_FRACTION,
    AnalysisConfig,
)


class TestTimeControlDerivation(unittest.TestCase):

    def test_bullet_values_are_unchanged(self):
        """The regression guarantee: at 60+0 the derived thresholds equal the
        constants every existing result in the repo was computed against."""
        cfg = AnalysisConfig()
        self.assertEqual(cfg.initial_time, 60.0)
        self.assertEqual(cfg.long_think_secs, 2.0)
        self.assertEqual(cfg.time_pressure_secs, 10.0)

    def test_fractions_are_the_documented_ones(self):
        self.assertAlmostEqual(LONG_THINK_FRACTION, 1 / 30)
        self.assertAlmostEqual(TIME_PRESSURE_FRACTION, 1 / 6)

    def test_derives_at_three_minutes(self):
        cfg = AnalysisConfig(initial_time=180.0)
        self.assertAlmostEqual(cfg.long_think_secs, 6.0)
        self.assertAlmostEqual(cfg.time_pressure_secs, 30.0)

    def test_instant_move_secs_is_absolute(self):
        """A one-second move means the same thing at any control -- it is a
        human motor-and-decision floor, not a share of the clock."""
        self.assertEqual(AnalysisConfig(initial_time=180.0).instant_move_secs,
                         AnalysisConfig(initial_time=60.0).instant_move_secs)

    def test_setting_initial_time_after_construction_redirives(self):
        """analyze.py's _config_from_args setattrs onto an already-built cfg.
        If the thresholds were snapshotted rather than derived on read, that
        path would silently keep bullet values at another time control."""
        cfg = AnalysisConfig()
        cfg.initial_time = 180.0
        self.assertAlmostEqual(cfg.long_think_secs, 6.0)
        self.assertAlmostEqual(cfg.time_pressure_secs, 30.0)

    def test_explicit_override_beats_derivation(self):
        cfg = AnalysisConfig(initial_time=180.0,
                             long_think_secs_override=4.5,
                             time_pressure_secs_override=12.0)
        self.assertEqual(cfg.long_think_secs, 4.5)
        self.assertEqual(cfg.time_pressure_secs, 12.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `venv/bin/python -m unittest testing.cheat_detection.test_tc_config -v`
Expected: FAIL — `ImportError: cannot import name 'LONG_THINK_FRACTION'`

- [ ] **Step 4: Implement the derivation**

In `cheat_detection/config.py`, add `Optional` to the typing import and add
these module-level constants directly above the `@dataclass` line:

```python
# Fractions of the initial clock that define the two clock-dependent feature
# thresholds. Both were chosen to reproduce the previously-hardcoded 60+0
# constants EXACTLY -- 60/30 = 2.0s and 60/6 = 10.0s -- so parameterising them
# leaves every existing bullet baseline, report and band table untouched.
#
# The /30 was already the documented intent. The /6 was derived backwards from
# the shipped 10.0 and lands on it exactly, which is reasonable evidence that a
# fraction is the right reading of "time pressure" rather than a coincidence.
# Open question flagged in the spec: the scramble may be partly *absolute* --
# 10s is roughly where humans stop calculating regardless of the starting clock
# -- in which case the right form is max(10.0, initial_time/6).
LONG_THINK_FRACTION = 1 / 30
TIME_PRESSURE_FRACTION = 1 / 6
```

Replace the `long_think_secs` and `time_pressure_secs` field definitions (and
their comment block) inside `AnalysisConfig` with:

```python
    # Initial clock in seconds for the corpus under analysis -- 60.0 is one
    # minute of bullet, the control everything in this repo was calibrated on.
    # Set it from --tc; the two thresholds below derive from it.
    initial_time: float = 60.0
    # Set these only to override the derivation; None = derive from
    # initial_time. They are properties rather than fields so that setting
    # initial_time after construction re-derives them (analyze.py's
    # _config_from_args mutates an already-built config).
    long_think_secs_override: Optional[float] = None
    time_pressure_secs_override: Optional[float] = None
```

Keep `blunder_wc_loss` where it is. Then add these two properties to the class,
directly above `cache_path`:

```python
    @property
    def long_think_secs(self) -> float:
        """emt above this counts as a "long think": the slow tail of the
        move-time distribution, the counterpart to instant_move_secs.

        Tracked because tuning against the fast tail alone hid a larger
        divergence -- the bot measured 0.067 against a human 0.115, and
        near-zero outside the midgame (opening 0.002 vs 0.031, endgame 0.007
        vs 0.044).
        """
        if self.long_think_secs_override is not None:
            return self.long_think_secs_override
        return self.initial_time * LONG_THINK_FRACTION

    @property
    def time_pressure_secs(self) -> float:
        """Clock below this = "time pressure" for the degradation features
        (acpl/blunders in the scramble)."""
        if self.time_pressure_secs_override is not None:
            return self.time_pressure_secs_override
        return self.initial_time * TIME_PRESSURE_FRACTION
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `venv/bin/python -m unittest testing.cheat_detection.test_tc_config -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Confirm nothing else constructed these as fields**

Run: `grep -rn "long_think_secs=\|time_pressure_secs=" --include=*.py .`
Expected: no output. If anything appears, it is a keyword argument to
`AnalysisConfig` that must be renamed to the `_override` form.

- [ ] **Step 7: Lint and commit**

```bash
venv/bin/ruff check cheat_detection/config.py testing/cheat_detection/test_tc_config.py
git add cheat_detection/config.py testing/cheat_detection/
git commit -m "feat(cheat_detection): derive clock thresholds from initial_time

long_think_secs = initial_time/30 and time_pressure_secs = initial_time/6,
both reproducing the shipped 60+0 constants (2.0s, 10.0s) exactly so every
existing baseline and report is untouched. Properties rather than fields so
that setting initial_time after construction re-derives them."
```

---

## Task 2: `--tc` flag on the analyser CLIs

**Files:**
- Modify: `cheat_detection/analyze.py:66-75` (`_config_from_args`) and its shared arg-adder
- Test: `testing/cheat_detection/test_tc_config.py` (extend)

**Interfaces:**
- Consumes: `AnalysisConfig.initial_time` from Task 1.
- Produces:
  - `cheat_detection.analyze.parse_tc_seconds(tc: str) -> float` — parses
    `"180+0"` to `180.0`; raises `ValueError` on anything unparseable.
  - `--tc` accepted by every `analyze.py` subcommand that builds a config.

- [ ] **Step 1: Write the failing test**

Append to `testing/cheat_detection/test_tc_config.py`:

```python
class TestParseTcSeconds(unittest.TestCase):

    def test_parses_base_and_increment(self):
        from cheat_detection.analyze import parse_tc_seconds
        self.assertEqual(parse_tc_seconds("180+0"), 180.0)
        self.assertEqual(parse_tc_seconds("60+0"), 60.0)

    def test_increment_is_optional(self):
        from cheat_detection.analyze import parse_tc_seconds
        self.assertEqual(parse_tc_seconds("300"), 300.0)

    def test_increment_does_not_change_the_base(self):
        """initial_time is the base clock; the increment is not folded in."""
        from cheat_detection.analyze import parse_tc_seconds
        self.assertEqual(parse_tc_seconds("180+2"), 180.0)

    def test_rejects_garbage(self):
        from cheat_detection.analyze import parse_tc_seconds
        for bad in ("", "-", "?", "blitz", "3+0min"):
            with self.assertRaises(ValueError):
                parse_tc_seconds(bad)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m unittest testing.cheat_detection.test_tc_config.TestParseTcSeconds -v`
Expected: FAIL — `ImportError: cannot import name 'parse_tc_seconds'`

- [ ] **Step 3: Implement the parser and wire the flag**

In `cheat_detection/analyze.py`, add near the top-level imports:

```python
import re

_TC_SECONDS_RE = re.compile(r"^(\d+)(?:\+(\d+))?$")


def parse_tc_seconds(tc: str) -> float:
    """Base clock in seconds from a "180+0"-style time control.

    The increment is parsed but deliberately discarded: initial_time means the
    starting clock, which is what the threshold fractions scale against.
    """
    m = _TC_SECONDS_RE.match((tc or "").strip())
    if not m:
        raise ValueError(f"bad time control {tc!r}; expected e.g. 180+0")
    return float(m.group(1))
```

In the function that adds the shared analysis arguments (the one already
defining `--min-moves`, `--test` and `--alpha`), add:

```python
    p.add_argument("--tc", default="60+0",
                   help="time control of the corpus, e.g. 180+0 (default "
                        "60+0). Sets the initial clock that long-think and "
                        "time-pressure thresholds derive from, and is checked "
                        "against each game's TimeControl header.")
    p.add_argument("--allow-tc-mismatch", action="store_true",
                   help="downgrade the TimeControl header check to a warning. "
                        "Mixing clocks muddies every timing feature, so this "
                        "is an escape hatch, not a normal mode.")
```

In `_config_from_args`, after the existing `setattr` loop and before the
`test_mode` block:

```python
    if getattr(args, "tc", None):
        cfg.initial_time = parse_tc_seconds(args.tc)
```

Note that `initial_time` is deliberately **not** added to the `setattr` loop's
attribute tuple — that loop copies raw argparse values, and `--tc` needs
parsing from a string first.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m unittest testing.cheat_detection.test_tc_config -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Verify the flag reaches the config end-to-end**

Run: `venv/bin/python -m cheat_detection.analyze report --help`
Expected: `--tc` and `--allow-tc-mismatch` both listed, no traceback.

- [ ] **Step 6: Lint and commit**

```bash
venv/bin/ruff check cheat_detection/analyze.py testing/cheat_detection/test_tc_config.py
git add cheat_detection/analyze.py testing/cheat_detection/test_tc_config.py
git commit -m "feat(cheat_detection): add --tc to set the corpus time control"
```

---

## Task 3: TimeControl header guard

**Files:**
- Modify: `cheat_detection/pgn_loader.py` (add error class and check function)
- Modify: `cheat_detection/pipeline.py:119` (apply in `iter_units`)
- Modify: `cheat_detection/elo_progression.py:102` (apply in `_collect_sequential`)
- Test: `testing/cheat_detection/test_tc_guard.py`

**Interfaces:**
- Consumes: `AnalysisConfig.initial_time` (Task 1), `args.allow_tc_mismatch` (Task 2).
- Produces:
  - `cheat_detection.pgn_loader.TimeControlMismatchError(Exception)`
  - `cheat_detection.pgn_loader.check_time_control(game: GameRecord, expected_secs: float, *, strict: bool = True) -> bool`
    — returns `True` when the game matches or cannot be checked; raises
    `TimeControlMismatchError` when it mismatches and `strict`; returns `False`
    (caller skips the game) when it mismatches and not `strict`.

**Why this exists:** CLAUDE.md's corpus policy already says to pin one exact
clock, because "mixing e.g. 30+0 with 60+0 muddies every timing feature". Today
that is a convention a human has to remember. This turns it into something the
code enforces, and it is the specific mistake this workstream is most exposed
to — the whole point is running the same analysis at a second control.

- [ ] **Step 1: Write the failing test**

Create `testing/cheat_detection/test_tc_guard.py`:

```python
"""The corpus-clock guard.

CLAUDE.md's corpus policy is to pin one exact time control, because mixing
clocks muddies every timing feature. This makes that policy enforceable rather
than remembered -- the failure it prevents is a silently blended population,
which produces numbers that look fine and mean nothing.
"""
import unittest

from cheat_detection.pgn_loader import (
    GameRecord,
    TimeControlMismatchError,
    check_time_control,
)


def _game(base_secs, tc="180+0"):
    return GameRecord(white="a", black="b", white_elo=2500, black_elo=2500,
                      time_control=tc, base_secs=base_secs, increment=0,
                      result="1-0", moves=[])


class TestCheckTimeControl(unittest.TestCase):

    def test_match_passes(self):
        self.assertTrue(check_time_control(_game(180), 180.0))

    def test_mismatch_raises_when_strict(self):
        with self.assertRaises(TimeControlMismatchError):
            check_time_control(_game(60, "60+0"), 180.0)

    def test_error_names_both_controls(self):
        """The message has to be actionable -- which corpus, which --tc."""
        with self.assertRaises(TimeControlMismatchError) as ctx:
            check_time_control(_game(60, "60+0"), 180.0)
        msg = str(ctx.exception)
        self.assertIn("60", msg)
        self.assertIn("180", msg)

    def test_mismatch_skips_when_not_strict(self):
        self.assertFalse(check_time_control(_game(60, "60+0"), 180.0,
                                            strict=False))

    def test_unknown_time_control_passes(self):
        """A missing or unparseable header cannot be checked, so it must not
        block analysis -- absence of evidence is not a mismatch."""
        self.assertTrue(check_time_control(_game(None, "-"), 180.0))

    def test_bullet_corpus_against_default_config_passes(self):
        """The existing 60+0 corpora must keep working untouched."""
        self.assertTrue(check_time_control(_game(60, "60+0"), 60.0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m unittest testing.cheat_detection.test_tc_guard -v`
Expected: FAIL — `ImportError: cannot import name 'TimeControlMismatchError'`

- [ ] **Step 3: Implement the guard**

Add to `cheat_detection/pgn_loader.py`, below the `GameRecord` dataclass:

```python
class TimeControlMismatchError(Exception):
    """A game's TimeControl header disagrees with the configured --tc."""


def check_time_control(game: GameRecord, expected_secs: float, *,
                       strict: bool = True) -> bool:
    """Whether this game's clock matches the analysis configuration.

    Returns True to analyse the game, False to skip it. Raises when the game
    mismatches and `strict` -- the default, because a blended corpus produces
    timing features that look plausible and mean nothing, which is worse than
    a crash.

    A game whose TimeControl is missing or unparseable (`base_secs is None`)
    passes: it cannot be checked, and absence of evidence is not a mismatch.
    """
    if game.base_secs is None:
        return True
    if float(game.base_secs) == float(expected_secs):
        return True
    if strict:
        raise TimeControlMismatchError(
            f"game has TimeControl {game.time_control!r} "
            f"({game.base_secs}s base) but the analysis is configured for "
            f"{expected_secs:g}s. Pin one clock -- mixing them muddies every "
            f"timing feature. Pass --tc {int(game.base_secs)}+0 to analyse "
            f"this corpus, or --allow-tc-mismatch to skip off-control games."
        )
    return False
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `venv/bin/python -m unittest testing.cheat_detection.test_tc_guard -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Apply the guard at both game-consumption sites**

In `cheat_detection/pipeline.py`, add `check_time_control` to the existing
`from .pgn_loader import ...` line, and inside `iter_units`'s loop over
`iter_games` (line ~119), immediately after the `for gi, game in enumerate(...)`
line:

```python
        if not check_time_control(game, cfg.initial_time,
                                  strict=getattr(cfg, "strict_tc", True)):
            continue
```

In `cheat_detection/elo_progression.py`, add `check_time_control` to its
`from .pgn_loader import GameRecord, iter_games` line, and add the identical
three lines inside `_collect_sequential`'s `for gi, game in enumerate(...)`
loop, immediately before the existing `if gi_filter is not None` check.

Add the backing field to `AnalysisConfig` in `cheat_detection/config.py`,
directly below `time_pressure_secs_override`:

```python
    # False downgrades the TimeControl header check to a skip; see
    # pgn_loader.check_time_control. Set from --allow-tc-mismatch.
    strict_tc: bool = True
```

And in `analyze.py:_config_from_args`, below the `--tc` block from Task 2:

```python
    if getattr(args, "allow_tc_mismatch", False):
        cfg.strict_tc = False
```

- [ ] **Step 6: Verify the guard fires on a real corpus**

This checks the guard against actual data, not a fixture. The bullet corpus is
60+0, so analysing it as 180+0 must fail fast.

```bash
venv/bin/python -m cheat_detection.analyze report \
  --pgn cheat_detection/corpora/bullet_1plus0_2300_plus.pgn \
  --baseline cheat_detection/baselines/bullet_1plus0_2300_plus.json \
  --tc 180+0 --max-games 5 --player nobody
```

Expected: `TimeControlMismatchError` mentioning both 60 and 180.

Then confirm the default path is unaffected:

```bash
venv/bin/python -m cheat_detection.analyze report \
  --pgn cheat_detection/corpora/bullet_1plus0_2300_plus.pgn \
  --baseline cheat_detection/baselines/bullet_1plus0_2300_plus.json \
  --max-games 5 --player nobody
```

Expected: no `TimeControlMismatchError` (the "no qualifying games" message for
`--player nobody` is the expected outcome and is fine).

- [ ] **Step 7: Lint and commit**

```bash
venv/bin/ruff check cheat_detection/pgn_loader.py cheat_detection/pipeline.py \
  cheat_detection/elo_progression.py cheat_detection/config.py \
  cheat_detection/analyze.py testing/cheat_detection/test_tc_guard.py
git add cheat_detection/ testing/cheat_detection/
git commit -m "feat(cheat_detection): enforce one corpus clock per analysis

Turns CLAUDE.md's pin-one-clock corpus policy into something the code checks.
A blended corpus produces timing features that look plausible and mean
nothing; --allow-tc-mismatch downgrades to skipping off-control games."
```

---

## Task 4: Thread the config through elo_progression's stats

**Files:**
- Modify: `cheat_detection/elo_progression.py:183-184` (delete the snapshots), `_stats`, `render`, `main`
- Test: `testing/cheat_detection/test_elo_progression_tc.py`

**Interfaces:**
- Consumes: `AnalysisConfig` (Tasks 1-2).
- Produces:
  - `_stats(mfeats: list[MoveFeatures], cfg: AnalysisConfig) -> dict[str, float]`
  - `render(by_band: dict, cfg: AnalysisConfig, min_n: int = 30) -> str`

**Why this is its own task:** `INSTANT_SECS` and `LONG_THINK_SECS` are captured
at *import time* from a default `AnalysisConfig()`. Tasks 1-3 make the config
TC-aware, but this module would keep computing `long_think_rate` against a
hardcoded 2.0s no matter what `--tc` said — producing a 180+0 band table with a
bullet threshold silently baked into one column. That is precisely the class of
error the spec's pre-committed reads depend on not happening.

- [ ] **Step 1: Write the failing test**

Create `testing/cheat_detection/test_elo_progression_tc.py`:

```python
"""The band table's long-think column must follow --tc.

INSTANT_SECS/LONG_THINK_SECS were module-level snapshots of a default config,
so before this change the 180+0 table would have counted "long thinks" against
bullet's 2.0s threshold -- a wrong number in a tracked calibration target that
nothing downstream would have caught.
"""
import unittest

from cheat_detection.config import AnalysisConfig
from cheat_detection.elo_progression import _stats
from cheat_detection.features import MoveFeatures


def _move(emt):
    return MoveFeatures(
        ply=0, phase="middlegame", rank=1, within_topk=True,
        matched_top1=True, matched_top2=True, matched_top3=True,
        cp_loss=0.0, wc_loss=0.0, ambiguity=1, sharpness=0.0,
        n_legal=30, is_blunder=False, emt=emt, clock_before=100.0,
    )


class TestStatsFollowConfig(unittest.TestCase):
    """Move times of 1.5s and 4.0s: at 60+0 (threshold 2.0s) one is a long
    think; at 180+0 (threshold 6.0s) neither is."""

    MOVES = [_move(1.5), _move(4.0)]

    def test_bullet_threshold(self):
        s = _stats(self.MOVES, AnalysisConfig(initial_time=60.0))
        self.assertAlmostEqual(s["long_think_rate"], 0.5)

    def test_three_minute_threshold(self):
        s = _stats(self.MOVES, AnalysisConfig(initial_time=180.0))
        self.assertAlmostEqual(s["long_think_rate"], 0.0)

    def test_instant_rate_is_unaffected_by_time_control(self):
        moves = [_move(0.5), _move(4.0)]
        for t in (60.0, 180.0):
            s = _stats(moves, AnalysisConfig(initial_time=t))
            self.assertAlmostEqual(s["instant_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
```

That constructor matches `MoveFeatures` as defined at
`cheat_detection/features.py:55-71` (all fields required except `kind`). If it
has since gained a field, fix the `_move` helper — do not change
`MoveFeatures` itself.

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m unittest testing.cheat_detection.test_elo_progression_tc -v`
Expected: FAIL — `TypeError: _stats() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Thread the config through**

In `cheat_detection/elo_progression.py`:

1. Delete the two module-level lines:

```python
INSTANT_SECS = AnalysisConfig().instant_move_secs
LONG_THINK_SECS = AnalysisConfig().long_think_secs
```

2. Change `def _stats(mfeats: list[MoveFeatures]) -> dict[str, float]:` to
   `def _stats(mfeats: list[MoveFeatures], cfg: AnalysisConfig) -> dict[str, float]:`
   and inside it replace `INSTANT_SECS` with `cfg.instant_move_secs` and
   `LONG_THINK_SECS` with `cfg.long_think_secs`.

3. Change `def render(by_band, min_n: int = 30) -> str:` to
   `def render(by_band, cfg: AnalysisConfig, min_n: int = 30) -> str:`.

4. In `render`, pass `cfg` to **every** `_stats(...)` call. There are six, one
   per table section (overall, sharpness buckets, eff-mob buckets, phases, and
   the two mix/analytics sections if they call it). Find them all with:

```bash
grep -n "_stats(" cheat_detection/elo_progression.py
```

5. In `render`, the header line interpolating `{INSTANT_SECS:g}` becomes
   `{cfg.instant_move_secs:g}`. Add a line recording the control the table was
   built at, immediately after it — without this the tracked artefact does not
   say which clock it describes:

```python
    lines.append(f"Built at a {cfg.initial_time:g}s initial clock; a \"long "
                 f"think\" is emt > {cfg.long_think_secs:g}s "
                 f"(initial_time/30).\n")
```

6. In `main`, add the `--tc` argument and use it:

```python
    ap.add_argument("--tc", default="60+0",
                    help="corpus time control, e.g. 180+0 (default 60+0)")
```

```python
    cfg = AnalysisConfig(workers=args.workers,
                         initial_time=parse_tc_seconds(args.tc))
```

with `from .analyze import parse_tc_seconds` added to the imports, and update
the `render(by_band)` call to `render(by_band, cfg)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `venv/bin/python -m unittest testing.cheat_detection.test_elo_progression_tc -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Verify no stale references remain**

Run: `grep -n "INSTANT_SECS\|LONG_THINK_SECS" cheat_detection/elo_progression.py`
Expected: no output.

- [ ] **Step 6: Lint and commit**

```bash
venv/bin/ruff check cheat_detection/elo_progression.py \
  testing/cheat_detection/test_elo_progression_tc.py
git add cheat_detection/elo_progression.py testing/cheat_detection/
git commit -m "fix(cheat_detection): band table thresholds follow --tc

INSTANT_SECS/LONG_THINK_SECS were import-time snapshots of a default config,
so a 180+0 table would have counted long thinks against bullet's 2.0s."
```

---

## Task 5: Phase 0 close-out — prove inertness at 60+0

**Files:**
- Modify: `cheat_detection/.gitignore` (negation for the new band table)
- No source changes expected. If this task needs one, an earlier task was wrong.

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: a verified-inert Phase 0, and a tracked path for the Phase 1 output.

**This is the gate.** Every existing bullet result in the repo was computed
against `long_think_secs=2.0` and `time_pressure_secs=10.0`. If Phase 0 moved
either, every baseline, every tracked band table and every conclusion in the
strength-dial spec silently changed meaning. The unit tests assert the
constants; this task asserts the *pipeline* built on them.

- [ ] **Step 1: Run the full analyser test suite**

Run: `venv/bin/python -m unittest discover -s testing/cheat_detection -v`
Expected: PASS, 15 tests across three files.

- [ ] **Step 2: Confirm untouched suites are still green**

```bash
venv/bin/python -m unittest discover -s testing/engine_components
venv/bin/python -m unittest discover -s testing/client
```

Expected: `engine_components` fully green. `client` shows 7 of 34 as
`_FailedTest` on `fastgrab._linux_x11` if this machine has no X11 — that is an
environment limitation, not a regression. Compare against a clean tree before
believing otherwise.

Do **not** run the engine parity harness: no engine code was touched, and it
false-fails under load.

- [ ] **Step 3: Regenerate a bullet report and diff it**

```bash
venv/bin/python -m cheat_detection.analyze report \
  --pgn cheat_detection/runs/strength_dial/phasef_control_a_seed870000.pgn \
  --player SimBotWhite SimBotBlack \
  --baseline cheat_detection/baselines/bullet_1plus0_2300_plus.json \
  --workers 3 --threads 2 \
  --out-json /tmp/claude-1001/-home-james-Documents-Projects-humanchessplayer2/074d5f1d-fb14-4139-b274-6a8a263452aa/scratchpad/repf_control_a_after.json
```

Then compare against the committed pre-change report:

```bash
venv/bin/python - <<'EOF'
import json
a = json.load(open("cheat_detection/runs/strength_dial/repf_control_a.report.json"))
b = json.load(open("/tmp/claude-1001/-home-james-Documents-Projects-humanchessplayer2/074d5f1d-fb14-4139-b274-6a8a263452aa/scratchpad/repf_control_a_after.json"))
def feats(r):
    for k in ("features", "bot", "bot_features"):
        if isinstance(r.get(k), dict):
            return r[k]
    return r
fa, fb = feats(a), feats(b)
bad = []
for k, va in fa.items():
    vb = fb.get(k)
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        if abs(va - vb) > 1e-9:
            bad.append((k, va, vb))
print("CHANGED:", bad or "none")
assert not bad, "Phase 0 was supposed to be inert at 60+0"
EOF
```

Expected: `CHANGED: none`.

If `long_think_rate` or `blunder_rate_timepressure` moved, the derivation is
wrong — check `LONG_THINK_FRACTION` and `TIME_PRESSURE_FRACTION` against Task
1's constants before touching anything else. If the PGN or report file is
absent (they are gitignored run output), substitute any other bullet PGN with a
committed report; if none exists, note that in the commit message and rely on
Steps 1-2.

If no pre-change report survives at all, run the same command twice — once
against a `git stash` of the changes, once with them applied — and diff those
two instead. The claim under test is "output is identical", and either route
establishes it.

- [ ] **Step 4: Track the forthcoming 180+0 band table**

The band tables are tracked by gitignore negation. Add the new one now, so the
Phase 1 run cannot produce an untracked artefact that is then lost. Append to
`cheat_detection/.gitignore`, directly after the existing
`!runs/elo_progression/report_longthink.md` line:

```
!runs/elo_progression/report_timing_180.md
```

- [ ] **Step 5: Commit**

```bash
git add cheat_detection/.gitignore
git commit -m "chore(cheat_detection): track the forthcoming 180+0 band table

Phase 0 verified inert at 60+0: full analyser suite green and a regenerated
bullet report byte-identical to the pre-change one."
```

---

## Task 6: Phase 1 — the 180+0 human band table

**Files:**
- Create: `cheat_detection/runs/elo_progression/run_band_table_180.py`
- Modify: `cheat_detection/.gitignore` (negation for the driver script)
- Output: `cheat_detection/runs/elo_progression/report_timing_180.md` (tracked)

**Interfaces:**
- Consumes: `--tc` (Task 2), the guard (Task 3), cfg-threaded stats (Task 4).
- Produces: the band table that Phase 2's design will be written against.

**Prerequisite, and it is the long pole:** there is no 180+0 corpus and no
local Lichess dump. A dump download must happen first, and it is larger than
the analysis it feeds. Do not start this task without one.

- [ ] **Step 1: Fetch the corpus**

`--band` takes `MIN MAX COUNT` with `+` for unbounded, and in band mode `--out`
is a *stem* the per-band filename derives from — so this writes
`blitz_3plus0_2300_+.pgn`, not the literal `--out` path.

```bash
zstd -dc <dump>.pgn.zst | venv/bin/python -m cheat_detection.fetch_corpus \
    --tc 180+0 --any-player --band 2300 + 40000 \
    --out cheat_detection/corpora/blitz_3plus0.pgn
```

`--any-player` means at least one player is 2300+, matching the bullet corpus
policy; the off-band opponent is excluded at analysis time by `--rating`.

- [ ] **Step 2: Sanity-check the corpus before spending an hour on it**

```bash
grep -c "^\[TimeControl" cheat_detection/corpora/blitz_3plus0_2300_+.pgn
grep "^\[TimeControl" cheat_detection/corpora/blitz_3plus0_2300_+.pgn | sort -u
```

Expected: one distinct value, `[TimeControl "180+0"]`. Anything else means the
fetch filter did not apply and the guard will stop the run anyway.

- [ ] **Step 3: Write the driver**

Create `cheat_detection/runs/elo_progression/run_band_table_180.py`:

```python
"""Phase 1: the 180+0 human rating-band table.

The second calibration point for the strength dial. Everything the dial knows
today is 60+0-specific: `common/strength_profiles.py` fits
`mean_emt = 0.1964*quickness + 0.6234` at one minute, and its own docstring
says longer controls "need their own band table and their own fit". This
produces that table.

**The pre-committed reads** (see the spec -- written before the data existed so
a marginal result cannot be talked into confirming the hypothesis afterwards).
Bullet references are from report_timing.md, overall table:

  1. Mean emt span across bands. Bullet: 1.26s (2100-2299) -> 1.00s (2800+),
     a 21% proportional drop. PREDICTED materially smaller at 180+0. If the
     span is >= bullet's, the hypothesis that pace decouples from rating is
     WRONG, quickness stays the primary axis, and the follow-up is a re-fit
     rather than a search for new levers.
  2. Top-1 match span. Bullet's adjacent bands differ by ~0.005, which is what
     caps the dial at ~200-300 Elo resolution. PREDICTED larger at 180+0 --
     this is what would make DIFFICULTY and eval_noise_scale worth sweeping
     here despite both failing at 1+0.
  3. Per-phase mean emt, opening-to-midgame ratio. The bot's
     `(base_time**0.2)/2` opening envelope makes its opening near-invariant
     across controls (0.57s at 60+0 vs 0.86s at 600+0, against a 10x clock);
     humans are predicted not to be. Prices an envelope refit.

Nothing bot-side runs here. No simulation arms, no sweeps, no changes to
strength_profiles.py.

~65 min plus the corpus fetch. Run from repo root:
    venv/bin/python cheat_detection/runs/elo_progression/run_band_table_180.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

CORPUS = ROOT / "cheat_detection/corpora/blitz_3plus0_2300_+.pgn"
OUT_MD = OUT_DIR / "report_timing_180.md"
TC = "180+0"
WORKERS = 6  # ~cores/2 at the default 2 engine threads


def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(
            f"no corpus at {CORPUS}. Fetch it first -- see the plan's Task 6 "
            f"step 1. This needs a Lichess dump and is the long pole."
        )
    if OUT_MD.exists():
        print(f"{OUT_MD.name} already present; delete it to re-run.")
        return

    start = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "cheat_detection.elo_progression",
         "--pgn", str(CORPUS), "--tc", TC,
         "--workers", str(WORKERS), "--out-md", str(OUT_MD)],
        cwd=ROOT, check=False)
    mins = (time.time() - start) / 60
    if proc.returncode != 0:
        raise SystemExit(f"band table FAILED rc={proc.returncode} "
                         f"after {mins:.1f} min")
    print(f"Wrote {OUT_MD} in {mins:.1f} min")
    print("Now read it against the three pre-committed reads in this file's "
          "docstring -- including the one that kills the hypothesis.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Track the driver**

Append to `cheat_detection/.gitignore`, beside the other negations:

```
!runs/elo_progression/run_band_table_180.py
```

- [ ] **Step 5: Smoke-test on a small slice before the full run**

```bash
venv/bin/python -m cheat_detection.elo_progression \
  --pgn "cheat_detection/corpora/blitz_3plus0_2300_+.pgn" \
  --tc 180+0 --max-games 50 --workers 2 \
  --out-md /tmp/claude-1001/-home-james-Documents-Projects-humanchessplayer2/074d5f1d-fb14-4139-b274-6a8a263452aa/scratchpad/smoke_180.md
```

Expected: completes without a `TimeControlMismatchError`, and the output's
header line reads `Built at a 180s initial clock` with a long-think threshold
of `6s`. If it says 2s, Task 4 did not take.

- [ ] **Step 6: Run the full band table**

```bash
venv/bin/python cheat_detection/runs/elo_progression/run_band_table_180.py
```

Expected: ~65 min, writes `report_timing_180.md`.

- [ ] **Step 7: Commit the tracked artefact**

```bash
git add cheat_detection/.gitignore \
  cheat_detection/runs/elo_progression/run_band_table_180.py \
  cheat_detection/runs/elo_progression/report_timing_180.md
git commit -m "feat(cheat_detection): 180+0 human rating-band table

The second calibration point for the strength dial. Read against the three
pre-committed comparisons in the driver docstring, including the one that
falsifies the pace-decoupling hypothesis."
```

- [ ] **Step 8: Report the three pre-committed reads**

Produce a comparison table — bullet figure, 180+0 figure, and whether each
prediction held — and state plainly whether read 1 falsified the hypothesis.
Do not design Phase 2 in this step; the spec deliberately stops here, and the
next design round is brainstormed against these numbers.

---

## Self-Review

**Spec coverage.** Phase 0's three hardcoded sites: derived thresholds (Task
1), `--tc` (Task 2), header guard (Task 3), `elo_progression` snapshots (Task
4). The `instant_move_secs`-stays-absolute requirement is asserted in Tasks 1
and 4. Per-TC baseline filename is not a code change — it is a `--out` argument
at Phase 2 time, and no baseline is built in this plan. Phase 1's corpus,
run, tracking and pre-committed reads are Task 6. The TC profile table is
correctly absent: the spec explicitly defers it.

**Known gap, deliberate.** The spec's open question about
`time_pressure_secs` possibly needing `max(10.0, initial_time/6)` is recorded
in Task 1's code comment but not implemented. It cannot be settled before the
Phase 1 data exists, and implementing a guess would be worse than carrying the
question.

**Type consistency.** `parse_tc_seconds` returns `float`, matching
`initial_time: float`, and is imported by both `analyze.py` (defines it) and
`elo_progression.py` (Task 4). `check_time_control` returns `bool` and is
called identically at both sites. `_stats(mfeats, cfg)` and `render(by_band,
cfg, min_n)` are used with those exact signatures in Task 4 steps 3 and 4.
`strict_tc` is defined in Task 3 step 5 and read via `getattr(cfg,
"strict_tc", True)` at both call sites.

**One instruction the implementer must not skip:** Task 4 step 1 says to check
`MoveFeatures`'s real field list before running the test. I wrote that
constructor from the field names used elsewhere in the codebase rather than
from the dataclass definition, so it may not match exactly.
