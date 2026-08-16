"""Phase E: clear the opening-book fast path's safety guard at a sample size
that can actually measure it.

The fast path (OPENING_BOOK_FAST_PATH) is validated on its intended effect --
opening instant-move rate 0.373 -> 0.567 against a human 0.565 (Phase D) --
but ships OFF, because it queues a book-derived premove and CLAUDE.md flags
added premove volume as twice-proven poison. That failure mode shows up in
`blunder_rate_timepressure`, and Phase C established that blunder-rate
comparisons carry no information at 150 games/arm: two identical-config arms
measured 0.0483 and 0.0396, a gap wider than either was from its baseline.

**Design note -- why 4 arms of 300 and not 2 of 600.** Same total compute,
but this way each condition has two independent seeds. That gives a pooled 600
games per condition AND an empirical within-condition spread, so the error bar
is measured rather than assumed. Assuming it is precisely the mistake that
produced the two retractions recorded in the strength-dial spec: the binomial
standard error (~0.0017/arm) understated the real between-seed spread
(~0.0062) by about 3.6x, and two confident conclusions were built on it.

Read the result as: is |control_pooled - fastpath_pooled| small relative to
the observed within-condition spread? If the two control seeds differ from
each other by as much as control differs from fastpath, the guard is still
unmeasured and no amount of reasoning about the mean rescues it.

Judged on:
  * `blunder_rate_timepressure` -- the premove-poison guard, primary
  * `blunder_rate`, `t1_rate`, `t2_rate` -- did anything else regress
  * `instant_move_rate` and per-phase instant rate -- confirm Phase D's gain
    survives at this sample size

Conventions match Phases A-D or the arms are not comparable with them: pure
self-play, complete games (`--simulate-full`), 60+0 -- 60 *seconds*, one
minute of bullet. Both conditions carry current shipped defaults, which
include REEVAL_ORDER="human".

~4.2h of simulation plus analysis. Run from repo root:
    venv/bin/python cheat_detection/runs/strength_dial/run_phase_e.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUT_DIR / "phase_e_driver.log"

GAMES = 300
WORKERS = 6  # ~cores/3 (see simulation/run.py docstring); adjust per-machine
TC = "60+0"
BASELINE = "cheat_detection/baselines/bullet_1plus0_2300_plus.json"

# (arm, flags, seed). Two seeds per condition -- see the design note above.
ARMS = [
    ("control_a",  [], 870000),
    ("fastpath_a", ["--opening-book-fast-path"], 890000),
    ("control_b",  [], 880000),
    ("fastpath_b", ["--opening-book-fast-path"], 900000),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _run(cmd: list[str], label: str, out: Path) -> None:
    start = time.time()
    # Stream progress rather than capturing it: an arm runs for over an hour,
    # and a run whose progress is invisible cannot be told from a hang.
    with open(out, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                              check=False)  # returncode handled below
    mins = (time.time() - start) / 60
    if proc.returncode != 0:
        log(f"{label}: FAILED rc={proc.returncode} after {mins:.1f} min "
            f"(see {out.name})")
        raise SystemExit(f"{label} failed")
    log(f"{label}: done in {mins:.1f} min")


def run_arm(arm: str, flags: list[str], seed: int) -> None:
    pgn = OUT_DIR / f"phasef_{arm}_seed{seed}.pgn"
    report = OUT_DIR / f"repf_{arm}.report.json"
    if pgn.exists():
        log(f"{arm}: PGN already present, skipping sim ({pgn.name})")
    else:
        log(f"{arm}: {' '.join(flags) or '(shipped defaults)'} seed={seed}")
        _run([sys.executable, "-m", "simulation.run",
              "--games", str(GAMES), "--tc", TC, "--seed", str(seed),
              "--workers", str(WORKERS), "--simulate-full", *flags,
              "--out", str(pgn), "--plain"],
             f"{arm} sim", OUT_DIR / f"phasef_{arm}.progress")

    if report.exists():
        log(f"{arm}: report already present, skipping analysis")
        return
    _run([sys.executable, "-m", "cheat_detection.analyze", "report",
          "--pgn", str(pgn), "--player", "SimBotWhite", "SimBotBlack",
          "--baseline", BASELINE, "--workers", "3", "--threads", "2",
          "--out-md", str(OUT_DIR / f"repf_{arm}.md"),
          "--out-json", str(report)],
         f"{arm} analysis", OUT_DIR / f"phasef_{arm}.analysis")


def main() -> None:
    log(f"Phase E start: {len(ARMS)} arms x {GAMES} games, {TC}, complete "
        f"games, workers={WORKERS}")
    log("Two seeds per condition: the within-condition spread is the error "
        "bar, and it is measured here rather than assumed.")
    for arm, flags, seed in ARMS:
        run_arm(arm, flags, seed)
    log("Phase E complete. Compare |control_pooled - fastpath_pooled| against "
        "the observed within-condition spread before concluding anything.")


if __name__ == "__main__":
    main()
