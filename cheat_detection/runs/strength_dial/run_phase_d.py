"""Phase D: validate the opening-book fast path at a sample size that can
price it.

The fast path (OPENING_BOOK_FAST_PATH, off by default) consults the opening
book before calculate_analytics rather than after, so a memorised move stops
paying for a full-width multipv scan plus an uncapped depth-12 sharpness scan.
It is the only lever that can raise the opening instant-move rate: Phase A
showed the engine's requested opening think time already sits below the
per-move compute floor, so no pacing knob touches it.

A 14-game smoke test measured opening instant rate 0.389 -> 0.567 against a
human 2100-2299 figure of 0.565, and compute-bypass moves 31.0% -> 39.1%. This
re-runs it at 150 games/arm so the aggregate number can actually be read: the
smoke test moved aggregate instant rate 0.279 -> 0.297 at about 1 se, which is
suggestive of nothing on its own.

Both arms carry the current shipped defaults, which now include
REEVAL_ORDER="human" (12b322d) -- so the control here is NOT the older Phase A
baseline, which predates that change.

**What this can and cannot settle.** Instant rate is measurable at 150
games/arm (se ~0.0045 against an expected opening effect near 0.18). The
safety question is not. The fast path queues a book-derived premove, and
CLAUDE.md flags added premove volume as twice-proven poison whose failure mode
shows up in `blunder_rate_timepressure` -- a blunder-rate metric, and Phase C
established that blunder-rate comparisons carry no information at this sample
size (real between-seed spread ~0.0062 against a binomial ~0.0017). So a clean
guard reading here is NOT evidence of safety; clearing that needs ~600
games/arm. Do not read this run as licence to flip the default on.

Conventions match Phases A-C: pure self-play, complete games
(`--simulate-full`), 60+0 -- 60 *seconds*, one minute of bullet -- 150 games.

Run from repo root:
    venv/bin/python cheat_detection/runs/strength_dial/run_phase_d.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUT_DIR / "phase_d_driver.log"

GAMES = 150
WORKERS = 6  # ~cores/3 (see simulation/run.py docstring); adjust per-machine
TC = "60+0"
BASELINE = "cheat_detection/baselines/bullet_1plus0_2300_plus.json"

# (arm, flags, seed). Seeds fixed per arm so the list can be trimmed.
ARMS = [
    ("control",  [], 850000),
    ("fastpath", ["--opening-book-fast-path"], 860000),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _run(cmd: list[str], label: str, out: Path) -> None:
    start = time.time()
    # Stream progress rather than capturing it: an arm runs for most of an
    # hour, and a run whose progress is invisible cannot be told from a hang.
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
    pgn = OUT_DIR / f"phasee_{arm}_seed{seed}.pgn"
    report = OUT_DIR / f"repe_{arm}.report.json"
    if pgn.exists():
        log(f"{arm}: PGN already present, skipping sim ({pgn.name})")
    else:
        log(f"{arm}: {' '.join(flags) or '(shipped defaults)'} seed={seed}")
        _run([sys.executable, "-m", "simulation.run",
              "--games", str(GAMES), "--tc", TC, "--seed", str(seed),
              "--workers", str(WORKERS), "--simulate-full", *flags,
              "--out", str(pgn), "--plain"],
             f"{arm} sim", OUT_DIR / f"phasee_{arm}.progress")

    if report.exists():
        log(f"{arm}: report already present, skipping analysis")
        return
    _run([sys.executable, "-m", "cheat_detection.analyze", "report",
          "--pgn", str(pgn), "--player", "SimBotWhite", "SimBotBlack",
          "--baseline", BASELINE, "--workers", "3", "--threads", "2",
          "--out-md", str(OUT_DIR / f"repe_{arm}.md"),
          "--out-json", str(report)],
         f"{arm} analysis", OUT_DIR / f"phasee_{arm}.analysis")


def main() -> None:
    log(f"Phase D start: {len(ARMS)} arms x {GAMES} games, {TC}, complete games")
    for arm, flags, seed in ARMS:
        run_arm(arm, flags, seed)
    log("Phase D complete. Judge on instant_move_rate (aggregate and per "
        "phase). blunder_rate_timepressure is NOT readable at this sample "
        "size -- see the module docstring before treating it as a guard.")


if __name__ == "__main__":
    main()
