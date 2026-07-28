"""Phase B of the strength-dial fit: the two accuracy axes.

Fits the rating mappings for the knobs that own top-1 match and blunder rate,
so they can join `quickness` in common/strength_profiles.CALIBRATED_KNOBS.
Today the dial sets pace only; these arms are what turn it into a strength
dial. See docs/superpowers/specs/2026-07-27-strength-dial-design.md.

Two axes, one factor at a time (this is a Jacobian, not a factorial):

  * eval_noise_scale -> t1_rate. Phase A already covered 0.55 / 0.75 / 0.95
    and showed the response saturating above 0.75 (0.75 -> 0.95 moved t1 by
    0.1 se). Those three arms are reused as-is; only the LOW side is missing,
    and it decides the dial's honest top end -- the live region's slope
    implies an unreachable ~0.03 to hit 2850's t1, so where the curve actually
    flattens is the open question.

  * midgame_breadth_strength_bonus -> blunder_rate. The existing 0.0380 ->
    0.0169 figure came from ADJUDICATED games and is not comparable to the
    complete-game human band table the dial is calibrated against, so it is
    re-measured here rather than imported.

Conventions match Phase A exactly, or the arms are not comparable with it:
pure self-play (win-probability features saturate in decided positions, so
unequal score rates cannot be compared), complete games (`--simulate-full`,
because the band table is built from complete human games and two of the four
dial targets are time-economy features), 60+0 -- that is 60 *seconds*, one
minute of bullet -- and 150 games per arm.

Run from repo root:
    venv/bin/python cheat_detection/runs/strength_dial/run_phase_b.py

Resumable: an arm whose PGN already exists is skipped.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUT_DIR / "phase_b.log"

GAMES = 150
WORKERS = 6  # ~cores/3 (see simulation/run.py docstring); adjust per-machine
TC = "60+0"

BASE_NOISE = 0.75    # common/constants.py HUMAN_EVAL_NOISE_SCALE

# (arm, extra simulation.run flags, seed). Seeds are fixed per arm so the list
# can be trimmed or split across machines without shifting any other arm.
ARMS = [
    # Low-noise side: where does the t1 response flatten?
    ("noise_040", ["--eval-noise-scale", "0.40"], 750000),
    ("noise_025", ["--eval-noise-scale", "0.25"], 760000),
    # Breadth -> blunder rate, re-measured on complete games.
    ("breadth_1", ["--midgame-breadth-bonus", "1"], 770000),
    ("breadth_2", ["--midgame-breadth-bonus", "2"], 780000),
    ("breadth_3", ["--midgame-breadth-bonus", "3"], 790000),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_arm(arm: str, flags: list[str], seed: int) -> Path:
    pgn = OUT_DIR / f"phaseb_{arm}_seed{seed}.pgn"
    if pgn.exists():
        log(f"{arm}: PGN already present, skipping ({pgn.name})")
        return pgn

    cmd = [
        sys.executable, "-m", "simulation.run",
        "--games", str(GAMES), "--tc", TC, "--seed", str(seed),
        "--workers", str(WORKERS), "--simulate-full",
        *flags,
        "--out", str(pgn), "--plain",
    ]
    log(f"{arm}: {' '.join(flags)} seed={seed}")
    start = time.time()
    # Stream progress rather than capturing it: an arm runs for most of an
    # hour, and a run whose progress is invisible cannot be told from a hang.
    progress = OUT_DIR / f"phaseb_{arm}.progress"
    with open(progress, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                              check=False)  # returncode handled below
    mins = (time.time() - start) / 60
    if proc.returncode != 0:
        log(f"{arm}: FAILED rc={proc.returncode} after {mins:.1f} min "
            f"(see {progress.name})")
        raise SystemExit(f"arm {arm} failed")
    log(f"{arm}: done in {mins:.1f} min -> {pgn.name}")
    return pgn


def main() -> None:
    log(f"Phase B start: {len(ARMS)} arms x {GAMES} games, {TC}, complete "
        f"games, workers={WORKERS}")
    log(f"(Phase A's noise {BASE_NOISE}/0.55/0.95 arms are reused, not re-run.)")
    for arm, flags, seed in ARMS:
        run_arm(arm, flags, seed)
    log("Phase B complete. Analyse against "
        "baselines/bullet_1plus0_2300_plus.json -- the COMPLETE-game "
        "baseline, not a truncated one.")


if __name__ == "__main__":
    main()
