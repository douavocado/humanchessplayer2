"""Phase A of the strength-dial fit: baseline + Jacobian probe.

Measures how the two candidate dial knobs -- eval_noise_scale (accuracy) and
quickness (speed) -- move the four aggregate human-likeness features the dial
targets: top-1 match rate, blunder rate, mean elapsed move time, and instant
rate. See docs/superpowers/specs/2026-07-27-strength-dial-design.md.

Two properties of these runs matter and differ from every earlier sweep:

  * **Pure self-play.** Both bots carry identical config, so the score rate is
    ~50% by construction. Win-probability features (blunder_rate, mean_wc_loss)
    saturate in decided positions, so arms with different score rates are not
    comparable on them -- equal-score self-play removes that confound.

  * **Complete games (`--simulate-full`).** The calibration table
    (cheat_detection/runs/elo_progression/report_timing.md) is built from
    complete human games, so the bot side must be complete too. The truncation
    matching used in the earlier bucket work was a device for that specific
    A/B; reusing it here would reintroduce exactly the mismatch it fixed.
    Adjudication exists to stop clock-race variance diluting an *Elo* signal,
    and we are measuring per-move features rather than results, so it buys
    nothing here and costs the whole scramble phase.

Phase A needs no engine changes: both knobs are already per-instance
constructor arguments and already exposed as simulation.run flags.

Run from repo root:
    venv/bin/python cheat_detection/runs/strength_dial/run_phase_a.py

Writes one PGN per arm into this directory plus an incrementally-updated
phase_a.log. Resumable: an arm whose PGN already exists is skipped, so the
script can be re-run after an interruption without redoing finished arms.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUT_DIR / "phase_a.log"

GAMES = 150
WORKERS = 6  # ~cores/3 (see simulation/run.py docstring); adjust per-machine
TC = "60+0"

# Shipped defaults, i.e. the point the Jacobian is taken around.
BASE_NOISE = 0.75    # common/constants.py HUMAN_EVAL_NOISE_SCALE
BASE_QUICK = 2.5     # common/constants.py QUICKNESS (bigger = slower)

# (arm, eval_noise_scale, quickness, seed). One knob off baseline at a time:
# this is a partial-derivative probe, not a factorial. Seeds are fixed per arm
# so the list can be trimmed or split across machines without shifting any
# other arm's seed or filename.
ARMS = [
    ("baseline",   BASE_NOISE, BASE_QUICK, 700000),
    ("noise_low",  0.55,       BASE_QUICK, 710000),
    ("noise_high", 0.95,       BASE_QUICK, 720000),
    ("quick_fast", BASE_NOISE, 2.0,        730000),
    ("quick_slow", BASE_NOISE, 3.0,        740000),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_arm(arm: str, noise: float, quick: float, seed: int) -> Path:
    pgn = OUT_DIR / f"phasea_{arm}_seed{seed}.pgn"
    if pgn.exists():
        log(f"{arm}: PGN already present, skipping ({pgn.name})")
        return pgn

    cmd = [
        sys.executable, "-m", "simulation.run",
        "--games", str(GAMES),
        "--tc", TC,
        "--seed", str(seed),
        "--workers", str(WORKERS),
        "--simulate-full",
        "--eval-noise-scale", str(noise),
        "--quickness", str(quick),
        "--out", str(pgn),
        "--plain",
    ]
    log(f"{arm}: noise={noise} quickness={quick} seed={seed}")
    log("  " + " ".join(cmd))
    start = time.time()
    # Stream progress to a per-arm file rather than capturing it: an arm runs
    # for the better part of an hour, and a run whose progress is invisible
    # until it finishes cannot be distinguished from one that has hung.
    progress = OUT_DIR / f"phasea_{arm}.progress"
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
    log(f"Phase A start: {len(ARMS)} arms x {GAMES} games, {TC}, "
        f"complete games, workers={WORKERS}")
    for arm, noise, quick, seed in ARMS:
        run_arm(arm, noise, quick, seed)
    log("Phase A complete. Analyse with cheat_detection against the "
        "complete-game 2300+ baseline; do NOT use a truncated baseline.")


if __name__ == "__main__":
    main()
