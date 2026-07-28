"""Phase C of the strength-dial work: the re-evaluation-ordering arms.

These are the arms behind the one behavioural default change of this line of
work -- REEVAL_ORDER "random" -> "human" (commit 12b322d) -- so they need to be
reproducible on demand rather than existing only as PGNs. They were originally
run as inline shell loops; this file re-declares them exactly, and the
resumable skip means re-running it will not redo work whose PGN is already
present.

What the arms answer:

  * `human` / `eval` (the arms formerly called "Phase C"): does the ordering of
    the re-evaluation draw move t1 at all, and along which axis? When there is
    not time to re-evaluate every root move, the ones that miss out stay at
    depth_considered 0 and take a ~60cp penalty, which in a quiet position
    dwarfs the real eval spread -- so the draw is effectively a
    disqualification. "eval" is the control: it should raise t1 by making the
    bot play *better*, which is the wrong direction for human-likeness.

  * `human_rep` / `human_br2` (formerly "Phase D"): does the `human` result
    replicate at a fresh seed, and does breadth stack on top of it? The
    replicate is the important one, and it is what retracted the original
    reading of these arms -- see below.

**Read the t1 column, not the blunder column.** The two identical-config
`human` arms (seeds 810000 and 830000) came out at blunder 0.0483 and 0.0396,
a gap wider than either is from the baseline. The binomial standard error
(~0.0017/arm) understates the real between-seed spread (~0.0062) by about
3.6x, so no blunder-rate comparison at 150 games carries information;
settling that feature needs roughly 600 games/arm. t1 behaves: the same two
arms agree to 0.8 se. An earlier reading of the seed-810000 arm claimed t1 and
blunder rate both rose -- that was a single-seed artefact and is retracted.

Conventions match Phases A and B exactly, or the arms are not comparable with
them: pure self-play, complete games (`--simulate-full`), 60+0 -- that is 60
*seconds*, one minute of bullet -- and 150 games per arm.

Run from repo root:
    venv/bin/python cheat_detection/runs/strength_dial/run_phase_c.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUT_DIR / "phase_c_driver.log"

GAMES = 150
WORKERS = 6  # ~cores/3 (see simulation/run.py docstring); adjust per-machine
TC = "60+0"

# (arm, prefix, flags, seed). Prefixes preserve the filenames the original
# inline runs produced, so this driver skips those arms rather than re-running
# them. Seeds are fixed per arm.
ARMS = [
    ("human",     "phasec", ["--reeval-order", "human"], 810000),
    ("eval",      "phasec", ["--reeval-order", "eval"], 820000),
    ("human_rep", "phased", ["--reeval-order", "human"], 830000),
    ("human_br2", "phased", ["--reeval-order", "human",
                             "--midgame-breadth-bonus", "2"], 840000),
]

BASELINE = "cheat_detection/baselines/bullet_1plus0_2300_plus.json"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _run(cmd: list[str], label: str, progress: Path) -> None:
    start = time.time()
    # Stream progress rather than capturing it: an arm runs for most of an
    # hour, and a run whose progress is invisible cannot be told from a hang.
    with open(progress, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                              check=False)  # returncode handled below
    mins = (time.time() - start) / 60
    if proc.returncode != 0:
        log(f"{label}: FAILED rc={proc.returncode} after {mins:.1f} min "
            f"(see {progress.name})")
        raise SystemExit(f"{label} failed")
    log(f"{label}: done in {mins:.1f} min")


def run_arm(arm: str, prefix: str, flags: list[str], seed: int) -> None:
    pgn = OUT_DIR / f"{prefix}_{arm}_seed{seed}.pgn"
    report = OUT_DIR / f"rep{prefix[-1]}_{arm}.report.json"
    if pgn.exists():
        log(f"{arm}: PGN already present, skipping sim ({pgn.name})")
    else:
        log(f"{arm}: {' '.join(flags)} seed={seed}")
        _run([sys.executable, "-m", "simulation.run",
              "--games", str(GAMES), "--tc", TC, "--seed", str(seed),
              "--workers", str(WORKERS), "--simulate-full", *flags,
              "--out", str(pgn), "--plain"],
             f"{arm} sim", OUT_DIR / f"{prefix}_{arm}.progress")

    if report.exists():
        log(f"{arm}: report already present, skipping analysis ({report.name})")
        return
    _run([sys.executable, "-m", "cheat_detection.analyze", "report",
          "--pgn", str(pgn), "--player", "SimBotWhite", "SimBotBlack",
          "--baseline", BASELINE, "--workers", "3", "--threads", "2",
          "--out-md", str(OUT_DIR / f"rep{prefix[-1]}_{arm}.md"),
          "--out-json", str(report)],
         f"{arm} analysis", OUT_DIR / f"{prefix}_{arm}.analysis")


def main() -> None:
    log(f"Phase C start: {len(ARMS)} arms x {GAMES} games, {TC}, complete "
        f"games, workers={WORKERS}")
    for arm, prefix, flags, seed in ARMS:
        run_arm(arm, prefix, flags, seed)
    log("Phase C complete. Judge on t1_rate; blunder_rate is not measurable "
        "at this sample size (see module docstring).")


if __name__ == "__main__":
    main()
