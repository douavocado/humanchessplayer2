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

**Corpus note.** The corpus is fetched by streaming a byte range of a Lichess
monthly dump through `fetch_corpus` (see the plan's Task 6). Download bandwidth
is the binding constraint, not the quota: measured ~64 KB/s, yielding ~68
qualifying games/min, so the corpus is sized by how long the fetch ran rather
than by `--band`'s count. That is fine for the three reads above -- they need
per-band precision on mean emt and t1, which tens of thousands of moves per
band already give. Set MIN_GAMES to the floor you are willing to read.

Run from repo root:
    venv/bin/python cheat_detection/runs/elo_progression/run_band_table_180.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

# fetch_corpus writes <out-stem>_<MIN>_<MAX>.pgn in --band mode, rendering an
# unbounded MAX as "plus" rather than the literal "+" passed on the CLI.
CORPUS = ROOT / "cheat_detection/corpora/blitz_3plus0_2300_plus.pgn"
OUT_MD = OUT_DIR / "report_timing_180.md"
TC = "180+0"
WORKERS = 6  # ~cores/2 at the default 2 engine threads
MIN_GAMES = 5000  # below this the per-band splits get too thin to read


def _count_games(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("[Event "):
                n += 1
    return n


def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(
            f"no corpus at {CORPUS}. Fetch it first -- see the plan's Task 6 "
            f"step 1. Bandwidth-bound, so allow hours, not minutes."
        )
    games = _count_games(CORPUS)
    if games < MIN_GAMES:
        raise SystemExit(
            f"{CORPUS.name} has only {games} games; want >= {MIN_GAMES} before "
            f"the per-band splits are worth reading. Let the fetch run longer, "
            f"or lower MIN_GAMES deliberately."
        )
    if OUT_MD.exists():
        print(f"{OUT_MD.name} already present; delete it to re-run.")
        return

    print(f"Building the {TC} band table from {games} games ...")
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
    print(f"Wrote {OUT_MD} in {mins:.1f} min ({games} games)")
    print("Now read it against the three pre-committed reads in this file's "
          "docstring -- including the one that kills the hypothesis.")


if __name__ == "__main__":
    main()
