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
            with open(OUT_DIR / f"sweepq_{q}.progress", "w") as progress_f:
                proc = subprocess.run(
                    [sys.executable, "-m", "simulation.run",
                     "--games", str(GAMES), "--tc", TC, "--seed", str(seed),
                     "--workers", str(WORKERS), "--simulate-full", "--plain",
                     "--a-quickness", str(q), "--b-quickness", str(q),
                     "--out", str(pgn)],
                    cwd=ROOT, check=False,
                    stdout=progress_f,
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
