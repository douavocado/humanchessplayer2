"""Part 1 of the 180+0 pacing plan: refit `quickness` -- outcome: falsified.

Result (100 games/arm, 180+0, pure self-play):

| quickness | mean emt | time forfeits |
|---|---|---|
| 2.0       | 2.276s   | 16% |
| 2.5 (shipped) | 2.574s | 20% |
| 3.2       | 3.208s   | 47% |
| 4.0       | 3.348s   | 41% |
| humans (40k games) | -- | 24.3% |

`quickness` only reaches the human mean emt (per-move convention, ~3.14s --
see `common/tc_profiles.py`'s docstring) at roughly double the human
time-forfeit rate. Per-phase means show why: endgame think time barely moves
across the sweep (1.30/1.47/1.33/1.45s) while midgame scales hard
(2.98s -> 4.27s), so raising the level overspends the midgame and starves the
endgame clock. Task 4 (inverting this fit onto per-band targets) and Task 5
(the opening envelope, which would have been fitted against Task 4's output)
were both cancelled as a result -- the plan halted here by explicit decision.
See `docs/superpowers/specs/2026-07-30-longer-tc-pacing-calibration-design.md`
("Outcome: Part 1 falsified") for the full record.

The pre-committed read this sweep was designed to check *also* failed: the
spec's model requires `open_mid` (opening/midgame emt ratio) to stay roughly
flat across arms, since `quickness` is supposed to be a global scale. Measured
`open_mid` was 0.364 / 0.333 / 0.299 / 0.309 -- not flat, a second signal that
the global-scale premise doesn't hold cleanly, independent of the forfeit-rate
finding above.

Conventions match the spec's measurement arm or the numbers are not
comparable: pure self-play, complete games (`--simulate-full`), 180+0, 100
games per arm. Phase means use cheat_detection's `_phase`, NOT the engine's
`phase_of_game` -- they are unrelated rules and mixing them invents a
phase-mix crisis that does not exist.

Reproduction (all arm PGNs already on disk; do not re-run the ~4h sweep
unless the underlying code changed):
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
import chess.pgn

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RESULT = OUT_DIR / "quickness_sweep_180.json"

GAMES = 100
TC = "180+0"
WORKERS = 6
ARMS = [(2.0, 920000), (2.5, 930000), (3.2, 940000), (4.0, 950000)]

# Per-move convention (mean over all moves, not over per-game/per-player
# means): measured directly from `cheat_detection/corpora/blitz_3plus0_2300_plus.pgn`
# (the same 40k-game / 3.14M-move corpus behind the 180+0 band table) via this
# file's own `profile()` helper, so it's apples-to-apples with the arms below.
# 3.283 (the figure this replaced) was a *per-unit* mean -- averaged over
# player-games first -- from a baseline JSON, and is not comparable to the
# arms' per-move mean_emt.
HUMAN_MEAN_EMT = 3.158
HUMAN_OPEN_MID = 0.377


def profile(pgn: Path) -> dict:
    """Realised mean emt, opening/midgame ratio, and forfeit rate.

    Phases use cheat_detection's `_phase`, not the engine's `phase_of_game`
    (see module docstring). `forfeit_rate` is the fraction of games in `pgn`
    whose `Termination` header is "Time forfeit" (see `simulation/pgn_writer.py`
    `TERMINATION_HEADER`) -- the metric that actually killed this plan.
    """
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

    n_games = 0
    n_forfeits = 0
    with open(pgn) as fh:
        while True:
            headers = chess.pgn.read_headers(fh)
            if headers is None:
                break
            n_games += 1
            if headers.get("Termination", "") == "Time forfeit":
                n_forfeits += 1
    forfeit_rate = n_forfeits / n_games if n_games else float("nan")

    return {
        "mean_emt": mean(allm),
        "opening": mean(op), "middlegame": mean(mid),
        "endgame": mean(byphase.get("endgame", [])),
        "open_mid": mean(op) / mean(mid) if op and mid else float("nan"),
        "n_moves": len(allm),
        "forfeit_rate": forfeit_rate,
        "n_games": n_games,
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
