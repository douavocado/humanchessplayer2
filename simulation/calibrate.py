"""Phase-0 calibration: measure the real components of per-move wall time.

The simulator charges the SimClock with modelled durations instead of
measuring them mid-simulation (simulation load would skew measurements).
This script builds those models on the machine the bot normally runs on:

  compute    Replay positions from a real bot PGN through a headless
             Engine.update_info + make_move and record wall time per call.
             Positions are replayed in game order with the game's real
             clock values, so mood/analytics state evolves as it did live.

  detection  Run the auto_calibration readback test on saved screenshots and
             parse its per-call vision timings (FEN extraction, turn
             detection, clock OCR).

Both write JSON into simulation/calibration/ (gitignored: machine-specific).

Examples (from repo root):
  venv/bin/python -m simulation.calibrate compute \
      --pgn cheat_detection/runs/bot_JXu2019_60plus0.pgn --player JXu2019 \
      --max-positions 150
  venv/bin/python -m simulation.calibrate detection
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

import chess

CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration")

# How many historical fens / clock entries to feed the engine. The engine
# uses at most 5 fens and reads the opponent's last 4 clock times.
FEN_HISTORY = 5


def _percentiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    vs = sorted(vals)
    def pct(p):
        return vs[min(len(vs) - 1, int(p / 100 * len(vs)))]
    return {
        "n": len(vs),
        "mean": statistics.fmean(vs),
        "p10": pct(10), "p50": pct(50), "p90": pct(90), "p99": pct(99),
        "min": vs[0], "max": vs[-1],
    }


def _build_info_dic(rec, ply_idx: int, side: bool):
    """Reconstruct the engine info_dic for the position before rec.moves[ply_idx].

    Returns None when clock data is missing (engine formulas need it).
    """
    moves = rec.moves
    base = float(rec.base_secs)

    lo = max(0, ply_idx - (FEN_HISTORY - 1))
    fens = [m.fen_before for m in moves[lo:ply_idx + 1]]
    last_moves = [m.move_uci for m in moves[lo:ply_idx]]

    self_clocks: list[float] = [base]
    opp_clocks: list[float] = [base]
    for m in moves[:ply_idx]:
        if m.clock_after is None:
            return None
        (self_clocks if m.mover == side else opp_clocks).append(float(m.clock_after))

    return {
        "side": side,
        "fens": fens,
        "last_moves": last_moves,
        "self_clock_times": self_clocks[-FEN_HISTORY:],
        "opp_clock_times": opp_clocks[-FEN_HISTORY:],
        "self_initial_time": base,
        "opp_initial_time": base,
        "self_rating": rec.white_elo if side == chess.WHITE else rec.black_elo,
        "opp_rating": rec.black_elo if side == chess.WHITE else rec.white_elo,
    }


def cmd_compute(args) -> int:
    from cheat_detection.pgn_loader import iter_games
    from common.board_information import phase_of_game
    from common.constants import DIFFICULTY
    from engine import Engine

    print(f"Loading Engine (playing_level={DIFFICULTY})...", flush=True)
    engine = Engine(playing_level=DIFFICULTY)
    records = []
    skipped = 0
    player = args.player.lower()

    try:
        for game_idx, rec in enumerate(iter_games(args.pgn, max_games=args.max_games)):
            if rec.base_secs is None:
                continue
            for ply_idx, mv in enumerate(rec.moves):
                if len(records) >= args.max_positions:
                    break
                if mv.mover_name.lower() != player:
                    continue
                info = _build_info_dic(rec, ply_idx, mv.mover)
                if info is None or info["self_rating"] is None:
                    skipped += 1
                    continue
                own_time = info["self_clock_times"][-1]
                try:
                    t0 = time.perf_counter()
                    engine.update_info(info)
                    t1 = time.perf_counter()
                    out = engine.make_move(log=False, seed=args.seed + ply_idx)
                    t2 = time.perf_counter()
                except Exception as e:  # bad scrape/edge position: skip, keep measuring
                    skipped += 1
                    print(f"  skip g{game_idx} ply{ply_idx}: {type(e).__name__}: {e}",
                          flush=True)
                    continue
                records.append({
                    "game": game_idx,
                    "ply": ply_idx,
                    "phase": phase_of_game(chess.Board(mv.fen_before)),
                    "own_time": own_time,
                    "update_secs": t1 - t0,
                    "move_secs": t2 - t1,
                    "total_secs": t2 - t0,
                    "time_take": out.get("time_take"),
                    "pondered": out.get("ponder_dic") is not None,
                })
                if len(records) % 20 == 0:
                    print(f"  measured {len(records)} positions...", flush=True)
            if len(records) >= args.max_positions:
                break
    finally:
        engine.close_engines()

    totals = [r["total_secs"] for r in records]
    by_phase = {}
    for ph in ("opening", "midgame", "endgame"):
        by_phase[ph] = _percentiles([r["total_secs"] for r in records if r["phase"] == ph])
    payload = {
        "kind": "engine_compute",
        "source_pgn": args.pgn,
        "player": args.player,
        "seed": args.seed,
        "summary": {"total_secs": _percentiles(totals), "by_phase": by_phase},
        "records": records,
        "skipped": skipped,
    }
    os.makedirs(CALIB_DIR, exist_ok=True)
    out_path = args.out or os.path.join(CALIB_DIR, "compute_time.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    s = payload["summary"]["total_secs"]
    print(f"\nWrote {len(records)} measurements -> {out_path} ({skipped} skipped)")
    if s:
        print(f"update_info+make_move wall time: mean={s['mean']:.3f}s "
              f"p50={s['p50']:.3f}s p90={s['p90']:.3f}s p99={s['p99']:.3f}s")
    return 0


_TIMING_LINE = re.compile(
    r"(FEN Extraction|Turn Detection|Clock OCR):\s*avg=([\d.]+)ms, "
    r"min=([\d.]+)ms, max=([\d.]+)ms \((\d+) calls\)")


def cmd_detection(args) -> int:
    cmd = [sys.executable, "-m", "auto_calibration.calibration_readback_test",
           "--screenshots", args.screenshots, "--profile", args.profile]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        print("readback test failed", file=sys.stderr)
        return 1

    parsed = {}
    for m in _TIMING_LINE.finditer(proc.stdout):
        key = m.group(1).lower().replace(" ", "_")
        parsed[key] = {"avg_ms": float(m.group(2)), "min_ms": float(m.group(3)),
                       "max_ms": float(m.group(4)), "calls": int(m.group(5))}
    if not parsed:
        print(proc.stdout[-2000:])
        print("Could not parse timing lines from readback output", file=sys.stderr)
        return 1

    payload = {
        "kind": "detection_latency",
        "screenshots": args.screenshots,
        "profile": args.profile,
        "vision_ms": parsed,
        # Screen capture itself can't run offline; ~8ms is typical for
        # fastgrab at 1080p. Refined later from live [PERF] scan logs.
        "capture_ms_estimate": 8.0,
    }
    os.makedirs(CALIB_DIR, exist_ok=True)
    out_path = args.out or os.path.join(CALIB_DIR, "detection_latency.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote detection timings -> {out_path}")
    for k, v in parsed.items():
        print(f"  {k}: avg={v['avg_ms']:.1f}ms over {v['calls']} calls")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="simulation.calibrate",
                                description="Measure per-move latency components")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compute", help="measure headless engine compute time")
    pc.add_argument("--pgn", required=True, help="real bot games (with %%clk)")
    pc.add_argument("--player", required=True, help="bot account name in the PGN")
    pc.add_argument("--max-games", type=int, default=50)
    pc.add_argument("--max-positions", type=int, default=150)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--out", help="output JSON (default simulation/calibration/compute_time.json)")
    pc.set_defaults(func=cmd_compute)

    pd = sub.add_parser("detection", help="measure vision latency from saved screenshots")
    pd.add_argument("--screenshots",
                    default="auto_calibration/offline_screenshots/desktop")
    pd.add_argument("--profile", default="desktop")
    pd.add_argument("--out", help="output JSON (default simulation/calibration/detection_latency.json)")
    pd.set_defaults(func=cmd_detection)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
