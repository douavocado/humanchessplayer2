"""Sweep OPENING_BREADTH_STRENGTH_BONUS / MIDGAME_BREADTH_STRENGTH_BONUS via
head-to-head adjudicated-Elo A/B matches: BaseBot (both bonuses 0, i.e.
current shipped default) vs a bot with one bonus raised, the other held at
0. One lever varied at a time -- this is a screen, not a full factorial.

Run from repo root:
    venv/bin/python cheat_detection/runs/breadth_sweep/run_sweep.py

Writes one PGN + adjudication JSON per matchup into this directory, plus an
incrementally-updated report.md (so progress is visible mid-sweep without
waiting for the whole thing to finish).

Resumable / splittable across machines: each matchup has a fixed seed keyed
by (lever, value) rather than by list position, and a matchup whose
`sweep_<lever><value>_seed<seed>.adjudicate.txt` already exists is skipped
(its result is read back from that file rather than re-run) -- so the same
script can be run on a second machine against a copy of this directory
without redoing already-finished matchups, and MATCHUPS can be freely
trimmed to just the remaining entries.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUT_DIR / "sweep.log"

GAMES = 150
WORKERS = 6  # ~cores/3 (see simulation/run.py docstring); adjust per-machine
TC = "60+0"

# (lever, value) -- midgame denser per the stated heuristic that midgame
# bonuses are expected to matter most; opening coarser as a secondary check.
# Seeds are fixed per (lever, value) so this list can be trimmed (e.g. to
# split remaining work across machines) without shifting any other entry's
# seed/filenames.
MATCHUPS = [
    ("midgame", 1, 500000),
    ("midgame", 2, 510000),
    ("midgame", 3, 520000),
    ("midgame", 5, 530000),
    ("opening", 2, 540000),
    ("opening", 5, 550000),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_matchup(lever: str, value: int, seed: int):
    bot_name = f"{lever.capitalize()}{value}Bot"
    pgn = OUT_DIR / f"sweep_{lever}{value}_seed{seed}.pgn"
    cmd = [
        sys.executable, "-m", "simulation.run",
        "--games", str(GAMES), "--tc", TC, "--seed", str(seed),
        "--workers", str(WORKERS), "--sides", "alternate",
        "--rating", "2450",
        "--a-name", "BaseBot", "--b-name", bot_name,
        "--out", str(pgn), "--plain",
    ]
    if lever == "midgame":
        cmd += ["--b-midgame-breadth-bonus", str(value)]
    else:
        cmd += ["--b-opening-breadth-bonus", str(value)]

    log(f"START matchup lever={lever} value={value} bot={bot_name} -> {pgn.name}")
    t0 = time.time()
    with open(OUT_DIR / f"sweep_{lever}{value}_seed{seed}.run.log", "w", encoding="utf-8") as run_log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=run_log, stderr=run_log)
    wall = time.time() - t0
    log(f"DONE matchup lever={lever} value={value} in {wall:.0f}s")
    return bot_name, pgn, wall


def adjudicate(pgn: Path, out_json: Path) -> str:
    cmd = [sys.executable, "-m", "simulation.adjudicate_result",
           "--pgn", str(pgn), "--out-json", str(out_json)]
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout


def parse_elo(stdout: str, bot_name: str):
    m = re.search(r"Adjudicated Elo delta \((.+?) - (.+?)\): ([+-]?[\d.]+)", stdout)
    if not m:
        return None
    p1, p2, val = m.group(1), m.group(2), float(m.group(3))
    if p1 == bot_name:
        return val
    if p2 == bot_name:
        return -val
    return None


def parse_raw_elo(stdout: str, bot_name: str):
    m = re.search(r"Raw Elo delta \((.+?) - (.+?)\): ([+-]?[\d.]+)", stdout)
    if not m:
        return None
    p1, p2, val = m.group(1), m.group(2), float(m.group(3))
    if p1 == bot_name:
        return val
    if p2 == bot_name:
        return -val
    return None


def write_report(results: list[dict]) -> None:
    lines = [
        "# Opening/midgame breadth-bonus sweep",
        "",
        "Head-to-head, BaseBot (opening/midgame bonus both 0 -- current "
        "shipped default) vs a bot with one bonus raised (other held at 0). "
        f"{GAMES} games/matchup, `--sides alternate`, TC {TC}, difficulty/"
        "rating otherwise identical (both 2450). Games are already "
        "clock-threshold-adjudicated live by the simulator default "
        "(`simulate_full=False`); the `simulation.adjudicate_result` "
        "post-hoc pass independently re-evaluates at the same cutoff as a "
        "check -- Raw and Adjudicated columns should closely agree.",
        "",
        "Positive Elo delta = the bonus made the bot stronger.",
        "",
        "| Lever | Bonus | Raw Elo delta | Adjudicated Elo delta | Wall (s) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        raw = f"{r['raw_elo']:+.1f}" if r['raw_elo'] is not None else "?"
        adj = f"{r['adj_elo']:+.1f}" if r['adj_elo'] is not None else "?"
        lines.append(f"| {r['lever']} | {r['value']} | {raw} | {adj} | {r['wall_secs']:.0f} |")
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Sweep starting: {len(MATCHUPS)} matchups, {GAMES} games each, {WORKERS} workers")
    results = []
    for lever, value, seed in MATCHUPS:
        bot_name = f"{lever.capitalize()}{value}Bot"
        pgn = OUT_DIR / f"sweep_{lever}{value}_seed{seed}.pgn"
        adjudicate_txt = OUT_DIR / f"{pgn.stem}.adjudicate.txt"
        if adjudicate_txt.exists():
            stdout = adjudicate_txt.read_text(encoding="utf-8")
            raw_elo, adj_elo = parse_raw_elo(stdout, bot_name), parse_elo(stdout, bot_name)
            log(f"SKIP matchup lever={lever} value={value} (already adjudicated: "
                f"raw_elo={raw_elo} adj_elo={adj_elo})")
            results.append({
                "lever": lever, "value": value, "bot_name": bot_name,
                "raw_elo": raw_elo, "adj_elo": adj_elo, "wall_secs": 0,
                "pgn": str(pgn),
            })
            write_report(results)
            continue

        bot_name, pgn, wall = run_matchup(lever, value, seed)
        out_json = pgn.with_suffix("").with_suffix(".adj.json")
        stdout = adjudicate(pgn, out_json)
        adjudicate_txt.write_text(stdout, encoding="utf-8")
        raw_elo = parse_raw_elo(stdout, bot_name)
        adj_elo = parse_elo(stdout, bot_name)
        log(f"RESULT lever={lever} value={value} raw_elo={raw_elo} adj_elo={adj_elo}")
        results.append({
            "lever": lever, "value": value, "bot_name": bot_name,
            "raw_elo": raw_elo, "adj_elo": adj_elo, "wall_secs": wall,
            "pgn": str(pgn),
        })
        write_report(results)
    log("Sweep complete.")


if __name__ == "__main__":
    main()
