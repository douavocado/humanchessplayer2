"""Investigates how human mistake patterns shift across rating bands, by
position character -- both eval-stakes (sharpness bucket, reusing
bucket_diagnostic.bucket_of) and structural complexity (Lucas analytics:
complexity, effective mobility, narrowness, activity -- the same metrics
engine.py computes live and decide_breadth branches on) -- plus game phase.
Goal: find *where* improvement concentrates as players get stronger,
rather than assuming it's uniform.

Motivation: engine.py's strength levers (DIFFICULTY, MIDGAME_PREMOVE_VETO_P,
...) apply uniformly -- turning one up raises "how careful" the bot is
everywhere at once, which risks over-flattening its mistake distribution
into something too consistent to look human (this is exactly what
DIFFICULTY=4 did: lower ACPL and blunder rate overall, but flagged by
cheat_detection's variance test as too-consistent game-to-game). If real
human improvement between rating bands is concentrated in specific
position types rather than uniform, that shape is a natural template for
shaping the bot's own strength progression instead of a blunt uniform
lever.

Lucas analytics are computed with a *separate*, wide-multipv (all legal
moves, matching engine.py's own calculate_analytics -- not the 5-candidate
eval cache used for sharpness/blunder features) Stockfish pass per
position, memoized within the run; not persistently cached like the main
eval cache, so re-running this script re-pays that cost.

Bands are fixed to the well-covered range of the existing corpus
(bullet_1plus0_2300_plus.pgn: 30k games, 2100-3100+ has thousands of games
per ~100-200pt band). Below ~2000 the corpus is too thin and, via Lichess
matchmaking, mostly reflects weaker players outmatched by 2300+ opponents
rather than how they play against peers -- a genuine 1200-2800 picture
would need a dedicated low-Elo fetch (out of scope here).

Usage:
  venv/bin/python -m cheat_detection.elo_progression \\
      --pgn cheat_detection/corpora/bullet_1plus0_2300_plus.pgn \\
      --workers 6 --out-md cheat_detection/runs/elo_progression/report.md
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as _queue
from collections import defaultdict
from typing import Optional

import chess
import chess.engine

from common.board_information import get_lucas_analytics
from common.constants import PATH_TO_STOCKFISH
from common.search_constants import EFF_MOB_TACTICAL_CUTOFF, STOCKFISH_HASH_MB, STOCKFISH_THREADS

from .bucket_diagnostic import BUCKETS, BUCKET_LABEL, bucket_of
from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer, open_cache_db
from .features import MoveFeatures, extract_move_features
from .pgn_loader import iter_games
from .pipeline import _sides_of_interest

BANDS = [
    (2100, 2300), (2300, 2400), (2400, 2500), (2500, 2600),
    (2600, 2700), (2700, 2800), (2800, 9999),
]
PHASES = ["opening", "middlegame", "endgame"]
LUCAS_KEYS = ["complexity", "win_prob", "eff_mob", "narrowness", "activity"]
EFF_MOB_BUCKETS = ["forced_tactical", "open"]

Unit = tuple[MoveFeatures, dict]


def band_label(band: tuple[int, int]) -> str:
    return f"{band[0]}+" if band[1] >= 9999 else f"{band[0]}-{band[1] - 1}"


def eff_mob_bucket(lucas: dict) -> str:
    return "forced_tactical" if lucas["eff_mob"] < EFF_MOB_TACTICAL_CUTOFF else "open"


def _lucas_for(fen: str, lucas_engine: chess.engine.SimpleEngine, memo: dict[str, dict]) -> dict:
    cached = memo.get(fen)
    if cached is not None:
        return cached
    board = chess.Board(fen)
    no_lines = board.legal_moves.count()
    analysis = lucas_engine.analyse(board, chess.engine.Limit(time=0.02), multipv=max(no_lines, 1))
    if isinstance(analysis, dict):
        analysis = [analysis]
    xcompl, xmlr, xemo, xnar, xact = get_lucas_analytics(board, analysis=analysis)
    result = {"complexity": xcompl, "win_prob": xmlr, "eff_mob": xemo, "narrowness": xnar, "activity": xact}
    memo[fen] = result
    return result


def _collect_sequential(pgn_path: str, cfg: AnalysisConfig, max_games: Optional[int],
                        gi_filter=None, progress_q=None) -> dict[tuple[int, int], list[Unit]]:
    """gi_filter(gi) -> bool restricts which games this call processes (used
    to stripe work across worker processes); None = every game."""
    by_band: dict[tuple[int, int], list[Unit]] = defaultdict(list)
    lucas_memo: dict[str, dict] = {}
    lucas_engine = chess.engine.SimpleEngine.popen_uci(PATH_TO_STOCKFISH)
    lucas_engine.configure({"Threads": STOCKFISH_THREADS, "Hash": STOCKFISH_HASH_MB})

    with EngineAnalyzer(cfg) as analyzer:
        for gi, game in enumerate(iter_games(pgn_path, max_games=max_games)):
            if gi_filter is not None and not gi_filter(gi):
                continue
            for band in BANDS:
                for color, name, elo in _sides_of_interest(game, None, band):
                    for rec in game.moves:
                        if rec.mover != color:
                            continue
                        pe = analyzer.analyse(rec.fen_before)
                        if not pe.candidates:
                            continue
                        played_cp = analyzer.eval_after_move(rec.fen_before, rec.move_uci)
                        mf = extract_move_features(rec, pe, played_cp, cfg)
                        if mf is None:
                            continue
                        lucas = _lucas_for(rec.fen_before, lucas_engine, lucas_memo)
                        by_band[band].append((mf, lucas))
            analyzer.flush()
            if progress_q is not None:
                progress_q.put(1)
            elif (gi + 1) % 1000 == 0:
                counts = {band_label(b): len(by_band[b]) for b in BANDS}
                print(f"  {gi + 1} games scanned ({len(lucas_memo)} unique lucas positions), "
                      f"moves collected so far: {counts}")

    lucas_engine.quit()
    return by_band


def _worker(worker_id: int, nworkers: int, pgn_path: str, cfg: AnalysisConfig,
            max_games: Optional[int], progress_q) -> dict[tuple[int, int], list[Unit]]:
    return _collect_sequential(pgn_path, cfg, max_games,
                               gi_filter=lambda gi: gi % nworkers == worker_id,
                               progress_q=progress_q)


def collect(pgn_path: str, cfg: AnalysisConfig, max_games: Optional[int] = None) -> dict[tuple[int, int], list[Unit]]:
    """Parallelised across cfg.workers processes (games striped by index,
    mirroring cheat_detection/parallel.py); workers<=1 is the sequential path."""
    workers = max(1, cfg.workers)
    if workers == 1:
        return _collect_sequential(pgn_path, cfg, max_games)

    open_cache_db(cfg).close()  # one-time legacy-JSON migration before forking out

    ctx = mp.get_context("spawn")
    by_band: dict[tuple[int, int], list[Unit]] = defaultdict(list)
    with ctx.Manager() as mgr:
        progress_q = mgr.Queue()
        with ctx.Pool(workers) as pool:
            jobs = [
                pool.apply_async(_worker, (wid, workers, pgn_path, cfg, max_games, progress_q))
                for wid in range(workers)
            ]
            pool.close()
            done = 0
            pending = list(jobs)
            while pending:
                try:
                    progress_q.get(timeout=0.5)
                    done += 1
                    if done % 500 == 0:
                        print(f"  {done} games scanned...")
                except _queue.Empty:
                    pass
                pending = [j for j in pending if not j.ready()]
            while True:
                try:
                    progress_q.get_nowait()
                    done += 1
                except _queue.Empty:
                    break
            for j in jobs:
                partial = j.get()  # re-raises worker exceptions
                for band, units in partial.items():
                    by_band[band].extend(units)
    return by_band


def _stats(mfeats: list[MoveFeatures]) -> dict[str, float]:
    n = len(mfeats)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "blunder_rate": sum(m.is_blunder for m in mfeats) / n,
        "mean_wc_loss": sum(m.wc_loss for m in mfeats) / n,
        "acpl": sum(m.cp_loss for m in mfeats) / n,
        "t1_rate": sum(m.matched_top1 for m in mfeats) / n,
    }


def _row(band, s) -> str:
    if s["n"] < 30:
        return f"| {band_label(band)} | {s['n']} | n/a | n/a | n/a |"
    return (f"| {band_label(band)} | {s['n']} | {s['blunder_rate']:.4f} | "
            f"{s['mean_wc_loss']:.4f} | {s['acpl']:.1f} |")


def render(by_band: dict[tuple[int, int], list[Unit]], min_n: int = 30) -> str:
    lines = ["# Human mistake-profile progression across rating bands\n"]
    lines.append("Pooled at the move level (not per-player-unit) -- descriptive trend, "
                  "not a significance test; sample sizes are large enough per band that "
                  "this is a reasonable simplification.\n")

    lines.append("## Overall (all positions)\n")
    lines.append("| Band | n moves | Blunder rate | Mean win-chance loss | ACPL | Top-1 match |")
    lines.append("|---|---|---|---|---|---|")
    for band in BANDS:
        s = _stats([mf for mf, _ in by_band[band]])
        if s["n"] < min_n:
            lines.append(f"| {band_label(band)} | {s['n']} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(f"| {band_label(band)} | {s['n']} | {s['blunder_rate']:.4f} | "
                      f"{s['mean_wc_loss']:.4f} | {s['acpl']:.1f} | {s['t1_rate']:.4f} |")
    lines.append("")

    lines.append("## By position-sharpness bucket (eval-stakes)\n")
    for b in BUCKETS:
        lines.append(f"### {BUCKET_LABEL[b]}\n")
        lines.append("| Band | n moves | Blunder rate | Mean win-chance loss | ACPL |")
        lines.append("|---|---|---|---|---|")
        for band in BANDS:
            s = _stats([mf for mf, _ in by_band[band] if bucket_of(mf) == b])
            lines.append(_row(band, s))
        lines.append("")

    lines.append(f"## By effective-mobility bucket (Lucas eff_mob, structural complexity -- "
                  f"cutoff {EFF_MOB_TACTICAL_CUTOFF}, same threshold decide_breadth uses)\n")
    for b in EFF_MOB_BUCKETS:
        label = "Forced/tactical (eff_mob < cutoff)" if b == "forced_tactical" else "Open (eff_mob >= cutoff)"
        lines.append(f"### {label}\n")
        lines.append("| Band | n moves | Blunder rate | Mean win-chance loss | ACPL |")
        lines.append("|---|---|---|---|---|")
        for band in BANDS:
            s = _stats([mf for mf, lu in by_band[band] if eff_mob_bucket(lu) == b])
            lines.append(_row(band, s))
        lines.append("")

    lines.append("## By game phase\n")
    for phase in PHASES:
        lines.append(f"### {phase}\n")
        lines.append("| Band | n moves | Blunder rate | Mean win-chance loss | ACPL |")
        lines.append("|---|---|---|---|---|")
        for band in BANDS:
            s = _stats([mf for mf, _ in by_band[band] if mf.phase == phase])
            lines.append(_row(band, s))
        lines.append("")

    lines.append("## Position-character mix per band (does the bucket distribution itself shift?)\n")
    lines.append("| Band | " + " | ".join(BUCKETS) + " | forced_tactical | open |")
    lines.append("|---|" + "---|" * (len(BUCKETS) + 2))
    for band in BANDS:
        pairs = by_band[band]
        n = len(pairs)
        if n < min_n:
            lines.append(f"| {band_label(band)} | " + " | ".join(["n/a"] * (len(BUCKETS) + 2)) + " |")
            continue
        sharp_counts = defaultdict(int)
        mob_counts = defaultdict(int)
        for mf, lu in pairs:
            sharp_counts[bucket_of(mf)] += 1
            mob_counts[eff_mob_bucket(lu)] += 1
        row = " | ".join(f"{100*sharp_counts[b]/n:.1f}%" for b in BUCKETS)
        row += " | " + " | ".join(f"{100*mob_counts[b]/n:.1f}%" for b in EFF_MOB_BUCKETS)
        lines.append(f"| {band_label(band)} | {row} |")
    lines.append("")

    lines.append("## Mean Lucas analytics per band (does the *context* itself shift with rating?)\n")
    lines.append("| Band | n moves | complexity | win_prob | eff_mob | narrowness | activity |")
    lines.append("|---|---|---|---|---|---|---|")
    for band in BANDS:
        pairs = by_band[band]
        n = len(pairs)
        if n < min_n:
            lines.append(f"| {band_label(band)} | {n} | " + " | ".join(["n/a"] * len(LUCAS_KEYS)) + " |")
            continue
        means = {k: sum(lu[k] for _, lu in pairs) / n for k in LUCAS_KEYS}
        lines.append(f"| {band_label(band)} | {n} | " +
                      " | ".join(f"{means[k]:.2f}" for k in LUCAS_KEYS) + " |")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    cfg = AnalysisConfig(workers=args.workers)
    print(f"Collecting mistake-profile data from {args.pgn} across bands {[band_label(b) for b in BANDS]} ...")
    by_band = collect(args.pgn, cfg, max_games=args.max_games)
    for band in BANDS:
        print(f"  {band_label(band)}: {len(by_band[band])} moves")

    md = render(by_band)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"Wrote {args.out_md}")
    else:
        print(md)


if __name__ == "__main__":
    main()
