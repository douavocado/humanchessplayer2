"""Diagnostic: bot vs human features conditioned on position character.

The main report collapses every move in a game into one aggregate feature
vector. This conditions on position type first: quiet (nothing at stake),
moderate, sharp+forced (one clearly-critical move -- a tactical shot), and
sharp+messy (several near-equally-good tries in a critical spot -- the
"which one?" positions). Buckets reuse the sharpness/ambiguity thresholds
already used elsewhere (engine.py's 0.25 "critical" / 0.10 "little at stake"
cutoffs; ambiguity==1 vs >=2 is the existing t1_rate_forced/ambiguous split).

Standalone and read-only: does not touch the baseline/report pipeline or its
JSON schema. Positions are re-read from the shared eval cache (FEN-keyed),
so re-running this over already-analysed PGNs is IO-bound, not Stockfish-bound.

Usage:
  venv/bin/python -m cheat_detection.bucket_diagnostic \
      --bot-pgn simulation/games/difficulty4_seed200000.pgn \
      --bot-player SimBotWhite SimBotBlack \
      --human-pgn cheat_detection/corpora/bullet_1plus0_2300_plus.pgn \
      --human-rating 2500 2800 \
      --out-md cheat_detection/runs/difficulty4/buckets_d4_vs2500.md
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Optional

from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer
from .features import MoveFeatures, extract_move_features
from .pgn_loader import iter_games, parse_tc_seconds
from .pipeline import _sides_of_interest

SHARP_HI = 0.25
SHARP_LO = 0.10

BUCKETS = ["quiet", "moderate", "sharp_forced", "sharp_messy"]
BUCKET_LABEL = {
    "quiet": "Quiet (sharpness < 0.10 -- little at stake)",
    "moderate": "Moderate (0.10 <= sharpness < 0.25)",
    "sharp_forced": "Sharp, one right answer (sharpness >= 0.25, ambiguity == 1 -- a tactical shot)",
    "sharp_messy": "Sharp and messy (sharpness >= 0.25, ambiguity >= 2 -- several critical tries)",
}
STATS = ["t1_rate", "t2_rate", "acpl", "blunder_rate", "mean_wc_loss"]
STAT_LABEL = {
    "t1_rate": "Top-1 match rate",
    "t2_rate": "Top-2 match rate",
    "acpl": "ACPL",
    "blunder_rate": "Blunder rate",
    "mean_wc_loss": "Mean win-chance loss",
}


def bucket_of(mf: MoveFeatures) -> str:
    if mf.sharpness < SHARP_LO:
        return "quiet"
    if mf.sharpness < SHARP_HI:
        return "moderate"
    return "sharp_forced" if mf.ambiguity == 1 else "sharp_messy"


def _unit_bucket_stats(mfeats: list[MoveFeatures], min_moves_bucket: int) -> dict[str, dict[str, Optional[float]]]:
    by_bucket: dict[str, list[MoveFeatures]] = defaultdict(list)
    for mf in mfeats:
        by_bucket[bucket_of(mf)].append(mf)
    out: dict[str, dict[str, Optional[float]]] = {}
    for b in BUCKETS:
        ms = by_bucket.get(b, [])
        n = len(ms)
        if n < min_moves_bucket:
            out[b] = {"n": n}
            continue
        out[b] = {
            "n": n,
            "t1_rate": sum(m.matched_top1 for m in ms) / n,
            "t2_rate": sum(m.matched_top2 for m in ms) / n,
            "acpl": sum(m.cp_loss for m in ms) / n,
            "blunder_rate": sum(m.is_blunder for m in ms) / n,
            "mean_wc_loss": sum(m.wc_loss for m in ms) / n,
        }
    return out


def collect_bucket_units(
    pgn_path: str,
    cfg: AnalysisConfig,
    player_filter: Optional[set[str]] = None,
    rating_band: Optional[tuple[int, int]] = None,
    max_games: Optional[int] = None,
    min_moves_total: int = 10,
    min_moves_bucket: int = 5,
    on_progress=None,
) -> list[dict[str, dict[str, Optional[float]]]]:
    """One dict-of-buckets per qualifying (player, game) unit."""
    units: list[dict[str, dict[str, Optional[float]]]] = []
    with EngineAnalyzer(cfg) as analyzer:
        for gi, game in enumerate(iter_games(pgn_path, max_games=max_games)):
            for color, name, elo in _sides_of_interest(game, player_filter, rating_band):
                mfeats: list[MoveFeatures] = []
                for rec in game.moves:
                    if rec.mover != color:
                        continue
                    pe = analyzer.analyse(rec.fen_before)
                    if not pe.candidates:
                        continue
                    played_cp = analyzer.eval_after_move(rec.fen_before, rec.move_uci)
                    mf = extract_move_features(rec, pe, played_cp, cfg)
                    if mf is not None:
                        mfeats.append(mf)
                if len(mfeats) < min_moves_total:
                    continue
                units.append(_unit_bucket_stats(mfeats, min_moves_bucket))
            analyzer.flush()
            if on_progress is not None:
                on_progress(gi + 1)
    return units


def _mean_std(xs: list[float]) -> tuple[Optional[float], Optional[float], int]:
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, 0
    mean = sum(xs) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return mean, std, n


def _welch_p(hmean, hstd, hn, bmean, bstd, bn) -> Optional[float]:
    if None in (hmean, hstd, bmean, bstd) or hn < 2 or bn < 2:
        return None
    vh, vb = hstd ** 2 / hn, bstd ** 2 / bn
    se2 = vh + vb
    if se2 <= 0:
        return None
    t = (bmean - hmean) / math.sqrt(se2)
    df = se2 ** 2 / (vh ** 2 / (hn - 1) + vb ** 2 / (bn - 1))
    try:
        from scipy import stats as _scipy_stats
        return float(2 * _scipy_stats.t.sf(abs(t), df))
    except ImportError:
        return math.erfc(abs(t) / math.sqrt(2))


def compare(human_units, bot_units) -> dict[str, dict[str, dict]]:
    """bucket -> stat -> {human_mean, human_std, human_n, bot_mean, bot_n, z, p}."""
    out: dict[str, dict[str, dict]] = {}
    for b in BUCKETS:
        out[b] = {}
        for s in STATS:
            hvals = [u[b].get(s) for u in human_units if s in u.get(b, {})]
            bvals = [u[b].get(s) for u in bot_units if s in u.get(b, {})]
            hmean, hstd, hn = _mean_std(hvals)
            bmean, bstd, bn = _mean_std(bvals)
            z = (bmean - hmean) / hstd if (hmean is not None and bmean is not None and hstd) else None
            p = _welch_p(hmean, hstd, hn, bmean, bstd, bn)
            out[b][s] = {
                "human_mean": hmean, "human_std": hstd, "human_n": hn,
                "bot_mean": bmean, "bot_std": bstd, "bot_n": bn,
                "z": z, "p": p,
            }
    return out


def render_md(cmp: dict, bot_label: str, human_label: str) -> str:
    lines = [f"# Bucketed diagnostic: {bot_label} vs {human_label}\n"]
    for b in BUCKETS:
        lines.append(f"## {BUCKET_LABEL[b]}\n")
        lines.append("| Stat | Human mean +/- std (n) | Bot mean (n) | z | p |")
        lines.append("|---|---|---|---|---|")
        for s in STATS:
            r = cmp[b][s]
            if r["human_mean"] is None or r["bot_mean"] is None:
                lines.append(f"| {STAT_LABEL[s]} | n/a | n/a | | |")
                continue
            hstr = f"{r['human_mean']:.3f} +/- {r['human_std']:.3f} ({r['human_n']})"
            bstr = f"{r['bot_mean']:.3f} ({r['bot_n']})"
            zstr = f"{r['z']:+.2f}" if r["z"] is not None else ""
            pstr = f"{r['p']:.4f}" if r["p"] is not None else ""
            flag = " ⚑" if r["z"] is not None and abs(r["z"]) >= 2 else ""
            lines.append(f"| {STAT_LABEL[s]} | {hstr} | {bstr} | {zstr}{flag} | {pstr} |")
        lines.append("")
    return "\n".join(lines)


def _progress(label):
    def cb(n):
        if n % 50 == 0:
            print(f"  {label}: {n} games...")
    return cb


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-pgn", required=True)
    ap.add_argument("--bot-player", nargs="+", required=True)
    ap.add_argument("--human-pgn", required=True)
    ap.add_argument("--human-rating", nargs=2, type=int, required=True)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--human-max-games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--tc", default="60+0",
                    help="time control of the corpora, e.g. 180+0 (default "
                         "60+0). Sets the initial clock that long-think and "
                         "time-pressure thresholds derive from.")
    ap.add_argument("--allow-tc-mismatch", action="store_true",
                    help="downgrade the TimeControl header check to a "
                         "warning/skip instead of raising.")
    return ap


def main():
    args = build_parser().parse_args()

    cfg = AnalysisConfig(workers=args.workers,
                         initial_time=parse_tc_seconds(args.tc),
                         strict_tc=not args.allow_tc_mismatch)

    print(f"Collecting bot buckets from {args.bot_pgn} ...")
    bot_units = collect_bucket_units(
        args.bot_pgn, cfg,
        player_filter={p.lower() for p in args.bot_player},
        max_games=args.max_games,
        on_progress=_progress("bot"),
    )
    print(f"  {len(bot_units)} bot units")

    band = (args.human_rating[0], args.human_rating[1])
    print(f"Collecting human buckets from {args.human_pgn} (rating {band}) ...")
    human_units = collect_bucket_units(
        args.human_pgn, cfg,
        rating_band=band,
        max_games=args.human_max_games,
        on_progress=_progress("human"),
    )
    print(f"  {len(human_units)} human units")

    cmp = compare(human_units, bot_units)
    md = render_md(cmp, bot_label=args.bot_pgn, human_label=f"{args.human_pgn} [{band[0]}-{band[1]}]")

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"Wrote {args.out_md}")
    if args.out_json:
        import json
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(cmp, fh, indent=2)
        print(f"Wrote {args.out_json}")
    if not args.out_md and not args.out_json:
        print(md)


if __name__ == "__main__":
    main()
