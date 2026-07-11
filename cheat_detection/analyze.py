"""CLI for the human-likeness diagnostic.

Quick path -- one command does fetch -> baseline -> report:

    venv/bin/python -m cheat_detection.analyze run \
        --user my_bot_account --rating 2300 2600 --perf bullet \
        --corpus cheat_detection/corpora/bullet_1plus0_2300_2600.pgn \
        --baseline cheat_detection/baselines/bullet_2300_2600.json \
        --out-md report.md

Individual steps (baseline / report / fetch-games) remain available below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .baseline import Baseline, build_baseline
from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer
from .fetch_lichess import fetch_user_games
from .orchestrate import DiagnosticSpec, run_diagnostic, save_result
from .pipeline import iter_units
from .report import build_report


def _progress_printer(label: str):
    return lambda n: print(f"  {label}: processed {n} games...", flush=True)


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--depth", type=int, help="Stockfish analysis depth (default 10)")
    p.add_argument("--multipv", type=int, help="candidate moves per position (default 5)")
    p.add_argument("--threads", type=int, help="Stockfish threads")
    p.add_argument("--hash", type=int, dest="hash_mb", help="Stockfish hash (MB)")
    p.add_argument("--min-moves", type=int, default=10,
                   help="minimum analysed moves for a unit to count (default 10)")
    p.add_argument("--test", choices=["effect-size", "welch"], dest="test_mode",
                   help="what flags a feature: effect-size |z| >= 2 vs human spread "
                        "(default), or a Welch two-sample t-test at --alpha")
    p.add_argument("--alpha", type=float, dest="flag_pvalue",
                   help="significance level for --test welch (default 0.05)")


def _config_from_args(args) -> AnalysisConfig:
    cfg = AnalysisConfig()
    for attr in ("depth", "multipv", "threads", "hash_mb", "flag_pvalue"):
        v = getattr(args, attr, None)
        if v is not None:
            setattr(cfg, attr, v)
    if getattr(args, "test_mode", None):
        cfg.test_mode = args.test_mode.replace("-", "_")
    return cfg


def cmd_baseline(args) -> int:
    cfg = _config_from_args(args)
    band = (args.rating[0], args.rating[1])
    print(f"Building baseline (rating {band[0]}-{band[1]}) from {args.pgn} ...")
    baseline = build_baseline(
        args.pgn, band, cfg,
        max_games=args.max_games, min_moves=args.min_moves,
        on_progress=_progress_printer("baseline"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    baseline.to_json(args.out)
    print(f"Wrote baseline with {baseline.n_units} human units -> {args.out}")
    return 0


def cmd_report(args) -> int:
    cfg = _config_from_args(args)
    baseline = Baseline.from_json(args.baseline)
    players = {p.lower() for p in args.player} if args.player else None

    bot_units = []
    with EngineAnalyzer(cfg) as analyzer:
        for unit in iter_units(
            args.pgn, analyzer, cfg,
            player_filter=players,
            max_games=args.max_games,
            min_moves=args.min_moves,
            on_progress=_progress_printer("report"),
        ):
            bot_units.append(unit)

    if not bot_units:
        print("No qualifying games found for the bot. Check --player and PGN.",
              file=sys.stderr)
        return 1

    md, report_dict = build_report(bot_units, baseline, cfg)

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"Wrote markdown report -> {args.out_md}")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2)
        print(f"Wrote JSON report -> {args.out_json}")
    if not args.out_md and not args.out_json:
        print(md)
    return 0


def cmd_fetch_games(args) -> int:
    fetch_user_games(args.user, args.out, max_games=args.max_games,
                     perf_type=args.perf, time_control=args.tc, on_log=print)
    return 0


def cmd_run(args) -> int:
    cfg = _config_from_args(args)
    spec = DiagnosticSpec(
        username=args.user,
        rating_band=(args.rating[0], args.rating[1]),
        perf=args.perf,
        time_control=args.tc,
        bot_pgn=args.pgn,
        baseline_path=args.baseline,
        corpus_pgn=args.corpus,
        bot_max_games=args.max_games,
        baseline_max_games=args.baseline_max_games,
        fetch_max_games=args.fetch_max_games,
    )
    workdir = args.workdir or os.path.join(os.path.dirname(__file__), "runs")
    try:
        result = run_diagnostic(
            cfg, spec, workdir,
            on_log=print,
            on_progress=_progress_printer("analysing"),
            on_phase=lambda ph: print(f"== phase: {ph} =="),
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.out_md or args.out_json:
        save_result(result,
                    args.out_md or os.path.join(workdir, "report.md"),
                    args.out_json or os.path.join(workdir, "report.json"))
        print(f"Saved report to {args.out_md or workdir}")
    else:
        print(result.markdown)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cheat_detection.analyze",
                                     description="Human-likeness diagnostic for the bot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prun = sub.add_parser("run", help="one-shot: fetch games -> ensure baseline -> report")
    prun.add_argument("--user", required=True, help="Lichess account of the bot")
    prun.add_argument("--rating", type=int, nargs=2, required=True, metavar=("MIN", "MAX"))
    prun.add_argument("--perf", default="bullet", help="bullet/blitz/rapid/classical")
    prun.add_argument("--tc", help="exact time control for the bot fetch, e.g. 60+0 "
                                   "(30+0 and 60+0 pacing differ ~2x; keep one clock)")
    prun.add_argument("--baseline", required=True, help="baseline JSON (loaded if present, else built)")
    prun.add_argument("--corpus", help="human corpus PGN to build the baseline if it doesn't exist")
    prun.add_argument("--pgn", help="bot PGN (skip fetch; use this file instead)")
    prun.add_argument("--max-games", type=int, default=300, help="cap bot games analysed")
    prun.add_argument("--baseline-max-games", type=int, help="cap baseline games when building")
    prun.add_argument("--fetch-max-games", type=int, default=300, help="games to download")
    prun.add_argument("--workdir", help="where fetched games / outputs go")
    prun.add_argument("--out-md", help="write markdown report here")
    prun.add_argument("--out-json", help="write JSON report here")
    _add_engine_args(prun)
    prun.set_defaults(func=cmd_run)

    pf = sub.add_parser("fetch-games", help="download a Lichess user's games as PGN")
    pf.add_argument("--user", required=True)
    pf.add_argument("--out", required=True)
    pf.add_argument("--max-games", type=int, default=300)
    pf.add_argument("--perf", default="bullet")
    pf.add_argument("--tc", help="exact time control to keep, e.g. 60+0")
    pf.set_defaults(func=cmd_fetch_games)

    pb = sub.add_parser("baseline", help="build a human baseline from a Lichess dump")
    pb.add_argument("--pgn", required=True, help="human PGN dump")
    pb.add_argument("--rating", type=int, nargs=2, required=True, metavar=("MIN", "MAX"),
                    help="rating band to include (both players' Elo filtered)")
    pb.add_argument("--out", required=True, help="output baseline JSON path")
    pb.add_argument("--max-games", type=int, help="cap games analysed (for speed)")
    _add_engine_args(pb)
    pb.set_defaults(func=cmd_baseline)

    pr = sub.add_parser("report", help="report bot vs. human baseline")
    pr.add_argument("--pgn", required=True, help="the bot's games")
    pr.add_argument("--player", nargs="+", help="bot account name(s) to profile")
    pr.add_argument("--baseline", required=True, help="baseline JSON from the 'baseline' step")
    pr.add_argument("--out-md", help="write markdown report here")
    pr.add_argument("--out-json", help="write JSON report here")
    pr.add_argument("--max-games", type=int, help="cap games analysed")
    _add_engine_args(pr)
    pr.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
