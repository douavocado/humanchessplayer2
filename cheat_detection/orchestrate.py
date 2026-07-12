"""One-call orchestration: fetch games -> ensure baseline -> report.

Both the CLI (`analyze.py run`) and the GUI drive this module, so the whole
pipeline lives in one place. Everything takes optional ``on_log`` / ``on_progress``
callbacks so a UI can show live status without the analysis code knowing about
any particular front-end.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

from .baseline import Baseline, build_baseline
from .config import AnalysisConfig
from .fetch_lichess import fetch_user_games
from .parallel import collect_units
from .pipeline import Unit
from .report import FeatureComparison, build_report, compare

LogFn = Optional[Callable[[str], None]]
ProgFn = Optional[Callable[[int], None]]


def _noop(*_a, **_k):
    return None


@dataclass
class DiagnosticResult:
    markdown: str
    report: dict
    comparisons: list[FeatureComparison]
    bot_units: list[Unit]
    baseline: Baseline


def generate_report(
    cfg: AnalysisConfig,
    bot_pgn: str,
    player: str,
    baseline: Baseline,
    max_games: Optional[int] = None,
    min_moves: int = 10,
    on_log: LogFn = None,
    on_progress: ProgFn = None,
    opponent_band: Optional[tuple[int, int]] = None,
    diff_range: Optional[tuple[Optional[int], Optional[int]]] = None,
) -> DiagnosticResult:
    """Analyse the bot's games and compare to a loaded baseline."""
    log = on_log or _noop
    players = {player.lower()} if player else None

    log(f"Analysing bot games from {os.path.basename(bot_pgn)} ...")
    bot_units: list[Unit] = collect_units(
        bot_pgn, cfg,
        player_filter=players,
        max_games=max_games,
        min_moves=min_moves,
        on_progress=on_progress,
        opponent_band=opponent_band,
        diff_range=diff_range,
    )

    if not bot_units:
        raise ValueError(
            f"No qualifying games for player '{player}' in {bot_pgn}."
        )

    md, report_dict = build_report(bot_units, baseline, cfg)
    comps = compare(bot_units, baseline, cfg)
    log(f"Report ready: {len(bot_units)} bot units vs {baseline.n_units} human units.")
    return DiagnosticResult(md, report_dict, comps, bot_units, baseline)


def ensure_baseline(
    cfg: AnalysisConfig,
    baseline_path: str,
    corpus_pgn: Optional[str],
    rating_band: tuple[int, int],
    max_games: Optional[int] = None,
    min_moves: int = 10,
    on_log: LogFn = None,
    on_progress: ProgFn = None,
    opponent_band: Optional[tuple[int, int]] = None,
    diff_range: Optional[tuple[Optional[int], Optional[int]]] = None,
) -> Baseline:
    """Load an existing baseline JSON, or build one from a corpus PGN."""
    log = on_log or _noop
    if os.path.exists(baseline_path):
        log(f"Loading existing baseline: {os.path.basename(baseline_path)}")
        baseline = Baseline.from_json(baseline_path)
        wanted = {
            "opponent_band": list(opponent_band) if opponent_band else None,
            "diff_range": list(diff_range) if diff_range else None,
        }
        if (opponent_band or diff_range) and (baseline.filters or {}) != wanted:
            log("WARNING: opponent/diff filters set, but the loaded baseline "
                f"was built with filters={baseline.filters} — the human side "
                "won't match the filtered bot population. Rebuild the baseline "
                "(delete the JSON) for a like-for-like comparison.")
        return baseline
    if not corpus_pgn:
        raise FileNotFoundError(
            f"Baseline {baseline_path} not found and no corpus given to build it."
        )
    log(f"Building baseline from {os.path.basename(corpus_pgn)} (this is the slow step)...")
    baseline = build_baseline(
        corpus_pgn, rating_band, cfg,
        max_games=max_games, min_moves=min_moves, on_progress=on_progress,
        opponent_band=opponent_band, diff_range=diff_range,
    )
    os.makedirs(os.path.dirname(os.path.abspath(baseline_path)), exist_ok=True)
    baseline.to_json(baseline_path)
    log(f"Baseline built: {baseline.n_units} human units -> {baseline_path}")
    return baseline


@dataclass
class DiagnosticSpec:
    username: str
    rating_band: tuple[int, int]
    perf: str = "bullet"                     # Lichess speed category
    time_control: Optional[str] = None       # exact clock, e.g. "60+0" (fetch filter)
    bot_pgn: Optional[str] = None            # if None, fetched from Lichess
    baseline_path: str = ""
    corpus_pgn: Optional[str] = None         # used to build baseline if missing
    bot_max_games: int = 300
    baseline_max_games: Optional[int] = None
    fetch_max_games: int = 300
    # Optional unit filters, applied to BOTH the baseline humans and the bot's
    # units so the two populations stay comparable (e.g. underdogs vs underdogs).
    opponent_band: Optional[tuple[int, int]] = None
    diff_range: Optional[tuple[Optional[int], Optional[int]]] = None


def run_diagnostic(
    cfg: AnalysisConfig,
    spec: DiagnosticSpec,
    workdir: str,
    on_log: LogFn = None,
    on_progress: ProgFn = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> DiagnosticResult:
    """Full pipeline: fetch bot games (if needed) -> ensure baseline -> report."""
    log = on_log or _noop
    phase = on_phase or _noop
    os.makedirs(workdir, exist_ok=True)

    # 1. Bot games.
    bot_pgn = spec.bot_pgn
    if not bot_pgn:
        phase("fetch")
        # TC-specific cache name so 60+0 and 30+0 fetches don't shadow each other.
        tc_tag = f"_{spec.time_control.replace('+', 'plus')}" if spec.time_control else ""
        bot_pgn = os.path.join(workdir, f"bot_{spec.username}{tc_tag}.pgn")
        if os.path.exists(bot_pgn):
            log(f"Using cached bot games: {bot_pgn}")
        else:
            fetch_user_games(
                spec.username, bot_pgn,
                max_games=spec.fetch_max_games, perf_type=spec.perf,
                time_control=spec.time_control,
                on_log=on_log,
            )

    # 2. Baseline.
    phase("baseline")
    baseline = ensure_baseline(
        cfg, spec.baseline_path, spec.corpus_pgn, spec.rating_band,
        max_games=spec.baseline_max_games, on_log=on_log, on_progress=on_progress,
        opponent_band=spec.opponent_band, diff_range=spec.diff_range,
    )

    # 3. Report.
    phase("report")
    return generate_report(
        cfg, bot_pgn, spec.username, baseline,
        max_games=spec.bot_max_games, on_log=on_log, on_progress=on_progress,
        opponent_band=spec.opponent_band, diff_range=spec.diff_range,
    )


def save_result(result: DiagnosticResult, md_path: str, json_path: str) -> None:
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(result.markdown)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result.report, fh, indent=2)
