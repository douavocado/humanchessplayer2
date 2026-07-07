"""Compare the bot's play against the human baseline and render a report.

The report is descriptive and comparison-based: for each feature it shows the
human distribution, the bot's value, and how many standard deviations apart
they are. It highlights where the bot is *least* human-like and explains what
each feature means, so you can decide what to tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .baseline import Baseline
from .config import AnalysisConfig
from .features import FEATURE_KEYS
from .pipeline import Unit

# Human-readable metadata: label, unit, and what a divergence means.
FEATURE_META: dict[str, tuple[str, str]] = {
    "t1_rate": ("Top-1 engine-move match rate", "fraction of moves equal to Stockfish's #1"),
    "t2_rate": ("Top-2 match rate", "fraction within Stockfish's top 2"),
    "t3_rate": ("Top-3 match rate", "fraction within Stockfish's top 3"),
    "acpl": ("Average centipawn loss", "overall accuracy; lower = stronger"),
    "acpl_opening": ("ACPL (opening)", "accuracy in the opening phase"),
    "acpl_middlegame": ("ACPL (middlegame)", "accuracy in the middlegame"),
    "acpl_endgame": ("ACPL (endgame)", "accuracy in the endgame"),
    "mean_wc_loss": ("Mean win-chance loss", "avg win-probability given up per move"),
    "blunder_rate": ("Blunder rate", "fraction of moves losing >=15% win chance"),
    "movetime_mean": ("Mean move time (s)", "average seconds per move"),
    "movetime_std": ("Move-time std (s)", "spread of move times"),
    "movetime_cv": ("Move-time variation", "std/mean; humans vary a lot, engines less"),
    "instant_move_rate": ("Instant-move rate", "fraction played in <1s"),
    "instant_in_sharp_rate": ("Instant moves in sharp positions",
                              "fraction of critical positions played in <1s -- a strong tell"),
    "corr_time_cploss": ("Time vs. move-loss correlation",
                         "humans think longer before worse moves; near 0 is unhuman"),
    "corr_time_sharpness": ("Time vs. position-sharpness correlation",
                            "humans slow down in critical positions"),
    "mean_ambiguity": ("Mean position ambiguity", "avg count of near-equal good moves faced"),
}


def _bot_summary(units: list[Unit]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for key in FEATURE_KEYS:
        vals = [u.features[key] for u in units if u.features.get(key) is not None]
        out[key] = (sum(vals) / len(vals)) if vals else None
    return out


@dataclass
class FeatureComparison:
    key: str
    human_mean: Optional[float]
    human_std: Optional[float]
    bot_value: Optional[float]
    zscore: Optional[float]
    flagged: bool


def compare(bot_units: list[Unit], baseline: Baseline, cfg: AnalysisConfig) -> list[FeatureComparison]:
    bot = _bot_summary(bot_units)
    comps: list[FeatureComparison] = []
    for key in FEATURE_KEYS:
        hs = baseline.stats.get(key, {})
        hmean, hstd = hs.get("mean"), hs.get("std")
        bval = bot.get(key)
        z = None
        flagged = False
        if bval is not None and hmean is not None and hstd:
            z = (bval - hmean) / hstd
            flagged = abs(z) >= cfg.flag_zscore
        comps.append(FeatureComparison(key, hmean, hstd, bval, z, flagged))
    return comps


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_markdown(
    comps: list[FeatureComparison],
    baseline: Baseline,
    bot_units: list[Unit],
    cfg: AnalysisConfig,
) -> str:
    n_moves = sum(u.n_moves for u in bot_units)
    scored = [c for c in comps if c.zscore is not None]
    mean_abs_z = (sum(abs(c.zscore) for c in scored) / len(scored)) if scored else None
    flagged = [c for c in comps if c.flagged]

    lines: list[str] = []
    lines.append("# Human-likeness diagnostic report\n")
    lines.append(
        f"- Baseline: {baseline.n_units} human units, "
        f"rating {baseline.rating_band[0]}-{baseline.rating_band[1]}\n"
        f"- Bot sample: {len(bot_units)} games/units, {n_moves} moves\n"
        f"- Engine: Stockfish depth {cfg.depth}, multipv {cfg.multipv}\n"
    )

    if mean_abs_z is not None:
        if mean_abs_z < 1.0:
            verdict = "within normal human variation"
        elif mean_abs_z < 2.0:
            verdict = "noticeably distinguishable from human play"
        else:
            verdict = "strongly distinguishable from human play"
        lines.append(
            f"\n**Overall divergence: mean |z| = {mean_abs_z:.2f}** "
            f"({verdict}). {len(flagged)} feature(s) beyond {cfg.flag_zscore}sigma.\n"
        )

    # Most divergent features first.
    lines.append("\n## Biggest divergences from human play\n")
    if flagged:
        for c in sorted(flagged, key=lambda c: -abs(c.zscore)):
            label, meaning = FEATURE_META.get(c.key, (c.key, ""))
            direction = "higher" if c.zscore > 0 else "lower"
            lines.append(
                f"- **{label}** ({direction} than human by {abs(c.zscore):.1f}sigma): "
                f"bot {_fmt(c.bot_value)} vs human {_fmt(c.human_mean)}+/-{_fmt(c.human_std)}. "
                f"_{meaning}._"
            )
    else:
        lines.append("_No feature exceeded the flag threshold; the bot tracks the human "
                     "baseline closely on all measured axes._")

    # Full table.
    lines.append("\n## All features\n")
    lines.append("| Feature | Human mean +/- std | Bot | z |")
    lines.append("|---|---|---|---|")
    for c in comps:
        label = FEATURE_META.get(c.key, (c.key, ""))[0]
        mark = " ⚑" if c.flagged else ""
        lines.append(
            f"| {label}{mark} | {_fmt(c.human_mean)} +/- {_fmt(c.human_std)} "
            f"| {_fmt(c.bot_value)} | {_fmt(c.zscore, 2)} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(bot_units, baseline, cfg):
    comps = compare(bot_units, baseline, cfg)
    md = render_markdown(comps, baseline, bot_units, cfg)
    report_dict = {
        "baseline": {"n_units": baseline.n_units, "rating_band": list(baseline.rating_band)},
        "bot": {"n_units": len(bot_units), "n_moves": sum(u.n_moves for u in bot_units)},
        "features": [
            {
                "key": c.key,
                "human_mean": c.human_mean,
                "human_std": c.human_std,
                "bot_value": c.bot_value,
                "zscore": c.zscore,
                "flagged": c.flagged,
            }
            for c in comps
        ],
    }
    return md, report_dict
