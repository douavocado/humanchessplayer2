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
    "t1_rate_ambiguous": ("Top-1 match rate (ambiguous positions)",
                          "engine agreement when several near-equal moves existed -- "
                          "human accuracy collapses here, engine accuracy doesn't"),
    "t1_rate_forced": ("Top-1 match rate (clear-best positions)",
                       "engine agreement when one move was clearly best; "
                       "high for everyone, so divergence here is less telling"),
    "acpl": ("Average centipawn loss", "overall accuracy; lower = stronger"),
    "acpl_opening": ("ACPL (opening)", "accuracy in the opening phase"),
    "acpl_middlegame": ("ACPL (middlegame)", "accuracy in the middlegame"),
    "acpl_endgame": ("ACPL (endgame)", "accuracy in the endgame"),
    "acpl_timepressure": ("ACPL under time pressure",
                          "accuracy with a low clock; humans degrade sharply in a "
                          "scramble, flat quality is a tell"),
    "blunder_rate_timepressure": ("Blunder rate under time pressure",
                                  "blunders with a low clock; humans blunder far more "
                                  "in a scramble"),
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


def _welch_test(
    hmean: Optional[float], hstd: Optional[float], hn: Optional[int],
    bmean: Optional[float], bstd: Optional[float], bn: Optional[int],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Welch two-sample t-test (unequal variances).

    Returns (t, df, two-sided p), or (None, None, None) when either sample is
    too small (n < 2) or both variances are zero.
    """
    import math

    if None in (hmean, hstd, bmean, bstd) or not hn or not bn or hn < 2 or bn < 2:
        return None, None, None
    vh, vb = hstd ** 2 / hn, bstd ** 2 / bn
    se2 = vh + vb
    if se2 <= 0:
        return None, None, None
    t = (bmean - hmean) / math.sqrt(se2)
    df = se2 ** 2 / (vh ** 2 / (hn - 1) + vb ** 2 / (bn - 1))
    return t, df, _two_sided_p_from_t(t, df)


def _two_sided_p_from_t(t: float, df: float) -> float:
    import math
    try:
        from scipy import stats as _scipy_stats
        return float(2 * _scipy_stats.t.sf(abs(t), df))
    except ImportError:  # pragma: no cover - venv ships scipy
        return math.erfc(abs(t) / math.sqrt(2))  # normal approximation


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _brown_forsythe(hvals: Optional[list[float]], bvals: Optional[list[float]]) -> Optional[float]:
    """Brown-Forsythe (median-centred Levene) test for equal variances.

    Answers "is the bot's game-to-game spread human-like?" — a bot can match
    every human mean while being far too consistent (or too erratic) from game
    to game. With two groups the ANOVA-on-|deviations| reduces to a pooled
    two-sample t-test on |x - group median|. Returns the two-sided p-value, or
    None when either side lacks the raw per-unit values or has n < 3.
    """
    import math

    if not hvals or not bvals or len(hvals) < 3 or len(bvals) < 3:
        return None
    hdev = [abs(x - _median(hvals)) for x in hvals]
    bdev = [abs(x - _median(bvals)) for x in bvals]
    nh, nb = len(hdev), len(bdev)
    mh, mb = sum(hdev) / nh, sum(bdev) / nb
    ss = sum((x - mh) ** 2 for x in hdev) + sum((x - mb) ** 2 for x in bdev)
    df = nh + nb - 2
    if ss <= 0:
        return None
    sp = math.sqrt(ss / df)
    se = sp * math.sqrt(1 / nh + 1 / nb)
    if se == 0:
        return None
    return _two_sided_p_from_t((mb - mh) / se, df)


@dataclass
class FeatureComparison:
    key: str
    human_mean: Optional[float]
    human_std: Optional[float]
    human_n: Optional[int]
    bot_value: Optional[float]
    bot_std: Optional[float]
    bot_n: Optional[int]
    zscore: Optional[float]        # effect size: (bot - human) / human_std
    t_stat: Optional[float]        # Welch two-sample t
    df: Optional[float]
    p_value: Optional[float]       # two-sided
    flagged: bool
    # Variance comparison (game-to-game consistency), independent of the mean:
    var_ratio: Optional[float]     # bot_std / human_std
    var_pvalue: Optional[float]    # Brown-Forsythe two-sided p
    var_flagged: bool


def compare(bot_units: list[Unit], baseline: Baseline, cfg: AnalysisConfig) -> list[FeatureComparison]:
    from .baseline import _summarize, collect_values

    bot = _summarize(bot_units)
    bot_values = collect_values(bot_units)
    human_values = baseline.values or {}
    comps: list[FeatureComparison] = []
    for key in FEATURE_KEYS:
        hs = baseline.stats.get(key, {})
        hmean, hstd, hn = hs.get("mean"), hs.get("std"), hs.get("n")
        bs = bot[key]
        bval, bstd, bn = bs["mean"], bs["std"], bs["n"]
        z = None
        if bval is not None and hmean is not None and hstd:
            z = (bval - hmean) / hstd
        t, df, p = _welch_test(hmean, hstd, hn, bval, bstd, bn)
        if cfg.test_mode == "welch":
            flagged = p is not None and p < cfg.flag_pvalue
        else:
            flagged = z is not None and abs(z) >= cfg.flag_zscore
        var_ratio = (bstd / hstd) if (bstd is not None and hstd) else None
        var_p = _brown_forsythe(human_values.get(key), bot_values.get(key))
        var_flagged = var_p is not None and var_p < cfg.flag_pvalue
        comps.append(FeatureComparison(
            key, hmean, hstd, hn, bval, bstd, bn, z, t, df, p, flagged,
            var_ratio, var_p, var_flagged))
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

    welch = cfg.test_mode == "welch"
    test_desc = (
        f"Welch two-sample t-test, flag at p < {cfg.flag_pvalue:g}"
        if welch else
        f"effect size vs human spread, flag at |z| >= {cfg.flag_zscore:g}"
    )

    lines: list[str] = []
    lines.append("# Human-likeness diagnostic report\n")
    lines.append(
        f"- Baseline: {baseline.n_units} human units, "
        f"rating {baseline.rating_band[0]}-{baseline.rating_band[1]}\n"
        f"- Bot sample: {len(bot_units)} games/units, {n_moves} moves\n"
        f"- Engine: Stockfish depth {cfg.depth}, multipv {cfg.multipv}\n"
        f"- Test: {test_desc}\n"
    )

    if mean_abs_z is not None:
        if mean_abs_z < 1.0:
            verdict = "within normal human variation"
        elif mean_abs_z < 2.0:
            verdict = "noticeably distinguishable from human play"
        else:
            verdict = "strongly distinguishable from human play"
        flag_desc = (
            f"significant at p < {cfg.flag_pvalue:g} (Welch)" if welch
            else f"beyond {cfg.flag_zscore:g}sigma"
        )
        lines.append(
            f"\n**Overall divergence: mean |z| = {mean_abs_z:.2f}** "
            f"({verdict}). {len(flagged)} feature(s) {flag_desc}.\n"
        )

    # Most divergent features first (by p when testing, by |z| otherwise).
    lines.append("\n## Biggest divergences from human play\n")
    if flagged:
        if welch:
            ordered = sorted(flagged, key=lambda c: c.p_value)
        else:
            ordered = sorted(flagged, key=lambda c: -abs(c.zscore))
        for c in ordered:
            label, meaning = FEATURE_META.get(c.key, (c.key, ""))
            direction = "higher" if (c.zscore or 0) > 0 else "lower"
            stat = (
                f"t = {_fmt(c.t_stat, 2)}, p = {_fmt(c.p_value, 4)}" if welch
                else f"{abs(c.zscore):.1f}sigma"
            )
            lines.append(
                f"- **{label}** ({direction} than human; {stat}): "
                f"bot {_fmt(c.bot_value)} vs human {_fmt(c.human_mean)}+/-{_fmt(c.human_std)}. "
                f"_{meaning}._"
            )
    else:
        lines.append("_No feature exceeded the flag threshold; the bot tracks the human "
                     "baseline closely on all measured axes._")

    # Game-to-game consistency: a bot can match every human mean while being
    # far too consistent (or erratic) across games. Brown-Forsythe on the
    # per-game values, flagged at the same alpha as the Welch test.
    var_flagged = [c for c in comps if c.var_flagged]
    lines.append("\n## Variance divergences (game-to-game consistency)\n")
    if var_flagged:
        for c in sorted(var_flagged, key=lambda c: c.var_pvalue):
            label, _ = FEATURE_META.get(c.key, (c.key, ""))
            direction = "more" if (c.var_ratio or 0) > 1 else "less"
            lines.append(
                f"- **{label}** varies {direction} game-to-game than human "
                f"(std ratio {_fmt(c.var_ratio, 2)}x, Brown-Forsythe "
                f"p = {_fmt(c.var_pvalue, 4)}): bot std {_fmt(c.bot_std)} vs "
                f"human std {_fmt(c.human_std)}."
            )
    elif any(c.var_pvalue is not None for c in comps):
        lines.append(f"_No feature's game-to-game spread differs significantly from human "
                     f"(Brown-Forsythe, alpha = {cfg.flag_pvalue:g})._")
    else:
        lines.append("_Not testable: the baseline lacks per-game values (rebuild it) or "
                     "there are too few games._")

    # Full table (all statistics shown regardless of which one flags).
    lines.append("\n## All features\n")
    lines.append("| Feature | Human mean +/- std | Bot | z | t | p | std ratio | var p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in comps:
        label = FEATURE_META.get(c.key, (c.key, ""))[0]
        mark = " ⚑" if c.flagged else ""
        vmark = " ⚑v" if c.var_flagged else ""
        lines.append(
            f"| {label}{mark}{vmark} | {_fmt(c.human_mean)} +/- {_fmt(c.human_std)} "
            f"| {_fmt(c.bot_value)} | {_fmt(c.zscore, 2)} "
            f"| {_fmt(c.t_stat, 2)} | {_fmt(c.p_value, 4)} "
            f"| {_fmt(c.var_ratio, 2)} | {_fmt(c.var_pvalue, 4)} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(bot_units, baseline, cfg):
    from .baseline import collect_values

    comps = compare(bot_units, baseline, cfg)
    md = render_markdown(comps, baseline, bot_units, cfg)
    report_dict = {
        "baseline": {"n_units": baseline.n_units, "rating_band": list(baseline.rating_band)},
        "bot": {"n_units": len(bot_units), "n_moves": sum(u.n_moves for u in bot_units)},
        "test_mode": cfg.test_mode,
        "flag_zscore": cfg.flag_zscore,
        "flag_pvalue": cfg.flag_pvalue,
        # Per-unit distributions for interactive drill-down (human side is
        # absent when the baseline predates the `values` field).
        "human_values": baseline.values,
        "bot_values": collect_values(bot_units),
        "features": [
            {
                "key": c.key,
                "human_mean": c.human_mean,
                "human_std": c.human_std,
                "human_n": c.human_n,
                "bot_value": c.bot_value,
                "bot_std": c.bot_std,
                "bot_n": c.bot_n,
                "zscore": c.zscore,
                "t_stat": c.t_stat,
                "df": c.df,
                "p_value": c.p_value,
                "flagged": c.flagged,
                "var_ratio": c.var_ratio,
                "var_pvalue": c.var_pvalue,
                "var_flagged": c.var_flagged,
            }
            for c in comps
        ],
    }
    return md, report_dict
