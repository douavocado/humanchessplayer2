"""Per-move and aggregate feature extraction.

Per-move features are ported from Irwin's ``AnalysedMove`` conventions
(move rank, ambiguity, winning-chances loss). Aggregate features follow
Kaladin's spirit: distributions and correlations over a player's whole
history (T1/T2/T3 match rates, ACPL by phase, move-time-vs-difficulty
correlation, timing variance).

Everything here is descriptive: the output is a set of interpretable numbers
per player, ready to be compared against a human baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import chess

from .config import AnalysisConfig
from .engine_analysis import PositionEval
from .pgn_loader import MoveRecord


def winning_chances(cp: int) -> float:
    """Map centipawns (mover's perspective) to a win probability in [0, 1].

    Uses the logistic form Lichess/Irwin use for evaluation-to-win-chance.
    """
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


# --- non-pawn material for phase detection ---
_NPM = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _non_pawn_material(board: chess.Board) -> int:
    total = 0
    for piece_type, val in _NPM.items():
        total += val * len(board.pieces(piece_type, chess.WHITE))
        total += val * len(board.pieces(piece_type, chess.BLACK))
    return total


def _phase(board: chess.Board, ply: int, cfg: AnalysisConfig) -> str:
    if ply < cfg.opening_plies:
        return "opening"
    if _non_pawn_material(board) <= cfg.endgame_npm:
        return "endgame"
    return "middlegame"


@dataclass
class MoveFeatures:
    ply: int
    phase: str
    rank: Optional[int]         # rank of played move among candidates (1-based), None if beyond top-k
    within_topk: bool
    matched_top1: bool
    matched_top2: bool
    matched_top3: bool
    cp_loss: float              # centipawn loss vs best (>= 0)
    wc_loss: float              # winning-chance loss vs best, in [0, 1]
    ambiguity: int              # number of candidate moves ~as good as the best
    sharpness: float            # win-prob spread across candidates (position criticality)
    n_legal: int
    is_blunder: bool
    emt: Optional[float]
    clock_before: Optional[float]


def extract_move_features(
    rec: MoveRecord, pe: PositionEval, played_cp: Optional[int], cfg: AnalysisConfig
) -> Optional[MoveFeatures]:
    if not pe.candidates:
        return None
    board = chess.Board(rec.fen_before)

    best_cp = pe.best_cp()
    best_wc = winning_chances(best_cp)

    rank = pe.rank_of(rec.move_uci)
    within = rank is not None

    if played_cp is None:
        return None
    played_wc = winning_chances(played_cp)

    cp_loss = max(0.0, best_cp - played_cp)
    wc_loss = max(0.0, best_wc - played_wc)

    # Ambiguity: candidates within the win-prob window of the best move.
    ambiguity = sum(
        1 for c in pe.candidates
        if best_wc - winning_chances(c.cp) <= cfg.ambiguity_wc_window
    )
    # Sharpness: how much win-prob separates the best from the worst *considered*
    # candidate. High when a position has both great and bad plausible moves.
    worst_wc = min(winning_chances(c.cp) for c in pe.candidates)
    sharpness = best_wc - worst_wc

    return MoveFeatures(
        ply=rec.ply,
        phase=_phase(board, rec.ply, cfg),
        rank=rank,
        within_topk=within,
        matched_top1=(rank == 1),
        matched_top2=(rank is not None and rank <= 2),
        matched_top3=(rank is not None and rank <= 3),
        cp_loss=cp_loss,
        wc_loss=wc_loss,
        ambiguity=ambiguity,
        sharpness=sharpness,
        n_legal=board.legal_moves.count(),
        is_blunder=wc_loss >= cfg.blunder_wc_loss,
        emt=rec.emt,
        clock_before=rec.clock_before,
    )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


# The feature keys the report and baseline compare on. Keep names stable.
FEATURE_KEYS = [
    "t1_rate", "t2_rate", "t3_rate",
    "t1_rate_ambiguous", "t1_rate_forced",
    "acpl", "acpl_opening", "acpl_middlegame", "acpl_endgame",
    "acpl_timepressure", "blunder_rate_timepressure",
    "mean_wc_loss", "blunder_rate",
    "movetime_mean", "movetime_std", "movetime_cv",
    "instant_move_rate", "instant_in_sharp_rate",
    "corr_time_cploss", "corr_time_sharpness",
    "mean_ambiguity",
]


def aggregate_features(moves: list[MoveFeatures], cfg: AnalysisConfig) -> dict[str, Optional[float]]:
    """Collapse a player's move-level features into one comparable vector."""
    if not moves:
        return {k: None for k in FEATURE_KEYS}

    n = len(moves)
    out: dict[str, Optional[float]] = {}

    # Engine-match rates (the classic cheat signal: high top-1 agreement).
    out["t1_rate"] = sum(m.matched_top1 for m in moves) / n
    out["t2_rate"] = sum(m.matched_top2 for m in moves) / n
    out["t3_rate"] = sum(m.matched_top3 for m in moves) / n

    # Selective accuracy: matching the engine on a forced/obvious move says
    # nothing; matching it when several near-equal candidates existed is the
    # discriminating signal (human T1 collapses there, engine T1 doesn't).
    ambiguous = [m for m in moves if m.ambiguity >= 2]
    forced = [m for m in moves if m.ambiguity == 1]
    out["t1_rate_ambiguous"] = (
        sum(m.matched_top1 for m in ambiguous) / len(ambiguous) if ambiguous else None
    )
    out["t1_rate_forced"] = (
        sum(m.matched_top1 for m in forced) / len(forced) if forced else None
    )

    # Accuracy overall and by phase.
    out["acpl"] = _mean([m.cp_loss for m in moves])
    for phase in ("opening", "middlegame", "endgame"):
        out[f"acpl_{phase}"] = _mean([m.cp_loss for m in moves if m.phase == phase])
    out["mean_wc_loss"] = _mean([m.wc_loss for m in moves])
    out["blunder_rate"] = sum(m.is_blunder for m in moves) / n

    # Time-pressure degradation: humans get markedly worse in a scramble;
    # a bot whose quality is flat regardless of clock is a strong tell.
    pressured = [
        m for m in moves
        if m.clock_before is not None and m.clock_before < cfg.time_pressure_secs
    ]
    out["acpl_timepressure"] = _mean([m.cp_loss for m in pressured])
    out["blunder_rate_timepressure"] = (
        sum(m.is_blunder for m in pressured) / len(pressured) if pressured else None
    )

    # Timing distribution (only moves with a known emt).
    timed = [m for m in moves if m.emt is not None]
    times = [m.emt for m in timed]
    out["movetime_mean"] = _mean(times)
    out["movetime_std"] = _std(times)
    mt_mean = out["movetime_mean"]
    out["movetime_cv"] = (out["movetime_std"] / mt_mean) if (out["movetime_std"] and mt_mean) else None

    if timed:
        out["instant_move_rate"] = sum(m.emt < cfg.instant_move_secs for m in timed) / len(timed)
        sharp = [m for m in timed if m.sharpness >= 0.25]
        out["instant_in_sharp_rate"] = (
            sum(m.emt < cfg.instant_move_secs for m in sharp) / len(sharp) if sharp else None
        )
        # The key human signal: do you spend longer on harder / costlier moves?
        out["corr_time_cploss"] = _pearson([m.emt for m in timed], [m.cp_loss for m in timed])
        out["corr_time_sharpness"] = _pearson([m.emt for m in timed], [m.sharpness for m in timed])
    else:
        out["instant_move_rate"] = None
        out["instant_in_sharp_rate"] = None
        out["corr_time_cploss"] = None
        out["corr_time_sharpness"] = None

    out["mean_ambiguity"] = _mean([float(m.ambiguity) for m in moves])
    return out
