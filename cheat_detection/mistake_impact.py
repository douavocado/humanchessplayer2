"""Diagnostic: what characterizes the moves that actually cost games --
and whether that profile is bot-specific or just how bullet chess is,
by comparing against real human games cut by the identical rule.

bucket_diagnostic.py asks "does the bot look human in position type X" --
useful for authenticity, but a validated fix there (moderate_breadth_fix)
turned out to move real game outcomes by roughly the noise floor when
measured against the bot's raw PGN result. That's because most games in
bullet end in a clock-race timeout, not a board result -- see
simulation/adjudicate_result.py, which re-scores each game from a real
Stockfish eval taken at the point where either side's clock first drops
below a quarter of their initial time (the same rule for bot and human
PGNs -- both carry [%clk] -- so the two populations are cut identically
and directly comparable).

This module: among games the *adjudicated* (clock-noise-stripped) result
says a side actually lost, which of the losing side's moves (before the
cutoff) did the most damage, and what characterizes them (game phase,
instant vs. deliberated, how far into the game, position
sharpness/ambiguity)? Reported against the base rate of the same side's
own moves, so it reads as "how concentrated is the damage", not just "what
does an average move look like". Run once on bot self-play and once on a
human sample at the same rating to see which of those concentration
patterns are bot-specific vs. just intrinsic to bullet.

Usage:
  venv/bin/python -m cheat_detection.mistake_impact \\
      --pgn simulation/games/ralph_stab_seed190000.pgn \\
      --player SimBotWhite SimBotBlack --out-md bot.md
  venv/bin/python -m cheat_detection.mistake_impact \\
      --pgn cheat_detection/corpora/bullet_1plus0_2300_plus.pgn \\
      --rating 2500 2800 --max-games 1500 --out-md human.md
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Optional

import chess.engine
import chess.pgn

from simulation.adjudicate_result import adjudicate_game

from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer
from .features import MoveFeatures, extract_move_features
from .pgn_loader import parse_game
from .pipeline import _sides_of_interest

PLY_THIRDS = ["first third", "middle third", "final third"]


def _ply_third(ply: int, cutoff_ply: int) -> str:
    frac = ply / max(cutoff_ply, 1)
    if frac < 1 / 3:
        return PLY_THIRDS[0]
    if frac < 2 / 3:
        return PLY_THIRDS[1]
    return PLY_THIRDS[2]


def _loser_color(adj: dict) -> Optional[bool]:
    import chess
    if adj["adjudicated_result"] == "1/2-1/2":
        return None
    return chess.BLACK if adj["adjudicated_result"] == "1-0" else chess.WHITE


def collect(pgn_path: str, cfg: AnalysisConfig,
            player_filter: Optional[set[str]] = None,
            rating_band: Optional[tuple[int, int]] = None,
            max_games: Optional[int] = None):
    losers_all_moves: list[MoveFeatures] = []
    losers_worst_move: list[tuple] = []
    n_games_with_loser = 0

    adj_engine = chess.engine.SimpleEngine.popen_uci(cfg.stockfish_path)
    adj_engine.configure({"Threads": cfg.threads, "Hash": cfg.hash_mb})

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as fh, EngineAnalyzer(cfg) as analyzer:
        gi = 0
        while True:
            if max_games is not None and gi >= max_games:
                break
            raw_game = chess.pgn.read_game(fh)
            if raw_game is None:
                break
            gi += 1
            rec = parse_game(raw_game)
            if rec is None:
                continue
            adj = adjudicate_game(raw_game, adj_engine)
            loser = _loser_color(adj)
            if loser is None:
                continue
            cutoff_ply = adj["cutoff_ply"]

            for color, name, elo in _sides_of_interest(rec, player_filter, rating_band):
                if color != loser:
                    continue
                mfeats: list[MoveFeatures] = []
                for mrec in rec.moves:
                    if mrec.mover != color or mrec.ply >= cutoff_ply:
                        continue
                    pe = analyzer.analyse(mrec.fen_before)
                    if not pe.candidates:
                        continue
                    played_cp = analyzer.eval_after_move(mrec.fen_before, mrec.move_uci)
                    mf = extract_move_features(mrec, pe, played_cp, cfg)
                    if mf is not None:
                        mfeats.append(mf)
                analyzer.flush()
                if not mfeats:
                    continue
                n_games_with_loser += 1
                losers_all_moves.extend(mfeats)
                worst = max(mfeats, key=lambda m: m.wc_loss)
                losers_worst_move.append((worst, cutoff_ply))

            if gi % 50 == 0:
                print(f"  {gi} games processed, {n_games_with_loser} adjudicated losses so far...")

    adj_engine.quit()
    return losers_all_moves, losers_worst_move, n_games_with_loser


def _dist(items: list[str]) -> dict[str, float]:
    c = Counter(items)
    n = len(items)
    return {k: round(100 * v / n, 1) for k, v in c.most_common()} if n else {}


def render(all_moves: list[MoveFeatures], worst_moves: list, cfg: AnalysisConfig, n_games: int, label: str) -> str:
    lines = [f"# Mistake-impact diagnostic: {label} (adjudicated losses only, n={n_games} games)\n"]

    lines.append(f"## Base rate: all of the losing side's moves up to the adjudication cutoff (n={len(all_moves)})\n")
    lines.append(f"- Phase mix: {_dist([m.phase for m in all_moves])}")
    lines.append(f"- Instant (emt < {cfg.instant_move_secs}s) fraction: "
                  f"{100*sum(1 for m in all_moves if (m.emt or 0) < cfg.instant_move_secs)/len(all_moves):.1f}%")
    lines.append(f"- Blunder rate: {100*sum(m.is_blunder for m in all_moves)/len(all_moves):.1f}%")
    lines.append(f"- Mean win-chance loss: {sum(m.wc_loss for m in all_moves)/len(all_moves):.4f}")
    has_kind = any(m.kind is not None for m in all_moves)
    if has_kind:
        lines.append(f"- Kind mix (bot-only signal, real Lichess PGNs never have this): "
                      f"{_dist([m.kind for m in all_moves])}")
    lines.append("")

    lines.append(f"## The single worst (highest win-chance-loss) move per adjudicated loss (n={len(worst_moves)})\n")
    worst_only = [m for m, _ in worst_moves]
    thirds = [_ply_third(m.ply, cutoff) for m, cutoff in worst_moves]
    lines.append(f"- Phase mix: {_dist([m.phase for m in worst_only])}")
    lines.append(f"- Position-in-game (ply / cutoff_ply): {_dist(thirds)}")
    lines.append(f"- Instant (emt < {cfg.instant_move_secs}s) fraction: "
                  f"{100*sum(1 for m in worst_only if (m.emt or 0) < cfg.instant_move_secs)/len(worst_only):.1f}%")
    lines.append(f"- Mean sharpness of the position: {sum(m.sharpness for m in worst_only)/len(worst_only):.3f}")
    lines.append(f"- Mean ambiguity: {sum(m.ambiguity for m in worst_only)/len(worst_only):.2f}")
    lines.append(f"- Mean win-chance loss: {sum(m.wc_loss for m in worst_only)/len(worst_only):.4f}")
    lines.append(f"- Fraction that are already a counted blunder (wc_loss >= {cfg.blunder_wc_loss}): "
                  f"{100*sum(m.is_blunder for m in worst_only)/len(worst_only):.1f}%")
    if has_kind:
        lines.append(f"- Kind mix: {_dist([m.kind for m in worst_only])}")
        instant_worst = [m for m in worst_only if (m.emt or 0) < cfg.instant_move_secs]
        if instant_worst:
            lines.append(f"- Kind mix of just the instant worst-mistakes (n={len(instant_worst)}): "
                          f"{_dist([m.kind for m in instant_worst])}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--player", nargs="+", help="bot player name(s) to filter to (self-play PGNs)")
    ap.add_argument("--rating", nargs=2, type=int, help="human rating band, e.g. 2500 2800")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    cfg = AnalysisConfig(workers=args.workers)
    player_filter = {p.lower() for p in args.player} if args.player else None
    rating_band = tuple(args.rating) if args.rating else None
    label = args.pgn if player_filter is None else f"{args.pgn} [{', '.join(args.player)}]"
    if rating_band:
        label += f" [{rating_band[0]}-{rating_band[1]}]"

    print(f"Collecting mistake-impact data from {label} ...")
    all_moves, worst_moves, n_games = collect(
        args.pgn, cfg, player_filter=player_filter, rating_band=rating_band, max_games=args.max_games)
    print(f"  {n_games} adjudicated-loss games, {len(all_moves)} candidate moves")

    md = render(all_moves, worst_moves, cfg, n_games, label)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"Wrote {args.out_md}")
    else:
        print(md)


if __name__ == "__main__":
    main()
