"""Writes simulated games as PGN with [%clk] tags.

Clock comments are quantised to whole seconds — the bot's real Lichess
exports carry integer-second clocks only, and cheat_detection derives move
times from clock diffs, so matching granularity matters (sub-second tags
would make simulated games measurably different from real ones).

Each move also carries a non-standard [%kind ...] comment tag (engine,
premove, ponder_hit, scramble, safe_ponder, scramble_hit -- see
client_model.py) -- the live decision mechanism that produced it. Real
Lichess exports never have this (there's no such label to export), so it's
only ever present on simulated PGNs; cheat_detection.pgn_loader parses it
as an optional field that's None when absent, precisely so bot and human
analysis code stays the same and only the bot side gets the extra signal.
"""

from __future__ import annotations

import chess
import chess.pgn

from .game_runner import SimGame

TERMINATION_HEADER = {
    "checkmate": "Normal",
    "resignation": "Normal",
    "draw": "Normal",
    "timeout": "Time forfeit",
    "max-plies": "Adjudication",
    "adjudicated": "Adjudication",
}


def game_to_pgn(sim: SimGame, round_no: int = 1,
                date: str = "????.??.??") -> chess.pgn.Game:
    game = chess.pgn.Game()
    inc = int(sim.increment)
    game.headers["Event"] = "Simulated bot self-play"
    game.headers["Site"] = "local simulation"
    game.headers["Date"] = date
    game.headers["Round"] = str(round_no)
    # Names/ratings live on the SimGame: with colour alternation they differ
    # game to game.
    game.headers["White"] = sim.white_name
    game.headers["Black"] = sim.black_name
    game.headers["WhiteElo"] = str(sim.white_elo)
    game.headers["BlackElo"] = str(sim.black_elo)
    game.headers["TimeControl"] = f"{int(sim.initial_time)}+{inc}"
    game.headers["Result"] = sim.result
    game.headers["Termination"] = TERMINATION_HEADER.get(sim.termination, "Normal")
    game.headers["SimSeed"] = str(sim.seed)

    node = game
    for mv in sim.moves:
        node = node.add_variation(chess.Move.from_uci(mv.move_uci))
        node.set_clock(int(mv.clock_after))
        node.comment += f" [%kind {mv.kind}]"
    return game


def write_games(sims: list[SimGame], out_path: str,
                date: str = "????.??.??") -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, sim in enumerate(sims, start=1):
            game = game_to_pgn(sim, round_no=i, date=date)
            fh.write(str(game) + "\n\n")
