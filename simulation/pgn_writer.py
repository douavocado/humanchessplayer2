"""Writes simulated games as PGN with [%clk] tags.

Clock comments are quantised to whole seconds — the bot's real Lichess
exports carry integer-second clocks only, and cheat_detection derives move
times from clock diffs, so matching granularity matters (sub-second tags
would make simulated games measurably different from real ones).
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
}


def game_to_pgn(sim: SimGame, white_name: str, black_name: str,
                white_elo: int, black_elo: int, round_no: int = 1,
                date: str = "????.??.??") -> chess.pgn.Game:
    game = chess.pgn.Game()
    inc = int(sim.increment)
    game.headers["Event"] = "Simulated bot self-play"
    game.headers["Site"] = "local simulation"
    game.headers["Date"] = date
    game.headers["Round"] = str(round_no)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["WhiteElo"] = str(white_elo)
    game.headers["BlackElo"] = str(black_elo)
    game.headers["TimeControl"] = f"{int(sim.initial_time)}+{inc}"
    game.headers["Result"] = sim.result
    game.headers["Termination"] = TERMINATION_HEADER.get(sim.termination, "Normal")
    game.headers["SimSeed"] = str(sim.seed)

    node = game
    for mv in sim.moves:
        node = node.add_variation(chess.Move.from_uci(mv.move_uci))
        node.set_clock(int(mv.clock_after))
    return game


def write_games(sims: list[SimGame], out_path: str, white_name: str,
                black_name: str, white_elo: int, black_elo: int,
                date: str = "????.??.??") -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, sim in enumerate(sims, start=1):
            game = game_to_pgn(sim, white_name, black_name,
                               white_elo, black_elo, round_no=i, date=date)
            fh.write(str(game) + "\n\n")
