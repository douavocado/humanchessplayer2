"""Parse PGN files into per-move records with elapsed move times.

Handles standard PGN as produced by Lichess (and the bot itself), reading
``[%clk H:MM:SS]`` tags to reconstruct each player's elapsed move time (emt).
Falls back to explicit ``[%emt]`` tags when present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional

import chess
import chess.pgn

_TC_RE = re.compile(r"^(\d+)(?:\+(\d+))?")
_TC_SECONDS_RE = re.compile(r"^(\d+)(?:\+(\d+))?$")


def parse_tc_seconds(tc: str) -> float:
    """Base clock in seconds from a "180+0"-style time control.

    The increment is parsed but deliberately discarded: initial_time means the
    starting clock, which is what the threshold fractions scale against.
    """
    m = _TC_SECONDS_RE.match((tc or "").strip())
    if not m:
        raise ValueError(f"bad time control {tc!r}; expected e.g. 180+0")
    return float(m.group(1))


@dataclass
class MoveRecord:
    ply: int                    # 0-based ply index within the game
    move_number: int            # standard move number (1-based)
    fen_before: str             # position the mover faced
    move_uci: str
    san: str
    mover: bool                 # chess.WHITE or chess.BLACK (True == White)
    mover_name: str
    emt: Optional[float]        # elapsed move time in seconds, if derivable
    clock_before: Optional[float]  # clock the mover started thinking with, seconds
    clock_after: Optional[float]  # clock remaining after the move, seconds
    kind: Optional[str] = None  # engine/premove/ponder_hit/scramble/... -- only
                                 # present on simulated PGNs (see pgn_writer.py);
                                 # None for real human exports, which have no such label


@dataclass
class GameRecord:
    white: str
    black: str
    white_elo: Optional[int]
    black_elo: Optional[int]
    time_control: str
    base_secs: Optional[int]
    increment: Optional[int]
    result: str
    moves: list[MoveRecord]

    def moves_by(self, name: str) -> list[MoveRecord]:
        """Moves played by the given player name (case-insensitive)."""
        low = name.lower()
        return [m for m in self.moves if m.mover_name.lower() == low]


class TimeControlMismatchError(Exception):
    """A game's TimeControl header disagrees with the configured --tc."""


def check_time_control(game: GameRecord, expected_secs: float, *,
                       strict: bool = True) -> bool:
    """Whether this game's clock matches the analysis configuration.

    Returns True to analyse the game, False to skip it. Raises when the game
    mismatches and `strict` -- the default, because a blended corpus produces
    timing features that look plausible and mean nothing, which is worse than
    a crash.

    A game whose TimeControl is missing or unparseable (`base_secs is None`)
    passes: it cannot be checked, and absence of evidence is not a mismatch.
    """
    if game.base_secs is None:
        return True
    if float(game.base_secs) == float(expected_secs):
        return True
    if strict:
        raise TimeControlMismatchError(
            f"game has TimeControl {game.time_control!r} "
            f"({game.base_secs}s base) but the analysis is configured for "
            f"{expected_secs:g}s. Pin one clock -- mixing them muddies every "
            f"timing feature. Pass --tc {int(game.base_secs)}+0 to analyse "
            f"this corpus, or --allow-tc-mismatch to skip off-control games."
        )
    return False


def _parse_time_control(tc: str) -> tuple[Optional[int], Optional[int]]:
    if not tc or tc in ("-", "?"):
        return None, None
    m = _TC_RE.match(tc.strip())
    if not m:
        return None, None
    base = int(m.group(1))
    inc = int(m.group(2)) if m.group(2) else 0
    return base, inc


def _to_int(v: Optional[str]) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_game(game: chess.pgn.Game) -> Optional[GameRecord]:
    """Convert a python-chess Game into a GameRecord, or None if unusable."""
    headers = game.headers
    white = headers.get("White", "?")
    black = headers.get("Black", "?")
    tc = headers.get("TimeControl", "-")
    base, inc = _parse_time_control(tc)

    board = game.board()
    if board.fen() != chess.STARTING_FEN:
        # Skip non-standard starting positions (variants / setups).
        return None

    # Track each side's previous clock reading to derive emt.
    prev_clock = {chess.WHITE: float(base) if base is not None else None,
                  chess.BLACK: float(base) if base is not None else None}

    records: list[MoveRecord] = []
    node = game
    ply = 0
    while node.variations:
        node = node.variation(0)
        mover = board.turn
        san = board.san(node.move)
        fen_before = board.fen()

        clock_after = node.clock()  # seconds remaining after this move, or None
        emt = node.emt()            # explicit emt tag, if any
        clock_before = prev_clock[mover]  # includes any increment from the previous move

        kind_match = re.search(r"%kind ([^\]\s]+)", node.comment or "")
        kind = kind_match.group(1) if kind_match else None

        if emt is None and clock_after is not None and prev_clock[mover] is not None:
            # emt = time_before - time_after + increment
            emt = prev_clock[mover] - clock_after + (inc or 0)
            if emt < 0:
                emt = None  # clock anomaly (e.g. berserk / correspondence); drop
        if clock_after is not None:
            prev_clock[mover] = clock_after

        records.append(MoveRecord(
            ply=ply,
            move_number=board.fullmove_number,
            fen_before=fen_before,
            move_uci=node.move.uci(),
            san=san,
            mover=mover,
            mover_name=white if mover == chess.WHITE else black,
            emt=emt,
            clock_before=clock_before,
            clock_after=clock_after,
            kind=kind,
        ))
        board.push(node.move)
        ply += 1

    if not records:
        return None

    return GameRecord(
        white=white,
        black=black,
        white_elo=_to_int(headers.get("WhiteElo")),
        black_elo=_to_int(headers.get("BlackElo")),
        time_control=tc,
        base_secs=base,
        increment=inc,
        result=headers.get("Result", "*"),
        moves=records,
    )


def iter_games(pgn_path: str, max_games: Optional[int] = None) -> Iterator[GameRecord]:
    """Stream GameRecords from a PGN file (handles multi-game files/dumps)."""
    count = 0
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            if max_games is not None and count >= max_games:
                break
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            rec = parse_game(game)
            if rec is None:
                continue
            yield rec
            count += 1
