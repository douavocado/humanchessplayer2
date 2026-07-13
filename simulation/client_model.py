"""Emulates the live client's per-move decision structure without vision or
mouse hardware.

Mirrors clients/mp_original.py's run_game branches in order:

  1. queued premove fires (if legal after the opponent's actual move)
  2. ponder-dic fast paths (exact hit / safe-premove-by-chance / <10s scramble)
  3. full engine path: update_info -> _decide_resign -> make_move

Each branch returns the move plus the seconds a live Lichess game would have
charged to our clock for it. Wait formulas are the exact functions the live
client calls (common/move_timing.py); latency components come from
LatencyModel.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import chess

from common.move_timing import (
    MOVE_DELAY,
    ponder_response_wait,
    resign_pause,
    scramble_response_wait,
)
from common.utils import check_safe_premove, scramble_fire_veto
from common.board_information import phase_of_game

from .latency_model import LatencyModel

# A fired premove registers essentially instantly; Lichess charges on the
# order of a tenth of a second (integer-second PGN clocks usually show 0).
PREMOVE_FIRE_SECS = 0.1


@dataclass
class MoveDecision:
    move_uci: Optional[str]     # None => resign
    charged_secs: float
    kind: str                   # premove | ponder_hit | safe_ponder |
                                # scramble | engine | resign


class SimClient:
    """One side of a simulated game, wrapping a headless Engine."""

    def __init__(self, engine, side: bool, latency: LatencyModel,
                 rng: random.Random, quickness: float,
                 initial_time: float, rating: int, opp_rating: int):
        self.engine = engine
        self.side = side
        self.latency = latency
        self.rng = rng
        self.quickness = quickness
        self.initial_time = float(initial_time)
        self.rating = rating
        self.opp_rating = opp_rating
        self.ponder_dic: dict = {}
        self.queued_premove: Optional[str] = None
        self.cursor_square: Optional[int] = None

    def new_game(self):
        self.ponder_dic = {}
        self.queued_premove = None
        self.cursor_square = None

    # ------------------------------------------------------------- helpers
    def _gesture(self, move_uci: str, own_time: float) -> float:
        move = chess.Move.from_uci(move_uci)
        secs, _ = self.latency.gesture_time(move, own_time, self.cursor_square)
        self.cursor_square = move.to_square
        return secs

    def _try_ponder_fast_paths(self, board: chess.Board, fens: list[str],
                               own_time: float) -> Optional[MoveDecision]:
        """The PONDER_DIC branches of run_game, in the live client's order."""
        board_fen = board.board_fen()
        if own_time > 10:
            entry = self.ponder_dic.get(board_fen)
            if entry is not None:
                wait = ponder_response_wait(self.initial_time, self.quickness,
                                            self.rng,
                                            pace_sf=self.engine.ponder_pace_sf)
                self.queued_premove = entry.get("premove")
                return MoveDecision(entry["move"],
                                    wait + self._gesture(entry["move"], own_time),
                                    "ponder_hit")
            if len(fens) >= 2 and self.ponder_dic:
                last_uci = list(self.ponder_dic.values())[-1]["move"]
                move_obj = chess.Move.from_uci(last_uci)
                dummy = board.copy()
                dummy.turn = self.side
                if move_obj in dummy.legal_moves:
                    last_board = chess.Board(fens[-2])
                    if check_safe_premove(last_board, last_uci):
                        prob = math.sqrt(1 / self.initial_time)
                        if self.initial_time < 200 and self.rng.random() < prob:
                            wait = ponder_response_wait(self.initial_time,
                                                        self.quickness, self.rng,
                                                        pace_sf=self.engine.ponder_pace_sf)
                            return MoveDecision(
                                last_uci,
                                wait + self._gesture(last_uci, own_time),
                                "safe_ponder")
        elif len(fens) >= 2 and self.ponder_dic:
            last_board = chess.Board(fens[-2])
            dummy = board.copy()
            dummy.turn = self.side
            # Fire eagerness and hang-blindness both follow this game's
            # character: a snappy game fires more stale moves, a low-skill
            # game doesn't check where they land (skill-gated veto -- an
            # unconditional veto collapsed the blunder tail to 0.37x human).
            prob = (30 - own_time) / 50 * (self.engine.game_scramble_fire_sf or 1.0)
            veto_p = self.engine.scramble_veto_p
            candidates = [chess.Move.from_uci(e["move"])
                          for e in list(self.ponder_dic.values())[-10:]
                          if chess.Move.from_uci(e["move"]) in dummy.legal_moves]
            for move_obj in candidates:
                if check_safe_premove(last_board, move_obj.uci()) or \
                        (self.rng.random() < prob
                         and not (self.rng.random() < veto_p
                                  and scramble_fire_veto(dummy, move_obj.uci()))):
                    wait = scramble_response_wait(self.rng)
                    return MoveDecision(
                        move_obj.uci(),
                        wait + self._gesture(move_obj.uci(), own_time),
                        "scramble")
        return None

    # ---------------------------------------------------------------- main
    def decide(self, board: chess.Board, fens: list[str],
               last_moves: list[str], self_clocks: list[float],
               opp_clocks: list[float]) -> MoveDecision:
        """Choose this side's move for the current position and the seconds
        a live game would charge for it."""
        own_time = self_clocks[-1]

        # 1. A queued premove fires the instant the opponent's move lands.
        if self.queued_premove is not None:
            premove = self.queued_premove
            self.queued_premove = None
            if chess.Move.from_uci(premove) in board.legal_moves:
                self.cursor_square = chess.Move.from_uci(premove).to_square
                return MoveDecision(premove, PREMOVE_FIRE_SECS, "premove")
            # Illegal after the opponent's actual move: Lichess cancels it.

        # Detection + server round-trip apply to every non-premove branch.
        overhead = (self.latency.detection_latency(own_time)
                    + self.latency.network_lag())

        # 2. Ponder-dic fast paths (no engine call at all).
        fast = self._try_ponder_fast_paths(board, fens, own_time)
        if fast is not None:
            fast.charged_secs += overhead
            return fast

        # 3. Full engine path.
        info = {
            "side": self.side,
            "fens": fens[-5:],
            "last_moves": last_moves[-4:],
            "self_clock_times": self_clocks[-8:],
            "opp_clock_times": opp_clocks[-8:],
            "self_initial_time": self.initial_time,
            "opp_initial_time": self.initial_time,
            "self_rating": self.rating,
            "opp_rating": self.opp_rating,
        }
        self.engine.update_info(info)

        if self.engine._decide_resign():
            return MoveDecision(None, overhead + resign_pause(self.rng),
                                "resign")

        out = self.engine.make_move(log=False,
                                    seed=self.rng.randrange(1_000_000))
        move_uci = out["move_made"]
        if chess.Move.from_uci(move_uci) not in board.legal_moves:
            raise RuntimeError(
                f"Engine returned illegal move {move_uci} in {board.fen()}")

        if out.get("ponder_dic"):
            self.ponder_dic.update(out["ponder_dic"])
        self.queued_premove = out.get("premove")

        # Realised think: the client sleeps up to (time_take - MOVE_DELAY),
        # but real compute overruns that floor on heavy positions.
        compute = self.latency.compute_time(phase_of_game(board))
        think = max(float(out["time_take"]) - MOVE_DELAY, compute)

        gesture = self._gesture(move_uci, own_time)
        return MoveDecision(move_uci, overhead + think + gesture, "engine")
