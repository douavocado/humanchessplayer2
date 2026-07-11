"""Runs one simulated bot-vs-bot game on a SimClock.

Two SimClients (each wrapping a headless Engine) alternate; every move
charges the mover's clock with the modelled live-game duration. No real
time passes beyond the engines' own computation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import chess

from .client_model import SimClient
from .clock import SimClock
from .latency_model import LatencyModel


@dataclass
class SimMove:
    ply: int
    move_uci: str
    san: str
    mover: bool
    charged_secs: float
    clock_after: float          # mover's remaining time after the move
    kind: str


@dataclass
class SimGame:
    result: str                 # "1-0" / "0-1" / "1/2-1/2"
    termination: str            # checkmate | resignation | timeout | draw | max-plies
    seed: int
    initial_time: float
    increment: float
    moves: list[SimMove] = field(default_factory=list)


@dataclass
class SimConfig:
    initial_time: float = 60.0
    increment: float = 0.0
    quickness: float = None          # None -> common.constants.QUICKNESS
    mouse_quickness: float = None    # None -> common.constants.MOUSE_QUICKNESS
    resolution_scale: float = None   # None -> common.constants.RESOLUTION_SCALE
    white_rating: int = 2450
    black_rating: int = 2450
    max_plies: int = 400

    def __post_init__(self):
        from common.constants import (QUICKNESS, MOUSE_QUICKNESS,
                                      RESOLUTION_SCALE)
        if self.quickness is None:
            self.quickness = QUICKNESS
        if self.mouse_quickness is None:
            self.mouse_quickness = MOUSE_QUICKNESS
        if self.resolution_scale is None:
            self.resolution_scale = RESOLUTION_SCALE


class GameRunner:
    """Plays simulated games; reuses the two (expensive) Engine instances
    across games, resetting per-game client state each time."""

    def __init__(self, white_engine, black_engine, cfg: SimConfig):
        self.cfg = cfg
        self.engines = {chess.WHITE: white_engine, chess.BLACK: black_engine}

    def play_game(self, seed: int,
                  on_progress: Optional[callable] = None) -> SimGame:
        cfg = self.cfg
        rng = random.Random(seed)
        latency = LatencyModel(rng, cfg.mouse_quickness, cfg.resolution_scale)
        clients = {
            side: SimClient(
                self.engines[side], side, latency, rng, cfg.quickness,
                cfg.initial_time,
                rating=cfg.white_rating if side == chess.WHITE else cfg.black_rating,
                opp_rating=cfg.black_rating if side == chess.WHITE else cfg.white_rating,
            )
            for side in (chess.WHITE, chess.BLACK)
        }
        for c in clients.values():
            c.new_game()

        clock = SimClock(cfg.initial_time, cfg.increment)
        board = chess.Board()
        fens = [board.fen()]
        last_moves: list[str] = []
        clocks = {chess.WHITE: [cfg.initial_time], chess.BLACK: [cfg.initial_time]}
        game = SimGame(result="*", termination="", seed=seed,
                       initial_time=cfg.initial_time, increment=cfg.increment)

        while True:
            if board.is_game_over(claim_draw=True):
                game.result = board.result(claim_draw=True)
                game.termination = ("checkmate" if board.is_checkmate()
                                    else "draw")
                break
            if len(game.moves) >= cfg.max_plies:
                game.result = "1/2-1/2"
                game.termination = "max-plies"
                break

            side = board.turn
            decision = clients[side].decide(
                board, fens, last_moves,
                self_clocks=clocks[side], opp_clocks=clocks[not side])

            if decision.move_uci is None:  # resignation
                clock.charge(side, decision.charged_secs)
                game.result = "0-1" if side == chess.WHITE else "1-0"
                game.termination = "resignation"
                break

            flagged = clock.charge(side, decision.charged_secs)
            if flagged:
                game.result = "0-1" if side == chess.WHITE else "1-0"
                game.termination = "timeout"
                break

            move = chess.Move.from_uci(decision.move_uci)
            san = board.san(move)
            board.push(move)
            fens.append(board.fen())
            last_moves.append(decision.move_uci)
            clocks[side].append(clock.time_left(side))
            game.moves.append(SimMove(
                ply=len(game.moves), move_uci=decision.move_uci, san=san,
                mover=side, charged_secs=decision.charged_secs,
                clock_after=clock.time_left(side), kind=decision.kind))
            if on_progress:
                on_progress(len(game.moves))

        return game
