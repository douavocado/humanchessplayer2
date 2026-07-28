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

from .adjudicate_result import CLOCK_THRESHOLD_FRACTION, adjudicate_position
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
    termination: str            # checkmate | resignation | timeout | draw | max-plies | adjudicated
    seed: int
    initial_time: float
    increment: float
    moves: list[SimMove] = field(default_factory=list)
    # Who actually sat on each side this game (set by GameRunner; supports
    # per-game colour alternation).
    white_name: str = "SimBotWhite"
    black_name: str = "SimBotBlack"
    white_elo: int = 2450
    black_elo: int = 2450
    # Set only when termination == "adjudicated": the Stockfish eval (White's
    # pov, centipawns) the result was decided from. None otherwise.
    adjudicated_cp: Optional[float] = None


@dataclass
class BotSpec:
    """One bot's persona: everything that can differ between the two sides."""
    name: str = "SimBot"
    rating: int = 2450
    difficulty: int = None           # engine playing_level; None -> DIFFICULTY
    quickness: float = None          # move-time pacing; None -> QUICKNESS
    mouse_quickness: float = None    # gesture speed; None -> MOUSE_QUICKNESS
    eval_noise_scale: float = None   # move-selection noise; None -> HUMAN_EVAL_NOISE_SCALE
    moderate_sharpness_breadth_bonus: int = None  # None -> MODERATE_SHARPNESS_BREADTH_BONUS
    midgame_premove_veto_p: float = None  # None -> MIDGAME_PREMOVE_VETO_P
    opening_breadth_strength_bonus: int = None  # None -> OPENING_BREADTH_STRENGTH_BONUS
    midgame_breadth_strength_bonus: int = None  # None -> MIDGAME_BREADTH_STRENGTH_BONUS
    ambiguity_forced_snap_delta: float = None  # None -> AMBIGUITY_FORCED_SNAP_DELTA
    ambiguity_messy_snap_delta: float = None   # None -> AMBIGUITY_MESSY_SNAP_DELTA
    opening_book_fast_path: bool = None        # None -> OPENING_BOOK_FAST_PATH
    ponder_time_per_position: float = None     # None -> PONDER_TIME_PER_POSITION
    game_ponder_width_base: float = None       # None -> GAME_PONDER_WIDTH_BASE
    target_rating: int = None                  # None -> no dial; see common/strength_profiles.py
    reeval_order: str = None                   # None -> REEVAL_ORDER

    def __post_init__(self):
        from common.constants import (DIFFICULTY, MOUSE_QUICKNESS, QUICKNESS,
                                      HUMAN_EVAL_NOISE_SCALE,
                                      AMBIGUITY_FORCED_SNAP_DELTA,
                                      AMBIGUITY_MESSY_SNAP_DELTA,
                                      OPENING_BOOK_FAST_PATH,
                                      GAME_PONDER_WIDTH_BASE)
        from common.search_constants import (MODERATE_SHARPNESS_BREADTH_BONUS,
                                             MIDGAME_PREMOVE_VETO_P,
                                             OPENING_BREADTH_STRENGTH_BONUS,
                                             MIDGAME_BREADTH_STRENGTH_BONUS,
                                             PONDER_TIME_PER_POSITION,
                                             REEVAL_ORDER)
        if self.difficulty is None:
            self.difficulty = DIFFICULTY
        if self.quickness is None:
            self.quickness = QUICKNESS
        if self.mouse_quickness is None:
            self.mouse_quickness = MOUSE_QUICKNESS
        if self.eval_noise_scale is None:
            self.eval_noise_scale = HUMAN_EVAL_NOISE_SCALE
        if self.moderate_sharpness_breadth_bonus is None:
            self.moderate_sharpness_breadth_bonus = MODERATE_SHARPNESS_BREADTH_BONUS
        if self.midgame_premove_veto_p is None:
            self.midgame_premove_veto_p = MIDGAME_PREMOVE_VETO_P
        if self.opening_breadth_strength_bonus is None:
            self.opening_breadth_strength_bonus = OPENING_BREADTH_STRENGTH_BONUS
        if self.midgame_breadth_strength_bonus is None:
            self.midgame_breadth_strength_bonus = MIDGAME_BREADTH_STRENGTH_BONUS
        if self.ambiguity_forced_snap_delta is None:
            self.ambiguity_forced_snap_delta = AMBIGUITY_FORCED_SNAP_DELTA
        if self.ambiguity_messy_snap_delta is None:
            self.ambiguity_messy_snap_delta = AMBIGUITY_MESSY_SNAP_DELTA
        if self.opening_book_fast_path is None:
            self.opening_book_fast_path = OPENING_BOOK_FAST_PATH
        if self.ponder_time_per_position is None:
            self.ponder_time_per_position = PONDER_TIME_PER_POSITION
        if self.game_ponder_width_base is None:
            self.game_ponder_width_base = GAME_PONDER_WIDTH_BASE
        if self.reeval_order is None:
            self.reeval_order = REEVAL_ORDER


@dataclass
class SimConfig:
    initial_time: float = 60.0
    increment: float = 0.0
    resolution_scale: float = None   # None -> common.constants.RESOLUTION_SCALE
    bot_a: BotSpec = field(default_factory=BotSpec)
    bot_b: BotSpec = field(default_factory=BotSpec)
    max_plies: int = 400
    # Default (False): stop the game and adjudicate from a Stockfish eval as
    # soon as either side's clock first drops below CLOCK_THRESHOLD_FRACTION
    # of initial_time (simulation/adjudicate_result.py), instead of playing
    # to a real conclusion. Most bullet games otherwise end on the clock
    # (56-62% "timeout" across every run generated pre-this-change), which
    # conflates move quality with clock-race variance and dilutes the Elo
    # signal from anything that only changes move quality -- see
    # cheat_detection/runs/adjudication/. --simulate-full (simulation/run.py)
    # restores the old full-length behaviour.
    simulate_full: bool = False

    def __post_init__(self):
        from common.constants import RESOLUTION_SCALE
        if self.resolution_scale is None:
            self.resolution_scale = RESOLUTION_SCALE


class GameRunner:
    """Plays simulated games; reuses the two (expensive) Engine instances
    across games, resetting per-game client state each time.

    ``engine_a``/``engine_b`` belong to ``cfg.bot_a``/``cfg.bot_b`` (they were
    built with that bot's difficulty and quickness); ``a_plays_white`` maps the
    bots onto the board per game, so colours can alternate without rebuilding
    engines.
    """

    def __init__(self, engine_a, engine_b, cfg: SimConfig):
        self.cfg = cfg
        self.engine_a = engine_a
        self.engine_b = engine_b

    def play_game(self, seed: int, a_plays_white: bool = True,
                  on_progress: Optional[callable] = None) -> SimGame:
        cfg = self.cfg
        rng = random.Random(seed)
        bots = {chess.WHITE: cfg.bot_a if a_plays_white else cfg.bot_b,
                chess.BLACK: cfg.bot_b if a_plays_white else cfg.bot_a}
        engines = {chess.WHITE: self.engine_a if a_plays_white else self.engine_b,
                   chess.BLACK: self.engine_b if a_plays_white else self.engine_a}
        clients = {
            side: SimClient(
                engines[side], side,
                LatencyModel(rng, bots[side].mouse_quickness, cfg.resolution_scale),
                rng, bots[side].quickness,
                cfg.initial_time,
                rating=bots[side].rating,
                opp_rating=bots[not side].rating,
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
                       initial_time=cfg.initial_time, increment=cfg.increment,
                       white_name=bots[chess.WHITE].name,
                       black_name=bots[chess.BLACK].name,
                       white_elo=bots[chess.WHITE].rating,
                       black_elo=bots[chess.BLACK].rating)

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
            if not cfg.simulate_full:
                threshold = CLOCK_THRESHOLD_FRACTION * cfg.initial_time
                if clocks[chess.WHITE][-1] < threshold or clocks[chess.BLACK][-1] < threshold:
                    game.result, game.adjudicated_cp = adjudicate_position(board, self.engine_a.stockfish_engine)
                    game.termination = "adjudicated"
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
