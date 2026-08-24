"""Guards the per-move clock budget and the reflective-pacing gate.

Why this exists rather than leaning on the parity harness: the harness
scenarios each decide one move from a fixed position with a full clock, so
they never reach either branch. The cap only bites on the extreme tail of the
mood long-think draws, and the reflective blend needs four logged opponent
clock times -- neither is reproducible from a single snapshot, so both need
pinning directly.

The behaviour being pinned: one move can overspend its even share of the
remaining clock by CLOCK_BUDGET_MAX_MULTIPLE and no more, and the blend
towards a slow opponent's tempo may always speed us up but may only slow us
down while we are still inside that share. Between them they are what stops a
3+0 game front-loading its whole clock into the first fifteen moves (logged
2026-08-24: 150s of 180s spent over 18 moves, worst single move 38.8s).
"""
import types
import unittest

import chess
import numpy as np

from common.constants import (
    CLOCK_BUDGET_MAX_MULTIPLE, CLOCK_BUDGET_MIN_MOVES_LEFT,
    CLOCK_BUDGET_TOTAL_MOVES,
)
from engine_components.decision_logic import (
    get_time_taken, move_time_budget, move_time_cap,
)

# A quiet middlegame position: the phase matters, because the opening
# envelope compresses base time to well under a second and no cap could ever
# bind there.
MIDGAME_FEN = "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 10"


def _board_at(fullmove_number):
    board = chess.Board(MIDGAME_FEN)
    board.fullmove_number = fullmove_number
    return board


def _engine(own_time, opp_times=None, fullmove_number=1, initial_time=180,
            mood="confident", pace_sf=None):
    return types.SimpleNamespace(
        log="",
        quickness=1.8,
        current_board=_board_at(fullmove_number),
        input_info={
            "self_initial_time": initial_time,
            "opp_initial_time": initial_time,
            "self_clock_times": [own_time],
            "opp_clock_times": opp_times if opp_times is not None else [own_time],
            "side": chess.WHITE,
            "opp_rating": None,
            "self_rating": None,
        },
        sharpness=0.15,
        ambiguity=None,
        ambiguity_forced_snap_delta=0.2,
        ambiguity_messy_snap_delta=0.2,
        game_snap_gate=0.65,
        game_pace_sf=pace_sf,
        opponent_just_blundered=False,
        just_blundered=False,
        mood=mood,
    )


class TestMoveTimeBudget(unittest.TestCase):
    def test_budget_is_the_even_share_of_the_remaining_clock(self):
        engine = _engine(own_time=180, fullmove_number=10)
        expected = 180 / (CLOCK_BUDGET_TOTAL_MOVES - 10)
        self.assertAlmostEqual(move_time_budget(engine), expected)

    def test_budget_shrinks_with_the_clock(self):
        early = move_time_budget(_engine(own_time=180, fullmove_number=15))
        late = move_time_budget(_engine(own_time=40, fullmove_number=15))
        self.assertLess(late, early)

    def test_budget_does_not_collapse_past_the_assumed_game_length(self):
        # A game running well past CLOCK_BUDGET_TOTAL_MOVES must not divide
        # the clock by zero (or by a negative move count).
        engine = _engine(own_time=30, fullmove_number=120)
        self.assertAlmostEqual(move_time_budget(engine),
                               30 / CLOCK_BUDGET_MIN_MOVES_LEFT)

    def test_cap_is_the_budget_multiple_when_the_clock_is_healthy(self):
        engine = _engine(own_time=170, fullmove_number=5)
        self.assertAlmostEqual(
            move_time_cap(engine),
            move_time_budget(engine) * CLOCK_BUDGET_MAX_MULTIPLE)
        # And that is far tighter than the old absolute clamp, which is the
        # whole point: 170 * 0.7 + 1 permits a 120-second move.
        self.assertLess(move_time_cap(engine), 170 * 0.7 + 1)

    def test_cap_never_exceeds_the_absolute_clamp(self):
        # The budget is the tighter of the two everywhere under the current
        # constants (budget * 4 <= own_time * 4/15 < own_time * 0.7 + 1), so
        # the absolute clamp is a backstop rather than the live rule. Pin the
        # ordering so it stays a backstop if the constants are ever retuned.
        for own_time in (2, 10, 45, 170):
            engine = _engine(own_time=own_time, fullmove_number=40)
            self.assertLessEqual(move_time_cap(engine), own_time * 0.7 + 1)

    def test_cap_still_shrinks_to_almost_nothing_in_a_scramble(self):
        engine = _engine(own_time=2, fullmove_number=40)
        self.assertLess(move_time_cap(engine), 1.0)


class TestCapAppliedToDecidedTime(unittest.TestCase):
    def test_a_long_think_is_capped_at_the_budget_multiple(self):
        # cautious's long-think branch on a full 3+0 clock produced the
        # 30-40 second moves seen in the logs.
        cap = None
        for seed in range(60):
            np.random.seed(seed)
            engine = _engine(own_time=170, fullmove_number=5, mood="cautious")
            cap = move_time_cap(engine)
            self.assertLessEqual(get_time_taken(engine), cap + 1e-9)
        self.assertIsNotNone(cap)

    def test_cap_binds_at_least_sometimes(self):
        # A guard that never fires is not a guard: at least one of these
        # draws must actually be trimmed, or the constants have drifted.
        trimmed = 0
        for seed in range(60):
            np.random.seed(seed)
            engine = _engine(own_time=170, fullmove_number=5, mood="cautious")
            cap = move_time_cap(engine)
            if abs(get_time_taken(engine) - cap) < 1e-9:
                trimmed += 1
        self.assertGreater(trimmed, 0)


class TestReflectivePacingGate(unittest.TestCase):
    """The blend towards the opponent's tempo is one-way once past budget."""

    def _decided(self, seed, opp_times, own_time, fullmove_number, mood):
        np.random.seed(seed)
        engine = _engine(own_time=own_time, opp_times=opp_times,
                         fullmove_number=fullmove_number, mood=mood)
        return get_time_taken(engine), engine.log

    def test_a_slow_opponent_cannot_stretch_a_move_past_budget(self):
        # The blend is the last draw get_time_taken makes, and it only runs
        # with four or more logged opponent clock times. So the same seed with
        # three of them yields exactly the pre-blend time, which is the
        # counterfactual to compare against.
        budget = move_time_budget(_engine(own_time=60, fullmove_number=25))
        slow = [180, 160, 140, 120]     # opponent burning 20s a move
        checked = 0
        for seed in range(200):
            before, _ = self._decided(seed, slow[1:], own_time=60,
                                      fullmove_number=25, mood="cautious")
            after, _ = self._decided(seed, slow, own_time=60,
                                     fullmove_number=25, mood="cautious")
            self.assertGreater(before, budget)   # the regime under test
            checked += 1
            self.assertLessEqual(after, before + 1e-9)
        self.assertGreater(checked, 0)

    def test_a_slow_opponent_may_still_stretch_a_move_inside_budget(self):
        # The mirror case, and the reason the gate is a budget test rather
        # than a blanket ban: a cheap move on a healthy clock is still allowed
        # to drift up towards a slower opponent's tempo.
        budget = move_time_budget(_engine(own_time=170, fullmove_number=5))
        slow = [180, 160, 140, 120]
        raised = 0
        for seed in range(200):
            before, _ = self._decided(seed, slow[1:], own_time=170,
                                      fullmove_number=5, mood="cocky")
            after, _ = self._decided(seed, slow, own_time=170,
                                     fullmove_number=5, mood="cocky")
            if before <= budget and after > before + 1e-9:
                raised += 1
        self.assertGreater(raised, 0)

    def test_a_fast_opponent_can_still_speed_us_up(self):
        # Opponent moving in ~0.2s against one spending ~4s: the blend pulls
        # our time down towards the faster tempo, and that direction is never
        # gated. Same seeds both arms, so the only difference is the blend
        # target.
        fast = [self._decided(seed, [180, 179.6, 179.2, 178.8], own_time=40,
                              fullmove_number=25, mood="cautious")[0]
                for seed in range(200)]
        steady = [self._decided(seed, [180, 172, 164, 156], own_time=40,
                                fullmove_number=25, mood="cautious")[0]
                  for seed in range(200)]
        self.assertLess(np.mean(fast), np.mean(steady))
