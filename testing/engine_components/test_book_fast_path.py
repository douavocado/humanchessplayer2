"""Tests for the opening-book fast path and the per-instance ponder knobs.

Both are levers on the *compute-bypass* channel, the only way the bot can
produce a sub-1s move: Phase A showed the engine's requested think time in the
opening already sits below the per-move compute floor, so no pacing knob moves
the instant rate (five arms varying eval_noise_scale and quickness held it
within 0.0087). See docs/superpowers/specs/2026-07-28-instant-move-channel-
design.md.

The book fast path consults the opening book *before* calculate_analytics
rather than after, so a memorised move stops paying for a full-width multipv
scan plus an uncapped depth-12 sharpness scan. It changes behaviour by
construction, so the load-bearing property here is that it is **off by
default** -- with the flag off nothing about update_info/make_move may differ,
which is what lets the engine parity harness stay green without --record.
"""
import unittest

import chess

from common.constants import OPENING_BOOK_FAST_PATH
from common.search_constants import PONDER_TIME_PER_POSITION
from engine import Engine


def _info(board, clock=45.0):
    return {"side": board.turn, "fens": [board.fen()], "last_moves": [],
            "self_clock_times": [clock], "opp_clock_times": [clock],
            "self_initial_time": 60.0, "opp_initial_time": 60.0}


class TestDefaultsAreInert(unittest.TestCase):

    def test_fast_path_ships_off(self):
        """Parity depends on this: the lever must land disabled."""
        self.assertFalse(OPENING_BOOK_FAST_PATH)


class TestBookFastPath(unittest.TestCase):
    """Needs real Stockfish + weights, like the parity harness."""

    @classmethod
    def setUpClass(cls):
        cls.off = Engine(log_file=None)
        cls.on = Engine(log_file=None, opening_book_fast_path=True)

    @classmethod
    def tearDownClass(cls):
        cls.off.close_engines()
        cls.on.close_engines()

    def test_disabled_engine_never_arms_the_fast_path(self):
        eng = self.off
        eng.update_info(_info(chess.Board()))
        self.assertIsNone(eng.book_fast_move)
        # ... and analytics still ran, exactly as today
        self.assertTrue(eng.analytics_updated)

    def test_enabled_engine_arms_on_a_book_hit_and_skips_analytics(self):
        """The whole point: a book hit must not pay for the scans."""
        eng = self.on
        eng.update_info(_info(chess.Board()))
        self.assertIsNotNone(eng.book_fast_move)
        self.assertFalse(eng.analytics_updated)

    def test_make_move_returns_the_book_move_and_flags_it(self):
        eng = self.on
        board = chess.Board()
        eng.update_info(_info(board))
        out = eng.make_move(log=False, seed=1234)
        self.assertTrue(out.get("book_fast"))
        self.assertIn(chess.Move.from_uci(out["move_made"]), board.legal_moves)
        self.assertGreater(out["time_take"], 0.0)

    def test_the_arming_flag_does_not_leak_into_the_next_move(self):
        """A book opening followed by an out-of-book position must not
        replay the stale book move -- update_info has to clear it."""
        eng = self.on
        eng.update_info(_info(chess.Board()))
        self.assertIsNotNone(eng.book_fast_move)
        # a position no opening book covers
        odd = chess.Board("4k3/8/8/3q4/8/8/4P3/4K3 w - - 0 30")
        eng.update_info(_info(odd))
        self.assertIsNone(eng.book_fast_move)
        self.assertTrue(eng.analytics_updated)

    def test_non_opening_positions_take_the_normal_path(self):
        eng = self.on
        mid = chess.Board(
            "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/R2QKB1R w KQ - 0 9")
        eng.update_info(_info(mid))
        self.assertIsNone(eng.book_fast_move)
        self.assertTrue(eng.analytics_updated)


class TestBookDerivedPremove(unittest.TestCase):
    """The fast path returns before the normal premove/ponder preparation, so
    without this it stops refilling the very channels it competes with -- a
    6-game smoke test showed premove 14.0% -> 5.8% and ponder_hit 16.7% ->
    13.2%. Queuing our book reply to the book's own predicted opponent reply
    recovers the premove half for two polyglot lookups and no search."""

    @classmethod
    def setUpClass(cls):
        cls.eng = Engine(log_file=None, opening_book_fast_path=True)

    @classmethod
    def tearDownClass(cls):
        cls.eng.close_engines()

    def test_start_position_yields_a_book_premove(self):
        from engine_components.opening_book import book_premove
        board = chess.Board()
        our_move = self.eng.book_fast_move  # not armed yet
        self.assertIsNone(our_move)
        self.eng.update_info(_info(board))
        after = board.copy()
        after.push(chess.Move.from_uci(self.eng.book_fast_move))
        pm = book_premove(self.eng, after)
        self.assertIsNotNone(pm)
        # legal in the position it would actually be fired from
        probe = after.copy()
        probe.push(chess.Move.from_uci(pm[1]))
        self.assertIn(chess.Move.from_uci(pm[0]), probe.legal_moves)

    def test_out_of_book_position_yields_none(self):
        from engine_components.opening_book import book_premove
        odd = chess.Board("4k3/8/8/3q4/8/8/4P3/4K3 b - - 0 30")
        self.assertIsNone(book_premove(self.eng, odd))

    def test_make_move_attaches_the_premove(self):
        eng = self.eng
        eng.update_info(_info(chess.Board()))
        out = eng.make_move(log=False, seed=77)
        self.assertTrue(out.get("book_fast"))
        self.assertIn("premove", out)


class TestPonderKnobs(unittest.TestCase):

    def test_defaults_resolve_to_the_module_constants(self):
        eng = Engine(log_file=None)
        try:
            self.assertEqual(eng.ponder_time_per_position,
                             PONDER_TIME_PER_POSITION)
        finally:
            eng.close_engines()

    def test_overrides_win(self):
        eng = Engine(log_file=None, ponder_time_per_position=0.03,
                     game_ponder_width_base=4.0)
        try:
            self.assertEqual(eng.ponder_time_per_position, 0.03)
            self.assertEqual(eng.game_ponder_width_base, 4.0)
        finally:
            eng.close_engines()

    def test_width_base_shifts_the_sampled_ponder_width(self):
        """The width draw must actually read the per-instance base, or the
        sweep arm that raises it is a no-op."""
        lo = Engine(log_file=None, game_ponder_width_base=0.0)
        hi = Engine(log_file=None, game_ponder_width_base=5.0)
        try:
            lo_w = [(lo._sample_game_character(), lo.game_ponder_width)[1]
                    for _ in range(30)]
            hi_w = [(hi._sample_game_character(), hi.game_ponder_width)[1]
                    for _ in range(30)]
            self.assertLess(sum(lo_w) / len(lo_w), sum(hi_w) / len(hi_w))
        finally:
            lo.close_engines()
            hi.close_engines()


if __name__ == "__main__":
    unittest.main()
