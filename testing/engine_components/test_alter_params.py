"""Guards the parameter plumbing between AlterMoveProbNN's two forward passes.

`forward` (torch, what train_model.py and evaluate_model.py drive) reads the
nn.Parameters directly. `forward_numpy` (what human_move_logic.py:135 runs in
the live engine) reads floats out of `self.params_dict`, populated by
`load_params_dict()` from `get_parameters_dict()`.

Every read in forward_numpy is `self.params_dict.get(name, <default>)`, so a
parameter missing from that dict is not an error -- it silently falls back to
the *untrained* default and the engine plays differently from the model that
was fitted. That is exactly what happened: the hand-written dict listed 16 of
19 parameters, and from the day the numpy path was introduced the live engine
ran with the interesting-move threshold 3.06x too high (exp(0.0) vs
exp(-1.118)), the threat-block exponent 2.24x too strong (1.0 vs 0.446), and
the solo-threat magnifier at 1.0 instead of 1.372. Measured over 7779 held-out
2300+ bullet positions it cost 1.70pp of top-1 human-move agreement and made
the two implementations disagree on the best move in 21.5% of positions.

So these tests pin the two properties that failure violated: the dict covers
every parameter, and the two implementations actually agree.
"""
import unittest

import chess
import torch

from common.constants import ALTER_MOVE_PROB_WEIGHTS_PTH
from models.alter_move_prob_nn import AlterMoveProbNN

# A quiet middlegame position with no promotion race, no advanced passed pawn
# and no fork threat, so the three forward_numpy-only rules stay silent and the
# two implementations are supposed to be computing the same function.
# The previous position and the move out of it, so the pair is reachable by
# construction: forward's takeback/newly-threatened/repeat rules run
# patch_fens over the two fens and go silent if it cannot link them, which
# would quietly stop this test exercising the parameters those rules use.
QUIET_PREV = "r2q1rk1/pp3ppp/2nbpn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 b - - 5 9"
QUIET_LAST_MOVE = "d6e7"
QUIET_FEN = "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 6 10"


class TestFixtureIsLinkable(unittest.TestCase):
    def test_prev_position_reaches_the_test_position(self):
        board = chess.Board(QUIET_PREV)
        move = chess.Move.from_uci(QUIET_LAST_MOVE)
        self.assertIn(move, board.legal_moves)
        board.push(move)
        self.assertEqual(board.fen(), QUIET_FEN)


class TestParameterDictCoverage(unittest.TestCase):
    def setUp(self):
        self.model = AlterMoveProbNN()
        self.model.load_state_dict(torch.load(ALTER_MOVE_PROB_WEIGHTS_PTH,
                                              weights_only=True))
        self.model.eval()

    def test_every_parameter_is_exported(self):
        exported = set(self.model.get_parameters_dict())
        declared = {name for name, _ in self.model.named_parameters()}
        self.assertEqual(exported, declared)

    def test_exported_values_are_the_trained_ones(self):
        exported = self.model.get_parameters_dict()
        for name, param in self.model.named_parameters():
            self.assertAlmostEqual(exported[name], float(param), places=6,
                                   msg="{} exported wrong".format(name))

    def test_the_three_that_were_missing_are_not_at_their_defaults(self):
        # If any of these ever reads back at its forward_numpy fallback, the
        # bug is back and every other test here would still pass by accident.
        exported = self.model.get_parameters_dict()
        for name, fallback in (("solo_factor_sf", 1.0),
                               ("threatened_lvl_diff_sf", 1.0),
                               ("interesting_move_threshold", 0.0)):
            self.assertIn(name, exported)
            self.assertNotAlmostEqual(exported[name], fallback, places=3)


class TestForwardPassesAgree(unittest.TestCase):
    """The live path must compute what the trained path computes."""

    @classmethod
    def setUpClass(cls):
        cls.model = AlterMoveProbNN()
        cls.model.load_state_dict(torch.load(ALTER_MOVE_PROB_WEIGHTS_PTH,
                                             weights_only=True))
        cls.model.eval()
        cls.model.load_params_dict()

    def _both(self, fen, prev_fen=None):
        board = chess.Board(fen)
        # A plausible-looking spread rather than a flat one: the interesting-
        # move threshold is a floor, so a flat input would hide a wrong one.
        moves = [m.uci() for m in board.legal_moves]
        move_dic = {u: p for u, p in zip(moves, [0.30, 0.18, 0.12, 0.09, 0.07,
                                                 0.05, 0.04, 0.03, 0.025, 0.02])}
        for u in moves[len(move_dic):]:
            move_dic[u] = 0.001
        total = sum(move_dic.values())
        move_dic = {k: v / total for k, v in move_dic.items()}
        prev = chess.Board(prev_fen) if prev_fen else None
        with torch.no_grad():
            torch_out, _ = self.model(move_dic, board, prev, None)
            numpy_out, _ = self.model.forward_numpy(move_dic, board, prev, None)
        return ({k: float(v) for k, v in torch_out.items()},
                {k: float(v) for k, v in numpy_out.items()})

    def test_same_probabilities_in_a_quiet_position(self):
        torch_out, numpy_out = self._both(QUIET_FEN, QUIET_PREV)
        self.assertEqual(set(torch_out), set(numpy_out))
        for uci in torch_out:
            self.assertAlmostEqual(torch_out[uci], numpy_out[uci], places=5,
                                   msg="diverged on {}".format(uci))

    def test_same_best_move(self):
        torch_out, numpy_out = self._both(QUIET_FEN, QUIET_PREV)
        self.assertEqual(max(torch_out, key=torch_out.get),
                         max(numpy_out, key=numpy_out.get))

    def test_agreement_is_not_an_artefact_of_a_defaulted_dict(self):
        # Reproduce the bug and assert the tests above would have caught it.
        broken = AlterMoveProbNN()
        broken.load_state_dict(torch.load(ALTER_MOVE_PROB_WEIGHTS_PTH,
                                          weights_only=True))
        broken.eval()
        partial = broken.get_parameters_dict()
        for name in ("solo_factor_sf", "threatened_lvl_diff_sf",
                     "interesting_move_threshold"):
            partial.pop(name)
        broken.load_params_dict(partial)
        board = chess.Board(QUIET_FEN)
        moves = [m.uci() for m in board.legal_moves]
        move_dic = {u: 1.0 / len(moves) for u in moves}
        with torch.no_grad():
            good, _ = broken(move_dic, board, chess.Board(QUIET_PREV), None)
            bad, _ = broken.forward_numpy(move_dic, board, chess.Board(QUIET_PREV), None)
        worst = max(abs(float(good[u]) - float(bad[u])) for u in good)
        self.assertGreater(worst, 1e-4)
