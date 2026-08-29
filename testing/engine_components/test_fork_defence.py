"""Guards the fork-defence path: `defends_against_fork` and the boost that
consumes it (`FORK_DEFENCE_SF` in `models/alter_move_prob_nn.py`).

Why this exists rather than leaning on the parity harness: none of the four
golden-master scenarios has an opponent fork threat at all
(`opponent_fork_threat` returns None for every one), so parity stays green
whatever this code does. It is not inert -- both regressions below are real
logged games where the bot dropped material.

Two failure modes are pinned.

1. `defends_against_fork` could not see a king step out of a fork
   (logs/sessions/2026-08-22_16-01-25, move 12). Nxf7+ forked Kd8, Rh8 and
   Bh6; the NN ranked Ke7 #3 and Ke8 #7 on its own, but the defence test
   rejected both, so neither got the boost and the bot played Bg7 (-311)
   instead of Ke8 (+421). Two separate bugs did it: the "cover the fork
   square" branch compared `PIECE_VALS[defender] <= forker_value` and
   PIECE_VALS[KING] is 1000, so a king could never cover anything; and the
   "move the forked piece" branch counted how many pieces the forker still
   attacked afterwards, which is blind to the check being gone and counts
   incidental pieces in the forker's radius. The fix asks `evaluate_fork`
   the same question it used to detect the threat -- after our move, is this
   still a fork? -- rather than re-deriving the geometry a second time.

2. The boost skipped any defence the net had already ranked at or above the
   interesting-move threshold. That made it structurally impossible to
   promote a genuine defence past a higher-ranked non-defence, which is
   exactly the 2026-08-22 shape (Ke7 at 0.112 behind Bg7 at 0.225). The gate
   was added for the 2026-08-05 Nc7+ game, where several *false* defences
   sat above the threshold; with (1) fixed those are rejected on their own
   merits, so the gate is gone and every defence is boosted.

Both un-altered move distributions below are copied from the engine logs of
the games in question, so these tests exercise the real alteration path
without depending on the piece-selector nets.
"""
import unittest

import chess
import torch

from common.board_information import defends_against_fork, opponent_fork_threat
from common.constants import ALTER_MOVE_PROB_WEIGHTS_PTH
from models import alter_move_prob_nn
from models.alter_move_prob_nn import AlterMoveProbNN

# logs/sessions/2026-08-22_16-01-25, move 12. White has just played Ne5,
# threatening Nxf7+ forking Kd8, Rh8 and Bh6.
NXF7_FEN = "r1bk3r/ppp2p2/5npb/2n1N3/2P1P3/2N5/PP3PP1/R3KB1R b KQ - 0 12"
NXF7_PREV = "r1bk3r/ppp2p2/5npb/2n1p3/2P1P3/2N2N2/PP3PP1/R3KB1R w KQ - 0 12"
NXF7_PREV_PREV = "r1bk3r/ppp2p1p/5nPb/2n1p3/2P1P3/2N2N2/PP3PP1/R3KB1R b KQ - 0 11"
NXF7_NN_PROBS = {
    "Ncxe4": 0.24347, "Nfxe4": 0.14836, "Ke7": 0.08549, "Rh7": 0.07812,
    "Be6": 0.06508, "Bg7": 0.05944, "Ke8": 0.05824, "c6": 0.05462,
    "Re8": 0.03763, "Rf8": 0.03222, "b6": 0.01601, "g5": 0.01559,
    "Ne6": 0.01317, "Bf4": 0.01283, "Rg8": 0.01183, "Bf8": 0.01141,
    "Ng4": 0.01102, "a6": 0.00745, "Bd2+": 0.00617, "a5": 0.00498,
    "Nfd7": 0.00467, "Bg4": 0.00434, "Nd3+": 0.00353, "Rb8": 0.00321,
    "Bg5": 0.0026, "Nh5": 0.00226, "Bd7": 0.00174, "Ncd7": 0.00171,
    "b5": 0.00098, "Na6": 0.00049, "Bf5": 0.00045, "Ne8": 0.0004,
    "Na4": 0.00021, "Ng8": 0.00014, "Bh3": 4e-05, "Nd5": 3e-05,
    "Nh7": 3e-05, "Bc1": 2e-05, "Be3": 1e-05, "Nb3": 1e-05,
}

# logs/sessions/2026-08-05_23-04-03, move 13. White has just played Nb5,
# threatening Nc7+ forking Ke8 and Ra8. The game the threshold gate was
# originally added for.
NC7_FEN = "rn2k1nr/pp3pb1/4p1p1/1N2P2p/2P5/2N5/PP4PP/R1B1KB1R b KQkq - 1 13"
NC7_NN_PROBS = {
    "Ne7": 0.30365, "Bxe5": 0.24269, "Nc6": 0.16358, "a6": 0.13147,
    "Bh6": 0.05074, "h4": 0.0415, "g5": 0.01322, "Nh6": 0.01041,
    "Na6": 0.00864, "Kf8": 0.0077, "Bf8": 0.00632, "Ke7": 0.00601,
    "Kd8": 0.00502, "f5": 0.00315, "Nd7": 0.00181, "Rh7": 0.00176,
    "Bf6": 0.00066, "a5": 0.00056, "Nf6": 0.00043, "b6": 0.00033,
    "f6": 0.00023, "Rh6": 0.00012, "Kd7": 3e-05,
}


def _defences(fen):
    """SAN of every legal move the classifier calls a fork defence."""
    board = chess.Board(fen)
    threat = opponent_fork_threat(board)
    assert threat is not None, "no fork threat in " + fen
    return {board.san(m) for m in board.legal_moves
            if defends_against_fork(board, m.uci(), threat)}


class TestDefendsAgainstFork(unittest.TestCase):

    def test_king_step_out_of_a_multi_target_fork(self):
        """Ke7/Ke8 answer Nxf7+; the old popcount test called them defenceless.

        The forker still attacks Rh8 and Bh6 from f7 after the king steps
        aside, which is what the old "does it still hit two pieces" test
        keyed on. What it misses is that Nxf7 is no longer check, so the
        knight simply hangs to the king.
        """
        defences = _defences(NXF7_FEN)
        self.assertIn("Ke7", defences)
        self.assertIn("Ke8", defences)

    def test_bishop_shuffle_is_not_a_defence(self):
        """Bg7 saves one forked piece and leaves the K+R fork on.

        The move actually played. It has the right *shape* (it moves a
        target), so only the evaluate_fork re-check can reject it.
        """
        self.assertNotIn("Bg7", _defences(NXF7_FEN))

    def test_king_can_cover_the_fork_square(self):
        """Kd8/Kd7 defend c7 against Nc7+, which PIECE_VALS[KING] forbade.

        `evaluate_fork` has already established the forker lands undefended,
        so a king that attacks the fork square just takes it -- the one
        defender for which the `<= forker_value` comparison is always wrong.
        """
        defences = _defences(NC7_FEN)
        self.assertIn("Kd8", defences)
        self.assertIn("Kd7", defences)

    def test_covering_defences_still_recognised(self):
        """The cases that already worked keep working.

        Na6 covers c7 with a piece the fork test always accepted; Bxe5
        covers it from e5 while winning a pawn; Bd2+ makes the fork move
        illegal by giving check.
        """
        self.assertIn("Na6", _defences(NC7_FEN))
        self.assertIn("Bxe5", _defences(NC7_FEN))
        self.assertIn("Bd2+", _defences(NXF7_FEN))

    def test_classifier_stays_selective(self):
        """A defence set that swallows the whole move list boosts nothing.

        The old geometry called *every* legal move a defence in a third of
        real fork positions (82 of 250 sampled from a logged game), which
        makes the multiplicative boost inert wherever it fires.
        """
        for fen in (NXF7_FEN, NC7_FEN):
            n_legal = chess.Board(fen).legal_moves.count()
            self.assertLess(len(_defences(fen)), n_legal, fen)


class TestForkDefenceBoost(unittest.TestCase):
    """The boost itself, driven by the logged NN distributions.

    Each test compares the real alteration path against itself with
    FORK_DEFENCE_SF neutralised to 1.0, which isolates this one feature from
    the dozen other multipliers in the same pass. The neutralised numbers
    reproduce the logged `Move_dic after alteration` line for the 2026-08-22
    position exactly (Bg7 0.225, Ke7 0.112, Rh7 0.102, Be6 0.085), which is
    what pins these fixtures to the games they came from.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = AlterMoveProbNN()
        cls.model.load_state_dict(torch.load(ALTER_MOVE_PROB_WEIGHTS_PTH, weights_only=True))
        cls.model.eval()
        cls.model.load_params_dict()

    def _altered(self, fen, san_probs, prev=None, prev_prev=None, fork_sf=None):
        board = chess.Board(fen)
        move_dic = {board.parse_san(san).uci(): p for san, p in san_probs.items()}
        original = alter_move_prob_nn.FORK_DEFENCE_SF
        if fork_sf is not None:
            alter_move_prob_nn.FORK_DEFENCE_SF = fork_sf
        try:
            altered, _ = self.model.forward_numpy(
                move_dic, board,
                chess.Board(prev) if prev else None,
                chess.Board(prev_prev) if prev_prev else None)
        finally:
            alter_move_prob_nn.FORK_DEFENCE_SF = original
        return {board.san(chess.Move.from_uci(u)): p for u, p in altered.items()}

    def test_defence_is_promoted_past_a_higher_ranked_non_defence(self):
        """Ke7 must end up top of the list, and the boost must widen its lead.

        The 2026-08-22 loss: search breadth was 1, so the single root move
        handed to Stockfish was Bg7, the one move that drops the exchange
        (Ke7 and Ke8 are worth +409 and +421 against Bg7's -311).

        This test used to assert that Bg7 led whenever FORK_DEFENCE_SF was
        neutralised, pinning the alteration's output at Bg7 0.225 / Ke7 0.112
        / Rh7 0.102 / Be6 0.085. Those were the numbers of a *bug*, not of the
        model: get_parameters_dict omitted solo_factor_sf,
        threatened_lvl_diff_sf and interesting_move_threshold, so forward_numpy
        read all three from its `.get` defaults and the live engine ran with
        the interesting-move threshold 3.06x too high and the threat-block
        exponent 2.24x too strong. With the parameters actually loaded, the
        neutral arm already puts Ke7 top at 0.149, so the fork boost was added
        to compensate for a blunder the plumbing had caused.

        The boost still earns its place -- it takes Ke7 from 0.149 to 0.208 and
        pushes the non-defences further down -- so what is pinned now is that
        it strictly widens Ke7's margin over Bg7 rather than that it rescues an
        ordering nothing else could reach.
        """
        boosted = self._altered(NXF7_FEN, NXF7_NN_PROBS, NXF7_PREV, NXF7_PREV_PREV)
        neutral = self._altered(NXF7_FEN, NXF7_NN_PROBS, NXF7_PREV, NXF7_PREV_PREV,
                                fork_sf=1.0)
        self.assertEqual(max(boosted, key=boosted.get), "Ke7")
        self.assertGreater(boosted["Ke7"], neutral["Ke7"])
        self.assertGreater(boosted["Ke8"], neutral["Ke8"])
        self.assertGreater(boosted["Ke7"] - boosted["Bg7"],
                           neutral["Ke7"] - neutral["Bg7"])

    def test_buried_defence_still_reached(self):
        """Na6 -- the 2026-08-05 case the threshold gate was added for.

        It sat at NN rank #9 on 0.9%. Removing the gate must not stop the
        boost reaching it. It does land lower than it did under the gate
        (0.055 against 0.088) because it no longer has the boost to itself,
        but the move that overtakes it is Bxe5, which is better -- see below.
        """
        boosted = self._altered(NC7_FEN, NC7_NN_PROBS)
        neutral = self._altered(NC7_FEN, NC7_NN_PROBS, fork_sf=1.0)
        self.assertGreater(boosted["Na6"], neutral["Na6"])

    def test_defence_the_net_already_liked_is_boosted_too(self):
        """Bxe5 answers the fork *and* is the engine's best move (-421 at
        depth 16, against -534 for the played Ne7 and -537 for Na6).

        It sat second on the net's own ranking, above the interesting-move
        threshold, so the old gate skipped it and the position played Ne7.
        Boosting every defence puts it top. This is the case the gate got
        backwards: a defence the net already rates is not a reason to
        withhold the boost.
        """
        boosted = self._altered(NC7_FEN, NC7_NN_PROBS)
        neutral = self._altered(NC7_FEN, NC7_NN_PROBS, fork_sf=1.0)
        self.assertEqual(max(neutral, key=neutral.get), "Ne7")
        self.assertEqual(max(boosted, key=boosted.get), "Bxe5")
        self.assertGreater(boosted["Bxe5"], neutral["Bxe5"])


if __name__ == "__main__":
    unittest.main()
