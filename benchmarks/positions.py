"""A fixed, self-contained position corpus.

Cross-device comparability requires every machine to benchmark *identical*
inputs, so the corpus is hardcoded here rather than sampled from a PGN.
That is the main thing separating this module from
simulation/calibrate.py's `compute` command, which replays a real bot PGN:
useful for calibrating the simulator against the games that machine
actually played, useless for comparing two machines that have no PGN in
common. `cheat_detection/` corpora and `logs/sessions/` are also gitignored
or pruned on a 7-day window, so neither survives a fresh clone.

Positions span all three phases because the engine picks a different
MoveScorer per phase and the search-breadth formulas differ, so a
single-phase corpus would misreport the mix a real game incurs.
"""

from __future__ import annotations

import chess

# (fen, label). Every position has the side to move NOT in check and a
# legal move available; validated by testing/benchmarks/test_positions.py.
POSITIONS = [
    # --- opening -----------------------------------------------------
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3",
     "italian_black_to_move"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 3",
     "two_knights_white"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 6 6",
     "giuoco_piano"),
    ("rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
     "slav_white"),
    # --- midgame -----------------------------------------------------
    ("r4rk1/pp2ppbp/2np1np1/q7/2P1P3/2N1B3/PP1QBPPP/3R1RK1 w - - 0 14",
     "dragon_castled"),
    ("r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 10",
     "quiet_iqp"),
    ("2r2rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 15",
     "sicilian_opp_castle"),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 11",
     "symmetrical_tension"),
    ("2rq1rk1/pp1bppbp/3p1np1/n7/2PNP3/2N1BP2/PP1QB1PP/R3K2R w KQ - 0 13",
     "sharp_sicilian"),
    # --- endgame -----------------------------------------------------
    ("8/p4kpp/4p3/5p2/1bP5/1P2P3/PB1p1PPP/5K2 w - - 0 31",
     "incident_promotion_race"),
    ("8/5ppp/4p3/3k4/8/4K3/5PPP/8 w - - 0 40", "king_and_pawns"),
    ("8/8/4k3/8/3R4/4K3/8/6r1 w - - 0 45", "rook_endgame"),
]

# The engine reads at most 5 historical fens and the opponent's last 4
# clock times; anything beyond that is dead weight in the info dict.
FEN_HISTORY = 5


def info_for(fen, own_time=45.0, opp_time=45.0, initial_time=60.0,
             rating=2300):
    """Build the engine `info_dic` for a bare FEN.

    Single-fen history with no last_moves is the shape
    testing/engine_components/test_reeval_order.py already drives the
    engine with, so it is a known-good input rather than a guess. It does
    mean history-dependent paths (opponent-blunder startle, patch_fens
    move linking) stay quiet -- deliberate, since those fire
    unpredictably and would add variance that has nothing to do with the
    machine.
    """
    board = chess.Board(fen)
    return {
        "side": board.turn,
        "fens": [fen],
        "last_moves": [],
        "self_clock_times": [own_time],
        "opp_clock_times": [opp_time],
        "self_initial_time": initial_time,
        "opp_initial_time": initial_time,
        "self_rating": rating,
        "opp_rating": rating,
    }
