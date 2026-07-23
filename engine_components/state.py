"""Position/game-boundary state tracking, extracted verbatim from engine.Engine.

new_game / update_info / calculate_analytics / _compute_sharpness together
own the engine's view of "what position are we in and what do we know about
it": syncing self.current_board from scraped fens+moves, detecting game
boundaries, and running the Stockfish scans that populate
self.stockfish_analysis / self.lucas_analytics / self.sharpness. Everything
downstream (decisions, pacing, pondering) reads state these functions set.

Second slice of the engine.py strangler-fig extraction (see
testing/engine_parity/ for the regression harness). Like simple_decisions.py,
each function is a verbatim move (self -> engine) -- no interface redesign.
Calls into other Engine methods not yet extracted (_sample_game_character,
_set_mood) go through the still-present Engine methods, so behaviour is
unaffected by what has or hasn't been pulled out yet.
"""
import time

import numpy as np

import chess
import chess.engine

from common.board_information import get_lucas_analytics
from common.utils import scraped_fen_sanity_issues, InvalidPositionError
from common.search_constants import SHARPNESS_SCAN_MULTIPV, SHARPNESS_SCAN_DEPTH


def new_game(engine):
    """ Signals a game boundary: resamples per-game character (pace,
        premove propensity). update_info also detects boundaries itself
        (ply going backwards), so calling this is optional -- it only makes
        the boundary exact for games that start deeper into the book than
        the last one ended.
    """
    engine._last_seen_ply = None
    engine._sample_game_character()


def update_info(engine, info_dic, auto_update_analytics=True):
    """ The engine is fed the following thins in the info_dic, which a dictionary
        of board information:
            - side: either chess.WHITE or chess.BLACK - indicates what side we are
            - fens: List of fens ordered with most recent fen last. Engine makes
                    use of at most 5 previous fens.
            - self_clock_times and opp_clock_times: List of past clock times
                    with most recent last. From this we can also work out last move times.
            - self_initial_time and opp_initial_time: Starting clock times for self and opp
            - last_moves: A list of moves made with most recent last. These moves are in uci
                    string format.

        This function should be called the first thing before making any calculations.
    """

    engine.log += "Received and updating info_dic: \n"
    engine.log += str(info_dic) + "\n"
    engine.input_info.update(info_dic)

    # if more than one historic fen and moves are given, then use this movestack to
    # also account for repetition
    if len(engine.input_info["last_moves"]) >= 1:
        test_board = chess.Board(engine.input_info["fens"][-len(engine.input_info["last_moves"])-1])
        for move_uci in engine.input_info["last_moves"]:
            if chess.Move.from_uci(move_uci) in test_board.legal_moves:
                test_board.push_uci(move_uci)
            else:
                engine.log += "ERROR: Could not sync last moves of input info with fen history at move {}. Defaulting to last known fen and wiping history. \n".format(move_uci)
                engine.current_board = chess.Board(engine.input_info["fens"][-1])
                engine.input_info["fens"] = engine.input_info["fens"][-1:]
                engine.input_info["last_moves"] = []
                break
        if test_board.fen() == engine.input_info["fens"][-1]:
            engine.current_board = test_board.copy()
        elif len(engine.input_info["last_moves"]) > 0:
            engine.log += "ERROR: Played through last_move list but got fen {} instead of {}. Defaulting to last known fen. \n".format(test_board.fen(), chess.Board(engine.input_info["fens"][-1]))
            engine.current_board = chess.Board(engine.input_info["fens"][-1])
    else:
        engine.log += "No last moves given, so resorting to last fen in history with no move_stack. \n"
        engine.current_board = chess.Board(engine.input_info["fens"][-1])
    # make sure the last fen entry is indeed our turn
    assert engine.current_board.turn == engine.input_info["side"]

    # Detect a game boundary: the engine instance persists across games
    # (live sessions and simulation batches alike), so when the ply count
    # goes backwards we are in a new game -- resample per-game character.
    ply = engine.current_board.ply()
    if engine.game_pace_sf is None or (engine._last_seen_ply is not None and ply < engine._last_seen_ply):
        engine._sample_game_character()
    engine._last_seen_ply = ply

    engine.analytics_updated = False

    if auto_update_analytics == True:
        engine.calculate_analytics()


def calculate_analytics(engine):
    """ Before any move making or human analysis is performed, statistics must be computed for
        the infomation dict self.input_info. This function must be called after every
        update_info, or everytime the info dic is updated.

        Returns None
    """
    engine.log += "Calculating analytics for current information dictionary. \n"
    engine.log += f"Evaluating from the board (capital letters are white pieces), FEN: {engine.current_board.fen()}: \n"
    engine.log += str(engine.current_board) + "\n"

    # Performing a quick initial analysis of the position
    engine.log += "Performing initial quick analysis perhaps used later by stockfish. \n"
    start = time.time()
    no_lines = len(list(engine.current_board.legal_moves))

    # Stockfish segfaults on structurally impossible positions (e.g. a
    # missing king from a corrupt screen scrape), taking any restarted
    # engine down with it. Refuse to analyse instead; the client treats
    # this as "rescan the board", not a fatal error. Turn is
    # authoritative here (update_info asserted turn == side), so the
    # turn-dependent OPPOSITE_CHECK test is safe to include -- unlike at
    # the client's scrape sites, where the raw FEN's turn is a guess.
    sanity_issues = scraped_fen_sanity_issues(engine.current_board, turn_reliable=True)
    if sanity_issues:
        engine.log += "ERROR: Refusing stockfish analysis of structurally impossible position {} ({}). \n".format(
            engine.current_board.fen(), sanity_issues)
        raise InvalidPositionError(
            "Position {} is structurally impossible: {}".format(
                engine.current_board.fen(), sanity_issues))

    try:
        analysis = engine.stockfish_engine.analyse(engine.current_board, limit=chess.engine.Limit(depth=10, time=0.02), multipv=no_lines)
    except chess.engine.EngineTerminatedError:
        # Engine crashed - try to restart it
        engine.log += "WARNING: Stockfish engine crashed, attempting restart... \n"
        print("[ENGINE] WARNING: Stockfish engine crashed, attempting restart...")
        try:
            engine.stockfish_engine = chess.engine.SimpleEngine.popen_uci(engine._stockfish_path)
            analysis = engine.stockfish_engine.analyse(engine.current_board, limit=chess.engine.Limit(depth=10, time=0.02), multipv=no_lines)
            engine.log += "Engine restart successful. \n"
            print("[ENGINE] Engine restart successful.")
        except Exception as e:
            engine.log += f"ERROR: Failed to restart engine: {e} \n"
            print(f"[ENGINE] ERROR: Failed to restart engine: {e}")
            raise

    if isinstance(analysis, dict): # sometimes analysis only gives one result and is not a list.
        analysis = [analysis]
    engine.stockfish_analysis = analysis
    end = time.time()
    engine.log += "Analysis computed in {} seconds. \n".format(end-start)
    # Getting lucas analytics for the position
    engine.log += "Calculating lucas analytics for the position. \n"
    xcomp, xmlr, xemo, xnar, xact = get_lucas_analytics(engine.current_board, analysis=engine.stockfish_analysis)
    lucas_dict = {"complexity": xcomp, "win_prob": xmlr, "eff_mob": xemo, "narrowness": xnar, "activity": xact}
    engine.lucas_analytics.update(lucas_dict)
    engine.log += "Lucas analytics: {} \n".format(lucas_dict)

    # Sharpness of the position (eval-stakes among the plausible moves).
    # This is the criterion for "complicated position" used when deciding
    # how long to think (see _get_time_taken).
    engine.sharpness = engine._compute_sharpness()
    engine.log += "Position sharpness: {} \n".format(engine.sharpness)
    print(f"[ENGINE] Position sharpness: {engine.sharpness:.3f}")

    # Now determine our player "mood" and set it as our mode for the rest of the calculations
    engine.mood = engine._set_mood()
    engine.log += "Setted mood to be: {} \n".format(engine.mood)
    print(f"[ENGINE] Setted mood to be: {engine.mood}")
    engine.analytics_updated = True


def compute_sharpness(engine):
    """ Measures how "sharp" (critical) the current position is, using the
        same definition as the cheat_detection human-likeness analyser:
        the spread in win-probability across the engine's top candidate
        moves. A position where the best move is winning and the next-best
        is losing is sharp (a lot is at stake); one where every candidate
        keeps roughly the same win-probability is not.

        Distinct from the Lucas analytics (which measure *structural*
        complexity -- number of good moves, mobility, activity): this keys
        off eval-stakes instead. Computed from a narrow, slightly deeper
        scan (multipv 5, depth 12) than the Lucas scan.

        Returns sharpness : float in [0, 1]. Falls back to 0.25 (the neutral
        / "critical" threshold) if the position can't be scanned, so a
        failed scan leaves the move-time pacing unchanged.
    """
    # Logistic centipawn -> win-probability, matching the analyser.
    def winning_chances(cp):
        return 1.0 / (1.0 + np.exp(-0.00368208 * cp))

    engine.sharpness_scan = None
    try:
        no_lines = min(SHARPNESS_SCAN_MULTIPV, len(list(engine.current_board.legal_moves)))
        if no_lines == 0:
            return 0.25
        analysis = engine.stockfish_engine.analyse(
            engine.current_board,
            limit=chess.engine.Limit(depth=SHARPNESS_SCAN_DEPTH),
            multipv=no_lines,
        )
        if isinstance(analysis, dict):
            analysis = [analysis]
        wcs = [
            winning_chances(entry["score"].pov(engine.current_board.turn).score(mate_score=10000))
            for entry in analysis
        ]
        if not wcs:
            return 0.25
        # Keep the per-move win chances: this scan is the engine's only
        # deep, narrow view of the position, and _chosen_move_wc_loss
        # reads it (the full-width analysis is capped at 20ms, so its
        # evals are too noisy to judge the chosen move's loss).
        engine.sharpness_scan = {
            entry["pv"][0].uci(): wc
            for entry, wc in zip(analysis, wcs) if entry.get("pv")
        }
        return max(wcs) - min(wcs)
    except Exception as e:
        engine.log += "WARNING: sharpness scan failed ({}); defaulting to neutral. \n".format(e)
        print(f"[ENGINE] WARNING: sharpness scan failed ({e}); defaulting to neutral 0.25.")
        return 0.25
