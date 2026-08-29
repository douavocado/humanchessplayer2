"""Per-move compute cost, broken into the stages that scale differently.

STAGES ARE NESTED, NOT A PARTITION. Each is timed by calling it directly,
so several are components of others and the column does not sum to
make_move_total:

    make_move_total                 end-to-end, what the client brackets
      analytics_total               runs inside update_info in production
        sf_scan_fullwidth             multipv full-width, Limit(depth=10, time=0.02)
        lucas_analytics               pure numpy/python over that scan
        sf_sharpness                  multipv 5, Limit(depth=12), no time cap
        set_mood                      a further short Stockfish probe
      nn_probabilities              the production NN probability path
        nn_move_scorer                the raw MoveScorer torch forward passes
      nn_alter                      AlterMoveProbNN.forward_numpy
      ponder                        elective; see compute_floor below

Why the split matters. The two Stockfish scans are limited differently, and
that changes what a slow device costs you:

  sf_scan_fullwidth is time-capped (20ms nominal, though a full-width
  multipv scan overshoots that: ~50ms measured). It reaches depth 4-7 of
  the 10 it asks for, so the depth limit never binds and the time limit
  is what is operative. A slower CPU
  therefore buys a shallower scan rather than a longer one, i.e. it
  degrades eval quality rather than adding latency, which no pacing knob
  can compensate for. Depth reached is recorded alongside wall time so
  that shows up in a comparison instead of hiding.

  sf_sharpness is depth-capped with no time limit, so a slower CPU pays
  in wall time here, straightforwardly.

COMPUTE FLOOR. make_move deliberately spends whatever is left of the think
budget on pondering (engine.py: time_left = time_take - time_spent, then
ponder(time_left / 1.15)), so make_move_total tracks time_take by
construction and comparing the two measures the design, not the device.
The device-limited quantity is make_move_total minus that elective ponder:
the fastest this machine can produce a move at all. That is the number
that decides whether a pacing knob is even reachable here -- the client
sleeps only (time_take - elapsed - MOVE_DELAY) and skips the sleep when
that is negative, so any intended move time below the floor is silently
unachievable.
"""

from __future__ import annotations

import re
import time

import chess

from benchmarks import positions as _positions
from benchmarks.stats import summarise
from common.board_information import get_lucas_analytics, phase_of_game
from common.move_timing import MOVE_DELAY
from common.search_constants import SHARPNESS_SCAN_DEPTH
from engine_components.state import compute_ambiguity

# Fixed workload for the raw engine probe. Depth-limited with no time cap,
# so wall time is a pure hardware measure; a dense midgame position because
# an endgame at fixed depth finishes too fast to resolve.
NPS_FEN = "r4rk1/pp2ppbp/2np1np1/q7/2P1P3/2N1B3/PP1QBPPP/3R1RK1 w - - 0 14"
NPS_DEPTH = 18

STAGES = ["update_info", "sf_scan_fullwidth", "lucas_analytics",
          "sf_sharpness", "set_mood", "analytics_total", "nn_move_scorer",
          "nn_probabilities", "nn_alter", "make_move_total", "compute_floor"]

# Indent depth in the report, encoding the nesting documented above so the
# columns are not mistakenly added up.
NESTING = {"sf_scan_fullwidth": 1, "lucas_analytics": 1, "sf_sharpness": 1,
           "set_mood": 1, "nn_move_scorer": 1}

# Stages whose wall time tracks CPU speed rather than being pinned by a
# time cap. The compare command reads this to say what a difference means.
MACHINE_VARIABLE = ["sf_sharpness", "nn_move_scorer", "nn_probabilities",
                    "nn_alter", "lucas_analytics", "compute_floor"]
TIME_CAPPED = ["sf_scan_fullwidth"]

# engine.py logs this after its elective ponder; parsed to separate the
# device-limited compute floor from budget the engine chose to spend.
_PONDER_SECS = re.compile(r"Took ([\d.eE+-]+) seconds for pondering")


def _analysis_depth(analysis):
    """Depth reached by a multipv analysis.

    Minimum across lines: the engine compares candidates against each
    other, so the shallowest line limits that comparison.
    """
    if not analysis:
        return None
    if isinstance(analysis, dict):
        analysis = [analysis]
    depths = [e.get("depth") for e in analysis if e.get("depth") is not None]
    return min(depths) if depths else None


def stockfish_nps_probe():
    """Raw engine throughput: fixed depth on a fixed position.

    Independent of the bot's own scan settings, so it stays comparable even
    if search_constants are retuned, and it is the cleanest single number
    for "how fast does Stockfish run on this machine".
    """
    import chess.engine

    from common.constants import PATH_TO_STOCKFISH
    from common.search_constants import STOCKFISH_HASH_MB, STOCKFISH_THREADS

    board = chess.Board(NPS_FEN)
    eng = chess.engine.SimpleEngine.popen_uci(PATH_TO_STOCKFISH)
    try:
        eng.configure({"Threads": STOCKFISH_THREADS, "Hash": STOCKFISH_HASH_MB})
        t0 = time.perf_counter()
        info = eng.analyse(board, chess.engine.Limit(depth=NPS_DEPTH))
        secs = time.perf_counter() - t0
    finally:
        eng.quit()
    return {"depth": NPS_DEPTH, "secs": secs,
            "nps": info.get("nps"), "nodes": info.get("nodes")}


def _measure_position(engine, fen, label, seed):
    """One position through every stage."""
    info = _positions.info_for(fen)
    rec = {"fen": fen, "label": label}

    # auto_update_analytics=False so calculate_analytics is timed on its own
    # rather than hidden inside update_info.
    t0 = time.perf_counter()
    engine.update_info(info, auto_update_analytics=False)
    rec["update_info"] = time.perf_counter() - t0

    board = engine.current_board
    phase = phase_of_game(board)
    rec["phase"] = phase
    rec["legal_moves"] = board.legal_moves.count()

    # --- components of calculate_analytics -------------------------------
    # The full-width time-capped scan, with production's exact limit
    # (engine_components/state.py).
    # Each component also writes the engine state that calculate_analytics
    # would have written, because the stages below genuinely depend on it
    # (adjust_human_prob reads engine.stockfish_analysis[0]). Assigning the
    # results we already have keeps the state production-faithful without
    # paying for either scan twice.
    no_lines = board.legal_moves.count()
    t0 = time.perf_counter()
    analysis = engine.stockfish_engine.analyse(
        board, limit=chess.engine.Limit(depth=10, time=0.02), multipv=no_lines)
    rec["sf_scan_fullwidth"] = time.perf_counter() - t0
    if isinstance(analysis, dict):  # single-line results come back unwrapped
        analysis = [analysis]
    rec["sf_scan_depth"] = _analysis_depth(analysis)
    engine.stockfish_analysis = analysis

    t0 = time.perf_counter()
    xcomp, xmlr, xemo, xnar, xact = get_lucas_analytics(board, analysis=analysis)
    rec["lucas_analytics"] = time.perf_counter() - t0
    engine.lucas_analytics.update({"complexity": xcomp, "win_prob": xmlr,
                                   "eff_mob": xemo, "narrowness": xnar,
                                   "activity": xact})

    t0 = time.perf_counter()
    engine.sharpness = engine._compute_sharpness()
    rec["sf_sharpness"] = time.perf_counter() - t0
    rec["sharpness"] = engine.sharpness
    engine.ambiguity = compute_ambiguity(engine.sharpness_scan)

    t0 = time.perf_counter()
    engine.mood = engine._set_mood()
    rec["set_mood"] = time.perf_counter() - t0
    engine.analytics_updated = True

    # Derived rather than measured. Timing calculate_analytics itself here
    # would re-run both scans against a transposition table the components
    # just warmed, and the second reading of the same work is understated
    # (CLAUDE.md flags this: its quick scan "leaves load-dependent
    # transposition-table state that perturbs later same-process scans").
    # Measured that way the composite came out *below* its own sharpness
    # component. Running the components once each, in production order, is
    # the reading that transfers between machines.
    #
    rec["analytics_total"] = (rec["sf_scan_fullwidth"] + rec["lucas_analytics"]
                              + rec["sf_sharpness"] + rec["set_mood"])

    # --- neural net ------------------------------------------------------
    # Mirrors human_move_logic.get_human_probabilities: the nets are trained
    # from white's perspective, so a black-to-move board is mirrored first.
    scorer = engine.human_scorers[phase]
    dummy = board.copy() if board.turn == chess.WHITE else board.mirror()
    t0 = time.perf_counter()
    scorer.get_move_dic(dummy, san=False, top=100)
    rec["nn_move_scorer"] = time.perf_counter() - t0

    # The production wrapper: adds a mate-threat check that itself calls
    # Stockfish, so this is strictly larger than nn_move_scorer.
    t0 = time.perf_counter()
    move_dic = engine.get_human_probabilities(board, phase, log=False)
    rec["nn_probabilities"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    engine._alter_move_prob_nn(move_dic, board, log=False)
    rec["nn_alter"] = time.perf_counter() - t0

    # --- end to end ------------------------------------------------------
    # update_info with analytics on, exactly as the client does, so the
    # total matches the window clients/mp_original.py brackets.
    engine.log = ""
    t0 = time.perf_counter()
    engine.update_info(info)
    out = engine.make_move(log=False, seed=seed)
    rec["make_move_total"] = time.perf_counter() - t0

    ponder = sum(float(m) for m in _PONDER_SECS.findall(engine.log))
    rec["ponder_secs"] = ponder
    rec["compute_floor"] = max(0.0, rec["make_move_total"] - ponder)
    rec["time_take"] = out.get("time_take")
    if rec["time_take"] is not None:
        rec["time_take"] = float(rec["time_take"])
        # The client sleeps (time_take - elapsed - MOVE_DELAY) and skips the
        # sleep when that is negative, so an intended time below the floor
        # cannot be produced on this machine.
        rec["floor_exceeds_intended"] = (
            rec["compute_floor"] > rec["time_take"] - MOVE_DELAY)
    return rec


def run(repeat=3, seed=4242, engine_kwargs=None, progress=True):
    from common.constants import DIFFICULTY
    from engine import Engine

    kwargs = {"playing_level": DIFFICULTY, "log_file": None}
    kwargs.update(engine_kwargs or {})

    if progress:
        print("  loading Engine (playing_level={})...".format(
            kwargs["playing_level"]), flush=True)
    t0 = time.perf_counter()
    engine = Engine(**kwargs)
    startup_secs = time.perf_counter() - t0

    records, skipped = [], []
    try:
        for rep in range(repeat):
            for idx, (fen, label) in enumerate(_positions.POSITIONS):
                try:
                    rec = _measure_position(engine, fen, label,
                                            seed + rep * 1000 + idx)
                except Exception as exc:  # noqa: BLE001 - keep measuring the rest
                    skipped.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                rec["rep"] = rep
                records.append(rec)
            if progress:
                print(f"  pass {rep + 1}/{repeat} done ({len(records)} measurements)", flush=True)
    finally:
        engine.close_engines()

    stages = {name: summarise([r[name] for r in records if name in r])
              for name in STAGES}
    by_phase = {}
    for ph in ("opening", "midgame", "endgame"):
        sub = [r for r in records if r.get("phase") == ph]
        if sub:
            by_phase[ph] = {n: summarise([r[n] for r in sub if n in r])
                            for n in STAGES}

    unreachable = [r for r in records if r.get("floor_exceeds_intended")]
    return {
        "engine_startup_secs": startup_secs,
        "stages": stages,
        "by_phase": by_phase,
        "sf_scan_depth": summarise([r["sf_scan_depth"] for r in records
                                    if r.get("sf_scan_depth") is not None]),
        "sharpness_scan_depth": SHARPNESS_SCAN_DEPTH,
        "nps_probe": stockfish_nps_probe(),
        "floor": {
            "move_delay": MOVE_DELAY,
            "n": len(records),
            "unreachable_count": len(unreachable),
            "unreachable_rate": (len(unreachable) / len(records)
                                 if records else None),
            "ponder_secs": summarise([r["ponder_secs"] for r in records
                                      if r.get("ponder_secs")]),
        },
        "records": records,
        "skipped": skipped,
    }


def _row(s):
    return "{:9.1f} {:9.1f} {:9.1f} {:9.1f}".format(
        s["mean"] * 1000, s["p50"] * 1000, s["p90"] * 1000, s["max"] * 1000)


def report(result):
    """Human-readable table."""
    lines = ["  stage                  mean_ms    p50_ms    p90_ms    max_ms",
             "  " + "-" * 64]
    for name in STAGES:
        s = result["stages"].get(name) or {}
        if not s.get("n"):
            continue
        label = "  " * NESTING.get(name, 0) + name
        tag = "   <- time-capped" if name in TIME_CAPPED else ""
        lines.append(f"  {label:<21s}" + _row(s) + tag)
    lines.append("  (nested, not a partition -- see module docstring)")

    depth = result.get("sf_scan_depth") or {}
    if depth.get("n"):
        lines.append("")
        lines.append("  full-width scan reached depth mean {:.2f} "
                     "(min {}, p90 {}) of the 10 requested".format(
                         depth["mean"], depth["min"], depth["p90"]))
        lines.append("    the time cap binds, not the depth cap: a slower "
                     "device loses depth here, not speed")

    nps = result.get("nps_probe") or {}
    if nps.get("nps"):
        lines.append("  stockfish depth {} probe: {:.2f}s at {:.2f} Mnps".format(
            nps["depth"], nps["secs"], nps["nps"] / 1e6))
    if result.get("engine_startup_secs"):
        lines.append("  engine startup (weights + 2 stockfish): {:.2f}s".format(
            result["engine_startup_secs"]))

    f = result.get("floor") or {}
    floor = (result["stages"].get("compute_floor") or {})
    if floor.get("n"):
        lines.append("")
        lines.append("  compute floor: p50 {:.0f}ms, p90 {:.0f}ms, "
                     "max {:.0f}ms".format(floor["p50"] * 1000,
                                           floor["p90"] * 1000,
                                           floor["max"] * 1000))
        lines.append("    fastest a move can be produced here; intended "
                     "times below this are unachievable")
        if f.get("n"):
            lines.append("    intended move time was below the floor on "
                         "{}/{} moves ({:.1f}%)".format(
                             f["unreachable_count"], f["n"],
                             f["unreachable_rate"] * 100))
    if result.get("skipped"):
        lines.append("  skipped {} position(s): {}".format(
            len(result["skipped"]), result["skipped"][0]["error"]))
    return "\n".join(lines)
