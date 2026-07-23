"""Golden-master parity harness for engine.py extractions.

Runs real Stockfish + real NN weights (no mocks) against a battery of fixed
scenarios (testing/engine_parity/scenarios.py) and compares the outputs
against recorded JSON snapshots. This is the regression gate for the
strangler-fig extraction of engine.py into engine_components/: after moving
a method's body into its own module and replacing it with a delegator, this
suite must still pass with zero diffs.

Usage:
    # Record/refresh golden snapshots from the current engine.py (only do
    # this deliberately -- it defines what "correct" means for every future
    # extraction step):
    venv/bin/python -m testing.engine_parity.test_snapshot --record

    # Normal check (what `unittest discover` runs):
    venv/bin/python -m unittest discover -s testing/engine_parity
"""
import json
import math
import os
import random
import sys
import unittest

import numpy as np
import torch

from engine import Engine
from testing.engine_parity.scenarios import SCENARIOS, SEEDS

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
# calculate_analytics (engine.py:1381) runs a depth=10/time=0.02s "quick"
# scan on self.stockfish_engine immediately before the depth-12 sharpness
# scan (engine.py:1447) reuses that *same* process. Under CPU load the 20ms
# scan reaches a different effective depth, leaving different
# hash-table/history-heuristic state behind, which measurably perturbs the
# later scan even though both runs request the same nominal depth. This is
# pre-existing engine.py behaviour (confirmed by repeated identical-code
# runs: observed drift up to ~0.06 on sharpness's [0,1] scale), not
# something this harness should paper over by mocking -- so continuous
# eval-derived floats get a tolerance well clear of that observed ceiling.
# Discrete outputs (move chosen, mood, every decide_*/check_obvious_move
# result) were rock-stable across dozens of repeated runs and are compared
# exactly below.
FLOAT_REL_TOL = 0.05
FLOAT_ABS_TOL = 0.15
# lucas_win_prob is read directly off that same 20ms-capped scan (not just
# perturbed by its leftover TT state, like sharpness is) and is on a 0-100
# scale -- a rare load spike was observed to swing it ~12 points in one run
# out of dozens otherwise identical. Give it its own, wider ceiling rather
# than loosening the global tolerance for every other field.
FIELD_ABS_TOL = {"lucas_win_prob": 25.0}


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_scenario(scenario, seed):
    """Drive one scenario through Engine and return a JSON-serialisable snapshot.

    Covers two layers: the three functions Step 3 of the revamp plan extracts
    into simple_decisions.py (called directly, so their inputs/outputs are
    pinned independently of make_move's own reseeding), and a full make_move
    call (the end-to-end contract). Future extraction steps should add more
    direct-method probes here as they touch other methods (see the revamp
    plan's backlog).
    """
    _seed_all(seed)
    engine = Engine(playing_level=scenario["playing_level"], quickness=scenario["quickness"])
    try:
        engine.update_info(scenario["info_dic"])

        obvious_move, obvious_move_found = engine.check_obvious_move()
        snapshot = {
            "sharpness": engine.sharpness,
            "lucas_win_prob": engine.lucas_analytics["win_prob"],
            "decide_resign": engine._decide_resign(),
            "decide_human_filters": engine._decide_human_filters(),
            "check_obvious_move": [obvious_move, obvious_move_found],
            # decide_human_filters isn't guaranteed False for any of the
            # fixed scenarios/seeds, so make_move alone never exercises
            # get_stockfish_move -- probe it directly too.
            "stockfish_move": engine.get_stockfish_move(log=False),
        }

        move_result = engine.make_move(log=False, seed=seed)
        snapshot["move_made"] = move_result["move_made"]
        snapshot["time_take"] = move_result["time_take"]
        snapshot["mood"] = engine.mood
        # make_move's premove search (get_premove) doesn't feed back into
        # move_made/time_take, so a crash there would fail this test but a
        # logic change to its output wouldn't -- record it explicitly.
        snapshot["premove"] = move_result.get("premove")
        return snapshot
    finally:
        engine.close_engines()


def _golden_path(name):
    return os.path.join(GOLDEN_DIR, "{}.json".format(name))


def _assert_snapshot_equal(test_case, expected, actual, path=""):
    if isinstance(expected, dict):
        test_case.assertIsInstance(actual, dict, path)
        test_case.assertEqual(set(expected), set(actual), "{}: key mismatch".format(path))
        for key in expected:
            _assert_snapshot_equal(test_case, expected[key], actual[key], "{}.{}".format(path, key))
    elif isinstance(expected, list):
        test_case.assertIsInstance(actual, list, path)
        test_case.assertEqual(len(expected), len(actual), "{}: length mismatch".format(path))
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_snapshot_equal(test_case, e, a, "{}[{}]".format(path, i))
    elif isinstance(expected, float):
        test_case.assertIsInstance(actual, (int, float), path)
        field_name = path.rsplit(".", 1)[-1]
        abs_tol = FIELD_ABS_TOL.get(field_name, FLOAT_ABS_TOL)
        test_case.assertTrue(
            math.isclose(expected, actual, rel_tol=FLOAT_REL_TOL, abs_tol=abs_tol),
            "{}: expected {!r}, got {!r}".format(path, expected, actual),
        )
    else:
        test_case.assertEqual(expected, actual, path)


class TestEngineParitySnapshot(unittest.TestCase):
    """One test per (scenario, seed) pair, generated below."""


def _make_test(name, seed):
    def test(self):
        golden_file = _golden_path("{}_{}".format(name, seed))
        if not os.path.exists(golden_file):
            self.fail(
                "No golden snapshot at {}. Run with --record to generate "
                "one from the current engine.py first.".format(golden_file)
            )
        with open(golden_file) as f:
            expected = json.load(f)
        actual = run_scenario(SCENARIOS[name], seed)
        _assert_snapshot_equal(self, expected, actual)

    test.__name__ = "test_{}_{}".format(name, seed)
    return test


for _name in SCENARIOS:
    for _seed in SEEDS:
        setattr(TestEngineParitySnapshot, "test_{}_{}".format(_name, _seed), _make_test(_name, _seed))


def record_all():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name, scenario in SCENARIOS.items():
        for seed in SEEDS:
            print("Recording {} seed={}...".format(name, seed))
            snapshot = run_scenario(scenario, seed)
            with open(_golden_path("{}_{}".format(name, seed)), "w") as f:
                json.dump(snapshot, f, indent=2, sort_keys=True)
                f.write("\n")
    print("Done.")


if __name__ == "__main__":
    if "--record" in sys.argv:
        record_all()
    else:
        unittest.main()
