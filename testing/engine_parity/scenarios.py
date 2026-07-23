"""Fixed input scenarios for the engine parity harness (see test_snapshot.py).

Each scenario is a complete `info_dic` (the shape `Engine.update_info` expects)
plus construction kwargs for `Engine`. Picked to cover materially different
code paths through `Engine`, not just different positions:

- midgame_reference: the worked example engine.py's own `__main__` block has
  used for years (real game fragment, includes move/takeback history).
- promotion_stop_incident: the live-flagged position from
  testing/client/test_promotion_stop.py (2026-07-14 incident).
- flag_race_scramble: own clock under 10s, black to move -- exercises the
  flag-race autopilot / scramble routing in _decide_human_filters and
  get_stockfish_move.
- resign_candidate: bare king vs king+queen, move 20, opponent well over
  10s -- exercises _decide_resign's material/win-probability branch.
"""

SCENARIOS = {
    "midgame_reference": {
        "quickness": 1.0,
        "playing_level": 6,
        "info_dic": {
            "fens": [
                "r2qk2r/pp3ppp/2p1pn2/4n3/1bPP4/P1N5/1P2QPPP/R1B2RK1 b kq - 1 12",
                "r2qk2r/pp3ppp/2p1pn2/4n3/2PP4/P1b5/1P2QPPP/R1B2RK1 w kq - 0 13",
                "r2qk2r/pp3ppp/2p1pn2/4P3/2P5/P1b5/1P2QPPP/R1B2RK1 b kq - 0 13",
                "r2qk2r/pp3ppp/2p1pn2/4b3/2P5/P7/1P2QPPP/R1B2RK1 w kq - 0 14",
                "r2qk2r/pp3ppp/2p1pn2/4Q3/2P5/P7/1P3PPP/R1B2RK1 b kq - 0 14",
                "r2q1rk1/pp3ppp/2p1pn2/4Q3/2P5/P7/1P3PPP/R1B2RK1 w - - 1 15",
                "r2q1rk1/pp3ppp/2p1pn2/4Q1B1/2P5/P7/1P3PPP/R4RK1 b - - 2 15",
                "rq3rk1/pp3ppp/2p1pn2/4Q1B1/2P5/P7/1P3PPP/R4RK1 w - - 3 16",
            ],
            "self_clock_times": [55, 55, 54, 53, 50, 49, 41, 40],
            "opp_clock_times": [57, 57, 56, 55, 54, 51, 50, 48],
            "last_moves": ["b4c3", "d4e5", "c3e5", "e2e5", "e8g8", "c1g5", "d8b8"],
            "side": True,
            "self_initial_time": 60,
            "opp_initial_time": 60,
            "opp_rating": 2543,
            "self_rating": 2560,
        },
    },
    "promotion_stop_incident": {
        "quickness": 1.0,
        "playing_level": 6,
        "info_dic": {
            "fens": ["8/p4kpp/4p3/5p2/1bP5/1P2P3/PB1p1PPP/5K2 w - - 0 31"],
            "self_clock_times": [40],
            "opp_clock_times": [35],
            "last_moves": [],
            "side": True,
            "self_initial_time": 60,
            "opp_initial_time": 60,
            "opp_rating": 2600,
            "self_rating": 2600,
        },
    },
    "flag_race_scramble": {
        "quickness": 1.0,
        "playing_level": 6,
        "info_dic": {
            "fens": ["r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 6 6"],
            "self_clock_times": [8],
            "opp_clock_times": [45],
            "last_moves": [],
            "side": False,
            "self_initial_time": 60,
            "opp_initial_time": 60,
            "opp_rating": 2500,
            "self_rating": 2500,
        },
    },
    "resign_candidate": {
        "quickness": 1.0,
        "playing_level": 6,
        "info_dic": {
            "fens": ["8/8/8/4k3/8/8/4K3/4q3 w - - 0 20"],
            "self_clock_times": [30],
            "opp_clock_times": [40],
            "last_moves": [],
            "side": True,
            "self_initial_time": 60,
            "opp_initial_time": 60,
            "opp_rating": 2500,
            "self_rating": 2500,
        },
    },
}

# Seeds used per scenario -- passed both to global random/numpy/torch seeding
# (before update_info, so per-game character sampling is reproducible) and to
# Engine.make_move(seed=...).
SEEDS = (743825, 12345)
