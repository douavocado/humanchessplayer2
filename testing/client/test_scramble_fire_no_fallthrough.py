"""Regression: firing a scramble ponder move must end the move loop.

The low-time ponder-fire branch in run_game ended in a bare `break`, which
only leaves the `for move_obj in candidate_moves` loop. Every other ponder
fast path in the same function uses `continue`. So after firing, control fell
through to the normal engine path, which builds engine_input_dic from the
now-superseded DYNAMIC_INFO, asks the engine for a move, and clicks that too -
a second move chosen for a position that no longer exists.

Caught live in 2026-08-22 19:47:11 at t=615.3s:

    615.274  ... last ponder move fxf5 ... By chance making this pondered move
    615.274  Clicking move f7f5  ->  Made pondered moves successfully.
    615.872  Received output_dic from engine: {'move_made': 'g4g3', ...}
    617.769  Clicking move g4g3 -> piece never left g4 - move did not register
    617.769  Clicking move g4g3 -> piece never left g4 - move did not register

There it failed harmlessly (out of turn), but in 2026-08-20 21:29:55 at
t=366.5s and t=369.7s the second move registered - h5f3 then g8f8, g7f6 then
e6a6, the second of each pair chosen for the pre-fire position. It fired
1-4 times per session in 7 of the last 14 sessions, always in a flag race.

That is an unvetted premove channel: it skips scramble_fire_veto and
check_safe_premove, fires after an opponent deviation and after the clock has
dropped, and CLAUDE.md records raising premove volume that way as twice-proven
poison for blunder and match rates.

The branch lives inside run_game's `while True`, so the invariant is
structural: the candidate-move loop must be followed by a guard that continues
the outer loop, and its retry must replay the move it just tried.
"""
import ast
import inspect
import unittest


def _run_game_ast():
    import clients.mp_original as mp
    source = inspect.getsource(mp)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_game":
            return node
    raise AssertionError("run_game not found in clients/mp_original.py")


def _find_candidate_loop(run_game):
    """The `for move_obj in candidate_moves:` loop and its parent block."""
    for parent in ast.walk(run_game):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block):
                if (isinstance(stmt, ast.For)
                        and isinstance(stmt.iter, ast.Name)
                        and stmt.iter.id == "candidate_moves"):
                    return stmt, block, i
    raise AssertionError(
        "the scramble candidate-move loop is gone; if it was renamed or "
        "restructured, re-check that firing a move still ends the move loop")


class ScrambleFireNoFallthroughTest(unittest.TestCase):

    def setUp(self):
        # clients.mp_original instantiates Engine() and CustomCursor() at
        # module level; the other client tests stub those, and importing after
        # them reuses the stubs. Parse the source rather than the module so
        # this test does not depend on import order.
        import sys
        from unittest.mock import MagicMock
        for mod in ("engine", "common.custom_cursor", "pyautogui"):
            sys.modules.setdefault(mod, MagicMock())
        self.run_game = _run_game_ast()
        self.loop, self.block, self.index = _find_candidate_loop(self.run_game)

    def test_firing_a_move_continues_the_outer_loop(self):
        """
        A bare `break` here only leaves the `for`. Something after the loop
        has to take the outer `while` round again, or the engine path runs on
        a position we have already moved from.
        """
        after = self.block[self.index + 1:]
        self.assertTrue(after, "nothing follows the candidate-move loop, so "
                               "firing a scramble move falls through to the "
                               "engine request")
        guard = after[0]
        self.assertIsInstance(
            guard, ast.If,
            "the statement after the candidate-move loop must be the guard "
            "that ends the move loop once a move has been fired")
        self.assertTrue(
            any(isinstance(n, ast.Continue) for n in ast.walk(guard)),
            "the guard after the candidate-move loop must `continue` the "
            "outer while loop, not fall through to the engine request")

    def test_the_guard_is_armed_when_a_move_is_fired(self):
        """The guard is only worth anything if the fire path sets its flag."""
        guard = self.block[self.index + 1]
        flags = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
        assigned = {
            t.id
            for stmt in ast.walk(self.loop) if isinstance(stmt, ast.Assign)
            for t in stmt.targets if isinstance(t, ast.Name)
        }
        self.assertTrue(
            flags & assigned,
            "the guard after the candidate-move loop tests a name the loop "
            "body never sets, so firing a move would not trigger it")

    def test_retry_replays_the_move_it_just_tried(self):
        """
        The retry used to call make_move(last_pondered_move_obj.uci()) - a
        name this branch never binds. It is only assigned in the own_time > 10
        branch, so a mouse slip during a scramble fire either replayed a move
        from an earlier position or raised NameError.
        """
        names = {n.id for n in ast.walk(self.loop) if isinstance(n, ast.Name)}
        self.assertNotIn(
            "last_pondered_move_obj", names,
            "the scramble fire branch must not reference "
            "last_pondered_move_obj; it is never bound here")
        self.assertIn("move_obj", names)


if __name__ == "__main__":
    unittest.main()
