"""Tuning constants for search breadth and candidate-move scan widths.

This module centralises the magic numbers that decide *how many moves the
bot considers* at each stage of move selection. Previously these were
hardcoded inline in `engine.py`, `engine_components/decision_logic.py`,
`engine_components/analyzer.py` and `engine_components/premover.py`.

The overall pipeline, and where each group of constants applies:

1. The neural network produces a probability-ranked list of "human" moves.
2. `_decide_breadth` (engine.py) / `decide_breadth` (decision_logic.py)
   decides how many of those top moves become Stockfish root moves.
   That decision starts from `DIFFICULTY` (see common/constants.py) and is
   adjusted by thinking time, position character and mood — the
   BREADTH_* / MOOD_* / KING_DANGER_* / EFF_MOB_* constants below.
3. Independent of the human-move filter, several raw Stockfish multipv
   scans use fixed widths — the *_SCAN_* constants below.
4. Re-evaluation and pondering deepen a subset of lines; their maximum
   depth grows with the square root of the search breadth — the
   MAX_CALC_DEPTH_* constants below.

Note: raising breadth/widths makes the bot stronger but less human-like
(and slower); the values here were chosen to mimic how a human's candidate
set widens or narrows with the position.
"""

# ---------------------------------------------------------------------------
# 1. Search breadth: thinking-time bonus
# ---------------------------------------------------------------------------
# When the bot has decided to spend longer on a move, it also considers more
# candidate moves (humans surveying more options during a long think).
# Tiers are (minimum target thinking time in seconds, extra root moves) and
# MUST be ordered by descending time: the first tier whose threshold the
# target time exceeds is applied, and only that one.
#
# Example with the default tiers: a 3s think gets +2 root moves; a 6s think
# gets +3; anything at or under 1.5s gets no bonus.
BREADTH_TIME_BONUS_TIERS = (
    (5.0, 3),
    (2.5, 2),
    (1.5, 1),
)

# ---------------------------------------------------------------------------
# 2. Search breadth: position-character adjustments
# ---------------------------------------------------------------------------
# These act on top of the base breadth (DIFFICULTY + time bonus). The logic
# branches on Lucas "effective mobility" (roughly: how many reasonable moves
# the position offers), king danger, and game phase.

# Lucas eff_mob below this means the position is cramped/forced — either the
# best move is obvious (recapture, mate-in-one) or a tactic is in progress.
EFF_MOB_TACTICAL_CUTOFF = 15

# Within the low-mobility branch, eff_mob must still exceed this for the
# position to count as a "tactical midgame" (very low mobility instead falls
# through to the default branch — the move is likely forced anyway).
EFF_MOB_FORCED_CUTOFF = 5

# king_danger() score above which the king is considered under real attack.
KING_DANGER_THRESHOLD = 500

# Extra root moves considered when the king is in danger — defensive
# resources must not be missed, whatever the phase or mobility.
KING_DANGER_BREADTH_BONUS = 10

# Breadth delta in a tactical midgame (low mobility, king safe): humans
# tunnel-vision on the tactic, so the search *narrows*.
TACTICAL_MIDGAME_BREADTH_DELTA = -1

# Endgames require precision, so breadth is forced up to a floor of
# max(<floor>, base + <bonus>). The low-mobility endgame branch uses a
# slightly higher floor / lower bonus than the open ("conversion") one.
ENDGAME_LOW_MOB_BREADTH_FLOOR = 8
ENDGAME_LOW_MOB_BREADTH_BONUS = 4
ENDGAME_BREADTH_FLOOR = 7
ENDGAME_BREADTH_BONUS = 5

# Breadth delta for a normal opening/midgame position with plenty of good
# moves: slightly narrower than base, since any sensible move is fine.
STANDARD_BREADTH_DELTA = -1

# ---------------------------------------------------------------------------
# 3. Search breadth: mood adjustments
# ---------------------------------------------------------------------------
# Applied last (see engine_components/mood_manager.py for how moods arise).
# Missing moods (e.g. "confident") get no adjustment. Breadth is always
# floored at 1 after this step.
MOOD_BREADTH_DELTAS = {
    "cocky": -1,    # feeling superior -> less careful
    "hurry": -1,    # low on clock -> snap decisions
    "cautious": +1, # worried -> double-checks more candidates
    "tilted": -2,   # frustrated -> plays impulsively
}

# ---------------------------------------------------------------------------
# 4. Fixed Stockfish scan widths (independent of the human-move filter)
# ---------------------------------------------------------------------------

# The initial per-move analysis (engine_components/analyzer.py) evaluates
# every legal move as its own multipv line so any move the human model
# suggests has an engine eval to cross-reference; capped for performance in
# absurdly wide positions.
INITIAL_SCAN_MULTIPV_CAP = 50

# The sharpness scan (engine.py:_compute_sharpness) measures the
# win-probability spread across the top candidates to drive move-time
# pacing. Width and depth MUST match the cheat_detection/ analyser's
# sharpness definition, or the bot's pacing and the offline human-likeness
# diagnostic will disagree about which positions are "critical".
SHARPNESS_SCAN_MULTIPV = 5
SHARPNESS_SCAN_DEPTH = 12

# Width of the quick scan of the position *after* the opponent's predicted
# reply, used to pick a premove / anticipated response (engine.py premove
# logic and engine_components/premover.py).
PREMOVE_SCAN_MULTIPV = 10

# ---------------------------------------------------------------------------
# 5. Re-evaluation / ponder depth scaling
# ---------------------------------------------------------------------------
# When re-evaluating top human lines (engine.py:get_human_move) or pondering
# (engine.py:ponder), lines may be followed several plies deep. Maximum depth
# scales as int(COEFF * sqrt(search_width)): wider searches earn deeper
# follow-up, but depth is capped because the human-probability model degrades
# with depth — too-deep recursion increases the chance of silly moves.
MAX_CALC_DEPTH_COEFF = 2.5

# ---------------------------------------------------------------------------
# 6. Ponder budget
# ---------------------------------------------------------------------------
# Nominal cost per pondered position (engine.py:make_move -> ponder). The
# leftover think budget divided by this sets variations_allowed, which in
# turn sets how many opponent replies the ponder covers (the depth>=2 rule
# in Engine.ponder). At 0.1 a typical bullet move's leftover budget only
# ever covered ONE reply -- the realised ponder width was budget-bound, not
# cap-bound (raising max_ponder_no measurably did nothing, ralph iter1/2).
# Lowering the per-position cost is the structural lever on the ponder-hit
# rate, the dominant instant-move channel: 2500-2800 humans fire 60% of
# sub-10s moves instantly off preparation. Cost: shallower per-position
# evals for the pondered replies.
PONDER_TIME_PER_POSITION = 0.05
