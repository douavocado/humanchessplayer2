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
# 2b. Search breadth: eval-stakes (sharpness) adjustment
# ---------------------------------------------------------------------------
# Breadth above reacts only to structural move-count (eff_mob) and king
# danger -- not eval-stakes. get_time_taken's "complicated position" trigger
# migrated to sharpness (win-probability spread across top candidates) a
# while back, but breadth never did: a real-stakes-but-no-single-forced-move
# position (moderate sharpness) gets no compensating widening the way quiet
# (<0.10, time cut) or sharp (>=0.25, king-danger bonus / tactical narrowing
# / intuition gate) positions do. The bucketed human-likeness diagnostic
# (cheat_detection/runs/difficulty4/buckets_*.md) showed this exact band is
# where the bot's blunder rate diverges most from human play, and diverges
# *further* -- not less -- at higher DIFFICULTY, i.e. widening breadth
# globally doesn't reach this band because decide_breadth isn't keying off
# the signal that identifies it. Thresholds match the diagnostic's bucket
# edges (and get_time_taken's sharpness thresholds) so "moderate" means the
# same thing everywhere. Applied as a flat bonus on top of whichever
# eff_mob/king-danger branch fired, mirroring how MOOD_BREADTH_DELTAS is
# layered on afterward.
#
# Magnitude calibration (cheat_detection/runs/moderate_breadth_fix/, 300-game
# D3 self-play, 2026-07-26): a +3 bonus over-corrected. It closed the gap
# against the 2800+ baseline (moderate-bucket blunder-rate z: +0.23 -> -0.08,
# win-chance-loss z: +0.28 -> -0.12 -- every |z| shrank) but overshot the
# bot's own native comparison population, 2500-2800 (blunder-rate z: +0.04
# -> -0.22, win-chance-loss z: +0.03 -> -0.29 -- that baseline was already
# near-perfectly calibrated pre-fix, so +3 pushed it past zero into a worse
# overshoot). Interpolating the two data points (bonus=0, bonus=3) against
# the 2500-2800 population's blunder rate puts the zero-crossing near +1;
# not re-verified by simulation at this exact value -- if retuning further,
# check the 2500-2800 bucket first since that is the population the sim
# bot's own rating (2450) actually sits in.
MODERATE_SHARPNESS_LO = 0.10
MODERATE_SHARPNESS_HI = 0.25
MODERATE_SHARPNESS_BREADTH_BONUS = 1

# ---------------------------------------------------------------------------
# 2c. Premove safety-vet probability (engine_components/premover.py)
# ---------------------------------------------------------------------------
# How often an ordinary (non-opening, non-scramble) premove gets a
# check_safe_premove second-guess before being allowed to fire, vs. trusted
# outright. Not a correctness bug fix so much as a discovered strength/
# human-likeness lever: raw search quality isn't the bottleneck (even a
# shallow Stockfish is far past human bullet strength) -- the bottleneck is
# how much a *fast, pre-committed* decision second-guesses itself, which is
# tunable the same way breadth and eval noise are. Validated once at the
# extreme (1.0 vs the old always-trust-it 0.0, cheat_detection/runs/
# adjudication/, 300-game D3 self-play under the default early-stopped
# simulation, 2026-07-26): premove's share of the instant-fire worst
# mistakes in adjudicated losses fell 61% -> 35%. Intermediate values are
# untested -- this constant exists to be swept, not treated as settled.
# Mirrors SCRAMBLE_VETO_P_* (common/constants.py) in spirit but is a flat
# probability rather than gated on a per-game skill draw, since (unlike the
# scramble branch, which must preserve an occasional human-catastrophe
# throw) there's no clock-pressure justification for ever skipping the
# check outside a flag race.
MIDGAME_PREMOVE_VETO_P = 1.0

# ---------------------------------------------------------------------------
# 2d. Per-phase breadth strength bonus (opening / midgame only)
# ---------------------------------------------------------------------------
# root_moves in get_human_move (engine.py) is human_move_ucis[:no_root_moves]
# -- the NN's top-N human-plausible ranking, truncated to the breadth decided
# here, BEFORE any of those candidates get a real Stockfish eval attached.
# The NN ranking, not engine search, is what actually bottlenecks strength:
# the eval every candidate in root_moves gets is already a genuine Stockfish
# score (from the pre-computed multipv analysis), so a move the NN ranks
# outside the window is never seen by the engine at all, however good it is.
# Widening breadth is therefore close to monotonically strengthening (more
# of the legal move list gets a real engine eval and a chance to win the
# final argmax) -- there is no meaningful "evaluation budget" tradeoff to
# widening it, unlike MODERATE_SHARPNESS_BREADTH_BONUS above, which reacts to
# eval-stakes rather than acting as a blanket strength dial. The cost is the
# obvious one: wider breadth also makes the bot look more like an engine and
# less like a human (fewer, more consistent mistakes) -- see the module
# docstring. That's why this isn't a single global knob: cheat_detection's
# elo_progression report (cheat_detection/runs/elo_progression/report.md,
# 8000-game human corpus, 2100-2900 pooled by rating band) found the human
# blunder-rate improvement with rating is concentrated almost entirely in
# game phase, not position character (sharpness/eff_mob buckets improve
# roughly proportionally, ~30-40% relative, at every rating band) -- opening
# blunder rate falls 56% relative from 2100-2299 to 2800+, middlegame 34%,
# endgame does not improve at all (+6.5%, noisy/non-monotonic). So the
# strength dial should widen breadth in the opening and (less so) the
# midgame, and explicitly NOT touch the endgame -- widening endgame breadth
# uniformly would make the bot's endgame play *more* consistent than real
# humans at any rating actually are, re-creating the same too-consistent
# failure mode DIFFICULTY raises across the board. Applied as a flat bonus
# on top of whichever eff_mob/king-danger/moderate-sharpness branch fired,
# same layering as MOOD_BREADTH_DELTAS. Both default to 0 (no behavioural
# change) -- these are unswept dials, not a calibrated fix; see
# engine.py/BotSpec for the per-instance overrides used to sweep them.
OPENING_BREADTH_STRENGTH_BONUS = 0
MIDGAME_BREADTH_STRENGTH_BONUS = 0

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

# Ambiguity is read off the *same* sharpness scan: how many candidates sit
# within this win-probability window of the best one. 1 means the position
# has a single right answer (a shot to recognise), >= 2 means several
# near-equal tries. The intuition snap gate splits on it -- humans snap
# recognisable positions far more often as they get stronger, while slowing
# down relatively on messy ones (docs/position-conditioned-human-likeness.md).
# Like the width/depth above this MUST match the analyser
# (cheat_detection/config.py:ambiguity_wc_window, same 0.05, also an
# inclusive `<=`), or the gate is tuned against a quantity we never measure.
AMBIGUITY_WC_WINDOW = 0.05

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

# ---------------------------------------------------------------------------
# 7. Opening book candidate width (engine_components/opening_book.py)
# ---------------------------------------------------------------------------
# How many polyglot entries at a matched position are eligible for
# weighted_choice; entries beyond this are excluded outright (not just
# down-weighted). The repertoire book is small and deliberately narrow by
# construction (rarely more than 2-3 real branches at any position), so its
# cap is tighter than the broad fallback book's.
OPENING_REPERTOIRE_TOP_N = 3
OPENING_BOOK_TOP_N = 5

# ---------------------------------------------------------------------------
# 8. Stockfish engine config (Threads / Hash)
# ---------------------------------------------------------------------------
# Every Stockfish call in the pipeline is wall-clock capped (most at 20ms;
# see engine_components/state.py, ponderer.py, premover.py) rather than
# depth-capped -- deliberately, to keep the bot's own move latency low. That
# cap is left alone here. What's untouched until now is *how much search*
# happens inside it: engine.stockfish_engine / ponder_stockfish_engine were
# never configured past UCI defaults (Threads=1, Hash=16MB). Raising these
# doesn't add any latency -- the wall-clock cap is unchanged -- it just lets
# more of the search actually happen within it (Stockfish's lazy-SMP scales
# effective node count with threads for the same wall time). Free strength
# with zero human-likeness footprint: it doesn't touch any modeled-behaviour
# constant, so it can't make the bot look less human, only make its
# within-budget evals less noisy.
# Kept modest: a live session runs one Engine instance (2 Stockfish
# processes configured this way), but simulation runs several worker
# processes each with two Engine instances -- Threads=2 there means real
# thread contention across workers, which would blunt (not reverse) the
# benefit rather than help further, so this isn't scaled up aggressively.
STOCKFISH_THREADS = 2
STOCKFISH_HASH_MB = 128


# ---------------------------------------------------------------------------
# 8. Re-evaluation ordering (Engine.get_human_move)
# ---------------------------------------------------------------------------
# Which root moves get the deeper second look, when there is not time for all
# of them. Moves that miss out are left at depth_considered 0 and take a ~60cp
# penalty (DEPTH_PENALTY x2 + ZERO_DEPTH_PENALTY), which in a quiet position
# is far larger than the real eval spread between candidates -- so missing out
# is effectively disqualification.
#
# Measured on 115 real positions: the engine picks 8.46 root moves in quiet
# positions but re-evaluates only 6.18, so ~2.3 candidates per position are
# knocked out, and the lottery is active in 63.2% of quiet positions (vs 30%
# of sharp ones, where the wider time budget covers everything).
#
#   "random" -- uniform random.sample, the shipped behaviour. Blind to both
#               move quality and human plausibility, so it disqualifies the
#               engine's best move about a quarter of the time it is present.
#   "human"  -- the most human-plausible candidates first (NN order). What a
#               human actually does: you calculate your most natural moves,
#               not a random subset.
#   "eval"   -- the highest-eval candidates first. Upper bound on the t1 gain,
#               but it also makes the bot *choose better*, which is the wrong
#               direction: the bot is already superhuman on error avoidance
#               (see the strength-dial spec's structural obstruction).
#
# Ships "random" so the parity harness stays green; the others are here to be
# measured.
REEVAL_ORDER = "random"
REEVAL_ORDERS = ("random", "human", "eval")
