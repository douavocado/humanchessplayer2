# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:33:26 2024

@author: xusem
"""
import math

from common.platform_compat import engine_path

# Resolved per-platform out of Engines/ -- ELF `-ubuntu` builds on Linux,
# `.exe` on Windows. See common/platform_compat.py:engine_path for the
# candidate names each platform looks for.
PATH_TO_STOCKFISH = engine_path("stockfish17")
PATH_TO_PONDER_STOCKFISH = engine_path("stockfish16")

# Small, hand-curated polyglot book of one account's real recurring opening
# habits (see assets/data/Opening_books/build_repertoire.py), consulted
# before the broad OPENING_BOOK_PATH -- see engine_components/opening_book.py.
OPENING_REPERTOIRE_PATH = "assets/data/Opening_books/repertoire.bin"

MOVE_FROM_WEIGHTS_OP_PTH = 'models/model_weights/piece_selector_opening_weights.pth'
MOVE_FROM_WEIGHTS_MID_PTH = 'models/model_weights/piece_selector_midgame_weights.pth'
MOVE_FROM_WEIGHTS_END_PTH = 'models/model_weights/piece_selector_endgame_weights.pth'
MOVE_FROM_WEIGHTS_TACTICS_PTH = 'models/model_weights/piece_selector_defensive_tactics_weights.pth'
MOVE_TO_WEIGHTS_MID_PTH = 'models/model_weights/piece_to_midgame_weights.pth'
MOVE_TO_WEIGHTS_END_PTH = 'models/model_weights/piece_to_endgame_weights.pth'
MOVE_TO_WEIGHTS_OP_PTH = 'models/model_weights/piece_to_opening_weights.pth'
MOVE_TO_WEIGHTS_TACTICS_PTH = 'models/model_weights/piece_to_defensive_tactics_weights.pth'
ALTER_MOVE_PROB_WEIGHTS_PTH = 'models/model_weights/alter_move_prob_nn_best.pth'

# Alter move probability model constants
WEIRD_MOVE_SD_DIC = {"opening" : 0.01,
                     "midgame" : 0.01,
                     "endgame" : 0.01,
                     }
LOWER_THRESH_SF = math.exp(-1.0406)
PROTECT_KING_SF = 0.7614
CAPTURE_EN_PRIS_SF = 0.7933
BREAK_PIN_SF = 1.2306
CAPTURE_SF = 1.1687
CAPTURE_SF_KING_DANGER = 1.1475
CAPTURABLE_SF = 0.8171
CHECK_SF_DIC = {"confident": 1.2022,
                "cocky": 1.4,
                "cautious": 1.1,
                "tilted": 1.6,
                "hurry": 1.9,
                "flagging": 1.6}
TAKEBACK_SF = 2.6469
# Boost for moves that deal with an opponent pawn ONE STEP from promoting
# (block/capture/cover the promotion square so a promoted piece cannot
# survive). A human sees a pawn on the 2nd rank like a gun on the table,
# but the NN under-ranks quiet king moves toward it and the weird-move
# penalty then crushes them -- a live game (2026-07-14) lost a won
# endgame because Ke2 (the only non-losing move, stopping d1=Q) never
# made the 8-move root set and g3 was played instead. Applied with the
# visibility floor, like takebacks/strengthening moves.
PROMOTION_STOP_SF = 3.5
# Boost for moves that blockade or newly attack a square on the promotion
# path of an opponent passed pawn that has already reached the 6th rank (3rd
# for a black pawn) -- a serious long-range threat well before it's one push
# from queening (PROMOTION_STOP_SF above only fires that late). A live game
# (2026-08-02) played Rxb2 -- grabbing a free pawn -- against a monster a6
# passer while Ra8 (blockading a8) and Nb6 (covering a8) sat at NN policy
# ranks #18 and #4 and never got root-evaluated at all. Smaller than
# PROMOTION_STOP_SF: the pawn still needs two pushes, not one, so it's a
# real concern rather than the loudest thing on the board.
ADVANCED_PASSED_PAWN_DEFENCE_SF = 2.2

# Least material a double attack must win before board_information counts it
# as a fork at all. Below roughly an exchange it isn't the kind of threat a
# human reorganises their move around, and counting it would fire the boost
# below on near-every position.
FORK_MIN_GAIN = 1.5

# Boost for moves that answer an opponent fork threat by moving the forked
# piece out of the double attack or covering the square the fork lands on
# (common.board_information.defends_against_fork). Applied to every defence,
# including ones the net already ranks highly: an earlier version skipped
# those, which made it impossible to promote a genuine defence past a
# higher-ranked non-defence -- see the block in
# models/alter_move_prob_nn.py. Diagnosed from a live game (2026-08-02) that
# met Nb5 with Nc6 and lost the exchange to Nc7+, while Na6 -- covering c7 --
# sat at NN policy rank #10 on 0.35%. At this value that position plays Na6
# in 9 of 10 seeds (mean eval -277 -> -104); 3.5 measured identically, so
# this is the low end of the plateau rather than the edge of an effect.
#
# Human-likeness caveat: forks are exactly what human players miss, so this
# is a strength-increasing knob. What keeps the bot fallible is the narrow
# defence definition (blocking the forker's route, defending the second
# target and counter-threats are all missed by design), not this constant.
# Not calibrated against cheat_detection yet.
FORK_DEFENCE_SF = 2.5
NEW_THREATENED_SF_DIC = {"confident":1.6287,
                        "cocky": 1.3,
                        "cautious": 1.8,
                        "tilted": 1.1,
                        "hurry": 1.3,
                        "flagging": 1.5}
EXCHANGE_SF_DIC = {"confident": 1.4049,
                        "cocky": 1.1,
                        "cautious": 1.2,
                        "tilted": 1.0,
                        "hurry": 1.7,
                        "flagging": 0.8}
EXCHANGE_K_DANGER_SF_DIC = {"confident": 1.1235,
                        "cocky": 0.9,
                        "cautious": 1.3,
                        "tilted": 1.0,
                        "hurry": 1.1,
                        "flagging": 1.0}
PASSED_PAWN_END_SF = 3.3780
SOLO_FACTOR_SF = 1.3324
THREATENED_LVL_DIFF_SF = 0.4485

# Noise penalties
DEPTH_PENALTY = 15
ZERO_DEPTH_PENALTY = 30
CAPTURE_BONUS = 10


QUICKNESS = 2.5 # adjust depending on computer fastness. The bigger the number the slower the moves made
# Per-game pace variation: at each game boundary the engine samples one
# multiplier (mean-preserving lognormal, sigma below, clipped to the range)
# and applies it to every think time that game (engine.py:_sample_game_pace).
# Humans' average pace swings game to game (mood, opponent, focus); without
# this the bot's per-game mean move time is unnaturally consistent. 0 disables.
GAME_PACE_SIGMA = 0.36
GAME_PACE_CLIP = (0.5, 2.0)
# Mean bias on the pace draw. 0.90 for the 2500-2800 comparison band:
# that population averages 1.16s/move vs the bot's 1.26s (z +0.3..+0.4
# across four 300-game runs) -- stronger bullet players simply move
# faster. The realised trim is smaller than x0.90 because the engine
# compute floor bounds short think times; also nudges near-boundary moves
# under the 1s instant cutoff and lifts movetime_cv toward the human mean.
GAME_PACE_MEAN = 0.90
# Same idea for premove appetite: one multiplier per game scaling every
# premove-search probability in make_move (full and takeback-only alike).
# Some games the bot premoves everything, some games barely at all --
# humans' premove usage swings similarly with mood/opponent. 0 disables.
# Premoves are the main source of 0-second moves (a decided think time can't
# beat the engine-compute floor), so this distribution's mean and spread set
# the instant-move rate and its game-to-game variance. Because the sf
# multiplies *probabilities*, the top of the draw saturates (a 3.0x game
# can't premove more than "always") -- the realized game-to-game spread
# comes mostly from the low side, hence the near-zero bottom clip (a game
# that barely premoves at all). Widened from (0.55, (0.3, 2.4)): the bot's
# per-game instant-move-rate std was 0.69x the human baseline's and, unlike
# most variance features, that one is also under-dispersed relative to
# innocent *single accounts* (see cheat_detection single-account
# calibration, 2026-07-12).
GAME_PREMOVE_SIGMA = 0.8
GAME_PREMOVE_CLIP = (0.1, 3.0)
# Mean bias on the premove draw. The lognormal is mean-preserving in the
# *multiplier*, but premove probabilities saturate at the top while the low
# tail genuinely removes premoves, so a mean-1.0 draw *lowers* the realised
# instant-move rate (measured -1.7pp when the spread was widened). This
# bias restores the mean without narrowing the spread. (Tried 1.40 for the
# 2500-2800 band's higher instant rate: it re-broke the TP blunder rate the
# v5 vetos had fixed -- the extra propensity lands in the hurry-mood <20s
# premove channel, which fires blind. Lift instants via ponder coverage /
# snap mean instead; those act above 10s.)
GAME_PREMOVE_MEAN = 1.25
# Per-game scale on the ponder-response wait only (move_timing.
# ponder_response_wait): ponder hits are ~30% of moves and sit at the
# 1-second instant-move boundary, so this widens the instant-move rate's
# game-to-game spread through the second fast-path channel. Kept separate
# from GAME_PACE_SIGMA deliberately: general think-time spread
# (movetime_mean std ratio) is already at innocent-account levels, and
# widening pace itself would push it past them. 0 disables.
GAME_PONDER_SNAP_SIGMA = 0.7
GAME_PONDER_SNAP_CLIP = (0.3, 2.6)
# Per-game scale on the <10s stale-ponder fire probability (the
# "(30-t)/50" scramble branch in both clients), driven by the same
# snappiness latent as the premove/ponder draws: humans differ hugely in
# scramble style (some fire everything, some think even at 5s -- corpus
# per-game scramble instant-rate std 0.195 vs the bot's 0.127 before
# this). exp(sigma*z), deliberately not mean-normalised: its mean ~1.13
# also lifts the scramble instant rate toward the human 0.51.
SCRAMBLE_FIRE_SF_SIGMA = 0.5
SCRAMBLE_FIRE_SF_CLIP = (0.5, 1.8)
# Mean bias < 1 (faster average ponder response): the instant-move cutoff
# is a threshold (1s integer clock), so a symmetric multiplier on the wait
# lowers P(instant) net -- bias the wait down to compensate, same reasoning
# as GAME_PREMOVE_MEAN above. (0.85 -> 0.75: the >=10s instant rate still
# sat 4.6pp under the human 0.285 after v3; 0.75 -> 0.65 for the 2500-2800
# band, whose instant rate is higher again -- 0.335 vs the bot's 0.279.)
#
# The premove and ponder-snap draws share ONE latent normal per game with
# opposite signs (a snappy game premoves more AND answers recognised
# replies faster). Independent draws half-cancel in the realised per-game
# instant rate; measured spread stayed at ~0.72x the human baseline's
# until the two channels were coupled. Marginals keep the sigmas/means
# above; only the correlation changes.
GAME_PONDER_SNAP_MEAN = 0.75
# (Tried 0.65 plus a per-game ponder-width cap draw for the 2500-2800
# band's higher instant rate: neither moved the realised instant rate --
# in bullet the ponder width is bound by the time budget in Engine.ponder,
# not the cap, and the lower snap mean only compressed instant_in_sharp
# variance. Quality scramble instants (the exact-hit fast path in the
# clients) are the working lever instead.)
# Global scale on the human-move eval noise sd (engine.py, both the
# get_stockfish_move and ponder noise sites). < 1 aligns the bot's
# non-error moves better with the engine's top choices without widening
# search breadth (no compute cost, unlike a DIFFICULTY bump). Added when
# the 2500-2800 comparison put every top-N match rate at z -0.4..-0.5
# while the error/blunder means were already on target. (0.85 measurably
# moved t1..t3 by +0.05-0.13 z with error means unchanged; 0.75 is the
# second dose -- the cluster still sat at z -0.33..-0.46.)
HUMAN_EVAL_NOISE_SCALE = 0.75
# Per-game ponder *coverage* cap (Engine.ponder max_ponder_no when no
# explicit width is passed), drawn per game off the snappiness latent:
# round(clip(BASE + SPREAD*z_snap + PRIVATE*eps, *CLIP)). A dead lever at
# the old 0.1s/position ponder cost (budget pinned realised width at ~1);
# at PONDER_TIME_PER_POSITION 0.05-0.06 the cap binds again, so a
# per-game draw modulates realised coverage -- the between-game
# ponder-hit-rate spread behind instant_move_rate variance, the one flag
# that persists at innocent-account sample size (57%). The LOW end does
# the work: a "narrow reader" game at width 1-2 gets structurally fewer
# hits; the wide end saturates against the time budget.
# Width 0 = a game that never pre-thinks (tilted/tired): the saturating
# multiplier channels can't make a strong LOW tail in per-game instant
# rate, but a ~5% no-ponder game fraction can -- that low tail is what
# the one remaining persistent variance flag (instant_move_rate) needs.
# BASE up 2.8 -> 3.0 compensates the mean for the width-0 games.
GAME_PONDER_WIDTH_BASE = 3.0
# Opening-book fast path: consult the book BEFORE calculate_analytics rather
# than after, so a memorised move stops paying for a full-width multipv scan
# plus an uncapped depth-12 sharpness scan.
#
# This is the only lever that can raise the instant-move rate in the opening.
# The per-move compute floor (screen detection + engine + mouse) means the
# engine's requested opening think time is ALREADY below it, so no pacing knob
# moves the instant rate -- Phase A varied eval_noise_scale 0.55-0.95 and
# quickness 2.0-3.0 and held instant rate within 0.0087 (per-arm se 0.0045).
# Only paths that bypass engine compute produce sub-1s moves.
#
# Ships OFF: unlike the other tuning levers here it changes behaviour by
# construction rather than being a no-op at its default, so the engine parity
# harness only stays green while this is False. Taking the fast path also
# skips check_obvious_move, which is acceptable because a real blunder takes
# the game OUT of book -- measured over 1,176 opening positions, 69.9% were
# book hits and 0 of those 822 hits had a hanging queen or rook.
OPENING_BOOK_FAST_PATH = False
GAME_PONDER_WIDTH_SPREAD = 1.4
GAME_PONDER_WIDTH_PRIVATE = 0.4
GAME_PONDER_WIDTH_CLIP = (0, 5)
# Per-game intuition gate: the probability of snapping (not deep-thinking) a
# sharp position, drawn uniformly from this range at each game boundary
# (mean 0.75). Trust-the-gut games snap ~95% of critical moves; grinding
# games stop and think on a good share of them. Widens the game-to-game
# spread of the long-think tail (move-time std). Raised from (0.45, 0.85):
# the bot's time-vs-sharpness correlation overshot the human baseline
# (~0.07 vs ~0.02) -- humans barely slow down for sharp positions in bullet.
GAME_SNAP_GATE_RANGE = (0.55, 0.95)
# Ambiguity split on that gate: sharp positions are not all alike. Humans snap
# a position with ONE right answer far more readily than one with several
# near-equal tries, and the gap widens with strength -- instant rate in
# sharp-forced positions climbs +13.2pp from 2100-2299 to 2800+ (vs +6.1pp in
# quiet), until at the top band sharp-forced is the *fastest* bucket (.464)
# while at the bottom it is indistinguishable from quiet. The bot shows the
# mirror symptom: corr_time_sharpness +0.056 vs a human +0.037, i.e. it slows
# down where strong humans speed up. These are additive offsets on the
# per-game draw above, so the trust-the-gut-vs-grinding character survives the
# split rather than being flattened by it.
#
# Both default to 0.0: the mechanism ships inert (bit-identical behaviour, no
# extra RNG draw) and the values come from a sweep, exactly as the breadth
# bonuses did. Grid and judging criteria:
# docs/superpowers/specs/2026-07-27-ambiguity-snap-gate-design.md
AMBIGUITY_FORCED_SNAP_DELTA = 0.0   # ambiguity == 1: raise the snap probability
AMBIGUITY_MESSY_SNAP_DELTA = 0.0    # ambiguity >= 2: lower it
# "Hesitation before the mistake" (engine.py:_adjust_time_for_move_loss):
# humans think longer in positions where they end up erring, giving a
# positive per-game correlation between move time and move loss (~ +0.10 in
# the 2300-2600 corpus) that the engine otherwise lacks -- its errors come
# from the human-probability sampling, independent of the decided think time.
# When the chosen move gives up at least WC_LOSS win probability, the think
# time is stretched by a Uniform(*RANGE) factor with probability PROB (the
# rest stay fast: snap blunders exist). Skipped when own clock < MIN_TIME --
# scramble errors are fast and must stay fast. The mirror side: when the
# chosen move is clean (loss <= SNAP_WC_LOSS), the think time is trimmed by
# Uniform(*SNAP_RANGE) with probability SNAP_PROB -- humans bang out moves
# they are sure of. The trim keeps the mean move time level despite the
# stretches and adds correlation from the fast side.
MISTAKE_HESITATION_WC_LOSS = 0.05
# 0.75 -> 0.88 and snap 0.6 -> 0.7 for the 2500-2800 band: its
# time-vs-loss correlation is stronger (pop mean 0.109; the bot sat at
# player_z -1.3..-1.6, the largest residual after the means centred).
MISTAKE_HESITATION_PROB = 0.88
MISTAKE_HESITATION_RANGE = (1.4, 2.6)
MISTAKE_HESITATION_MIN_TIME = 10
MISTAKE_SNAP_WC_LOSS = 0.02
MISTAKE_SNAP_PROB = 0.7
MISTAKE_SNAP_RANGE = (0.65, 0.9)
# Opponent-blunder "act startled" reaction (mood_manager.check_opp_blunder,
# decision_logic.py's opponent_just_blundered doubling of base_time). The
# absolute eval swing is only a cheap pre-filter to skip the enpris scan on
# quiet moves -- it is deliberately loose (a fixed cp threshold saturates in
# already-winning positions, where hanging a whole rook barely moves the
# score) and the real gate is EN_PRIS_MIN_VALUE: is any opponent piece,
# either just moved or merely left behind, worth at least this many pawns
# to win back in an exchange (calculate_threatened_levels).
OPP_BLUNDER_EVAL_SWING_MIN = 50
OPP_BLUNDER_EN_PRIS_MIN_VALUE = 3
# The startle reaction itself (decision_logic.py's base_time *= multiplier)
# still fires under time pressure -- a human doesn't stop being surprised
# just because their clock is low -- but the reaction is muted rather than
# skipped outright the way MISTAKE_HESITATION/SNAP are: below
# OPP_BLUNDER_STARTLE_LOW_TIME_THRESHOLD seconds of own clock, the added
# think time is roughly halved (multiplier 2.0 -> 1.5).
OPP_BLUNDER_STARTLE_MULTIPLIER = 2.0
OPP_BLUNDER_STARTLE_MULTIPLIER_LOW_TIME = 1.5
OPP_BLUNDER_STARTLE_LOW_TIME_THRESHOLD = 10
# Per-move clock budget (decision_logic.move_time_budget / move_time_cap).
# The only ceiling on think time used to be `own_time*0.7 + 1`, which with a
# full 3+0 clock permits a 127-second move: every multiplier downstream of the
# phase envelope (the mood long-think tail, reflective pacing, the rating
# factor) could compound unchecked and the clamp never bound until the clock
# was nearly gone. The budget is instead the even share of the remaining clock
# over the moves we still expect to play, and one move may overspend it by
# BUDGET_MAX_MULTIPLE. A long think is still a long think; it just cannot eat
# a whole phase of the game. TOTAL_MOVES is the assumed game length in full
# moves (the 180+0 self-play arm averaged 98.6 plies), and MIN_MOVES_LEFT
# keeps the budget from collapsing once a game runs past it. The same budget
# gates the reflective-pacing blend: matching a slow opponent's tempo is a
# luxury only affordable while we are inside it.
#
# MAX_MULTIPLE is set so the cap touches the extreme tail only. Replayed over
# every decided time in logs/sessions (909 moves at 60+0, 868 at 180+0), 4.0
# leaves p95 move time unmoved at both controls (3.1s -> 2.9s and 8.9s ->
# 8.9s) and costs 6% / 4% of mean emt, while the single worst move falls from
# 38.8s to 19.3s and the worst game gives back 55s of clock over 18 moves.
CLOCK_BUDGET_TOTAL_MOVES = 50
CLOCK_BUDGET_MIN_MOVES_LEFT = 15
CLOCK_BUDGET_MAX_MULTIPLE = 4.0
# Flag-race autopilot (engine.py:get_stockfish_move): in a deep scramble a
# human does not distinguish "+mate" from "+800" -- both read as "winning" --
# and plays on instinct: shuffling, missing mates, occasionally stalemating
# or throwing the win outright. Without this the bot's endgame ACPL tail is
# unhumanly thin (humans: ~20% of games contain a 300+ acpl endgame, tail to
# 9000+ from thrown mates; the bot's safe fast paths produce almost none).
# Below FLAG_RACE_TIME seconds, evals are capped in the move-appeal
# formula, so among winning moves the choice is driven by mouse distance
# and noise, exactly like a human flag race.
#
# The cap is per-game, not constant: a flat cap made the bot blunder too
# often but too gently and too evenly across games (TP blunder rate mean
# 0.134 vs human 0.082, over-dispersed 1.24x, while the TP/endgame ACPL
# tails stayed thin -- humans blunder *less often but occasionally
# catastrophically*, and unevenly game to game). Each game draws
# game_scramble_skill u ~ Uniform(0,1) (engine.py:_sample_game_character):
#   eval cap        = CAP_MIN + u * (CAP_MAX - CAP_MIN)   (mean ~700)
#   blind-move prob = BLIND_P_MAX * (1-u)^2 per scramble decision
# A "blind move" drops the eval term entirely for that decision -- pure
# hand-distance + noise -- which is what hangs a mate or throws a won
# ending and creates the human 3000+ ACPL catastrophe tail that capped
# evals structurally cannot. The (1-u)^2 shaping concentrates disasters in
# a minority of games. Costs real win rate in flag races - that's the
# point. FLAG_RACE_EVAL_CAP is the fallback before the first per-game draw.
FLAG_RACE_TIME = 10
FLAG_RACE_EVAL_CAP = 450
# CAP_MIN raised from 300 after v1 validation: a floor below the old flat
# 450 made low-skill games sloppier than before *on top of* their blind
# moves, raising the TP blunder-rate mean and over-dispersion instead of
# lowering them. With a higher floor the frequent-medium-error channel
# shrinks and the blind moves alone carry the catastrophe tail.
FLAG_RACE_CAP_MIN = 550
FLAG_RACE_CAP_MAX = 1100
FLAG_RACE_BLIND_P_MAX = 0.10
# How strongly the scramble safety vetos (scramble_fire_veto in the
# clients, check_safe_premove in get_premove's flag-race branch) apply:
# p = BASE + RANGE * game_scramble_skill. Tuning history: absolute vetos
# (p=1) cut TP blunders to 0.112 but collapsed the ACPL body/tail
# (endgame std ratio 0.37x); a full 0..1 skill gate restored the tail but
# re-opened the hang-fire channel (TP blunder mean 0.136, dispersion
# 1.41x). Mostly-on with mild skill leak is the calibrated middle.
SCRAMBLE_VETO_P_BASE = 0.75
SCRAMBLE_VETO_P_RANGE = 0.25
# Conversion progress in the scramble move-appeal formula
# (stockfish_move_logic.get_stockfish_move, via
# board_information.move_progress_score). In a decided position the eval
# term of that formula carries almost no information: every candidate is
# "winning", so the spread across all legal moves collapses (measured on a
# rook-up 7-piece ending: 0.31 appeal points uncapped, exactly 0.00 once
# the flag-race cap bites, and a deeper scan doesn't help -- even depth 16
# spreads those moves by under 100cp because no single move converts).
# Hand distance spreads ~3.0 over the same moves, so the ranking became
# "shortest mouse travel", and since re-moving the piece you just moved
# costs zero linkage distance, the cheapest legal option is to put it
# straight back -- a shuffle attractor. Real humans scrambling in a won
# ending are fast and error-prone but *directional*: they push the passer,
# eat the pawn. These terms restore that direction without touching the
# flag-race catastrophe tail (they apply on blind moves too -- a human
# following a memorised plan without checking it is safe is exactly how a
# won ending gets thrown).
#
# MEASURED_PLACEHOLDER
#
# Only applied when clearly winning (the best candidate is at least
# PROGRESS_MIN_EVAL): repeating and shuffling are legitimate human play
# when holding a worse position or angling for a draw.
PROGRESS_MIN_EVAL = 150
# How far below the best *perceived* eval (i.e. after the flag-race cap, or
# nothing at all on a blind move) a move may sit and still be offered the
# plan bonus. Keeps the plan a tie-breaker among moves that all read as
# winning rather than something that overrules a visible drop.
PROGRESS_EVAL_TOLERANCE = 100
# Sized against that ~3.0 hand-distance spread: penalties have to be able
# to outweigh it outright, bonuses only have to break ties among moves of
# similar travel cost.
PROGRESS_REPEAT_PENALTY = 4.0
PROGRESS_UNDO_PENALTY = 2.5
# Captures scale with what is taken (pawn 0.4, knight 1.2, rook 2.0).
PROGRESS_CAPTURE_BONUS = 2.0
PROGRESS_PAWN_PUSH_BONUS = 1.5
PROGRESS_PASSED_PAWN_BONUS = 1.5
PROGRESS_PROMOTION_BONUS = 5.0
# Per rank beyond the 4th that the pushed pawn ends on.
PROGRESS_PAWN_ADVANCE_COEFF = 0.4
# No king-march term: measured and rejected, see move_progress_score's
# docstring in common/board_information.py.
DIFFICULTY = 3 # engine difficulty
MOUSE_QUICKNESS = 4 # number between 0 and 10. Bigger the number the slower we are with mouse movements
RESOLUTION_SCALE = 2.0  # Set to 2.0 for 4K, 1.0 for 1080p - adjusts mouse curve point density

# BENCHMARKS
"""
DIFFICULTY   |   QUICKNESS   |    ELO
    3        |     2.2      |  ~2500


"""
