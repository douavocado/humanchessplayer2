# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:33:26 2024

@author: xusem
"""
import math

PATH_TO_STOCKFISH = "Engines/stockfish17-ubuntu"
PATH_TO_PONDER_STOCKFISH = "Engines/stockfish16-ubuntu"

MOVE_FROM_WEIGHTS_OP_PTH = 'models/model_weights/piece_selector_opening_weights.pth'
MOVE_FROM_WEIGHTS_MID_PTH = 'models/model_weights/piece_selector_midgame_weights.pth'
MOVE_FROM_WEIGHTS_END_PTH = 'models/model_weights/piece_selector_endgame_weights.pth'
MOVE_FROM_WEIGHTS_TACTICS_PTH = 'models/model_weights/piece_selector_defensive_tactics_weights.pth'
MOVE_TO_WEIGHTS_MID_PTH = 'models/model_weights/piece_to_midgame_weights.pth'
MOVE_TO_WEIGHTS_END_PTH = 'models/model_weights/piece_to_endgame_weights.pth'
MOVE_TO_WEIGHTS_OP_PTH = 'models/model_weights/piece_to_opening_weights.pth'
MOVE_TO_WEIGHTS_TACTICS_PTH = 'models/model_weights/piece_to_defensive_tactics_weights.pth'

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
# bias restores the mean without narrowing the spread.
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
# sat 4.6pp under the human 0.285 after v3.)
#
# The premove and ponder-snap draws share ONE latent normal per game with
# opposite signs (a snappy game premoves more AND answers recognised
# replies faster). Independent draws half-cancel in the realised per-game
# instant rate; measured spread stayed at ~0.72x the human baseline's
# until the two channels were coupled. Marginals keep the sigmas/means
# above; only the correlation changes.
GAME_PONDER_SNAP_MEAN = 0.75
# Per-game intuition gate: the probability of snapping (not deep-thinking) a
# sharp position, drawn uniformly from this range at each game boundary
# (mean 0.75). Trust-the-gut games snap ~95% of critical moves; grinding
# games stop and think on a good share of them. Widens the game-to-game
# spread of the long-think tail (move-time std). Raised from (0.45, 0.85):
# the bot's time-vs-sharpness correlation overshot the human baseline
# (~0.07 vs ~0.02) -- humans barely slow down for sharp positions in bullet.
GAME_SNAP_GATE_RANGE = (0.55, 0.95)
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
MISTAKE_HESITATION_PROB = 0.75
MISTAKE_HESITATION_RANGE = (1.4, 2.6)
MISTAKE_HESITATION_MIN_TIME = 10
MISTAKE_SNAP_WC_LOSS = 0.02
MISTAKE_SNAP_PROB = 0.6
MISTAKE_SNAP_RANGE = (0.65, 0.9)
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
DIFFICULTY = 3 # engine difficulty
MOUSE_QUICKNESS = 4 # number between 0 and 10. Bigger the number the slower we are with mouse movements
RESOLUTION_SCALE = 2.0  # Set to 2.0 for 4K, 1.0 for 1080p - adjusts mouse curve point density

# BENCHMARKS
"""
DIFFICULTY   |   QUICKNESS   |    ELO
    3        |     2.2      |  ~2500


"""
