"""Per-game character sampling, extracted verbatim from engine.Engine.

sample_game_character draws the multipliers/gates that vary *between*
games what per-move noise can't (pace, premove propensity, snap gate,
ponder-snap, scramble fire/skill, ponder width) -- see the docstring
below for the coupling rationale (several draws share one latent so
independent noise doesn't half-cancel the realised instant-move rate).
scramble_veto_p and ponder_pace_sf stay Engine properties (read
externally by clients/mp_original.py and simulation/client_model.py as
engine.scramble_veto_p / engine.ponder_pace_sf) but delegate here.

Seventh slice of the engine.py strangler-fig extraction (see
testing/engine_parity/ for the regression harness) -- this is exactly
the kind of cross-cutting per-game state that made a naive component
split awkward last time (one latent feeding multiple components with
correlated signs). Verbatim move (self -> engine), no interface redesign.
"""
import numpy as np

from common.constants import (
    GAME_PACE_SIGMA, GAME_PACE_CLIP, GAME_PACE_MEAN,
    GAME_PREMOVE_SIGMA, GAME_PREMOVE_CLIP, GAME_SNAP_GATE_RANGE,
    GAME_PREMOVE_MEAN, GAME_PONDER_SNAP_MEAN,
    GAME_PONDER_SNAP_SIGMA, GAME_PONDER_SNAP_CLIP,
    SCRAMBLE_FIRE_SF_SIGMA, SCRAMBLE_FIRE_SF_CLIP,
    GAME_PONDER_WIDTH_BASE, GAME_PONDER_WIDTH_SPREAD,
    GAME_PONDER_WIDTH_PRIVATE, GAME_PONDER_WIDTH_CLIP,
    SCRAMBLE_VETO_P_BASE, SCRAMBLE_VETO_P_RANGE,
)


def lognormal_sf(sigma, clip, mean=1.0):
    """ One lognormal draw with the given mean (median slightly below
        it), clipped. sigma <= 0 disables (returns exactly `mean`). """
    if sigma <= 0:
        return float(mean)
    return float(np.clip(mean * np.random.lognormal(-sigma**2 / 2, sigma), *clip))


def sample_game_character(engine):
    """ Draws this game's character multipliers -- game-to-game variation
        that per-move noise cannot produce:

        - game_pace_sf: applied to think times in _get_time_taken, so one
          game is played uniformly faster or slower than another.
        - game_premove_sf: applied to the premove-search probabilities in
          make_move, so one game premoves eagerly and another rarely.
        - game_snap_gate: the intuition-gate probability in
          _get_time_taken, so one game snaps most sharp positions on gut
          feel and another stops to think on most of them.
        - game_ponder_snap_sf: applied (with pace) to the ponder-response
          wait only, so one game bangs out recognised replies and another
          double-checks them -- the second channel of the instant-move
          rate's game-to-game spread.
        - game_scramble_skill: this game's flag-race composure, setting
          the scramble eval cap and blind-move probability in
          get_stockfish_move; one game scrambles cleanly, another throws
          a won ending.
    """
    engine.game_pace_sf = lognormal_sf(GAME_PACE_SIGMA, GAME_PACE_CLIP,
                                           mean=GAME_PACE_MEAN)
    # One latent snappiness draw drives both fast-path channels with
    # opposite signs (premove propensity up <=> ponder wait down):
    # independent draws half-cancel in the realised per-game instant
    # rate, coupling them makes the spreads add. Marginals are the same
    # lognormals as before (mean * exp(sigma*z - sigma^2/2)).
    z_snap = float(np.random.randn())
    engine.game_premove_sf = float(np.clip(
        GAME_PREMOVE_MEAN * np.exp(GAME_PREMOVE_SIGMA * z_snap - GAME_PREMOVE_SIGMA**2 / 2),
        *GAME_PREMOVE_CLIP))
    engine.game_ponder_snap_sf = float(np.clip(
        GAME_PONDER_SNAP_MEAN * np.exp(-GAME_PONDER_SNAP_SIGMA * z_snap - GAME_PONDER_SNAP_SIGMA**2 / 2),
        *GAME_PONDER_SNAP_CLIP))
    # Same latent, positive sign, deliberately un-normalised mean (~1.13):
    # a snappy game also fires more stale ponder moves in the scramble.
    engine.game_scramble_fire_sf = float(np.clip(
        np.exp(SCRAMBLE_FIRE_SF_SIGMA * z_snap), *SCRAMBLE_FIRE_SF_CLIP))
    engine.game_snap_gate = float(np.random.uniform(*GAME_SNAP_GATE_RANGE))
    engine.game_scramble_skill = float(np.random.uniform(0, 1))
    # Ponder coverage rides the same snappiness latent: how many
    # opponent replies this game prepares for. Only effective now that
    # PONDER_TIME_PER_POSITION lets the budget fill widths > 1 (see
    # constants) -- the low end (width 1-2 games) carries the
    # between-game instant-rate spread.
    engine.game_ponder_width = int(round(np.clip(
        GAME_PONDER_WIDTH_BASE + GAME_PONDER_WIDTH_SPREAD * z_snap
        + GAME_PONDER_WIDTH_PRIVATE * float(np.random.randn()),
        *GAME_PONDER_WIDTH_CLIP)))
    engine.log += "Sampled per-game character: pace {:.3f}, premove propensity {:.3f}, snap gate {:.3f}, ponder snap {:.3f}, scramble fire {:.3f}, scramble skill {:.3f}, ponder width {} \n".format(
        engine.game_pace_sf, engine.game_premove_sf, engine.game_snap_gate,
        engine.game_ponder_snap_sf, engine.game_scramble_fire_sf, engine.game_scramble_skill,
        engine.game_ponder_width)
    print(f"[ENGINE] Sampled per-game character: pace {engine.game_pace_sf:.3f}, "
          f"premove propensity {engine.game_premove_sf:.3f}, snap gate {engine.game_snap_gate:.3f}, "
          f"ponder snap {engine.game_ponder_snap_sf:.3f}, scramble fire {engine.game_scramble_fire_sf:.3f}, "
          f"scramble skill {engine.game_scramble_skill:.3f}")


def scramble_veto_p(engine):
    """ Probability the scramble safety vetos apply this game (see
        SCRAMBLE_VETO_P_* in constants); 1.0 before the first draw.
        Clients read this so live and sim can't drift. """
    if engine.game_scramble_skill is None:
        return 1.0
    return SCRAMBLE_VETO_P_BASE + SCRAMBLE_VETO_P_RANGE * engine.game_scramble_skill


def ponder_pace_sf(engine):
    """ Combined per-game scale for the ponder-response wait (pace x
        ponder-snap), 1.0 before the first per-game draw. Clients and the
        simulator both read this so the coupling can't drift. """
    return (engine.game_pace_sf or 1.0) * (engine.game_ponder_snap_sf or 1.0)
