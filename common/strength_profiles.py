"""The target-rating dial: rating -> engine knob vector.

Maps a requested playing strength onto the knobs that reproduce that human
rating band's *measured behaviour*, taken from
`cheat_detection/runs/elo_progression/report_timing.md` (2.28M human moves,
7 bands x 8 features). Design and derivation:
`docs/superpowers/specs/2026-07-27-strength-dial-design.md`.

Two things this module deliberately refuses to do.

**It does not offer precision that was never measured.** A 150-game arm
resolves top-1 match to ~0.005 and adjacent 100-Elo human bands differ by
0.005, so the honest resolution is ~200-300 Elo. Callers therefore get one of
`STRENGTH_LEVELS`, roughly 250 apart, and `resolve` reports which one was
actually used. Asking for 2700 and 2750 returns the same bot, and the caller
can see that from `effective_rating` rather than from this docstring.

**It does not move knobs whose rating mapping has not been fitted.** Only
`quickness` is calibrated today, from the Phase A regression
`mean_emt = 0.1964 * quickness + 0.6234` (~20 se across the probed range,
spanning the entire band table). `eval_noise_scale`, the breadth bonuses and
the instant-move channel all have owners in the control surface but no fitted
rating mapping yet, so they stay at their module defaults and are absent from
`CALIBRATED_KNOBS`. A dial that guessed at unmeasured mappings would be worse
than one that admits it currently controls pace alone.

Consequence worth stating at the call site: a bot set to 2850 plays at the
right *speed* for 2850 while its move-agreement and error rates remain those
of the shipped defaults.

**Scope: 60+0 only** -- that is 60 *seconds* (one minute of bullet, 1+0 in
standard notation), the units `simulation.run --tc` takes. The regression
above is a 60+0 fit and its intercept has no meaning at another time control;
longer controls need their own band table and their own fit.
"""

# Supported dial settings. Spaced ~250 Elo, matching the measured resolution,
# and bounded by the human band table's own extent (midpoints 2200..2850) --
# extrapolating past the data would invent behaviour no human exhibited here.
STRENGTH_LEVELS = (2200, 2450, 2700, 2850)

# Knobs this module is entitled to set. Anything absent has no fitted rating
# mapping and must be left alone; see the module docstring.
CALIBRATED_KNOBS = ("quickness",)

# Per-level knob values.
#
# quickness inverts the Phase A fit onto each level's target mean elapsed move
# time, interpolated from the human band table (2200: 1.26s, 2450: 1.19s,
# 2700: 1.065s, 2850: 1.00s). Bigger quickness = slower play, so it falls as
# rating rises.
_PROFILES = {
    2200: {"quickness": 3.241},
    2450: {"quickness": 2.885},
    2700: {"quickness": 2.248},
    2850: {"quickness": 1.918},
}


def snap_rating(rating):
    """ Nearest supported level to `rating`, clamped to the table's extent.

        Returns int, always a member of STRENGTH_LEVELS.
    """
    return min(STRENGTH_LEVELS, key=lambda lvl: (abs(lvl - rating), lvl))


def resolve(rating):
    """ Knob overrides reproducing `rating`'s measured human behaviour.

        Returns a dict of knob name -> value covering exactly
        CALIBRATED_KNOBS, plus "effective_rating": the level actually used
        after snapping, so callers can see the dial's real granularity
        instead of assuming their request was honoured verbatim.
    """
    level = snap_rating(rating)
    out = dict(_PROFILES[level])
    out["effective_rating"] = level
    return out
