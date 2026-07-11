# cheat_detection — human-likeness diagnostic

An **offline** self-diagnostic that applies the *techniques* of Lichess's
open-source cheat detectors ([Irwin](https://github.com/clarkerubber/irwin)'s
per-move engine analysis, [Kaladin](https://github.com/lichess-org/kaladin)'s
aggregate statistics) to one question:

> How far does this bot's play sit from a real human distribution, and on which
> axes is it least human-like?

It is **comparison-based and descriptive**. It measures the bot against a human
reference corpus feature-by-feature and reports the divergence. It does not
score against, or target, any particular detector — the output is "here is where
your move-quality / timing profile differs from humans," which is a research and
tuning signal, not an evasion checklist.

## What it measures

Per move (Irwin-style, from Stockfish multi-PV analysis):
- **Move rank** among the engine's candidates; top-1/2/3 match rates.
- **Centipawn loss** and **winning-chance loss** vs. the best move.
- **Ambiguity** — how many near-equal good moves the position offered.
- **Sharpness** — win-probability spread across candidates (position criticality).

Aggregated per player (Kaladin-style distributions):
- **T1/T2/T3 engine-match rates** — the classic accuracy signal.
- **ACPL** overall and split by opening / middlegame / endgame.
- **Blunder rate**, mean win-chance loss.
- **Move-time** mean, std, coefficient of variation.
- **Instant-move rate**, and instant moves *in sharp positions* (a strong tell).
- **Correlation of move time with move-loss and with position sharpness** — humans
  think longer before hard/critical moves; a flat correlation is un-human.

See `features.py:FEATURE_KEYS` for the full, stable list.

## Inputs

- **The bot's games**: any PGN with `[%clk ...]` tags (as Lichess exports, and as
  the bot itself can log). Clocks are needed for the timing features.
- **A human baseline**: a [Lichess database dump](https://database.lichess.org)
  (standard multi-game PGN with `WhiteElo`/`BlackElo` and clocks). Filter to a
  rating band and time control matching the bot.

## Getting a human corpus

Download a monthly dump from https://database.lichess.org/#standard_games
(`.pgn.zst`). These are large (tens of GB) and mix every rating and time
control, so stream them through `fetch_corpus` to keep only games matching your
bot's band — no need to decompress to disk:

```bash
zstd -dc lichess_db_standard_rated_2024-01.pgn.zst | \
    venv/bin/python -m cheat_detection.fetch_corpus \
        --category blitz --rating 1800 2100 --max-games 4000 \
        --out cheat_detection/corpora/blitz_1800_2100.pgn
```

`--category` is one of bullet/blitz/rapid/classical (classified by
`base + 40*increment`, as Lichess does); or set `--base-min/--base-max`
explicitly. Prefer `--tc 60+0` to pin one **exact** clock — a whole category
mixes pacing (people move roughly twice as fast in 30+0 as in 60+0), which
muddies every timing feature. Both players' Elo must fall in `--rating`. Games
without clocks are dropped by default (the timing features need them).

For a smaller, targeted baseline of *specific* strong human players instead of a
random sample, the Lichess API also exports a user's games with clocks:
`https://lichess.org/api/games/user/<name>?clocks=true&rated=true&perfType=blitz`.

## Usage

Everything runs from the repo root with the repo venv.

### Quickest: the GUI

```bash
venv/bin/python -m cheat_detection.gui
```

Enter the bot's Lichess account, rating band and time control, point at a
baseline (or a corpus PGN to build one), and hit **Run diagnostic**. Analysis
runs in a background thread; progress streams to the log and the result renders
as a z-score chart plus a sortable feature table. Click any feature row to
drill down into its distribution — a histogram of human per-game values overlaid
with the bot's per-game values, human mean and ±2σ marked (**⟵ Overview**
returns to the z-score chart). **Open report JSON...** views a previously saved
report instantly (no recompute), including the drill-downs.

### One command: `run`

Fetch the bot's games, ensure a baseline exists (load or build), and report — in
a single call. This is what the GUI drives under the hood:

```bash
venv/bin/python -m cheat_detection.analyze run \
    --user JXu2019 --rating 2300 2600 --perf bullet \
    --baseline cheat_detection/baselines/bullet_1plus0_2300_2600.json \
    --corpus   cheat_detection/corpora/bullet_1plus0_2300_2600.pgn \
    --max-games 300 --baseline-max-games 250 \
    --out-md report.md
```

If `--baseline` exists it's loaded; otherwise it's built from `--corpus` (the
slow step) and saved. Pass `--pgn <file>` to skip the Lichess fetch and use a
local PGN. Fetched games and outputs go under `cheat_detection/runs/`.

### Just download a user's games

```bash
venv/bin/python -m cheat_detection.analyze fetch-games \
    --user JXu2019 --perf bullet --tc 60+0 --max-games 300 --out bot.pgn
```

`--tc` keeps only games at that exact clock. The Lichess API can only filter by
speed category server-side, so up to `--max-games` games of `--perf` are
downloaded and non-matching clocks discarded locally — if fewer survive than
you wanted, raise `--max-games`. The same flag exists on `run` (and as the
"exact clock" field in the GUI, default `60+0` to match the shipped baseline);
there the fetched file is cached per-clock as `runs/bot_<user>_<tc>.pgn`.

(Set `LICHESS_TOKEN` in the environment to raise the API rate limit.)

### Manual steps

The `baseline` and `report` subcommands remain for fine control.

### 1. Build a human baseline (once per rating band)

```bash
venv/bin/python -m cheat_detection.analyze baseline \
    --pgn human_dump.pgn --rating 1800 2100 \
    --out cheat_detection/baselines/blitz_1800_2100.json \
    --max-games 3000
```

Only players whose Elo falls in `--rating` are included. `--max-games` caps how
much of the dump is analysed (analysis is the slow part). The baseline JSON
stores the mean/std of every feature across human "units" (one unit = one
player's moves in one game), plus the raw per-unit values (used by the GUI's
distribution drill-down; baselines built before this field exists still load,
falling back to a normal-fit curve).

### 2. Report the bot against that baseline

```bash
venv/bin/python -m cheat_detection.analyze report \
    --pgn my_bot_games.pgn --player my_bot_account \
    --baseline cheat_detection/baselines/blitz_1800_2100.json \
    --out-md report.md --out-json report.json
```

`--player` names the account(s) whose moves to profile (omit to profile every
player in the file). The report shows, per feature, the human mean±std, the
bot's value, and the z-score, and calls out the biggest divergences with a
plain-English explanation of each.

### Flagging mode: effect size vs. Welch t-test

Two statistics are always computed and shown per feature; `--test` (CLI) or the
**Test** dropdown (GUI) picks which one decides the ⚑ flags:

- **`effect-size`** (default): flag when
  `|z| = |bot_mean − human_mean| / human_std ≥ 2`. Sample-size independent —
  it asks "is the bot's average far outside normal human game-to-game
  variation?" A small systematic bias never flags, no matter how many games.
- **`welch`**: Welch two-sample t-test (bot games vs. baseline games, unequal
  variances); flag when the two-sided p-value < `--alpha` (default 0.05, the
  **α** field in the GUI). Sensitivity grows with sample size, so with enough
  games even small-but-real systematic differences flag. Needs ≥2 games on
  each side and a baseline that stores per-feature `n` (all do).

The Welch test treats each game as an independent observation from both
populations, so with hundreds of games expect it to flag more features than
effect-size does — statistically detectable ≠ practically distinguishable.

## Configuration

Defaults live in `config.py` (`AnalysisConfig`): Stockfish **depth 10**,
**multipv 5**, using the repo's `PATH_TO_STOCKFISH` (Stockfish 17). Override per
run with `--depth`, `--multipv`, `--threads`, `--hash`, `--min-moves`.

Depth 10 was chosen after benchmarking (~24× faster than depth 18 with only
modest shifts in baseline stats). If you change depth, rebuild the baseline at
the same depth — comparing bot games analysed at one depth against a baseline
built at another skews every engine-derived feature.

## Performance & caching

Engine analysis dominates runtime. Every analysed position is cached to
`cheat_detection/cache/analysis_d<depth>_mpv<multipv>.json`, keyed by FEN, so:
- re-running a report reuses cached evals instantly (no engine needed for hits);
- the bot's and the humans' games share the cache when positions overlap;
- interrupted runs resume where they left off.

Delete the cache file to force fresh analysis (e.g. after changing depth — though
the depth/multipv are already in the filename, so different settings don't clash).

## Layout

| File | Role |
|---|---|
| `config.py` | tunables (`AnalysisConfig`) |
| `gui.py` | tkinter GUI (run diagnostics, chart + table view) |
| `orchestrate.py` | one-call pipeline: fetch → ensure baseline → report |
| `fetch_lichess.py` | download a Lichess user's games via the API |
| `fetch_corpus.py` | filter a Lichess dump to a rating band + time control |
| `pgn_loader.py` | PGN → per-move records with elapsed move times |
| `engine_analysis.py` | Stockfish multi-PV with on-disk cache |
| `features.py` | per-move + aggregate feature extraction |
| `pipeline.py` | PGN → analysis → features → per-unit vectors |
| `baseline.py` | build/load the human reference distribution |
| `report.py` | compare bot vs. baseline, render markdown/JSON |
| `analyze.py` | CLI (`baseline` and `report` subcommands) |

## Notes & limitations

- **Not** a reproduction of Lichess's actual models. Irwin's neural nets and
  Kaladin's CNN are trained on labelled Lichess data (including signals like
  window-blur events) that aren't in a PGN. This tool uses the same *feature
  ideas* but a transparent statistical comparison instead of a black-box score.
- The baseline is only as representative as the dump you feed it — match the
  rating band **and** time control to the bot for meaningful z-scores.
- A "unit" needs `--min-moves` analysed moves (default 10) to count, so very
  short games are dropped.
