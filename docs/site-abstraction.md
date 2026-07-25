# Site abstraction: separating "which site" from "which screen"

Status: **all four stages implemented.** Verification results are recorded
under each stage below.

chess.com interaction is now implemented too: resign is a single click with
no confirmation step, and the lobby sequence (Play -> Play Online ->
time-control dropdown -> control -> Start Game) is calibrated from the
find_new_game_* screenshots.

## The problem

`auto_calibration` profiles were built to answer *"where are things on this
user's screen?"* — resolution, board origin, clock boxes, piece/digit
templates, colour scheme. That is a property of the **device**.

Adding chess.com showed that a second, independent axis has been quietly
riding along inside the same profile: *"how does this site behave?"* — how a
game start and end are recognised, what a result banner is, whether clock
position carries meaning, how you resign. That is a property of the **site**,
and it is currently hardcoded into the client (and, in one place, into the
profile schema itself).

The two axes are genuinely independent: the same user on the same monitor
needs two different sets of *behaviour*, and the same site needs different
*coordinates* on a 1080p laptop versus a 4K desktop. Anything that tries to
express both with one concept ends up with a profile like
`chess_com_desktop.json`, which today stores **six identical clock boxes**
under `play/start1/start2/end1/end2/end3` — six keys that exist only because
*lichess* moves its clock when the game state changes.

## What broke, concretely

Three symptoms found while getting the chess.com profile to 100% board and
clock accuracy:

1. **chess.com end-of-game detection does not work at all.** Not "works
   badly" — the code path is silently skipped. See below.
2. **The clock-position end signal is meaningless on chess.com.** lichess
   repositions its clock when a game ends, which is what
   `game_over_found()` keys off. chess.com's clock sits at identical
   coordinates in every game state, so "a clock is readable at the end-state
   coordinates" is true during normal play and carries no information.
3. **Site assumptions leak into device data.** The `start1/start2/end1/end2/end3`
   clock-state vocabulary is a lichess-ism embedded in the calibration JSON
   schema, `ChessConfig.get_clock_position(clock_type, state)`, and every
   `capture_*_clock(state=...)` call site.

### Why chess.com's banner detection is currently dead code

`check_game_end()` (`clients/mp_original.py:1140`) is already multi-signal,
and Method 2 *is* the banner approach:

| # | Signal | Source |
|---|---|---|
| 1 | Board outcome (checkmate/stalemate) | `DYNAMIC_INFO["fens"][-1]` |
| 2 | Result image match | `result_region` + result templates |
| 2b | Game-over message in notation panel | `game_over_message_found()` |
| 3 | Clock readable at end coords, *not* at play coords | `game_over_found()` |

Method 2 never runs for chess.com because
`TemplateExtractor.load_result_templates()`
(`auto_calibration/template_extractor.py:676`) is all-or-nothing:

```python
filename_map = {
    'white_win': 'whitewin_result.png',
    'black_win': 'blackwin_result.png',
    'draw':      'draw_result.png',
}
for result_type, filename in filename_map.items():
    filepath = result_dir / filename
    if not filepath.exists():
        return None          # <-- one missing file disables ALL result detection
```

The chess.com profile has `white_win.png` / `black_win.png` (wrong filenames)
and no draw example at all, so this returns `None`, `_get_result_references()`
falls back to the legacy `chessimage/*.png` lichess references, and those
never match a chess.com modal.

Note the sibling loader `load_game_over_message_templates()` (line 706)
already does this correctly — each template is optional and a profile without
one simply skips that message. The result loader should follow the same rule.

### A correction on severity

An earlier note in this work described `game_over_found()` as *the* live
game-over signal. In the active client it is **gated**: Method 3 only consults
it when the play-position clock is *unreadable* first
(`clients/mp_original.py:1206`). Since chess.com's play clock is always
readable, that path effectively cannot fire there. The readback test's
"False Ends" metric calls `game_over_found()` in isolation and therefore
overstates production risk for `mp_original.py`. The older
`clients/mp_client.py:604` and
`clients/multiprocessing_client_one_game.py:591` *do* call it unguarded.

## The split

| Concern | Belongs to | Examples |
|---|---|---|
| Pixel geometry | **Profile** (device) | board origin/size, clock boxes, notation/rating regions, resign button coords |
| Rendering assets | **Profile** (device) | piece templates, digit templates, colour scheme, `RESOLUTION_SCALE` |
| Game lifecycle | **Site** | how a new game / game end / result is recognised |
| UI semantics | **Site** | does clock position encode state? is there a result modal or a side panel? |
| Interaction | **Site** | resign flow, new-game flow, back-to-lobby, promotion picker |
| Display quirks | **Site** | clock number format, premove rendering |

A profile then *binds* a device layout to a site, rather than implying one.

### Observed per-site behaviour

Everything below was measured from screenshots during the calibration work,
not assumed.

| Behaviour | lichess | chess.com |
|---|---|---|
| Clock position varies by game state | **Yes** (drives `start1..end3`) | **No** — identical in all states |
| Clock format, over a minute | `00:57.1` (zero-padded, tenths shown) | `2:57` (not zero-padded) |
| Clock format, under ~20s | — | `0:11.3` (switches to tenths) |
| Clock rendering styles | one | **three** (grey chip / white chip / plain on dark) |
| Clock has a leading status icon | No | Yes (fragments into digit-like regions) |
| Game-end presentation | result box + notation-panel messages | **modal banner over the board centre**, height varies by ending |
| Board theme affects piece templates | — | no (dark square is zeroed; cream light square is shared) |
| Board theme affects highlight colours | yes | yes — `colour_scheme` must be re-fitted per theme |
| Premoves drawn on the board | (unverified) | **Yes** — piece shown on its destination |
| Player info position | beside the clock, right panel | **above and below the board** |
| Player rows swap with our colour | yes | **no** — we are always the bottom player |
| Rating presentation | its own element, a bare number | inline after the username: `squishypup (1453)` |

The premove row matters beyond cosmetics: chess.com's board genuinely shows a
position that is *not* the server position while a premove is queued. Any
move-linking logic that assumes the scraped placement equals the confirmed
game state can be wrong by one or more plies on chess.com. (This is what made
three of the ground-truth labels in
`auto_calibration/offline_screenshots/chess_com/` wrong until they were
corrected.)

## Proposed design

### 1. Profiles declare their site

Add to `calibration_info` in each calibration JSON:

```json
"calibration_info": { "site": "lichess", ... }
```

`ChessConfig.get_site()` returns it, **defaulting to `"lichess"`** when
absent so every existing profile keeps working untouched. `main.py` gets an
optional `--site` override for testing.

Deliberately *not* auto-detected from the screen at runtime: the bot already
knows which site it was pointed at, and a misdetection would silently change
game-lifecycle behaviour mid-session. Auto-detection can be added later as a
sanity check that *warns*, rather than as the source of truth.

### 2. A `sites/` package

```
sites/
  __init__.py     # get_site(name) registry
  base.py         # Site ABC + capability declarations
  lichess.py      # LichessSite  - verbatim current behaviour
  chess_com.py    # ChessComSite
```

```python
class Site(ABC):
    name: str

    # --- capability declarations (data, not behaviour) ---
    clock_position_varies_by_state: bool
    clock_shows_tenths_below_seconds: int | None
    renders_premoves_on_board: bool
    result_template_files: dict[str, str]   # logical name -> filename
    optional_result_templates: frozenset[str]

    # --- lifecycle detection ---
    def detect_new_game(self, ctx, expected_time=None) -> int | None: ...
    def detect_game_end(self, ctx, game_state) -> GameEndSignal | None: ...
    def detect_result(self, ctx) -> str | None: ...

    # --- interaction (stage 2) ---
    def resign(self, ctx) -> bool: ...
    def start_new_game(self, ctx, time_control: str) -> bool: ...
    def back_to_lobby(self, ctx) -> bool: ...
    def complete_promotion(self, ctx, move_uci: str) -> None: ...
```

`ctx` is a **narrow protocol** (capture functions + config + click helpers),
not the client module itself. Sites must not import `clients.mp_original`, or
the import cycle makes the seam unusable. Concretely `ctx` supplies
`capture_board()`, `capture_clock(which, state)`, `capture_region(name)`,
`click(x, y, ...)` and `config`.

`GameEndSignal` carries *which* signal fired and any result string, so the
existing debug-screenshot and log lines keep their current detail.

### 3. Per-site lifecycle

**`LichessSite.detect_game_end`** is today's Methods 1 → 2 → 2b → 3, in
order, unchanged. This is the correctness anchor: the refactor must not alter
lichess behaviour at all.

**`ChessComSite.detect_game_end`** is Methods 1 → 2 only:

- Method 3 is **dropped entirely**, not merely reordered — its premise
  (clock position encodes game state) is false here, so keeping it can only
  produce false positives.
- Method 2b (notation-panel message) has no chess.com equivalent yet; the
  aborted/didn't-move endings need their own investigation.

### 4. chess.com banner detection

This section is **measured**, not proposed — the geometry below comes from the
eight end-state screenshots now in
`auto_calibration/offline_screenshots/chess_com/`.

#### The fixed `result_region` cannot work

Modal bounding boxes, in absolute screen pixels (board at x=785, y=185,
size=1872, so board centre is (1721, 1121)):

| Screenshot | modal x | w | y | h | centre |
|---|---|---|---|---|---|
| `won_timeout` | 1514 | 407 | 875 | 492 | (1717, 1121) |
| `i_win_checkmate` | 1521 | 400 | 875 | 492 | (1721, 1121) |
| `aborted` | 1521 | 400 | 884 | 474 | (1721, 1121) |
| `draw` | 1521 | 400 | 940 | 392 | (1721, 1136) |

So the modal is **always horizontally fixed** (`board_centre_x − 200`, width
400 ≈ 0.214 × board size) and **vertically centred on the board centre**, with
only its *height* varying by ending type (392–492).

The profile's current `result_region` is `{x:1550, y:850, w:350, h:100}` — a
fixed strip near the top of the *tallest* modal. Against `draw`, whose modal
starts at y=940, that strip lands entirely on board squares above the modal.
A fixed sub-region is therefore not merely fragile, it is wrong for at least
one real ending.

Worse, the title's offset *within* the modal also varies, because each ending
has a different layout (a trophy inline with the title, a large icon above
it, or neither):

| Ending | title | offset below modal top |
|---|---|---|
| `won_timeout` | "You Won!" | ~20–49 px |
| `draw` | "Draw" | ~56–84 px |
| `aborted` | "Game Aborted" | ~207–234 px |

So even anchoring to the modal's own top edge does not give a fixed title
position. The title must be **searched for anywhere within the modal**, which
is exactly what `game_over_message_found()` already does
(`cv2.matchTemplate(...).max()` over a region) — that pattern should simply be
reused rather than reinvented.

#### Modal presence is a trivially reliable "did it end" signal

Measuring the fraction of near-black pixels (`max(B,G,R) < 70`) inside a fixed
band centred on the board centre (±200 x, ±280 y) across the whole chess.com
corpus:

| Group | dark fraction |
|---|---|
| 8 ended games (`aborted`, `draw`, `won_timeout`, both checkmates, both resigns, `i_win_on_time`) | **0.401 – 0.647** |
| 21 in-play frames | **0.000 – 0.067** |

A **6× margin** with no overlap. Any threshold near 0.20 separates them
perfectly. This is far more robust than template-matching a title, and it is
wording-independent, so it survives message variants we have no screenshots
of (disconnects, "Black won by resignation", localised text).

Recommended two-step design:

1. **Did the game end?** — dark-fraction test in the fixed centre band.
   Cheap (one crop, one comparison) and wording-independent.
2. **What was the result?** — template-search the result titles *within* the
   modal box, reusing the `game_over_message_found()` search pattern.

Only step 2 needs templates, so a missing or unrecognised ending still yields
"the game is over" rather than the bot playing on into a dead board.

Two implementation cautions:

- **Do not take the largest dark connected component.** chess.com renders an
  ad panel immediately right of the modal which is also dark; in
  `i_lose_checkmate` the two merge into one 715-px-wide component. Constrain
  to the known geometry (centred, ~0.214 × board size wide) instead.
- The board occlusion is an *orthogonal* confirmation: once the modal covers
  the centre, `scraped_fen_sanity_issues()` (`common/utils.py`) will start
  rejecting the scrape. That signal is already computed and can corroborate
  step 1 for free.

#### Ending coverage now available

With the three screenshots added, chess.com's corpus covers `white_win`,
`black_win`, `draw`, `aborted` and `won_timeout`. Still missing: disconnect
and abandonment endings. Note `aborted` is a genuinely distinct case — the
board is still the starting position, so board-outcome detection cannot fire
and the modal is the *only* signal.

### 5. chess.com new-game detection

The mirror problem of section 4, and harder: nothing on chess.com marks a game
as having *begun*. Lichess moves its clock into a start-state position, which
is a positive signal; chess.com's clock never moves, and the lobby renders a
preview board **at the starting position** beside a clock showing the
**selected time control**. So "full clock over a start-like board, no result
modal" — every signal the site otherwise has — is equally true of
`/play/online` and of a game two seconds old.

Two live false positives on 2026-07-25 came from exactly that: the bot
announced a new game on opening the Play Online page, and again while sitting
in the seek queue, then played into a board nobody was moving. The green
**Start Game** button was the only lobby discriminator and it is not one — it
is *absent* while a seek is running, replaced by "Searching..".

The fix is positive evidence of a **game page**: the move-navigation bar
(`|< < > > >|`) under the moves list, which exists on every game page and on
no lobby screen. It is read relatively — the fraction of the band brighter
than that band's own median — so a theme change moves content and background
together. Measured across the whole chess.com corpus:

| Group | bright fraction of the band |
|---|---|
| every game page (playing, ended, aborted) | **0.063 – 0.135** |
| every lobby screen (home, /play, New Game panel, searching) | **0.000** |

The lobby panel is *perfectly* flat there, so the 0.03 threshold guards
against stray antialiasing rather than splitting a distribution.

Two deliberate calls:

- A capture failure returns `None`, not `False`. A window narrower than the
  calibration puts the panel outside the captured area, and failing closed
  there would mean the bot never finds a game again; unknown degrades to the
  previous behaviour instead.
- **No "is the clock ticking?" confirmation.** Re-reading both clocks ~1.3s
  apart is the strongest possible liveness proof and would also cover an
  aborted game whose modal had been dismissed, but it costs 1.3s off our own
  clock at the start of every game — material in bullet — and the
  modal-presence test already covers ended and aborted games with a 6×
  margin.

*Verification:* against the 34 full-resolution chess.com screenshots, the
three genuine game starts are still detected (including the one white move in
case), all four lobby screens and every mid-game and ended-game frame are
refused — 0 mismatches. Pinned by
`testing/client/test_chess_com_new_game_page.py`, which synthesises the bands
from measured greys because the screenshots are gitignored.

### 6. The pre-seek guard is a site question too

`new_game()` refuses to click through the lobby while a game is being played,
because those clicks land on the running game. The test was Lichess's — a
clock readable at a live clock position, with a visible end-of-game screen as
the escape — and the client applied it to every site.

On chess.com it is never false, for the same reason Method 3 was dropped from
`detect_game_end`: one clock position, never moves, and a finished game goes
on showing the time it ended with. The only thing between that and a refused
seek was the result modal, and chess.com swaps the modal for its analysis
panel a few seconds after the game ends. Across all 57
`new_game_blocked_live_game` screenshots in `logs/`, the modal dark fraction
was **0.000–0.126** — every one below the 0.20 threshold, i.e. no modal on
screen in any of them. Every session on 2026-07-25 shows the shape:

```
GAME 1 ENDED
[ERROR] Tried to seek a new game but a live game appears to be on screen
[WARN ] Game 2 skipped, seeking again
[INFO ] chess.com: found Start Game button at (2939, 308)   <- only the retry sought
```

So the bot never sought after its own games; it sought on the *retry* once the
new-game wait had timed out, which is why a too-short wait and this guard
looked like one bug.

The question moved to `Site.live_game_on_screen()`, which returns the reason a
game looks live or `None`. The default is the Lichess test unchanged.
chess.com overrides it with the one unambiguous statement of liveness it has:
**is a clock ticking** — two reads 1.2s apart, a drop of 1..6s. The wait is
affordable here because it happens between games, which is exactly why the
same test is *not* used in `detect_new_game` (section 5), where it would cost
a second of our own clock at the start of every game.

*Verification:* all 56 historically-blocked screens now permit the seek; a
live game whose clock is moving still blocks it. Pinned by
`testing/client/test_seek_guard.py`, both sites.

### 7. Not every site difference belongs in `sites/`

The player/rating rows are the counter-example, and worth recording because
the instinct to reach for `sites/` was wrong.

Every one of the 19 games on 2026-07-25 logged `Detected ratings: Opponent:
None, Self: None`. Two causes, and they sit in different layers:

- **Where the rating is.** Lichess keeps player info beside the clock in the
  right panel; chess.com puts it above and below the *board* and never swaps
  the two rows by colour. `CoordinateCalculator` only knew the Lichess shape,
  so fitting chess.com placed the 70px crop 100px right of the number — it
  was reading `53)`.
- **What the rating looks like.** chess.com writes it inline after a
  username of unpredictable length, so even a well-placed crop is a line of
  text, not a number. `capture_rating` did `int()` on the whole string.

Neither belongs in `sites/`. The first is a *where*, which is
`auto_calibration/`'s job — so `CoordinateCalculator` gained a `site` and a
chess.com branch deriving the rows from the board in step units, and
`offline_fitter --site` records the choice in `calibration_info` (the same
field `Site` binding already reads). The second is not site-specific at all
once stated properly: `rating_from_words()` prefers a bracketed 3-4 digit
word and otherwise accepts a bare number only as the line's sole word, which
is precisely the Lichess crop. One reader, both sites.

The rule this suggests: `sites/` is for *behaviour over time* — what a game
start looks like, when a game has ended, what a click flow does.
Presentation and position, even when they differ per site, stay in the
vision layer with the site as a parameter.

*Verification:* 14/14 readings on the chess.com game-start screens (0
before), every value on the mid-game production frames inside a plausible
band, and all 26 Lichess screenshots returning exactly their previous
values. Pinned by `testing/client/test_rating_ocr.py`.

## Migration stages

Each stage is independently shippable and independently verifiable.

**Stage 0 — schema + loader (no behaviour change)**
- Add `site` to `calibration_info`; `ChessConfig.get_site()` defaults to `lichess`.
- Make `load_result_templates()` per-site: take the filename map from the
  site, treat missing entries as optional (matching
  `load_game_over_message_templates`), return what exists.
- Rename the chess.com result templates to the site's declared filenames.
- *Verification (done):* Lichess readback and client tests byte-identical.

**Stage 1 — lifecycle detection behind `Site`** (the part that unblocks chess.com)
- Create `sites/`; move `new_game_found`, `game_over_found`, `check_game_end`
  bodies into `LichessSite` unchanged; client calls `SITE.detect_*`.
- Implement `ChessComSite` lifecycle + banner detection (modal-presence test
  for *whether*, template search within the modal for *which*).
- *Verification (done):*
  - Lichess readback identical to the Stage 0 baseline; client tests 11/11.
  - `LichessSite` proven equivalent to the pre-refactor client logic by
    running both implementations side by side over every Lichess screenshot
    across three FEN-history scenarios: **78 comparisons, 0 mismatches** for
    game-end detection and **78 comparisons, 0 mismatches** for new-game
    detection. (When building that harness, note that the extracted original
    `check_game_end` silently returns False if `game_over_found` is missing -
    its Method 3 sits under a bare `except` - which looks like a real
    mismatch until the helper is supplied.)
  - chess.com end detection: **29/29** — all 8 endings detected with the
    correct result name, all 21 in-play frames negative.

**Stage 2 — interaction behind `Site`** *(implemented)*
- `resign()`, `start_new_game()`, `back_to_lobby()`, `berserk()` moved behind
  the interface, with a `SiteActions` record carrying the client capabilities
  a site needs (click, sleep, log). The split keeps *policy* in the client -
  the caps-lock human-interference guard, the log buffer, the humanised
  mouse - and moves only *where to click, in what order* into the site, so
  sites still never import a client.
- `supports_berserk` / `supports_back_to_lobby` let the client skip controls a
  site does not have, instead of clicking where Lichess happens to put them.
- *Verification (done):* these paths have no offline coverage, so they were
  checked by equivalence rather than behaviour - the original functions and
  the new site were each driven with a recording click stub and their click
  sequences compared exactly: `berserk`, `back_to_lobby`, `resign` and all
  ten time-control lobby fallbacks produce **identical (x, y, tolerance,
  clicks)** sequences.
- chess.com's lobby is calibrated from the `find_new_game_*` screenshots.
  Two details are worth keeping: the green **Start Game** button is located
  by colour rather than position, because opening the time-control dropdown
  pushes it down the panel (measured at y=308 collapsed, y=788 open) and a
  missed click would land on the variant selector and silently change the
  game type; and an **unsupported time control is refused rather than
  approximated**, since chess.com's default grid has no 5+3 and clicking the
  nearest cell would start a game whose pacing the engine then gets wrong.
- *Verification (done):* every planned click was replayed against the
  screenshots and checked to land inside its intended control - all five
  lobby steps, and all nine time-control cells dead-centre. The Start Game
  colour search returns the right button in both the collapsed and open
  states, and `None` on an in-game screenshot, so it cannot invent a button
  where there is no lobby.
- **Promotion needs no handling.** `find_clicks()` ignores the promotion
  piece in the UCI and only clicks from/to, which is correct: both sites are
  configured to auto-queen, so the from/to click completes the move. The
  consequence is that underpromotion cannot be played - an accepted
  limitation of the mouse interface, not an outstanding bug.

**Stage 3 — retire the lichess clock-state vocabulary**
- Replace `state="start1"|...` with site-declared states; chess.com's profile
  stops carrying six identical clock boxes.
- Touches the calibration schema, the offline fitter, and every client.

## Verification strategy

The technique that caught a real regression during the clock work should be
the standard gate for every stage:

```
git stash push
python -m auto_calibration.calibration_readback_test --screenshots <dir> --profile <p> > head.txt
git stash pop
python -m auto_calibration.calibration_readback_test --screenshots <dir> --profile <p> > after.txt
diff head.txt after.txt
```

For lichess, **any** diff during Stages 0–1 is a bug, since those stages are
meant to be behaviour-preserving there. Note that the readback test's
"False Ends" check models `game_over_found()` in isolation rather than
production's guarded use; once `ChessComSite` drops the clock signal, that
metric should be made site-aware too or it will keep reporting 100% for
chess.com forever.

## Board themes: resolved, one theme per profile

**Decision: theme is a profile concern.** Piece and digit templates already
live per-profile, so this is had for free; a user running two themes runs two
profiles.

This was then checked rather than assumed, because the three new end-state
screenshots arrived on a **green** board theme while the entire existing
chess.com play corpus is **brown**. Running the current (brown-derived) piece
templates against the green-theme boards:

| Screenshot | extracted placement | verdict |
|---|---|---|
| `aborted` | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` | **exactly** the starting position — correct, the game was aborted before a move |
| `draw` | `r1bqk2r/ppp1bppp/2np1n2/8/2B1p3/3P1N2/PPP2PPP/RNBQ1RK1` | coherent Italian-structure position |

Piece detection is therefore **more theme-robust than expected**, and the
mechanism explains why: `remove_background_colours()` zeroes strongly coloured
pixels, which removes the dark square entirely (green spread ≈ 50, brown ≈ 77,
both well above the threshold), while the near-neutral cream light square
(spread ≈ 8) is retained identically by both themes. A theme that changed the
*light* square to a saturated colour would not transfer this way.

The genuinely theme-sensitive part is the **highlight colour table**
(`colour_scheme`, consumed by `_build_highlight_colours()`), which drives
last-move and turn detection. That is already per-profile, so the split holds
— but it means a theme change requires re-fitting colours even though the
piece templates would have survived.

## Open questions

1. **Remaining endings** — disconnect and abandonment endings are still
   uncaptured. Lower priority now that modal *presence* detection is
   wording-independent (see above), so an unrecognised ending degrades to
   "game over, result unknown" rather than to a hang.
2. **Test-corpus defect (pre-existing, found while measuring).**
   `calibration_readback_test.py:500` derives state from the *filename* and
   sets `is_actual_end = 'end' in state_hint`. `i_win_on_time.png` is a
   genuine end screenshot (it carries `result:black_win` and scores 0.644 on
   the modal test) but its filename has no `end`, so it is counted as an
   in-play frame and inflates the false-end metric. `is_actual_end` should be
   driven by the presence of a `result:` label, not the filename.
3. **Premove semantics** — does the client's move-linking need to know that
   chess.com draws unconfirmed premoves? Likely yes; needs its own
   investigation against `update_dynamic_info_from_fullimage`.
4. **Promotion** — resolved: both sites auto-queen, so clicking from/to
   completes the move. Underpromotion is unreachable through the mouse
   interface and is accepted as such.
5. **Older clients** — `mp_client.py` and `multiprocessing_client_one_game.py`
   call `game_over_found()` unguarded. Migrate them, or formally retire them?
