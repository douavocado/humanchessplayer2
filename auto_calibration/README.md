# Chess Board Auto-Calibration

Automatic detection of chess board position and UI elements (clocks, notation, ratings) for the chess bot. Works across different monitors, resolutions, and scaling factors.

## Quick Start

### Live Calibration (Recommended)

```bash
# Activate your environment
conda activate humanchess2

# Run calibration with a chess game visible on screen
python -m auto_calibration.calibrator --live
```

### Offline Calibration (From Screenshots)

```bash
# Calibrate from a saved screenshot
python -m auto_calibration.calibrator --screenshot ./my_screenshot.png

# Calibrate from multiple screenshots
python -m auto_calibration.calibrator --offline ./screenshots/
```

## How It Works

### Phase 1: Board Detection

The system uses colour segmentation to find the Lichess chess board:

1. **Downscale** the image 8× for fast initial scanning
2. **Create colour masks** for Lichess light (cream) and dark (green) squares
3. **Find the largest connected region** matching board colours
4. **Verify 8×8 grid pattern** by checking alternating cell intensities
5. **Refine** position on full resolution

### Phase 2: Clock Detection

Clocks are found using your existing `read_clock()` function:

1. **Calculate search region** - clocks are to the right of the board
2. **Find clock X position** by testing multiple X values
3. **Sweep Y-axis** in 3px increments
4. **Validate with `read_clock()`** - if it returns a time, the position is correct
5. **Cluster nearby detections** to handle slight variations

### Phase 3: Coordinate Calculation

Once board and clocks are found, notation and rating positions are derived:

- **Notation panel**: Centred between top and bottom clocks
- **Rating positions**: Fixed offset from clock X position

## Commands

### Calibrator

```bash
# Live calibration (5 second countdown)
python -m auto_calibration.calibrator --live

# Live with custom countdown
python -m auto_calibration.calibrator --live --countdown 10

# From single screenshot
python -m auto_calibration.calibrator --screenshot ./screenshot.png

# From screenshots directory
python -m auto_calibration.calibrator --offline ./calibration_screenshots/

# Verify existing configuration
python -m auto_calibration.calibrator --verify

# Skip visualisation
python -m auto_calibration.calibrator --live --no-visualise
```

### Screenshot Collector

For capturing screenshots of different game states:

```bash
# Single capture with 3s delay
python -m auto_calibration.screenshot_collector

# Capture with state hint
python -m auto_calibration.screenshot_collector --state play

# Interactive mode (prompts for each state)
python -m auto_calibration.screenshot_collector --interactive

# Multiple captures
python -m auto_calibration.screenshot_collector --count 5
```

## Clock States

The system handles 6 different clock positions that vary based on game phase:

| State | Description |
|-------|-------------|
| `play` | Normal gameplay (after several moves) |
| `start1` | Game start, before first move |
| `start2` | After first move played |
| `end1` | Game over (by resignation) |
| `end2` | Game over (by timeout) |
| `end3` | Game over (by checkmate/stalemate/draw) |

When calibrating from a single screenshot, missing states are estimated based on known Y-offsets.

## Output Files

### Configuration

After calibration, `chess_config.json` is created/updated:

```json
{
  "calibration_info": {
    "timestamp": "2025-01-15T14:30:22",
    "method": "live",
    "board_confidence": 0.85,
    "clock_states_detected": ["bottom_clock.play", "top_clock.play"]
  },
  "coordinates": {
    "board": {"x": 543, "y": 179, "width": 848, "height": 848},
    "bottom_clock": {
      "play": {"x": 1420, "y": 742, "width": 147, "height": 44},
      "start1": {"x": 1420, "y": 756, "width": 147, "height": 44},
      ...
    },
    ...
  }
}
```

### Debug Visualisations

Each calibration run creates a timestamped folder in `calibration_outputs/`:

```
calibration_outputs/2025-01-15_14-30-22/
├── 01_original.png           # Input screenshot
├── 02_colour_segmentation.png # Colour masks
├── 03_board_detection.png    # Board with grid overlay
├── 04_clock_detection.png    # Detected clock positions
├── 05_final_overlay.png      # All elements combined
├── extracted_regions/        # Individual extracted regions
│   ├── board.png
│   ├── bottom_clock_play.png
│   ├── notation.png
│   └── ...
└── report.txt                # Text summary
```

## Integration

### Using Calibrated Coordinates

```python
from auto_calibration.config import get_config

# Get configuration
config = get_config()

# Get board position
x, y, step = config.get_board_info()

# Get clock position for specific state
cx, cy, cw, ch = config.get_clock_position('bottom_clock', 'play')

# Check if using fallback (no calibration)
if config.is_using_fallback():
    print("Run calibration!")
```

### Backward Compatibility

The calibration integrates with `chessimage/image_scrape_utils.py`:

```python
# These still work exactly the same:
from chessimage.image_scrape_utils import capture_board, capture_bottom_clock
```

## Troubleshooting

### Board Detection Fails

- Ensure a Lichess game is visible on screen
- The board should show the standard green/cream theme
- Try browser zoom at 100%
- Avoid obstructing the board with windows

### Clock Detection Fails

- Clocks must show actual time values (not "--:--")
- Run during active gameplay, not before game starts
- Ensure clock area is not obstructed

### Low Confidence

- Run calibration during active gameplay
- Try capturing multiple screenshots with different game states
- Use offline fitting with multiple screenshots

### Multi-Monitor Setup

The system captures the entire screen by default. If you have multiple monitors:

```bash
# The board should be clearly visible
# The system will find the largest chess board pattern
python -m auto_calibration.calibrator --live
```

## File Structure

```
auto_calibration/
├── __init__.py               # Module init
├── calibrator.py             # Main entry point (coordinate calibration)
├── board_detector.py         # Board detection
├── clock_detector.py         # Clock detection
├── coordinate_calculator.py  # Derived coordinates
├── visualiser.py             # Debug visualisation
├── config.py                 # Configuration loading/saving
├── offline_fitter.py         # Offline fitting
├── screenshot_collector.py   # Screenshot helper
├── utils.py                  # Shared utilities
├── template_extractor.py     # Template extraction core logic
├── shadow_calibrator.py      # Passive template extraction
├── interactive_calibrator.py # Manual template capture GUI
├── chess_config.json         # Generated configuration
├── calibration_screenshots/  # Saved screenshots
├── calibration_outputs/      # Debug outputs
└── templates/                # Extracted templates
    ├── digits/               # Clock digit templates (0-9)
    ├── pieces/               # Chess piece templates
    └── results/              # Game result templates
```

## Performance

- **Calibration time**: ~2-5 seconds
- **Board detection**: Uses 8× downsampling for speed
- **Clock detection**: 1D Y-sweep (not 2D grid search)
- **Runtime overhead**: Zero (simple coordinate lookup)

---

## Template Calibration

When switching to a new setup (different resolution, scaling, or theme), you may need to recalibrate **templates** for:
- Clock digits (0-9)
- Chess pieces
- Game result labels (1-0, 0-1, ½-½)

### Shadow Calibration (Recommended)

The shadow calibrator runs passively during normal gameplay, automatically extracting templates as they appear:

```bash
# Run in background for 5 minutes while you play
python -m auto_calibration.shadow_calibrator --duration 5

# Check current template status
python -m auto_calibration.shadow_calibrator --status

# Reset and start fresh
python -m auto_calibration.shadow_calibrator --reset all
```

**What it captures:**
- **Pieces**: Extracted from the starting position when a new game begins
- **Digits**: Captured as the clock counts down (one game typically covers most digits)
- **Results**: Captured when games end (requires winning as white, as black, and drawing)

### Interactive Calibration (Manual Override)

For templates that shadow calibration cannot capture (e.g., specific game results), use the interactive tool:

```bash
# Guided mode - walks through all missing templates
python -m auto_calibration.interactive_calibrator --guided

# Capture specific templates
python -m auto_calibration.interactive_calibrator --digit 7
python -m auto_calibration.interactive_calibrator --result black_win
python -m auto_calibration.interactive_calibrator --pieces
```

### Typical Calibration Workflow

1. **Run coordinate calibration first** (if new setup):
   ```bash
   python -m auto_calibration.calibrator --live
   ```

2. **Start shadow calibration** and play a few games:
   ```bash
   python -m auto_calibration.shadow_calibrator --duration 10
   ```

3. **Check what's missing**:
   ```bash
   python -m auto_calibration.shadow_calibrator --status
   ```

4. **Use interactive mode** for any remaining templates:
   ```bash
   python -m auto_calibration.interactive_calibrator --guided
   ```

### Template Storage

Extracted templates are stored in `auto_calibration/templates/`:
- `digits/0.png` through `digits/9.png` - Clock digit templates
- `pieces/w_rook.png`, `pieces/b_knight.png`, etc. - Chess piece templates  
- `results/whitewin_result.png`, `blackwin_result.png`, `draw_result.png` - Result labels

Progress is tracked in `templates/extraction_progress.json`.
