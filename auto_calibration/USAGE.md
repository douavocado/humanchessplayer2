# Auto-Calibration Usage Guide

## Quick Start

### 1. Run Calibration
```bash
conda activate humanchess2
cd auto_calibration
python calibrator.py
```

### 2. That's It!
Your system now automatically uses calibrated coordinates instead of hardcoded values.

## What Changed

### Before (Hardcoded):
```python
# In chessimage/image_scrape_utils.py
BOTTOM_CLOCK_X = 1420  # Fixed for your specific setup
BOTTOM_CLOCK_Y = 742
START_X = 543
START_Y = 179
```

### After (Auto-Calibrated):
```python
# Automatically loaded from chess_config.json
START_X, START_Y, STEP = get_board_info()  # Device-independent
BOTTOM_CLOCK_X, BOTTOM_CLOCK_Y, _, _ = get_clock_info('bottom_clock', 'play')
```

## How Your Existing Code Works

### Your Functions Work Unchanged
```python
# These still work exactly the same:
from clients.mp_original import new_game_found, game_over_found
from chessimage.image_scrape_utils import capture_bottom_clock

# All state variations still work:
capture_bottom_clock('play')    # Normal gameplay
capture_bottom_clock('start1')  # New game start (your START_Y equivalent)
capture_bottom_clock('start2')  # New game alt (your START_Y_2 equivalent) 
capture_bottom_clock('end1')    # Game over (your END_Y equivalent)
capture_bottom_clock('end2')    # Game over alt (your END_Y_2 equivalent)
capture_bottom_clock('end3')    # Game over alt (your END_Y_3 equivalent)
```

### Fallback System
If `chess_config.json` is missing, the system automatically falls back to your original hardcoded coordinates.

## Status Messages

When you import your modules, you'll see:
- `✅ Loaded chess configuration from: auto_calibration/chess_config.json`
- `📍 Using auto-calibrated coordinates`

Or if no config file:
- `⚠️ Auto-calibration not available, using hardcoded coordinates`

## When to Recalibrate

Run `python calibrator.py` again when:
- Screen resolution changes
- Browser zoom level changes  
- Monitor setup changes
- Chess site updates their layout
- Clock detection starts failing

## Files Created

- `auto_calibration/chess_config.json` - Your calibrated coordinates
- `auto_calibration/calibrator.py` - The calibration tool
- `auto_calibration/config_loader.py` - Configuration loading system

## Performance

- **Calibration**: ~10 seconds one-time process
- **Runtime**: Identical performance to hardcoded coordinates
- **Memory**: Minimal overhead (small JSON config)

## Integration Status

✅ **Fully Integrated** - Your existing code works without any changes  
✅ **Backward Compatible** - Falls back to hardcoded values if needed  
✅ **State-Aware** - Handles all your START_Y_2, END_Y_2, END_Y_3 variations  
✅ **Device Independent** - Works across different screens and zoom levels  

Your chess bot now automatically adapts to different devices while maintaining the same performance and functionality! 🎯
