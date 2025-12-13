# Chess Board Auto-Calibration Tool

This tool automatically detects your chess board position and generates device-independent coordinate configuration, replacing hardcoded screen coordinates with adaptive values.

## Quick Start

### 1. Run Calibration
```bash
# Activate your environment
conda activate humanchess2

# Run the calibration tool
cd auto_calibration
python calibrator.py
```

### 2. Follow Instructions
- Open a chess game (Chess.com or Lichess)
- Ensure board and clocks are visible
- Wait for the 5-second countdown
- The tool will automatically detect and save coordinates

### 3. Use Configuration
Your main.py will now automatically use the calibrated coordinates instead of hardcoded values.

## What It Does

### Automatically Detects:
- ✅ **Chess board position** (any size/zoom level)
- ✅ **Clock positions** for all game states (play, start1/start2, end1/end2/end3)
- ✅ **UI element locations** (notation, ratings)
- ✅ **State-specific coordinate variations** (handles UI shifts during different game phases)

### Generates Configuration:
- 📄 **chess_config.json** - Complete coordinate configuration
- 🔧 **Device-independent** - Works across different screens/resolutions
- 🎯 **State-aware** - Handles position shifts like your START_Y_2, END_Y_2, END_Y_3
- 🚀 **Drop-in replacement** - No code changes needed in main system

## Files Overview

- **`calibrator.py`** - Main calibration tool (run this)
- **`config_loader.py`** - Configuration loading system for main code
- **`board_detector.py`** - Chess board detection using template matching
- **`coordinate_calculator.py`** - UI coordinate calculation relative to board
- **`calibration_utils.py`** - Utility functions for image processing

## Integration

### For Main System
The calibration automatically integrates with your existing code:

```python
# Instead of hardcoded:
BOTTOM_CLOCK_X = 1420
BOTTOM_CLOCK_Y = 742

# Now uses:
from auto_calibration.config_loader import get_clock_position
x, y, w, h = get_clock_position('bottom_clock', 'play')
```

### State-Specific Coordinates
All your state variations are automatically mapped:

```python
# Replaces your manual coordinate variations:
get_clock_position('bottom_clock', 'play')    # Normal gameplay
get_clock_position('bottom_clock', 'start1')  # New game (like START_Y)
get_clock_position('bottom_clock', 'start2')  # New game alt (like START_Y_2)
get_clock_position('bottom_clock', 'end1')    # Game over (like END_Y)
get_clock_position('bottom_clock', 'end2')    # Game over alt (like END_Y_2)
get_clock_position('bottom_clock', 'end3')    # Game over alt (like END_Y_3)
```

## Configuration File

The generated `chess_config.json` contains:

```json
{
  "calibration_info": {
    "timestamp": "2025-09-12T20:36:39",
    "board_detection": { "method": "corner_template", "confidence": 0.84 },
    "validation_success_rate": 0.818
  },
  "coordinates": {
    "board": { "x": 543, "y": 179, "width": 848, "height": 848 },
    "bottom_clock": {
      "play": { "x": 1420, "y": 742, "width": 147, "height": 44 },
      "start1": { "x": 1420, "y": 756, "width": 147, "height": 44 },
      "start2": { "x": 1420, "y": 770, "width": 147, "height": 44 }
    },
    "top_clock": { /* similar structure */ },
    "notation": { /* notation area coordinates */ },
    "rating": { /* rating area coordinates */ }
  }
}
```

## Fallback System

If no configuration file exists, the system automatically falls back to your original hardcoded coordinates, ensuring your system never breaks.

## When to Recalibrate

Run calibration again when:
- 🖥️ **Screen resolution changes**
- 🔍 **Browser zoom level changes**
- 🖱️ **Monitor setup changes** (add/remove monitors)
- 🌐 **Chess site layout updates**
- ❌ **Clock detection starts failing**

## Troubleshooting

### Board Detection Fails
- Ensure chess game is fully visible
- Try 100% browser zoom
- Use during active gameplay (best detection)
- Close other windows that might interfere

### Low Success Rate
- Run during active game with visible clocks
- Ensure good contrast between UI elements
- Try different game states (start vs play vs end)

### Integration Issues
- Check that `chess_config.json` exists
- Verify file permissions
- Check console output for error messages

## Performance

- **Calibration**: One-time process (~10 seconds)
- **Runtime**: Zero performance impact (simple coordinate lookup)
- **Memory**: Minimal (small JSON config in memory)
- **Same speed**: Identical performance to hardcoded coordinates

## Example Usage

```bash
# Run calibration
python calibrator.py

# Check status in your main code
from auto_calibration.config_loader import print_config_status
print_config_status()

# Use in your existing functions
from auto_calibration.config_loader import get_clock_position

def capture_bottom_clock(state="play"):
    x, y, w, h = get_clock_position('bottom_clock', state)
    return SCREEN_CAPTURE.capture((x, y, w, h))
```

This tool eliminates the need for manual coordinate calibration while maintaining full compatibility with your existing codebase! 🎯
