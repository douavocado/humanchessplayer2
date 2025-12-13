# Chess Board Auto-Calibration System

This module provides device-independent auto-calibration for chess board detection and UI element positioning. It automatically detects the chess board position and calculates coordinates for clocks, notation areas, and other UI elements relative to the board.

## Features

- **Multi-scale template matching** for board detection
- **Device-independent coordinate mapping** 
- **State-specific coordinate detection** - handles UI shifts during different game phases
- **Automatic template scale detection** for digit recognition
- **Configuration validation** and testing
- **Visual diagnostics** and debugging tools
- **Interactive calibration interface**

## Quick Start

1. **Run the auto-calibration:**
   ```bash
   cd testing/autoboard_calibration
   python auto_calibrator.py
   ```

2. **Follow the on-screen instructions:**
   - Ensure a chess game is open and visible
   - Wait for the 5-second countdown
   - The system will automatically detect and calibrate

3. **Test your calibration:**
   - Use the interactive interface to test clock detection
   - View diagnostic visualizations

## Files

### Core Modules

- **`auto_calibrator.py`** - Main calibration interface and orchestrator
- **`board_detector.py`** - Chess board detection using template matching
- **`coordinate_mapper.py`** - UI element coordinate mapping relative to board
- **`visualizer.py`** - Diagnostic visualization and validation tools

### Usage Examples

#### Basic Calibration
```python
from auto_calibrator import AutoCalibrator

calibrator = AutoCalibrator()
config = calibrator.run_full_calibration()

if config:
    print("Calibration successful!")
    # Configuration is automatically saved
else:
    print("Calibration failed")
```

#### Load and Test Existing Configuration
```python
from coordinate_mapper import CoordinateMapper
import json

# Load saved configuration
mapper = CoordinateMapper()
config = mapper.load_config("auto_calibration_config.json")

# Test with current screen
calibrator = AutoCalibrator()
test_results = calibrator.test_clock_detection(config)
print(test_results)
```

#### Create Diagnostic Report
```python
from visualizer import CalibrationVisualizer
import json

# Load configuration
with open("auto_calibration_config.json", 'r') as f:
    config = json.load(f)

# Generate diagnostic report
visualizer = CalibrationVisualizer()
output_dir = visualizer.create_diagnostic_report(config)
print(f"Diagnostic report saved to: {output_dir}")
```

## Configuration Format

The auto-calibration generates a JSON configuration file with the following structure:

```json
{
  "board_detection": {
    "method": "template_matching",
    "position": [543, 179, 848, 848],
    "confidence": 0.85,
    "scale": 1.2
  },
  "template_scale": 1.1,
  "timestamp": "2024-01-15T10:30:00",
  "ui_elements": {
    "bottom_clock": {
      "play": {"x": 1420, "y": 742, "width": 147, "height": 44},
      "start1": {"x": 1420, "y": 756, "width": 147, "height": 44},
      "start2": {"x": 1420, "y": 770, "width": 147, "height": 44}
    },
    "top_clock": {
      "play": {"x": 1420, "y": 424, "width": 147, "height": 44},
      "start1": {"x": 1420, "y": 396, "width": 147, "height": 44}
    },
    "notation": {"x": 1458, "y": 591, "width": 166, "height": 104},
    "rating": {
      "opp_white": {"x": 1755, "y": 458, "width": 40, "height": 24},
      "own_white": {"x": 1755, "y": 706, "width": 40, "height": 24}
    }
  }
}
```

## Detection Methods

### 1. Template Matching
- Uses existing board images as templates
- Tests multiple scales (0.5x to 2.0x)
- Most reliable for consistent setups

### 2. Corner Detection  
- Creates synthetic checkerboard corner template
- Fallback when main template fails
- Estimates full board size from corner match

### 3. Edge Detection
- Detects board using line detection
- Last resort method
- Works with high-contrast board edges

## State-Specific Coordinate Handling

The system automatically accounts for UI position shifts that occur during different game phases:

### **Game States Detected:**
- **`play`** - Normal gameplay with active clocks
- **`start1`** - New game initialization (primary position)
- **`start2`** - New game initialization (alternative position)
- **`end1`** - Game over (resigned/timeout)
- **`end2`** - Game over (aborted)
- **`end3`** - Game over (alternative position)

### **Why This Matters:**
Chess sites like Chess.com and Lichess slightly reposition UI elements during different phases:
- Clock positions shift by 14-28 pixels between start and play states
- End game screens have different layouts than active gameplay
- Your original hardcoded `START_Y_2`, `END_Y_2`, `END_Y_3` variables handle these shifts

### **How It Works:**
1. **Analyzes current screen** to detect which state is active
2. **Tests all state positions** to find the best clock detection
3. **Ranks positions by confidence** based on clock readability
4. **Generates state-specific coordinates** for each game phase
5. **Maintains compatibility** with your existing `new_game_found()` and `game_over_found()` logic

## Troubleshooting

### Board Detection Fails

**Symptoms:** "Could not detect chess board" error

**Solutions:**
1. Ensure chess game is fully visible and not obstructed
2. Try different browser zoom levels (90%, 100%, 110%)
3. Improve contrast between board and background
4. Close other windows that might interfere
5. Use a standard chess site layout (Chess.com, Lichess)

### Low Clock Detection Success Rate

**Symptoms:** Clock validation shows <50% success rate

**Solutions:**
1. Ensure clocks are visible during calibration
2. Try calibrating during an active game
3. Check if site layout has changed
4. Verify browser zoom level is consistent
5. Re-run calibration with different timing

### Template Scale Issues

**Symptoms:** Digit recognition fails consistently

**Solutions:**
1. Check if browser zoom has changed
2. Try different display scaling settings
3. Ensure clock fonts are standard
4. Re-run calibration to detect new scale

## Integration with Main Code

To integrate with your existing chess client:

1. **Generate configuration:**
   ```bash
   python auto_calibrator.py
   ```

2. **Modify your client to use relative coordinates:**
   ```python
   # Load auto-calibration config
   with open('path/to/auto_calibration_config.json', 'r') as f:
       config = json.load(f)
   
   # Use calculated coordinates instead of hardcoded ones
   def capture_bottom_clock(state="play"):
       coords = config['ui_elements']['bottom_clock'][state]
       return SCREEN_CAPTURE.capture((
           coords['x'], coords['y'], 
           coords['width'], coords['height']
       ))
   ```

3. **Handle scale adjustment:**
   ```python
   # Adjust template scale for digit recognition
   template_scale = config['template_scale']
   scaled_templates = scale_digit_templates(TEMPLATES, template_scale)
   ```

## Performance Considerations

- **Board detection** runs only during calibration (once per setup)
- **Runtime coordinate lookups** are simple dictionary access (fast)
- **Template scaling** is pre-calculated and cached
- **No performance impact** on normal gameplay after calibration

## System Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- fastgrab (for screenshots)
- matplotlib (optional, for interactive visualization)

## Known Limitations

1. **Site-specific layouts:** Calibrated for Chess.com/Lichess standard layouts
2. **Browser zoom:** Requires consistent zoom level between calibration and use
3. **Multiple monitors:** May need recalibration when changing monitor setup
4. **Dark modes:** May affect detection accuracy depending on contrast

## Future Improvements

- [ ] Support for more chess sites
- [ ] Dynamic recalibration during runtime
- [ ] Machine learning-based element detection
- [ ] Mobile/tablet support
- [ ] Configuration profiles for different setups
