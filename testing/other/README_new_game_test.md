# New Game Position Tester

This script tests the click positions used in the `new_game()` function from `clients/mp_original.py` without actually performing the final clicks. It's designed to help verify that the theoretical click positions are well calibrated.

## Purpose

The `new_game()` function in the main client calculates specific screen coordinates for:
1. **Play button** - to start a new game
2. **Time control buttons** - to select 1+0 or 3+0 time controls

This test script allows you to verify these positions are correct by hovering the mouse cursor over them without actually clicking.

## How it Works

The script replicates the exact coordinate calculations from the original `new_game()` function:

### Play Button Position
```python
play_button_x = START_X - 1.9 * STEP
play_button_y = START_Y - 0.4 * STEP
```

### Time Control Positions
- **1+0 control**: `START_X + 1.7*STEP, START_Y + 0.7*STEP`
- **3+0 control**: `START_X + 5.7*STEP, START_Y + 0.7*STEP`

## Usage

### Basic Usage
```bash
cd testing/other/
python test_new_game_positions.py
```

### What the Script Does

1. **Displays coordinate calculations** - Shows the math behind each position
2. **Tests cursor positioning** - Moves cursor to each calculated position
3. **Applies tolerance** - Uses the same random tolerance as the original function
4. **Provides verification time** - Pauses at each position so you can verify accuracy

### Test Modes

The script offers three test modes:

1. **Automatic Test**: Tests all positions sequentially
2. **Interactive Test**: Waits for user input between each position
3. **Both**: Runs automatic test followed by interactive test

### Sample Output

```
New Game Position Tester
Current mouse position: (960, 540)

COORDINATE CALCULATIONS
Base coordinates:
  START_X = 543
  START_Y = 179
  STEP = 106

Play button calculation:
  X = START_X - 1.9*STEP = 543 - 1.9*106 = 341.6
  Y = START_Y - 0.4*STEP = 179 - 0.4*106 = 136.6

1. Testing PLAY BUTTON position:
Hovering at Play Button: (347.2, 141.1) [original: (341.6, 136.6), tolerance: ±5]
```

## Safety Features

- **Caps Lock Check**: Mimics the original function's caps lock safety check
- **No Actual Clicking**: Only moves cursor, never clicks
- **Position Restoration**: Returns mouse to original position when done
- **Failsafe**: PyAutoGUI failsafe remains enabled

## Requirements

- Python 3.x
- PyAutoGUI (`pip install pyautogui`)
- NumPy (`pip install numpy`)

**Note**: The script is self-contained and doesn't require the full project dependencies. It extracts only the necessary constants and uses a simplified cursor implementation.

## Verification Process

1. Run the script
2. Watch the cursor move to each calculated position
3. Verify that the cursor lands on the correct UI elements:
   - Play button for starting new games
   - Time control buttons (1+0 and 3+0)
4. Note any misalignments and adjust constants if needed

## Constants Used

The script uses the same constants as the main application:
- `START_X = 543` - Chess board left edge
- `START_Y = 179` - Chess board top edge  
- `STEP = 106` - Size of each chess square

These may need adjustment if the chess.com interface changes or if using a different screen resolution.

## Troubleshooting

- **Import errors**: Make sure you have `pyautogui` and `numpy` installed (`pip install pyautogui numpy`)
- **Position inaccuracies**: Check that START_X, START_Y, and STEP values match your screen setup
- **Permission errors**: On some systems, you may need to grant accessibility permissions for cursor control
- **Display scaling**: If using high-DPI displays, coordinate calculations may need adjustment

## Fixed Issues

- ✅ **File path issues**: Script now uses hardcoded constants instead of importing from modules with relative path dependencies
- ✅ **Custom cursor dependency**: Replaced with simplified PyAutoGUI-based cursor movement
- ✅ **Working directory issues**: Script handles path resolution automatically
