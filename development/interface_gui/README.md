# Screenshot Tracker GUI

A minimalist GUI application for mouse tracking and screenshot capture functionality.

## Features

- **Tracking Mode**: Records mouse coordinates when a configurable hotkey is pressed
- **Screenshot Capture**: Takes screenshots of specified bounding boxes using fastgrab
- **Configurable Hotkeys**: Customise both tracking and screenshot keys
- **Real-time Display**: Shows captured screenshots directly in the GUI
- **Timestamped Output**: All events are logged with timestamps

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python screenshot_tracker.py
```

## Usage

### Mouse Tracking
1. Enable "Tracking Mode" by ticking the checkbox
2. Set your preferred tracking key (default: 'space')
3. Click "Update" to apply the new key
4. Press the tracking key anywhere on your system to record mouse coordinates

### Screenshot Capture
1. Enter the bounding box coordinates (x, y, width, height)
2. Set your preferred screenshot key (default: 'f1')
3. Click "Update" to apply the new key
4. Press the screenshot key anywhere on your system to capture the specified area
5. The screenshot will be displayed in the right panel

### Manual Screenshot
You can also take screenshots manually by clicking the "Take Screenshot Now" button.

## Key Configuration

- **Tracking Key**: Any single character ('a', 'b', etc.) or special key names ('space', 'enter', 'ctrl', etc.)
- **Screenshot Key**: Function keys ('f1', 'f2', etc.) or any other key combination

## Dependencies

- `fastgrab`: High-performance screen capture
- `pynput`: Global hotkey detection
- `pillow`: Image processing and display
- `tkinter`: GUI framework (built-in with Python)

## Notes

- The application uses global hotkeys, so they work system-wide
- Screenshots are scaled down for display if they're too large
- All actions are logged with timestamps in the output panel
- The GUI shows the availability status of fastgrab in the status bar 