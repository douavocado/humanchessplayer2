# Auto-Calibration Module

Automated chess board detection and UI calibration for Lichess. This module fits coordinates for the board, clocks, and other UI elements, and extracts piece/digit templates from saved screenshots.

## Quick Start (Offline Fitting)

The preferred way to calibrate is using a directory of screenshots:

```bash
# Fit coordinates and extract all templates from a directory with debug visualisations
python -m auto_calibration.offline_fitter --dir ./my_screenshots/ --profile laptop --extract-all --visualise
```

### Fitting from a single screenshot

If you only have one screenshot, you must provide a state hint (e.g., `play`, `start1`, `end1`):

```bash
# Fit from a single file
python -m auto_calibration.offline_fitter --file ./shot.png --state play --profile laptop --visualise
```

## Testing Calibration

Verify your calibration accuracy against ground truth screenshots (requires `_fen.txt` files for ground truth):

```bash
# Run read-back test to verify FEN and Clock OCR
python -m auto_calibration.calibration_readback_test --screenshots ./my_screenshots/ --profile laptop --debug
```

### Key Arguments for Read-back Test:
- `--bottom [w|b|auto]`: Set which colour is at the bottom (defaults to auto-detect).
- `--debug`: Saves detailed crop and OCR debug images to `calibration_debug/latest/`.
- `--create-new`: Saves results in a timestamped folder instead of overwriting `latest`.

## Directory Structure

### Core Scripts
*   `offline_fitter.py`: Main entry point for fitting calibration from screenshots.
*   `calibration_readback_test.py`: Verifies calibration accuracy against ground truth.
*   `board_detector.py`: Core logic for finding the chess board using colour segmentation.
*   `clock_detector.py`: Core logic for finding and reading clocks.
*   `coordinate_calculator.py`: Calculates final relative coordinates for all UI elements.
*   `template_extractor.py`: Utility for extracting piece and digit templates from screenshots.
*   `visualiser.py`: Generates the debug visualisation overlays found in `calibration_outputs/`.
*   `colour_extractor.py`: Extracts theme colours (board, highlights) from the UI.
*   `panel_detector.py`: Detects UI panels and move lists.
*   `button_detector.py`: Dynamically detects Lichess UI buttons (Play, Rematch, etc.).

### Data and Outputs
*   `calibrations/`: Stores generated JSON profiles (e.g., `laptop.json`).
*   `templates/`: Stores extracted piece and digit images per profile.
*   `calibration_outputs/`: Timestamped folders containing debug visualisations of the fitting process.
*   `calibration_debug/`: Output directory for `calibration_readback_test.py` verification images.
*   `offline_screenshots/`: Recommended location for storing calibration screenshots.
*   `debug_calibration.ipynb`: Jupyter notebook for interactive debugging of calibration logic.
