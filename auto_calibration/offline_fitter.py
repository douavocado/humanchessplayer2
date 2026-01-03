#!/usr/bin/env python3
"""
Offline Fitting Module

Fits calibration from saved screenshots instead of live capture.
Useful for:
- Capturing rare game states (stalemate, etc.)
- Calibrating on a different machine than where screenshots were taken
- Re-running calibration without an active game
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import json
import re

from .utils import (
    load_image,
    get_screenshots_directory,
    remove_background_colours,
    get_output_directory,
    ensure_directory,
)
from .board_detector import BoardDetector
from .clock_detector import ClockDetector
from .coordinate_calculator import CoordinateCalculator
from .visualiser import CalibrationVisualiser
from .config import save_config
from .template_extractor import TemplateExtractor
from .colour_extractor import extract_colour_scheme


def detect_digit_positions(clock_image: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Detect digit positions within a clock image.
    
    Analyses the clock to find where the 4 time digits (MM:SS) are located,
    returning positions as fractions of clock width for resolution independence.
    
    Args:
        clock_image: BGR or grayscale clock image
        
    Returns:
        Dictionary with d1_start, d1_end, d2_start, d2_end, d3_start, d3_end, d4_start, d4_end
        as fractions of width (0.0 to 1.0), or None if detection failed.
    """
    if clock_image is None:
        return None
    
    # Convert to grayscale and process
    if clock_image.ndim == 3:
        processed = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
    else:
        processed = clock_image.copy()
    
    if processed.size == 0 or processed.ndim != 2:
        return None
    
    img_height, img_width = processed.shape
    
    # Find dark content (digits are dark on light background)
    dark_mask = processed < 128
    col_sums = dark_mask.sum(axis=0)
    
    # Find content regions
    threshold = 2
    content = col_sums > threshold
    transitions = np.where(np.diff(content.astype(int)) != 0)[0]
    
    if len(transitions) < 6:
        return None
    
    # Extract and split regions (handle "1:" being detected as single region)
    raw_regions = []
    for i in range(0, len(transitions) - 1, 2):
        start = transitions[i] + 1
        end = transitions[i + 1] + 1 if i + 1 < len(transitions) else img_width
        if end - start > 3:
            raw_regions.append((start, end))
    
    # Split regions where there's a big jump in column sums (digit vs colon)
    regions = []
    for start, end in raw_regions:
        region_sums = col_sums[start:end]
        if len(region_sums) == 0:
            continue
        
        max_sum = region_sums.max()
        min_sum = region_sums[region_sums > 0].min() if (region_sums > 0).any() else 0
        
        if max_sum > 20 and min_sum < 10 and max_sum > 3 * min_sum:
            # Split thin digit from colon/decimal
            for j in range(len(region_sums) - 1):
                if region_sums[j] < 10 and region_sums[j+1] > 20:
                    regions.append((start, start + j + 1))
                    regions.append((start + j + 1, end))
                    break
            else:
                regions.append((start, end))
        else:
            regions.append((start, end))
    
    # Filter for digit-sized regions
    digit_regions = []
    for start, end in regions:
        width = end - start
        if width < 3:
            continue
        avg_col_sum = col_sums[start:end].mean()
        if width >= 6 and avg_col_sum < 25:
            digit_regions.append((start, end))
    
    if len(digit_regions) < 4:
        return None
    
    # Add small padding and convert to fractions
    padding_left = 3
    padding_right = 2
    
    def to_fraction(start, end):
        s = max(0, start - padding_left)
        e = min(img_width, end + padding_right)
        return s / img_width, e / img_width
    
    d1_frac = to_fraction(digit_regions[0][0], digit_regions[0][1])
    d2_frac = to_fraction(digit_regions[1][0], digit_regions[1][1])
    d3_frac = to_fraction(digit_regions[2][0], digit_regions[2][1])
    d4_frac = to_fraction(digit_regions[3][0], digit_regions[3][1])
    
    return {
        'd1_start': d1_frac[0], 'd1_end': d1_frac[1],
        'd2_start': d2_frac[0], 'd2_end': d2_frac[1],
        'd3_start': d3_frac[0], 'd3_end': d3_frac[1],
        'd4_start': d4_frac[0], 'd4_end': d4_frac[1]
    }


class OfflineFitter:
    """
    Fits calibration from saved screenshots.
    
    Supports multiple screenshots with different game states,
    combining them into a single comprehensive configuration.
    
    Can also extract templates and colours for complete profile calibration.
    """
    
    def __init__(self, screenshots_dir: Optional[Path] = None, profile_name: Optional[str] = None):
        """
        Initialise offline fitter.
        
        Args:
            screenshots_dir: Directory containing screenshots.
                           If None, uses default location.
            profile_name: Name of the profile for template storage.
                         If None, templates go to generic location.
        """
        if screenshots_dir is None:
            screenshots_dir = get_screenshots_directory()
        self.screenshots_dir = Path(screenshots_dir)
        self.profile_name = profile_name
        
        self.board_detector = BoardDetector()
        self.clock_detector = ClockDetector()
        self.calculator = CoordinateCalculator()
        
        # Template extractor for the profile
        if profile_name:
            template_dir = Path(__file__).parent / "templates" / profile_name
        else:
            template_dir = Path(__file__).parent / "templates"
        self.template_extractor = TemplateExtractor(template_dir)
    
    def fit_from_screenshots(self, 
                            screenshot_paths: Optional[List[str]] = None,
                            state_hints: Optional[Dict[str, str]] = None,
                            visualise: bool = False,
                            output_root: Optional[Path] = None) -> Optional[Dict]:
        """
        Fit calibration from multiple screenshots.
        
        Args:
            screenshot_paths: List of paths to screenshots.
                            If None, uses all images in screenshots_dir.
            state_hints: Optional mapping of filename to state hint.
                        e.g., {'screenshot_001.png': 'play', 'screenshot_002.png': 'start1'}
            visualise: Whether to save visualisations per screenshot.
            output_root: Optional root directory for visualisations.
        
        Returns:
            Complete calibration configuration, or None if failed.
        """
        base_output_dir: Optional[Path] = None
        if visualise:
            base_output_dir = Path(output_root) if output_root else Path(get_output_directory())
            ensure_directory(base_output_dir)
        
        # Get screenshot paths
        if screenshot_paths is None:
            screenshot_paths = self._find_screenshots()
        
        if not screenshot_paths:
            print("No screenshots found")
            return None
        
        print(f"Found {len(screenshot_paths)} screenshots")
        
        # Process each screenshot
        all_detections = []
        board_detection = None
        
        for path in screenshot_paths:
            print(f"\nProcessing: {Path(path).name}")
            
            # Load image
            image = load_image(path)
            if image is None:
                print(f"  Failed to load image")
                continue
            
            # Detect board (use first successful detection)
            if board_detection is None:
                board_detection = self.board_detector.detect(image)
                if board_detection:
                    print(f"  Board detected: ({board_detection['x']}, {board_detection['y']}) "
                          f"size={board_detection['size']} conf={board_detection['confidence']:.2f}")
            
            if board_detection is None:
                print(f"  Skipping (no board detected)")
                continue
            
            # Detect clocks
            self.clock_detector.set_board(board_detection)
            clock_detection = self.clock_detector.detect(image)
            
            if clock_detection:
                # Get state hint from filename or provided hints
                filename = Path(path).name
                state_hint = None
                
                if state_hints and filename in state_hints:
                    state_hint = state_hints[filename]
                else:
                    state_hint = self._extract_state_from_filename(filename)
                
                all_detections.append({
                    'path': path,
                    'image': image,
                    'board': board_detection,
                    'clocks': clock_detection,
                    'state_hint': state_hint
                })
                
                print(f"  Clocks detected: {clock_detection['detection_count']} positions")
                if state_hint:
                    print(f"  State hint: {state_hint}")

                # Optional visualisations per screenshot/state
                if visualise and base_output_dir:
                    # Build coordinates for overlays, using state_hint to relabel clocks
                    coords_for_vis, relabelled_clocks = self._build_coordinates_for_visualisation(
                        board_detection, clock_detection, state_hint
                    )
                    state_name = state_hint or "unknown"
                    shot_dir = base_output_dir / state_name / Path(path).stem
                    ensure_directory(shot_dir)
                    visualiser = CalibrationVisualiser(shot_dir)
                    # Pass relabelled clocks so visualiser shows correct state labels
                    visualiser.visualise_all(image, board_detection, relabelled_clocks, coords_for_vis)
                    print(f"  Visualisation saved to: {visualiser.get_output_dir()}")
            else:
                print(f"  No clocks detected")
        
        if not all_detections:
            print("\nNo valid detections from any screenshot")
            return None
        
        # Combine detections
        combined = self._combine_detections(all_detections, board_detection)
        
        return combined

    def _build_coordinates_for_visualisation(
        self,
        board_detection: Dict,
        clock_detection: Optional[Dict],
        state_hint: Optional[str] = None
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Build coordinates for visual overlays without affecting the combined fit.
        
        If state_hint is provided, relabels the detected 'play' clocks to the
        appropriate state and only includes states from that family.
        
        Returns:
            (coordinates, relabelled_clock_detection) tuple
        """
        if board_detection is None:
            return None, None

        # Determine which state family we're in based on state_hint
        if state_hint:
            if state_hint.startswith('start'):
                state_family = 'start'
                family_states = ['start1', 'start2']
            elif state_hint.startswith('end'):
                state_family = 'end'
                family_states = ['end1', 'end2', 'end3']
            else:
                state_family = 'play'
                family_states = ['play']
        else:
            state_family = None
            family_states = None

        # If we have a state_hint, relabel detected 'play' clocks to that state
        if state_hint and clock_detection:
            relabelled_detection = {'detection_count': clock_detection.get('detection_count', 0)}
            if 'clock_x' in clock_detection:
                relabelled_detection['clock_x'] = clock_detection['clock_x']
            
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in clock_detection:
                    relabelled_detection[clock_type] = {}
                    for state, coords in clock_detection[clock_type].items():
                        # Relabel 'play' to the state_hint
                        new_state = state_hint if state == 'play' else state
                        relabelled_detection[clock_type][new_state] = coords
            
            clock_detection_to_use = relabelled_detection
        else:
            clock_detection_to_use = clock_detection

        self.calculator.set_board(board_detection)
        self.calculator.set_clocks(clock_detection_to_use)
        coordinates = self.calculator.calculate_all()

        # Only include clocks from the relevant state family
        if family_states:
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in coordinates:
                    # Filter to only family states
                    filtered = {s: c for s, c in coordinates[clock_type].items() 
                               if s in family_states}
                    coordinates[clock_type] = filtered

        return coordinates, clock_detection_to_use
    
    def _load_ground_truth(self, screenshot_path: str) -> Dict[str, Any]:
        """
        Load ground truth data from a corresponding _fen.txt file.
        Format:
          Line 1: FEN
          Line 2: top:SECONDS
          Line 3: bottom:SECONDS
          Line 4: result:white_win|black_win|draw (optional)
          Line 5: move:e2e4 (optional)
        """
        path = Path(screenshot_path)
        gt_file = path.parent / f"{path.stem}_fen.txt"
        
        gt = {'fen': None, 'top_time': None, 'bottom_time': None, 'result': None, 'move': None, 'side': None}
        
        if gt_file.exists():
            try:
                lines = gt_file.read_text().splitlines()
                if len(lines) >= 1:
                    gt['fen'] = lines[0].strip()
                
                for line in lines[1:]:
                    line = line.strip().lower()
                    if line.startswith('top:'):
                        gt['top_time'] = int(line.split(':')[1].strip())
                    elif line.startswith('bottom:'):
                        gt['bottom_time'] = int(line.split(':')[1].strip())
                    elif line.startswith('result:'):
                        gt['result'] = line.split(':')[1].strip()
                    elif line.startswith('move:'):
                        gt['move'] = line.split(':')[1].strip()
                    elif line.startswith('side:'):
                        gt['side'] = line.split(':')[1].strip()
            except Exception as e:
                print(f"  Warning: Error loading ground truth from {gt_file.name}: {e}")
                
        return gt

    def _find_screenshots(self) -> List[str]:
        """Find all screenshot files in the screenshots directory."""
        if not self.screenshots_dir.exists():
            return []
        
        extensions = ['.png', '.jpg', '.jpeg']
        screenshots = []
        
        for ext in extensions:
            screenshots.extend(self.screenshots_dir.glob(f'*{ext}'))
            screenshots.extend(self.screenshots_dir.glob(f'*{ext.upper()}'))
        
        return sorted([str(p) for p in screenshots])
    
    def _extract_state_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract game state hint from filename.
        """
        states = ['play', 'start1', 'start2', 'end1', 'end2', 'end3',
                  'start', 'end', 'resign', 'draw', 'stalemate', 'checkmate',
                  'white_win', 'black_win', 'whitewin', 'blackwin']
        
        filename_lower = filename.lower()
        
        for state in states:
            if state in filename_lower:
                if state == 'start':
                    return 'start1'
                elif state in ['end', 'resign', 'stalemate', 'checkmate']:
                    return 'end1'
                elif state in ['white_win', 'whitewin']:
                    return 'white_win'
                elif state in ['black_win', 'blackwin']:
                    return 'black_win'
                elif state == 'draw':
                    return 'draw'
                else:
                    return state
        
        return None
    
    def _combine_detections(self, detections: List[Dict],
                           board_detection: Dict) -> Dict:
        """
        Combine multiple screenshot detections into single config using
        state-aware grouping (start*, play, end*).
        """
        self.calculator.set_board(board_detection)

        def collect_positions(clock_type: str, labels: List[str], k_clusters: int = 1):
            """
            Collect clock positions for a state family and optionally cluster them.
            Labels length should match k_clusters when k_clusters > 1.
            """
            # Gather raw detections that match the state family
            samples = []
            for det in detections:
                clocks = det['clocks']
                hint = det.get('state_hint')
                if clock_type not in clocks:
                    continue
                for state_name, coords in clocks[clock_type].items():
                    actual_state = hint if hint and state_name == 'play' else state_name
                    if any(actual_state.startswith(prefix.rstrip('123')) for prefix in labels):
                        samples.append(coords)
            if not samples:
                return {}
            if k_clusters == 1:
                # Average all samples
                x = int(np.mean([s['x'] for s in samples]))
                y = int(np.mean([s['y'] for s in samples]))
                w = int(np.mean([s.get('width', 147) for s in samples]))
                h = int(np.mean([s.get('height', 44) for s in samples]))
                
                # STRICT FITTING: Tighten vertical coordinates to the actual digits
                # Extract crop and find horizontal sums to find content
                # (Assuming the first sample is representative for fitting)
                y_final, h_final = y, h
                try:
                    # Find which detection this sample belongs to
                    for det in detections:
                        img = det['image']
                        for s_name, s_coords in det['clocks'][clock_type].items():
                            if s_coords['y'] == y: # Found the matching image
                                crop = img[y:y+h, x:x+w]
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                if np.mean(binary) > 127: binary = 255 - binary
                                h_sums = np.sum(binary > 0, axis=1)
                                v_content = np.where(h_sums > (binary.shape[1] * 0.05))[0]
                                if len(v_content) > 0:
                                    v_start, v_end = v_content[0], v_content[-1]
                                    # Center the digits in a fixed height crop (e.g. 44px)
                                    target_h = 44 
                                    digit_h = v_end - v_start
                                    y_final = y + v_start - (target_h - digit_h) // 2
                                    h_final = target_h
                                break
                except:
                    pass
                
                return {labels[0]: {'x': x, 'y': y_final, 'width': w, 'height': h_final}}
            # Sort by y then split into k roughly equal groups (top to bottom)
            samples_sorted = sorted(samples, key=lambda c: (c['y'], c['x']))
            n = len(samples_sorted)
            if n < k_clusters:
                # Not enough samples; map what we have in order, leave others missing
                clustered = {}
                for i, sample in enumerate(samples_sorted):
                    if i >= len(labels):
                        break
                    
                    v_margin = int(sample.get('height', 44) * 0.1)
                    clustered[labels[i]] = {
                        'x': sample['x'],
                        'y': sample['y'] - v_margin,
                        'width': sample.get('width', 147),
                        'height': sample.get('height', 44) + v_margin * 2,
                    }
                return clustered
            # Split into k contiguous groups
            clusters = []
            size = n // k_clusters
            remainder = n % k_clusters
            start = 0
            for i in range(k_clusters):
                end = start + size + (1 if i < remainder else 0)
                clusters.append(samples_sorted[start:end])
                start = end
            clustered = {}
            for i, group in enumerate(clusters):
                if i >= len(labels) or not group:
                    continue
                
                x = int(np.mean([g['x'] for g in group]))
                y = int(np.mean([g['y'] for g in group]))
                w = int(np.mean([g.get('width', 147) for g in group]))
                h_avg = int(np.mean([g.get('height', 44) for g in group]))
                
                # STRICT FITTING: Tighten vertical coordinates to the actual digits
                y_final, h_final = y, h_avg
                try:
                    # Find a sample from this group to use for vertical alignment
                    sample = group[0]
                    for det in detections:
                        img = det['image']
                        found = False
                        for s_name, s_coords in det['clocks'][clock_type].items():
                            if s_coords['y'] == sample['y']:
                                crop = img[y:y+h_avg, x:x+w]
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                if np.mean(binary) > 127: binary = 255 - binary
                                h_sums = np.sum(binary > 0, axis=1)
                                v_content = np.where(h_sums > (binary.shape[1] * 0.05))[0]
                                if len(v_content) > 0:
                                    v_start, v_end = v_content[0], v_content[-1]
                                    target_h = 44
                                    digit_h = v_end - v_start
                                    y_final = y + v_start - (target_h - digit_h) // 2
                                    h_final = target_h
                                found = True
                                break
                        if found: break
                except:
                    pass
                
                clustered[labels[i]] = {
                    'x': x,
                    'y': y_final,
                    'width': w,
                    'height': h_final,
                }
            return clustered

        def build_clock_dict(clock_type: str):
            # Play: single average
            play_pos = collect_positions(clock_type, ['play'], k_clusters=1)
            # Start: two clusters
            start_pos = collect_positions(clock_type, ['start1', 'start2'], k_clusters=2)
            # End: three clusters
            end_pos = collect_positions(clock_type, ['end1', 'end2', 'end3'], k_clusters=3)
            combined = {}
            combined.update(play_pos)
            combined.update(start_pos)
            combined.update(end_pos)
            return combined

        # Build clocks with grouping
        bottom_clocks = build_clock_dict('bottom_clock')
        top_clocks = build_clock_dict('top_clock')

        clock_x_vals = []
        for det in detections:
            if 'clock_x' in det['clocks']:
                clock_x_vals.append(det['clocks']['clock_x'])
        clock_x = int(np.mean(clock_x_vals)) if clock_x_vals else (board_detection['x'] + board_detection['size'] + 29)

        combined_clocks = {
            'bottom_clock': bottom_clocks,
            'top_clock': top_clocks,
            'clock_x': clock_x,
            'detection_count': len(bottom_clocks) + len(top_clocks)
        }

        self.calculator.set_clocks(combined_clocks)
        coordinates = self.calculator.calculate_all()

        # Estimate missing states to fill gaps
        estimated_clocks = self.calculator.estimate_missing_clock_states(combined_clocks)
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in estimated_clocks:
                if clock_type not in coordinates:
                    coordinates[clock_type] = {}
                for state, coords in estimated_clocks[clock_type].items():
                    if state not in coordinates[clock_type]:
                        coordinates[clock_type][state] = coords

        config = {
            'calibration_info': {
                'method': 'offline_fitting',
                'screenshots_used': len(detections),
                'board_confidence': board_detection.get('confidence', 0),
                'clock_states_detected': list(bottom_clocks.keys()) + list(top_clocks.keys())
            },
            'coordinates': coordinates
        }

        digit_positions = self._detect_digit_positions_from_detections(detections, coordinates)
        if digit_positions:
            config['digit_positions'] = digit_positions
            print(f"✓ Digit positions calibrated")
        else:
            print("⚠ Could not calibrate digit positions, will use fallback")

        board_width = board_detection.get('size', 848)
        piece_size = board_width // 8
        config['template_info'] = {
            'piece_size': piece_size,
            'digit_size': [30, 44],
        }

        return config
    
    def extract_templates_and_colours(
        self,
        detections: List[Dict],
        board_detection: Dict,
        config: Dict
    ) -> Dict:
        """
        Extract piece templates, digit templates, and colour scheme from detections.
        Collects multiple instances of each template and averages them with sub-pixel alignment.
        """
        print("\n📋 Extracting templates and colours...")
        
        board_x = board_detection['x']
        board_y = board_detection['y']
        board_size = board_detection['size']
        step = board_size // 8
        
        # Buffers to collect multiple instances for averaging
        piece_instances = {p: [] for p in "RNBQKPrnbqkp"}
        digit_instances = {str(i): [] for i in range(10)}
        result_instances = {'white_win': [], 'black_win': [], 'draw': []}
        
        # Buffer for colour schemes from different frames
        colour_schemes = []
        
        for detection in detections:
            image = detection.get('image')
            path = detection.get('path')
            
            if image is None or path is None:
                continue
            
            # Load ground truth
            gt = self._load_ground_truth(path)
            fen = gt.get('fen')
            gt_move = gt.get('move')
            state_hint = detection.get('state_hint', '')
            
            # Extract board region
            board_img = image[board_y:board_y + board_size, board_x:board_x + board_size]
            if board_img.size == 0:
                continue
            
            # 1. Collect Piece Instances
            bottom = gt.get('side', 'w')
            if fen:
                # Extract all pieces in this board (returns one per type found)
                pieces = self.template_extractor.extract_pieces_from_fen(board_img, fen, bottom=bottom)
                for piece, template in pieces.items():
                    if piece in piece_instances:
                        piece_instances[piece].append(template)
            
            # 2. Collect Digit Instances
            digit_positions = config.get('digit_positions')
            if digit_positions:
                times = {'top_clock': gt.get('top_time'), 'bottom_clock': gt.get('bottom_time')}
                for clock_type in ['bottom_clock', 'top_clock']:
                    if clock_type in detection['clocks'] and times[clock_type] is not None:
                        known_time = times[clock_type]
                        # Use first state available for this clock type
                        state = next(iter(detection['clocks'][clock_type]))
                        c = detection['clocks'][clock_type][state]
                        clock_img = image[c['y']:c['y']+c['height'], c['x']:c['x']+c['width']]
                        
                        if clock_img.size > 0:
                            digits = self.template_extractor.extract_digits_from_clock(
                                clock_img, digit_positions, known_time
                            )
                            for val, template in digits.items():
                                if str(val) in digit_instances:
                                    digit_instances[str(val)].append(template)
            
            # 3. Collect Result Instances
            # Determine result from:
            # 1. Ground truth 'result' field
            # 2. State hint (if it's already 'white_win', etc.)
            # 3. FEN (if it's a checkmate/draw position)
            result_type = gt.get('result')
            
            if not result_type:
                if state_hint in result_instances:
                    result_type = state_hint
                elif fen:
                    import chess
                    try:
                        board = chess.Board(fen)
                        if board.is_game_over():
                            res = board.result()
                            if res == "1-0": result_type = 'white_win'
                            elif res == "0-1": result_type = 'black_win'
                            elif res == "1/2-1/2": result_type = 'draw'
                    except:
                        pass
            
            if result_type and 'result_region' in config['coordinates']:
                r = config['coordinates']['result_region']
                res_img = image[r['y']:r['y']+r['height'], r['x']:r['x']+r['width']]
                if res_img.size > 0:
                    result_instances[result_type].append(res_img)
            
            # 4. Extract Colour Scheme
            # Use FEN and ground truth move if available for precise sampling
            highlighted_squares = None
            if gt_move:
                import chess
                try:
                    move = chess.Move.from_uci(gt_move)
                    highlighted_squares = [move.from_square, move.to_square]
                except Exception:
                    pass
            
            current_scheme = extract_colour_scheme(
                board_img, step=step, fen=fen, bottom=bottom, 
                highlighted_squares=highlighted_squares
            )
            if current_scheme:
                colour_schemes.append(current_scheme)

        # --- Finalize Colour Scheme (Average across frames) ---
        if colour_schemes:
            final_scheme = {}
            
            # Base colours: average all
            for key in ['light_square', 'dark_square', 'premove_light', 'premove_dark']:
                vals = [s[key] for s in colour_schemes if key in s]
                if vals:
                    final_scheme[key] = np.mean(vals, axis=0).astype(int).tolist()
            
            # Highlights: prefer extracted over estimated
            from .colour_extractor import _estimate_highlight_colour
            for key in ['highlight_light', 'highlight_dark']:
                # Find "real" highlights (those that were actually extracted)
                real_highlights = [s[key] for s in colour_schemes if key in s]
                
                if real_highlights:
                    final_scheme[key] = np.mean(real_highlights, axis=0).astype(int).tolist()
                    print(f"  ✓ {key} extracted from {len(real_highlights)} frames")
                else:
                    # Fallback to estimate based on the finalized base colour
                    is_light = (key == 'highlight_light')
                    base_colour = final_scheme['light_square'] if is_light else final_scheme['dark_square']
                    final_scheme[key] = _estimate_highlight_colour(base_colour, is_light=is_light)
                    print(f"  ⚠ {key} estimated (no highlights found in any frame)")
            
            config['colour_scheme'] = final_scheme
            print(f"  ✓ Colour scheme finalized (from {len(colour_schemes)} frames)")
        else:
            print("  ⚠ Could not extract colour scheme, will use fallback")

        # --- Finalize Piece Templates (use first clean instance) ---
        pieces_extracted = 0
        print(f"  Finalizing pieces (selecting best instance from {sum(len(v) for v in piece_instances.values())} total)...")
        for piece, instances in piece_instances.items():
            if not instances:
                continue
            
            # Use the first instance directly instead of averaging
            # Averaging tends to blur templates when pieces come from different
            # backgrounds (light/dark squares, highlighted/normal, etc.)
            template = instances[0]
            if template is not None:
                if self.template_extractor.save_piece_template(piece, template, overwrite=True):
                    pieces_extracted += 1

        # --- Finalize Digit Templates (pick best instance) ---
        digits_extracted = 0
        print(f"  Finalizing digits (selecting best instance from {sum(len(v) for v in digit_instances.values())} total)...")
        for digit_str, instances in digit_instances.items():
            if not instances:
                continue
            
            digit_val = int(digit_str)
            
            # Find the "best" instance. For digits, we want the one that is most complete.
            # Trimming logic in extract_digits_from_clock means the widest image 
            # (before resizing) is likely the most complete one.
            # But since they are all resized to 30x44, we check which one has the 
            # highest pixel density (most white pixels), which often indicates a full digit.
            best_instance = instances[0]
            max_density = 0
            for inst in instances:
                density = np.sum(inst > 127)
                if density > max_density:
                    max_density = density
                    best_instance = inst
            
            if best_instance is not None:
                if self.template_extractor.save_digit_template(digit_val, best_instance, overwrite=True):
                    digits_extracted += 1

        # --- Fill in missing digits from fallbacks ---
        missing_digits = [i for i in range(10) if not self.template_extractor.progress["digits"][str(i)]]
        if missing_digits:
            print(f"  Generating {len(missing_digits)} missing digits ({', '.join(map(str, missing_digits))}) from fallbacks...")
            for digit in missing_digits:
                if self.template_extractor.generate_digit_from_fallback(digit):
                    digits_extracted += 1

        # --- Finalize Result Templates (Align & Average) ---
        results_extracted = 0
        print(f"  Finalizing results (averaging multiple instances)...")
        for res_type, instances in result_instances.items():
            if not instances:
                continue
            
            # Result templates are in color (BGR)
            # Need to align and average color images
            if len(instances) == 1:
                avg_template = instances[0]
            else:
                # Basic averaging for color images (no sub-pixel alignment for now to keep it simple,
                # but result region is usually very stable)
                stacked = np.stack(instances, axis=0)
                avg_template = np.median(stacked, axis=0).astype(np.uint8)
            
            if self.template_extractor.save_result_template(res_type, avg_template, overwrite=True):
                results_extracted += 1
        
        # Update template info
        if 'template_info' not in config:
            config['template_info'] = {}
        config['template_info']['extracted_from'] = str(self.screenshots_dir)
        config['template_info']['pieces_extracted'] = pieces_extracted
        config['template_info']['digits_extracted'] = digits_extracted
        config['template_info']['results_extracted'] = results_extracted
        
        print(f"\n{self.template_extractor.get_completion_summary()}")
        
        return config
    
    def _extract_digit_templates(self, detections: List[Dict], config: Dict) -> int:
        """
        Extract digit templates from clock images using ground truth times.
        """
        digit_positions = config.get('digit_positions')
        if not digit_positions:
            return 0
        
        extracted = 0
        
        for detection in detections:
            image = detection.get('image')
            clocks = detection.get('clocks', {})
            path = detection.get('path')
            
            if image is None or path is None:
                continue
                
            # Load ground truth for this specific screenshot
            gt = self._load_ground_truth(path)
            
            # Use times from ground truth if available
            times = {
                'top_clock': gt.get('top_time'),
                'bottom_clock': gt.get('bottom_time')
            }
            
            # Extract clock region and digits
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type not in clocks or times[clock_type] is None:
                    continue
                
                known_time = times[clock_type]
                
                for state, clock_coords in clocks[clock_type].items():
                    x = clock_coords.get('x', 0)
                    y = clock_coords.get('y', 0)
                    w = clock_coords.get('width', 220)
                    h = clock_coords.get('height', 40)
                    
                    clock_img = image[y:y+h, x:x+w]
                    if clock_img.size == 0:
                        continue
                    
                    count = self.template_extractor.extract_digits_from_known_time(
                        clock_img, digit_positions, known_time, overwrite=True
                    )
                    extracted += count
        
        return extracted
    
    def _detect_digit_positions_from_detections(self, detections: List[Dict],
                                                 coordinates: Dict) -> Optional[Dict]:
        """
        Detect digit positions from the best available clock image.
        
        Args:
            detections: List of detection results with images.
            coordinates: Calculated coordinates.
            
        Returns:
            Digit positions as fractions, or None if detection failed.
        """
        # Try to find a 'start1' or 'start2' detection as these show initial time
        for detection in detections:
            if detection['state_hint'] in ['start1', 'start2']:
                image = detection.get('image')
                clocks = detection.get('clocks', {})
                
                if image is not None and 'bottom_clock' in clocks:
                    # Extract clock region from image
                    for state, clock_coords in clocks['bottom_clock'].items():
                        if 'x' in clock_coords and 'y' in clock_coords:
                            x = clock_coords['x']
                            y = clock_coords['y']
                            w = clock_coords.get('width', 220)
                            h = clock_coords.get('height', 40)
                            
                            clock_img = image[y:y+h, x:x+w]
                            if clock_img.size > 0:
                                positions = detect_digit_positions(clock_img)
                                if positions:
                                    return positions
        
        # Fallback: try any detection
        for detection in detections:
            image = detection.get('image')
            clocks = detection.get('clocks', {})
            
            if image is not None and 'bottom_clock' in clocks:
                for state, clock_coords in clocks['bottom_clock'].items():
                    if 'x' in clock_coords and 'y' in clock_coords:
                        x = clock_coords['x']
                        y = clock_coords['y']
                        w = clock_coords.get('width', 220)
                        h = clock_coords.get('height', 40)
                        
                        clock_img = image[y:y+h, x:x+w]
                        if clock_img.size > 0:
                            positions = detect_digit_positions(clock_img)
                            if positions:
                                return positions
        
        return None
    
    def fit_from_single_screenshot(self, screenshot_path: str,
                                   state_hint: Optional[str] = None,
                                   visualise: bool = True) -> Optional[Dict]:
        """
        Fit calibration from a single screenshot.
        
        Args:
            screenshot_path: Path to screenshot.
            state_hint: Optional state hint for the screenshot.
            visualise: Whether to create debug visualisations.
        
        Returns:
            Calibration configuration, or None if failed.
        """
        # Load image
        image = load_image(screenshot_path)
        if image is None:
            print(f"Failed to load: {screenshot_path}")
            return None
        
        print(f"Processing: {screenshot_path}")
        
        # Detect board
        board_detection = self.board_detector.detect(image)
        if board_detection is None:
            print("Board detection failed")
            return None
        
        print(f"Board: ({board_detection['x']}, {board_detection['y']}) "
              f"size={board_detection['size']} conf={board_detection['confidence']:.2f}")
        
        # Detect clocks
        self.clock_detector.set_board(board_detection)
        clock_detection = self.clock_detector.detect(image)
        
        if clock_detection:
            print(f"Clocks: {clock_detection['detection_count']} positions detected")
        else:
            print("Clock detection failed")
        
        # Calculate coordinates
        self.calculator.set_board(board_detection)
        self.calculator.set_clocks(clock_detection)
        coordinates = self.calculator.calculate_all()
        
        # Estimate missing states
        if clock_detection:
            estimated = self.calculator.estimate_missing_clock_states(clock_detection)
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in estimated:
                    for state, coords in estimated[clock_type].items():
                        if state not in coordinates.get(clock_type, {}):
                            if clock_type not in coordinates:
                                coordinates[clock_type] = {}
                            coordinates[clock_type][state] = coords
        
        # Build config
        config = {
            'calibration_info': {
                'method': 'offline_single',
                'source_screenshot': screenshot_path,
                'board_confidence': board_detection.get('confidence', 0),
                'state_hint': state_hint
            },
            'coordinates': coordinates
        }
        
        # Visualise
        if visualise:
            visualiser = CalibrationVisualiser()
            outputs = visualiser.visualise_all(image, board_detection, 
                                               clock_detection, coordinates)
            print(f"\nDebug output saved to: {visualiser.get_output_dir()}")
        
        return config


def fit_from_screenshots(
    screenshots_dir: Optional[str] = None,
    save_to_config: bool = True,
    output_path: Optional[str] = None,
    profile_name: Optional[str] = None,
    extract_all: bool = False,
    visualise: bool = False,
    output_root: Optional[str] = None
) -> Optional[Dict]:
    """
    Convenience function to fit from screenshots directory.
    
    Args:
        screenshots_dir: Directory with screenshots.
        save_to_config: Whether to save result to chess_config.json.
        output_path: Optional output file path. If None, uses default.
        profile_name: Name of the profile (for template storage).
        extract_all: If True, also extract templates and colours.
        visualise: Whether to save visualisations per screenshot.
        output_root: Optional root directory for visualisations.
    
    Returns:
        Configuration dictionary, or None if failed.
    """
    fitter = OfflineFitter(
        Path(screenshots_dir) if screenshots_dir else None,
        profile_name=profile_name
    )
    config = fitter.fit_from_screenshots(
        visualise=visualise,
        output_root=Path(output_root) if output_root else None
    )
    
    if config and extract_all:
        # Re-process to get detections for template extraction
        # This is a bit redundant but keeps the API clean
        screenshot_paths = fitter._find_screenshots()
        
        detections = []
        board_detection = None
        
        for path in screenshot_paths:
            image = load_image(path)
            if image is None:
                continue
            
            if board_detection is None:
                board_detection = fitter.board_detector.detect(image)
            
            if board_detection:
                fitter.clock_detector.set_board(board_detection)
                clock_detection = fitter.clock_detector.detect(image)
                
                if clock_detection:
                    state_hint = fitter._extract_state_from_filename(Path(path).name)
                    detections.append({
                        'path': path,
                        'image': image,
                        'board': board_detection,
                        'clocks': clock_detection,
                        'state_hint': state_hint
                    })
        
        if detections and board_detection:
            config = fitter.extract_templates_and_colours(detections, board_detection, config)
    
    if config and save_to_config:
        saved_path = save_config(config, output_path=output_path)
        print(f"\nConfiguration saved to: {saved_path}")
    
    return config


def fit_from_single(screenshot_path: str,
                   state_hint: Optional[str] = None,
                   save_to_config: bool = True,
                   output_path: Optional[str] = None,
                   visualise: bool = True) -> Optional[Dict]:
    """
    Convenience function to fit from single screenshot.
    
    Args:
        screenshot_path: Path to screenshot.
        state_hint: Optional state hint.
        save_to_config: Whether to save result to chess_config.json.
        output_path: Optional output file path.
        visualise: Whether to create debug visualisations.
    
    Returns:
        Configuration dictionary, or None if failed.
    """
    fitter = OfflineFitter()
    config = fitter.fit_from_single_screenshot(screenshot_path, state_hint, visualise=visualise)
    
    if config and save_to_config:
        saved_path = save_config(config, output_path=output_path)
        print(f"\nConfiguration saved to: {saved_path}")
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Offline calibration fitter - fits coordinates, extracts templates and colours"
    )
    parser.add_argument("--dir", type=str, help="Screenshots directory")
    parser.add_argument("--file", type=str, help="Single screenshot file")
    parser.add_argument("--state", type=str, help="State hint for single file (required when using --file)")
    parser.add_argument("--no-save", action="store_true", help="Don't save to config")
    parser.add_argument("--extract-all", action="store_true",
                       help="Extract templates and colours in addition to coordinates")
    parser.add_argument("--visualise", "--visualize", action="store_true",
                       help="Save debug visualisations in calibration_outputs/")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--profile", type=str,
                             help="Save calibration to a named profile (auto_calibration/calibrations/<profile>.json)")
    output_group.add_argument("--output", type=str,
                             help="Save calibration to an explicit JSON path")
    
    args = parser.parse_args()

    resolved_output: Optional[str] = None
    profile_name: Optional[str] = None
    
    if not args.no_save:
        if getattr(args, "output", None):
            resolved_output = args.output
        elif getattr(args, "profile", None):
            profile_name = args.profile
            resolved_output = str(Path(__file__).parent / "calibrations" / f"{args.profile}.json")
    
    if args.file:
        if not args.state:
            parser.error("--state is required when using --file")
        fit_from_single(
            args.file, 
            args.state, 
            not args.no_save, 
            output_path=resolved_output,
            visualise=args.visualise
        )
    elif args.dir:
        fit_from_screenshots(
            args.dir,
            not args.no_save,
            output_path=resolved_output,
            profile_name=profile_name,
            extract_all=args.extract_all,
            visualise=args.visualise
        )
    else:
        # Use default screenshots directory
        fit_from_screenshots(
            None,
            not args.no_save,
            output_path=resolved_output,
            profile_name=profile_name,
            extract_all=args.extract_all,
            visualise=args.visualise
        )
