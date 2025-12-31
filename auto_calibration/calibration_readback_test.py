#!/usr/bin/env python3
"""
Offline calibration read-back checker.

Given a directory of screenshots and a calibration profile, this script:
1) Loads the calibrated coordinates.
2) Crops the board and clocks from each screenshot.
3) Runs FEN and clock OCR to verify that the calibrated positions work.

Usage:
  python -m auto_calibration.calibration_readback_test \
    --screenshots auto_calibration/offline_screenshots/laptop \
    --profile laptop \
    --bottom w
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

# Set profile env before importing chessimage utilities so they pick up the profile
def set_profile_env(profile: Optional[str]) -> None:
    if profile:
        os.environ["HCP_CALIBRATION_PROFILE"] = profile


def extract_state_from_filename(filename: str) -> Optional[str]:
    """
    Mirror the heuristic used by offline_fitter to label screenshot state.
    """
    states = ['play', 'start1', 'start2', 'end1', 'end2', 'end3',
              'start', 'end', 'resign', 'draw', 'stalemate', 'checkmate']
    name = filename.lower()
    for state in states:
        if state in name:
            if state == 'start':
                return 'start1'
            if state in ['end', 'resign', 'draw', 'stalemate', 'checkmate']:
                return 'end1'
            return state
    return None


def load_ground_truth(screenshot_path: Path) -> dict:
    """
    Load ground truth data from corresponding _fen.txt file.
    Returns dict with 'fen', 'top_time', 'bottom_time', and 'result'.
    """
    res = {'fen': None, 'top_time': None, 'bottom_time': None, 'result': None}
    fen_file = screenshot_path.parent / f"{screenshot_path.stem}_fen.txt"
    if fen_file.exists():
        content = fen_file.read_text().strip()
        lines = content.split('\n')
        if lines:
            res['fen'] = lines[0].strip()
            for line in lines[1:]:
                line = line.lower().strip()
                if line.startswith('top:'):
                    try: res['top_time'] = int(line.split(':')[1])
                    except: pass
                elif line.startswith('bottom:'):
                    try: res['bottom_time'] = int(line.split(':')[1])
                    except: pass
                elif line.startswith('result:'):
                    res['result'] = line.split(':')[1].strip()
    return res


def extract_board_position(fen: str) -> str:
    """Extract just the board position part of a FEN (first field)."""
    return fen.split()[0] if fen else ""


def compare_board_positions(detected: str, ground_truth: str) -> dict:
    """
    Compare two board positions (FEN board part only).
    Returns dict with match statistics.
    """
    import chess
    
    detected_pos = extract_board_position(detected)
    gt_pos = extract_board_position(ground_truth)
    
    try:
        detected_board = chess.Board(detected_pos + " w - - 0 1")
        gt_board = chess.Board(gt_pos + " w - - 0 1")
    except Exception:
        return {'valid': False, 'correct': 0, 'total': 64, 'accuracy': 0.0, 'errors': []}
    
    correct = 0
    errors = []
    
    for sq in range(64):
        detected_piece = detected_board.piece_at(sq)
        gt_piece = gt_board.piece_at(sq)
        
        if detected_piece == gt_piece:
            correct += 1
        else:
            sq_name = chess.square_name(sq)
            det_str = detected_piece.symbol() if detected_piece else '.'
            gt_str = gt_piece.symbol() if gt_piece else '.'
            errors.append({
                'square': sq,
                'name': sq_name,
                'gt': gt_str,
                'det': det_str
            })
    
    return {
        'valid': True,
        'correct': correct,
        'total': 64,
        'accuracy': correct / 64 * 100,
        'errors': errors
    }


def load_dependencies(profile: Optional[str]):
    """
    Defer heavy imports until after env is set.
    """
    set_profile_env(profile)
    from auto_calibration.config import ChessConfig
    from auto_calibration.utils import load_image
    from chessimage.image_scrape_utils import (
        get_fen_from_image, read_clock, PIECE_TEMPLATES, STEP, PIECE_STEP,
        remove_background_colours, template_match_f, w_rook, INDEX_MAPPER,
        TEMPLATES, multitemplate_match_f, compare_result_images
    )
    import cv2
    import numpy as np
    return (ChessConfig, load_image, get_fen_from_image, read_clock,
            PIECE_TEMPLATES, STEP, PIECE_STEP, remove_background_colours,
            template_match_f, w_rook, INDEX_MAPPER, TEMPLATES,
            multitemplate_match_f, compare_result_images, cv2, np)


def iter_screenshots(root: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg")
    return sorted([p for p in root.iterdir() if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="Check FEN and clock OCR using calibrated coordinates.")
    parser.add_argument("--screenshots", "-s", type=str, required=True,
                        help="Directory of screenshots to test.")
    parser.add_argument("--profile", "-p", type=str, default=None,
                        help="Calibration profile name (e.g., laptop).")
    parser.add_argument("--bottom", "-b", type=str, default="auto", choices=["w", "b", "auto"],
                        help="Bottom colour for FEN extraction. 'auto' detects from bottom-left corner.")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast_mode for FEN extraction.")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Save debug images showing board crop and template info.")
    parser.add_argument("--create-new", action="store_true",
                        help="Save results in a timestamped folder instead of just 'latest'.")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit to first N screenshots.")
    args = parser.parse_args()

    screenshots_dir = Path(args.screenshots)
    if not screenshots_dir.exists():
        raise SystemExit(f"Screenshot directory not found: {screenshots_dir}")

    (ChessConfig, load_image, get_fen_from_image, read_clock,
     PIECE_TEMPLATES, STEP, PIECE_STEP, remove_background_colours,
     template_match_f, w_rook, INDEX_MAPPER, TEMPLATES,
     multitemplate_match_f, compare_result_images, cv2, np) = load_dependencies(args.profile)
    
    # Load result templates if profile is provided
    result_templates = None
    if args.profile:
        try:
            from auto_calibration.template_extractor import TemplateExtractor
            t_dir = Path(__file__).parent / "templates" / args.profile
            extractor = TemplateExtractor(template_dir=str(t_dir))
            result_templates = extractor.load_result_templates()
            if result_templates:
                print(f"Loaded result templates: {list(result_templates.keys())}")
            else:
                print(f"No result templates found in {t_dir}")
        except Exception as e:
            print(f"Warning: could not load result templates: {e}")
    
    # Reverse mapper for similarity score calculation
    SYMBOL_TO_INDEX = {v: k for k, v in INDEX_MAPPER.items()}

    def debug_clock_detection(clock_img, detected_time, gt_time, name, shot_debug_dir):
        """
        Create a combined debug image for clock detection mismatches.
        """
        if clock_img is None or clock_img.size == 0:
            return

        # Create clock folder
        clock_debug_dir = shot_debug_dir / "clock"
        clock_debug_dir.mkdir(exist_ok=True)

        # Process image same as read_clock
        if clock_img.ndim == 3:
            gray = cv2.cvtColor(clock_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = clock_img.copy()
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = 255 - binary

        # Find regions (same logic)
        projection = np.max(binary, axis=0) > 0
        regions = []
        start = None
        for i, val in enumerate(projection):
            if val and start is None: start = i
            elif not val and start is not None:
                if i - start > 2: regions.append((start, i))
                start = None
        if start is not None: regions.append((start, len(projection)))

        template_h, template_w = TEMPLATES.shape[1:3]
        
        # Identify digits in each region
        digit_info = []
        for r_start, r_end in regions:
            region_img = binary[:, r_start:r_end]
            h_proj = np.max(region_img, axis=1) > 0
            h_sum = np.sum(h_proj)
            if h_sum < binary.shape[0] * 0.3 or (r_end - r_start) < 2:
                continue
            region_resized = cv2.resize(region_img, (template_w, template_h), interpolation=cv2.INTER_AREA)
            
            # Get scores for all digits manually to get info
            T = TEMPLATES.astype(np.float32)
            I = region_resized.astype(np.float32)
            T_means = T.mean(axis=(1, 2), keepdims=True)
            I_mean = I.mean()
            T_primes = T - T_means
            I_prime = I - I_mean
            T_denom = (T_primes ** 2).sum(axis=(1, 2))
            I_denom = (I_prime ** 2).sum()
            denoms = np.sqrt(T_denom * I_denom) + 1e-10
            nums = (T_primes * I_prime).sum(axis=(1, 2))
            scores = nums / denoms
            
            best_digit = int(scores.argmax())
            best_score = float(scores[best_digit])
            
            digit_info.append({
                'img': region_resized,
                'detected': best_digit if best_score >= 0.4 else None,
                'score': best_score
            })

        num_detected = len(digit_info)
        
        # Truncate digit_info to exclude tenths/hundredths (only keep MM:SS part)
        if num_detected >= 5:
            digit_info = digit_info[:4]
            num_detected = 4

        # Infer ground truth digits based on number of regions (now max 4)
        gt_digits = []
        if gt_time is not None:
            mins = gt_time // 60
            secs = gt_time % 60
            d_m1, d_m2 = mins // 10, mins % 10
            d_s1, d_s2 = secs // 10, secs % 10
            
            if num_detected == 4:
                gt_digits = [d_m1, d_m2, d_s1, d_s2]
            elif num_detected == 3:
                gt_digits = [d_m2, d_s1, d_s2]
            elif num_detected == 2:
                gt_digits = [d_s1, d_s2]
            elif num_detected == 1:
                gt_digits = [d_s2]

        # Create debug visualization
        padding = 20
        label_width = 160 # Space for row labels
        v_spacing = 70
        col_width = 80
        digit_w = template_w
        digit_h = template_h
        row_height = digit_h + v_spacing
        
        header_h = binary.shape[0] + 120
        num_rows = 3
        
        canvas_w = max(binary.shape[1] + padding * 2, label_width + num_detected * col_width + padding, 450)
        canvas_h = header_h + num_rows * row_height + padding
        
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        
        # Draw binary full clock
        cv2.putText(canvas, f"Clock: {name} (Det: {detected_time}s, GT: {gt_time}s)", (padding, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        canvas[45:45+binary.shape[0], padding:padding+binary.shape[1]] = binary
        
        # Row Labels
        y_row1 = header_h
        y_row2 = y_row1 + row_height
        y_row3 = y_row2 + row_height
        
        cv2.putText(canvas, "Detected Digits", (padding, y_row1 + digit_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 200, 1)
        cv2.putText(canvas, "Detected Templates", (padding, y_row2 + digit_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 200, 1)
        cv2.putText(canvas, "GT Templates", (padding, y_row3 + digit_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 200, 1)

        # Draw digits and templates
        for i, info in enumerate(digit_info):
            x = label_width + i * col_width
            
            # Row 1: Actual digits from binary
            det_str = str(info['detected']) if info['detected'] is not None else "?"
            cv2.putText(canvas, f"Det: {det_str}", (x, y_row1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
            cv2.putText(canvas, f"Sc: {info['score']:.3f}", (x, y_row1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 1)
            canvas[y_row1:y_row1+digit_h, x:x+digit_w] = info['img']
            cv2.rectangle(canvas, (x-1, y_row1-1), (x+digit_w, y_row1+digit_h), 150, 1)

            # Row 2: Detected templates
            if info['detected'] is not None:
                cv2.putText(canvas, f"{info['detected']}", (x, y_row2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 180, 1)
                canvas[y_row2:y_row2+digit_h, x:x+digit_w] = TEMPLATES[info['detected']]
            cv2.rectangle(canvas, (x-1, y_row2-1), (x+digit_w, y_row2+digit_h), 100, 1)

            # Row 3: Ground truth templates
            gt_digit = gt_digits[i] if i < len(gt_digits) else None
            if gt_digit is not None:
                cv2.putText(canvas, f"{gt_digit}", (x, y_row3 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 180, 1)
                canvas[y_row3:y_row3+digit_h, x:x+digit_w] = TEMPLATES[gt_digit]
            cv2.rectangle(canvas, (x-1, y_row3-1), (x+digit_w, y_row3+digit_h), 100, 1)

        cv2.imwrite(str(clock_debug_dir / f"clock_debug_{name}.png"), canvas)

    def detect_bottom_colour(board_img) -> str:
        """
        Auto-detect which colour is at the bottom of the board.
        Checks multiple squares on the back rank for White pieces.
        Returns 'w' if white at bottom, 'b' if black at bottom.
        """
        h, w = board_img.shape[:2]
        step_x = w / 8.0
        step_y = h / 8.0
        
        # Check all 8 squares on the bottom rank for white pieces
        white_score = 0
        black_score = 0
        
        for col in range(8):
            # Extract square at bottom rank (row 7)
            y1, y2 = int(7 * step_y), int(8 * step_y)
            x1, x2 = int(col * step_x), int((col + 1) * step_x)
            sq = board_img[y1:y2, x1:x2]
            
            if sq.ndim == 3:
                sq_proc = remove_background_colours(sq).astype(np.uint8)
            else:
                sq_proc = sq.copy()
            
            sq_resized = cv2.resize(sq_proc, (PIECE_STEP, PIECE_STEP))
            
            # Check brightness of non-zero pixels
            mask = sq_resized > 1
            if mask.any():
                brightness = sq_resized[mask].mean()
                # White pieces are very bright (>150), Black are dark (<100)
                if brightness > 150:
                    white_score += 1
                elif brightness < 100:
                    black_score += 1
        
        # If more white pieces on bottom rank, white is at bottom
        if white_score > black_score:
            return 'w'
        # If more black pieces on bottom rank, black is at bottom
        if black_score > white_score:
            return 'b'
            
        # Fallback to checking the top rank if bottom was inconclusive
        white_score_top = 0
        black_score_top = 0
        for col in range(8):
            y1, y2 = 0, int(step_y)
            x1, x2 = int(col * step_x), int((col + 1) * step_x)
            sq = board_img[y1:y2, x1:x2]
            if sq.ndim == 3:
                sq_proc = remove_background_colours(sq).astype(np.uint8)
            else:
                sq_proc = sq.copy()
            sq_resized = cv2.resize(sq_proc, (PIECE_STEP, PIECE_STEP))
            mask = sq_resized > 1
            if mask.any():
                brightness = sq_resized[mask].mean()
                if brightness > 150: white_score_top += 1
                elif brightness < 100: black_score_top += 1
        
        # If more black pieces at top, white is at bottom
        if black_score_top > white_score_top:
            return 'w'
        return 'b'

    cfg_path = Path(__file__).parent / "calibrations"
    config_file = None
    if args.profile:
        candidate = cfg_path / f"{args.profile}.json"
        if candidate.exists():
            config_file = str(candidate)
    config = ChessConfig(config_file=config_file)

    coords = config.get_coordinates()
    board = coords['board']

    shots = iter_screenshots(screenshots_dir)
    if not shots:
        raise SystemExit(f"No screenshots found in {screenshots_dir}")

    if args.limit:
        shots = shots[:args.limit]

    print(f"Using calibration: {config_file or 'fallback'}")
    print(f"Board: x={board['x']}, y={board['y']}, size={board['width']}x{board['height']}")
    print(f"STEP={STEP}, PIECE_STEP={PIECE_STEP}")
    print(f"PIECE_TEMPLATES shape: {PIECE_TEMPLATES.shape}")
    print(f"Found {len(shots)} screenshots\n")

    # Debug output directory
    debug_dir = None
    if args.debug:
        debug_root = Path(__file__).parent / "calibration_debug"
        debug_dir = debug_root / "latest"
        
        # Wipe latest
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Debug output: {debug_dir}\n")
        
        # Save templates for inspection
        templates_dir = debug_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        piece_names = ['w_rook', 'w_knight', 'w_bishop', 'w_king', 'w_queen', 'w_pawn',
                       'b_rook', 'b_knight', 'b_bishop', 'b_king', 'b_queen', 'b_pawn']
        for i, name in enumerate(piece_names):
            cv2.imwrite(str(templates_dir / f"{name}.png"), PIECE_TEMPLATES[i])
        print(f"Saved {len(piece_names)} template images to {templates_dir}")

    # Track summary statistics
    all_comparisons = []
    clock_results = {'bottom': 0, 'top': 0, 'total': 0}
    result_stats = {'correct': 0, 'total_with_gt': 0, 'detected_total': 0}
    
    for shot in shots:
        state_hint = extract_state_from_filename(shot.name) or "play"
        img = load_image(str(shot))
        if img is None:
            print(f"[{shot.name}] ❌ could not load image")
            continue

        # Crop board
        bx, by, bw, bh = board['x'], board['y'], board['width'], board['height']
        board_img = img[by:by+bh, bx:bx+bw]

        if args.debug and debug_dir:
            shot_debug_dir = debug_dir / shot.stem
            shot_debug_dir.mkdir(exist_ok=True)
            
            # Save board crop
            cv2.imwrite(str(shot_debug_dir / "board_crop.png"), board_img)
            
            # Save board with background removed
            if board_img.ndim == 3:
                board_processed = remove_background_colours(board_img).astype(np.uint8)
            else:
                board_processed = board_img.copy()
            cv2.imwrite(str(shot_debug_dir / "board_processed.png"), board_processed)
            
            # Save individual squares
            squares_dir = shot_debug_dir / "squares"
            squares_dir.mkdir(exist_ok=True)
            step_x = bw / 8.0
            step_y = bh / 8.0
            for row in range(8):
                for col in range(8):
                    y1, y2 = int(row * step_y), int((row + 1) * step_y)
                    x1, x2 = int(col * step_x), int((col + 1) * step_x)
                    square_img = board_processed[y1:y2, x1:x2]
                    cv2.imwrite(str(squares_dir / f"r{row}_c{col}.png"), square_img)

        # Load ground truth if exists
        gt_data = load_ground_truth(shot)
        gt_fen = gt_data['fen']
        gt_top = gt_data['top_time']
        gt_bot = gt_data['bottom_time']

        # Determine bottom colour
        if args.bottom == 'auto' and gt_fen:
            # Try both orientations and pick the one that matches ground truth best
            fen_w = get_fen_from_image(board_img, bottom='w', fast_mode=args.fast)
            fen_b = get_fen_from_image(board_img, bottom='b', fast_mode=args.fast)
            
            comp_w = compare_board_positions(fen_w, gt_fen)
            comp_b = compare_board_positions(fen_b, gt_fen)
            
            # Debug: show both orientations
            if args.debug:
                print(f"    DEBUG: bottom='w' -> {comp_w['accuracy']:.1f}% ({comp_w['correct']}/64)")
                print(f"    DEBUG: bottom='b' -> {comp_b['accuracy']:.1f}% ({comp_b['correct']}/64)")
            
            if comp_w['accuracy'] >= comp_b['accuracy']:
                fen = fen_w
                detected_bottom = 'w'
                comparison = comp_w
            else:
                fen = fen_b
                detected_bottom = 'b'
                comparison = comp_b
        else:
            if args.bottom == 'auto':
                detected_bottom = detect_bottom_colour(board_img)
            else:
                detected_bottom = args.bottom
            
            fen = get_fen_from_image(board_img, bottom=detected_bottom, fast_mode=args.fast)
            comparison = None
            if gt_fen:
                comparison = compare_board_positions(fen, gt_fen)
        
        if gt_fen:
            all_comparisons.append({'name': shot.name, 'comparison': comparison, 'gt_fen': gt_fen})
            
        # Misclassified square debugging
        if args.debug and comparison and comparison['errors']:
            misclassified_dir = shot_debug_dir / "misclassified"
            misclassified_dir.mkdir(exist_ok=True)
            
            # Re-process board to get processed square images
            if board_img.ndim == 3:
                board_processed = remove_background_colours(board_img).astype(np.uint8)
            else:
                board_processed = board_img.copy()
            
            for err in comparison['errors']:
                sq_name = err['name']
                gt_sym = err['gt']
                det_sym = err['det']
                
                # Determine image row and col from chess square
                if detected_bottom == 'w':
                    row = 7 - (err['square'] // 8)
                    col = err['square'] % 8
                else:
                    row = 7 - ((63 - err['square']) // 8)
                    col = (63 - err['square']) % 8
                
                step_x = bw / 8.0
                step_y = bh / 8.0
                y1, y2 = int(row * step_y), int((row + 1) * step_y)
                x1, x2 = int(col * step_x), int((col + 1) * step_x)
                sq_img = board_processed[y1:y2, x1:x2]
                sq_resized = cv2.resize(sq_img, (PIECE_STEP, PIECE_STEP))
                
                # Helper for score calculation
                def get_score(sq, sym):
                    if sym == '.': return 0.0
                    idx = SYMBOL_TO_INDEX.get(sym)
                    if idx is None: return 0.0
                    return float(template_match_f(np.expand_dims(sq, 0), PIECE_TEMPLATES[idx])[0])

                score_det = get_score(sq_resized, det_sym)
                score_gt = get_score(sq_resized, gt_sym)

                # Create combined debug image
                padding = 20
                text_height = 60
                canvas_h = PIECE_STEP + text_height + padding
                canvas_w = PIECE_STEP * 3 + padding * 4
                
                canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                
                def draw_block(img, x, label, score=None):
                    # Labels
                    cv2.putText(canvas, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                    if score is not None:
                        cv2.putText(canvas, f"Score: {score:.4f}", (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 1)
                    
                    # Image
                    y_start = text_height
                    h, w = img.shape
                    canvas[y_start:y_start+h, x:x+w] = img
                    # Draw boundary to show template edges
                    cv2.rectangle(canvas, (x - 1, y_start - 1), (x + w, y_start + h), 150, 1)

                # 1. Actual square
                draw_block(sq_resized, padding, "Actual Square")
                
                # 2. Detected template
                det_img = np.zeros((PIECE_STEP, PIECE_STEP), dtype=np.uint8)
                if det_sym != '.':
                    idx = SYMBOL_TO_INDEX.get(det_sym)
                    if idx is not None: det_img = PIECE_TEMPLATES[idx]
                draw_block(det_img, PIECE_STEP + padding * 2, f"Detected: {det_sym}", score_det)
                
                # 3. Ground truth template
                gt_img = np.zeros((PIECE_STEP, PIECE_STEP), dtype=np.uint8)
                if gt_sym != '.':
                    idx = SYMBOL_TO_INDEX.get(gt_sym)
                    if idx is not None: gt_img = PIECE_TEMPLATES[idx]
                draw_block(gt_img, PIECE_STEP * 2 + padding * 3, f"Expected: {gt_sym}", score_gt)
                
                # Save combined debug image
                debug_img_path = misclassified_dir / f"{sq_name}_{gt_sym}_to_{det_sym}.png"
                cv2.imwrite(str(debug_img_path), canvas)

        # Crop clocks
        def crop_clock(clock_type: str):
            if clock_type in coords and state_hint in coords[clock_type]:
                c = coords[clock_type][state_hint]
            elif clock_type in coords and 'play' in coords[clock_type]:
                c = coords[clock_type]['play']
            else:
                return None
            return img[c['y']:c['y']+c['height'], c['x']:c['x']+c['width']]

        bot_clock_img = crop_clock('bottom_clock')
        top_clock_img = crop_clock('top_clock')
        bot_time = read_clock(bot_clock_img) if bot_clock_img is not None else None
        top_time = read_clock(top_clock_img) if top_clock_img is not None else None

        if args.debug and debug_dir:
            shot_debug_dir = debug_dir / shot.stem
            if bot_clock_img is not None:
                cv2.imwrite(str(shot_debug_dir / "bottom_clock.png"), bot_clock_img)
            if top_clock_img is not None:
                cv2.imwrite(str(shot_debug_dir / "top_clock.png"), top_clock_img)
            
            # Clock debugging if mismatch with GT
            if gt_bot is not None and bot_time != gt_bot:
                debug_clock_detection(bot_clock_img, bot_time, gt_bot, "bottom", shot_debug_dir)
            if gt_top is not None and top_time != gt_top:
                debug_clock_detection(top_clock_img, top_time, gt_top, "top", shot_debug_dir)
        
        # Track clock results
        clock_results['total'] += 1
        if bot_time is not None:
            clock_results['bottom'] += 1
        if top_time is not None:
            clock_results['top'] += 1
        
        # Track clock accuracy if GT exists
        if gt_bot is not None:
            clock_results.setdefault('bottom_correct', 0)
            clock_results.setdefault('bottom_with_gt', 0)
            clock_results['bottom_with_gt'] += 1
            if bot_time == gt_bot: clock_results['bottom_correct'] += 1
        if gt_top is not None:
            clock_results.setdefault('top_correct', 0)
            clock_results.setdefault('top_with_gt', 0)
            clock_results['top_with_gt'] += 1
            if top_time == gt_top: clock_results['top_correct'] += 1
        
        # Result detection (only for end states)
        detected_result = None
        result_score = 0
        gt_result = gt_data['result']
        if 'end' in state_hint:
            if result_templates and 'result_region' in coords:
                r = coords['result_region']
                res_y, res_x, res_h, res_w = r['y'], r['x'], r['height'], r['width']
                result_img = img[res_y:res_y+res_h, res_x:res_x+res_w]
                
                best_res = None
                best_score = 0
                for res_type, template in result_templates.items():
                    score = compare_result_images(result_img, template)
                    if score > best_score:
                        best_score = score
                        best_res = res_type
                
                if best_score > 0.7:
                    detected_result = best_res
                    result_score = best_score
                    result_stats['detected_total'] += 1
                
                if args.debug and debug_dir:
                    cv2.imwrite(str(shot_debug_dir / "result_region.png"), result_img)
            
            if gt_result:
                result_stats['total_with_gt'] += 1
                if detected_result == gt_result:
                    result_stats['correct'] += 1
        
        print(f"[{shot.name}] state={state_hint}, bottom={detected_bottom}")
        print(f"  Detected: {fen}")
        if gt_fen:
            print(f"  Expected: {gt_fen}")
            if gt_bot is not None or gt_top is not None:
                print(f"  Expected Time: bottom={gt_bot}  top={gt_top}")
            if comparison and comparison['valid']:
                acc = comparison['accuracy']
                correct = comparison['correct']
                status = "✅" if acc == 100 else "⚠️" if acc >= 80 else "❌"
                print(f"  Accuracy: {status} {correct}/64 squares ({acc:.1f}%)")
                if comparison['errors'] and acc < 100:
                    # Show first 10 errors
                    errs = [f"{e['name']}:{e['gt']}->{e['det']}" for e in comparison['errors'][:10]]
                    more = f" (+{len(comparison['errors'])-10} more)" if len(comparison['errors']) > 10 else ""
                    print(f"  Errors:   {', '.join(errs)}{more}")
        print(f"  Time: bottom={bot_time}  top={top_time}")
        if detected_result or gt_result:
            res_str = f"  Result: detected={detected_result} (score={result_score:.2f})"
            if gt_result:
                status = "✅" if detected_result == gt_result else "❌"
                res_str += f" expected={gt_result} {status}"
            print(res_str)
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_comparisons:
        total_correct = sum(c['comparison']['correct'] for c in all_comparisons if c['comparison']['valid'])
        total_squares = sum(c['comparison']['total'] for c in all_comparisons if c['comparison']['valid'])
        overall_acc = total_correct / total_squares * 100 if total_squares > 0 else 0
        
        perfect = sum(1 for c in all_comparisons if c['comparison']['valid'] and c['comparison']['accuracy'] == 100)
        good = sum(1 for c in all_comparisons if c['comparison']['valid'] and 80 <= c['comparison']['accuracy'] < 100)
        bad = sum(1 for c in all_comparisons if c['comparison']['valid'] and c['comparison']['accuracy'] < 80)
        
        print(f"Board detection:")
        print(f"  Overall accuracy: {total_correct}/{total_squares} squares ({overall_acc:.1f}%)")
        print(f"  Perfect (100%):   {perfect}/{len(all_comparisons)}")
        print(f"  Good (80-99%):    {good}/{len(all_comparisons)}")
        print(f"  Bad (<80%):       {bad}/{len(all_comparisons)}")
    
    if clock_results['total'] > 0:
        print(f"Clock detection:")
        print(f"  Bottom clocks:    {clock_results['bottom']}/{clock_results['total']} detected", end="")
        if 'bottom_with_gt' in clock_results:
            acc = clock_results['bottom_correct'] / clock_results['bottom_with_gt'] * 100
            print(f" ({clock_results['bottom_correct']}/{clock_results['bottom_with_gt']} correct, {acc:.1f}%)")
        else:
            print()
            
        print(f"  Top clocks:       {clock_results['top']}/{clock_results['total']} detected", end="")
        if 'top_with_gt' in clock_results:
            acc = clock_results['top_correct'] / clock_results['top_with_gt'] * 100
            print(f" ({clock_results['top_correct']}/{clock_results['top_with_gt']} correct, {acc:.1f}%)")
        else:
            print()

    if result_stats['detected_total'] > 0 or result_stats['total_with_gt'] > 0:
        print(f"Result detection:")
        print(f"  Total results:    {result_stats['detected_total']} detected")
        if result_stats['total_with_gt'] > 0:
            acc = result_stats['correct'] / result_stats['total_with_gt'] * 100
            print(f"  Accuracy:         {result_stats['correct']}/{result_stats['total_with_gt']} correct ({acc:.1f}%)")
        else:
            print("  Accuracy:         N/A (no ground truth)")

    if args.create_new and debug_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = debug_root / timestamp
        shutil.copytree(debug_dir, run_dir)
        print(f"\nResults also saved to: {run_dir}")


if __name__ == "__main__":
    main()

