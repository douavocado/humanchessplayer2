#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:22:09 2025

@author: james
"""
import time
import sys
import random

import argparse

# Import only the constants module first, delay other imports
import common.constants as constants

parser = argparse.ArgumentParser(description="Setting game mode")
parser.add_argument("-t", "--time", type=int, default=60,
                    help="The time control in seconds")
parser.add_argument("-i", "--increment", type=int, default=0,
                    help="The time control increcrement in seconds")
parser.add_argument("-g", "--games", type=int, default=5,
                    help="The number of games to play of the specified time control. Only used when not in tournament mode.")
parser.add_argument("-a", "--arena", help="Tournament arena mode",
                        action="store_true")
parser.add_argument("-b", "--berserk", help="Always beserk in tournament arena mode",
                        action="store_true")
parser.add_argument("-d", "--difficulty", type=int, default=None,
                    help="Engine difficulty level (overrides default from constants)")
parser.add_argument("-q", "--quickness", type=float, default=None,
                    help="Engine quickness (overrides default from constants)")
parser.add_argument("-m", "--mouse-quickness", type=float, default=None,
                    help="Mouse quickness (overrides default from constants)")
parser.add_argument("--debug", help="Debug/dry-run mode: only test new game detection with visualisations",
                        action="store_true")
parser.add_argument("--debug-interval", type=float, default=2.0,
                    help="Interval between detection attempts in debug mode (seconds)")
parser.add_argument("--offline", help="Use offline screenshots instead of live capture (for testing without a game)",
                        action="store_true")
parser.add_argument("--offline-dir", type=str, default="auto_calibration/offline_screenshots",
                    help="Directory containing offline screenshots (default: auto_calibration/offline_screenshots)")
args = parser.parse_args()

# Set the values, using command line arguments if provided, otherwise use defaults from constants
DIFFICULTY = args.difficulty if args.difficulty is not None else constants.DIFFICULTY
QUICKNESS = args.quickness if args.quickness is not None else constants.QUICKNESS
MOUSE_QUICKNESS = args.mouse_quickness if args.mouse_quickness is not None else constants.MOUSE_QUICKNESS

# Update the constants module with the final values so other modules can access them
constants.DIFFICULTY = DIFFICULTY
constants.QUICKNESS = QUICKNESS
constants.MOUSE_QUICKNESS = MOUSE_QUICKNESS

# Now import mp_original after constants are set
from clients.mp_original import run_game, await_new_game, set_game, new_game, back_to_lobby, berserk

# Debug mode imports (only loaded when needed)
def run_debug_mode(interval=2.0, offline=False, offline_dir="auto_calibration/offline_screenshots"):
    """
    Debug/dry-run mode for testing new game detection.
    
    Continuously attempts to detect a new game, saving debug visualisations
    and logging detailed information about the detection process.
    
    Args:
        interval: Time between detection attempts (seconds)
        offline: If True, use offline screenshots instead of live capture
        offline_dir: Directory containing offline screenshots
    """
    import os
    import cv2
    import numpy as np
    from datetime import datetime
    from pathlib import Path
    
    from chessimage.image_scrape_utils import (
        capture_bottom_clock, capture_top_clock, capture_board,
        read_clock, get_clock_info, get_coordinates, get_board_info,
        remove_background_colours, TEMPLATES, multitemplate_match_f
    )
    
    # Create debug output directory
    debug_dir = Path("debug_detection_output")
    debug_dir.mkdir(exist_ok=True)
    
    # Create log file
    log_file = debug_dir / f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log(message, also_print=True):
        """Log message to file and optionally print."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        formatted = f"[{timestamp}] {message}"
        if also_print:
            print(formatted)
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')
    
    def save_debug_image(img, name, subdir=None):
        """Save debug image with timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        if subdir:
            save_dir = debug_dir / subdir
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = debug_dir
        filepath = save_dir / f"{name}_{timestamp}.png"
        cv2.imwrite(str(filepath), img)
        return filepath
    
    def analyse_clock_image(clock_img, state_name):
        """Analyse a clock image and return detailed info."""
        result = {
            'state': state_name,
            'shape': clock_img.shape if clock_img is not None else None,
            'clock_value': None,
            'digit_scores': [],
            'success': False
        }
        
        if clock_img is None:
            result['error'] = "Image is None"
            return result
        
        # Process image for OCR
        if clock_img.ndim == 3:
            processed = remove_background_colours(clock_img, thresh=1.6).astype(np.uint8)
        else:
            processed = clock_img.copy()
        
        result['processed_shape'] = processed.shape
        
        if processed.size == 0 or processed.ndim != 2:
            result['error'] = "Processed image invalid"
            return result
        
        img_height, img_width = processed.shape
        result['width'] = img_width
        result['height'] = img_height
        
        # Get template dimensions
        template_h, template_w = TEMPLATES.shape[1:3]
        
        # Try to get digit positions from config
        try:
            from auto_calibration.config import get_config
            config = get_config()
            digit_pos = config.get_digit_positions()
        except:
            digit_pos = None
        
        if digit_pos:
            # Use calibrated positions (as fractions of width)
            d1_start = int(digit_pos['d1_start'] * img_width)
            d1_end = int(digit_pos['d1_end'] * img_width)
            d2_start = int(digit_pos['d2_start'] * img_width)
            d2_end = int(digit_pos['d2_end'] * img_width)
            d3_start = int(digit_pos['d3_start'] * img_width)
            d3_end = int(digit_pos['d3_end'] * img_width)
            d4_start = int(digit_pos['d4_start'] * img_width)
            d4_end = int(digit_pos['d4_end'] * img_width)
        else:
            # Fallback to scaled positions
            ORIGINAL_WIDTH = 147.0
            scale = img_width / ORIGINAL_WIDTH
            d1_start, d1_end = 0, int(30 * scale)
            d2_start, d2_end = int(34 * scale), int(64 * scale)
            d3_start, d3_end = int(83 * scale), int(113 * scale)
            d4_start, d4_end = int(117 * scale), min(int(147 * scale), img_width)
        
        # Extract digit regions
        d1 = processed[:, d1_start:d1_end]
        d2 = processed[:, d2_start:d2_end]
        d3 = processed[:, d3_start:d3_end]
        d4 = processed[:, d4_start:d4_end]
        
        digits = [d1, d2, d3, d4]
        digit_positions = [(d1_start, d1_end), (d2_start, d2_end), (d3_start, d3_end), (d4_start, d4_end)]
        
        result['digit_positions'] = digit_positions
        
        # Resize and match each digit
        digit_values = []
        for i, (d, pos) in enumerate(zip(digits, digit_positions)):
            if d.size == 0:
                result['digit_scores'].append({'digit': i+1, 'error': 'Empty region'})
                digit_values.append(None)
                continue
            
            try:
                d_resized = cv2.resize(d, (template_w, template_h), interpolation=cv2.INTER_AREA)
                
                # Get matching scores for all templates
                T = TEMPLATES.astype(float)
                I = d_resized.astype(float)
                w, h = d_resized.shape
                T_primes = T - np.expand_dims(1/(w*h)*T.sum(axis=(1,2)), (-1,-2))
                I_prime = I - np.expand_dims(1/(w*h)*I.sum(), (-1))
                T_denom = (T_primes**2).sum(axis=(1,2))
                I_denom = (I_prime**2).sum()
                denoms = np.sqrt(T_denom*I_denom) + 10**(-10)
                nums = (T_primes*np.expand_dims(I_prime,0)).sum(axis=(1,2))
                scores = nums/denoms
                
                best_match = int(scores.argmax())
                best_score = float(scores.max())
                
                result['digit_scores'].append({
                    'digit': i+1,
                    'position': pos,
                    'best_match': best_match,
                    'best_score': round(best_score, 3),
                    'all_scores': [round(s, 3) for s in scores.tolist()]
                })
                
                if best_score >= 0.5:
                    digit_values.append(best_match)
                else:
                    digit_values.append(None)
                    
            except Exception as e:
                result['digit_scores'].append({'digit': i+1, 'error': str(e)})
                digit_values.append(None)
        
        # Calculate clock value if all digits matched
        if all(d is not None for d in digit_values):
            result['clock_value'] = digit_values[0] * 600 + digit_values[1] * 60 + digit_values[2] * 10 + digit_values[3]
            result['success'] = True
            result['digits'] = digit_values
        else:
            result['digits'] = digit_values
        
        return result
    
    def extract_clock_from_screenshot(screenshot, clock_type, state):
        """Extract clock region from a full screenshot using calibrated coordinates."""
        try:
            x, y, w, h = get_clock_info(clock_type, state)
            # Extract region from screenshot
            clock_img = screenshot[y:y+h, x:x+w].copy()
            if clock_img.ndim == 3 and clock_img.shape[2] == 4:
                clock_img = clock_img[:, :, :3]  # Remove alpha channel if present
            return clock_img
        except Exception as e:
            return None
    
    def extract_board_from_screenshot(screenshot):
        """Extract board region from a full screenshot using calibrated coordinates."""
        try:
            board_x, board_y, step = get_board_info()
            board_size = step * 8
            board_img = screenshot[board_y:board_y+board_size, board_x:board_x+board_size].copy()
            if board_img.ndim == 3 and board_img.shape[2] == 4:
                board_img = board_img[:, :, :3]
            return board_img
        except Exception as e:
            return None
    
    def analyse_board_image(board_img, bottom='w'):
        """
        Analyse a board image and return detailed FEN detection info.
        
        Args:
            board_img: BGR board image
            bottom: 'w' if white is at bottom, 'b' if black is at bottom
            
        Returns:
            Dictionary with detection results
        """
        from chessimage.image_scrape_utils import (
            get_fen_from_image, PIECE_TEMPLATES, STEP, PIECE_STEP,
            remove_background_colours, multitemplate_multimatch
        )
        import chess
        
        result = {
            'success': False,
            'fen': None,
            'piece_count': 0,
            'shape': board_img.shape if board_img is not None else None,
            'step': STEP,
            'piece_step': PIECE_STEP,
            'square_details': [],
            'error': None
        }
        
        if board_img is None:
            result['error'] = "Board image is None"
            return result
        
        try:
            # Get FEN
            fen = get_fen_from_image(board_img, bottom=bottom)
            result['fen'] = fen
            
            # Parse FEN to count pieces
            board = chess.Board(fen)
            result['piece_count'] = len(board.piece_map())
            result['board_valid'] = board.is_valid()
            
            # Get detailed square-by-square analysis
            if board_img.ndim == 3:
                processed = remove_background_colours(board_img).astype(np.uint8)
            else:
                processed = board_img.copy()
            
            # Extract all 64 squares
            images = [processed[x*STEP:x*STEP+PIECE_STEP, y*STEP:y*STEP+PIECE_STEP] 
                     for x in range(8) for y in range(8)]
            images = np.stack(images, axis=0)
            
            # Run template matching
            valid_squares, argmaxes = multitemplate_multimatch(images, PIECE_TEMPLATES)
            
            piece_chars = 'RNBKQPrnbkqp'
            
            # Analyse each square
            for sq in range(64):
                row, col = sq // 8, sq % 8
                file_letter = chr(ord('a') + col)
                rank_number = 8 - row
                square_name = f"{file_letter}{rank_number}"
                
                # Calculate match scores for this square
                sq_img = images[sq]
                T = PIECE_TEMPLATES.astype(float)
                I = sq_img.astype(float)
                w, h = sq_img.shape
                T_primes = T - np.expand_dims(1/(w*h)*T.sum(axis=(1,2)), (-1,-2))
                I_prime = I - 1/(w*h)*I.sum()
                T_denom = (T_primes**2).sum(axis=(1,2))
                I_denom = (I_prime**2).sum()
                denoms = np.sqrt(T_denom * I_denom) + 1e-10
                nums = (T_primes * np.expand_dims(I_prime, 0)).sum(axis=(1,2))
                scores = nums / denoms
                
                best_idx = int(scores.argmax())
                best_score = float(scores.max())
                
                is_valid = sq in valid_squares
                detected_piece = piece_chars[argmaxes[sq]] if is_valid else None
                
                result['square_details'].append({
                    'square': square_name,
                    'row': row,
                    'col': col,
                    'detected_piece': detected_piece,
                    'best_match': piece_chars[best_idx],
                    'best_score': round(best_score, 3),
                    'is_valid': is_valid
                })
            
            result['success'] = True
            result['detected_pieces'] = sum(1 for s in result['square_details'] if s['is_valid'])
            
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def create_board_debug_image(board_img, analysis):
        """
        Create a debug image showing detected pieces on the board.
        
        Overlays detection results on the board image.
        """
        if board_img is None:
            return None
        
        debug_img = board_img.copy()
        step = analysis.get('step', 207)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = step / 100  # Scale font based on square size
        thickness = max(1, int(step / 50))
        
        for sq_info in analysis.get('square_details', []):
            row, col = sq_info['row'], sq_info['col']
            x = col * step
            y = row * step
            
            piece = sq_info.get('detected_piece')
            score = sq_info.get('best_score', 0)
            
            if piece:
                # Draw piece letter
                color = (0, 255, 0) if score > 0.7 else (0, 255, 255)  # Green if high confidence, yellow otherwise
                text = piece
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x + (step - text_size[0]) // 2
                text_y = y + (step + text_size[1]) // 2
                cv2.putText(debug_img, text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Draw score in corner
            score_text = f"{score:.2f}"
            cv2.putText(debug_img, score_text, (x + 2, y + int(step * 0.15)), 
                       font, font_scale * 0.4, (255, 255, 255), 1)
        
        return debug_img
    
    def save_problematic_squares(board_img, analysis, output_dir):
        """
        Save individual square images for squares with low confidence or misdetections.
        
        Helps diagnose template matching issues.
        """
        from chessimage.image_scrape_utils import STEP, PIECE_STEP, remove_background_colours, PIECE_TEMPLATES
        
        if board_img is None or not analysis.get('success'):
            return
        
        # Process board to get grayscale squares
        if board_img.ndim == 3:
            processed = remove_background_colours(board_img).astype(np.uint8)
        else:
            processed = board_img.copy()
        
        squares_dir = Path(output_dir) / "squares"
        squares_dir.mkdir(parents=True, exist_ok=True)
        
        problem_count = 0
        for sq_info in analysis.get('square_details', []):
            row, col = sq_info['row'], sq_info['col']
            score = sq_info.get('best_score', 0)
            is_valid = sq_info.get('is_valid', False)
            detected = sq_info.get('detected_piece')
            square_name = sq_info.get('square', f'{row}_{col}')
            
            # Save if: low confidence, unexpected detection, or potential misdetection
            should_save = (is_valid and score < 0.6) or (not is_valid and score > 0.4)
            
            if should_save and problem_count < 20:  # Limit to avoid too many files
                # Extract square
                sq_img = processed[row*STEP:row*STEP+PIECE_STEP, col*STEP:col*STEP+PIECE_STEP]
                sq_rgb = board_img[row*STEP:row*STEP+PIECE_STEP, col*STEP:col*STEP+PIECE_STEP]
                
                # Save both processed and RGB versions
                status = 'detected' if is_valid else 'empty'
                piece_str = detected if detected else 'none'
                filename = f"{square_name}_{status}_{piece_str}_{score:.2f}"
                
                cv2.imwrite(str(squares_dir / f"{filename}_processed.png"), sq_img)
                cv2.imwrite(str(squares_dir / f"{filename}_rgb.png"), sq_rgb)
                problem_count += 1
        
        if problem_count > 0:
            log(f"    Saved {problem_count} problematic square images to {squares_dir}")
        
        # Also save the piece templates for comparison
        templates_dir = squares_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        piece_chars = 'RNBKQPrnbkqp'
        for i, char in enumerate(piece_chars):
            cv2.imwrite(str(templates_dir / f"template_{char}.png"), PIECE_TEMPLATES[i])
    
    # Load offline screenshots if in offline mode
    offline_screenshots = {}
    if offline:
        offline_path = Path(offline_dir)
        if not offline_path.exists():
            print(f"ERROR: Offline directory not found: {offline_path.absolute()}")
            return
        
        # Map screenshot filenames to states
        state_mapping = {
            'start1.png': 'start1',
            'start2.png': 'start2',
            'play.png': 'play',
            'end1.png': 'end1',
            'end2.png': 'end2',
            'draw_stalemate.png': 'draw'
        }
        
        for filename, state in state_mapping.items():
            filepath = offline_path / filename
            if filepath.exists():
                img = cv2.imread(str(filepath))
                if img is not None:
                    offline_screenshots[state] = img
                    print(f"✓ Loaded offline screenshot: {filename} ({img.shape})")
                else:
                    print(f"✗ Failed to load: {filename}")
            else:
                print(f"- Not found: {filename}")
        
        if not offline_screenshots:
            print("ERROR: No offline screenshots loaded!")
            return
    
    mode_str = "OFFLINE" if offline else "LIVE"
    print("\n" + "=" * 70)
    print(f"DEBUG MODE ({mode_str}) - New Game Detection Testing")
    print("=" * 70)
    print(f"Debug output directory: {debug_dir.absolute()}")
    print(f"Log file: {log_file}")
    if offline:
        print(f"Offline screenshots: {len(offline_screenshots)} loaded")
    else:
        print(f"Detection interval: {interval} seconds")
        print("Press Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    # Log calibration info
    log("=== CALIBRATION INFO ===")
    try:
        coords = get_coordinates()
        log(f"Coordinates loaded successfully")
        
        # Board info
        board_x, board_y, step = get_board_info()
        log(f"Board: ({board_x}, {board_y}) step={step}")
        
        # Clock info for each state
        for clock_type in ['bottom_clock', 'top_clock']:
            log(f"\n{clock_type.upper()}:")
            for state in ['play', 'start1', 'start2', 'end1', 'end2']:
                try:
                    x, y, w, h = get_clock_info(clock_type, state)
                    log(f"  {state:8s}: ({x:4d}, {y:4d}) size={w:3d}x{h:2d}")
                except Exception as e:
                    log(f"  {state:8s}: ERROR - {e}")
    except Exception as e:
        log(f"ERROR loading coordinates: {e}")
    
    if offline:
        log("\n=== OFFLINE DETECTION TEST ===\n")
        
        # Test each loaded screenshot
        for screenshot_state, screenshot in offline_screenshots.items():
            log(f"\n{'='*50}")
            log(f"Testing screenshot: {screenshot_state}.png")
            log(f"Screenshot shape: {screenshot.shape}")
            log(f"{'='*50}")
            
            # Create subdirectory for this screenshot
            attempt_dir = f"offline_{screenshot_state}"
            
            # Extract and analyse board
            board_img = None
            try:
                board_img = extract_board_from_screenshot(screenshot)
                if board_img is not None:
                    save_debug_image(board_img, "board", attempt_dir)
                    log(f"Board extracted: shape={board_img.shape}")
                    
                    # Analyse board for both orientations
                    for bottom in ['w', 'b']:
                        analysis = analyse_board_image(board_img, bottom=bottom)
                        
                        if analysis['success']:
                            log(f"  Board (bottom={bottom}): FEN={analysis['fen']}")
                            log(f"    Pieces detected: {analysis['detected_pieces']}/32, Valid board: {analysis.get('board_valid', False)}")
                            
                            # Create and save debug overlay image
                            debug_overlay = create_board_debug_image(board_img, analysis)
                            if debug_overlay is not None:
                                save_debug_image(debug_overlay, f"board_debug_{bottom}", attempt_dir)
                            
                            # Save problematic squares for detailed analysis
                            save_problematic_squares(board_img, analysis, debug_dir / attempt_dir)

                            # Log any squares with low confidence
                            low_conf_squares = [s for s in analysis['square_details'] 
                                              if s['is_valid'] and s['best_score'] < 0.6]
                            if low_conf_squares:
                                log(f"    Low confidence pieces:")
                                for sq in low_conf_squares[:5]:  # Show first 5
                                    log(f"      {sq['square']}: {sq['detected_piece']} (score={sq['best_score']})")
                            
                            # Check for common issues
                            empty_board = analysis['fen'].split()[0] == '8/8/8/8/8/8/8/8'
                            if empty_board:
                                log(f"    ⚠️  WARNING: Empty board detected! Check piece templates.")
                        else:
                            log(f"  Board (bottom={bottom}): FAILED - {analysis.get('error', 'Unknown error')}")
                else:
                    log(f"ERROR: Failed to extract board")
            except Exception as e:
                log(f"ERROR extracting/analysing board: {e}")
                import traceback
                log(f"  {traceback.format_exc()}")
            
            # Test clock extraction for different states
            # For start1.png, we test 'start1' state; for start2.png, test 'start2' state, etc.
            clock_states_to_test = ['start1', 'start2', 'play', 'end1', 'end2']
            detection_results = {}
            
            for clock_state in clock_states_to_test:
                try:
                    # Extract clock region from screenshot
                    clock_img = extract_clock_from_screenshot(screenshot, 'bottom_clock', clock_state)
                    
                    if clock_img is None or clock_img.size == 0:
                        log(f"  ✗ {clock_state:8s}: Failed to extract region")
                        detection_results[clock_state] = {'error': 'Failed to extract'}
                        continue
                    
                    # Save original extracted image
                    save_debug_image(clock_img, f"clock_bottom_{clock_state}", attempt_dir)
                    
                    # Analyse the clock image
                    analysis = analyse_clock_image(clock_img, clock_state)
                    detection_results[clock_state] = analysis
                    
                    # Save processed image
                    if clock_img is not None and clock_img.ndim == 3:
                        processed = remove_background_colours(clock_img, thresh=1.6).astype(np.uint8)
                        save_debug_image(processed, f"clock_bottom_{clock_state}_processed", attempt_dir)
                    
                    # Log results
                    if analysis['success']:
                        log(f"  ✓ {clock_state:8s}: READ SUCCESS - {analysis['clock_value']}s (digits: {analysis.get('digits', [])})")
                    else:
                        log(f"  ✗ {clock_state:8s}: FAILED - shape={analysis.get('shape')}, width={analysis.get('width', 'N/A')}")
                        # Log digit details for failed attempts
                        for d_info in analysis.get('digit_scores', []):
                            if 'error' in d_info:
                                log(f"      Digit {d_info['digit']}: {d_info['error']}")
                            else:
                                log(f"      Digit {d_info['digit']}: best={d_info['best_match']} score={d_info['best_score']:.3f}")
                    
                except Exception as e:
                    log(f"  ✗ {clock_state:8s}: EXCEPTION - {e}")
                    import traceback
                    log(f"      {traceback.format_exc()}")
                    detection_results[clock_state] = {'error': str(e)}
            
            # Summary for this screenshot
            log(f"\n--- Summary for {screenshot_state}.png ---")
            successful_states = [s for s, r in detection_results.items() if r.get('success')]
            if successful_states:
                log(f"✓ Successfully read states: {successful_states}")
                # Check if this is a "start" screenshot and if start detection worked
                if screenshot_state in ['start1', 'start2']:
                    if screenshot_state in successful_states:
                        log(f"🎮 WOULD DETECT NEW GAME from {screenshot_state}.png!")
                        # Play sound
                        try:
                            sound_file = "assets/audio/new_game_found.mp3"
                            os.system("mpg123 -q " + sound_file)
                        except:
                            pass
                    else:
                        log(f"⚠️  This is a {screenshot_state} screenshot but {screenshot_state} state was NOT successfully read!")
            else:
                log(f"✗ No states successfully read - detection would FAIL")
        
        log(f"\n\n{'='*70}")
        log(f"=== OFFLINE TEST COMPLETE ===")
        log(f"{'='*70}")
        log(f"Debug images saved to: {debug_dir.absolute()}")
        log(f"Check the extracted clock images to diagnose issues.")
        
    else:
        # Live mode - continuous detection
        log("\n=== STARTING DETECTION LOOP ===\n")
        
        detection_count = 0
        success_count = 0
        
        try:
            while True:
                detection_count += 1
                log(f"\n--- Detection Attempt #{detection_count} ---")
                
                # Create subdirectory for this attempt
                attempt_dir = f"attempt_{detection_count:04d}"
                
                # Capture and analyse board
                board_img = None
                try:
                    board_img = capture_board()
                    if board_img is not None:
                        save_debug_image(board_img, "board", attempt_dir)
                        log(f"Board captured: shape={board_img.shape}")
                        
                        # Analyse board (assume white at bottom by default)
                        analysis = analyse_board_image(board_img, bottom='w')
                        
                        if analysis['success']:
                            log(f"  FEN: {analysis['fen']}")
                            log(f"  Pieces: {analysis['detected_pieces']}/32, Valid: {analysis.get('board_valid', False)}")
                            
                            # Create debug overlay
                            debug_overlay = create_board_debug_image(board_img, analysis)
                            if debug_overlay is not None:
                                save_debug_image(debug_overlay, "board_debug", attempt_dir)
                            
                            # Save problematic squares for detailed analysis
                            save_problematic_squares(board_img, analysis, debug_dir / attempt_dir)

                            # Check for empty board
                            if analysis['fen'].split()[0] == '8/8/8/8/8/8/8/8':
                                log(f"  ⚠️  WARNING: Empty board detected!")
                        else:
                            log(f"  Board analysis FAILED: {analysis.get('error', 'Unknown')}")
                except Exception as e:
                    log(f"ERROR capturing/analysing board: {e}")
                
                # Test each clock state
                clock_states = ['start1', 'start2', 'play', 'end1', 'end2']
                detection_results = {}
                
                for state in clock_states:
                    try:
                        # Capture clock image
                        clock_img = capture_bottom_clock(state=state)
                        
                        # Save original image
                        save_debug_image(clock_img, f"clock_bottom_{state}", attempt_dir)
                        
                        # Analyse the clock image
                        analysis = analyse_clock_image(clock_img, state)
                        detection_results[state] = analysis
                        
                        # Save processed image
                        if clock_img is not None and clock_img.ndim == 3:
                            processed = remove_background_colours(clock_img, thresh=1.6).astype(np.uint8)
                            save_debug_image(processed, f"clock_bottom_{state}_processed", attempt_dir)
                        
                        # Log results
                        if analysis['success']:
                            log(f"  ✓ {state:8s}: READ SUCCESS - {analysis['clock_value']}s (digits: {analysis.get('digits', [])})")
                        else:
                            log(f"  ✗ {state:8s}: FAILED - shape={analysis.get('shape')}, width={analysis.get('width', 'N/A')}")
                            # Log digit details for failed attempts
                            for d_info in analysis.get('digit_scores', []):
                                if 'error' in d_info:
                                    log(f"      Digit {d_info['digit']}: {d_info['error']}")
                                else:
                                    log(f"      Digit {d_info['digit']}: best={d_info['best_match']} score={d_info['best_score']:.3f}")
                        
                    except Exception as e:
                        log(f"  ✗ {state:8s}: EXCEPTION - {e}")
                        detection_results[state] = {'error': str(e)}
                
                # Check if new game was detected (start1 or start2 success)
                game_detected = False
                detected_time = None
                
                for state in ['start1', 'start2']:
                    if detection_results.get(state, {}).get('success'):
                        game_detected = True
                        detected_time = detection_results[state]['clock_value']
                        break
                
                if game_detected:
                    success_count += 1
                    log(f"\n🎮 NEW GAME DETECTED! Time: {detected_time} seconds")
                    log(f"   Detection rate: {success_count}/{detection_count} ({100*success_count/detection_count:.1f}%)")
                    
                    # Play sound
                    try:
                        sound_file = "assets/audio/new_game_found.mp3"
                        os.system("mpg123 -q " + sound_file)
                    except:
                        pass
                else:
                    log(f"\n⏳ No new game detected. ({success_count}/{detection_count} successful)")
                
                # Wait before next attempt
                time.sleep(interval)
                
        except KeyboardInterrupt:
            log(f"\n\n=== DEBUG MODE STOPPED ===")
            log(f"Total attempts: {detection_count}")
            log(f"Successful detections: {success_count}")
            log(f"Success rate: {100*success_count/detection_count:.1f}%" if detection_count > 0 else "N/A")
            log(f"Debug images saved to: {debug_dir.absolute()}")
            print("\nDebug mode stopped.")

def verify_and_patch_constants():
    """Verify that other modules can see the overridden constants and patch them if needed"""
    import clients.mp_original as mp_original
    
    # Patch the constants in mp_original module
    mp_original.DIFFICULTY = DIFFICULTY
    mp_original.QUICKNESS = QUICKNESS  
    mp_original.MOUSE_QUICKNESS = MOUSE_QUICKNESS
    
    # Re-initialize the engine with the correct difficulty if it was overridden
    if args.difficulty is not None:
        from engine import Engine
        mp_original.ENGINE = Engine(playing_level=DIFFICULTY)
        print(f"Re-initialized engine with difficulty: {DIFFICULTY}")
    
    print("\n--- Constants Verification ---")
    print(f"main.py - DIFFICULTY: {DIFFICULTY}, QUICKNESS: {QUICKNESS}, MOUSE_QUICKNESS: {MOUSE_QUICKNESS}")
    print(f"constants module - DIFFICULTY: {constants.DIFFICULTY}, QUICKNESS: {constants.QUICKNESS}, MOUSE_QUICKNESS: {constants.MOUSE_QUICKNESS}")
    print(f"mp_original module - DIFFICULTY: {mp_original.DIFFICULTY}, QUICKNESS: {mp_original.QUICKNESS}, MOUSE_QUICKNESS: {mp_original.MOUSE_QUICKNESS}")
    print(f"mp_original.ENGINE.playing_level: {mp_original.ENGINE.playing_level}")
    print("--- End Verification ---\n")

# Run verification and patching if any constants were overridden  
if args.difficulty is not None or args.quickness is not None or args.mouse_quickness is not None:
    verify_and_patch_constants()

print("Engine Difficulty: {}, Quickness: {}, Mouse Quickness: {}".format(DIFFICULTY, QUICKNESS, MOUSE_QUICKNESS))

# Debug mode - only test detection, don't play
if args.debug or args.offline:
    run_debug_mode(
        interval=args.debug_interval,
        offline=args.offline,
        offline_dir=args.offline_dir
    )
    sys.exit()

if args.arena == True:
    # in tournament mode
    while True:
        time.sleep(0.5)
        res = await_new_game(timeout=300)
        if res is not None:
            set_game(res)
            if args.berserk:
                # beserk mode
                berserk()
                time.sleep(0.5)
            run_game(arena=True)
            print("Finished tournament game.")
            # go back to lobby. This can be done by clicking where the resign button is once
            time.sleep(random.randint(1,3))
            back_to_lobby()
else:
    if args.time == 60:
        tc_str = "1+0"
    elif args.time == 180:
        tc_str = "3+0"
    else:
        raise Exception("Time control not recognised: {}".format(args.time))
    games = args.games
    for i in range(games):
        time.sleep(0.5)
        res = await_new_game(timeout=5)
        if res is not None:
            set_game(res)
            run_game(arena=False)
            print("Finished game {}".format(i+1))
            if i < games-1:
                new_game(tc_str)
        elif i < games-1:
            print("Skipped game, trying to seek again.")
            new_game(tc_str)
sys.exit()