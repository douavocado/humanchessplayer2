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
            
            # Extract and save board
            try:
                board_img = extract_board_from_screenshot(screenshot)
                if board_img is not None:
                    save_debug_image(board_img, "board", attempt_dir)
                    log(f"Board extracted: shape={board_img.shape}")
                else:
                    log(f"ERROR: Failed to extract board")
            except Exception as e:
                log(f"ERROR extracting board: {e}")
            
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
                try:
                    board_img = capture_board()
                    if board_img is not None:
                        save_debug_image(board_img, "board", attempt_dir)
                        log(f"Board captured: shape={board_img.shape}")
                except Exception as e:
                    log(f"ERROR capturing board: {e}")
                
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