#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for verifying click positions in new_game function
Tests the theoretical click positions without actually clicking

Created for testing new_game button positions and time control selections
"""

import sys
import os
import time
import numpy as np
import subprocess

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Save current working directory
original_cwd = os.getcwd()

try:
    import pyautogui
    
    # Extract just the constants we need to avoid image loading issues
    # These values are from chessimage/image_scrape_utils.py
    START_X = 543  # Chess board left edge
    START_Y = 179  # Chess board top edge  
    STEP = 106     # Size of each chess square
    
    print(f"✓ Loaded constants successfully")
    print(f"Project root: {project_root}")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have pyautogui and numpy installed")
    print(f"Project root: {project_root}")
    sys.exit(1)
except Exception as e:
    print(f"Error during setup: {e}")
    sys.exit(1)

# Restore original working directory
os.chdir(original_cwd)

class SimpleCursor:
    """Simplified cursor class that uses pyautogui directly"""
    
    @staticmethod
    def move_to(point, duration=0.5, steady=True):
        """Move cursor to point with given duration"""
        pyautogui.moveTo(point[0], point[1], duration=duration)

class NewGamePositionTester:
    """Test class for validating new_game click positions"""
    
    def __init__(self):
        self.cursor = SimpleCursor()
        self.start_x = START_X
        self.start_y = START_Y
        self.step = STEP
        
        print(f"Initialised with START_X={START_X}, START_Y={START_Y}, STEP={STEP}")
    
    def is_capslock_on(self):
        """Check if caps lock is on (copied from original function)"""
        try:
            output = subprocess.check_output('xset q | grep LED', shell=True)
            if output[65] == 48:
                return False
            elif output[65] == 49:
                return True
            return False
        except Exception as e:
            print(f"Warning: Could not check caps lock status: {e}")
            return False
    
    def hover_mouse(self, x, y, tolerance=0, duration=0.5, label=""):
        """Hover mouse at position with tolerance (without clicking)"""
        new_x = x + tolerance * (np.random.random() - 0.5)
        new_y = y + tolerance * (np.random.random() - 0.5)
        
        print(f"Hovering at {label}: ({new_x:.1f}, {new_y:.1f}) [original: ({x}, {y}), tolerance: ±{tolerance/2}]")
        
        # Move cursor to position
        self.cursor.move_to([new_x, new_y], duration=duration, steady=True)
        
        # Wait a moment to see the position
        time.sleep(1.5)
        
        return new_x, new_y
    
    def test_new_game_positions(self, time_control="1+0", test_both_controls=True):
        """Test the new_game function positions without actually clicking"""
        
        print("=" * 60)
        print("TESTING NEW GAME CLICK POSITIONS")
        print("=" * 60)
        
        # Check caps lock (as original function does)
        if self.is_capslock_on():
            print("WARNING: Caps lock is on - original function would abort here")
            return False
        else:
            print("✓ Caps lock is off - proceeding with test")
        
        print(f"\nTesting positions for time control: {time_control}")
        print(f"Reference coordinates: START_X={self.start_x}, START_Y={self.start_y}, STEP={self.step}")
        
        # Test play button position
        print("\n1. Testing PLAY BUTTON position:")
        play_button_x = self.start_x - 1.9 * self.step
        play_button_y = self.start_y - 0.4 * self.step
        
        actual_x, actual_y = self.hover_mouse(
            play_button_x, play_button_y, 
            tolerance=10, 
            duration=np.random.uniform(0.3, 0.7),
            label="Play Button"
        )
        
        print(f"   Expected: ({play_button_x:.1f}, {play_button_y:.1f})")
        print(f"   With tolerance: ({actual_x:.1f}, {actual_y:.1f})")
        
        # Simulate the wait that would happen after clicking play button
        print("   Simulating 1.5s wait after play button click...")
        time.sleep(1.5)
        
        # Test time control positions
        if test_both_controls:
            time_controls = ["1+0", "3+0"]
        else:
            time_controls = [time_control]
            
        for tc in time_controls:
            print(f"\n2. Testing TIME CONTROL position for {tc}:")
            
            if tc == "1+0":
                to_x = self.start_x + 1.7 * self.step
                to_y = self.start_y + 0.7 * self.step
            elif tc == "3+0":
                to_x = self.start_x + 5.7 * self.step
                to_y = self.start_y + 0.7 * self.step
            else:
                print(f"   Unknown time control: {tc}")
                continue
            
            actual_x, actual_y = self.hover_mouse(
                to_x, to_y, 
                tolerance=20, 
                duration=np.random.uniform(0.3, 0.7),
                label=f"Time Control {tc}"
            )
            
            print(f"   Expected: ({to_x:.1f}, {to_y:.1f})")
            print(f"   With tolerance: ({actual_x:.1f}, {actual_y:.1f})")
        
        return True
    
    def test_coordinate_calculations(self):
        """Test and display coordinate calculations"""
        print("\n" + "=" * 60)
        print("COORDINATE CALCULATIONS")
        print("=" * 60)
        
        print(f"Base coordinates:")
        print(f"  START_X = {self.start_x}")
        print(f"  START_Y = {self.start_y}")
        print(f"  STEP = {self.step}")
        
        print(f"\nPlay button calculation:")
        play_x = self.start_x - 1.9 * self.step
        play_y = self.start_y - 0.4 * self.step
        print(f"  X = START_X - 1.9*STEP = {self.start_x} - 1.9*{self.step} = {play_x}")
        print(f"  Y = START_Y - 0.4*STEP = {self.start_y} - 0.4*{self.step} = {play_y}")
        
        print(f"\n1+0 time control calculation:")
        tc1_x = self.start_x + 1.7 * self.step
        tc1_y = self.start_y + 0.7 * self.step
        print(f"  X = START_X + 1.7*STEP = {self.start_x} + 1.7*{self.step} = {tc1_x}")
        print(f"  Y = START_Y + 0.7*STEP = {self.start_y} + 0.7*{self.step} = {tc1_y}")
        
        print(f"\n3+0 time control calculation:")
        tc3_x = self.start_x + 5.7 * self.step
        tc3_y = self.start_y + 0.7 * self.step
        print(f"  X = START_X + 5.7*STEP = {self.start_x} + 5.7*{self.step} = {tc3_x}")
        print(f"  Y = START_Y + 0.7*STEP = {self.start_y} + 0.7*{self.step} = {tc3_y}")
    
    def interactive_test(self):
        """Interactive test mode - allows manual verification"""
        print("\n" + "=" * 60)
        print("INTERACTIVE TEST MODE")
        print("=" * 60)
        print("Watch the cursor movement and verify positions are correct")
        print("Press Enter after each position to continue...")
        
        # Test play button
        input("\nPress Enter to test PLAY BUTTON position...")
        play_x = self.start_x - 1.9 * self.step
        play_y = self.start_y - 0.4 * self.step
        self.hover_mouse(play_x, play_y, tolerance=10, duration=1.0, label="Play Button")
        
        # Test 1+0 control
        input("\nPress Enter to test 1+0 TIME CONTROL position...")
        tc1_x = self.start_x + 1.7 * self.step
        tc1_y = self.start_y + 0.7 * self.step
        self.hover_mouse(tc1_x, tc1_y, tolerance=20, duration=1.0, label="1+0 Time Control")
        
        # Test 3+0 control
        input("\nPress Enter to test 3+0 TIME CONTROL position...")
        tc3_x = self.start_x + 5.7 * self.step
        tc3_y = self.start_y + 0.7 * self.step
        self.hover_mouse(tc3_x, tc3_y, tolerance=20, duration=1.0, label="3+0 Time Control")
        
        print("\nInteractive test completed!")

def main():
    """Main test function"""
    print("New Game Position Tester")
    print("This script tests the click positions used in new_game() without actually clicking")
    
    # Disable pyautogui failsafe for testing
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    
    # Get current mouse position
    current_x, current_y = pyautogui.position()
    print(f"Current mouse position: ({current_x}, {current_y})")
    
    # Create tester
    tester = NewGamePositionTester()
    
    # Show coordinate calculations
    tester.test_coordinate_calculations()
    
    # Test mode selection
    print("\n" + "=" * 60)
    print("SELECT TEST MODE")
    print("=" * 60)
    print("1. Automatic test (tests all positions)")
    print("2. Interactive test (manual verification)")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip()
        if not choice:
            choice = "1"
            
        if choice in ["1", "3"]:
            print("\nRunning automatic test...")
            success = tester.test_new_game_positions(test_both_controls=True)
            if success:
                print("✓ Automatic test completed")
            else:
                print("✗ Automatic test failed")
        
        if choice in ["2", "3"]:
            print("\nRunning interactive test...")
            tester.interactive_test()
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    
    # Return mouse to original position
    print(f"\nReturning mouse to original position: ({current_x}, {current_y})")
    pyautogui.moveTo(current_x, current_y)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
