#!/usr/bin/env python3
"""
Test script for the auto-calibration system.

This script provides simple tests to verify the calibration system works correctly.
"""

import sys
import time
import json
from pathlib import Path

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "chessimage"))

from board_detector import BoardDetector
from coordinate_mapper import CoordinateMapper
from auto_calibrator import AutoCalibrator
from visualizer import CalibrationVisualizer

def test_board_detection():
    """Test basic board detection functionality."""
    print("Testing board detection...")
    
    detector = BoardDetector()
    
    print("Please ensure a chess game is visible on screen.")
    print("Starting detection in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    result = detector.find_chess_board(max_attempts=1)
    
    if result:
        print(f"✅ Board detection successful!")
        print(f"   Method: {result['method']}")
        print(f"   Position: {result['position']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        return True
    else:
        print("❌ Board detection failed")
        return False

def test_coordinate_mapping():
    """Test coordinate mapping functionality."""
    print("\nTesting coordinate mapping...")
    
    # First detect board
    detector = BoardDetector()
    board_data = detector.find_chess_board(max_attempts=1)
    
    if not board_data:
        print("❌ Cannot test coordinate mapping - board detection failed")
        return False
    
    # Test coordinate mapping
    mapper = CoordinateMapper()
    mapper.set_board_position(board_data)
    
    screenshot_img = detector.screen_capture.capture()
    if screenshot_img is None:
        print("❌ Failed to capture screenshot for coordinate mapping")
        return False
    
    try:
        config = mapper.generate_coordinate_config(screenshot_img)
        print("✅ Coordinate mapping successful!")
        print(f"   Generated {len(config['ui_elements'])} UI element types")
        return True
    except Exception as e:
        print(f"❌ Coordinate mapping failed: {e}")
        return False

def test_full_calibration():
    """Test the full calibration process."""
    print("\nTesting full calibration process...")
    
    calibrator = AutoCalibrator()
    
    try:
        config = calibrator.run_full_calibration(save_config=False)
        
        if config:
            print("✅ Full calibration successful!")
            return True, config
        else:
            print("❌ Full calibration failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Full calibration error: {e}")
        return False, None

def test_configuration_validation(config):
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    if not config:
        print("❌ No configuration to validate")
        return False
    
    calibrator = AutoCalibrator()
    calibrator.config = config
    
    try:
        test_results = calibrator.test_clock_detection()
        
        if 'error' in test_results:
            print(f"❌ Validation failed: {test_results['error']}")
            return False
        
        # Count successful detections
        total_tests = 0
        successful_tests = 0
        
        for clock_type, clock_data in test_results.items():
            for state, result in clock_data.items():
                total_tests += 1
                if result['success']:
                    successful_tests += 1
        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        print(f"✅ Validation complete: {successful_tests}/{total_tests} ({success_rate:.1%}) success rate")
        
        return success_rate > 0.3  # At least 30% success rate
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def test_visualization(config):
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    if not config:
        print("❌ No configuration to visualize")
        return False
    
    try:
        visualizer = CalibrationVisualizer()
        
        # Test basic visualization
        annotated = visualizer.visualize_detection_results(config)
        
        if annotated is not None:
            print("✅ Visualization successful!")
            
            # Test diagnostic report generation
            output_dir = visualizer.create_diagnostic_report(config, "test_diagnostic_output")
            print(f"✅ Diagnostic report created: {output_dir}")
            
            return True
        else:
            print("❌ Visualization failed")
            return False
            
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("CHESS BOARD AUTO-CALIBRATION TEST SUITE")
    print("=" * 60)
    print()
    
    print("This test suite will verify that the auto-calibration system works correctly.")
    print("Please ensure a chess game is open and visible on your screen before continuing.")
    print()
    
    input("Press Enter to start tests...")
    
    # Track test results
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Board Detection
    if test_board_detection():
        tests_passed += 1
    
    # Test 2: Coordinate Mapping
    if test_coordinate_mapping():
        tests_passed += 1
    
    # Test 3: Full Calibration
    calibration_success, config = test_full_calibration()
    if calibration_success:
        tests_passed += 1
    
    # Test 4: Configuration Validation
    if test_configuration_validation(config):
        tests_passed += 1
    
    # Test 5: Visualization
    if test_visualization(config):
        tests_passed += 1
    
    # Final Results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {tests_passed/total_tests:.1%}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The auto-calibration system is working correctly.")
    elif tests_passed >= 3:
        print("⚠️  Most tests passed. The system should work but may have some issues.")
    else:
        print("❌ Multiple tests failed. Please check your setup and try again.")
    
    print("\nNext steps:")
    if tests_passed >= 3:
        print("✅ You can now integrate this auto-calibration system with your main chess client")
        print("✅ Use the generated configuration to replace hardcoded coordinates")
    else:
        print("❌ Please resolve the test failures before using the auto-calibration system")
        print("❌ Check the troubleshooting section in README.md")

if __name__ == "__main__":
    main()
