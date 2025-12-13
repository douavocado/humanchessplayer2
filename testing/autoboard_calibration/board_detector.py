#!/usr/bin/env python3
"""
Chess Board Auto-Detection Module

This module provides functionality to automatically detect chess board position
and calculate UI element coordinates relative to the board for device-independent
screen scraping.
"""

import cv2
import numpy as np
from fastgrab import screenshot
import os
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import time

# Add parent directories to path to import existing modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

from utils import remove_background_colours, SCREEN_CAPTURE

class BoardDetector:
    """Detects chess board position and calculates UI element coordinates."""
    
    def __init__(self):
        self.screen_capture = SCREEN_CAPTURE
        self.template_scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
        self.board_template = None
        self.load_board_template()
        
    def load_board_template(self):
        """Load and prepare chess board template for detection."""
        # Try to use existing board examples
        template_paths = [
            parent_dir / "chessimage" / "example_board.png",
            parent_dir / "chessimage" / "example_board_2.png"
        ]
        
        for path in template_paths:
            if path.exists():
                template = cv2.imread(str(path))
                if template is not None:
                    # Convert to grayscale and resize to standard size
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    # Resize to a standard size for consistent matching
                    self.board_template = cv2.resize(template_gray, (400, 400))
                    print(f"Loaded board template from {path}")
                    return
        
        print("Warning: No board template found. You may need to provide a reference board image.")
    
    def create_corner_template(self) -> np.ndarray:
        """Create a simple chess board corner template for detection."""
        # Create a simple 4x4 checkerboard pattern
        template = np.zeros((80, 80), dtype=np.uint8)
        square_size = 20
        
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    y1, y2 = i * square_size, (i + 1) * square_size
                    x1, x2 = j * square_size, (j + 1) * square_size
                    template[y1:y2, x1:x2] = 255
        
        return template
    
    def detect_board_multi_scale(self, screenshot_img: np.ndarray, 
                                 template: np.ndarray) -> Tuple[Optional[Tuple[int, int, float]], float]:
        """
        Detect chess board using multi-scale template matching.
        
        Returns:
            ((x, y, scale), confidence) or (None, 0) if not found
        """
        if screenshot_img is None or template is None:
            return None, 0
        
        # Convert screenshot to grayscale
        if len(screenshot_img.shape) == 3:
            screenshot_gray = cv2.cvtColor(screenshot_img, cv2.COLOR_BGR2GRAY)
        else:
            screenshot_gray = screenshot_img
        
        best_match = None
        best_confidence = 0
        best_scale = 1.0
        
        for scale in self.template_scales:
            # Resize template
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
            
            # Skip if template is larger than screenshot
            if (scaled_template.shape[0] > screenshot_gray.shape[0] or 
                scaled_template.shape[1] > screenshot_gray.shape[1]):
                continue
            
            # Perform template matching
            result = cv2.matchTemplate(screenshot_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_match = (max_loc[0], max_loc[1], scale)
                best_scale = scale
        
        return best_match, best_confidence
    
    def detect_board_edges(self, screenshot_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Alternative method: Detect board by finding straight lines and corners.
        
        Returns:
            (x, y, width, height) of detected board or None
        """
        if screenshot_img is None:
            return None
        
        # Convert to grayscale
        if len(screenshot_img.shape) == 3:
            gray = cv2.cvtColor(screenshot_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = screenshot_img
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return None
        
        # Find horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            rho, theta = line[0]
            # Check if line is roughly horizontal
            if abs(theta) < np.pi/8 or abs(theta - np.pi) < np.pi/8:
                horizontal_lines.append((rho, theta))
            # Check if line is roughly vertical
            elif abs(theta - np.pi/2) < np.pi/8:
                vertical_lines.append((rho, theta))
        
        # If we have enough lines, try to find board bounds
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Sort lines to find extremes
            h_rhos = [abs(rho) for rho, theta in horizontal_lines]
            v_rhos = [abs(rho) for rho, theta in vertical_lines]
            
            if h_rhos and v_rhos:
                min_h, max_h = min(h_rhos), max(h_rhos)
                min_v, max_v = min(v_rhos), max(v_rhos)
                
                # Estimate board position (this is a rough approximation)
                x, y = int(min_v), int(min_h)
                width = int(max_v - min_v)
                height = int(max_h - min_h)
                
                # Sanity check: board should be roughly square
                if 0.5 < width/height < 2.0 and width > 200 and height > 200:
                    return (x, y, width, height)
        
        return None
    
    def find_chess_board(self, max_attempts: int = 3) -> Optional[Dict]:
        """
        Main function to find chess board on screen.
        
        Returns:
            Dictionary with board position and metadata or None
        """
        print("Searching for chess board on screen...")
        
        best_detection = None
        best_confidence = 0
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            
            # Capture screenshot
            screenshot_img = self.screen_capture.capture()
            if screenshot_img is None:
                print(f"Failed to capture screenshot on attempt {attempt + 1}")
                continue
            
            # Method 1: Template matching with existing board template
            if self.board_template is not None:
                match, confidence = self.detect_board_multi_scale(screenshot_img, self.board_template)
                print(f"Template matching confidence: {confidence:.3f}")
                
                if match and confidence > best_confidence:
                    best_confidence = confidence
                    x, y, scale = match
                    template_height, template_width = self.board_template.shape
                    scaled_width = int(template_width * scale)
                    scaled_height = int(template_height * scale)
                    
                    best_detection = {
                        'method': 'template_matching',
                        'position': (x, y, scaled_width, scaled_height),
                        'confidence': confidence,
                        'scale': scale,
                        'template_size': (template_width, template_height)
                    }
            
            # Method 2: Try with corner template if main template fails
            if best_confidence < 0.6:
                corner_template = self.create_corner_template()
                match, confidence = self.detect_board_multi_scale(screenshot_img, corner_template)
                print(f"Corner template confidence: {confidence:.3f}")
                
                if match and confidence > best_confidence:
                    best_confidence = confidence
                    x, y, scale = match
                    # Estimate full board size from corner
                    corner_height, corner_width = corner_template.shape
                    estimated_board_size = int(corner_width * scale * 2.5)  # Rough estimate
                    
                    best_detection = {
                        'method': 'corner_template',
                        'position': (x, y, estimated_board_size, estimated_board_size),
                        'confidence': confidence,
                        'scale': scale,
                        'template_size': (corner_width, corner_height)
                    }
            
            # Method 3: Edge detection fallback
            if best_confidence < 0.4:
                edge_result = self.detect_board_edges(screenshot_img)
                if edge_result:
                    x, y, width, height = edge_result
                    print(f"Edge detection found board at ({x}, {y}) size {width}x{height}")
                    
                    best_detection = {
                        'method': 'edge_detection',
                        'position': (x, y, width, height),
                        'confidence': 0.5,  # Assign medium confidence for edge detection
                        'scale': 1.0,
                        'template_size': None
                    }
                    best_confidence = 0.5
            
            # If we found a good match, break early
            if best_confidence > 0.7:
                break
                
            # Wait a bit before next attempt
            if attempt < max_attempts - 1:
                time.sleep(1)
        
        if best_detection:
            print(f"Board detected using {best_detection['method']} with confidence {best_confidence:.3f}")
            print(f"Position: {best_detection['position']}")
            return best_detection
        else:
            print("Failed to detect chess board")
            return None


if __name__ == "__main__":
    # Test the board detector
    detector = BoardDetector()
    
    print("Starting board detection test...")
    print("Please ensure a chess board is visible on your screen.")
    
    time.sleep(3)  # Give user time to set up
    
    result = detector.find_chess_board()
    
    if result:
        print("\n=== BOARD DETECTION SUCCESSFUL ===")
        print(f"Method: {result['method']}")
        print(f"Position (x, y, width, height): {result['position']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Scale factor: {result['scale']:.2f}")
    else:
        print("\n=== BOARD DETECTION FAILED ===")
        print("Could not detect chess board on screen.")
        print("Make sure:")
        print("1. A chess game is open and visible")
        print("2. The board is not obstructed")
        print("3. There's good contrast between board and background")
