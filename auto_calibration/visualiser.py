#!/usr/bin/env python3
"""
Calibration Visualisation Module

Creates debug visualisations for calibration results.
Generates multi-panel debug images and summary reports.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from .utils import (
    draw_rectangle,
    draw_grid,
    put_text,
    save_image,
    ensure_directory,
    get_output_directory,
    extract_region,
    create_board_colour_mask
)


class CalibrationVisualiser:
    """
    Creates debug visualisations for calibration results.
    
    Generates:
    - Original screenshot with overlays
    - Colour segmentation masks
    - Board detection visualisation
    - Clock sweep visualisation
    - Final overlay with all elements
    - Extracted region samples
    - Text summary report
    """
    
    # Colours for different elements (BGR)
    COLOUR_BOARD = (0, 255, 0)       # Green
    COLOUR_GRID = (0, 200, 0)        # Dark green
    COLOUR_CLOCK_PLAY = (0, 0, 255)  # Red
    COLOUR_CLOCK_OTHER = (255, 128, 0)  # Orange
    COLOUR_NOTATION = (0, 255, 255)  # Yellow
    COLOUR_RATING = (255, 0, 255)    # Magenta
    COLOUR_TEXT = (255, 255, 255)    # White
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialise visualiser.
        
        Args:
            output_dir: Directory for output files. If None, creates timestamped dir.
        """
        if output_dir is None:
            output_dir = get_output_directory()
        self.output_dir = Path(output_dir)
        ensure_directory(self.output_dir)
        
        # Create subdirectory for extracted regions
        self.regions_dir = self.output_dir / "extracted_regions"
        ensure_directory(self.regions_dir)
    
    def visualise_all(self, image: np.ndarray,
                     board_detection: Optional[Dict],
                     clock_detection: Optional[Dict],
                     coordinates: Optional[Dict]) -> Dict[str, str]:
        """
        Create all visualisations.
        
        Args:
            image: Original screenshot.
            board_detection: Board detection result.
            clock_detection: Clock detection result.
            coordinates: Final coordinates.
        
        Returns:
            Dictionary mapping visualisation names to file paths.
        """
        outputs = {}
        
        # 1. Save original screenshot
        path = self.output_dir / "01_original.png"
        save_image(str(path), image)
        outputs['original'] = str(path)
        
        # 2. Colour segmentation
        if board_detection:
            path = self._visualise_colour_segmentation(image)
            outputs['colour_segmentation'] = str(path)
        
        # 3. Board detection
        if board_detection:
            path = self._visualise_board_detection(image, board_detection)
            outputs['board_detection'] = str(path)
        
        # 4. Clock detection
        if clock_detection:
            path = self._visualise_clock_detection(image, board_detection, clock_detection)
            outputs['clock_detection'] = str(path)
        
        # 5. Final overlay
        if coordinates:
            path = self._visualise_final_overlay(image, coordinates)
            outputs['final_overlay'] = str(path)
        
        # 6. Extract and save regions
        if coordinates:
            region_paths = self._extract_and_save_regions(image, coordinates)
            outputs['regions'] = region_paths
        
        # 7. Generate text report
        path = self._generate_report(board_detection, clock_detection, coordinates)
        outputs['report'] = str(path)
        
        return outputs
    
    def _visualise_colour_segmentation(self, image: np.ndarray) -> Path:
        """Visualise colour segmentation masks."""
        # Create mask
        mask = create_board_colour_mask(image)
        
        # Convert to colour for visualisation
        mask_colour = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Create side-by-side comparison
        h, w = image.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = image
        combined[:, w:] = mask_colour
        
        # Add labels
        combined = put_text(combined, "Original", 10, 30)
        combined = put_text(combined, "Colour Mask", w + 10, 30)
        
        path = self.output_dir / "02_colour_segmentation.png"
        save_image(str(path), combined)
        return path
    
    def _visualise_board_detection(self, image: np.ndarray,
                                   board_detection: Dict) -> Path:
        """Visualise board detection result."""
        vis = image.copy()
        
        x = board_detection['x']
        y = board_detection['y']
        size = board_detection['size']
        confidence = board_detection.get('confidence', 0)
        
        # Draw board boundary
        vis = draw_rectangle(vis, x, y, size, size,
                            self.COLOUR_BOARD, thickness=3)
        
        # Draw 8x8 grid
        vis = draw_grid(vis, x, y, size, 8, self.COLOUR_GRID, thickness=1)
        
        # Add info text
        info = f"Board: ({x}, {y}) {size}x{size} conf={confidence:.2f}"
        vis = put_text(vis, info, 10, 30)
        
        step = size // 8
        vis = put_text(vis, f"Step size: {step}px", 10, 60)
        
        path = self.output_dir / "03_board_detection.png"
        save_image(str(path), vis)
        return path
    
    def _visualise_clock_detection(self, image: np.ndarray,
                                   board_detection: Optional[Dict],
                                   clock_detection: Dict) -> Path:
        """Visualise clock detection results."""
        vis = image.copy()
        
        # Draw board if available
        if board_detection:
            x = board_detection['x']
            y = board_detection['y']
            size = board_detection['size']
            vis = draw_rectangle(vis, x, y, size, size, self.COLOUR_BOARD, thickness=2)
        
        # Draw clock positions
        y_offset = 30
        
        for clock_type in ['top_clock', 'bottom_clock']:
            if clock_type not in clock_detection:
                continue
            
            for state, coords in clock_detection[clock_type].items():
                cx = coords['x']
                cy = coords['y']
                cw = coords.get('width', 147)
                ch = coords.get('height', 44)
                
                # Use different colour for 'play' state
                colour = self.COLOUR_CLOCK_PLAY if state == 'play' else self.COLOUR_CLOCK_OTHER
                
                vis = draw_rectangle(vis, cx, cy, cw, ch, colour, thickness=2)
                
                # Label
                label = f"{clock_type.split('_')[0]}_{state}"
                if 'time_value' in coords and coords['time_value'] is not None:
                    label += f" ({coords['time_value']}s)"
                vis = put_text(vis, label, cx, cy - 5, colour, font_scale=0.4)
        
        # Add summary
        detection_count = clock_detection.get('detection_count', 0)
        clock_x = clock_detection.get('clock_x', 0)
        vis = put_text(vis, f"Clock X: {clock_x}, Detections: {detection_count}", 10, 30)
        
        path = self.output_dir / "04_clock_detection.png"
        save_image(str(path), vis)
        return path
    
    def _visualise_final_overlay(self, image: np.ndarray,
                                 coordinates: Dict) -> Path:
        """Create final overlay with all detected elements."""
        vis = image.copy()
        
        # Draw board
        if 'board' in coordinates:
            b = coordinates['board']
            vis = draw_rectangle(vis, b['x'], b['y'], b['width'], b['height'],
                                self.COLOUR_BOARD, thickness=3)
            vis = draw_grid(vis, b['x'], b['y'], b['width'], 8,
                           self.COLOUR_GRID, thickness=1)
        
        # Draw clocks
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type not in coordinates:
                continue
            
            for state, coords in coordinates[clock_type].items():
                colour = self.COLOUR_CLOCK_PLAY if state == 'play' else self.COLOUR_CLOCK_OTHER
                vis = draw_rectangle(vis, coords['x'], coords['y'],
                                    coords['width'], coords['height'],
                                    colour, thickness=2)
        
        # Draw notation
        if 'notation' in coordinates:
            n = coordinates['notation']
            vis = draw_rectangle(vis, n['x'], n['y'], n['width'], n['height'],
                                self.COLOUR_NOTATION, thickness=2)
            vis = put_text(vis, "notation", n['x'], n['y'] - 5,
                          self.COLOUR_NOTATION, font_scale=0.4)
        
        # Draw ratings
        if 'rating' in coordinates:
            for name, coords in coordinates['rating'].items():
                vis = draw_rectangle(vis, coords['x'], coords['y'],
                                    coords['width'], coords['height'],
                                    self.COLOUR_RATING, thickness=1)
        
        # Add legend
        legend_y = 30
        vis = put_text(vis, "Legend:", 10, legend_y)
        legend_y += 25
        vis = put_text(vis, "Green: Board", 10, legend_y, self.COLOUR_BOARD, font_scale=0.5)
        legend_y += 20
        vis = put_text(vis, "Red: Clock (play)", 10, legend_y, self.COLOUR_CLOCK_PLAY, font_scale=0.5)
        legend_y += 20
        vis = put_text(vis, "Orange: Clock (other)", 10, legend_y, self.COLOUR_CLOCK_OTHER, font_scale=0.5)
        legend_y += 20
        vis = put_text(vis, "Yellow: Notation", 10, legend_y, self.COLOUR_NOTATION, font_scale=0.5)
        legend_y += 20
        vis = put_text(vis, "Magenta: Rating", 10, legend_y, self.COLOUR_RATING, font_scale=0.5)
        
        path = self.output_dir / "05_final_overlay.png"
        save_image(str(path), vis)
        return path
    
    def _extract_and_save_regions(self, image: np.ndarray,
                                  coordinates: Dict) -> Dict[str, str]:
        """Extract and save individual regions."""
        paths = {}
        
        # Extract board
        if 'board' in coordinates:
            b = coordinates['board']
            region = extract_region(image, b['x'], b['y'], b['width'], b['height'])
            if region is not None:
                path = self.regions_dir / "board.png"
                save_image(str(path), region)
                paths['board'] = str(path)
        
        # Extract clocks
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type not in coordinates:
                continue
            
            for state, coords in coordinates[clock_type].items():
                region = extract_region(image, coords['x'], coords['y'],
                                       coords['width'], coords['height'])
                if region is not None:
                    path = self.regions_dir / f"{clock_type}_{state}.png"
                    save_image(str(path), region)
                    paths[f"{clock_type}_{state}"] = str(path)
        
        # Extract notation
        if 'notation' in coordinates:
            n = coordinates['notation']
            region = extract_region(image, n['x'], n['y'], n['width'], n['height'])
            if region is not None:
                path = self.regions_dir / "notation.png"
                save_image(str(path), region)
                paths['notation'] = str(path)
        
        # Extract ratings
        if 'rating' in coordinates:
            for name, coords in coordinates['rating'].items():
                region = extract_region(image, coords['x'], coords['y'],
                                       coords['width'], coords['height'])
                if region is not None:
                    path = self.regions_dir / f"rating_{name}.png"
                    save_image(str(path), region)
                    paths[f"rating_{name}"] = str(path)
        
        return paths
    
    def _generate_report(self, board_detection: Optional[Dict],
                        clock_detection: Optional[Dict],
                        coordinates: Optional[Dict]) -> Path:
        """Generate text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("CALIBRATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")
        
        # Board detection
        lines.append("-" * 40)
        lines.append("BOARD DETECTION")
        lines.append("-" * 40)
        
        if board_detection:
            lines.append(f"Position: ({board_detection['x']}, {board_detection['y']})")
            lines.append(f"Size: {board_detection['size']}x{board_detection['size']}")
            lines.append(f"Step size: {board_detection.get('step', board_detection['size'] // 8)}")
            lines.append(f"Confidence: {board_detection.get('confidence', 0):.3f}")
            lines.append(f"Method: {board_detection.get('method', 'unknown')}")
        else:
            lines.append("Board detection FAILED")
        
        lines.append("")
        
        # Clock detection
        lines.append("-" * 40)
        lines.append("CLOCK DETECTION")
        lines.append("-" * 40)
        
        if clock_detection:
            lines.append(f"Clock X position: {clock_detection.get('clock_x', 'N/A')}")
            lines.append(f"Total detections: {clock_detection.get('detection_count', 0)}")
            lines.append("")
            
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in clock_detection:
                    lines.append(f"{clock_type.upper()}:")
                    for state, coords in clock_detection[clock_type].items():
                        time_val = coords.get('time_value', 'N/A')
                        estimated = " (estimated)" if coords.get('estimated', False) else ""
                        lines.append(f"  {state}: ({coords['x']}, {coords['y']}) - {time_val}s{estimated}")
                    lines.append("")
        else:
            lines.append("Clock detection FAILED or not performed")
        
        lines.append("")
        
        # Final coordinates
        lines.append("-" * 40)
        lines.append("FINAL COORDINATES")
        lines.append("-" * 40)
        
        if coordinates:
            if 'board' in coordinates:
                b = coordinates['board']
                lines.append(f"Board: ({b['x']}, {b['y']}) [{b['width']}x{b['height']}]")
            
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in coordinates:
                    lines.append(f"\n{clock_type.upper()}:")
                    for state, coords in coordinates[clock_type].items():
                        lines.append(f"  {state}: ({coords['x']}, {coords['y']})")
            
            if 'notation' in coordinates:
                n = coordinates['notation']
                lines.append(f"\nNotation: ({n['x']}, {n['y']}) [{n['width']}x{n['height']}]")
            
            if 'rating' in coordinates:
                lines.append("\nRatings:")
                for name, coords in coordinates['rating'].items():
                    lines.append(f"  {name}: ({coords['x']}, {coords['y']})")
        else:
            lines.append("No coordinates calculated")
        
        lines.append("")
        lines.append("=" * 60)
        
        # Write report
        path = self.output_dir / "report.txt"
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        
        return path
    
    def get_output_dir(self) -> Path:
        """Get the output directory path."""
        return self.output_dir


def visualise_calibration(image: np.ndarray,
                         board_detection: Optional[Dict],
                         clock_detection: Optional[Dict],
                         coordinates: Optional[Dict],
                         output_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Convenience function to create all visualisations.
    
    Args:
        image: Original screenshot.
        board_detection: Board detection result.
        clock_detection: Clock detection result.
        coordinates: Final coordinates.
        output_dir: Output directory.
    
    Returns:
        Dictionary mapping visualisation names to file paths.
    """
    visualiser = CalibrationVisualiser(output_dir)
    return visualiser.visualise_all(image, board_detection, clock_detection, coordinates)
