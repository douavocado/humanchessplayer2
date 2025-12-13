#!/usr/bin/env python3
"""
Calibration Visualizer

Tools for visualizing calibration results, debugging detection issues,
and validating the auto-calibration system.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from fastgrab import screenshot

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "chessimage"))

from board_detector import BoardDetector
from coordinate_mapper import CoordinateMapper
from utils import simple_read_clock

class CalibrationVisualizer:
    """Visualizes calibration results and provides debugging tools."""
    
    def __init__(self):
        self.screen_capture = screenshot.Screenshot()
        self.colors = {
            'board': (0, 255, 0),      # Green
            'clock': (255, 0, 0),      # Red  
            'notation': (0, 0, 255),   # Blue
            'rating': (255, 255, 0),   # Yellow
            'detected': (255, 0, 255)  # Magenta
        }
    
    def visualize_detection_results(self, config: Dict, save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a visual representation of detection results.
        
        Args:
            config: Configuration dictionary from calibration
            save_path: Optional path to save the visualization
            
        Returns:
            Annotated screenshot image
        """
        # Capture current screenshot
        screenshot_img = self.screen_capture.capture()
        if screenshot_img is None:
            raise ValueError("Failed to capture screenshot")
        
        # Create a copy for annotation
        annotated = screenshot_img.copy()
        
        # Draw board detection
        if 'board_detection' in config:
            board_pos = config['board_detection']['position']
            self._draw_rectangle(annotated, board_pos, self.colors['board'], 
                               f"Board ({config['board_detection']['method']})", thickness=3)
        
        # Draw UI elements
        if 'ui_elements' in config:
            for element_type, element_data in config['ui_elements'].items():
                color = self.colors.get(element_type.split('_')[0], (128, 128, 128))
                
                if element_type in ['bottom_clock', 'top_clock']:
                    self._draw_clock_positions(annotated, element_data, color, element_type)
                elif element_type == 'notation':
                    pos = (element_data['x'], element_data['y'], 
                          element_data['width'], element_data['height'])
                    self._draw_rectangle(annotated, pos, self.colors['notation'], "Notation")
                elif element_type == 'rating':
                    self._draw_rating_positions(annotated, element_data, self.colors['rating'])
        
        # Add legend
        self._add_legend(annotated)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"Visualization saved to {save_path}")
        
        return annotated
    
    def _draw_rectangle(self, image: np.ndarray, position: Tuple, color: Tuple, 
                       label: str, thickness: int = 2):
        """Draw a labeled rectangle on the image."""
        x, y, w, h = position
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Add label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(image, (x, y - label_size[1] - 10), 
                     (x + label_size[0] + 10, y), color, -1)
        cv2.putText(image, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_clock_positions(self, image: np.ndarray, clock_data: Dict, color: Tuple, clock_type: str):
        """Draw all clock position states."""
        for state, coords in clock_data.items():
            pos = (coords['x'], coords['y'], coords['width'], coords['height'])
            label = f"{clock_type.replace('_', ' ').title()} ({state})"
            self._draw_rectangle(image, pos, color, label, thickness=1)
    
    def _draw_rating_positions(self, image: np.ndarray, rating_data: Dict, color: Tuple):
        """Draw rating positions."""
        for rating_type, coords in rating_data.items():
            pos = (coords['x'], coords['y'], coords['width'], coords['height'])
            label = f"Rating ({rating_type})"
            self._draw_rectangle(image, pos, color, label, thickness=1)
    
    def _add_legend(self, image: np.ndarray):
        """Add a color legend to the image."""
        legend_items = [
            ("Board", self.colors['board']),
            ("Clock", self.colors['clock']),
            ("Notation", self.colors['notation']),
            ("Rating", self.colors['rating'])
        ]
        
        # Position legend in top-right corner
        legend_x = image.shape[1] - 200
        legend_y = 30
        
        # Draw legend background
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (legend_x + 180, legend_y + len(legend_items) * 25 + 10), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (legend_x + 180, legend_y + len(legend_items) * 25 + 10), 
                     (255, 255, 255), 1)
        
        # Draw legend items
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            cv2.rectangle(image, (legend_x, y_pos), (legend_x + 15, y_pos + 15), color, -1)
            cv2.putText(image, label, (legend_x + 25, y_pos + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def test_clock_regions_visual(self, config: Dict) -> Dict:
        """
        Test clock regions and create visual feedback.
        
        Returns:
            Dictionary with test results and extracted clock images
        """
        screenshot_img = self.screen_capture.capture()
        if screenshot_img is None:
            return {'error': 'Failed to capture screenshot'}
        
        results = {
            'clock_images': {},
            'clock_results': {},
            'annotated_screenshot': None
        }
        
        # Create annotated version
        annotated = screenshot_img.copy()
        
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in config['ui_elements']:
                results['clock_images'][clock_type] = {}
                results['clock_results'][clock_type] = {}
                
                for state, coords in config['ui_elements'][clock_type].items():
                    x, y = coords['x'], coords['y']
                    w, h = coords['width'], coords['height']
                    
                    # Extract clock region
                    clock_region = screenshot_img[y:y+h, x:x+w]
                    results['clock_images'][clock_type][state] = clock_region
                    
                    # Test clock reading
                    try:
                        time_value = simple_read_clock(clock_region)
                        success = time_value is not None
                        results['clock_results'][clock_type][state] = {
                            'success': success,
                            'value': time_value
                        }
                        
                        # Color code the rectangle based on success
                        color = (0, 255, 0) if success else (0, 0, 255)  # Green if success, red if fail
                        
                    except Exception as e:
                        results['clock_results'][clock_type][state] = {
                            'success': False,
                            'error': str(e)
                        }
                        color = (0, 0, 255)  # Red for error
                    
                    # Draw rectangle on annotated image
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    label = f"{clock_type}_{state}"
                    cv2.putText(annotated, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        results['annotated_screenshot'] = annotated
        return results
    
    def create_diagnostic_report(self, config: Dict, output_dir: str = "diagnostic_output"):
        """
        Create a comprehensive diagnostic report with images and analysis.
        """
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(exist_ok=True)
        
        print(f"Creating diagnostic report in {output_path}")
        
        # 1. Main visualization
        print("Creating main visualization...")
        main_viz = self.visualize_detection_results(config)
        cv2.imwrite(str(output_path / "01_main_visualization.png"), main_viz)
        
        # 2. Clock region tests
        print("Testing clock regions...")
        clock_test_results = self.test_clock_regions_visual(config)
        
        if 'annotated_screenshot' in clock_test_results:
            cv2.imwrite(str(output_path / "02_clock_test_results.png"), 
                       clock_test_results['annotated_screenshot'])
        
        # 3. Individual clock images
        print("Saving individual clock regions...")
        if 'clock_images' in clock_test_results:
            for clock_type, clock_states in clock_test_results['clock_images'].items():
                for state, clock_img in clock_states.items():
                    filename = f"03_clock_{clock_type}_{state}.png"
                    cv2.imwrite(str(output_path / filename), clock_img)
        
        # 4. Generate text report
        print("Generating text report...")
        self._create_text_report(config, clock_test_results, output_path / "04_diagnostic_report.txt")
        
        # 5. Save configuration
        config_path = output_path / "05_configuration.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Diagnostic report complete! Check {output_path}")
        
        return str(output_path)
    
    def _create_text_report(self, config: Dict, test_results: Dict, output_file: Path):
        """Create a text-based diagnostic report."""
        with open(output_file, 'w') as f:
            f.write("CHESS BOARD AUTO-CALIBRATION DIAGNOSTIC REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Configuration summary
            f.write("CONFIGURATION SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Board Position: {config['board_detection']['position']}\n")
            f.write(f"Detection Method: {config['board_detection']['method']}\n")
            f.write(f"Confidence: {config['board_detection']['confidence']:.3f}\n")
            f.write(f"Template Scale: {config.get('template_scale', 'Unknown')}\n")
            f.write(f"Timestamp: {config.get('timestamp', 'Unknown')}\n\n")
            
            # Clock test results
            f.write("CLOCK DETECTION TEST RESULTS\n")
            f.write("-" * 30 + "\n")
            
            if 'clock_results' in test_results:
                total_tests = 0
                successful_tests = 0
                
                for clock_type, clock_data in test_results['clock_results'].items():
                    f.write(f"\n{clock_type.replace('_', ' ').title()}:\n")
                    
                    for state, result in clock_data.items():
                        total_tests += 1
                        status = "✅ PASS" if result['success'] else "❌ FAIL"
                        f.write(f"  {state}: {status}")
                        
                        if result['success']:
                            successful_tests += 1
                            if 'value' in result and result['value'] is not None:
                                time_str = f"{result['value']//60}:{result['value']%60:02d}"
                                f.write(f" (Time: {time_str})")
                        else:
                            if 'error' in result:
                                f.write(f" (Error: {result['error']})")
                        
                        f.write("\n")
                
                f.write(f"\nOverall Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)\n")
            
            # UI Elements summary
            f.write("\nUI ELEMENT POSITIONS\n")
            f.write("-" * 30 + "\n")
            for element_type, element_data in config['ui_elements'].items():
                f.write(f"\n{element_type}:\n")
                if isinstance(element_data, dict):
                    for sub_type, coords in element_data.items():
                        if isinstance(coords, dict) and 'x' in coords:
                            f.write(f"  {sub_type}: ({coords['x']}, {coords['y']}) [{coords['width']}x{coords['height']}]\n")
                        else:
                            f.write(f"  {sub_type}: {coords}\n")
    
    def show_interactive_visualization(self, config: Dict):
        """Show interactive matplotlib visualization."""
        try:
            screenshot_img = self.screen_capture.capture()
            if screenshot_img is None:
                print("Failed to capture screenshot")
                return
            
            # Convert BGR to RGB for matplotlib
            screenshot_rgb = cv2.cvtColor(screenshot_img, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(screenshot_rgb)
            
            # Draw board
            if 'board_detection' in config:
                x, y, w, h = config['board_detection']['position']
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                       edgecolor='green', facecolor='none', 
                                       label='Chess Board')
                ax.add_patch(rect)
            
            # Draw UI elements
            if 'ui_elements' in config:
                for element_type, element_data in config['ui_elements'].items():
                    if element_type in ['bottom_clock', 'top_clock']:
                        for state, coords in element_data.items():
                            x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
                            color = 'red' if 'bottom' in element_type else 'blue'
                            rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                                   edgecolor=color, facecolor='none', alpha=0.7)
                            ax.add_patch(rect)
                            ax.text(x, y-5, f"{element_type}_{state}", fontsize=8, color=color)
            
            ax.set_title("Chess Board Auto-Calibration Results")
            ax.legend()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for interactive visualization")
        except Exception as e:
            print(f"Error creating visualization: {e}")


def main():
    """Main function for running visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize calibration results")
    parser.add_argument("--config", default="auto_calibration_config.json",
                       help="Configuration file to visualize")
    parser.add_argument("--output", default="diagnostic_output",
                       help="Output directory for diagnostic files")
    parser.add_argument("--interactive", action="store_true",
                       help="Show interactive matplotlib visualization")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Please run auto_calibrator.py first to generate a configuration.")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create visualizer
    visualizer = CalibrationVisualizer()
    
    # Generate diagnostic report
    output_dir = visualizer.create_diagnostic_report(config, args.output)
    
    # Show interactive visualization if requested
    if args.interactive:
        visualizer.show_interactive_visualization(config)


if __name__ == "__main__":
    main()
