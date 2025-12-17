#!/usr/bin/env python3
"""
Template Extractor - Core logic for extracting calibration templates.

Extracts:
- Clock digit templates (0-9) from clock images
- Chess piece templates from board screenshots
- Result label templates from game end states

These templates are used for fast template matching during gameplay.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json


# Standard dimensions for templates
DIGIT_TEMPLATE_SIZE = (30, 44)  # (width, height) - matches original templates
PIECE_TEMPLATE_SIZE = (106, 106)  # Standard piece size at 1080p equivalent
RESULT_TEMPLATE_SIZE = (50, 30)  # Result region size


def remove_background_colours(img: np.ndarray, thresh: float = 1.04) -> np.ndarray:
    """
    Remove coloured background, keeping only grayscale-ish pixels.
    Used for isolating digits and pieces from coloured backgrounds.
    """
    if img is None or img.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    
    if img.ndim == 2:
        return img  # Already grayscale
    
    img_f = img.astype(np.float32)
    b, g, r = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    eps = 1e-10
    t = thresh - 1
    
    mask = (
        (np.abs(b / (g + eps) - 1.0) < t) &
        (np.abs(b / (r + eps) - 1.0) < t) &
        (np.abs(g / (r + eps) - 1.0) < t)
    )
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray * mask).astype(np.uint8)


class TemplateExtractor:
    """
    Extracts and manages calibration templates for template matching.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialise the template extractor.
        
        Args:
            template_dir: Directory to store extracted templates.
                         Defaults to auto_calibration/templates/
        """
        if template_dir is None:
            self.template_dir = Path(__file__).parent / "templates"
        else:
            self.template_dir = Path(template_dir)
        
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Track extraction progress
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict:
        """Load extraction progress from file."""
        progress_file = self.template_dir / "extraction_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "digits": {str(i): False for i in range(10)},
            "pieces": {
                "R": False, "N": False, "B": False, "K": False, "Q": False, "P": False,
                "r": False, "n": False, "b": False, "k": False, "q": False, "p": False
            },
            "results": {
                "white_win": False,
                "black_win": False,
                "draw": False
            },
            "last_updated": None
        }
    
    def _save_progress(self):
        """Save extraction progress to file."""
        self.progress["last_updated"] = datetime.now().isoformat()
        progress_file = self.template_dir / "extraction_progress.json"
        with open(progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_missing_items(self) -> Dict[str, List[str]]:
        """
        Get list of items that still need to be extracted.
        
        Returns:
            Dictionary with keys 'digits', 'pieces', 'results' containing
            lists of missing item identifiers.
        """
        missing = {
            "digits": [k for k, v in self.progress["digits"].items() if not v],
            "pieces": [k for k, v in self.progress["pieces"].items() if not v],
            "results": [k for k, v in self.progress["results"].items() if not v]
        }
        return missing
    
    def is_complete(self) -> bool:
        """Check if all templates have been extracted."""
        missing = self.get_missing_items()
        return all(len(v) == 0 for v in missing.values())
    
    def get_completion_summary(self) -> str:
        """Get a human-readable summary of extraction progress."""
        missing = self.get_missing_items()
        
        digits_done = 10 - len(missing["digits"])
        pieces_done = 12 - len(missing["pieces"])
        results_done = 3 - len(missing["results"])
        
        lines = [
            f"📊 Template Extraction Progress:",
            f"   Digits:  {digits_done}/10  {'✓' if digits_done == 10 else '(' + ','.join(missing['digits']) + ' missing)'}",
            f"   Pieces:  {pieces_done}/12  {'✓' if pieces_done == 12 else '(' + ','.join(missing['pieces']) + ' missing)'}",
            f"   Results: {results_done}/3   {'✓' if results_done == 3 else '(' + ','.join(missing['results']) + ' missing)'}"
        ]
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Digit Extraction
    # -------------------------------------------------------------------------
    
    def extract_digits_from_clock(
        self,
        clock_img: np.ndarray,
        digit_positions: Dict[str, float],
        known_time: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract digit templates from a clock image.
        
        Args:
            clock_img: BGR image of the clock region
            digit_positions: Dictionary with d1_start, d1_end, d2_start, etc.
                            as fractions of clock width
            known_time: If provided, the known time in seconds (e.g., 180 for 3:00)
                       This allows us to label the extracted digits correctly.
        
        Returns:
            Dictionary mapping digit value (0-9) to template image
        """
        if clock_img is None or clock_img.size == 0:
            return {}
        
        # Process image
        if clock_img.ndim == 3:
            processed = remove_background_colours(clock_img, thresh=1.6)
        else:
            processed = clock_img.copy()
        
        h, w = processed.shape[:2]
        
        # Extract digit regions using calibrated positions
        def get_digit_region(start_key: str, end_key: str) -> np.ndarray:
            start = int(digit_positions[start_key] * w)
            end = int(digit_positions[end_key] * w)
            region = processed[:, start:end]
            if region.size == 0:
                return None
            # Resize to standard template size
            return cv2.resize(region, DIGIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        
        d1 = get_digit_region('d1_start', 'd1_end')
        d2 = get_digit_region('d2_start', 'd2_end')
        d3 = get_digit_region('d3_start', 'd3_end')
        d4 = get_digit_region('d4_start', 'd4_end')
        
        extracted = {}
        
        if known_time is not None:
            # Convert time to digit values
            minutes = known_time // 60
            seconds = known_time % 60
            
            d1_val = minutes // 10
            d2_val = minutes % 10
            d3_val = seconds // 10
            d4_val = seconds % 10
            
            if d1 is not None:
                extracted[d1_val] = d1
            if d2 is not None:
                extracted[d2_val] = d2
            if d3 is not None:
                extracted[d3_val] = d3
            if d4 is not None:
                extracted[d4_val] = d4
        
        return extracted
    
    def save_digit_template(self, digit: int, template: np.ndarray, overwrite: bool = False) -> bool:
        """
        Save a digit template to disk.
        
        Args:
            digit: The digit value (0-9)
            template: Grayscale template image
            overwrite: If True, overwrite existing template
        
        Returns:
            True if saved successfully
        """
        if not 0 <= digit <= 9:
            return False
        
        digit_dir = self.template_dir / "digits"
        digit_dir.mkdir(exist_ok=True)
        
        filepath = digit_dir / f"{digit}.png"
        
        if filepath.exists() and not overwrite:
            return False
        
        # Ensure template is the right size
        if template.shape != (DIGIT_TEMPLATE_SIZE[1], DIGIT_TEMPLATE_SIZE[0]):
            template = cv2.resize(template, DIGIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(str(filepath), template)
        self.progress["digits"][str(digit)] = True
        self._save_progress()
        return True
    
    def load_digit_templates(self) -> Optional[np.ndarray]:
        """
        Load all digit templates as a stacked array for template matching.
        
        Returns:
            10xHxW array of digit templates (0-9), or None if incomplete
        """
        digit_dir = self.template_dir / "digits"
        templates = []
        
        for i in range(10):
            filepath = digit_dir / f"{i}.png"
            if not filepath.exists():
                return None
            
            template = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if template is None:
                return None
            
            # Ensure consistent size
            if template.shape != (DIGIT_TEMPLATE_SIZE[1], DIGIT_TEMPLATE_SIZE[0]):
                template = cv2.resize(template, DIGIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
            
            templates.append(template)
        
        return np.stack(templates, axis=0)
    
    # -------------------------------------------------------------------------
    # Piece Extraction
    # -------------------------------------------------------------------------
    
    def extract_pieces_from_starting_position(
        self,
        board_img: np.ndarray,
        bottom: str = 'w'
    ) -> Dict[str, np.ndarray]:
        """
        Extract all piece templates from a starting position screenshot.
        
        The starting position is deterministic, so we know exactly where each piece is:
        - Row 0 (rank 8 if white bottom): r n b q k b n r
        - Row 1 (rank 7 if white bottom): p p p p p p p p
        - Row 6 (rank 2 if white bottom): P P P P P P P P
        - Row 7 (rank 1 if white bottom): R N B Q K B N R
        
        Args:
            board_img: BGR image of the chess board (must be 8x8 squares)
            bottom: 'w' if white is at bottom, 'b' if black
        
        Returns:
            Dictionary mapping piece symbol to template image
        """
        if board_img is None or board_img.size == 0:
            return {}
        
        h, w = board_img.shape[:2]
        step = w // 8
        
        # Process image to remove background colours
        processed = remove_background_colours(board_img)
        
        def get_square(row: int, col: int) -> np.ndarray:
            """Extract a square from the board."""
            y1, y2 = row * step, (row + 1) * step
            x1, x2 = col * step, (col + 1) * step
            region = processed[y1:y2, x1:x2]
            # Resize to standard template size
            return cv2.resize(region, (PIECE_TEMPLATE_SIZE[0], PIECE_TEMPLATE_SIZE[1]), 
                            interpolation=cv2.INTER_AREA)
        
        # Define piece positions for white at bottom
        # (row, col, piece_symbol)
        if bottom == 'w':
            positions = [
                # Black pieces (rows 0-1)
                (0, 0, 'r'), (0, 1, 'n'), (0, 2, 'b'), (0, 3, 'q'),
                (0, 4, 'k'), (0, 5, 'b'), (0, 6, 'n'), (0, 7, 'r'),
                (1, 0, 'p'),  # Just need one pawn
                # White pieces (rows 6-7)
                (7, 0, 'R'), (7, 1, 'N'), (7, 2, 'B'), (7, 3, 'Q'),
                (7, 4, 'K'), (7, 5, 'B'), (7, 6, 'N'), (7, 7, 'R'),
                (6, 0, 'P'),  # Just need one pawn
            ]
        else:
            # Black at bottom - pieces are mirrored
            positions = [
                # White pieces (rows 0-1 when black at bottom)
                (0, 0, 'R'), (0, 1, 'N'), (0, 2, 'B'), (0, 3, 'K'),
                (0, 4, 'Q'), (0, 5, 'B'), (0, 6, 'N'), (0, 7, 'R'),
                (1, 0, 'P'),
                # Black pieces (rows 6-7 when black at bottom)
                (7, 0, 'r'), (7, 1, 'n'), (7, 2, 'b'), (7, 3, 'k'),
                (7, 4, 'q'), (7, 5, 'b'), (7, 6, 'n'), (7, 7, 'r'),
                (6, 0, 'p'),
            ]
        
        extracted = {}
        for row, col, piece in positions:
            if piece not in extracted:  # Only extract each piece type once
                extracted[piece] = get_square(row, col)
        
        return extracted
    
    def save_piece_template(self, piece: str, template: np.ndarray, overwrite: bool = False) -> bool:
        """
        Save a piece template to disk.
        
        Args:
            piece: Piece symbol (R, N, B, Q, K, P for white; r, n, b, q, k, p for black)
            template: Grayscale template image
            overwrite: If True, overwrite existing template
        
        Returns:
            True if saved successfully
        """
        valid_pieces = {'R', 'N', 'B', 'Q', 'K', 'P', 'r', 'n', 'b', 'q', 'k', 'p'}
        if piece not in valid_pieces:
            return False
        
        piece_dir = self.template_dir / "pieces"
        piece_dir.mkdir(exist_ok=True)
        
        # Use descriptive filename
        colour = 'w' if piece.isupper() else 'b'
        piece_name = {
            'R': 'rook', 'r': 'rook',
            'N': 'knight', 'n': 'knight',
            'B': 'bishop', 'b': 'bishop',
            'Q': 'queen', 'q': 'queen',
            'K': 'king', 'k': 'king',
            'P': 'pawn', 'p': 'pawn'
        }[piece]
        
        filepath = piece_dir / f"{colour}_{piece_name}.png"
        
        if filepath.exists() and not overwrite:
            return False
        
        cv2.imwrite(str(filepath), template)
        self.progress["pieces"][piece] = True
        self._save_progress()
        return True
    
    def load_piece_templates(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all piece templates.
        
        Returns:
            Dictionary mapping piece symbol to template, or None if incomplete
        """
        piece_dir = self.template_dir / "pieces"
        templates = {}
        
        piece_files = {
            'R': 'w_rook.png', 'N': 'w_knight.png', 'B': 'w_bishop.png',
            'Q': 'w_queen.png', 'K': 'w_king.png', 'P': 'w_pawn.png',
            'r': 'b_rook.png', 'n': 'b_knight.png', 'b': 'b_bishop.png',
            'q': 'b_queen.png', 'k': 'b_king.png', 'p': 'b_pawn.png'
        }
        
        for piece, filename in piece_files.items():
            filepath = piece_dir / filename
            if not filepath.exists():
                return None
            
            template = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if template is None:
                return None
            
            templates[piece] = template
        
        return templates
    
    # -------------------------------------------------------------------------
    # Result Template Extraction
    # -------------------------------------------------------------------------
    
    def save_result_template(
        self, 
        result_type: str, 
        template: np.ndarray, 
        overwrite: bool = False
    ) -> bool:
        """
        Save a game result template to disk.
        
        Args:
            result_type: 'white_win', 'black_win', or 'draw'
            template: BGR image of the result region
            overwrite: If True, overwrite existing template
        
        Returns:
            True if saved successfully
        """
        valid_types = {'white_win', 'black_win', 'draw'}
        if result_type not in valid_types:
            return False
        
        result_dir = self.template_dir / "results"
        result_dir.mkdir(exist_ok=True)
        
        filename_map = {
            'white_win': 'whitewin_result.png',
            'black_win': 'blackwin_result.png',
            'draw': 'draw_result.png'
        }
        
        filepath = result_dir / filename_map[result_type]
        
        if filepath.exists() and not overwrite:
            return False
        
        cv2.imwrite(str(filepath), template)
        self.progress["results"][result_type] = True
        self._save_progress()
        return True
    
    def load_result_templates(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all result templates.
        
        Returns:
            Dictionary with 'white_win', 'black_win', 'draw' templates,
            or None if incomplete
        """
        result_dir = self.template_dir / "results"
        templates = {}
        
        filename_map = {
            'white_win': 'whitewin_result.png',
            'black_win': 'blackwin_result.png',
            'draw': 'draw_result.png'
        }
        
        for result_type, filename in filename_map.items():
            filepath = result_dir / filename
            if not filepath.exists():
                return None
            
            template = cv2.imread(str(filepath))
            if template is None:
                return None
            
            templates[result_type] = template
        
        return templates
    
    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------
    
    def extract_all_from_starting_position(
        self,
        board_img: np.ndarray,
        bottom: str = 'w',
        overwrite: bool = False
    ) -> int:
        """
        Extract all piece templates from a starting position.
        
        Args:
            board_img: Screenshot of the chess board at starting position
            bottom: 'w' or 'b' indicating which colour is at bottom
            overwrite: If True, overwrite existing templates
        
        Returns:
            Number of templates saved
        """
        pieces = self.extract_pieces_from_starting_position(board_img, bottom)
        
        saved = 0
        for piece, template in pieces.items():
            if self.save_piece_template(piece, template, overwrite):
                saved += 1
        
        return saved
    
    def extract_digits_from_known_time(
        self,
        clock_img: np.ndarray,
        digit_positions: Dict[str, float],
        time_seconds: int,
        overwrite: bool = False
    ) -> int:
        """
        Extract digit templates from a clock showing a known time.
        
        Args:
            clock_img: Screenshot of the clock region
            digit_positions: Calibrated digit positions
            time_seconds: The known time being displayed (in seconds)
            overwrite: If True, overwrite existing templates
        
        Returns:
            Number of NEW templates saved (not counting already-extracted)
        """
        digits = self.extract_digits_from_clock(clock_img, digit_positions, time_seconds)
        
        saved = 0
        for digit, template in digits.items():
            # Only save if we don't have this digit yet (or overwrite is True)
            if overwrite or not self.progress["digits"][str(digit)]:
                if self.save_digit_template(digit, template, overwrite):
                    saved += 1
        
        return saved
    
    def reset_progress(self, category: Optional[str] = None):
        """
        Reset extraction progress.
        
        Args:
            category: 'digits', 'pieces', 'results', or None for all
        """
        if category is None or category == 'digits':
            self.progress["digits"] = {str(i): False for i in range(10)}
        if category is None or category == 'pieces':
            self.progress["pieces"] = {
                "R": False, "N": False, "B": False, "K": False, "Q": False, "P": False,
                "r": False, "n": False, "b": False, "k": False, "q": False, "p": False
            }
        if category is None or category == 'results':
            self.progress["results"] = {
                "white_win": False, "black_win": False, "draw": False
            }
        self._save_progress()


if __name__ == "__main__":
    # Quick test
    extractor = TemplateExtractor()
    print(extractor.get_completion_summary())
    print("\nMissing items:", extractor.get_missing_items())
