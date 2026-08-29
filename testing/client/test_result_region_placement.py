"""Regression: the Lichess result region has to land on the score line.

The region was placed at a fixed offset from the clock's left edge
(clock_x + 155 * scale). On the fitted 4K desktop profile that put it at
x 3069-3244, running off the right-hand edge of the notation panel, which
ends at 3165. The score line ("1-0" / "0-1" / "1/2-1/2") is *centred* in
that panel and actually sits at x 2955-2979.

So result_template detection never matched on Lichess. Every game end fell
through to the last-resort clock-position fallback instead: 6 of 6 games in
2026-08-22 19:47:11, and the same in every other Lichess session on record.
Scoring the six saved endings against the profile templates gave 0.20-0.35
against a 0.8 threshold. The cost was that the bot kept clicking at a dead
board - a failed move, a retry, and a deliberate mouse slip - in 4 of those
6 games before the fallback noticed.

The height mattered too. The old 68px crop swallowed the termination
sentence under the score ("White resigned", "White time out", "Checkmate"),
which differs per ending: across 23 real Lichess endings a 68px crop
fragmented into 14 clusters, while the score line alone separates cleanly
into 1-0 / 0-1 / draw (within-cluster min 0.88, cross-cluster max 0.53, and
worst score 0.16 against 238 mid-game frames).

Calibration JSONs are gitignored, so these assert the invariants the
calculator has to satisfy rather than reading a fitted profile.
"""
import unittest

from auto_calibration.coordinate_calculator import CoordinateCalculator

# The 4K desktop fit these coordinates come from (auto_calibration/
# calibrations/desktop.json, 2026-07-01): a 1656px board with the clocks in
# the right-hand panel.
BOARD = {'x': 1095, 'y': 189, 'size': 1656}
CLOCKS = {
    'clock_x': 2767,
    'top_clock': {'play': {'x': 2767, 'y': 839, 'width': 219, 'height': 42}},
    'bottom_clock': {'play': {'x': 2767, 'y': 1158, 'width': 219, 'height': 42}},
}

# Measured on the six game-end screenshots of session 2026-08-22 19:47:11.
# The score line is identical in all of them to within a couple of pixels.
SCORE_LINE_X = (2955, 2979)
SCORE_LINE_Y = (935, 947)
# The sentence underneath, which must stay out of the crop.
TERMINATION_LINE_Y = (965, 981)


class ResultRegionPlacementTest(unittest.TestCase):

    def setUp(self):
        calc = CoordinateCalculator(BOARD, CLOCKS, site="lichess")
        coords = calc.calculate_all()
        self.region = coords['result_region']
        self.notation = coords['notation']

    def test_region_sits_inside_the_notation_panel(self):
        """The old clock-relative offset ran off the panel's right edge."""
        self.assertGreaterEqual(self.region['x'], self.notation['x'])
        self.assertLessEqual(
            self.region['x'] + self.region['width'],
            self.notation['x'] + self.notation['width'],
            "the result region extends past the notation panel it reads from")

    def test_region_covers_the_score_line(self):
        left, right = self.region['x'], self.region['x'] + self.region['width']
        self.assertLess(left, SCORE_LINE_X[0])
        self.assertGreater(right, SCORE_LINE_X[1])

        top, bottom = self.region['y'], self.region['y'] + self.region['height']
        self.assertLessEqual(top, SCORE_LINE_Y[0] + 1)
        self.assertGreaterEqual(bottom, SCORE_LINE_Y[1])

    def test_region_excludes_the_termination_sentence(self):
        """
        It says *how* the game ended, so including it gives a different
        template for every ending and matches nothing.
        """
        bottom = self.region['y'] + self.region['height']
        self.assertLess(
            bottom, TERMINATION_LINE_Y[0],
            "the result crop reaches the termination sentence, which varies "
            "per ending and breaks template matching")


class ResultTemplateTest(unittest.TestCase):
    """
    A fitted profile's Lichess templates must be crops of the region above.

    Extracted templates live under auto_calibration/templates/, which is
    gitignored along with the calibration JSONs, so this only runs where a
    desktop profile has actually been fitted. The placement invariants above
    are the part that travels with the repo.
    """

    NAMES = ("whitewin", "blackwin", "draw")

    def setUp(self):
        import cv2
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        self.dir = repo_root / "auto_calibration/templates/desktop/results"
        missing = [n for n in self.NAMES
                   if not (self.dir / f"{n}_result.png").exists()]
        if missing:
            self.skipTest(
                "no fitted desktop result templates ({} missing); run the "
                "offline fitter to produce them".format(", ".join(missing)))
        self.templates = {
            name: cv2.imread(str(self.dir / f"{name}_result.png"))
            for name in self.NAMES
        }
        for name, img in self.templates.items():
            self.assertIsNotNone(img, f"{name}_result.png is unreadable")

    def test_templates_match_the_region_shape(self):
        calc = CoordinateCalculator(BOARD, CLOCKS, site="lichess")
        region = calc.calculate_all()['result_region']
        for name, img in self.templates.items():
            with self.subTest(template=name):
                self.assertEqual(
                    img.shape[:2], (region['height'], region['width']),
                    f"{name}_result.png is not a crop of the result region; "
                    "compare_result_images trims to the smaller of the two, "
                    "so a mismatched template silently compares a sub-crop")

    def test_templates_carry_actual_text(self):
        """
        The previous whitewin/blackwin templates were near-blank grey
        rectangles cut from the mis-placed region - they could never match a
        real ending, and a blank template is a false-positive risk besides.
        """
        for name, img in self.templates.items():
            with self.subTest(template=name):
                gray = img.mean(axis=2)
                ink = int((gray < 140).sum())
                self.assertGreater(
                    ink, 30,
                    f"{name}_result.png has almost no dark pixels, so it is "
                    "not a picture of a score line")

    def test_the_three_results_are_distinguishable(self):
        from chessimage.image_scrape_utils import compare_result_images
        names = sorted(self.templates)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                with self.subTest(pair=(a, b)):
                    self.assertLess(
                        compare_result_images(self.templates[a], self.templates[b]),
                        0.8,
                        f"{a} and {b} templates are not separable at the "
                        "detection threshold")


if __name__ == "__main__":
    unittest.main()
