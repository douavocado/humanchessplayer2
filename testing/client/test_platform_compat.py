"""Guards on common/platform_compat.py, the OS-dependent seam.

The reason this file exists: the Linux/X11 path is the one this project was
built and tuned against, and neither the parity harness nor the calibration
profiles can run on Windows to catch a regression in it. So the binary-name
resolution -- the one place where a careless edit could silently hand Linux a
*different Stockfish* than the one every measurement was taken with -- gets
asserted explicitly, on both platforms, from either host OS.

`engine_path` reads the module-level IS_WINDOWS, so both arms are reachable
from either host by patching it; REPO_ROOT is patched to a tmpdir so the
results do not depend on which binaries happen to be on this machine.
"""

import os
import tempfile
import unittest
from pathlib import Path

from common import platform_compat as pc


class EnginePathTest(unittest.TestCase):
    """Which binary each platform picks out of Engines/."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "Engines").mkdir()
        self._saved = (pc.REPO_ROOT, pc.IS_WINDOWS)
        pc.REPO_ROOT = self.root

    def tearDown(self):
        pc.REPO_ROOT, pc.IS_WINDOWS = self._saved
        self._tmp.cleanup()

    def _touch(self, name):
        (self.root / "Engines" / name).write_bytes(b"")

    def test_linux_picks_the_historically_hardcoded_names(self):
        """The pre-refactor constants were these exact two paths.

        If this breaks, Linux is running a different engine than every
        tuning measurement in the repo was taken against.
        """
        pc.IS_WINDOWS = False
        self._touch("stockfish17-ubuntu")
        self._touch("stockfish16-ubuntu")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish17-ubuntu")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish16")),
                         "stockfish16-ubuntu")

    def test_linux_prefers_ubuntu_suffix_over_bare_name(self):
        pc.IS_WINDOWS = False
        self._touch("stockfish17-ubuntu")
        self._touch("stockfish17")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish17-ubuntu")

    def test_windows_prefers_the_windows_exe(self):
        pc.IS_WINDOWS = True
        self._touch("stockfish17-windows.exe")
        self._touch("stockfish17.exe")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish17-windows.exe")

    def test_windows_accepts_the_alternate_names(self):
        pc.IS_WINDOWS = True
        self._touch("stockfish16-x86-64-avx2.exe")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish16")),
                         "stockfish16-x86-64-avx2.exe")

    def test_windows_lone_engine_serves_every_slot(self):
        """One unnamed Windows binary covers both the play and ponder slots.

        Concurrent engine processes do not need distinct executables --
        popen_uci spawns a fresh process from the same file each call -- so
        the 17/16 split is not a requirement to reproduce on Windows.
        """
        pc.IS_WINDOWS = True
        self._touch("stockfish18-windows.exe")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish18-windows.exe")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish16")),
                         "stockfish18-windows.exe")

    def test_exact_version_beats_the_lone_engine_fallback(self):
        """Adding properly-named binaries later must take effect with no code
        change -- otherwise the fallback would pin the wrong engine forever."""
        pc.IS_WINDOWS = True
        self._touch("stockfish18-windows.exe")
        self._touch("stockfish17-windows.exe")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish17-windows.exe")

    def test_two_unnamed_engines_are_not_guessed_between(self):
        """Ambiguity must not silently pick one; fall through to the
        canonical name so the failure says what is missing."""
        pc.IS_WINDOWS = True
        self._touch("stockfish18-windows.exe")
        self._touch("stockfish19-windows.exe")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish17-windows.exe")

    def test_lone_engine_fallback_is_windows_only(self):
        """Linux resolution must stay exact: it is the platform every tuning
        measurement in the repo was taken on."""
        pc.IS_WINDOWS = False
        self._touch("stockfish18-ubuntu")
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish17-ubuntu")

    def test_missing_binary_falls_back_to_the_canonical_name(self):
        """Nothing on disk: return the name the user is expected to supply,
        so the eventual launch failure names the right file rather than an
        empty string or a path from the wrong platform."""
        pc.IS_WINDOWS = True
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish17-windows.exe")
        pc.IS_WINDOWS = False
        self.assertEqual(os.path.basename(pc.engine_path("stockfish17")),
                         "stockfish17-ubuntu")

    def test_paths_are_absolute(self):
        """Resolved against the repo root, not the CWD -- so the bot does not
        have to be launched from the project directory."""
        pc.IS_WINDOWS = False
        self._touch("stockfish17-ubuntu")
        self.assertTrue(os.path.isabs(pc.engine_path("stockfish17")))


class ScreenCaptureContractTest(unittest.TestCase):
    """The capture contract every consumer relies on.

    Callers slice ``[:, :, :3]`` off the result and hand it to OpenCV as BGR,
    so the array must be 4-channel BGRA. mss and fastgrab both satisfy this;
    dxcam would not.
    """

    def test_backend_is_constructed_lazily(self):
        """chessimage builds its SCREEN_CAPTURE at import time. If that
        construction touched the display, importing the module would again
        require a capture library and an active session -- which is exactly
        what used to make 7 client test modules unimportable."""
        cap = pc.make_screen_capture()
        self.assertTrue(hasattr(cap, "capture"))

    @unittest.skipUnless(pc.IS_WINDOWS, "exercises the mss-backed Windows arm")
    def test_windows_capture_returns_writable_bgra(self):
        cap = pc.make_screen_capture()
        img = cap.capture((0, 0, 32, 16))
        self.assertEqual(img.shape, (16, 32, 4))
        self.assertEqual(img.dtype.name, "uint8")
        # mss hands back a read-only view of a buffer it reuses between grabs;
        # several call sites do not .copy(), so the shim must return its own.
        self.assertTrue(img.flags.writeable)


if __name__ == "__main__":
    unittest.main()
