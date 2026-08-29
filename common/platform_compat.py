"""Every place this codebase has to care which OS it is running on.

The project was written for Linux/X11 and that path must stay behaviourally
identical -- the parity harness and the calibration profiles are the only
regression evidence there is, and neither can run on Windows. So the rule is:
each OS-dependent decision lives in exactly one function here, with the Linux
arm a verbatim copy of what the call site used to do inline. Nothing outside
this module should branch on ``sys.platform``.

What differs, and why:

* **Screen capture** -- ``fastgrab`` is an X11-only C extension. The Windows
  arm reimplements its contract on top of ``mss``. See ``make_screen_capture``.
* **DPI awareness** -- Windows-only, and the single most dangerous difference;
  see ``init``.
* **Caps lock** -- the manual kill-switch. X11 reads it out of ``xset q``.
* **Sound** -- ``mpg123`` is not a thing on Windows.
* **Engine binaries** -- ``Engines/`` ships ELF builds with ``-ubuntu`` names.
* **Tesseract** -- the OCR engine is a system install, not a pip one, and its
  Windows installer does not put itself on PATH. See ``configure_tesseract``.
* **Process groups** -- ``os.killpg``/``start_new_session`` are POSIX-only.
"""

import os
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

REPO_ROOT = Path(__file__).resolve().parent.parent

_dpi_initialised = False


# --------------------------------------------------------------------------
# Process-wide init
# --------------------------------------------------------------------------

def init():
    """Process-wide setup that must happen before anything touches the screen.

    On Windows this declares DPI awareness, which is the difference between
    the bot working and the bot clicking in the wrong place *with no error*.

    An un-aware process is lied to by Windows: on a 1920x1080 panel at 150%
    scaling it sees a 1280x720 desktop, and every coordinate it reads or
    writes is scaled by 2/3 behind its back. Capture backends report real
    physical pixels, so calibration would be fitted against 1920-wide frames
    while PyAutoGUI clicked at 1280-wide coordinates. Declaring awareness
    puts both sides in physical pixels, which is what the Linux/X11 path has
    always been in -- so no calibration or coordinate maths has to change.

    Must run before the capture backend is constructed (``mss`` caches monitor
    geometry when the instance is created) and before any Tk window exists.
    Idempotent, and a no-op off Windows.
    """
    global _dpi_initialised
    if _dpi_initialised or not IS_WINDOWS:
        return
    _dpi_initialised = True

    import ctypes

    # Newest first. -4 is DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, and note
    # the Context variant lives in user32 -- shcore only has the older
    # SetProcessDpiAwareness. Each call throws or returns falsey if the OS
    # predates it, or if awareness was already set (e.g. by an app manifest),
    # in which case there is nothing to do anyway.
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(-4):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # system-aware only
    except (AttributeError, OSError):
        pass


# --------------------------------------------------------------------------
# Screen capture
# --------------------------------------------------------------------------

class _MssCapture:
    """fastgrab's ``Screenshot`` contract, backed by ``mss``.

    The contract is implicit in the call sites and load-bearing: every
    consumer slices ``[:, :, :3]`` off the result, so the array must be
    4-channel **BGRA**, in that order. ``mss`` hands back BGRA natively
    (its ``.rgb`` property is the converted one -- don't reach for it);
    ``dxcam``, if anyone ever swaps it in here, returns RGB and would need
    a conversion, or the piece templates silently stop matching.

    ``mss`` instances are not thread-safe, so each capture object owns its
    own and is created lazily -- after :func:`init` has set DPI awareness.
    """

    def __init__(self):
        self._sct = None

    def _backend(self):
        if self._sct is None:
            init()  # geometry is cached on construction; awareness must precede it
            import mss
            # mss 10 renamed the factory to MSS and deprecated the old name;
            # mss 9 only has the old one, and requirements allows either.
            factory = getattr(mss, "MSS", None) or mss.mss
            self._sct = factory()
        return self._sct

    def capture(self, region=None):
        """``region`` is ``(x, y, width, height)``; omit it for the whole screen.

        Returns a fresh, writable ``(h, w, 4)`` uint8 BGRA array. The copy is
        deliberate: mss reuses its internal buffer between grabs and hands back
        a read-only view of it, while fastgrab returns a fresh array. Several
        call sites (e.g. ``image_scrape_utils.capture_board``) do not ``.copy()``
        and would otherwise be reading a buffer that the next grab overwrites.
        """
        import numpy as np  # local: keeps numpy off common.constants' import path

        sct = self._backend()
        if region is None:
            # monitors[0] is the bounding box of all monitors, which is what
            # fastgrab's root-window grab returns. monitors[1] would be just
            # the primary -- identical on a single-monitor setup, wrong on two.
            monitor = sct.monitors[0]
        else:
            x, y, width, height = region
            monitor = {"left": int(x), "top": int(y),
                       "width": int(width), "height": int(height)}

        shot = sct.grab(monitor)
        return np.array(shot, dtype=np.uint8)  # (h, w, 4), BGRA, writable copy


def make_screen_capture():
    """The screen-capture backend for this platform.

    Returns an object exposing ``capture()`` and ``capture((x, y, w, h))``,
    both yielding a 4-channel BGRA ``ndarray``.
    """
    if IS_WINDOWS:
        return _MssCapture()
    from fastgrab import screenshot
    return screenshot.Screenshot()


# --------------------------------------------------------------------------
# Keyboard
# --------------------------------------------------------------------------

def is_capslock_on():
    """Caps lock toggle state -- the bot's manual pause/kill switch."""
    if IS_WINDOWS:
        import ctypes
        # Low bit of GetKeyState is the toggle state (as opposed to the high
        # bit, which is whether the key is currently held down).
        return bool(ctypes.windll.user32.GetKeyState(0x14) & 1)  # VK_CAPITAL

    # Verbatim from the original inline X11 implementation, including its
    # implicit `return None` when the byte is neither '0' nor '1'.
    if subprocess.check_output('xset q | grep LED', shell=True)[65] == 48:
        return False
    elif subprocess.check_output('xset q | grep LED', shell=True)[65] == 49:
        return True


# --------------------------------------------------------------------------
# Sound
# --------------------------------------------------------------------------

def play_sound(sound_file):
    """Play a notification sound. Never raises -- these are only chimes."""
    if IS_WINDOWS:
        _play_sound_windows(sound_file)
        return
    # Unchanged from the original call sites, blocking included.
    os.system("mpg123 -q " + sound_file)


def _play_sound_windows(sound_file):
    import winsound
    # winsound plays WAV only, and the assets are mp3. Prefer a converted
    # sibling if one has been dropped next to it, else fall back to a system
    # beep so the audible cue is not lost entirely.
    wav = Path(sound_file).with_suffix(".wav")
    try:
        if wav.is_file():
            winsound.PlaySound(str(wav),
                               winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            winsound.MessageBeep()
    except (OSError, RuntimeError):
        pass


# --------------------------------------------------------------------------
# Tesseract OCR
# --------------------------------------------------------------------------

# Where the Windows installer actually puts it. It offers an "add to PATH"
# checkbox that is off by default, so a perfectly good install is routinely
# invisible to ``pytesseract`` - which shells out to a bare ``tesseract`` and
# reports the miss as "tesseract is not installed".
_WINDOWS_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def configure_tesseract():
    """Point ``pytesseract`` at the Tesseract binary, and say whether it worked.

    ``pytesseract`` is only a wrapper: ``pip install -r requirements.txt``
    satisfies the import but not the executable behind it, so the OCR paths
    (ratings, and the calibration button detector) fail at *run* time rather
    than import time, and silently -- ``read_rating`` just returns ``None``
    every game. This is the one runtime prerequisite the repo cannot ship
    itself, so the least it can do is find an install that is present.

    Leaves a working PATH lookup alone, so a normal Linux install (and a
    Windows one whose PATH box *was* ticked) behaves exactly as before.

    Returns:
        The resolved executable path, or None if Tesseract could not be found.
    """
    try:
        import pytesseract
    except ImportError:
        return None

    import shutil

    configured = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(configured):
        return configured

    if IS_WINDOWS:
        for candidate in _WINDOWS_TESSERACT_CANDIDATES:
            if os.path.isfile(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                return candidate

        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidate = os.path.join(local, "Programs", "Tesseract-OCR",
                                     "tesseract.exe")
            if os.path.isfile(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                return candidate

    return None


# --------------------------------------------------------------------------
# Engine binaries
# --------------------------------------------------------------------------

def engine_path(stem):
    """Resolve a Stockfish binary from ``Engines/`` for this platform.

    ``stem`` is the version-ish prefix, e.g. ``"stockfish17"``. Returns an
    absolute path resolved against the repo root rather than the CWD, so the
    bot no longer has to be launched from the project directory.

    An exact version match always wins. Failing that, on Windows, a *lone*
    Stockfish executable in ``Engines/`` serves every slot -- see
    ``_sole_windows_engine``.

    If nothing matches at all, the first (canonical) candidate is returned
    anyway so that the eventual launch failure names the file you are
    expected to supply, rather than something empty or from the wrong OS.
    """
    if IS_WINDOWS:
        candidates = [f"{stem}-windows.exe", f"{stem}.exe",
                      f"{stem}-x86-64-avx2.exe"]
    else:
        candidates = [f"{stem}-ubuntu", stem, f"{stem}-x86-64-avx2"]

    engines = REPO_ROOT / "Engines"
    for name in candidates:
        path = engines / name
        if path.is_file():
            return str(path)

    sole = _sole_windows_engine(engines)
    if sole is not None:
        return sole

    return str(engines / candidates[0])


def _sole_windows_engine(engines):
    """The single Windows Stockfish in ``Engines/``, if there is exactly one.

    The repo asks for two engines by name (``stockfish17`` for play,
    ``stockfish16`` for pondering), but that split is historical rather than
    a calibration requirement -- it came from a belief that concurrent engine
    processes needed distinct binaries, which they do not: ``popen_uci``
    spawns a fresh process from the same executable every call. So on Windows
    one binary is allowed to serve both slots.

    Deliberately narrow. It fires only when the version-specific lookup found
    nothing *and* the directory is unambiguous. Drop properly-named
    ``stockfish17-windows.exe``/``stockfish16-windows.exe`` in later and the
    exact matches win before this is ever consulted, so moving to per-version
    binaries needs no code change. Two unnamed engines and it declines to
    guess.

    Note the version actually in use is whatever is on disk: if that is not
    the Stockfish the repo's timing and strength tables were measured
    against, engine-derived behaviour will differ from those numbers and the
    parity harness is expected to disagree.
    """
    if not IS_WINDOWS:
        return None
    found = sorted(p for p in engines.glob("stockfish*.exe") if p.is_file())
    return str(found[0]) if len(found) == 1 else None


# --------------------------------------------------------------------------
# Subprocess groups
# --------------------------------------------------------------------------

def process_group_popen_kwargs():
    """Popen kwargs that put the child in its own killable process group."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def terminate_process_group(proc):
    """Kill ``proc`` and everything it spawned (workers, Stockfish children)."""
    if IS_WINDOWS:
        # taskkill /T walks the child tree. A CTRL_BREAK_EVENT would only
        # reach the direct child, leaving orphaned Stockfish processes behind.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        return
    import signal
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
