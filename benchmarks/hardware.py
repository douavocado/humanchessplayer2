"""Machine fingerprint.

A timing number is meaningless without knowing what produced it, and these
JSONs are meant to be compared across devices weeks apart. Everything here
is read from the standard library or an already-required dependency --
psutil is deliberately not used, since it is not in requirements.txt and a
benchmark that needs a new dependency installed is one that will not get
run on the machine you most want to measure.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import time


def _cpu_name():
    """Marketing CPU name. platform.processor() returns a family/stepping
    string on Windows ("AMD64 Family 23 ...") which is nearly useless for
    telling two machines apart, so prefer the OS's own record."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _total_ram_gb():
    if sys.platform == "win32":
        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return round(st.ullTotalPhys / 1024 ** 3, 1)
    else:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 1024 ** 2, 1)
        except OSError:
            pass
    return None


def _stockfish_identity(path):
    """The engine's own `uci` banner. Which Stockfish is on disk changes
    both speed and strength, and the repo ships different versions per
    platform (SF16 on Windows, SF17 on Linux), so a comparison that
    ignores this can attribute a version difference to hardware."""
    try:
        proc = subprocess.run([path], input="uci\nquit\n", capture_output=True,
                              text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("id name"):
            return line[len("id name"):].strip()
    return None


def cpu_reference_score():
    """A tiny fixed workload, so two machines stay comparable even when
    neither CPU name means anything to the reader.

    Reported as seconds (lower is faster). Single-threaded on purpose:
    Stockfish gets its own thread budget (STOCKFISH_THREADS) and torch its
    own, so a multi-core score would blur the per-core speed that actually
    sets latency on a mostly-serial per-move path.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.random((512, 512), dtype=np.float64)
    b = rng.random((512, 512), dtype=np.float64)
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        a @ b
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def describe(include_reference_score=True):
    import numpy as np
    import torch

    from common.constants import (
        PATH_TO_PONDER_STOCKFISH,
        PATH_TO_STOCKFISH,
        RESOLUTION_SCALE,
    )
    from common.search_constants import STOCKFISH_HASH_MB, STOCKFISH_THREADS

    info = {
        "cpu": _cpu_name(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "numpy": np.__version__,
        "stockfish_play": _stockfish_identity(PATH_TO_STOCKFISH),
        "stockfish_ponder": _stockfish_identity(PATH_TO_PONDER_STOCKFISH),
        "stockfish_threads": STOCKFISH_THREADS,
        "stockfish_hash_mb": STOCKFISH_HASH_MB,
        "resolution_scale": RESOLUTION_SCALE,
    }
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = None
    if include_reference_score:
        info["cpu_reference_secs"] = cpu_reference_score()
    return info
