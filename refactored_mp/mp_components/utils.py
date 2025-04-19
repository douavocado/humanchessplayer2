#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the multiprocessing chess client.
"""

import subprocess
import os
import datetime
import platform

from .constants import LOG_FILE

# Global log string buffer for the current session
LOG = ""

def is_capslock_on():
    """
    Checks if Caps Lock is currently active.
    Supports Linux (using xset) and potentially other systems if needed.
    Returns True if Caps Lock is on, False otherwise.
    """
    system = platform.system()
    if system == "Linux":
        try:
            # The original command checks a specific byte (65) which might be fragile.
            # A more robust check might involve parsing the output more carefully.
            # Sticking to the original logic for functional equivalence for now.
            output = subprocess.check_output('xset q | grep LED', shell=True, text=True)
            # Example output: "  LED mask:  00000002" (Caps Lock on) or "  LED mask:  00000000" (off)
            # The original check looked at byte 65, which corresponded to '1' or '0'.
            # Let's try a slightly more robust check based on the mask value.
            # This assumes the Caps Lock bit is the second bit (mask 0x2).
            if 'LED mask:' in output:
                 mask_hex = output.split('LED mask:')[1].split()[0]
                 mask_int = int(mask_hex, 16)
                 return (mask_int & 0x2) != 0 # Check if the Caps Lock bit is set
            # Fallback to original logic if parsing fails or format changes
            return output[65] == '1'
        except (subprocess.CalledProcessError, IndexError, ValueError) as e:
            print(f"Warning: Could not determine Caps Lock state on Linux: {e}")
            return False # Default to False if check fails
    elif system == "Windows":
        # Placeholder for Windows implementation if needed
        # import ctypes
        # return ctypes.WinDLL("User32.dll").GetKeyState(0x14) & 1 # VK_CAPITAL = 0x14
        print("Warning: Caps Lock check not implemented for Windows.")
        return False
    elif system == "Darwin": # macOS
        # Placeholder for macOS implementation if needed
        # Requires different approach, possibly involving IOKit or external tools
        print("Warning: Caps Lock check not implemented for macOS.")
        return False
    else:
        print(f"Warning: Caps Lock check not implemented for OS: {system}")
        return False


def append_log(message: str):
    """Appends a message to the global log buffer."""
    global LOG
    LOG += message


def write_log():
    """ Writes the accumulated log buffer to the log file and clears the buffer. """
    global LOG
    if not LOG: # Don't write if buffer is empty
        return
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
            log_file.write(LOG)
        LOG = "" # Clear the buffer after successful write
    except IOError as e:
        print(f"Error writing to log file {LOG_FILE}: {e}")
        # Optionally, decide whether to keep LOG content or discard on error