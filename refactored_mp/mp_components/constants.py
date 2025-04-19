#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants used across the multiprocessing chess client application.
"""

import os
import datetime

# --- Configuration Constants ---

# Maximum number of recent FENs to store
FEN_NO_CAP = 8

# Time gap between full screen scrapes (in seconds)
SCRAPE_EVERY = 0.5

# Base delay added to engine calculation time for making a move (accounts for scraping, updates, etc.)
MOVE_DELAY = 0.25

# Delay after a drag-move for the piece to visually snap before next action (in seconds)
DRAG_MOVE_DELAY = 0.07

# Delay after a click-move for the piece to visually snap before next action (in seconds)
CLICK_MOVE_DELAY = 0.03

# --- Engine/Gameplay Settings ---

# Corresponds to engine's playing strength/depth (adjust as needed)
# Higher value generally means stronger play but more calculation time.
DIFFICULTY = 5 # Example value, adjust based on original script's intent or engine capabilities

# Influences how quickly the bot appears to react/move. Lower is faster.
QUICKNESS = 1.0 # Example value, adjust based on original script's intent

# Influences the speed of simulated mouse movements. Lower is faster.
MOUSE_QUICKNESS = 100 # Example value, adjust based on original script's intent

# --- Logging ---

LOG_DIRECTORY = os.path.join(os.getcwd(), "Client_logs")
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)
LOG_FILE = os.path.join(LOG_DIRECTORY, str(datetime.datetime.now()).replace(" ", "").replace(":", "_") + '.txt')

# --- Sound Files ---
# Ensure these files exist in the root directory or provide correct paths
NEW_GAME_SOUND = "new_game_found.mp3"
ALERT_SOUND = "alert.mp3"

# --- Error File Directory ---
ERROR_FILE_DIRECTORY = os.path.join(os.getcwd(), "Error_files")
if not os.path.exists(ERROR_FILE_DIRECTORY):
    os.makedirs(ERROR_FILE_DIRECTORY)