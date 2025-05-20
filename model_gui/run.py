#!/usr/bin/env python3
"""
Launcher script for the Chess MoveScorer GUI
"""
import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import chess
        import PIL
        import cairosvg
        import torch
        return True, "All dependencies found"
    except ImportError as e:
        return False, str(e)

def install_dependencies():
    """Attempt to install missing dependencies"""
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_file):
        return False, "Requirements file not found"
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        return True, "Dependencies installed successfully"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to install dependencies: {str(e)}"

def check_model_weights():
    """Check if model weights are available"""
    weights_paths = [
        "../models/model_weights/piece_selector_opening_weights.pth",
        "../models/model_weights/piece_to_opening_weights.pth",
        "../models/model_weights/piece_selector_midgame_weights.pth",
        "../models/model_weights/piece_to_midgame_weights.pth",
        "../models/model_weights/piece_selector_endgame_weights.pth",
        "../models/model_weights/piece_to_endgame_weights.pth",
        "../models/model_weights/piece_selector_defensive_tactics_weights.pth",
        "../models/model_weights/piece_to_defensive_tactics_weights.pth",
    ]
    
    missing = []
    for path in weights_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if not os.path.exists(full_path):
            missing.append(path)
    
    if missing:
        return False, f"Missing model weights: {', '.join(missing)}"
    
    return True, "Model weights found"

def main():
    """Main launcher function"""
    # First check dependencies
    deps_ok, deps_msg = check_dependencies()
    
    if not deps_ok:
        # Try to install dependencies if missing
        print(f"Missing dependencies: {deps_msg}")
        print("Attempting to install dependencies...")
        
        install_ok, install_msg = install_dependencies()
        if not install_ok:
            print(f"Error: {install_msg}")
            if tk._default_root is None:
                tk.Tk().withdraw()  # Create a root window but hide it
            messagebox.showerror("Error", f"Failed to install dependencies: {install_msg}")
            return 1
        
        print(install_msg)
    
    # Check model weights
    weights_ok, weights_msg = check_model_weights()
    if not weights_ok:
        print(f"Error: {weights_msg}")
        if tk._default_root is None:
            tk.Tk().withdraw()
        messagebox.showerror("Error", f"{weights_msg}\n\nPlease ensure model weights are in the correct location.")
        return 1
    
    print(weights_msg)
    
    # Launch the application
    try:
        from app import main
        main()
        return 0
    except Exception as e:
        print(f"Error launching application: {e}")
        if tk._default_root is None:
            tk.Tk().withdraw()
        messagebox.showerror("Error", f"Failed to launch application: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 