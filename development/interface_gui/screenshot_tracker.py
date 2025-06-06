#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screenshot Tracker GUI
A minimalist GUI for mouse tracking and screenshot capture functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from PIL import Image, ImageTk
import numpy as np
import os
from tkinter import filedialog

try:
    from fastgrab import screenshot
    FASTGRAB_AVAILABLE = True
except ImportError:
    FASTGRAB_AVAILABLE = False
    print("Warning: fastgrab not available. Install with: pip install fastgrab")

try:
    from pynput import mouse, keyboard
    from pynput.keyboard import Key, KeyCode
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Install with: pip install pynput")


class ScreenshotTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Screenshot Tracker")
        self.root.geometry("800x600")
        
        # Initialize screenshot capture
        if FASTGRAB_AVAILABLE:
            self.screen_capture = screenshot.Screenshot()
        else:
            self.screen_capture = None
            
        # State variables
        self.tracking_mode = False
        self.mouse_listener = None
        self.keyboard_listener = None
        self.tracking_key = 'space'  # Default tracking key
        self.screenshot_key = 'f1'   # Default screenshot key
        
        # Screenshot variables
        self.last_screenshot = None
        self.screenshot_image_label = None
        
        self.setup_ui()
        self.start_listeners()
        
    def setup_ui(self):
        """Set up the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Tracking section
        tracking_frame = ttk.LabelFrame(main_frame, text="Mouse Tracking", padding="5")
        tracking_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        tracking_frame.columnconfigure(2, weight=1)
        
        # Tracking mode toggle
        self.tracking_var = tk.BooleanVar()
        tracking_check = ttk.Checkbutton(
            tracking_frame, 
            text="Enable Tracking Mode", 
            variable=self.tracking_var,
            command=self.toggle_tracking_mode
        )
        tracking_check.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        # Tracking key configuration
        ttk.Label(tracking_frame, text="Tracking Key:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.tracking_key_var = tk.StringVar(value=self.tracking_key)
        tracking_key_entry = ttk.Entry(tracking_frame, textvariable=self.tracking_key_var, width=10)
        tracking_key_entry.grid(row=1, column=1, sticky=tk.W, padx=(0, 5))
        tracking_key_entry.bind('<Return>', self.update_tracking_key)
        
        ttk.Button(
            tracking_frame, 
            text="Update", 
            command=self.update_tracking_key
        ).grid(row=1, column=2, sticky=tk.W)
        
        # Screenshot section
        screenshot_frame = ttk.LabelFrame(main_frame, text="Screenshot Capture", padding="5")
        screenshot_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        screenshot_frame.columnconfigure(1, weight=1)
        
        # Screenshot key configuration
        ttk.Label(screenshot_frame, text="Screenshot Key:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.screenshot_key_var = tk.StringVar(value=self.screenshot_key)
        screenshot_key_entry = ttk.Entry(screenshot_frame, textvariable=self.screenshot_key_var, width=10)
        screenshot_key_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        screenshot_key_entry.bind('<Return>', self.update_screenshot_key)
        
        ttk.Button(
            screenshot_frame, 
            text="Update", 
            command=self.update_screenshot_key
        ).grid(row=0, column=2, sticky=tk.W)
        
        # BBox input
        bbox_frame = ttk.Frame(screenshot_frame)
        bbox_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        bbox_frame.columnconfigure(1, weight=1)
        bbox_frame.columnconfigure(3, weight=1)
        bbox_frame.columnconfigure(5, weight=1)
        bbox_frame.columnconfigure(7, weight=1)
        
        ttk.Label(bbox_frame, text="BBox (x, y, width, height):").grid(row=0, column=0, columnspan=8, sticky=tk.W, pady=(0, 2))
        
        ttk.Label(bbox_frame, text="X:").grid(row=1, column=0, sticky=tk.W)
        self.bbox_x_var = tk.StringVar(value="100")
        ttk.Entry(bbox_frame, textvariable=self.bbox_x_var, width=8).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(2, 5))
        
        ttk.Label(bbox_frame, text="Y:").grid(row=1, column=2, sticky=tk.W)
        self.bbox_y_var = tk.StringVar(value="100")
        ttk.Entry(bbox_frame, textvariable=self.bbox_y_var, width=8).grid(row=1, column=3, sticky=(tk.W, tk.E), padx=(2, 5))
        
        ttk.Label(bbox_frame, text="W:").grid(row=1, column=4, sticky=tk.W)
        self.bbox_w_var = tk.StringVar(value="200")
        ttk.Entry(bbox_frame, textvariable=self.bbox_w_var, width=8).grid(row=1, column=5, sticky=(tk.W, tk.E), padx=(2, 5))
        
        ttk.Label(bbox_frame, text="H:").grid(row=1, column=6, sticky=tk.W)
        self.bbox_h_var = tk.StringVar(value="150")
        ttk.Entry(bbox_frame, textvariable=self.bbox_h_var, width=8).grid(row=1, column=7, sticky=(tk.W, tk.E), padx=(2, 0))
        
        # Manual screenshot button
        ttk.Button(
            screenshot_frame, 
            text="Take Screenshot Now", 
            command=self.take_screenshot
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Text output for mouse coordinates
        self.output_text = scrolledtext.ScrolledText(output_frame, width=30, height=15)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Screenshot display area
        screenshot_display_frame = ttk.LabelFrame(main_frame, text="Screenshot Display", padding="5")
        screenshot_display_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        screenshot_display_frame.columnconfigure(0, weight=1)
        screenshot_display_frame.rowconfigure(0, weight=1)
        
        # Create a frame for the screenshot with scrollbars
        self.screenshot_canvas = tk.Canvas(screenshot_display_frame, bg='white')
        h_scrollbar = ttk.Scrollbar(screenshot_display_frame, orient=tk.HORIZONTAL, command=self.screenshot_canvas.xview)
        v_scrollbar = ttk.Scrollbar(screenshot_display_frame, orient=tk.VERTICAL, command=self.screenshot_canvas.yview)
        self.screenshot_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        self.screenshot_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Save screenshot controls
        save_frame = ttk.Frame(screenshot_display_frame)
        save_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Label(save_frame, text="Save as:").grid(row=0, column=0, sticky=tk.W)
        self.save_filename_var = tk.StringVar(value="screenshot.png")
        save_entry = ttk.Entry(save_frame, textvariable=self.save_filename_var, width=25)
        save_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(2, 5))
        ttk.Button(save_frame, text="Save Screenshot", command=self.save_screenshot).grid(row=0, column=2, sticky=tk.W)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Fastgrab: " + ("Available" if FASTGRAB_AVAILABLE else "Not Available"))
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def toggle_tracking_mode(self):
        """Toggle tracking mode on/off"""
        self.tracking_mode = self.tracking_var.get()
        status = "enabled" if self.tracking_mode else "disabled"
        self.log_output(f"Tracking mode {status}")
        self.update_status(f"Tracking mode {status}")
        
    def update_tracking_key(self, event=None):
        """Update the tracking key"""
        new_key = self.tracking_key_var.get().strip().lower()
        if new_key:
            self.tracking_key = new_key
            self.log_output(f"Tracking key updated to: {self.tracking_key}")
            self.restart_listeners()
        
    def update_screenshot_key(self, event=None):
        """Update the screenshot key"""
        new_key = self.screenshot_key_var.get().strip().lower()
        if new_key:
            self.screenshot_key = new_key
            self.log_output(f"Screenshot key updated to: {self.screenshot_key}")
            self.restart_listeners()
            
    def get_mouse_position(self):
        """Get current mouse position"""
        if PYNPUT_AVAILABLE:
            try:
                position = mouse.Controller().position
                return position
            except:
                return None
        return None
        
    def log_output(self, message):
        """Add message to output text area"""
        timestamp = time.strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)
        
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        
    def on_key_press(self, key):
        """Handle key press events"""
        try:
            # Convert key to string for comparison
            if hasattr(key, 'char') and key.char:
                key_str = key.char.lower()
            elif hasattr(key, 'name'):
                key_str = key.name.lower()
            else:
                key_str = str(key).replace('Key.', '').lower()
                
            # Handle tracking key
            if self.tracking_mode and key_str == self.tracking_key:
                position = self.get_mouse_position()
                if position:
                    self.log_output(f"Mouse position: ({position[0]}, {position[1]})")
                else:
                    self.log_output("Could not get mouse position")
                    
            # Handle screenshot key
            elif key_str == self.screenshot_key:
                threading.Thread(target=self.take_screenshot, daemon=True).start()
                
        except Exception as e:
            pass  # Ignore key handling errors
            
    def take_screenshot(self):
        """Take a screenshot of the specified bounding box"""
        if not self.screen_capture:
            self.log_output("Error: fastgrab not available")
            messagebox.showerror("Error", "fastgrab is not available. Please install it with: pip install fastgrab")
            return
            
        try:
            # Get bounding box values
            x = int(self.bbox_x_var.get())
            y = int(self.bbox_y_var.get())
            width = int(self.bbox_w_var.get())
            height = int(self.bbox_h_var.get())
            
            # Take screenshot
            self.update_status("Taking screenshot...")
            screenshot_array = self.screen_capture.capture((x, y, width, height)).copy()
            
            # Convert to PIL Image (RGB)
            if screenshot_array.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(screenshot_array[:, :, :3])  # Remove alpha channel
            else:  # RGB
                pil_image = Image.fromarray(screenshot_array[:, :, :3])
                
            self.display_screenshot(pil_image)
            self.log_output(f"Screenshot taken: ({x}, {y}, {width}, {height})")
            self.update_status("Screenshot captured successfully")
            
        except ValueError as e:
            error_msg = "Invalid bounding box values. Please enter valid integers."
            self.log_output(f"Error: {error_msg}")
            messagebox.showerror("Error", error_msg)
        except Exception as e:
            error_msg = f"Screenshot failed: {str(e)}"
            self.log_output(f"Error: {error_msg}")
            messagebox.showerror("Error", error_msg)
            self.update_status("Screenshot failed")
            
    def display_screenshot(self, pil_image):
        """Display the screenshot in the GUI, centred with an outline"""
        try:
            # Scale image if it's too large
            max_display_size = 400
            img_width, img_height = pil_image.size
            
            if img_width > max_display_size or img_height > max_display_size:
                # Calculate scaling factor
                scale = min(max_display_size / img_width, max_display_size / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_width, img_height = pil_image.size
            
            # Convert to PhotoImage for tkinter
            self.last_screenshot = ImageTk.PhotoImage(pil_image)
            self.last_pil_image = pil_image.copy()
            
            # Clear canvas and display image
            self.screenshot_canvas.delete("all")
            # Get canvas size
            canvas_width = int(self.screenshot_canvas.winfo_width())
            canvas_height = int(self.screenshot_canvas.winfo_height())
            if canvas_width == 1 and canvas_height == 1:
                # If not yet rendered, use default size
                canvas_width = 400
                canvas_height = 400
            # Calculate centre position
            x0 = (canvas_width - img_width) // 2
            y0 = (canvas_height - img_height) // 2
            # Draw image at centre
            self.screenshot_canvas.create_image(x0, y0, anchor=tk.NW, image=self.last_screenshot)
            # Draw outline rectangle
            self.screenshot_canvas.create_rectangle(
                x0, y0, x0 + img_width, y0 + img_height,
                outline="red", width=2
            )
            # Update scroll region
            self.screenshot_canvas.configure(scrollregion=self.screenshot_canvas.bbox("all"))
            
        except Exception as e:
            self.log_output(f"Error displaying screenshot: {str(e)}")
            
    def start_listeners(self):
        """Start keyboard and mouse listeners"""
        if not PYNPUT_AVAILABLE:
            self.log_output("Warning: pynput not available - hotkeys will not work")
            return
            
        try:
            # Start keyboard listener
            self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
            self.keyboard_listener.start()
            self.log_output("Keyboard listener started")
        except Exception as e:
            self.log_output(f"Failed to start keyboard listener: {str(e)}")
            
    def restart_listeners(self):
        """Restart the listeners (for key updates)"""
        self.stop_listeners()
        time.sleep(0.1)  # Small delay
        self.start_listeners()
        
    def stop_listeners(self):
        """Stop all listeners"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
            
    def on_closing(self):
        """Handle window closing"""
        self.stop_listeners()
        self.root.destroy()

    def save_screenshot(self):
        """Save the currently displayed screenshot to saved_screenshots/ with the specified filename"""
        # Get filename
        filename = self.save_filename_var.get().strip()
        if not filename:
            self.log_output("Please enter a filename.")
            return
        # Ensure .png extension
        if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
            filename += '.png'
        # Ensure directory exists
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_screenshots')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        # Save the image
        try:
            # Reconstruct PIL image from PhotoImage
            # Use the last PIL image if available
            if hasattr(self, 'last_pil_image') and self.last_pil_image is not None:
                pil_image = self.last_pil_image
            else:
                self.log_output("No PIL image available to save.")
                return
            pil_image.save(save_path)
            self.log_output(f"Screenshot saved as {save_path}")
            self.update_status(f"Screenshot saved: {filename}")
        except Exception as e:
            self.log_output(f"Failed to save screenshot: {str(e)}")


def main():
    """Main function to run the application"""
    # Check dependencies
    if not FASTGRAB_AVAILABLE:
        print("Warning: fastgrab not found. Install with: pip install fastgrab")
    if not PYNPUT_AVAILABLE:
        print("Warning: pynput not found. Install with: pip install pynput")
        
    root = tk.Tk()
    app = ScreenshotTracker(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main() 