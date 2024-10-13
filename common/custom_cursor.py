#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 01:36:44 2024

@author: james
"""

from humancursor import SystemCursor
import pyautogui

class CustomCursor(SystemCursor):
    def __init__(self):
        super().__init__()
    
    def fake_drag(self, from_point: list or tuple, to_point: list or tuple, duration: int or float or [float, float] or (float, float) = None, steady=False):
        """Drags from a certain point, and releases to another"""
        if isinstance(duration, (list, tuple)):
            first_duration, second_duration = duration
        elif isinstance(duration, (float, int)):
            first_duration = second_duration = duration / 2
        else:
            first_duration = second_duration = None

        self.move_to(from_point, duration=first_duration)
        pyautogui.mouseDown()
        self.move_to(to_point, duration=second_duration, steady=steady)
        self.move_to(from_point, duration=second_duration, steady=steady)
        pyautogui.mouseUp()