#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 01:36:44 2024

@author: james
"""

from humancursor import SystemCursor
from humancursor.utilities.human_curve_generator import HumanizeMouseTrajectory
from humancursor.utilities.calculate_and_randomize import generate_random_curve_parameters
import pyautogui
import random

class CustomCursor(SystemCursor):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def move_to(point: list or tuple, duration: int or float = None, human_curve=None, steady=False):
        from_point = pyautogui.position()
        
        if abs(from_point[0]-point[0]) < 2:
            point[0] += 5
        if abs(from_point[1] - point[1]) < 2:
            point[1] += 5
        
        if not human_curve:
            (
                offset_boundary_x,
                offset_boundary_y,
                knots_count,
                distortion_mean,
                distortion_st_dev,
                distortion_frequency,
                tween,
                target_points,
            ) = generate_random_curve_parameters(
                pyautogui, from_point, point
            )
            if steady:
                offset_boundary_x, offset_boundary_y = 10, 10
                distortion_mean, distortion_st_dev, distortion_frequency = 1.2, 1.2, 1
            human_curve = HumanizeMouseTrajectory(
                from_point,
                point,
                offset_boundary_x=offset_boundary_x,
                offset_boundary_y=offset_boundary_y,
                knots_count=knots_count,
                distortion_mean=distortion_mean,
                distortion_st_dev=distortion_st_dev,
                distortion_frequency=distortion_frequency,
                tween=tween,
                target_points=target_points,
            )

        if duration is None:
            duration = random.uniform(0.5, 2.0)
        pyautogui.PAUSE = duration / len(human_curve.points)
        for pnt in human_curve.points:
            pyautogui.moveTo(pnt)
        pyautogui.moveTo(point)

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