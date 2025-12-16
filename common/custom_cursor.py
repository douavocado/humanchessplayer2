#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 01:36:44 2024

@author: james

Custom cursor with lightweight human-like Bezier curve mouse movement.
Replaces humancursor library for better performance.
"""

import pyautogui
import random
import math
import time

def _bezier_point(t, p0, p1, p2, p3):
    """Calculate point on cubic Bezier curve at parameter t."""
    u = 1 - t
    return (
        u*u*u * p0[0] + 3*u*u*t * p1[0] + 3*u*t*t * p2[0] + t*t*t * p3[0],
        u*u*u * p0[1] + 3*u*u*t * p1[1] + 3*u*t*t * p2[1] + t*t*t * p3[1]
    )

def _ease_out_quad(t):
    """Easing function: slow down towards end (more human-like)."""
    return t * (2 - t)

def _generate_human_curve(start, end, num_points=15, curve_spread=0.3):
    """
    Generate a human-like curved path using cubic Bezier curves.
    
    Args:
        start: (x, y) start position
        end: (x, y) end position
        num_points: Number of points in the path (fewer = faster)
        curve_spread: How much the curve deviates (0.0-1.0)
    
    Returns:
        List of (x, y) points along the curve
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.sqrt(dx*dx + dy*dy)
    
    # For very short distances, use direct path with slight curve
    if distance < 50:
        curve_spread *= 0.3
    
    # Calculate perpendicular direction for curve offset
    if distance > 0:
        perp_x = -dy / distance
        perp_y = dx / distance
    else:
        perp_x, perp_y = 0, 1
    
    # Random curve direction (left or right of straight line)
    curve_direction = random.choice([-1, 1])
    
    # Control points for cubic Bezier
    # Offset them perpendicular to the line for natural curve
    offset1 = distance * curve_spread * random.uniform(0.3, 0.7) * curve_direction
    offset2 = distance * curve_spread * random.uniform(0.2, 0.5) * curve_direction
    
    # First control point: ~1/3 along the line, offset perpendicular
    ctrl1 = (
        start[0] + dx * 0.3 + perp_x * offset1 + random.uniform(-5, 5),
        start[1] + dy * 0.3 + perp_y * offset1 + random.uniform(-5, 5)
    )
    
    # Second control point: ~2/3 along the line, offset perpendicular  
    ctrl2 = (
        start[0] + dx * 0.7 + perp_x * offset2 + random.uniform(-3, 3),
        start[1] + dy * 0.7 + perp_y * offset2 + random.uniform(-3, 3)
    )
    
    # Generate points along the curve with easing
    points = []
    for i in range(num_points):
        # Use easing for more natural speed (slow at start/end)
        t_linear = i / (num_points - 1) if num_points > 1 else 1
        t = _ease_out_quad(t_linear)
        
        point = _bezier_point(t, start, ctrl1, ctrl2, end)
        
        # Add tiny random jitter for more natural feel
        jitter = max(1, distance * 0.005)
        point = (
            point[0] + random.uniform(-jitter, jitter),
            point[1] + random.uniform(-jitter, jitter)
        )
        points.append(point)
    
    # Ensure we end exactly at the target
    points[-1] = end
    
    return points


def _generate_minimal_curve(start, end):
    """
    Generate a minimal curved path with just 3-4 waypoints.
    Uses pyautogui's built-in movement between points for speed.
    
    Returns:
        List of (x, y, duration) tuples - waypoints with timing
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.sqrt(dx*dx + dy*dy)
    
    # Very short distance - just go direct
    if distance < 100:
        return [(end[0], end[1])]
    
    # Calculate perpendicular for curve offset
    if distance > 0:
        perp_x = -dy / distance
        perp_y = dx / distance
    else:
        return [(end[0], end[1])]
    
    # Random curve direction and magnitude
    curve_dir = random.choice([-1, 1])
    curve_amount = distance * random.uniform(0.08, 0.18) * curve_dir
    
    # Single midpoint with curve offset
    mid = (
        start[0] + dx * 0.5 + perp_x * curve_amount + random.uniform(-3, 3),
        start[1] + dy * 0.5 + perp_y * curve_amount + random.uniform(-3, 3)
    )
    
    return [mid, end]


class CustomCursor:
    """
    Lightweight cursor with human-like movement.
    Uses custom Bezier curves instead of humancursor library for speed.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def quick_move_to(point: list or tuple, duration: float = 0.15, resolution_scale: float = 1.0):
        """
        Fast human-like mouse movement using minimal waypoints.
        Uses pyautogui's built-in smooth movement between waypoints for speed.
        
        Args:
            point: Target (x, y) coordinates
            duration: Movement duration in seconds
            resolution_scale: Scale factor (ignored in minimal mode)
        """
        start = pyautogui.position()
        end = (int(point[0]), int(point[1]))
        
        # Generate minimal curved path (1-2 waypoints)
        waypoints = _generate_minimal_curve(start, end)
        
        # Disable pyautogui's built-in pause
        old_pause = pyautogui.PAUSE
        pyautogui.PAUSE = 0
        
        try:
            if len(waypoints) == 1:
                # Direct movement with slight curve built into pyautogui
                pyautogui.moveTo(end[0], end[1], duration=duration, tween=pyautogui.easeOutQuad)
            else:
                # Move through waypoint then to end
                mid = waypoints[0]
                # Split duration: 40% to midpoint, 60% to end
                pyautogui.moveTo(int(mid[0]), int(mid[1]), duration=duration * 0.4, tween=pyautogui.easeOutQuad)
                pyautogui.moveTo(end[0], end[1], duration=duration * 0.6, tween=pyautogui.easeOutQuad)
        finally:
            pyautogui.PAUSE = old_pause
    
    @staticmethod
    def move_to(point: list or tuple, duration: float = None, human_curve=None, steady=False):
        """
        Human-like mouse movement (legacy interface).
        Uses minimal waypoints for speed.
        
        Args:
            point: Target (x, y) coordinates
            duration: Movement duration (if None, uses random 0.15-0.3s)
            human_curve: Ignored (for compatibility)
            steady: If True, uses less curve deviation
        """
        if duration is None:
            duration = random.uniform(0.15, 0.3)
        
        start = pyautogui.position()
        end = (int(point[0]), int(point[1]))
        
        # Use minimal curve for speed
        waypoints = _generate_minimal_curve(start, end)
        
        old_pause = pyautogui.PAUSE
        pyautogui.PAUSE = 0
        
        try:
            if len(waypoints) == 1:
                pyautogui.moveTo(end[0], end[1], duration=duration, tween=pyautogui.easeOutQuad)
            else:
                mid = waypoints[0]
                pyautogui.moveTo(int(mid[0]), int(mid[1]), duration=duration * 0.4, tween=pyautogui.easeOutQuad)
                pyautogui.moveTo(end[0], end[1], duration=duration * 0.6, tween=pyautogui.easeOutQuad)
        finally:
            pyautogui.PAUSE = old_pause
    
    def drag_and_drop(self, from_point: list or tuple, to_point: list or tuple, 
                      duration=None):
        """
        Drag from one point to another with human-like movement.
        
        Args:
            from_point: Starting (x, y) coordinates
            to_point: Ending (x, y) coordinates
            duration: Total duration or [from_duration, to_duration]
        """
        if isinstance(duration, (list, tuple)):
            from_duration, to_duration = duration
        elif isinstance(duration, (float, int)):
            from_duration = to_duration = duration / 2
        else:
            from_duration = to_duration = 0.15
        
        self.move_to(from_point, duration=from_duration)
        pyautogui.mouseDown()
        self.move_to(to_point, duration=to_duration)
        pyautogui.mouseUp()
    
    def fake_drag(self, from_point: list or tuple, to_point: list or tuple, 
                  duration=None, steady=False):
        """Drags from a certain point, and releases to another (legacy)."""
        if isinstance(duration, (list, tuple)):
            first_duration, second_duration = duration
        elif isinstance(duration, (float, int)):
            first_duration = second_duration = duration / 2
        else:
            first_duration = second_duration = 0.2

        self.move_to(from_point, duration=first_duration)
        pyautogui.mouseDown()
        self.move_to(to_point, duration=second_duration, steady=steady)
        self.move_to(from_point, duration=second_duration, steady=steady)
        pyautogui.mouseUp()