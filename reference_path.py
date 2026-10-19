# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:33:11 2026

@author: Teja3
"""

import numpy as np

def straight_road(length=200, n=400):
    # A perfectly straight lane, 200 meters long
    x = np.linspace(0, length, n)
    y = np.zeros(n)       # y=0 means center of lane
    return x, y

def curved_road(length=200, n=400):
    # A gentle s-curve - good for testing
    x = np.linspace(0, length, n)
    y = 2.0 * np.sin(2 * np.pi * x / 100)
    return x, y

def get_reference_at(x_current, ref_x, ref_y):
    # Find the closest point on the path to current car position
    # Returns: desired y (lateral position) and desired heading
    idx = np.argmin(np.abs(ref_x - x_current))
    idx = np.clip(idx, 0, len(ref_x) - 2)
    
    y_ref = ref_y[idx]
    
    # Desired heading = angle of the path at this point
    dx = ref_x[idx+1] - ref_x[idx]
    dy = ref_y[idx+1] - ref_y[idx]
    psi_ref = np.arctan2(dy, dx)
    
    return y_ref, psi_ref