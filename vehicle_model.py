# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:02:00 2026

@author: Teja3
"""

import numpy as np

class BicycleModel:
    def __init__(self, L=2.9, dt=0.05):
        # L = wheelbase in meters (distance front to rear axle)
        # dt = time step in seconds (0.05 = 20 updates per second)
        self.L = L
        self.dt = dt
        
    def step(self, state, steering, v=10.0):
        # state = [x, y, psi]
        # steering = front wheel angle (radians)
        # v = constant speed (m/s), default 10 m/s = 36 km/h
        x, y, psi = state
        
        # The equations of motion (kinematic bicycle model)
        dx = v * np.cos(psi)
        dy = v * np.sin(psi)
        dpsi = (v / self.L) * np.tan(steering)
        
        # Euler integration: new_state = old_state + derivative * dt
        x_new = x + dx * self.dt
        y_new = y + dy * self.dt
        psi_new = psi + dpsi * self.dt
        
        return np.array([x_new, y_new, psi_new])