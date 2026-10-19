# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:08:44 2026

@author: Teja3
"""

import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, v=10.0, L=2.9):
        # Linearised bicycle model: state=[e_y, e_psi], input=steering
        # A and B matrices come from linearising the bicycle equations
        self.A = np.array([[0, v],
                          [0, 0]])
        
        self.B = np.array([[0],
                          [v / L]])
        
        # Q: penalise lateral error heavily, heading error moderately
        self.Q = np.diag([10.0, 1.0])
        
        # R: small value = aggressive steering, large = gentle
        self.R = np.array([[0.1]])
        
        # solve Riccati equation to get optimal gain k
        p = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ p
        print(f"LQR gain K = {self.K}")
        
    def compute(self, e_y, e_psi):
        # error state vector
        e = np.array([e_y, e_psi])
        # steering = -k * error (negetive feedback)
        steering = -self.K @ e
        # clamp to physical limits: +/- 30 degrees in radians
        return np.clip(steering[0], -0.52, 0.52)
