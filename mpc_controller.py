# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:19:06 2026

@author: Teja3
"""

import numpy as np
from scipy.optimize import minimize

class MPCController:
    def __init__(self, N=10, dt=0.05, v=10.0, L=2.9):
        self.N = N   # prediction horizon
        self.dt = dt
        
        self.A = np.array([[1, v*dt],
                           [0, 1]])
        self.B = np.array([[0   ],
                           [v*dt/L]])
        self.Q = np.diag([10.0, 1.0])
        self.R = np.array([[0.1]])
        self.steer_max = 0.52
    def _cost(self, U, x0):
        cost = 0.0
        x = x0.copy()
        for k in range(self.N):
            u = np.array([U[k]])
            # running cost : error + effort
            cost += x @ self.Q @ x + u @ self.R @ u
            # propagate state forward one step
            x = self.A @ x + self.B @ u
        # terminal cost
        cost += x @ self.Q @ x
        return cost
        
    def compute(self, e_y, e_psi):
        x0 = np.array([e_y, e_psi])
        # Initial guess : all zeros (go straight)
        U0 = np.zeros(self.N)
        # steering angle bounds : -30deg to +30 deg for all N steps
        bounds = [(-self.steer_max, self.steer_max)] * self.N
        # solve the optimization
        result = minimize(
            self._cost, U0, args=(x0,),
            method='SLSQP',
            bounds=bounds ,
            options={'maxiter' : 50, 'ftol' : 1e-4}
        )
        # Apply only the first input (receding horizon)
        if result.success:
            return float(result.x[0])
        else:
            return 0.0
                           
