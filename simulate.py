# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:48:49 2026

@author: Teja3
"""

import numpy as np
import time
from vehicle_model import BicycleModel
from lqr_controller import LQRController
from mpc_controller import MPCController
from reference_path import curved_road, get_reference_at

def run_simulation(controller, path_x, path_y, n_steps=300):
    car = BicycleModel ()
    state = np.array([0.0, 0.5, 0.0])  # start 0.5m off-center
    
    history_y = []
    history_ref = []
    history_err = []
    history_steer = []
    times    = []
    
    for i in range(n_steps):
        x, y, psi = state
        y_ref, psi_ref = get_reference_at(x, path_x, path_y)
        
        e_y = y - y_ref
        e_psi = psi - psi_ref
        
        t0 = time.perf_counter()
        steering = controller.compute(e_y, e_psi)
        times.append(time.perf_counter() - t0)
        
        state = car.step(state, steering)
        
        history_y.append(y)
        history_ref.append(y_ref)
        history_err.append(abs(e_y))
        history_steer.append(steering)
        
    rms_error = np.sqrt(np.mean(np.array(history_err)**2))
    avg_time = np.mean(times) * 1000  # ms
    return history_y, history_ref, history_err, history_steer, rms_error, avg_time

# Run both the controllers on same path
path_x, path_y = curved_road()
lqr = LQRController()
mpc = MPCController()

lqr_y, lqr_ref, lqr_err, lqr_steer, lqr_rms, lqr_t = run_simulation(lqr, path_x, path_y)
mpc_y, mpc_ref, mpc_err, mpc_steer, mpc_rms, mpc_t = run_simulation(mpc, path_x, path_y)

print(f"LQR - RMS error: {lqr_rms:.4f} m | avg compute: {lqr_t:.3f} ms")
print(f"MPC - RMS error: {mpc_rms:.4f} m | avg compute: {mpc_t:.3f} ms")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# plot 1 : path tracking
axes[0].plot(lqr_ref, 'k--', label='Reference', linewidth=1)
axes[0].plot(lqr_y, 'b-', label=f'LQR RMS={lqr_rms:.4f}m')
axes[0].plot(mpc_y, 'g-', label=f'MPC RMS={mpc_rms:.4f}m')
axes[0].set_ylabel('Lateral position (m)')
axes[0].set_title('Lane keeping: LQR vs MPC')
axes[0].legend()
axes[0].grid(True)

# plot 2 : tracking error over time
axes[1].plot(lqr_err, 'b-', label=f'LQR RMS={lqr_rms:.4f}m')
axes[1].plot(mpc_err, 'g-', label=f'MPC RMS={mpc_rms:.4f}m')
axes[1].set_ylabel('|Lateral error| (m)')
axes[1].legend()
axes[1].grid(True)

# plot 3 : steering commands
axes[2].plot(lqr_steer, 'b-', label='LQR steering')
axes[2].plot(mpc_steer, 'g-', label='MPC steering')
axes[2].axhline(0.52, color='r', linestyle='--', linewidth=0.8, label= '+/- 30 deg limit')
axes[2].axhline(-0.52, color='r', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Steering angle (rad)')
axes[2].set_xlabel('Time steps')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('results.png', dpi=150)
plt.show()
print("plot Saved as results.png")


    
