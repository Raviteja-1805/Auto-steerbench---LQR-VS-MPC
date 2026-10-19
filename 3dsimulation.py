# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:27:42 2026

@author: Teja3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from  scipy.linalg import solve_continuous_are
from  scipy.optimize import minimize
from  mpl_toolkits.mplot3d import Axes3D


# __ Bicycle model
class BicycleModel:
    def __init__(self, L=2.9, dt=0.05):
        self.L = L  # wheelbase in meters
        self.dt = dt # time step in seconds

    def step(self, state, steering, v=10.0):
        x, y, psi = state
        dx = v * np.cos(psi)
        dy = v * np.sin(psi)
        dpsi = (v / self.L) * np.tan(steering)
        return np.array([x + dx*self.dt,
                         y + dy*self.dt,
                         psi + dpsi*self.dt])

# __Reference path (S-curve road)
def make_path(n=600):
    x = np.linspace(0, 300, n)
    y = 3.0 * np.sin(2* np.pi * x /120)
    return x, y

def get_ref(x_cur, ref_x, ref_y):
    idx = np.clip(np.argmin(np.abs(ref_x - x_cur)), 0, len(ref_x)-2)
    y_r =ref_y[idx]
    psi_r = np.arctan2(ref_y[idx+1] - ref_y[idx],
                       ref_x[idx+1] - ref_x[idx])
    return y_r, psi_r

# __LQR controller
class LQR:
    def __init__(self, V=10.0, L=2.9):
        A = np.array([[0, V], [0, 0]])
        B = np.array([[0], [V/L]])
        Q = np.diag([10.0, 1.0])
        R = np.array([[0.1]])
        p = solve_continuous_are(A, B, Q, R)
        self.k = np.linalg.inv(R) @ B.T @ p

    def compute(self, e_y, e_psi):
        return float(np.clip((-self.k @ np.array ([e_y, e_psi]))[0], -0.52, 0.52))

# __MPC controller
class MPC:
    def __init__(self, N=8, dt=0.05, V=10.0, L=2.9):
        self.N = N
        self.dt = dt
        self.V = V
        self.L = L
        self.q1 = 10.0  # lateral error weight
        self.q2 = 1.0   # heading error weight
        self.r1 = 0.1   # steering effort weight

    def _predict(self, e_y, e_psi, U):
        x1 = float(np.ravel(args[0])[0])
        x2 = float(np.ravel(args[1])[0])
        cost = 0.0
        for k in range(self.N):
            u = float(np.ravel(U)[k])
            # Cost at this step
            cost += self.q1 * x1*x1 + self.q2 * x2*x2 + self.r1 * u*u
            # propagate state forward one step
            x1_new = x1 + self.V * self.dt * x2
            x2_new = x2 + (self.V / self.L) * self.dt * u
            x1, x2 = x1_new, x2_new
        cost += self.q1 * x1*x1 + self.q2 * x2*x2
        return float(cost)

    def compute(self, e_y, e_psi):
        ey = float(np.ravel(np.array([e_y]))[0])
        eps = float(np.ravel(np.array([e_psi]))[0])

        def cost_fn(U):
            x1, x2 = ey, eps
            c = 0.0
            for k in range(self.N):
                u = float(U[k])
                c += self.q1*x1*x1 + self.q2*x2*x2 + self.r1*u*u
                x1 = x1 + self.V * self.dt * x2
                x2 = x2 + (self.V / self.L) * self.dt * u
            c += self.q1*x1*x1 + self.q2*x2*x2
            return c
        
        res = minimize(cost_fn,
            np.zeros(self.N),
            method='SLSQP',
            bounds=[(-0.52, 0.52)]*self.N,
            options={'maxiter' :30, 'ftol':1e-3})
        return float(res.x[0]) if res.success else 0.0

# __Run simulations
def run(ctrl, path_x, path_y, y_offset=0.0, n=300):
    car = BicycleModel()
    state = np.array([0.0, y_offset + 0.8, 0.0])
    xs, ys = [state[0]], [state[1]]
    for _ in range(n):
        y_r, psi_r = get_ref(state[0], path_x, path_y)
        y_r += y_offset
        e_y = state[1] - y_r
        e_psi = state[2] - psi_r
        s  = ctrl.compute(e_y, e_psi)
        state = car.step(state, s)
        xs.append(state[0])
        ys.append(state[1])
    return np.array(xs), np.array(ys)


print("Generating reference path..")
path_x, path_y = make_path()

print("Running LQR simulation..")
lqr = LQR()
lx, ly = run(lqr, path_x, path_y, y_offset=0.0)

print("Running MPC simulation (takes about 20 seconds)...")
mpc = MPC()
mx, my = run(mpc, path_x, path_y, y_offset=4.0)  #4m lane offset = second lane

# Reference center lines for both lanes
ref_xs = path_x[:len(lx)]
ref_lqr = np.array([path_y[np.clip(np.argmin(np.abs(path_x - xi)), 0, len(path_x)-1)]
                    for xi in ref_xs])
ref_mpc = ref_lqr + 4.0

# __3D fig setup
fig = plt.figure(figsize=(16, 9))

# __Set the background of the whole figure to navy
fig.patch.set_facecolor('navy')

ax = fig .add_subplot(111, projection='3d')

# set the bacground of the 3D plot area to navy as well
ax.set_facecolor('navy')

# __ Draw the road surface (grey strips)________________________________
# draw road lane markings along the path
for i in range(0, len (ref_xs), 5):
    xi = ref_xs[i]
    yi = ref_lqr[i]
    # Road edges (white lines on both sides)
    ax.plot([xi, xi], [yi - 3.5, yi + 7.5], [0, 0],
            color='grey', linewidth=0.4, alpha=0.3)

#__ Reference center lines (dashed)
ax.plot(ref_xs, ref_lqr, np.zeros_like(ref_xs),
        linestyle='dashed', color='dodgerblue',
        linewidth=1.0, alpha=0.5, label='LQR reference lane')


ax.plot(ref_xs, ref_mpc, np.zeros_like(ref_xs),
        linestyle='dashed', color='limegreen',
        linewidth=1.0, alpha=0.5, label='MPC reference lane')

#__Actual car trajectories (solid lines)
ax.plot(lx, ly, np.zeros_like(lx),
        color='deepskyblue', linewidth=2.5,
        label='LQR trajectory')

ax.plot(mx, my, np.zeros_like(mx),
        color='lime', linewidth=2.5,
        label='MPC trajectory')

#__Animated car markers
# LQR car = blue circle markers
lqr_car, = ax.plot([lx[0]], [ly[0]], [0.3],
                   marker='o', color='deepskyblue',
                   markersize=14, zorder=5)

#MPC car = green square marker
mpc_car, = ax.plot([mx[0]], [my[0]], [0.3],
                   marker='s', color='lime',
                   markersize=14, zorder=5)

#__ Axis labeling and styling
ax.set_xlabel('X position (m)', color='white', labelpad=10)
ax.set_ylabel('y position (m)', color='White', labelpad=10)
ax.set_zlabel('Z (m)',          color='white', labelpad=5)
ax.tick_params(colors='lightgrey')

# Make the pane walls transparent (just show the grid lines)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# pane edge color = dark navy
ax.xaxis.pane.set_edgecolor('midnightblue')
ax.yaxis.pane.set_edgecolor('midnightblue')
ax.zaxis.pane.set_edgecolor('midnightblue')

ax.set_zlim(0, 3)
ax.set_title('Lane Keeping 3D - LQR (blue) vs MPC (green)',
             color='white', fontsize=14, pad=20)

ax.legend(loc='upper left',
          facecolor='midnightblue',
          labelcolor='White',
          fontsize=10)

#__ RMS error results printed on the figure
lqr_rms = np.sqrt(np.mean((ly - ref_lqr[:len(ly)])**2))
mpc_rms = np.sqrt(np.mean((my - ref_mpc[:len(my)])**2))

fig.text(0.02, 0.15,
         f'LQR RMS error: {lqr_rms:.4f}m',
         color='deepskyblue', fontsize=11)
fig.text(0.02, 0.11,
         f'MPC RMS error: {mpc_rms:.4f}m',
         color='lime', fontsize=11)


#__ Animation: move the car markers along the track
def update(frame):
    i = min(frame * 3, len(lx) - 1) # move 3 steps per frame = faster animation

    #Move Lqr car
    lqr_car.set_data([lx[i]],[ly[i]])
    lqr_car.set_3d_properties([0.3])

    #Move MPC car
    j = min(i, len(mx) - 1)
    mpc_car.set_data([mx[j]], [my[j]])
    mpc_car.set_3d_properties([0.3])

    # slowly rotate the camera for a cinematic effect
    ax.view_init(elev=25, azim=30 + frame * 0.15)

    return lqr_car, mpc_car

ani = animation.FuncAnimation(fig, update,
                              frames=100,
                              interval=80,
                              blit=False)
plt.tight_layout()
plt.savefig('lanekeeping_3d_clear.png', dpi=150,
            facecolor='navy')
plt.show()

print(f"\nResults:")
print(f" LQR RMS error : {lqr_rms:.4f} m")
print(f" MPC RMS error : {mpc_rms:.4f} m")
print("saved: lanekeeping_3d_clear.png")
print("Saving animation as gif... ")
ani.save('lanekeeping_3d.gif', writer='pillow', fps=20, dpi=100)
print("Saved: lanekeeping_3d.gif")
plt.show()

    



