from __future__ import division

import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from model import Model
from simulator import Simulator
from plotter import Plotter

np.set_printoptions(suppress=True, precision=4)

__author__ = 'belousov'


# ============================================================================
#                              Initialization
# ============================================================================
# ----------------------------- Create model ------------------------------- #
# Initial state
x0 = ca.DMatrix([0, 0, 0, 5, 5, 10, 5, 0])
# Final cost of coordinate discrepancy
w_cl = 1e1
# Running cost on controls
R = ca.diagcat([1, 0])
# Discretization step
dt = 0.1
model = Model(x0, (w_cl, R), dt)

# -------------------------- Create simulator ------------------------------ #
# Time horizon
l = 20
# Nominal controls for simulation
u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, l))
u_all[:, 'v'] = 2
u_all[:, 'phi'] = ca.pi/3
simulator = Simulator(model, l, u_all)

# ---------------------------- Create plotter ------------------------------ #
plotter = Plotter()


# ============================================================================
#                           Simulate a trajectory
# ============================================================================
# Get a single noise-free trajectory
x_all = simulator.draw_trajectory()

# Plot trajectory
fig, ax = plt.subplots(figsize=(6, 6))
plotter.plot_ball_trajectory('Ball trajectory', ax, x_all)
plotter.plot_catcher_trajectory('Catcher trajectory', ax, x_all)
ax.grid(True)
plt.show()

# Plot trajectory 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plotter.plot_ball_trajectory_3D('Ball trajectory 3D', ax, x_all)
plotter.plot_catcher_trajectory_3D('Catcher trajectory 3D', ax, x_all)
ax.set_zlim(0, 6)
plt.show()

























