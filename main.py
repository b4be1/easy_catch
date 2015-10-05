from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from model import Model
from simulator import Simulator
from planner import Planner
from plotter import Plotter

np.set_printoptions(suppress=True, precision=4)

__author__ = 'belousov'


# ============================================================================
#                              Initialization
# ============================================================================
# ----------------------------- Create model ------------------------------- #
# Initial state
m0 = ca.DMatrix([0, 0, 0, 5, 5, 10, 5, 0])
S0 = ca.DMatrix.eye(m0.size()) * 0.25
# Discretization step
dt = 0.1
# System noise matrix
M = ca.DMatrix.eye(m0.size()) * 1e-3
# Final cost of coordinate discrepancy
w_cl = 1e1
# Running cost on controls
R = 1e-1 * ca.diagcat([1, 0])
# Create model
model = Model((m0, S0), dt, M, (w_cl, R))

# ------------------------- Simulator parameters --------------------------- #
# Time horizon
l = 10
# Nominal controls for simulation
u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, l))
u_all[:, 'v'] = 2
u_all[:, 'phi'] = ca.pi/2


# ============================================================================
#                 Simulate trajectory and observations in 2D
# ============================================================================
x_all = Simulator.simulate_trajectory(model, u_all)
z_all = Simulator.simulate_observed_trajectory(model, x_all, u_all)
b_all = Simulator.filter_observed_trajectory(model, z_all, u_all)

# Plot 2D
_, ax = plt.subplots(figsize=(6, 6))
Plotter.plot_trajectory(ax, x_all, u_all)
Plotter.plot_observed_ball_trajectory(ax, z_all)
Plotter.plot_filtered_trajectory(ax, b_all)


# ============================================================================
#                   Plot trajectory and observations in 3D
# ============================================================================

# Plot 3D
fig_3D = plt.figure(figsize=(12, 8))
ax_3D = fig_3D.add_subplot(111, projection='3d')
Plotter.plot_trajectory_3D(ax_3D, x_all, u_all)


# ============================================================================
#                             Plan trajectory
# ============================================================================
plan = Planner.create_plan(model, l)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Plot 2D
_, ax = plt.subplots(figsize=(6, 6))
Plotter.plot_trajectory(ax, x_all, u_all)


# ============================================================================
#                         Model predictive control
# ============================================================================
# for k in range(l):
#     plan = Planner.create_plan(model, l)




plt.show()


























