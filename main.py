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
# Initial mean
m0 = ca.DMatrix([0, 0, 0, 5, 5, 10, 5, 0, ca.pi/2])
# Initial covariance
S0 = ca.diagcat([1, 1, 1, 1, 1, 1, 0.5, 0.5, 1e-2]) * 0.25
# Hypercovariance
L0 = ca.DMatrix.eye(m0.size()) * 1e-5
# Discretization step
dt = 0.1
# Number of Runge-Kutta integration intervals per time step
n_rk = 1
# Reaction time (in units of dt)
n_delay = 3
# System noise matrix
M = ca.DMatrix.eye(m0.size()) * 1e-3
M[-3:, -3:] = ca.DMatrix.eye(3) * 1e-5  # catcher's dynamics is less noisy
# Final cost of coordinate discrepancy
w_cl = 1e1
# Running cost on controls
R = 1e-1 * ca.diagcat([1, 1])
# Create model
model = Model((m0, S0, L0), dt, n_rk, n_delay, M, (w_cl, R))


# ============================================================================
#                             Plan trajectory
# ============================================================================
# Find optimal controls
plan = Planner.create_plan(model)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Plot 2D
_, ax = plt.subplots(figsize=(6, 6))
Plotter.plot_trajectory(ax, x_all)

# Simulate ebelief propagation
eb_all = Simulator.simulate_eb_trajectory(model, u_all)

# Plot 2D
_, ax = plt.subplots(figsize=(12, 12))
Plotter.plot_plan(ax, eb_all)

# ============================================================================
#                   Simulate trajectory and observations
# ============================================================================
# Nominal controls for simulation
u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, 10))
u_all[:, 'v'] = 2
u_all[:, 'w'] = 0.5

# Initial state is drawn from N(m0, S0)
model.init_x0()

# Simulate
x_all = Simulator.simulate_trajectory(model, u_all)
z_all = Simulator.simulate_observed_trajectory(model, x_all)
b_all = Simulator.filter_observed_trajectory(model, z_all, u_all)

# Plot 2D
_, ax = plt.subplots(figsize=(6, 6))
Plotter.plot_trajectory(ax, x_all)
Plotter.plot_observed_ball_trajectory(ax, z_all)
Plotter.plot_filtered_trajectory(ax, b_all)

# Plot 3D
fig_3D = plt.figure(figsize=(12, 8))
ax_3D = fig_3D.add_subplot(111, projection='3d')
Plotter.plot_trajectory_3D(ax_3D, x_all)

plt.show()


# ============================================================================
#                         Model predictive control
# ============================================================================
# ----------------------------- Simulation --------------------------------- #
# Reset the model in case it was used before
model.set_initial_state(m0, m0, S0)
model.init_x0()

# Prepare a place to store simulation results
X_all = []
U_all = []
XP_all = []
B_all = []

# Simulator: simulate first n_delay time-steps with static controls
# u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, model.n_delay))
# u_all[:, 'phi'] = ca.pi/2
# x_all = Simulator.simulate_trajectory(model, x0, u_all)
# z_all = Simulator.simulate_observed_trajectory(model, x_all, u_all)
# b_all = Simulator.filter_observed_trajectory(model, z_all, u_all)


# Iterate until the ball hits the ground
while model.n != 0:
    # Planner: plan for n time steps
    plan = Planner.create_plan(model)
    xp_all = plan.prefix['X']
    u_all = plan.prefix['U']

    # Simulator: execute the first action
    x_all = Simulator.simulate_trajectory(model, [u_all[0]])
    z_all = Simulator.simulate_observed_trajectory(model, x_all)
    b_all = Simulator.filter_observed_trajectory(model, z_all, [u_all[0]])

    # Save simulation results
    X_all.append(x_all)
    U_all.append(u_all)
    XP_all.append(xp_all)
    B_all.append(b_all)

    # Advance time
    model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])

# ------------------------------- Plotting --------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Appearance
axes[0].set_title("Model predictive control, simulation")
axes[1].set_title("Model predictive control, plans")
for ax in axes:
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.grid(True)
    ax.set_aspect('equal')

# Plot
for k, _ in enumerate(X_all):
    # Planning
    Plotter.plot_trajectory(axes[1], XP_all[k])
    fig.canvas.draw()

    # Simulation
    Plotter.plot_trajectory(axes[0], X_all[k])
    Plotter.plot_filtered_trajectory(axes[0], B_all[k])
    fig.canvas.draw()
    plt.waitforbuttonpress()
















