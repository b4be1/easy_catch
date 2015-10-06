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
# Initial mean
m0 = ca.DMatrix([0, 0, 0, 5, 5, 10, 5, 0])
# Initial covariance
S0 = ca.DMatrix.eye(m0.size()) * 0.25
# Discretization step
dt = 0.1
# Reaction time (in units of dt)
n_delay = 3
# System noise matrix
M = ca.DMatrix.eye(m0.size()) * 1e-3
# Final cost of coordinate discrepancy
w_cl = 1e1
# Running cost on controls
R = 1e-1 * ca.diagcat([1, 0])
# Create model
model = Model((m0, S0), dt, n_delay, M, (w_cl, R))

# ------------------------- Simulator parameters --------------------------- #
# Time horizon
n = 10
# Nominal controls for simulation
u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, n))
u_all[:, 'v'] = 2
u_all[:, 'phi'] = ca.pi/2


# ============================================================================
#                 Simulate trajectory and observations in 2D
# ============================================================================
# Initial state is drawn from the Gaussian distribution
x0 = Simulator.draw_initial_state(model)
x_all = Simulator.simulate_trajectory(model, x0, u_all)
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
plan = Planner.create_plan(model, n)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Plot 2D
_, ax = plt.subplots(figsize=(6, 6))
Plotter.plot_trajectory(ax, x_all, u_all)


# ============================================================================
#                         Model predictive control
# ============================================================================
# ----------------------------- Simulation --------------------------------- #
# Simulator: draw initial state from the Gaussian distribution
model.set_initial_condition(m0, S0)
x0 = Simulator.draw_initial_state(model)

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
while True:
    # Planner: estimate how many planning steps are required
    n = Planner.estimate_planning_horizon_length(model, dt)

    # Quit if horizon is zero
    if n == 0:
        break

    # Planner: plan for n time steps
    plan = Planner.create_plan(model, n)
    xp_all = plan.prefix['X']
    u_all = plan.prefix['U']

    # Simulator: execute the first action
    x_all = Simulator.simulate_trajectory(model, x0, [u_all[0]])
    z_all = Simulator.simulate_observed_trajectory(model, x_all, [u_all[0]])
    b_all = Simulator.filter_observed_trajectory(model, z_all, [u_all[0]])

    # Save simulation results
    X_all.append(x_all)
    U_all.append(u_all)
    XP_all.append(xp_all)
    B_all.append(b_all)

    # Advance time
    x0 = x_all[-1]
    model.set_initial_condition(b_all[-1, 'm'], b_all[-1, 'S'])

# ------------------------------- Plotting --------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Appearance
axes[0].set_title("Model predictive control, simulation")
axes[1].set_title("Model predictive control, plans")
for ax in axes:
    ax.grid(True)
    ax.set_aspect('equal')

# Plot
for k, _ in enumerate(X_all):
    # Simulation
    Plotter.plot_trajectory(axes[0], X_all[k], U_all[k])
    Plotter.plot_filtered_trajectory(axes[0], B_all[k])
    plt.waitforbuttonpress()
    fig.canvas.draw()

    # Planning
    Plotter.plot_trajectory(axes[1], XP_all[k], U_all[k])
    plt.waitforbuttonpress()
    fig.canvas.draw()




plt.show()























