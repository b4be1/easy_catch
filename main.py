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
# Final cost of uncertainty
w_S = 1e1
# Running cost on controls
R = 1e-1 * ca.diagcat([1, 1])


# Model creation wrapper
def new_model():
    return Model((m0, S0, L0), dt, n_rk, n_delay, M, (w_cl, w_S, R))

# Create model
model = new_model()


# ============================================================================
#                             Plan trajectory
# ============================================================================
# Find optimal controls
plan = Planner.create_plan(model)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Simulate ebelief trajectory
eb_all = Simulator.simulate_eb_trajectory(model, u_all)

# Plot 2D
_, ax = plt.subplots(figsize=(12, 12))
Plotter.plot_plan(ax, eb_all)


# ============================================================================
#                             Belief planning
# ============================================================================
# Find optimal controls
plan = Planner.create_belief_plan(model, plan)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Simulate ebelief trajectory
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
# Create models for simulation and planning
model = new_model()
model_p = new_model()

# Simulator: simulate first n_delay time-steps with zero controls
u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, model.n_delay))
x_all = Simulator.simulate_trajectory(model, u_all)
z_all = Simulator.simulate_observed_trajectory(model, x_all)
b_all = Simulator.filter_observed_trajectory(model, z_all, u_all)

# Store simulation results
X_all = x_all.cast()
U_all = u_all.cast()
B_all = b_all.cast()

# Advance time
model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])

# Iterate until the ball hits the ground
EB_all = []
k = 0  # pointer to current catcher observation (= now - n_delay)
while model.n != 0:
    # Reaction delay compensation
    eb_all_head = Simulator.simulate_eb_trajectory(model_p,
                            model_p.u.repeated(U_all[:, k:k+model_p.n_delay]))
    model_p.set_initial_state(eb_all_head[-1, 'm'],
                              eb_all_head[-1, 'm'],
                              eb_all_head[-1, 'L'] + eb_all_head[-1, 'S'])
    if model_p.n == 0:
        break

    # Planner: plan for model_p.n time steps
    plan = Planner.create_plan(model_p)
    u_all = model_p.u.repeated(ca.horzcat(plan['U']))

    # Simulator: simulate ebelief trajectory for plotting
    eb_all_tail = Simulator.simulate_eb_trajectory(model_p, u_all)

    # Simulator: execute the first action
    x_all = Simulator.simulate_trajectory(model, [u_all[0]])
    z_all = Simulator.simulate_observed_trajectory(model, x_all)
    b_all = Simulator.filter_observed_trajectory(model, z_all, [u_all[0]])

    # Save simulation results
    X_all.appendColumns(x_all.cast()[:, 1:])  # 0'th state is already included
    U_all.appendColumns(u_all.cast()[:, 0])   # save only the first control
    B_all.appendColumns(b_all.cast()[:, 1:])  # 0'th belief is also included
    EB_all.append([eb_all_head, eb_all_tail])

    # Advance time
    model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])
    model_p.set_initial_state(model_p.b(B_all[:, k+1])['m'],
                              model_p.b(B_all[:, k+1])['m'],
                              model_p.b(B_all[:, k+1])['S'])
    k += 1

# ------------------------------- Plotting --------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Appearance
axes[0].set_title("Model predictive control, simulation")
axes[1].set_title("Model predictive control, planning")
for ax in axes:
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.grid(True)
    ax.set_aspect('equal')

# Plot the first piece
head = 0
x_piece = model.x.repeated(X_all[:, head:head+n_delay+1])
b_piece = model.b.repeated(B_all[:, head:head+n_delay+1])
Plotter.plot_trajectory(axes[0], x_piece)
Plotter.plot_filtered_trajectory(axes[0], b_piece)
fig.canvas.draw()

# Advance time
head += n_delay

# Plot the rest
for k, _ in enumerate(EB_all):
    # Clear old plan
    axes[1].clear()
    axes[1].set_title("Model predictive control, planning")
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    axes[1].grid(True)
    axes[1].set_aspect('equal')

    # Show new plan
    plt.waitforbuttonpress()
    Plotter.plot_plan(axes[1], EB_all[k][0])
    fig.canvas.draw()
    plt.waitforbuttonpress()
    Plotter.plot_plan(axes[1], EB_all[k][1])
    fig.canvas.draw()

    # Simulate one step
    x_piece = model.x.repeated(X_all[:, head:head+2])
    b_piece = model.b.repeated(B_all[:, head:head+2])
    plt.waitforbuttonpress()
    Plotter.plot_trajectory(axes[0], x_piece)
    Plotter.plot_filtered_trajectory(axes[0], b_piece)
    fig.canvas.draw()

    # Advance time
    head += 1
















