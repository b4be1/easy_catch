from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import csv

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
# Initial condition
x_b0 = y_b0 = z_b0 = 0
vx_b0 = 10
vy_b0 = 7
vz_b0 = 15

x_c0 = 30
y_c0 = 2
vx_c0 = vy_c0 = 0
phi0 = ca.arctan2(y_b0-y_c0, x_b0-x_c0)  # direction towards the ball
if phi0 < 0:
    phi0 += 2 * ca.pi
psi0 = 0

# Initial mean
m0 = ca.DMatrix([x_b0, y_b0, z_b0, vx_b0, vy_b0, vz_b0,
                 x_c0, y_c0, vx_c0, vy_c0, phi0, psi0])
# Initial covariance
S0 = ca.diagcat([1, 1, 0, 1, 1, 1,
                 0.5, 0.5, 0.5, 0.5, 1e-2, 1e-2]) * 0.25
# Hypercovariance
L0 = ca.DMatrix.eye(m0.size()) * 1e-5
# Discretization step
dt = 0.1
# Number of Runge-Kutta integration intervals per time step
n_rk = 1
# Reaction time (in units of dt)
n_delay = 2
# System noise matrix
M = ca.DMatrix.eye(m0.size()) * 1e-3
M[-6:, -6:] = ca.DMatrix.eye(6) * 1e-5  # catcher's dynamics is less noisy
# Observation noise
N_min = 1e-2  # when looking directly at the ball
N_max = 1e1   # when the ball is 90 degrees from the gaze direction
# Final cost: w_cl * distance_between_ball_and_catcher
w_cl = 1e3
# Running cost on controls: u.T * R * u
R = 1e0 * ca.diagcat([1, 1e-1, 1e1, 1e-1])
# Final cost of uncertainty: w_Sl * tr(S)
w_Sl = 1e3
# Running cost of uncertainty: w_S * tr(S)
w_S = 1e2
# Control limits
F_c1, F_c2 = 7.5, 2.5
w_max = 2 * ca.pi
psi_max = 0.8 * ca.pi/2

# Model creation wrapper
def new_model():
    return Model((m0, S0, L0), dt, n_rk, n_delay, (M, N_min, N_max),
                 (w_cl, R, w_Sl, w_S), (F_c1, F_c2, w_max, psi_max))

# Create model
model = new_model()


# ============================================================================
#                             Plan trajectory
# ============================================================================
# Find optimal controls
plan, lam_x, lam_g = Planner.create_plan(model)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Simulate ebelief trajectory
eb_all = Simulator.simulate_eb_trajectory(model, u_all)

# Plot 2D
fig, ax = plt.subplots()
fig.tight_layout()
handles = Plotter.plot_plan(ax, eb_all)
ax.legend(handles=handles, loc='upper left')
ax.set_aspect('equal')

# Plot 3D
# fig_3D = plt.figure(figsize=(10, 10))
# ax_3D = fig_3D.add_subplot(111, projection='3d')
# Plotter.plot_trajectory_3D(ax_3D, x_all)


# ============================================================================
#                           Belief space planning
# ============================================================================
# Find optimal controls
plan = Planner.create_belief_plan(model, warm_start=True,
                                  x0=plan, lam_x0=lam_x, lam_g0=lam_g)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Simulate ebelief trajectory
eb_all = Simulator.simulate_eb_trajectory(model, u_all)

# Plot 2D
fig, ax = plt.subplots()
fig.tight_layout()
handles = Plotter.plot_plan(ax, eb_all)
ax.legend(handles=handles, loc='upper left')
ax.set_aspect('equal')

# Plot 3D
# fig_3D = plt.figure(figsize=(10, 10))
# ax_3D = fig_3D.add_subplot(111, projection='3d')
# Plotter.plot_trajectory_3D(ax_3D, x_all)

# ============================================================================
#                   Simulate trajectory and observations
# ============================================================================
# Nominal controls for simulation
# u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, 15))
# u_all[:, 'v'] = 5

# Initial state is drawn from N(m0, S0)
# model.init_x0()

# Simulate
x_all = Simulator.simulate_trajectory(model, u_all)
z_all = Simulator.simulate_observed_trajectory(model, x_all)
b_all = Simulator.filter_observed_trajectory(model, z_all, u_all)

# Plot 2D
# fig, ax = plt.subplots(figsize=(10, 10))
# fig.tight_layout()
# Plotter.plot_trajectory(ax, x_all)
# Plotter.plot_observed_ball_trajectory(ax, z_all)
# Plotter.plot_filtered_trajectory(ax, b_all)

# Plot 3D
# fig_3D = plt.figure(figsize=(10, 10))
# ax_3D = fig_3D.add_subplot(111, projection='3d')
# Plotter.plot_trajectory_3D(ax_3D, x_all)
#
# plt.show()


# ============================================================================
#                         Model predictive control
# ============================================================================
# ----------------------------- Simulation --------------------------------- #
# Create models for simulation and planning
model = new_model()
model_p = new_model()

# Run MPC
X_all, Z_all, B_all, EB_all = Simulator.mpc(model, model_p)

# Cast simulation results for ease of use
x_all = model.x.repeated(X_all)
z_all = model.z.repeated(Z_all)
b_all = model.b.repeated(B_all)

# ---------------------- Step-by-step plotting ----------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.tight_layout()
xlim = (-5, 35)
ylim = (-5, 35)
Plotter.plot_mpc(fig, axes, xlim, ylim,
                 model, X_all, Z_all, B_all, EB_all)


# -------------------------- Plot full simulation -------------------------- #
# Plot 2D
fig, ax = plt.subplots()
fig.tight_layout()
handles = Plotter.plot_trajectory(ax, x_all)
handles.extend(Plotter.plot_observed_ball_trajectory(ax, z_all))
handles.extend(Plotter.plot_filtered_trajectory(ax, b_all))
ax.legend(handles=handles, loc='upper left')
ax.set_aspect('equal')

# Plot 3D
# fig_3D = plt.figure(figsize=(10, 10))
# ax_3D = fig_3D.add_subplot(111, projection='3d')
# Plotter.plot_trajectory_3D(ax_3D, model.x.repeated(X_all))


# ------------------- Optic acceleration cancellation ---------------------- #
fig = Plotter.plot_heuristics(model, x_all)


# ============================================================================
#                  Save/load trajectory to/from a csv-file
# ============================================================================
def save_trajectory(x_all, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for k in range(len(x_all[:])):
            writer.writerow(list(x_all[k]))


def load_trajectory(model, filename):
    x_all = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            x_all.append(row)
    return model.x.repeated(ca.DMatrix(np.array(x_all, dtype=np.float64)).T)




















