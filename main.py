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
vy_b0 = 10
vz_b0 = 15

x_c0 = 25
y_c0 = 15
vx_c0 = vy_c0 = 0
phi0 = ca.arctan2(y_b0-y_c0, x_b0-x_c0)  # direction towards the ball
if phi0 < 0:
    phi0 += 2 * ca.pi
psi0 = 0

# Initial mean
m0 = ca.DMatrix([x_b0, y_b0, z_b0, vx_b0, vy_b0, vz_b0,
                 x_c0, y_c0, vx_c0, vy_c0, phi0, psi0])
# Initial covariance
S0 = ca.diagcat([1, 1, 1, 1, 1, 1,
                 0.5, 0.5, 0.5, 0.5, 1e-2, 1e-2]) * 0.25
# Hypercovariance
L0 = ca.DMatrix.eye(m0.size()) * 1e-5
# Discretization step
dt = 0.2
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
w_cl = 1e2
# Running cost on facing the ball: w_c * face_the_ball
w_c = 0
# Running cost on controls: u.T * R * u
R = 1e-1 * ca.diagcat([1, 1, 1, 1e-2])
# Final cost of uncertainty: w_Sl * tr(S)
w_Sl = 1e2
# Running cost of uncertainty: w_S * tr(S)
w_S = 1e1
# Control limits
F_c1, F_c2 = 7.5, 2.5
w_max = 4 * ca.pi
psi_max = 0.8 * ca.pi/2

# Model creation wrapper
def new_model():
    return Model((m0, S0, L0), dt, n_rk, n_delay, (M, N_min, N_max),
                 (w_cl, w_c, R, w_Sl, w_S), (F_c1, F_c2, w_max, psi_max))

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
_, ax = plt.subplots(figsize=(12, 12))
Plotter.plot_plan(ax, eb_all)

# Plot 3D
fig_3D = plt.figure(figsize=(12, 8))
ax_3D = fig_3D.add_subplot(111, projection='3d')
Plotter.plot_trajectory_3D(ax_3D, x_all)


# ============================================================================
#                           Belief space planning
# ============================================================================
# Find optimal controls
plan = Planner.create_belief_plan(model, warm_start=False,
                                  x0=plan, lam_x0=lam_x, lam_g0=lam_g)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Simulate ebelief trajectory
eb_all = Simulator.simulate_eb_trajectory(model, u_all)

# Plot 2D
_, ax = plt.subplots(figsize=(12, 12))
Plotter.plot_plan(ax, eb_all)

# Plot 3D
fig_3D = plt.figure(figsize=(12, 8))
ax_3D = fig_3D.add_subplot(111, projection='3d')
Plotter.plot_trajectory_3D(ax_3D, x_all)

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
_, ax = plt.subplots(figsize=(10, 10))
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
Z_all = z_all.cast()
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
    belief_plan = Planner.create_belief_plan(model_p, plan)
    u_all = model_p.u.repeated(ca.horzcat(belief_plan['U']))

    # Simulator: simulate ebelief trajectory for plotting
    eb_all_tail = Simulator.simulate_eb_trajectory(model_p, u_all)

    # Simulator: execute the first action
    x_all = Simulator.simulate_trajectory(model, [u_all[0]])
    z_all = Simulator.simulate_observed_trajectory(model, x_all)
    b_all = Simulator.filter_observed_trajectory(model, z_all, [u_all[0]])

    # Save simulation results
    X_all.appendColumns(x_all.cast()[:, 1:])  # 0'th state is already included
    Z_all.appendColumns(z_all.cast()[:, 1:])
    U_all.appendColumns(u_all.cast()[:, 0])   # save only the first control
    B_all.appendColumns(b_all.cast()[:, 1:])
    EB_all.append([eb_all_head, eb_all_tail])

    # Advance time
    model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])
    model_p.set_initial_state(model_p.b(B_all[:, k+1])['m'],
                              model_p.b(B_all[:, k+1])['m'],
                              model_p.b(B_all[:, k+1])['S'])
    k += 1


# ----------------------------- Plotting ----------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Appearance
axes[0].set_title("Model predictive control, simulation")
axes[1].set_title("Model predictive control, planning")
for ax in axes:
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 40)
    ax.grid(True)
    ax.set_aspect('equal')

# Plot the first piece
head = 0
x_piece = model.x.repeated(X_all[:, head:head+n_delay+1])
z_piece = model.z.repeated(Z_all[:, head:head+n_delay+1])
b_piece = model.b.repeated(B_all[:, head:head+n_delay+1])
Plotter.plot_trajectory(axes[0], x_piece)
Plotter.plot_observed_ball_trajectory(axes[0], z_piece)
Plotter.plot_filtered_trajectory(axes[0], b_piece)
fig.canvas.draw()

# Advance time
head += n_delay

# Plot the rest
for k, _ in enumerate(EB_all):
    # Clear old plan
    axes[1].clear()
    axes[1].set_title("Model predictive control, planning")
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 40)
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
    z_piece = model.z.repeated(Z_all[:, head:head+2])
    b_piece = model.b.repeated(B_all[:, head:head+2])
    plt.waitforbuttonpress()
    Plotter.plot_trajectory(axes[0], x_piece)
    Plotter.plot_observed_ball_trajectory(axes[0], z_piece)
    Plotter.plot_filtered_trajectory(axes[0], b_piece)
    fig.canvas.draw()

    # Advance time
    head += 1

x_all = model.x.repeated(X_all)
z_all = model.z.repeated(Z_all)
b_all = model.b.repeated(B_all)


# -------------------------- Plot full simulation -------------------------- #
# Plot 2D
_, ax = plt.subplots(figsize=(12, 12))
Plotter.plot_trajectory(ax, x_all)
Plotter.plot_observed_ball_trajectory(ax, z_all)
Plotter.plot_filtered_trajectory(ax, b_all)

# Plot 3D
fig_3D = plt.figure(figsize=(12, 8))
ax_3D = fig_3D.add_subplot(111, projection='3d')
Plotter.plot_trajectory_3D(ax_3D, model.x.repeated(X_all))


# ------------------- Optic Acceleration Cancellation ---------------------- #
n = len(x_all[:])
oac = []
for k in range(n):
    x_b = x_all[k, ca.veccat, ['x_b', 'y_b']]
    x_c = x_all[k, ca.veccat, ['x_c', 'y_c']]
    r_bc_xy = ca.norm_2(x_b - x_c)
    z_b = x_all[k, 'z_b']
    tan_phi = ca.arctan2(z_b, r_bc_xy)
    oac.append(tan_phi)

# Plot 2D
t_all = np.linspace(0, (n-1)*dt, n)
_, ax = plt.subplots(figsize=(8, 8))
ax.plot(t_all, oac)
ax.grid(True)


# ----------------------- Constant Bearing Angle --------------------------- #
cba = []
for k in range(n):
    x_b = x_all[k, ca.veccat, ['x_b', 'y_b']]
    x_c = x_all[k, ca.veccat, ['x_c', 'y_c']]
    r_cb = x_b - x_c
    r_cb_unit = r_cb / ca.norm_2(r_cb)
    cba.append(ca.arccos(r_cb_unit[0]))  # cos of the angle with x-axis

# Plot 2D
_, ax = plt.subplots(figsize=(8, 8))
ax.plot(t_all, cba)
ax.grid(True)


# ========================================================================
#                  Save/load trajectory to/from a csv-file
# ========================================================================
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





















