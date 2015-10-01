from __future__ import division

import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

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
x0 = ca.DMatrix([0, 0, 0, 5, 5, 10, 5, 0])
# Final cost of coordinate discrepancy
w_cl = 1e1
# Running cost on controls
R = 1e-1 * ca.diagcat([1, 0])
# Discretization step
dt = 0.1
model = Model(x0, (w_cl, R), dt)

# -------------------------- Create simulator ------------------------------ #
# Time horizon
l = 10
# Nominal controls for simulation
u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, l))
u_all[:, 'v'] = 2
u_all[:, 'phi'] = ca.pi/3
simulator = Simulator()


# ============================================================================
#                           Simulate a trajectory
# ============================================================================
# Get a single noise-free trajectory
x_all = simulator.simulate_trajectory(model, u_all)
Plotter.plot_trajectory(x_all)
Plotter.plot_trajectory_3D(x_all)


# ============================================================================
#                             Plan a trajectory
# ============================================================================
plan = Planner.create_plan(model, l)
x_all = plan.prefix['X']
Plotter.plot_trajectory(x_all)





















