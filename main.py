from __future__ import division

import casadi as ca
import casadi.tools as cat

import numpy as np

np.set_printoptions(suppress=True, precision=4)

__author__ = 'belousov'


# %% =========================================================================
#                                Parameters
# ============================================================================

# Number of time intervals
N = 10


# %% =========================================================================
#                                  Model
# ============================================================================

# State
x = cat.struct_symSX(['T', 'x_b', 'y_b', 'z_b',
                      'vx_b', 'vy_b', 'vz_b',
                      'x_c', 'y_c'])

# Control
u = cat.struct_symSX(['v', 'phi'])


# %% =========================================================================
#                             Initial condition
# ============================================================================

# Initial state
x0 = ca.DMatrix([5, 0, 0, 0, 5, 5, 10, 5, 0])

# Nominal controls
U0 = u.repeated(ca.DMatrix.zeros(u.size, N))
U0[:,'v'] = 3


# %% =========================================================================
#                             Continuous dynamics
# ============================================================================

ode = cat.struct_SX(x)
ode['T'] = 0
ode['x_b'] = x['vx_b']
ode['y_b'] = x['vy_b']
ode['z_b'] = x['vz_b']







































