from __future__ import division

import numpy as np
from numpy.random import multivariate_normal as normal

import casadi as ca
import casadi.tools as cat

__author__ = 'belousov'


class Model:

    # Gravitational constant on the surface of the Earth
    g = 9.81

    def __init__(self, (m0, S0, L0), dt, n_rk, n_delay,
                 M, (w_cl, R, w_Sl, w_S), (v1, v2, w_max)):
        # Discretization time step, cannot be changed after creation
        self.dt = dt

        # Number of Runge-Kutta integration steps
        self.n_rk = n_rk

        # MPC reaction delay (in units of dt)
        self.n_delay = n_delay

        # State x
        self.x = cat.struct_symSX(['x_b', 'y_b', 'z_b',
                                   'vx_b', 'vy_b', 'vz_b',
                                   'x_c', 'y_c', 'phi'])
        # Control u
        self.u = cat.struct_symSX(['v', 'w', 'theta'])

        # Observation z
        self.z = cat.struct_symSX(['x_b', 'y_b', 'z_b', 'x_c', 'y_c', 'phi'])

        # Belief b = (mu, Sigma)
        self.b = cat.struct_symSX([
            cat.entry('m', struct=self.x),
            cat.entry('S', shapestruct=(self.x, self.x))
        ])

        # Extended belief eb = (mu, Sigma, L) for MPC and plotting
        self.eb = cat.struct_symSX([
            cat.entry('m', struct=self.x),
            cat.entry('S', shapestruct=(self.x, self.x)),
            cat.entry('L', shapestruct=(self.x, self.x))
        ])

        # Sizes
        self.nx = self.x.size
        self.nu = self.u.size
        self.nz = self.z.size

        # Initial state
        [self.x0,
         self.m0, self.S0, self.L0,
         self.b0, self.eb0] = self._state_init(m0, S0, L0)

        # Dynamics
        [self.f, self.F, self.Fj_x,
         self.h, self.hj_x] = self._dynamics_init()

        # Noise, system noise covariance matrix M = M(x, u)
        self.M = self._create_system_covariance_function(M)
        # State-dependent observation noise covariance matrix N = N(x, u)
        self.N = self._create_observation_covariance_function()

        # Noisy dynamics
        [self.Fn, self.hn] = self._noisy_dynamics_init()

        # Kalman filters
        [self.EKF, self.BF, self.EBF] = self._filters_init()

        # Cost functions: final and running
        self.cl = self._create_final_cost_function(w_cl)
        self.c = self._create_running_cost_function(R)

        # Cost functions: final and running uncertainty
        self.cSl = self._create_final_uncertainty_cost(w_Sl)
        self.cS = self._create_uncertainty_cost(w_S)

        # Control limits
        self.v1, self.v2, self.w_max = v1, v2, w_max

        # Number of simulation steps till the ball hits the ground
        self.n = self._estimate_simulation_duration()

    # ========================================================================
    #                           Initial condition
    # ========================================================================
    def set_initial_state(self, x0, m0, S0):
        self.x0 = self.x(x0)
        self.m0 = self.x(m0[:])
        self.S0 = self.x.squared(ca.densify(S0))
        self.b0['m'] = self.m0
        self.b0['S'] = self.S0
        self.eb0['m'] = self.m0
        self.eb0['S'] = self.S0
        self.n = self._estimate_simulation_duration()

    def init_x0(self):
        self.x0 = self._draw_initial_state(self.m0, self.S0)

    def _state_init(self, m0, S0, L0):
        m0 = self.x(m0[:])
        S0 = self.x.squared(ca.densify(S0))
        L0 = self.x.squared(ca.densify(L0))

        x0 = self._draw_initial_state(m0, S0)

        b0 = self.b()
        b0['m'] = m0
        b0['S'] = S0

        eb0 = self.eb()
        eb0['m'] = m0
        eb0['S'] = S0
        eb0['L'] = L0

        return [x0, m0, S0, L0, b0, eb0]

    def _draw_initial_state(self, m0, S0):
        m0_array = np.array(m0[...]).ravel()
        S0_array = S0.cast()
        return self.x(normal(m0_array, S0_array))

    def _estimate_simulation_duration(self):
        # 1. Unpack mean z-coordinate and z-velocity
        z_b0 = self.m0['z_b']
        vz_b0 = self.m0['vz_b']
        # 2. Use kinematic equation of the ball to find time
        T = (vz_b0 + ca.sqrt(vz_b0 ** 2 + 2 * self.g * z_b0)) / self.g
        # 3. Divide time by time-step duration
        return int(float(T) // self.dt)

    # ========================================================================
    #                                Dynamics
    # ========================================================================
    def _dynamics_init(self):
        # Continuous dynamics x_dot = f(x, u)
        f = self._create_continuous_dynamics()

        # Discrete dynamics x_next = F(x, u)
        F = self._discretize_dynamics(f)

        # Linearize discrete dynamics dx_next/dx
        Fj_x = F.jacobian('x')

        # Observation function z = h(x)
        h = self._create_observation_function()

        # Linearize observation function dz/dx
        hj_x = h.jacobian('x')

        return [f, F, Fj_x, h, hj_x]

    def _noisy_dynamics_init(self):
        # Discrete dynamics x_next = F(x, u) + sqrt(M(x, u)) * m, m ~ N(0, I)
        Fn = self._noisy_discrete_dynamics

        # Noisy observation function
        hn = self._noisy_observation_function

        return [Fn, hn]

    def _create_continuous_dynamics(self):
        # Unpack arguments
        [x_b, y_b, z_b, vx_b, vy_b, vz_b, x_c, y_c, phi] = self.x[...]
        [v, w, theta] = self.u[...]

        # Define the governing ordinary differential equation (ODE)
        rhs = cat.struct_SX(self.x)
        rhs['x_b'] = vx_b
        rhs['y_b'] = vy_b
        rhs['z_b'] = vz_b
        rhs['vx_b'] = 0
        rhs['vy_b'] = 0
        rhs['vz_b'] = -self.g
        rhs['x_c'] = v * ca.cos(phi + theta)
        rhs['y_c'] = v * ca.sin(phi + theta)
        rhs['phi'] = w

        op = {'input_scheme': ['x', 'u'],
              'output_scheme': ['x_dot']}
        return ca.SXFunction('Continuous dynamics',
                             [self.x, self.u], [rhs], op)

    def _discretize_dynamics(self, f):
        op = {'input_scheme': ['x', 'u'],
              'output_scheme': ['x_next']}
        # [x_dot] = f([self.x, self.u])
        # x_next = self.x + self.dt * x_dot
        # return ca.SXFunction('Discrete dynamics',
        #                      [self.x, self.u], [x_next], op)
        return ca.SXFunction('Discrete dynamics', [self.x, self.u],
                ca.simpleRK(f, self.n_rk)([self.x, self.u, self.dt]), op)

    def _create_observation_function(self):
        # Define the observation
        rhs = cat.struct_SX(self.z)
        for label in self.z.keys():
            rhs[label] = self.x[label]

        op = {'input_scheme': ['x'],
              'output_scheme': ['z']}
        return ca.SXFunction('Observation function',
                             [self.x], [rhs], op)


    # ========================================================================
    #                            Noisy dynamics
    # ========================================================================
    def _noisy_discrete_dynamics(self, (x, u)):
        [x_next] = self.F([x, u])
        [M] = self.M([x, u])
        x_next += normal(np.zeros(self.nx), M)
        return [x_next]

    def _noisy_observation_function(self, (x,)):
        [z] = self.h([x])
        [N] = self.N([x])
        z += normal(np.zeros(self.nz), N)
        return [z]


    # ========================================================================
    #                               Noise
    # ========================================================================
    # This function does nothing now: M(x, u) = M
    def _create_system_covariance_function(self, M):
        op = {'input_scheme': ['x', 'u'],
              'output_scheme': ['M']}
        return ca.SXFunction('System covariance',
                             [self.x, self.u], [M], op)

    def _create_observation_covariance_function(self):
        d = ca.veccat([ca.cos(self.x['phi']),
                       ca.sin(self.x['phi'])])
        r = ca.veccat([self.x['x_b'] - self.x['x_c'],
                       self.x['y_b'] - self.x['y_c']])
        r_cos_omega = ca.mul(d.T, r)
        cos_omega = r_cos_omega / (ca.norm_2(r) + 1e-6)

        # Look at the ball and be close to the ball
        N = self.z.squared(ca.SX.zeros(self.nz, self.nz))
        N['x_b', 'x_b'] = ca.mul(r.T, r) * (1 - cos_omega) + 1e-2
        N['y_b', 'y_b'] = ca.mul(r.T, r) * (1 - cos_omega) + 1e-2

        op = {'input_scheme': ['x'],
              'output_scheme': ['N']}
        return ca.SXFunction('Observation covariance',
                             [self.x], [N], op)


    # ========================================================================
    #                           Kalman filters
    # ========================================================================
    def _filters_init(self):
        # Extended Kalman Filter b_next = EKF(b, u, z)
        EKF = self._create_EKF()

        # Belief dynamics
        BF = self._create_BF()

        # Extended belief dynamics
        EBF = self._create_EBF()

        return [EKF, BF, EBF]

    def _create_EKF(self):
        """Extended Kalman filter"""
        b_next = cat.struct_SX(self.b)

        # Compute the mean
        [mu_bar] = self.F([self.b['m'], self.u])

        # Compute linearization
        [A, _] = self.Fj_x([self.b['m'], self.u])
        [C, _] = self.hj_x([mu_bar])

        # Get system and observation noises, as if the state was mu_bar
        [M] = self.M([self.b['m'], self.u])
        [N] = self.N([self.b['m']])

        # Predict the covariance
        S_bar = ca.mul([A, self.b['S'], A.T]) + M

        # Compute the inverse
        P = ca.mul([C, S_bar, C.T]) + N
        P_inv = ca.inv(P)

        # Kalman gain
        K = ca.mul([S_bar, C.T, P_inv])

        # Predict observation
        [z_bar] = self.h([mu_bar])

        # Update equations
        b_next['m'] = mu_bar + ca.mul([K, self.z - z_bar])
        b_next['S'] = ca.mul(ca.DMatrix.eye(self.nx) - ca.mul(K, C), S_bar)

        # (b, u, z) -> b_next
        op = {'input_scheme': ['b', 'u', 'z'],
              'output_scheme': ['b_next']}
        return ca.SXFunction('Extended Kalman filter',
                             [self.b, self.u, self.z], [b_next], op)

    def _create_BF(self):
        """Belief dynamics"""
        b_next = cat.struct_SX(self.b)

        # Compute the mean
        [mu_bar] = self.F([self.b['m'], self.u])

        # Compute linearization
        [A, _] = self.Fj_x([self.b['m'], self.u])
        [C, _] = self.hj_x([mu_bar])

        # Get system and observation noises, as if the state was mu_bar
        [M] = self.M([self.b['m'], self.u])
        [N] = self.N([self.b['m']])

        # Predict the covariance
        S_bar = ca.mul([A, self.b['S'], A.T]) + M

        # Compute the inverse
        P = ca.mul([C, S_bar, C.T]) + N
        P_inv = ca.inv(P)

        # Kalman gain
        K = ca.mul([S_bar, C.T, P_inv])

        # Update equations
        b_next['m'] = mu_bar
        b_next['S'] = ca.mul(ca.DMatrix.eye(self.nx) - ca.mul(K, C), S_bar)

        # (b, u) -> b_next
        op = {'input_scheme': ['b', 'u'],
              'output_scheme': ['b_next']}
        return ca.SXFunction('Belief dynamics',
                             [self.b, self.u], [b_next], op)

    def _create_EBF(self):
        """Extended belief dynamics"""
        eb_next = cat.struct_SX(self.eb)

        # Compute the mean
        [mu_bar] = self.F([self.eb['m'], self.u])

        # Compute linearization
        [A, _] = self.Fj_x([self.eb['m'], self.u])
        [C, _] = self.hj_x([mu_bar])

        # Get system and observation noises, as if the state was mu_bar
        [M] = self.M([self.eb['m'], self.u])
        [N] = self.N([self.eb['m']])

        # Predict the covariance
        S_bar = ca.mul([A, self.eb['S'], A.T]) + M

        # Compute the inverse
        P = ca.mul([C, S_bar, C.T]) + N
        P_inv = ca.inv(P)

        # Kalman gain
        K = ca.mul([S_bar, C.T, P_inv])

        # Update equations
        eb_next['m'] = mu_bar
        eb_next['S'] = ca.mul(ca.DMatrix.eye(self.nx) - ca.mul(K, C), S_bar)
        eb_next['L'] = ca.mul([A, self.eb['L'], A.T]) + ca.mul([K, C, S_bar])

        # (eb, u) -> eb_next
        op = {'input_scheme': ['eb', 'u'],
              'output_scheme': ['eb_next']}
        return ca.SXFunction('Extended belief dynamics',
                             [self.eb, self.u], [eb_next], op)

    # ========================================================================
    #                           Cost functions
    # ========================================================================
    def _create_final_cost_function(self, w_cl):
        # Final position
        x_b = self.x[ca.veccat, ['x_b', 'y_b']]
        x_c = self.x[ca.veccat, ['x_c', 'y_c']]
        dx_bc = x_b - x_c

        final_cost = 0.5 * ca.mul(dx_bc.T, dx_bc)
        op = {'input_scheme': ['x'],
              'output_scheme': ['cl']}
        return ca.SXFunction('Final cost', [self.x],
                             [w_cl * final_cost], op)

    def _create_running_cost_function(self, R):
        running_cost = 0.5 * ca.mul([self.u.cat.T, R * self.dt, self.u.cat])
        op = {'input_scheme': ['x', 'u'],
              'output_scheme': ['c']}
        return ca.SXFunction('Running cost', [self.x, self.u],
                             [running_cost], op)

    def _create_final_uncertainty_cost(self, w_Sl):
        final_uncertainty_cost = 0.5 * w_Sl * ca.trace(self.b['S'])
        op = {'input_scheme': ['b'],
              'output_scheme': ['cSl']}
        return ca.SXFunction('Final uncertainty cost', [self.b],
                             [final_uncertainty_cost], op)

    def _create_uncertainty_cost(self, w_S):
        running_uncertainty_cost = 0.5 * w_S * ca.trace(self.b['S'])
        op = {'input_scheme': ['b'],
              'output_scheme': ['cS']}
        return ca.SXFunction('Running uncertainty cost', [self.b],
                             [running_uncertainty_cost], op)

    # ========================================================================
    #                      Function called by Planner
    # ========================================================================
    def _set_control_limits(self, lbx, ubx):
        # v >= 0
        lbx['U', :, 'v'] = 0
        # w >= -w_max
        lbx['U', :, 'w'] = -self.w_max
        # w <= w_max
        ubx['U', :, 'w'] = self.w_max
        # theta >= -ca.pi
        lbx['U', :, 'theta'] = -ca.pi
        # theta <= ca.pi
        ubx['U', :, 'theta'] = ca.pi













