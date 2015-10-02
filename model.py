from __future__ import division

import casadi as ca
import casadi.tools as cat

__author__ = 'belousov'


class Model:

    # Gravitational constant on the surface of the Earth
    g = 9.81

    def __init__(self, x0, dt, M, (w_cl, R)):
        # State x
        self.x = cat.struct_symSX(['x_b', 'y_b', 'z_b',
                                   'vx_b', 'vy_b', 'vz_b',
                                   'x_c', 'y_c'])
        # Control u
        self.u = cat.struct_symSX(['v', 'phi'])

        # Observation z
        self.z = cat.struct_symSX(['x_b', 'y_b', 'z_b', 'x_c', 'y_c'])

        # Sizes
        self.nx = self.x.size
        self.nu = self.u.size
        self.nz = self.z.size

        # Initial state x0
        self.x0 = x0

        # Dynamics
        [self.f, self.F, self.Fj_x,
         self.h, self.hj_x] = self._dynamics_init(dt)

        # Cost functions: final and running
        self.cl = self._create_final_cost_function(w_cl)
        self.c = self._create_running_cost_function(R * dt)

        # System noise covariance matrix M
        self.M = M

        # State-dependent observation noise covariance matrix N = N(x)
        self.N = self._create_observation_covariance_function()

    def _dynamics_init(self, dt):
        # Continuous dynamics x_dot = f(x, u)
        f = self._create_continuous_dynamics()

        # Discrete dynamics x_next = F(x, u)
        F = self._discretize(f, dt)

        # Linearize discrete dynamics dx_next/dx
        Fj_x = F.jacobian('x')

        # Observation function z = h(x)
        h = self._create_observation_function()

        # Linearize observation function dz/dx
        hj_x = h.jacobian('x')

        return [f, F, Fj_x, h, hj_x]

    def _create_continuous_dynamics(self):
        # Unpack arguments
        [x_b, y_b, z_b, vx_b, vy_b, vz_b, x_c, y_c] = self.x[...]
        [v, phi] = self.u[...]

        # Define the ordinary differential equation (ODE)
        rhs = cat.struct_SX(self.x)
        rhs['x_b'] = vx_b
        rhs['y_b'] = vy_b
        rhs['z_b'] = vz_b
        rhs['vx_b'] = 0
        rhs['vy_b'] = 0
        rhs['vz_b'] = -self.g
        rhs['x_c'] = v * ca.cos(phi)
        rhs['y_c'] = v * ca.sin(phi)

        op = {'input_scheme': ['x', 'u'],
              'output_scheme': ['x_dot']}
        return ca.SXFunction('Continuous dynamics',
                             [self.x, self.u], [rhs], op)

    def _discretize(self, continuous_dynamics, dt):
        """Continuous dynamics is discretized with time step dt

        :param continuous_dynamics: f: [x, u] -> x_dot
        :param dt: time step
        :return: discrete_dynamics: F: [x, u] -> x_next
        """
        [x_dot] = continuous_dynamics([self.x, self.u])
        x_next = self.x + dt * x_dot

        op = {'input_scheme': ['x', 'u'],
              'output_scheme': ['x_next']}
        return ca.SXFunction('Discrete dynamics',
                             [self.x, self.u], [x_next], op)

    def _create_observation_function(self):
        # Define the observation
        rhs = cat.struct_SX(self.z)
        for label in self.z.keys():
            rhs[label] = self.x[label]

        op = {'input_scheme': ['x'],
              'output_scheme': ['z']}
        return ca.SXFunction('Observation function',
                             [self.x], [rhs], op)

    def _create_observation_covariance_function(self):
        d = ca.veccat([ca.cos(self.u['phi']),
                       ca.sin(self.u['phi'])])
        r = ca.veccat([self.x['x_b'] - self.x['x_c'],
                       self.x['y_b'] - self.x['y_c']])
        r_cos_omega = ca.mul(d.T, r)
        cos_omega = r_cos_omega / (ca.norm_2(r) + 1e-2)

        # Look at the ball and be close to the ball
        N = self.z.squared(ca.SX.zeros(self.nz, self.nz))
        N['x_b', 'x_b'] = ca.mul(r.T, r) * (1 - cos_omega) + 1e-2
        N['y_b', 'y_b'] = ca.mul(r.T, r) * (1 - cos_omega) + 1e-2

        op = {'input_scheme': ['x', 'u'],
              'output_scheme': ['N']}
        return ca.SXFunction('Observation covariance',
                             [self.x, self.u], [N], op)

    def _create_final_cost_function(self, w_cl):
        # Final position
        x_b = self.x[ca.veccat, ['x_b', 'y_b']]
        x_c = self.x[ca.veccat, ['x_c', 'y_c']]
        dx_bc = x_b - x_c

        final_cost = 0.5 * ca.mul(dx_bc.T, dx_bc)
        op = {'input_scheme': ['x'],
              'output_scheme': ['cl']}
        return ca.SXFunction('Final cost',
                             [self.x], [w_cl * final_cost], op)

    def _create_running_cost_function(self, R):
        cost = 0.5 * ca.mul([self.u.cat.T, R, self.u.cat])
        op = {'input_scheme': ['x', 'u'],
              'output_scheme': ['c']}
        return ca.SXFunction('Running cost', [self.x, self.u], [cost], op)

    @staticmethod
    def _set_control_limits(lbx, ubx):
        # v >= 0
        lbx['U', :, 'v'] = 0

        # -pi <= phi <= pi
        lbx['U', :, 'phi'] = -ca.pi; ubx['U', :, 'phi'] = ca.pi










