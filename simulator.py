from __future__ import division

import numpy as np
from numpy.random import multivariate_normal as normal

import casadi as ca

__author__ = 'belousov'


class Simulator:

    @staticmethod
    def simulate_trajectory(model, u_all):
        n = u_all.shape[1]
        xk = model.x0
        x_all = [xk]
        for k in range(n):
            [xk_next] = model.F([xk, u_all[k]])
            xk_next += normal(np.zeros(model.nx), model.M)
            x_all.append(xk_next)
            xk = xk_next
        x_all = model.x.repeated(ca.horzcat(x_all))
        return x_all

    @staticmethod
    def simulate_observed_trajectory(model, x_all, u_all):
        n = u_all.shape[1]
        z_all = model.h([x_all[0]])
        for k in range(1, n+1):
            [zk] = model.h([x_all[k]])
            [N] = model.N([x_all[k], u_all[k-1]])
            zk += normal(np.zeros(model.nz), N)
            z_all.append(zk)
        z_all = model.z.repeated(ca.horzcat(z_all))
        return z_all

    @staticmethod
    def filter_observed_trajectory(model, ??):














