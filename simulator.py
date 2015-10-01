from __future__ import division

import casadi as ca

__author__ = 'belousov'


class Simulator:
    """Simulate dynamics"""

    @staticmethod
    def simulate_trajectory(model, u_all):
        n = u_all.shape[1]
        xk = model.x0
        x_all = [xk]
        for k in range(n):
            [xk_next] = model.F([xk, u_all[k]])
            x_all.append(xk_next)
            xk = xk_next
        x_all = model.x.repeated(ca.horzcat(x_all))
        return x_all

















