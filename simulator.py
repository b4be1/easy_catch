from __future__ import division

import casadi as ca

__author__ = 'belousov'


class Simulator:
    """Simulate a model"""

    def __init__(self, model, l, u_all):
        """
        :param model: Model instance
        :param N: Simulation horizon
        :param u_all: Nominal controls
        :return: None
        """
        # Model to be simulated
        self.model = model

        # Number of simulation steps
        self.l = l

        # Nominal trajectory for simulations
        self.u_all = u_all

    def draw_trajectory(self):
        xk = self.model.x0
        x_all = [xk]
        for k in range(self.l):
            [xk_next] = self.model.F([xk, self.u_all[k]])
            x_all.append(xk_next)
            xk = xk_next
        x_all = self.model.x.repeated(ca.horzcat(x_all))
        return x_all

















