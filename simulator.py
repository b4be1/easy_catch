from __future__ import division

import casadi as ca

__author__ = 'belousov'


class Simulator:

    # ========================================================================
    #                           True noisy trajectory
    # ========================================================================
    @staticmethod
    def simulate_trajectory(model, u_all):
        xk = model.x0.cat
        x_all = [xk]
        for uk in u_all[:]:
            [xk_next] = model.Fn([xk, uk])
            x_all.append(xk_next)
            xk = xk_next
        x_all = model.x.repeated(ca.horzcat(x_all))
        return x_all

    # ========================================================================
    #                              Observations
    # ========================================================================
    @staticmethod
    def simulate_observed_trajectory(model, x_all):
        z_all = []
        for xk in x_all[:]:
            [zk] = model.hn([xk])
            z_all.append(zk)
        z_all = model.z.repeated(ca.horzcat(z_all))
        return z_all

    # ========================================================================
    #                         Filtered observations
    # ========================================================================
    @staticmethod
    def filter_observed_trajectory(model, z_all, u_all):
        n = len(u_all[:])
        bk = model.b0
        b_all = [bk]
        for k in range(1, n+1):
            [bk_next] = model.EKF([bk, u_all[k-1], z_all[k]])
            b_all.append(bk_next)
            bk = bk_next
        b_all = model.b.repeated(ca.horzcat(b_all))
        return b_all

    # ========================================================================
    #                       Extended belief trajectory
    # ========================================================================
    @staticmethod
    def simulate_eb_trajectory(model, u_all):
        ebk = model.eb0
        eb_all = [ebk]
        for uk in u_all[:]:
            [ebk_next] = model.EBF([ebk, uk])
            eb_all.append(ebk_next)
            ebk = ebk_next
        eb_all = model.eb.repeated(ca.horzcat(eb_all))
        return eb_all
























