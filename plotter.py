from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Patch, Ellipse

import numpy as np
import casadi as ca

__author__ = 'belousov'


class Plotter:
    # ========================================================================
    #                                  2D
    # ========================================================================
    # -------------------------- Helper methods ---------------------------- #
    @staticmethod
    def _create_ellipse(mu, cov):
        if len(mu) != 2 and cov.shape != (2, 2):
            raise TypeError('Arguments should be 2D')

        s = 6 # 6 -> 95%; 9.21 -> 99%
        w, v = np.linalg.eigh(cov)
        alpha = np.rad2deg(np.arctan2(v[1, 1], v[1, 0]))
        width = 2 * np.sqrt(s * w[1])
        height = 2 * np.sqrt(s * w[0])

        # Create the ellipse
        return Ellipse(mu, width, height, alpha,
                       fill=True, color='y', alpha=0.1)

    @staticmethod
    def _plot_arrows(name, ax, x, y, phi):
        x_vec = ca.cos(phi)
        y_vec = ca.sin(phi)
        ax.quiver(x, y, x_vec, y_vec,
                  units='xy', angles='xy', scale=1, width=0.08,
                  headwidth=4, headlength=6, headaxislength=5,
                  color='r', alpha=0.8, lw=0.1)
        return [Patch(color='red', label=name)]

    @staticmethod
    def _plot_arrows_3D(name, ax, x, y, phi, psi):
        x = ca.veccat(x)
        y = ca.veccat(y)
        z = ca.DMatrix.zeros(x.size())
        phi = ca.veccat(phi)
        psi = ca.veccat(psi)
        x_vec = ca.cos(psi) * ca.cos(phi)
        y_vec = ca.cos(psi) * ca.sin(phi)
        z_vec = ca.sin(psi)
        ax.quiver(x + x_vec, y + y_vec, z + z_vec,
                  x_vec, y_vec, z_vec,
                  color='r', alpha=0.8)
        return [Patch(color='red', label=name)]

    # ---------------------------- Trajectory ------------------------------ #
    @classmethod
    def plot_trajectory(cls, ax, x_all):
        cls._plot_trajectory('Ball trajectory', ax, x_all, ('x_b', 'y_b'))
        cls._plot_trajectory('Catcher trajectory', ax, x_all, ('x_c', 'y_c'))
        cls._plot_arrows('Catcher gaze', ax,
                         x_all[:, 'x_c'], x_all[:, 'y_c'], x_all[:, 'phi'])
        ax.grid(True)

    @staticmethod
    def _plot_trajectory(name, ax, x_all, (xl, yl)):
        x = x_all[:, xl]
        y = x_all[:, yl]
        return ax.plot(x, y, label=name, lw=0.8, alpha=0.8, color='g',
                       marker='o', markersize=4, fillstyle='none')

    # --------------------------- Observations ----------------------------- #
    @classmethod
    def plot_observed_ball_trajectory(cls, ax, z_all):
        x = z_all[:, 'x_b']
        y = z_all[:, 'y_b']
        return [ax.scatter(x, y, label='Observed ball trajectory',
                           c='m', marker='+', s=60)]

    # ------------------------ Filtered trajectory ------------------------- #
    @classmethod
    def plot_filtered_trajectory(cls, ax, b_all):
        # Plot line
        cls._plot_filtered_ball_mean('Filtered ball trajectory', ax, b_all)

        # Plot ellipses
        cls._plot_filtered_ball_cov('Filtered ball covariance', ax, b_all)

    @staticmethod
    def _plot_filtered_ball_mean(name, ax, b_all):
        x = b_all[:, 'm', 'x_b']
        y = b_all[:, 'm', 'y_b']
        return ax.plot(x, y, label=name, marker='.', color='m',
                       lw=0.8, alpha=0.9)

    @classmethod
    def _plot_filtered_ball_cov(cls, name, ax, b_all):
        for k in range(b_all.shape[1]):
            e = cls._create_ellipse(b_all[k, 'm', ['x_b', 'y_b']],
                                b_all[k, 'S', ['x_b', 'y_b'], ['x_b', 'y_b']])
            e.set_color('c')
            ax.add_patch(e)
        return [Patch(color='cyan', alpha=0.1, label=name)]

    # ------------------------ Planned trajectory -------------------------- #
    @classmethod
    def plot_plan(cls, ax, eb_all):
        """Complete plan"""
        cls._plot_plan(ax, eb_all, ('x_b', 'y_b'))
        cls._plot_plan(ax, eb_all, ('x_c', 'y_c'))
        cls._plot_arrows('Catcher gaze', ax,
                         eb_all[:, 'm', 'x_c'],
                         eb_all[:, 'm', 'y_c'],
                         eb_all[:, 'm', 'phi'])
        # Appearance
        ax.grid(True)

    @classmethod
    def _plot_plan(cls, ax, eb_all, (xl, yl)):
        """Plan for one object (ball or catcher)"""
        [plan_m] = cls._plot_plan_m('Plan', ax,
                                    eb_all[:, 'm', xl],
                                    eb_all[:, 'm', yl])
        [plan_S] = cls._plot_plan_S('Posterior', ax,
                                    eb_all[:, 'm', [xl, yl]],
                                    eb_all[:, 'S', [xl, yl], [xl, yl]])
        [plan_L] = cls._plot_plan_L('Prior', ax,
                                    eb_all[:, 'm', [xl, yl]],
                                    eb_all[:, 'L', [xl, yl], [xl, yl]])
        [plan_SL] = cls._plot_plan_SL('Prior + posterior', ax,
                                      eb_all[:, 'm', [xl, yl]],
                                      eb_all[:, 'S', [xl, yl], [xl, yl]],
                                      eb_all[:, 'L', [xl, yl], [xl, yl]])
        return [plan_m, plan_S, plan_L, plan_SL]

    @staticmethod
    def _plot_plan_m(name, ax, x, y):
        """Planned mean"""
        return ax.plot(x, y, label=name, lw=0.7,
                       alpha=0.9, marker='.', color='b')

    @classmethod
    def _plot_plan_S(cls, name, ax, mus, covs):
        """Planned posterior"""
        for k in range(len(mus)):
            e = cls._create_ellipse(mus[k], covs[k])
            e.set_fill(False)
            e.set_color('r')
            e.set_alpha(0.4)
            ax.add_patch(e)
        return [Patch(color='red', alpha=0.4, label=name)]

    @classmethod
    def _plot_plan_L(cls, name, ax, mus, covs):
        """Planned prior"""
        for k in range(len(mus)):
            e = cls._create_ellipse(mus[k], covs[k])
            ax.add_patch(e)
        return [Patch(color='yellow', alpha=0.1, label=name)]

    @classmethod
    def _plot_plan_SL(cls, name, ax, mus, covs, lcovs):
        """Planned prior + posterior"""
        for i in range(len(mus)):
            e = cls._create_ellipse(mus[i], covs[i]+lcovs[i])
            e.set_fill(False)
            e.set_color('g')
            e.set_alpha(0.1)
            ax.add_patch(e)
        return [Patch(color='green', alpha=0.1, label=name)]


    # ========================================================================
    #                               3D
    # ========================================================================
    @classmethod
    def plot_trajectory_3D(cls, ax, x_all):
        cls._plot_ball_trajectory_3D('Ball trajectory 3D', ax, x_all)
        cls._plot_catcher_trajectory_3D('Catcher trajectory 3D', ax, x_all)
        cls._plot_arrows_3D('Catcher gaze', ax,
                         x_all[:, 'x_c'], x_all[:, 'y_c'],
                         x_all[:, 'phi'], x_all[:, 'psi'])

    @staticmethod
    def _plot_ball_trajectory_3D(name, ax, x_all):
        return ax.scatter3D(x_all[:, 'x_b'],
                            x_all[:, 'y_b'],
                            x_all[:, 'z_b'],
                            label=name, color='g')

    @staticmethod
    def _plot_catcher_trajectory_3D(name, ax, x_all):
        return ax.scatter3D(x_all[:, 'x_c'],
                            x_all[:, 'y_c'],
                            0,
                            label=name, color='g')
