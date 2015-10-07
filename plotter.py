from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Patch, Ellipse

import numpy as np
import casadi as ca

__author__ = 'belousov'


class Plotter:
    # ========================================================================
    #                                  2D
    # ========================================================================
    # ---------------------------- Trajectory ------------------------------ #
    @classmethod
    def plot_trajectory(cls, ax, x_all):
        cls._plot_ball_trajectory('Ball trajectory', ax, x_all)
        cls._plot_catcher_trajectory('Catcher trajectory', ax, x_all)
        cls._plot_arrows('Catcher gaze', ax, x_all)
        ax.grid(True)

    @staticmethod
    def _plot_ball_trajectory(name, ax, x_all):
        x = x_all[:, 'x_b']
        y = x_all[:, 'y_b']
        return ax.plot(x, y, label=name, lw=0.8, alpha=0.8, color='g',
                       marker='o', markersize=4, fillstyle='none')

    @staticmethod
    def _plot_catcher_trajectory(name, ax, x_all):
        x = x_all[:, 'x_c']
        y = x_all[:, 'y_c']
        return ax.plot(x, y, label=name, lw=0.8, alpha=0.8, color='g',
                       marker='o', markersize=4, fillstyle='none')

    @staticmethod
    def _plot_arrows(name, ax, x_all):
        x = x_all[:, 'x_c']
        y = x_all[:, 'y_c']
        phi = x_all[:, 'phi']
        x_vec = ca.cos(phi)
        y_vec = ca.sin(phi)
        ax.quiver(x, y, x_vec, y_vec,
                  units='xy', angles='xy', scale=2, headwidth=4,
                  color='r', lw=0.1)
        return [Patch(color='red', label=name)]

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

    # ========================================================================
    #                               3D
    # ========================================================================
    @classmethod
    def plot_trajectory_3D(cls, ax, x_all):
        cls._plot_ball_trajectory_3D('Ball trajectory 3D', ax, x_all)
        cls._plot_catcher_trajectory_3D('Catcher trajectory 3D', ax, x_all)

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
