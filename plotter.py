import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Patch

import casadi as ca

__author__ = 'belousov'


class Plotter:
    # ========================================================================
    #                                  2D
    # ========================================================================
    # ---------------------------- Trajectory ------------------------------ #
    @classmethod
    def plot_trajectory(cls, ax, x_all, u_all):
        cls._plot_ball_trajectory('Ball trajectory', ax, x_all)
        cls._plot_catcher_trajectory('Catcher trajectory', ax, x_all)
        cls._plot_arrows('Catcher gaze', ax, x_all, u_all)
        ax.grid(True)

    @staticmethod
    def _plot_ball_trajectory(name, ax, x_all):
        return ax.plot(x_all[:, 'x_b'],
                       x_all[:, 'y_b'],
                       label=name, lw=0.8, alpha=0.8, color='g',
                       marker='o', markersize=4, fillstyle='none')

    @staticmethod
    def _plot_catcher_trajectory(name, ax, x_all):
        return ax.plot(x_all[:, 'x_c'],
                       x_all[:, 'y_c'],
                       label=name, lw=0.8, alpha=0.8, color='g',
                       marker='o', markersize=4, fillstyle='none')

    @staticmethod
    def _plot_arrows(name, ax, x_all, u_all):
        x = x_all[:-1, 'x_c']
        y = x_all[:-1, 'y_c']
        phi = u_all[:, 'phi']
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

    # ========================================================================
    #                               3D
    # ========================================================================
    @classmethod
    def plot_trajectory_3D(cls, ax, x_all, u_all):
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
