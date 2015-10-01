import matplotlib.pyplot as plt

__author__ = 'belousov'


class Plotter:
    """Plot simulation results"""

    # ========================================================================
    #                               2D
    # ========================================================================
    @staticmethod
    def plot_trajectory(x_all):
        fig, ax = plt.subplots(figsize=(6, 6))
        Plotter._plot_ball_trajectory('Ball trajectory', ax, x_all)
        Plotter._plot_catcher_trajectory('Catcher trajectory', ax, x_all)
        ax.grid(True)
        plt.show()

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

    # ========================================================================
    #                               3D
    # ========================================================================
    @staticmethod
    def plot_trajectory_3D(x_all):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        Plotter._plot_ball_trajectory_3D('Ball trajectory 3D', ax, x_all)
        Plotter._plot_catcher_trajectory_3D('Catcher trajectory 3D', ax, x_all)
        plt.show()

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
