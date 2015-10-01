__author__ = 'belousov'


class Plotter:
    """Plot simulation results"""

    # ------------------------------------------------------------------------
    #                               2D
    # ------------------------------------------------------------------------
    def plot_ball_trajectory(self, name, ax, x_all):
        return ax.plot(x_all[:, 'x_b'],
                       x_all[:, 'y_b'],
                       label=name, lw=0.8, alpha=0.8, color='g',
                       marker='o', markersize=4, fillstyle='none')

    def plot_catcher_trajectory(self, name, ax, x_all):
        return ax.plot(x_all[:, 'x_c'],
                       x_all[:, 'y_c'],
                       label=name, lw=0.8, alpha=0.8, color='g',
                       marker='o', markersize=4, fillstyle='none')

    # ------------------------------------------------------------------------
    #                               3D
    # ------------------------------------------------------------------------
    def plot_ball_trajectory_3D(self, name, ax, x_all):
        return ax.scatter3D(x_all[:, 'x_b'],
                            x_all[:, 'y_b'],
                            x_all[:, 'z_b'],
                            label=name, color='g')

    def plot_catcher_trajectory_3D(self, name, ax, x_all):
        return ax.scatter3D(x_all[:, 'x_c'],
                            x_all[:, 'y_c'],
                            0,
                            label=name, color='g')
