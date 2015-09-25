__author__ = 'belousov'


class Plotter:
    """Plot simulation results"""

    def plot_trajectory(self, name, ax, x_all):
        return ax.plot(x_all[:, 'x_b'],
                       x_all[:, 'y_b'],
                       label=name, lw=0.8, alpha=0.8, color='g',
                       marker='o', markersize=4, fillstyle='none')

    def plot_trajectory_3D(self, name, ax, x_all):
        return ax.scatter3D(x_all[:, 'x_b'],
                            x_all[:, 'y_b'],
                            x_all[:, 'z_b'],
                            label=name, c='b', lw='0.1')
