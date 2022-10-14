
def _2d_samples(fig, samples, color):
    ax = fig.add_subplot(111)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticklabels([])
    ax.scatter(samples[:, 0], samples[:, 1], c=color, s=0.5, alpha=0.5)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])


def _3d_samples(fig, samples, color):
    ax = fig.add_subplot(111, projection='3d')
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=color, s=0.5, alpha=0.5)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])


def _4d_samples(fig, samples, color):
    # create all axes and turn the ticks and labels off
    axs = [[] for a in range(4)]
    for i in range(3):
        for j in range(i + 1):
            ax = fig.add_subplot(4, 4, i * 4 + j + 1)
            axs[i].append(ax)
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_ticklabels([])
    for j in range(4):
        ax = fig.add_subplot(4, 4, 13 + j, projection='3d', proj_type='ortho')
        axs[-1].append(ax)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])

    # 2d scatter plots for the first 3 rows
    for i, ax_list in enumerate(axs[:-1]):
        for j, ax in enumerate(ax_list):
            ax.scatter(samples[:, i + 1], samples[:, j], s=0.1, c=color, alpha=0.5)

    # 3d scatter plot for the last row
    for j in range(4):
        ax = axs[-1][j]
        indices = list(range(4))
        indices.remove(j)
        ax.scatter(samples[:, indices[0]], samples[:, indices[1]], samples[:, indices[2]], s=0.1, c=color)
