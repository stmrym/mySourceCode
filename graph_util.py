import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from multiprocessing import Pool

def _plot(kwargs):
    plot_data, save_name, cmap, vmin, vmax = kwargs.values()
    # Initialize plt        
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(plot_data, vmin=vmin, vmax=vmax, cmap=cmap)
    # ax.axis("off")
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')

    fig.colorbar(im, cax=cax)
    plt.tight_layout()
    fig.patch.set_alpha(0)
    plt.savefig(save_name)


def plot_heatmap(**kwargs):
    p = Pool(1)
    p.map(_plot, [kwargs])
    p.close()