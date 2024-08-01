import cv2
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from multiprocessing import Pool

def _plot(plot_data: np.ndarray, save_name: str, cmap: str, **kwargs):
    # Initialize plt        
    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots(dpi=100, tight_layout=True)
    im = ax.imshow(plot_data, cmap=cmap, **kwargs)
    # im = ax.imshow(plot_data, cmap=cmap)
    # ax.axis("off")
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad='15%')

    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.tight_layout()
    # fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.04)

def _plot_wrapper(kwargs):
    return _plot(**kwargs)

def plot_heatmap(**kwargs):
    p = Pool(1)
    p.map(_plot_wrapper, [kwargs])
    p.close()


def cv2_heatmap(image: np.ndarray, save_name: str, cmap: int = cv2.COLORMAP_JET) -> None:
    #######
    # image: [0,1] normalized ndarray
    #######
    image = (np.clip(image, 0, 1)*255).astype(np.uint8)
    image_heatmap = cv2.applyColorMap(image, colormap=cmap)
    cv2.imwrite(save_name, image_heatmap)

def cv2_alpha_heatmap(map: np.ndarray, image: np.ndarray, save_name: str, alpha: float, beta: float, cmap: int = cv2.COLORMAP_JET) -> None:
    #######
    # map: [0,1] normalized ndarray
    # image: [0,255] uint8 color image
    #######
    map = (np.clip(map, 0, 1)*255).astype(np.uint8)
    image_heatmap = cv2.applyColorMap(map, colormap=cmap).astype(np.float32)

    blended = image_heatmap * alpha + image.astype(np.float32) * (1 - alpha) + beta
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.imwrite(save_name, blended)