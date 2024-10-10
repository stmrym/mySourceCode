from functools import wraps
import time
import random
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from MulticoreTSNE import MulticoreTSNE as TSNE
# from skilearn.manifold import TSNE


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print("{} min in {}".format(elapsed_time/60, func.__name__))
        return result
    return wrapper

def crop_center(pil_img, crop_size=480):
    w, h = pil_img.size
    return pil_img.crop(((w - crop_size) // 2, (h - crop_size) // 2,
                         (w + crop_size) // 2, (h + crop_size) // 2))

def get_image_list(dir_path_list, n_sample=3000):

    total_image_paths = []
    for dir_path in dir_path_list:
        image_paths = sorted([p for p in Path(dir_path).glob('**/*.png') if re.search('blur_gamma|Blur', str(p)) ])
        total_image_paths += image_paths

    total_image_paths = random.sample(total_image_paths, n_sample)
    # total_image_paths = total_image_paths[250:350]

    # images_np: (T, H, W, C)
    image_list = [crop_center(Image.open(p)) for p in tqdm(total_image_paths)]
    print(len(image_list))
    return image_list, total_image_paths


@stop_watch
def fit(image_list):

    images_np = np.stack(image_list, axis=0)
    images_np = images_np.reshape(images_np.shape[0], -1)

    print(images_np.shape)
    tsne = TSNE(n_jobs=4, perplexity=10)
    images_reduced = tsne.fit_transform(images_np)
    print(images_reduced.shape)
    return images_reduced

def imscatter_transform(image, image_path):
    if 'GOPRO' in str(image_path):
        return ImageOps.expand(image, border=20, fill='cyan')
    elif 'BSD' in str(image_path):
        return ImageOps.expand(image, border=20, fill='magenta')
    else:
        print('need to include "GOPRO" or "BSD" in file path.')
        exit()


def imscatter(x, y, image_list, images_path, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()

    # images_np: [T, H, W, C]
    im_list = [OffsetImage(imscatter_transform(image, path), zoom=zoom) for image, path in zip(image_list, images_path)]
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im in zip(x, y, im_list):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.autoscale()
    return artists
    

if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    dir_path_list = [
        '../dataset/BSD_3ms24ms/test',
        '../dataset/GOPRO_Large/test'
    ]

    image_list, total_images_path = get_image_list(dir_path_list, n_sample=500)
    images_reduced = fit(image_list)

    imscatter(images_reduced[:,0], images_reduced[:,1], image_list, total_images_path, ax=ax, zoom=0.03)
    # plt.scatter(images_reduced[:,0], images_reduced[:,1])
    # fig.savefig('tsne.png', bbox_inches='tight', pad_inches=0)
    plt.show()


