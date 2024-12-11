import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def load_images_from_folder(folder, num_images=None):
    images_l = []
    images_path_l = sorted(Path(folder).rglob('**/*.png'))

    with tqdm(images_path_l) as pbar:
        for i, img_path in enumerate(pbar):
            pbar.set_description(str(img_path))
            if num_images and i >= num_images:
                break
            img = Image.open(img_path).convert('L')  # 画像をグレースケールに変換
            images_l.append(np.array(img))
    return images_l


def load_single_image_from_path(path):
    images_l = []
    img = Image.open(path).convert('L')
    images_l.append(np.array(img))
    return images_l
    

def plot_histogram(images, color, label):
    all_pixels = np.concatenate([img.flatten() for img in images])
    print(all_pixels.shape)
    plt.hist(all_pixels, bins=256, range=(0, 255), color=color, density=True, alpha=0.7, label=label)
    plt.title("Histogram of Image Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")


if __name__ == '__main__':

    # folder_path = '../dataset/BSD_2ms16ms/test/blur/'
    # images_l = load_images_from_folder(folder_path)

    image_path = '/mnt/d/results/20241210/00000040_blur.png'
    images_l = load_single_image_from_path(image_path)

    print(len(images_l))
    plot_histogram(images_l, color='tab:blue', label='Blurred')
    

    # folder_path = '../dataset/BSD_2ms16ms_unsharp/test/blur/'
    # images_l = load_images_from_folder(folder_path)
    # print(len(images_l))
    # plot_histogram(images_l, color='tab:orange', label='BSD_2ms16ms + UnsharpMask')
    # plt.legend()
    # plt.savefig('hist_normal_unsharp.png', bbox_inches='tight', pad_inches=0.04, dpi=300)

    # folder_path = '../dataset/BSD_2ms16ms_unsharp2/test/blur/'
    # images_l = load_images_from_folder(folder_path)
    # print(len(images_l))
    # plot_histogram(images_l, color='tab:green', label='BSD_2ms16ms + ModifiedUnsharpMask')
    # plt.legend()
    # plt.savefig('hist_normal_unsharp2.png', bbox_inches='tight', pad_inches=0.04, dpi=300)


    image_path = '/mnt/d/results/20241210/067_00000040_default_UM.png'
    images_l = load_single_image_from_path(image_path)
    print(len(images_l))
    plot_histogram(images_l, color='tab:orange', label='+ UnsharpMask')

    plt.legend()
    plt.savefig('hist_000040_unsharp.png', bbox_inches='tight', pad_inches=0.04, dpi=300)