import torch
from torchvision import models, transforms
from torcheval.metrics import FrechetInceptionDistance
import numpy as np
from skimage.util import view_as_windows
import cv2
import matplotlib.pyplot as plt


def extract_patches(image, patch_size, overlap_fraction):
    h, w, c = image.shape
    step = int(patch_size[0] * overlap_fraction)
    
    patches = []
    for i in range(0, h - patch_size[0] + 1, step):
        for j in range(0, w - patch_size[1] + 1, step):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            patches.append(patch)
    
    return np.array(patches)


if __name__ == '__main__':


    # 画像の前処理
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 実画像と生成画像の例（ランダムな画像を使用）
    real_images = torch.randn(10, 3, 299, 299)  # 実画像のバッチ
    generated_images = torch.randn(10, 3, 299, 299)  # 生成画像のバッチ

    # 前処理を適用
    real_images = torch.stack([preprocess(image) for image in real_images])
    generated_images = torch.stack([preprocess(image) for image in generated_images])

    # Frechet Inception Distance (FID) の計算
    fid_metric = FrechetInceptionDistance()

    # 実画像と生成画像の特徴を更新
    fid_metric.update(real_images, real=True)
    fid_metric.update(generated_images, real=False)

    # FIDスコアの計算
    fid_score = fid_metric.compute()
    print("FID Score:", fid_score)



    image = cv2.imread('deblurred.png')
    patch_size = (256, 256)
    print(image.shape)
    overlap_fraction = 7 / 8  # オーバーラップ量 (1/8)

    patches = extract_patches(image, patch_size, overlap_fraction)


    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(patches[i])
        ax.axis('off')
    plt.savefig('test_patch.png')