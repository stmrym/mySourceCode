import os
import glob
import cv2
from distort_images import change_constrast, add_gaussian_noise


src_dir = '/mnt/d/dataset/chronos/004'
dst_dir = '/mnt/d/dataset/chronos/004_s20'
os.makedirs(os.path.join(dst_dir, 'blur'), exist_ok=True)
os.makedirs(os.path.join(dst_dir, 'sharp'), exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(src_dir, '**/*.png'), recursive=True))

for image_path in image_paths:

    image = cv2.imread(image_path)
    alpha = 0.7
    beta = 0
    sigma = 10
    dst_image = add_gaussian_noise(image, sigma=sigma)
    # dst_image = change_constrast(image, alpha=alpha, beta=beta)
    save_path = image_path.replace(src_dir, dst_dir)
    print(save_path)
    cv2.imwrite(save_path, dst_image)

