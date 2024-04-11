import os
import glob
import numpy as np
import cv2
from distort_images import change_constrast, add_noise

mode = 'single'

# src_dir = '/mnt/d/dataset/chronos/004'
src_dir = '/mnt/d/results/20240410/STDAN/004_a03_b0'
# dst_dir = '/mnt/d/dataset/chronos/004_s10'
dst_dir = '/mnt/d/results/20240410/STDAN/004_a03a1'

if mode == 'single':
    os.makedirs(os.path.join(dst_dir), exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(src_dir, '*.png'), recursive=True))

    for path in image_paths:
        print(path)
        
        image = cv2.imread(path)
        alpha = 1/0.3
        beta = 0
        sigma = 20

        # noise = np.random.normal(0, sigma, image.shape)

        dst_image = change_constrast(image, alpha=alpha, beta=beta)
        cv2.imwrite(path.replace(src_dir, dst_dir), dst_image)

elif mode == 'pair':

    os.makedirs(os.path.join(dst_dir, 'blur'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'sharp'), exist_ok=True)


    blur_image_paths = sorted(glob.glob(os.path.join(src_dir, 'blur', '*.png'), recursive=True))
    sharp_image_paths = sorted(glob.glob(os.path.join(src_dir, 'sharp', '*.png'), recursive=True))

    for blur_path, sharp_path in zip(blur_image_paths, sharp_image_paths):
        print(blur_path, sharp_path)
        
        blur_image = cv2.imread(blur_path)
        sharp_image = cv2.imread(sharp_path)

        alpha = 0.3
        beta = 0
        sigma = 20
        
        print(blur_image[0:3,0:3,0])
        print(sharp_image[0:3,0:3,0])


        noise = np.random.normal(0, sigma, blur_image.shape)

        print('^^^^^^^^^^^^')
        print(noise[0:3,0:3,0])

        dst_blur_image = add_noise(blur_image, noise)
        dst_sharp_image = add_noise(sharp_image, noise)

        print('------------')
        print(dst_blur_image[0:3,0:3,0])
        print(dst_sharp_image[0:3,0:3,0])
        exit()

        # dst_image = change_constrast(image, alpha=alpha, beta=beta)
        cv2.imwrite(blur_path.replace(src_dir, dst_dir), dst_blur_image)
        cv2.imwrite(sharp_path.replace(src_dir, dst_dir),dst_sharp_image)


