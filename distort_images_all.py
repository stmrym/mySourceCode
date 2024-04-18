import os
import glob
import numpy as np
import cv2
from distort_images import change_constrast, add_noise
from add_raw_noise import add_raw_noise
from unprocess_torch import random_ccm, random_gains

mode = 'pair'
blur_dirname = 'blur_gamma'
sharp_dirname = 'sharp'
# src_dir = '/mnt/d/dataset/chronos/004'
src_dir = '/home/moriyamasota/dataset/GOPRO_Large/train/GOPR0372_07_01'
# src_dir = '/mnt/d/results/20240410/STDAN/004'
dst_dir = '/home/moriyamasota/dataset/GOPRO_Large_raw/train/GOPR0372_07_01'
# dst_dir = '/mnt/d/dataset/chronos/004_raw'
# dst_dir = '/mnt/d/results/20240410/STDAN/004_raw'

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

    os.makedirs(os.path.join(dst_dir, blur_dirname), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, sharp_dirname), exist_ok=True)

    random_ccm_tensor = random_ccm() 
    random_gains_list = random_gains() 

    blur_image_paths = sorted(glob.glob(os.path.join(src_dir, blur_dirname, '*.png'), recursive=True))
    sharp_image_paths = sorted(glob.glob(os.path.join(src_dir, sharp_dirname, '*.png'), recursive=True))

    for blur_path, sharp_path in zip(blur_image_paths, sharp_image_paths):
        print(blur_path, sharp_path)
        
        blur_image = cv2.cvtColor(cv2.imread(blur_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        sharp_image = cv2.cvtColor(cv2.imread(sharp_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        images = np.stack((sharp_image, blur_image), axis=0)
        images = (images/255).astype(np.float32)

        # Input: np array normalized [0,1] of shape (b, h, w, c)
        # batch :0 becomes GT
        # batch :1 becomes noisy       
        dst_sharp_image, dst_blur_image = add_raw_noise(images, random_ccm_tensor, random_gains_list)

        dst_blur_image = cv2.cvtColor((dst_blur_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        dst_sharp_image = cv2.cvtColor((dst_sharp_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # dst_image = change_constrast(image, alpha=alpha, beta=beta)
        cv2.imwrite(blur_path.replace(src_dir, dst_dir), dst_blur_image)
        cv2.imwrite(sharp_path.replace(src_dir, dst_dir),dst_sharp_image)


