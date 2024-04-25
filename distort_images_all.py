import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

from distort_images import change_constrast
from add_raw_noise import add_raw_noise
from unprocess_torch import random_ccm, random_gains, random_noise_levels


def make_dataset_each_seq(src_dir, dst_dir, blur_dirname, sharp_dirname, seq=None):

    if seq is not None:
        blur_image_paths = sorted(glob.glob(os.path.join(src_dir, seq, blur_dirname, '*.png'), recursive=True))
        sharp_image_paths = sorted(glob.glob(os.path.join(src_dir, seq, sharp_dirname, '*.png'), recursive=True))
        os.makedirs(os.path.join(dst_dir, seq, blur_dirname), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, seq, sharp_dirname), exist_ok=True)

    else:
        blur_image_paths = sorted(glob.glob(os.path.join(src_dir, blur_dirname, '*.png'), recursive=True))
        sharp_image_paths = sorted(glob.glob(os.path.join(src_dir, sharp_dirname, '*.png'), recursive=True))
        os.makedirs(os.path.join(dst_dir, blur_dirname), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, sharp_dirname), exist_ok=True)

    random_ccm_tensor = random_ccm() 
    random_gains_list = random_gains() 
    lambda_shot, lambda_read = random_noise_levels()

    rng = np.random.default_rng()
    contrast = rng.uniform(0.3, 1.0)
    contrast = 1
    brightness = 0.0

    print(seq, contrast)

    for blur_path, sharp_path in zip(tqdm(blur_image_paths), sharp_image_paths):
        
        blur_image = cv2.cvtColor(cv2.imread(blur_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        sharp_image = cv2.cvtColor(cv2.imread(sharp_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        images = np.stack((sharp_image, blur_image), axis=0)
        images = (images/255).astype(np.float32)

        # Input: np array normalized [0,1] of shape (b, h, w, c)
        # batch :0 becomes GT
        # batch :1 becomes noisy       
        dst_sharp_image, dst_blur_image = add_raw_noise(images, random_ccm_tensor, random_gains_list, lambda_shot, lambda_read,
                                                        contrast, brightness)

        dst_blur_image = cv2.cvtColor((dst_blur_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        dst_sharp_image = cv2.cvtColor((dst_sharp_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        cv2.imwrite(blur_path.replace(src_dir, dst_dir), dst_blur_image)
        cv2.imwrite(sharp_path.replace(src_dir, dst_dir),dst_sharp_image)


if __name__ == '__main__':

    mode = 'pair'
    blur_dirname = 'blur'
    sharp_dirname = 'sharp'
    src_dir = '/home/moriyamasota/dataset/chronos/test'
    dst_dir = '/home/moriyamasota/dataset/chronos_raw/test'
    
    assert src_dir != dst_dir, 'Do not equal src_dir to dst_dir'

    if mode == 'single':
        os.makedirs(os.path.join(dst_dir), exist_ok=True)
        image_paths = sorted(glob.glob(os.path.join(src_dir, '*.png'), recursive=True))
        for path in image_paths:
            image = cv2.imread(path)
            dst_image = change_constrast(image, alpha=1/0.3, beta=0)
            cv2.imwrite(path.replace(src_dir, dst_dir), dst_image)

    elif mode == 'pair':
        # single_seq -> seqs = [None]
        # multi_seqs -> seqs = [seq1, seq2, ...]
        seqs = [None] if os.path.isdir(os.path.join(src_dir, blur_dirname))\
                else sorted([f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))])

        for seq in seqs:
            make_dataset_each_seq(src_dir, dst_dir, blur_dirname, sharp_dirname, seq)

