from __future__ import annotations
import cv2
import glob
import os
from tqdm import tqdm

def resize(file_paths: list[str], base_dir: str, save_dir: str, size: tuple[int]) -> None: 
    
    assert file_paths != [], 'file paths empty.'
    for file_path in tqdm(file_paths):
        img = cv2.imread(file_path)
        resized_img = cv2.resize(img, size)
        save_path = file_path.replace(base_dir, save_dir)
        dir_path = os.path.dirname(save_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(save_path, resized_img)

def crop(file_paths: list[str], base_dir: str, save_dir: str, size: tuple[int]) -> None:

    assert file_paths != [], 'file paths empty.'
    for file_path in tqdm(file_paths):
        img = cv2.imread(file_path)
        H, W, _ = img.shape 
        w, h = size
        # cropped_img = img[(H-h)//2:(H+h)//2, (W-w)//2:(W+w)//2, :]
        cropped_img = img[0:h, (W-w)//2:(W+w)//2, :]        
        save_path = file_path.replace(base_dir, save_dir)
        dir_path = os.path.dirname(save_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(save_path, cropped_img)





if __name__ == '__main__':

    # base_dir = '../dataset/Mi11Lite/test'
    # file_paths = sorted(glob.glob(os.path.join(base_dir, '*/*.png'), recursive=True))
    # save_dir = '../dataset/Mi11Lite_cropped/test'

    base_dir = '../dataset/Mi11Lite/test/VID_20240523_164838'
    file_paths = sorted(glob.glob(os.path.join(base_dir, '*.png'), recursive=True))
    save_dir = '../dataset/Mi11Lite_cropped/test/VID_20240523_164838_2'

    # resize(file_paths, base_dir, save_dir, (1280, 720))
    crop(file_paths, base_dir, save_dir, (1280, 720))