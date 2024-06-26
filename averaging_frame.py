import cv2
import os
import glob
import numpy as np
import shutil
import rawpy

def averaging_frames(frame_dir_path, blur_dir_path, sharp_dir_path, avg_step, ext='png', save_ext='png'):

    contrast = 2
    brightness = 0.00
    all_frames = sorted(glob.glob(os.path.join(frame_dir_path, '*.' + ext)))
    os.makedirs(blur_dir_path, exist_ok=True)
    os.makedirs(sharp_dir_path, exist_ok=True)

    if ext == 'dng':
        with rawpy.imread(all_frames[0]) as raw:
            rgb = raw.postprocess()
        h, w, c = rgb.shape
    else:
        h, w, c = cv2.imread(all_frames[0]).shape

    for i, start_idx in enumerate(range(0, len(all_frames), avg_step)):
        frames = all_frames[start_idx:start_idx+avg_step]
        total_img = np.zeros((h, w, c))
        for frame in frames:
            img = cv2.imread(frame)
            img_scaled = (img/255.).astype(np.float32)
            img_scaled = np.power(img_scaled, 2.2)

            total_img += img_scaled
            if frame == frames[len(frames)//2]:
                center_img = img_scaled*contrast + brightness


        avg_img = total_img * contrast / len(frames)
        avg_img = avg_img + brightness
        avg_img = np.power(avg_img, 1/2.2)
        center_img = np.power(center_img, 1/2.2)
        blur_img = np.clip(avg_img*255, a_min=0, a_max=255).astype(np.uint8)
        sharp_img = np.clip(center_img*255, a_min=0, a_max=255).astype(np.uint8)
        blur_name = os.path.join(blur_dir_path, f'{str(i).zfill(5)}.{save_ext}')
        sharp_name = os.path.join(sharp_dir_path, f'{str(i).zfill(5)}.{save_ext}')
        
        cv2.imwrite(blur_name, blur_img)
        cv2.imwrite(sharp_name, sharp_img)
        # shutil.copy(frames[len(frames)//2], sharp_name)
        print(f'{blur_name} saved.')


if __name__ == '__main__':
    averaging_frames('/mnt/d/Chronos/0425-182036', '/mnt/d/dataset/chronos/0425-182036/blur', '/mnt/d/dataset/chronos/0425-182036/sharp', avg_step=80)
