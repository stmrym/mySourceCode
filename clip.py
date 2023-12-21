import cv2
import glob
import os

files = sorted(glob.glob('../dataset/ADAS_15fps_resized/test/*/*/*.jpg', recursive=True))

clip_freq = 2
count = 0
for filename in files:
    if count % clip_freq == 0:
        os.remove(filename)
        print(f'{filename} deleted.')
    count += 1