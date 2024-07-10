import os
import cv2
import numpy as np


path = '/mnt/d/results/20240712/00000123_blur.png'
save_name = '/mnt/d/results/20240712/2_00000123_blur.png'
img = cv2.imread(path)
print(img.shape)
assert img is not None, 'image empty.'

h, w, _ = img.shape
# BSD 1
# x1, y1, w1, h1 = [205, 185, 90, 90]
# x2, y2, w2, h2 = [230, 385, 100, 70]
# BSD 2
# x1, y1, w1, h1 = [110, 60, 320, 130]
# x2, y2, w2, h2 = [135, 280, 140, 100]
# GOPRO
x1, y1, w1, h1 = [425, 210, 45, 45]
x2, y2, w2, h2 = [290, 0, 130, 60]

color1 = (0,0,255)
color2 = (0,255,0)
thick_size = 3
mode = 'bottom' # 'bottom' or 'right' 

# calculate resize ratio
if mode == 'bottom':
    k1 = h2*w / (h2*w1 + h1*w2)
    k2 = h1*w / (h2*w1 + h1*w2)
elif mode == 'right':
    k1 = w2*h / (h1*w2 + h2*w1)
    k2 = w1*h / (h1*w2 + h2*w1)

# clipping
img1 = img[y1:y1+h1, x1:x1+w1, :]
img2 = img[y2:y2+h2, x2:x2+w2, :]
clip1 = cv2.resize(img1, dsize=None, fx=k1, fy=k1, interpolation=cv2.INTER_NEAREST)
clip2 = cv2.resize(img2, dsize=None, fx=k2, fy=k2, interpolation=cv2.INTER_NEAREST)

# draw line
cv2.rectangle(clip1, (1, 1), (clip1.shape[1]-2, clip1.shape[0]-2), color=color1, thickness=thick_size)
cv2.rectangle(img, (x1, y1, w1, h1), color=color1, thickness=thick_size)
cv2.rectangle(clip2, (1, 1), (clip2.shape[1]-2, clip2.shape[0]-2), color=color2, thickness=thick_size)
cv2.rectangle(img, (x2, y2, w2, h2), color=color2, thickness=thick_size)

print(clip1.shape)
print(clip2.shape)

if mode == 'bottom':
    bottom = np.hstack((clip1, clip2))
    out = np.vstack((img, bottom))
elif mode == 'right':
    right = np.vstack((clip1, clip2))
    out = np.hstack((img, right))

cv2.imwrite(save_name, out)



