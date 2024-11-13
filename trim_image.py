import cv2
import numpy as np

src_path = '/mnt/d/results/20241115/1_in_00038.png'
dst_path = '/mnt/d/results/20241115/1_in_00038_changed.png'


contrast = 10000
brightness = 0

src = cv2.imread(src_path)
src = (src/255.0).astype(np.float32)
dst = src * contrast + brightness 
dst = np.clip(dst*255, 0, 255).astype(np.uint8)

# dst = src[180:550,700:,:]
cv2.imwrite(dst_path, dst)