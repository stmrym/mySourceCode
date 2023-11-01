import cv2
import os
import numpy as np

comp_path = '/mnt/d/dataset/real_RR_result/3_input.png'
img_path = '/mnt/d/dataset/real_RR_result/3_e0400.png'



tuple_path = os.path.splitext(img_path)

comp = cv2.imread(comp_path)
img = cv2.imread(img_path)

cv2.imwrite(tuple_path[0] + '_diff_1' + tuple_path[1], cv2.absdiff(comp, img))

comp = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

comp_scaled = (comp/255.).astype(np.float32)
img_scaled = (img/255.).astype(np.float32)

diff_scaled = (img_scaled - comp_scaled)/2 + 0.5
diff = np.clip(diff_scaled*255, a_min=0, a_max=255).astype(np.uint8)

cv2.imwrite(tuple_path[0] + '_diff_2' + tuple_path[1], diff)


