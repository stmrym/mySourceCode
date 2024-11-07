import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('Mandrill.bmp').astype(np.float32)
h, w, c = im.shape

print(h,w,c)
uniform_noise = np.random.uniform(-1, 1, (h,w,c))*128
normal_noise = np.random.normal(0, 0.1, (h,w,c))
normal_noise = (normal_noise - normal_noise.min())/(normal_noise.max() - normal_noise.min())*256 - 128


# uniform_noise_img = np.clip(uniform_noise, 0, 255).astype(np.uint8)
# normal_noise_img = np.clip(normal_noise, 0, 255).astype(np.uint8)

fig, ax = plt.subplots()
# plt.hist(uniform_noise_img.flatten(), bins=255)
plt.hist(normal_noise.flatten(), bins=255)
plt.savefig('uniform.png')

# cv2.imwrite('normal_img.png', normal_noise_img)
# cv2.imwrite('uniform_img.png', uniform_noise_img)