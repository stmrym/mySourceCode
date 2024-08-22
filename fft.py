import cv2
import numpy as np

# path = '/mnt/d/results/SIP_202408/presentation_fig/lena_std.bmp'
path = '/mnt/d/results/SIP_202408/presentation_fig/00000079_b.png'
save_path = '/mnt/d/results/SIP_202408/presentation_fig/fft_00000079_b.png'

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = (img/255.0).astype(np.float32)

fft = np.fft.fftshift(np.fft.fft2(img))
fft_abs = np.absolute(fft)

fft_abs = 20*np.log(fft_abs)


fft_abs = np.clip(fft_abs, -105, 241)
fft_norm = (fft_abs - fft_abs.min()) / (fft_abs.max() - fft_abs.min()) 

# fft_img = np.clip((fft_abs / fft_abs.max() * 255), 0, 255).astype(np.uint8)
fft_img = np.clip((fft_norm * 255), 0, 255).astype(np.uint8)


print(fft_abs.max(), fft_abs.min())
print(fft_norm.max(), fft_norm.min())
print(fft_norm)
cv2.imwrite(save_path, fft_img)


# im.save(dst)