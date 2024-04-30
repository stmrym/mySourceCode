import cv2
import glob

files = sorted(glob.glob('/home/moriyamasota/dataset/chronos/test/0306-222503/*/*.png', recursive=True))

for filename in files:
    img = cv2.imread(filename)
    H, W, _ = img.shape
    img2 = img[180:900, 320:1600, :]
    # img2 = cv2.resize(img, (W//2, H//2))
    cv2.imwrite(filename, img2)
    print(filename)