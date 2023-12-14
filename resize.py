import cv2
import glob

files = sorted(glob.glob('../dataset/night_blur_resized/test/input/001/*.png', recursive=True))


for filename in files:
    img = cv2.imread(filename)
    H, W, _ = img.shape
    img2 = cv2.resize(img, (W//4, H//4))
    cv2.imwrite(filename, img2)
    print(filename)