import cv2
import glob

files = sorted(glob.glob('../dataset/ADAS_resized/test/*/*/*.jpg', recursive=True))


for filename in files:
    img = cv2.imread(filename)
    H, W, _ = img.shape
    img2 = cv2.resize(img, (W//2, H//2))
    cv2.imwrite(filename, img2)
    print(filename)