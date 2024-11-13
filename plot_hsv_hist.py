import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

class ImageProperty:
    def __init__(self, label, image_path, color):
        self.label = label
        self.image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.merge([gray_image, gray_image, gray_image])
        
        self.color = color


def plot_hists(images_l, save_prefix, mask=None):
    
    hsv_l = [cv2.cvtColor(image_p.image, cv2.COLOR_BGR2HSV) for image_p in images_l]

    hsv_dict = OrderedDict(
        Hue = [cv2.calcHist([hsv], [0], mask, [256], [0, 256]) for hsv in hsv_l],
        Saturation = [cv2.calcHist([hsv], [1], mask, [256], [0, 256]) for hsv in hsv_l],
        Value = [cv2.calcHist([hsv], [2], mask, [256], [0, 256]) for hsv in hsv_l]
    )
    
    for title, hists in hsv_dict.items():
        plt.figure()
        plt.title(title)
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        for hist, image_p in zip(hists, images_l):
            plt.plot(hist, color=image_p.color, label=image_p.label, alpha=0.7)
        plt.xlim([0, 256])
        plt.legend()
        plt.savefig(save_prefix + '_' + title[0] + '.png', dpi=200)



if __name__ == '__main__':

    images_l = [
        ImageProperty(label='GT', color='black', image_path='/mnt/d/results/20241115/00000014.png'),
        ImageProperty(label='ESTDAN', color='tab:blue', image_path='/mnt/d/results/20241115/3_out_00000014.png'),
        ImageProperty(label='Real_ESTDAN', color='tab:red', image_path='/mnt/d/results/20241115/4_CT_00000014.png')
    ]

    save_prefix = 'g_hist'
    plot_hists(images_l, save_prefix)
