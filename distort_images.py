from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(img_array, savename, ver='bgr'):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()

    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Number of pixels')

    ax.set_xlim([-0.5,255.5])
    ax.set_ylim([0,30000])

    if ver == 'bgr':
        color_list = ['blue', 'green', 'red']
    elif ver == 'yuv':
        color_list = ['black', 'tab:red', 'tab:blue']

    # ax.hist(img_array, color=['red', 'green', 'blue'], range=(-0.5,255.5), histtype='step', bins=256)
    ax.hist(img_array, color=color_list, range=(-0.5,255.5), histtype='step', bins=256)

    ax.minorticks_on()
    # ax.grid(which = "both", axis="x")
    # ax.grid(axis='y')

    plt.tight_layout()
    plt.savefig(savename, dpi=300)


def equalize_hist(img):
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_y = img_yuv[:,:,0]
        img_yuv[:,:,0] = cv2.equalizeHist(img_y)
        dst_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        print(dst_img.dtype)
    elif len(img.shape) == 2:
        dst_img = cv2.equalizeHist(img)
    return dst_img

def change_constrast(img, alpha=1, beta=0):
    dst_img = alpha*img + beta
    dst_img = np.clip(dst_img*255, 0, 255).astype(np.uint8)
    return dst_img


def add_noise(img, noise):
    dst_img = img.astype(np.float32) + noise
    dst_img = np.clip(dst_img, 0, 255).astype(np.uint8)
    return dst_img



def main():

    # 2024-11-12T073940__CT_
    # 2024-11-12T080924__
    # 2024-11-14T040540__STDAN_

    # VID_20240523_165150   38
    # VID_20240523_163120   81

    path = '../STDAN_modified/exp_log/test/2024-11-14T040540__STDAN_/epoch-0400_Mi11Lite_output/VID_20240523_165150/00038.png'
    out_name = '00038_stdan.png'
    
    img = cv2.imread(path).astype(np.float32)
    img /= 255.

    alpha = 10**(-0.04)
    beta = 0.06
    sigma = 20

    # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # plot_hist(img_yuv.reshape(-1,3), '/mnt/d/results/20240403/' + out_name + '_yuv_in_hist.png', ver='yuv')
    # plot_hist(img.reshape(-1,3), '/mnt/d/results/20240403/' + out_name + '_in_hist.png', ver='bgr')


    # img_yuv[:,:,0] = change_constrast(img_yuv[:,:,0], alpha=alpha, beta=beta)
    # dst_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    dst_img = change_constrast(img, alpha=alpha, beta=beta)
    # dst_img = np.clip(img*255, 0, 255).astype(np.uint8)
    dst_img = dst_img[50:550,750:1250,:]
    # noise = np.random.normal(0, sigma, img.shape)
    # dst_img = add_noise(img, noise = noise)




    # dst_img_yuv = cv2.cvtColor(dst_img, cv2.COLOR_BGR2YUV)
    cv2.imwrite(out_name, dst_img)
    # plot_hist(dst_img_yuv.reshape(-1,3), '/mnt/d/results/20240403/' + out_name + '_yuv_out_hist.png', ver='yuv')
    # plot_hist(dst_img.reshape(-1,3), '/mnt/d/results/20240403/' + out_name + '_out_hist.png', ver='bgr')

if __name__ == '__main__':
    main()