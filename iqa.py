import cv2
from brisque import BRISQUE
import numpy as np
from stdan.metrics import NIQE
import matplotlib.pyplot as plt
from tqdm import tqdm




def calc_metric(metric, img, A, B):
    Z = np.zeros_like(A)
    # print(A)
    # exit()
    # for i in tqdm(range(A.shape[0])):
    #     for j in range(A.shape[1]):
    #         a = A[i,j]
    #         b = B[i,j]
    #         trans_img = np.clip((a * img + b)*255, 0, 255)
    #         Z[i,j] = metric.get_score(trans_img)
    #         # Z[i,j] = metric.calculate((a * img + b)*255)
    #         print(f'a:{a:.3f}, b:{b:.3f}, z:{Z[i,j]:.3f} min:{trans_img.min()}, max:{trans_img.max()}')



    return Z


# 2024-11-12T073940__CT_
# 2024-11-12T080924__
# 2024-11-14T040540__STDAN_

# VID_20240523_165150

if __name__ == '__main__':

    image_path = '../STDAN_modified/exp_log/test/2024-11-14T040540__STDAN_/epoch-0400_Mi11Lite_output/VID_20240523_165150/00038.png'
    image = cv2.imread(image_path)
    save_name = 'test_38_STDAN_80.png'
    metric = BRISQUE()

    a_min, a_max = -2, 1
    b_min, b_max = -1.0, 0.999
    a = np.linspace(a_min, a_max, 80)
    b = np.linspace(b_min, b_max, 80)
    x = np.linspace(a_min, a_max, 100)


    A, B = np.meshgrid(a,b)
    mask = (B >= -10**A) & (B < 1)
    A_masked = A[mask]
    B_masked = B[mask]
    # niqe = NIQE(crop_border=0)
    
    # metric = NIQE(crop_border=0)

    z_values = []
    for a, b in zip(tqdm(A_masked), B_masked):
        tran_img = np.clip((10**a * image/255 + b)*255, 0, 255).astype(np.uint8)
        value = metric.get_score(tran_img)
        # value = metric.calculate(tran_img)
        z_values.append(value)
        # print(a,b, value)

    brisque_score = metric.get_score(image)

    # Z = calc_metric(metric, image/255.0, A, B)
    Z = np.zeros_like(A)
    Z[mask] = z_values
    Z[Z == 0] = np.nan

    min_indices = np.where(Z == np.min(Z[Z>0]))
    min_x = A[min_indices]
    min_y = B[min_indices]

    plt.figure(figsize=(6, 6)) 
    # contour = plt.contourf(A, B, Z, levels=50, cmap='viridis', vmin=np.min(Z[Z>0]))
    plt.pcolormesh(A, B, Z, shading='auto', cmap='plasma_r', vmin=np.min(Z[Z>0]))


    # plt.colorbar(contour, label='BRISQUE')
    plt.colorbar(label='BRISQUE')


    y1 = -10**x
    plt.plot(x, y1, 'k-')  
    y2 = np.ones_like(x) * b_max
    plt.plot(x, y2, 'k-')
    plt.xlim([a_min, a_max])
    plt.ylim([b_min, b_max])

    plt.xlabel('log(A)') 
    plt.ylabel('B') 
    # plt.xscale('log')
    plt.scatter(0, 0, color='blue', marker='*', s=100, label=f'(0, 0): {brisque_score:.2f}')
    plt.scatter(min_x, min_y, color='red', marker='*', s=100, label=f'({min_x[0]:.2f}, {min_y[0]:.2f}): {np.min(Z[Z>0]):.2f}')
    plt.legend(loc='lower left')
    plt.savefig(save_name, bbox_inches='tight', dpi=200, pad_inches=0.1)

    

    print(f"BRISQUEスコア: {brisque_score}")
