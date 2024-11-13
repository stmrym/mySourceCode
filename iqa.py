import cv2
from brisque import BRISQUE
import numpy as np
from stdan.metrics import NIQE
import matplotlib.pyplot as plt
from tqdm import tqdm




def calc_metric(metric, img, A, B):
    Z = np.zeros_like(A)
    for i in tqdm(range(A.shape[0])):
        for j in range(A.shape[1]):
            a = A[i,j]
            b = B[i,j]
            Z[i,j] = metric.get_score(a * img + b)
            print(f'a:{a:.3f}, b:{b:.3f}, z:{Z[i,j]:.3f}')

    return Z




if __name__ == '__main__':

    image_path = '/mnt/d/results/20241115/1_in_00038.png'
    image = cv2.imread(image_path)
    image = (image / 255.0).astype(np.float32)


    a = np.linspace(1e-1, 1e+3, 10)
    b = np.linspace(-10, 1, 10)
    x = np.linspace(0, 5, 100)


    A, B = np.meshgrid(a,b)
    brisque = BRISQUE()
    Z = calc_metric(brisque, image, A, B)

    plt.figure(figsize=(6, 6)) 
    contour = plt.contourf(A, B, Z, levels=50, cmap='viridis')
    plt.xlabel('a') 
    plt.ylabel('b') 
    plt.xscale('log')
    plt.colorbar(contour, label='Mean Value of B = a*I + b')

    y1 = -np.exp(x)
    plt.plot(x, y1, 'k-', label='y = -e^x')  
    y2 = np.ones_like(x)
    plt.plot(x, y2, 'k-', label='y = 1')

    plt.ylim([1e-3, 1])


    plt.savefig('test.png', bbox_inches='tight', pad_inches=0.1)

    exit()
    brisque = BRISQUE()
    niqe = NIQE()
    brisque_score = brisque.get_score(image)

    print(f"BRISQUEスコア: {brisque_score}")
