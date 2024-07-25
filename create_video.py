import cv2
from PIL import Image
import os
import glob
from tqdm import tqdm
from pygifsicle import gifsicle


def create_gif(path: str, savename: str) -> None:

    path = '../dataset/chronos/test/0306-222113/blur'
    savename = '../dataset/0306-222113.gif'

    file_list = sorted(glob.glob(os.path.join(path, '*.png')))
    # file_list = file_list[2:-3]

    pictures = []

    for file in tqdm(file_list):
        img = Image.open(file).quantize()
        img = img.resize((img.width//2, img.height//2))
        pictures.append(img)

    pictures[0].save(savename,save_all=True, append_images=pictures[1:], optimize=True, loop=0)

    gifsicle(sources=savename, destination=savename ,optimize=False,colors=256,options=["--optimize=3"])

def create_mp4(path: str, savename: str) -> None:

    file_list = sorted(glob.glob(os.path.join(path, '*.png')))
    # file_list = file_list[2:-3]

    img0 = cv2.imread(file_list[0])
    h, w, _ = img0.shape
    k = 4

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')                                                                                                                                                         
    video = cv2.VideoWriter(savename,fourcc, 15., (w//k, h//k))                                                                                                                                                
                                                                                                                                                                                                            
    for fname in file_list:                                                                                                                                                                                  
        img = cv2.imread(fname)     
        img = cv2.resize(img, (w//k, h//k))                                                                                                                                                                            
        video.write(img)                                                                                                                                                                                        
    video.release()


if __name__ == '__main__':

    path = '../dataset/BSD_3ms24ms/test/074/Blur/RGB'
    # path = '../dataset/GOPRO_Large/test/GOPR0862_11_00/blur_gamma'

    # savename = '../dataset/0306-222113.gif'
    savename = '../dataset/074.mp4'


    create_mp4(path, savename)