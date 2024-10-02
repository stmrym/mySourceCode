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

def create_mp4(path: str, savename: str, seq: str = 'all') -> None:

    if seq == 'all':
        base_path = path.split('%s')[0]
        seq_list = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    else:
        seq_list = [seq]
    print(seq_list)

    for seq in seq_list:
        file_list = sorted(glob.glob(os.path.join(path % seq, '*.png')))
        # file_list = file_list[2:-3]

        img0 = cv2.imread(file_list[0])
        h, w, _ = img0.shape
        k = 4

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')                                                                                                                                                         
        video = cv2.VideoWriter(savename % seq,fourcc, 10., (w//k, h//k))                                                                                                                                                
                                                                                                                                                                                                                
        for fname in tqdm(file_list):                                                                                                                                                                                  
            img = cv2.imread(fname)     
            img = cv2.resize(img, (w//k, h//k))                                                                                                                                                                            
            video.write(img)                                                                                                                                                                                        
        video.release()


if __name__ == '__main__':

    # path = '../dataset/BSD_3ms24ms/test/%s/Blur/RGB'
    # path = '../dataset/GOPRO_Large/test/GOPR0862_11_00/blur_gamma'
    # path = '../dualBR/output/realBR/%s/RGB'
    path = '../dualBR/output/20240829_b4_1/realBR/%s/RGB'
    # path = '/mnt/d/dataset/realBR/test/%s/RS/RGB'

    # savename = '../dataset/BSD_3ms24ms/%s.mp4'
    # savename = '../dualBR/output/realBR/%s.mp4'
    savename = '../dualBR/output/20240829_b4_1/video/%s.mp4'
    # savename = '/mnt/d/dataset/realBR/test/%s.mp4'

    create_mp4(path, savename, seq='all')