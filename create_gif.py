from PIL import Image
import os
import glob
from tqdm import tqdm
from pygifsicle import gifsicle


path = '../dataset/GOPRO_Large/test/GOPR0854_11_00/blur_gamma'
savename = '../dataset/GOPR0854_11_00.gif'
# savename = '../dataset/chronos/test/0425-182036/blur.gif'

file_list = sorted(glob.glob(os.path.join(path, '*.png')))
# file_list = file_list[2:-3]

pictures = []

for file in tqdm(file_list):
    img = Image.open(file).quantize()
    img = img.resize((img.width//2, img.height//2))
    pictures.append(img)



pictures[0].save(savename,save_all=True, append_images=pictures[1:], optimize=True, loop=0)

gifsicle(sources=savename, destination=savename ,optimize=False,colors=256,options=["--optimize=3"]
)

