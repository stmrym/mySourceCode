from PIL import Image
import os
import glob
from tqdm import tqdm
from pygifsicle import gifsicle


path = '../dataset/chronos/test/0425-182036/blur'
savename = '../dataset/chronos/test/0425-182036/blur.gif'

file_list = sorted(glob.glob(os.path.join(path, '*.png')))
# file_list = file_list[2:-3]

pictures = []

for file in file_list:
    img = Image.open(file).quantize()
    # img_resize = img.resize((img.width//2, img.height//2))
    pictures.append(img)



pictures[0].save(savename,save_all=True, append_images=pictures[1:], optimize=True, loop=0)

gifsicle(sources=savename, destination=savename ,optimize=False,colors=256,options=["--optimize=3"]
)

