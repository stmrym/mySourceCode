from PIL import Image
import os
import glob
from tqdm import tqdm
from pygifsicle import gifsicle


path = '../STDAN_modified/exp_log/train/20231221_2023-12-21T185537_STDAN_Stack_BSD_3ms24ms_GOPRO/visualization/BSD_3ms24ms/epoch-0700/output/027'
savename = '../STDAN_modified/exp_log/train/20231221_2023-12-21T185537_STDAN_Stack_BSD_3ms24ms_GOPRO/visualization/BSD_3ms24ms/epoch-0700/output/027.gif'

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

