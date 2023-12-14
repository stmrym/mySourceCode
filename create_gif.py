from PIL import Image
import os
import glob
from tqdm import tqdm
from pygifsicle import gifsicle

# path = '../STDAN_modified/exp_log/test/20231129_STDAN_Stack_night_blur_ckpt-epoch-0050/mmflow_npy/001'
# savename = '../STDAN_modified/exp_log/test/20231129_STDAN_Stack_night_blur_ckpt-epoch-0050/mmflow_npy.gif'
# path = '../dataset/night_blur_resized/test/input/001'
# savename = '../dataset/night_blur_resized/test/001.gif'
path = '../Tracking-Anything-with-DEVA/example/output/Visualizations'
savename = '../Tracking-Anything-with-DEVA/example/output/Visualizations.gif'

file_list = sorted(glob.glob(os.path.join(path, '*.png')))
file_list = file_list[2:-3]

pictures = []

for file in file_list:
    img = Image.open(file).quantize()
    # img_resize = img.resize((img.width//2, img.height//2))
    pictures.append(img)



pictures[0].save(savename,save_all=True, append_images=pictures[1:], optimize=True, loop=0)

gifsicle(sources=savename, destination=savename ,optimize=False,colors=256,options=["--optimize=3"]
)

