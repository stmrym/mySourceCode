from PIL import Image
import os
import glob
from tqdm import tqdm
from pygifsicle import gifsicle

path = '../datasets/BSD_3ms24ms'
file_type_list = ['Blur', 'Sharp']

for file_type in file_type_list:

    gif_path = os.path.join(path, 'gif', file_type)

    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    seq_list = sorted(os.listdir(os.path.join(path, 'test')))

    for seq in tqdm(seq_list):

        pictures = []

        file_list = sorted(glob.glob(os.path.join(path, 'test', seq, file_type, 'RGB', '*.png')))
        file_list = file_list[2:-2]

        for file in file_list:
            img = Image.open(file).quantize()
            img_resize = img.resize((img.width//2, img.height//2))
            pictures.append(img_resize)
        
        

        pictures[0].save(os.path.join(gif_path, seq + '.gif'),save_all=True, append_images=pictures[1:], optimize=True, loop=0)

        gifsicle(sources=os.path.join(gif_path, seq + '.gif'), destination=os.path.join(gif_path, seq + '.gif') ,optimize=False,colors=256,options=["--optimize=3"]
)

