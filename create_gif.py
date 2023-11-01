from PIL import Image
import os
import glob
from tqdm import tqdm

# path = '/home/moriyamasota/STDAN/exp_log/test/STDAN_Stack_REDS_ckpt-epoch-0400'
path = '/home/moriyamasota/datasets/REDS_flip'
file_type = 'input'

gif_path = os.path.join(path, 'gif')

if not os.path.exists(gif_path):
    os.makedirs(gif_path)

if file_type in ['input', 'GT']:
    seq_list = sorted(os.listdir(os.path.join(path, 'test')))
else:
    seq_list = sorted(os.listdir(os.path.join(path, 'output')))

for seq in tqdm(seq_list):

    pictures = []

    if file_type in ['input', 'GT']:
        # print(os.path.join(path, 'test', seq, file_type, '*.png'))
        file_list = sorted(glob.glob(os.path.join(path, 'test', seq, file_type, '*.png')))
        file_list = file_list[2:-2]
    else:
        file_list = sorted(glob.glob(os.path.join(path, 'output', seq, '*.png')))

    for file in file_list:
        img = Image.open(file).quantize()
        pictures.append(img)

    pictures[0].save(os.path.join(gif_path, seq + '.gif'),save_all=True, append_images=pictures[1:], optimize=True, loop=0)

