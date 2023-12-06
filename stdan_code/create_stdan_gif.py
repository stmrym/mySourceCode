from PIL import Image
import argparse
import os
import glob
from tqdm import tqdm
from pygifsicle import gifsicle

parser = argparse.ArgumentParser(description='create STDAN output gif file.')

parser.add_argument('mode', help="'seg', 'output'")
parser.add_argument('exp_name', help="'STDAN_modified/exp_log/test/<exp_name>'")

args = parser.parse_args()

mode = args.mode
exp_name = args.exp_name

if mode in ['output', 'seg']:
    path = '../../STDAN_modified/exp_log/test/' + exp_name

if mode == 'output':
    file_type_list = ['output', 'flow_forward', 'flow_hsv']
elif mode == 'seg':    
    file_type_list = ['seg', 'blended']


for file_type in file_type_list:

    gif_path = os.path.join(path, 'gif', file_type)

    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    seq_list = sorted(os.listdir(os.path.join(path, file_type)))

    for seq in tqdm(seq_list):

        pictures = []

        file_list = sorted(glob.glob(os.path.join(path, file_type, seq, '*.png')))

        for file in file_list:
            img = Image.open(file).quantize()
            
            if file_type in ['flow_forward', 'flow_hsv']:
                resize_w = img.width//4
                resize_h = img.height//4
            else:
                resize_w = img.width//2
                resize_h = img.height//2

            img_resize = img.resize((resize_w, resize_h))
            pictures.append(img_resize)

        pictures[0].save(os.path.join(gif_path, seq + '.gif'),save_all=True, append_images=pictures[1:], optimize=True, loop=0)

        gifsicle(sources=os.path.join(gif_path, seq + '.gif'), destination=os.path.join(gif_path, seq + '.gif') ,optimize=False,colors=256,options=["--optimize=3"])
        print(f'saved {os.path.join(gif_path, seq + ".gif")}')

