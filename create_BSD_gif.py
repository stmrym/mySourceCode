from PIL import Image
import os
import glob
from tqdm import tqdm

path = '/mnt/d/dataset/BSD_3ms24ms'
file_type = 'valid'

gif_path = os.path.join(path, 'gif')

if not os.path.exists(gif_path):
    os.makedirs(gif_path)
    os.makedirs(os.path.join(gif_path, 'Blur'))
    os.makedirs(os.path.join(gif_path, 'Sharp'))


seq_list = sorted(os.listdir(os.path.join(path, file_type)))

for seq in tqdm(seq_list):

    pictures = []

    for input_type in ['Blur', 'Sharp']:
        file_list = sorted(glob.glob(os.path.join(path, file_type, seq, input_type, 'RGB', '*.png')))

        for file in file_list:
            img = Image.open(file)
            pictures.append(img)

        if pictures == []:
            print(os.path.join(path, file_type, seq, input_type, 'RGB'))
            print(file_list)
            print('picture empty')
            exit()
        pictures[0].save(os.path.join(gif_path, input_type, file_type + '_' + seq + '.gif'),save_all=True, append_images=pictures[1:], optimize=True, loop=0)

