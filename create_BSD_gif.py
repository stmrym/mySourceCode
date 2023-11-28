from PIL import Image
import os
import glob
from tqdm import tqdm

# path = '/mnt/d/dataset/BSD_3ms24ms'
path = '../dataset/BSD_3ms24ms'
file_type = 'test'

gif_path = os.path.join(path, 'gif', file_type)

if not os.path.exists(gif_path):
    os.makedirs(os.path.join(gif_path, 'Blur'), exist_ok=True)
    os.makedirs(os.path.join(gif_path, 'Sharp'), exist_ok=True)


seq_list = sorted(os.listdir(os.path.join(path, file_type)))

for seq in tqdm(seq_list):

    pictures = []

    for input_type in ['Blur', 'Sharp']:
        file_list = sorted(glob.glob(os.path.join(path, file_type, seq, input_type, 'RGB', '*.png')))
        file_list = file_list[2:-2]

        for file in file_list:
            img = Image.open(file)
            resize_w = img.width // 2
            resize_h = img.height // 2
            img = img.resize((resize_w, resize_h))
            pictures.append(img)


        if pictures == []:
            print(os.path.join(path, file_type, seq, input_type, 'RGB'))
            print(file_list)
            print('picture empty')
            exit()
        pictures[0].save(os.path.join(gif_path, input_type, seq + '.gif'),save_all=True, append_images=pictures[1:], optimize=True, loop=0)

