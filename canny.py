import cv2
import os
import glob
from tqdm import tqdm

mode = 'input'
exp_name = '20231124_STDAN_Stack_BSD_3ms24ms_ckpt-epoch-0455'
dataset_path = '../dataset/BSD_3ms24ms'

if mode == 'input':
    file_type_list = ['Blur', 'Sharp']
    path = dataset_path
elif mode == 'output':
    file_type_list = ['output']
    path = '../STDAN_modified/exp_log/test/' + exp_name


for file_type in file_type_list:



    if mode == 'input':
        seq_list = sorted(os.listdir(os.path.join(path, 'test')))
    else:
        seq_list = sorted(os.listdir(os.path.join(path, file_type)))


    for seq in tqdm(seq_list):
        
        if mode == 'input':
            output_path = os.path.join(path, 'canny', file_type, seq)
        else:
            output_path = os.path.join(path, 'canny', seq)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if mode == 'input':
            file_list = sorted(glob.glob(os.path.join(path, 'test', seq, file_type, 'RGB', '*.png')))            
        else:
            file_list = sorted(glob.glob(os.path.join(path, file_type, seq, '*.png')))

        for file in file_list:
            basename = os.path.splitext(os.path.basename(file))[0]
            img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(img, 200, 200)

            cv2.imwrite(os.path.join(output_path, basename + '.png'), edge)
