import argparse
import cv2
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mmflow.datasets import visualize_flow

from argparse import ArgumentParser
from typing import Sequence

import cv2
import numpy as np
from numpy import ndarray

from mmflow.apis import inference_model, init_model

mode = 'pwcnet' # 'npy' or 'pwcnet'
base_dir = '../STDAN_modified/exp_log/test/20231129_STDAN_Stack_real_night_blur_ckpt-epoch-0400'

if mode == 'npy':
    flow_dir = os.path.join(base_dir, 'flow_npy')
    save_path = os.path.join(base_dir, 'mmflow_npy')
    seq_list = [f for f in os.listdir(flow_dir) if os.path.isdir(os.path.join(flow_dir, f))]
elif mode == 'pwcnet':
    img_dir = os.path.join(base_dir, 'output')
    save_path = os.path.join(base_dir, 'mmflow_pwc')
    seq_list = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
    seq_list = ['003']
    
    model = init_model(
        config='../STDAN_modified/mmflow/configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py',
        checkpoint='../STDAN_modified/mmflow/checkpoints/pwcnet_ft_4x1_300k_sintel_final_384x768.pth',
        device='cuda:0')



for seq in seq_list:

    if mode == 'npy':
        npy_files = sorted(glob.glob(os.path.join(flow_dir, seq, '*.npy')))
        for npy_file in npy_files:
            flow_npy = np.load(npy_file)
            
            basename = os.path.splitext(os.path.basename(npy_file))[0]

            flow_map = visualize_flow(flow_npy, None)
            # visualize_flow return flow map with RGB order
            flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)

            os.makedirs(os.path.join(save_path, seq), exist_ok=True)

            cv2.imwrite(os.path.join(save_path, seq, basename + '.png'), flow_map)
            print(f'saved {os.path.join(save_path, seq, basename + ".png")}')


    elif mode == 'pwcnet':
        img_files = sorted(glob.glob(os.path.join(img_dir, seq, '*.png')))
        for i in range(0, len(img_files) - 1):
            img1 = cv2.imread(img_files[i])
            img2 = cv2.imread(img_files[i + 1])
            # estimate flow
            result = inference_model(model, img1, img2)

            flow_map = visualize_flow(result, None)
            # visualize_flow return flow map with RGB order
            flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
            
            basename = os.path.splitext(os.path.basename(img_files[i]))[0]

            os.makedirs(os.path.join(save_path, seq), exist_ok=True)

            cv2.imwrite(os.path.join(save_path, seq, basename + '.png'), flow_map)
            print(f'saved {os.path.join(save_path, seq, basename + ".png")}')


