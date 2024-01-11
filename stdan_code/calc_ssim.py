import numpy as np
import argparse
import glob
import os
import math
from skimage.metrics import structural_similarity as ssim
import cv2
import pandas as pd
from tqdm import tqdm

'''
calculating SSIMs of all test video sequences

[Input]: Result Images

 230213_release_V2/
  ├ BasicSR/
  │  └ results/ 
  │     └ [Result Images]
  └ SSIM_graph/


[Output]: SSIM .csv files of each video sequence  (./SSIM_csv/xxx.csv)
'''

parser = argparse.ArgumentParser(description='make ssim.csv file from test results.')

parser.add_argument('exp_name', help="e.g., 20231129_STDAN_Stack_BSD_3ms24ms_ckpt-epoch-0905")
parser.add_argument('--gt_path', help="e.g., ../../dataset/BSD3ms24ms/test", default='../../dataset/BSD_3ms24ms/test')

args = parser.parse_args()


# calculate convolution (for LPF)
def valid_convolve(xx, size):
    b = np.ones(size)/size
    xx_mean = np.convolve(xx, b, mode="same")

    n_conv = math.ceil(size/2)

    xx_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= size/(i+n_conv)
        xx_mean[-i] *= size/(i + n_conv - (size % 2))

    return xx_mean


class Calc_SSIM():
    def __init__(self, fdir):
        self.fdir = fdir
        self.files = sorted(glob.glob(os.path.join(self.fdir, '*.png')))
        self.ssim_list = []
        self.ssim_LPF_list = []
        self.filter_size = 3
        self.resize_ratio = 2
    
    def road_ith_img_resize(self, i):   # downsampling x2 for EDVR results
        self.image = cv2.cvtColor(cv2.imread(self.files[i]), cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.image.shape
        self.image = cv2.resize(self.image, (self.w // self.resize_ratio, self.h // self.resize_ratio), interpolation=cv2.INTER_AREA)   

    def road_img(self, img_path):  # for Single Image Reflection Removal (SIRR) results (Zhang et al.)
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

    def append_ssim(self, GT_image):
        self.ssim_list.append(ssim(GT_image, self.image))

    def calc_LPF(self):
        self.ssim_LPF_list = valid_convolve(self.ssim_list, self.filter_size)

exp_name = args.exp_name
gt_path = args.gt_path
base_path = os.path.join('..', '..', 'STDAN_modified', exp_name)
output_path = os.path.join(base_path, 'output')

seq_list = [f for f in sorted(os.listdir(output_path)) if os.path.isdir(os.path.join(output_path, f))]

for seq in seq_list:
        
    print(seq)
    gt = Calc_SSIM(fdir = os.path.join(gt_path, seq, 'Sharp', 'RGB'))
    output = Calc_SSIM(fdir = os.path.join(output_path, seq))
    gt.files = gt.files[2:-2]

    assert len(gt.files) == len(output.files), f"len(gt)={len(gt.files)}, len(output)={len(output.files)} don't match"

    for gt_file, output_file in tqdm(zip(gt.files, output.files)):
        
        assert os.path.basename(gt_file) == os.path.basename(output_file), f"basenames gt_file={os.path.basename(gt_file)} don't match"
        
        gt.road_img(gt_file)
        output.road_img(output_file)
        output.append_ssim(gt.image)

        output.calc_LPF()

    frame = [os.path.splitext(os.path.basename(f))[0] for f in gt.files]

    df = pd.DataFrame(
        data={  'frame'             :   frame,
                'output'            :   output.ssim_list,
                'output_LPF'        :   output.ssim_LPF_list,
                }
    )

    save_path = os.path.join(base_path,'SSIM_csv')
    # save_path = os.path.join(os.environ['HOME'], 'STDAN', 'exp_log', 'test', exp_name,'SSIM_csv')
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    df.to_csv(os.path.join(save_path, seq + '.csv'), index=False) # save to .csv
    print(f'saved {os.path.join(save_path, seq + ".csv")}')

