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

parser.add_argument('--output_path', required = True, help="e.g., ./exp_log/WO_Motion_small_2024-02-08T161225_STDAN_Stack_BSD_3ms24ms_GOPRO/visualization/epoch-0200")
parser.add_argument('--save_dir', required = True, help="e.g., ./exp_log/20231129_STDAN_Stack_BSD_3ms24ms_ckpt-epoch-0905")
parser.add_argument('--gt_path', required = True, nargs='+', help="e.g., ../../dataset/BSD3ms24ms/test ../../dataset/GOPRO_Large/test")

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
    def __init__(self, file_paths):
        self.files = file_paths
        self.psnr_list = []
        self.ssim_list = []
        self.ssim_LPF_list = []
        self.filter_size = 3
        self.resize_ratio = 2
    
    def road_ith_img_resize(self, i):   # downsampling x2 for EDVR results
        self.image = cv2.cvtColor(cv2.imread(self.files[i]), cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.image.shape
        self.image = cv2.resize(self.image, (self.w // self.resize_ratio, self.h // self.resize_ratio), interpolation=cv2.INTER_AREA)   

    def load_img(self, img_path):  # for Single Image Reflection Removal (SIRR) results (Zhang et al.)
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

    def append_ssim(self, GT_image):
        self.ssim_list.append(ssim(GT_image, self.image))

    def append_psnr(self, GT_image):
        self.psnr_list.append(cv2.PSNR(GT_image, self.image))
    def calc_LPF(self):
        self.ssim_LPF_list = valid_convolve(self.ssim_list, self.filter_size)



seq_list = [f for f in sorted(os.listdir(args.output_path)) if os.path.isdir(os.path.join(args.output_path, f))]

for seq in seq_list:
        
    print(seq)

    output_frame_list = [f.split('/')[-1] for f in sorted(glob.glob(os.path.join(args.output_path, seq, '*.png')))]

    for gt_path in args.gt_path:
        #  GOPRO
        if os.path.isdir(os.path.join(gt_path, seq, 'sharp')):
            gt_frame_list = [os.path.join(gt_path, seq, 'sharp', output_frame) for output_frame in output_frame_list]

        # BSD_3ms24ms
        elif os.path.isdir(os.path.join(gt_path, seq, 'Sharp', 'RGB')):
            gt_frame_list = [os.path.join(gt_path, seq, 'Sharp', 'RGB', output_frame) for output_frame in output_frame_list]


    output = Calc_SSIM(file_paths = sorted(glob.glob(os.path.join(args.output_path, seq, '*.png'))))
    gt = Calc_SSIM(file_paths = gt_frame_list)

    assert len(output.files) != 0, f'len(output)={len(output.files)}'
    assert len(gt.files) != 0, f'len(gt)={len(gt.files)}'
    assert len(gt.files) == len(output.files), f"len(gt)={len(gt.files)}, len(output)={len(output.files)} don't match"

    for gt_file, output_file in tqdm(zip(gt.files, output.files)):
        
        assert os.path.basename(gt_file) == os.path.basename(output_file), f"basenames gt_file={os.path.basename(gt_file)} don't match"
        
        gt.load_img(gt_file)
        output.load_img(output_file)
        output.append_psnr(gt.image)
        output.append_ssim(gt.image)

        output.calc_LPF()
    frame = [os.path.splitext(os.path.basename(f))[0] for f in gt.files]

    df = pd.DataFrame(
        data={  'frame'             :   frame,
                'output_PSNR'       :   output.psnr_list,
                'output'            :   output.ssim_list,
                'output_LPF'        :   output.ssim_LPF_list,
                }
    )

    save_path = os.path.join(args.save_dir,'SSIM_csv')
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    df.to_csv(os.path.join(save_path, seq + '.csv'), index=False) # save to .csv
    print(f'saved {os.path.join(save_path, seq + ".csv")}')

