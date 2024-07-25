import numpy as np
import glob
import os
import math
from skimage.util.arraycrop import crop
from skimage.metrics import structural_similarity as ssim
import cv2
import pandas as pd
from tqdm import tqdm
import lpips
import torch
from typing import List
from matplotlib import cm
from graph_util import plot_heatmap, cv2_heatmap

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

def gray2cmap_numpy(image_np: np.ndarray, cmap_name: str = 'bwr') -> np.ndarray:
    colormap = cm.get_cmap(cmap_name, 256)
    # (H, W) -> (H, W, 3)
    converted_array = colormap(image_np)[:,:,0:3]
    return converted_array


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


class Metric_dataframe():
    def __init__(self, metric_types, seq_dict, rgb_range=1.0, pixel_max=1, shave=4):
        self.metric_types = metric_types
        self.seq_dict = seq_dict
        self.rgb_range = rgb_range
        self.pixel_max = pixel_max
        self.shave = shave
        if 'LPIPS' in self.metric_types:
            self.loss_fn_alex = lpips.LPIPS(net='alex')
        
        # avg_df initialize
        self.avg_df = pd.DataFrame(index = self.seq_dict.keys())        
        for metric_type in self.metric_types:
            self.avg_df['avg' + metric_type] = 0.0
            self.avg_df['sd' + metric_type] = 0.0

    def load_flow_npz(self, flow_npz_path):
        self._flow_npz = np.load(flow_npz_path)    

    def preprocess_each_seq(self, seq, **kwargs):
        # initialize values_dict
        # {'PSNR': [], 'SSIM':[], ...}
        self.values_dict = dict(zip(self.metric_types, [np.zeros([0]) for _ in range(len(self.metric_types))]))

        if 'masked_SSIM' in self.metric_types:
            for npz_path in kwargs['npz_path_list']:
                self.npz_path = npz_path % seq
                if os.path.isfile(self.npz_path):
                    break
            
            assert os.path.isfile(self.npz_path) == True, 'flow_npz does not exist'
            self.load_flow_npz(self.npz_path)

    def calc_metric(self, metric_name, output_image, gt_image, **kwargs):
        if metric_name == 'PSNR':
            self._img1 = output_image[self.shave:-self.shave, self.shave:-self.shave, :]/255.0
            self._img2 = gt_image[self.shave:-self.shave, self.shave:-self.shave, :]/255.0
            self._mse = np.mean((self._img1/self.rgb_range - self._img2/self.rgb_range)**2)
            self._value = 20 * np.log10(self.pixel_max / np.sqrt(self._mse))
            
        elif metric_name == 'SSIM':
            gt = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            output = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            self._value = ssim(output, gt, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                        data_range=255.0)
        elif metric_name == 'LPIPS':
            output_tensor = torch.from_numpy(output_image.astype(np.float32)/255).clone()
            gt_tensor = torch.from_numpy(gt_image.astype(np.float32)/255).clone()
            d = self.loss_fn_alex(output_tensor.permute(2,0,1), gt_tensor.permute(2,0,1)).detach().numpy()
            self._value = d[0,0,0,0]
        elif metric_name == 'masked_SSIM':
            # generate SSIM map
            gt = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            output = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            _, self.ssim_map = ssim(output, gt, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                    data_range=255.0, gradient=False, full=True)
            
            # flow normalized
            self._flow = self._flow_npz[kwargs['basename']]
            H, W, _ = self._flow.shape
            M = kwargs['scale_k'] * np.minimum(H, W)
            self.flow_mag = np.sqrt(self._flow[:,:,0]**2 + self._flow[:,:,1]**2) / M
            self.flow_mag = np.minimum(self.flow_mag, 1)

            self.masked_ssim_map = self.flow_mag * self.ssim_map               

            # plot_heatmap(plot_data=ssim_map, save_name='../STDAN_modified/debug_results/' + str(self.count) + 'ssim.png', vmin=0, vmax=1, cmap='jet')
            # plot_heatmap(plot_data=self.flow_mag, save_name='../STDAN_modified/debug_results/' + str(self.count) + 'flow.png', vmin=0, vmax=1, cmap='jet')
            # plot_heatmap(plot_data=self.masked_ssim_map, save_name='../STDAN_modified/debug_results/' + str(self.count) + 'masked.png', vmin=0, vmax=1, cmap='jet')

            truncate = 3.5
            sigma = 1.5
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
            pad = (win_size - 1) // 2
            # compute (weighted) mean of ssim. Use float64 for accuracy.
            self._value = crop(self.masked_ssim_map, pad).mean(dtype=np.float64)

        elif metric_name == 'i_masked_SSIM':
            self.i_masked_ssim_map = (1 - self.flow_mag) * self.ssim_map    

            # plot_heatmap(plot_data=self.i_masked_ssim_map, save_name='../STDAN_modified/debug_results/' + str(self.count) + 'i_masked.png', vmin=0, vmax=1, cmap='jet')

            truncate = 3.5
            sigma = 1.5
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
            pad = (win_size - 1) // 2
            self._value = crop(self.i_masked_ssim_map, pad).mean(dtype=np.float64)
            
        return self._value
    
    def calc_metric_all(self, **kwargs):
        for metric_name in self.metric_types:
            self.value = self.calc_metric(metric_name=metric_name, **kwargs)
            self.values_dict[metric_name] = np.append(self.values_dict[metric_name], self.value)

    def postprocess_each_seq(self, seq, output_path_list, save_dir):
        # make df each seq and save
        self.frame = [os.path.splitext(os.path.basename(f))[0] for f in output_path_list]
        self.data = {'seq':seq, 'frame':self.frame}
        for metric_name in self.metric_types:  # make data
            self.data[metric_name] = self.values_dict[metric_name]
            self.avg_df.at[seq, 'avg' + metric_name] = self.values_dict[metric_name].mean()
            self.avg_df.at[seq, 'sd' + metric_name] = np.sqrt(self.values_dict[metric_name].var())

        self.df = pd.DataFrame(data=self.data)
        # save each dataframe
        self._save_path = os.path.join(save_dir,'metrics_csv')
        if not os.path.isdir(self._save_path):
            os.makedirs(self._save_path, exist_ok=True)
        self.df.to_csv(os.path.join(self._save_path, seq + '.csv'), index=False) # save to .csv

    def final_process(self):
        # save avg_df, stack_df
        for metric_name in self.metric_types:
            self.avg_df.at['Avg.', 'avg' + metric_name] = self.avg_df['avg' + metric_name].mean()
            self.avg_df.at['Avg.', 'sd' + metric_name] = np.sqrt(self.avg_df['avg' + metric_name].var())
        self.avg_df.to_csv(os.path.join(save_dir, 'avg_metrics.csv')) # save to .csv


def prepare_seq_dict(output_path: str, seq_select: str, gt_paths: List[str]) -> dict:

    if seq_select == 'all':
        seq_list = [f for f in sorted(os.listdir(output_path)) if (os.path.isdir(os.path.join(output_path, f)) and 'metrics_csv' not in f)]
    else:
        seq_list = [seq_select]
    
    seq_dict = {}

    for seq in seq_list:    # make output_path_list and gt_path_list         
        output_frame_list = [f.split('/')[-1] for f in sorted(glob.glob(os.path.join(output_path, seq, '*.png')))]
        output_path_list = sorted(glob.glob(os.path.join(output_path, seq, '*.png')))
        for gt_path in gt_paths:
            if os.path.isdir(os.path.join(gt_path, seq, 'sharp')): #  GOPRO
                gt_path_list = [os.path.join(gt_path, seq, 'sharp', output_frame) for output_frame in output_frame_list]
            elif os.path.isdir(os.path.join(gt_path, seq, 'Sharp', 'RGB')):  # BSD_3ms24ms
                gt_path_list = [os.path.join(gt_path, seq, 'Sharp', 'RGB', output_frame) for output_frame in output_frame_list]
        assert len(output_path_list) == len(gt_path_list), f'output {len(output_path_list)}, GT {len(gt_path_list)} do not match.'
        seq_dict[seq] = (output_path_list, gt_path_list)

    return seq_dict


def calc_metrics(seq_dict: dict, save_dir: str, metric_type_list: List[str], **kwargs) -> None:

    metrics = Metric_dataframe(metric_type_list, seq_dict)
    for seq, (output_path_list, gt_path_list) in seq_dict.items():
        
        # start processing each seq
        print(seq)
        metrics.preprocess_each_seq(seq, **kwargs)
        
        for output_path, gt_path in zip(tqdm(output_path_list), gt_path_list):
            assert os.path.basename(output_path) == os.path.basename(gt_path), f"basenames gt_file={os.path.basename(gt_path)} don't match"
            gt = cv2.imread(gt_path)
            output = cv2.imread(output_path)
            kwargs['output_image'] = output
            kwargs['gt_image'] = gt
            kwargs['basename'] = os.path.basename(gt_path)
            metrics.calc_metric_all(**kwargs)
        metrics.postprocess_each_seq(seq, output_path_list, save_dir)

    metrics.final_process()


def calc_ssim_map(seq_dict: str, save_dir: str, grayscale: bool = True, mode: str = 'img') -> None:
    # calc and output ssim_map and gradient of ssim_map
    for seq, (output_path_list, gt_path_list) in seq_dict.items():
        print(seq)
        for output_path, gt_path in zip(tqdm(output_path_list), gt_path_list):
            assert os.path.basename(output_path) == os.path.basename(gt_path), f"basenames gt_file={os.path.basename(gt_path)} don't match"

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            output = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            ssim_value, grad, ssim_map = ssim(output, gt, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                    data_range=255.0, gradient=True, full=True)

            save_map_path = os.path.join(save_dir, 'map_' + seq)
            if not os.path.isdir(save_map_path):
                os.makedirs(save_map_path, exist_ok=True)
            save_name = os.path.join(save_map_path, os.path.basename(output_path))

            if mode == 'fig':
                plot_heatmap(plot_data=ssim_map, save_name=save_name, vmin=0, vmax=1, cmap='jet')
            elif mode == 'img':
                cv2_heatmap(ssim_map, save_name, cmap=cv2.COLORMAP_JET)
            


if __name__ == '__main__':
    
    ##########
    # exp = '2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO'
    # exp = '2024-05-23T102821_ESTDAN_v2_BSD_3ms24ms_GOPRO'
    # exp = 'F_2024-05-29T122237_STDAN_BSD_3ms24ms_GOPRO'
    # exp = '2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO'
    exp = '2024-07-17T094452__STDAN_BSD_3ms24ms'
    # metric_type_list = ['PSNR', 'SSIM', 'LPIPS']
    # metric_type_list = ['SSIM']
    metric_type_list = ['SSIM', 'masked_SSIM', 'i_masked_SSIM']
    output_path = '../STDAN_modified/exp_log/train/%s/visualization/epoch-1200_output' % exp
    seq_select = 'all'
    save_dir = '../STDAN_modified/exp_log/train/%s/visualization/c1200_out_ssim' % exp
    gt_paths = ['../dataset/GOPRO_Large/test', '../dataset/BSD_3ms24ms/test']
    #########

    kwargs = {
        'npz_path_list' : ['../dataset/GOPRO_Large/flow_sharp/%s.npz', '../dataset/BSD_3ms24ms/flow_sharp/%s.npz'],
        'scale_k' : 0.10,
        'mode' : 'img'
    }

    # 2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO
    # 2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO

    seq_dict = prepare_seq_dict(output_path, seq_select, gt_paths)

    calc_metrics(seq_dict, save_dir, metric_type_list,**kwargs)


    