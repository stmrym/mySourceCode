from collections import OrderedDict
import cv2
import importlib
import json
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import yaml

from metrics_pyiqa.utils.tensor_util import filepathlist2tensor, filepath2tensor



def check_save_path(save_name):
    if not isinstance(save_name, Path):
        save_name = Path(save_name)
    save_parent = save_name.parent
    assert save_parent.exists() and save_parent.is_dir(), f'{save_parent} dir does not exist'
    assert not save_name.exists(), f'{save_name} already exists'


def build_metrics(yaml_opt, device):
    metric_dict = OrderedDict()
    for name, params in yaml_opt.items():
        module = importlib.import_module(f'metrics_pyiqa.{name}')
        metric = getattr(module, name)
        metric_dict[name] = metric(device=device, **params) if params is not None else metric(device=device)

    print(f'metric = {metric_dict.keys()}')
    return metric_dict


def zip_seq_frame_dict(recons_base_path, lq_base_path, gt_base_path, specific_seq):

    seq_frame_path_dict = OrderedDict()

    seq_path_l = get_seq_path(recons_base_path, specific_seq)
    for seq_path in seq_path_l[0:2]:
        seq = seq_path.name
        recons_path_l = sorted(list((seq_path).rglob('*.png')))
        gt_path_l = [Path(gt_base_path) / seq / recons_path.name for recons_path in recons_path_l]
        lq_path_l = [Path(lq_base_path) / seq / recons_path.name for recons_path in recons_path_l]

        seq_frame_path_dict[seq] = (recons_path_l, gt_path_l, lq_path_l)
    
    return seq_frame_path_dict


def get_seq_path(img_base_path, specific_seq=''):
    if specific_seq == '':
        img_seq_path_l = sorted([f for f in Path(img_base_path).iterdir() if f.is_dir()])
    else:
        img_seq_path_l = [Path(img_base_path) / specific_seq]
    return img_seq_path_l


if __name__ == '__main__':
    
    # for input
    # img1_base_path = '../dataset/BSD_1ms8ms_comp/test/blur'
    # img2_base_path = '../dataset/BSD_1ms8ms/test/GT'
    # save_name = '../dataset/BSD_1ms8ms_comp/input_brisque.json'

    lq_base_path = '../dataset/Mi11Lite/test'
    gt_base_path = '../dataset/Mi11Lite/test'
    recons_base_path = '../STDAN_modified/exp_log/train/2024-12-09T194830__VT3__ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/epoch-0300_Mi11Lite_output/'

    save_name = 'input_pyiqa_NIQE_LR.json'
    
    # for test output
    # img1_base_path = '../STDAN_modified/exp_log/test/2024-11-12T073940__CT_/epoch-0400_BSD_1ms8ms_output/'
    # img2_base_path = '../dataset/BSD_1ms8ms/test/GT'
    # save_name = '../STDAN_modified/exp_log/test/2024-11-12T073940__CT_/epoch-0400_BSD_1ms8ms_output_brisque.json'

    # img1_base_path = '../STDAN_modified/exp_log/train/2024-12-09T194830__VT3__ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/epoch-0300_Mi11Lite_output/'
    # img2_base_path = '../dataset/Mi11Lite/test'
    # save_name = 'VT3_e0300.json'

    # for train output
    # img1_base_path = '../STDAN_modified/exp_log/train/2024-11-12T120452_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/epoch-0400_BSD_2ms16ms_output/'
    # img2_base_path = '../dataset/BSD_2ms16ms/test/GT'
    # save_name = '../STDAN_modified/exp_log/train/2024-11-12T120452_ESTDAN_v3_BSD_3ms24ms_GOPRO/epoch-0400_BSD_2ms16ms_output.json'
    
    # specific_seq = 'VID_20240523_165150'
    specific_seq = ''

    device = 'cuda:0'

    check_save_path(save_name)
    print(f'lq_base_path = {lq_base_path}')
    print(f'gt_base_path = {gt_base_path}')
    print(f'recons_base_path = {gt_base_path}')
    print(f'save_name = {save_name}')

    with open('calc_metrics_iqa_json.yml', mode='r') as f:
        opt = yaml.safe_load(f)    
    metric_dict = build_metrics(opt, device)
    results_dict = OrderedDict()

    zipped_path_dict = zip_seq_frame_dict(recons_base_path, lq_base_path, gt_base_path, specific_seq)

    for seq, (recons_path_l, gt_path_l, lq_path_l) in zipped_path_dict.items():
        with tqdm(lq_path_l) as lq_path_l_pbar:
            for lq_path, gt_path, recons_path in zip(lq_path_l_pbar, gt_path_l, recons_path_l):
                lq_path_l_pbar.set_description(f'[seq]={lq_path.parent}, [frame]={lq_path.name}')
                
                # load tensor from path
                lq_tensor = filepath2tensor(lq_path, device)
                gt_tensor = filepath2tensor(gt_path, device)
                recons_tensor = filepath2tensor(recons_path, device)

                

                # calc_metrics
                for name, metric in metric_dict.items():
                    result = metric.calculate(recons=recons_tensor, gt=gt_tensor, lq=lq_tensor)
                    results_dict.setdefault(name, {}).setdefault(seq, {})[lq_path.name] = result

                # Average for each seq
                for name in metric_dict.keys():
                    results_dict[name].setdefault('Average', {})[seq] = np.mean(list(results_dict[name][seq].values()))
    
    # Average for each seq
    for name in metric_dict.keys():
        results_dict[name]['TotalAverage'] = np.mean(list(results_dict[name]['Average'].values()))

    with open(save_name, 'w') as f:
        json.dump(results_dict, f, indent=4, sort_keys=True)

            
        
