import cv2
import json
import os
import sys
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from stdan import metrics

def check_save_path(save_name):
    if not isinstance(save_name, Path):
        save_name = Path(save_name)
    save_parent = save_name.parent
    assert save_parent.exists() and save_parent.is_dir(), f'{save_parent} dir does not exist'
    print(save_name.exists(), save_name.is_file())
    assert not save_name.exists(), f'{save_name} already exists'

def build_metrics(yaml_opt):
    metric_dict = OrderedDict()
    for name, params in yaml_opt.items():
        metric = getattr(metrics, name)
        metric_dict[name] = metric(**params) if params is not None else metric()
    print(metric_dict)
    return metric_dict


if __name__ == '__main__':
    
    # img1_base_path = '../BSSTNet/dataset/BSD_1ms8ms/test/blur'
    # img2_base_path = '../BSSTNet/datasets/BSD_1ms8ms/test/GT'
    # save_name = '../BSSTNet/datasets/BSD_1ms8ms/input.json'

    img1_base_path = '../STDAN_modified/exp_log/test/2024-11-12T073940__CT_/epoch-0400_Mi11Lite_output/'
    img2_base_path = '../dataset/Mi11Lite/test'
    save_name = '../STDAN_modified/exp_log/test/2024-11-12T073940__CT_/epoch-0400_Mi11Lite_output.json'

    # img1_base_path = '../dataset/BSD_1ms8ms_comp/test/blur'
    # img2_base_path = '../dataset/BSD_1ms8ms/test/GT'
    # save_name = '../dataset/BSD_1ms8ms_comp/input.json'

    # img1_base_path = '../dataset/Mi11Lite/test'
    # img2_base_path = '../dataset/Mi11Lite/test'
    # save_name = '../dataset/Mi11Lite/input.json'

    check_save_path(save_name)

    with open('calc_metrics_json.yml', mode='r') as f:
        opt = yaml.safe_load(f)    
    metric_dict = build_metrics(opt)

    results_dict = OrderedDict()

    img1_seq_path_l = sorted([f for f in Path(img1_base_path).iterdir() if f.is_dir()])
    for img1_seq_path in img1_seq_path_l:
        seq = img1_seq_path.name
        img1_path_l = sorted(list((img1_seq_path).rglob('*.png')))

        for img1_path in tqdm(img1_path_l, desc=f'{seq}'):    
            # read images
            img2_path = Path(img2_base_path) / seq / img1_path.name

            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            # calc_metrics
            for name, metric in metric_dict.items():
                result = metric.calculate(img1=img1, img2=img2)
                results_dict.setdefault(name, {}).setdefault(seq, {})[img1_path.name] = result

        # Average for each seq
        for name in metric_dict.keys():
            results_dict[name].setdefault('Average', {})[seq] = np.mean(list(results_dict[name][seq].values()))
    
    # Average for each seq
    for name in metric_dict.keys():
        results_dict[name]['TotalAverage'] = np.mean(list(results_dict[name]['Average'].values()))

    with open(save_name, 'w') as f:
        json.dump(results_dict, f, indent=4, sort_keys=True)

            
        
