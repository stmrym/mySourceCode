import cv2
import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from stdan.data_transforms import *


def build_transforms(yaml_opt):
    transform_l = []
    for name, params in yaml_opt.items():
        if name in globals():
            transform = globals()[name](**params)
        print(transform)
        transform_l.append(transform)
    return transform_l


if __name__ == '__main__':
    
    base_tmpl = '../dataset/BSD_2ms16ms_original/test'
    file_tmpl = 'Blur/RGB'
    dst_dir = '../dataset/BSD_2ms16ms_comp/test/blur'


    with open('create_val_transformed_dataset.yml', mode='r') as f:
        opt = yaml.safe_load(f)
    
    transform_l = build_transforms(opt)

    base_dir = base_tmpl.split('%s')[0]
    seqs = sorted([f for f in Path(base_dir).iterdir() if f.is_dir()])
    for seq in seqs:
        frame_path_l = sorted(list((seq / file_tmpl).rglob('*.png')))
        os.makedirs(Path(dst_dir) / seq.name, exist_ok=True)

        # read images
        imgs = []
        for frame_path in frame_path_l:
            img = (cv2.imread(str(frame_path)) / 255.0).astype(np.float32)
            imgs.append(img)
        
        # apply transforms
        for t in tqdm(transform_l, desc=str(seq.name)):
            t_imgs, _ = t(imgs, imgs)
        
        # save transformed images
        save_path_l = [Path(dst_dir) / seq.name / f.name for f in frame_path_l]
        for t_img, save_path in zip(t_imgs, save_path_l):
            t_img_uint = np.clip((t_img*255), 0, 255).astype(np.uint8)
            cv2.imwrite(str(save_path), t_img_uint)





