import cv2
import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from stdan.data_transforms import *


def build_transform(yaml_opt):
    transform_l = []
    for name, params in yaml_opt.items():
        if name in globals():
            transform = globals()[name](**params)
        print(transform)
        transform_l.append(transform)
    return transform_l


def read_seqs(base_dir, seq=None):
    # Read all seqs
    if seq is None:
        seqs = sorted([f for f in Path(base_dir).iterdir() if f.is_dir()])
    # Read specific seq
    else:
        seqs = [Path(base_dir) / seq]

    return seqs


def read_file_path(seq, file_name=None):
    # Read all frames
    if file_name is None:
        frame_path_l = sorted(list((seq).rglob('*.png')))
        os.makedirs(Path(dst_dir) / seq.name, exist_ok=True)
    # Read specific frame
    else:
        frame_path_l = [seq / file_name]

    return frame_path_l        


def read_images(frame_path_l):
    imgs = []
    for frame_path in frame_path_l:
        print(frame_path)
        img = (cv2.imread(str(frame_path)) / 255.0).astype(np.float32)
        imgs.append(img)
    return imgs

def save_images(dst_dir, t_imgs):
    if len(t_imgs) > 1:    
        save_path_l = [Path(dst_dir) / seq.name / f.name for f in frame_path_l]
    else:
        save_path_l = [Path(seq.name + '_' + frame_path_l[0].name)]

    for t_img, save_path in zip(t_imgs, save_path_l):
        t_img_uint = np.clip((t_img*255), 0, 255).astype(np.uint8)
        cv2.imwrite(str(save_path), t_img_uint)

if __name__ == '__main__':
    
    # base_tmpl = '/mnt/d/dataset/BSD_2ms16ms/test'
    base_dir = '../dataset/BSD_2ms16ms/train/blur'
    dst_dir = '../dataset/BSD_2ms16ms_comp2/test/blur'
    
    seq = '067'
    file_name = '00000040.png'


    with open('create_val_transformed_dataset.yml', mode='r') as f:
        opt = yaml.safe_load(f)
    
    transform_l = build_transform(opt)
    seqs = read_seqs(base_dir, seq)

    for seq in seqs:
        # read frame paths
        frame_path_l = read_file_path(seq, file_name)
        # read images
        imgs_lq = read_images(frame_path_l)
        
        # apply transforms
        for t in tqdm(transform_l, desc=str(seq.name)):
            imgs_lq, _ = t(imgs_lq, imgs_lq)
        
        # save transformed images
        save_images(dst_dir, imgs_lq)





