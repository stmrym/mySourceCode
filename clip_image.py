import os
import cv2
import numpy as np
from typing import List


def clip_image() -> None:

    path = '/mnt/d/results/SIP_202408/presentation_fig/000019_b.png'
    save_path = '/mnt/d/results/SIP_202408/presentation_fig/000019_b_clip.png'

    img = cv2.imread(path)
    h, w, _ = img.shape
    cut_size = int(640/480 * np.minimum(h, w))

    clipped_image = img[:,(w-cut_size)//2:(w-cut_size)//2 + cut_size,:]

    print(clipped_image.shape)
    cv2.imwrite(save_path, clipped_image)
    

def add_patch_inside(path: str, save_name: str, thick_size: int, scale_list: List[float], loc_list: List[str], xywh_list: List[tuple], color_list: List[tuple], **kwargs) -> None:
    img = cv2.imread(path)
    assert img is not None, 'image empty.'
    img_h, img_w, _ = img.shape

    
    for (x, y, w, h), color, loc, scale in zip(xywh_list, color_list, loc_list, scale_list):

        clip = cv2.resize(img[y:y+h, x:x+w], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # draw line
        cv2.rectangle(clip, (1, 1), (clip.shape[1]-2, clip.shape[0]-2), color=color, thickness=thick_size)
        cv2.rectangle(img, (x, y, w, h), color=color, thickness=thick_size)
        clip_h, clip_w, _ = clip.shape

        if loc == 'ul':
            img[:clip_h, :clip_w] = clip
        elif loc == 'ur':
            img[:clip_h:, -clip_w:] = clip
        elif loc == 'bl':
            img[-clip_h:, :clip_w] = clip
        elif loc == 'br':
            img[-clip_h:, -clip_w:] = clip
        else:
            print('invalid loc')
            exit()

    cv2.imwrite(save_name, img)



def add_patch_outside(path: str, save_name: str, thick_size: int, xywh_list: List[tuple], color_list: List[tuple], **kwargs) -> None:

    img = cv2.imread(path)
    assert img is not None, 'image empty.'
    img_h, img_w, _ = img.shape

    # calculate resize ratio
    w_list = [xyhw[2] for xyhw in xywh_list]
    denom = 0
    for i in range(len(w_list)):
        # h1*h2*...*wi*...*hn
        denom += w_list[i] * np.prod([xyhw[3] for j, xyhw in enumerate(xywh_list) if j != i])

    bottom = []
    # creating patches
    for i, ((x, y, w, h), color) in enumerate(zip(xywh_list, color_list)):
        k = img_w * np.prod([xyhw[3] for j, xyhw in enumerate(xywh_list) if j != i]) / denom
        # clipping
        clip = cv2.resize(img[y:y+h, x:x+w], dsize=None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)

        # draw line
        cv2.rectangle(clip, (1, 1), (clip.shape[1]-2, clip.shape[0]-2), color=color, thickness=thick_size)
        cv2.rectangle(img, (x, y, w, h), color=color, thickness=thick_size)
        bottom.append(clip)
        print(clip.shape)

    stack = np.hstack(bottom)
    out = np.vstack((img, stack))
    cv2.imwrite(save_name, out)



if __name__ == '__main__':

    name = '00000143_e_woLf'
    kwargs = {
        'path' : '/mnt/d/results/SIP_202408/result_src/%s.png' % name,
        'save_name' : '/mnt/d/results/SIP_202408/result_fig/%s.png' % name,
        'xywh_list' : [
            # 00000143
            (340, 270, 100, 60),
            # 003012
            # (1000, 60, 150, 140),
            # 00000020
            # (255, 200, 170, 90),
            # (470, 260, 140, 140),
        ],
        'color_list': [
            (0,255,0),
            (0,0,255),
            (255,0,0)
        ],
        'thick_size' : 4,
        'scale_list' : [3.5,],
        'loc_list': ['ur',]
    }

    # add_patch_outside(**kwargs)
    add_patch_inside(**kwargs)
    # clip_image()