import os
import cv2
import numpy as np
from typing import List


def clip_center_image() -> None:

    path = '/mnt/d/results/20241210/00000040_blur.png'
    save_path = '/mnt/d/results/20241210/00000040_blur_clip_unsharp.png'

    img = cv2.imread(path)
    h, w, _ = img.shape
    cut_size = int(640/480 * np.minimum(h, w))

    clipped_image = img[:,(w-cut_size)//2:(w-cut_size)//2 + cut_size,:]

    print(clipped_image.shape)
    cv2.imwrite(save_path, clipped_image)




def clip_image(path: str, save_name: str, scale_list: List[float], xywh_list: List[tuple], **kwargs) -> None:

    img = cv2.imread(path)

    x, y, w, h = xywh_list[0]
    scale = scale_list[0]

    clipped = cv2.resize(img[y:y+h, x:x+w], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    print(clipped.shape)
    cv2.imwrite(save_name, clipped)
    

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



def add_patch_outside(path: str, save_name: str, thick_size: int, xywh_list: List[tuple], color_list: List[tuple], loc: str, **kwargs) -> None:

    img = cv2.imread(path)
    assert img is not None, 'image empty.'
    img_h, img_w, _ = img.shape

    # calculate resize ratio
    w_list = [xyhw[2] for xyhw in xywh_list]
    h_list = [xyhw[3] for xyhw in xywh_list]

    denom = 0

    if loc == 'b':
        for i in range(len(w_list)):
            # h1*h2*...*wi*...*hn
            denom += w_list[i] * np.prod([h for j, h in enumerate(h_list) if j != i])

        bottom = []
        # creating patches
        for i, ((x, y, w, h), color) in enumerate(zip(xywh_list, color_list)):
            k = img_w * np.prod([h for j, h in enumerate(h_list) if j != i]) / denom
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

    elif loc == 'r':
        for i in range(len(h_list)):
            # w1*w2*...*hi*...*wn
            denom += h_list[i] * np.prod([w for j, w in enumerate(w_list) if j != i])

        bottom = []
        # creating patches
        for i, ((x, y, w, h), color) in enumerate(zip(xywh_list, color_list)):
            k = img_h * np.prod([w for j, w in enumerate(w_list) if j != i]) / denom
            # clipping
            clip = cv2.resize(img[y:y+h, x:x+w], dsize=None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)

            # draw line
            cv2.rectangle(clip, (1, 1), (clip.shape[1]-2, clip.shape[0]-2), color=color, thickness=thick_size)
            cv2.rectangle(img, (x, y, w, h), color=color, thickness=thick_size)
            bottom.append(clip)

        stack = np.vstack(bottom)
        out = np.hstack((img, stack))
        cv2.imwrite(save_name, out)       
    else:
        print(f'Invalid loc: {loc}')
        exit()


if __name__ == '__main__':

    name = '5_00045_1_b'
    kwargs = {
        'path' : '/mnt/d/python/master_thesis/source/%s.png' % name,
        'save_name' : '/mnt/d/python/master_thesis/%s_clip_unsharp.png' % name,
        'xywh_list' : [
            # 000090
            # (30, 30, 280, 280)
            # 00000002_b
            # (20, 315, 130, 80),
            # 00034
            # (1030, 270, 120, 90),
            # 00081
            # (50, 50, 200, 130),
            # (760, 270, 150, 150)
            # 00045
            (728, 270, 200, 90),
            (770, 435, 130, 70)
        ],
        'color_list': [
            # (0,255,0),
            (0,0,255),
            (255,0,0)
        ],
        'thick_size' : 4,
        'scale_list' : [4,],
        'loc_list': ['ur',],
        'loc': 'b'    # 'b' bottom or 'r' right
    }

    add_patch_outside(**kwargs)
    # add_patch_inside(**kwargs)
    # clip_image(**kwargs)
    # clip_image()