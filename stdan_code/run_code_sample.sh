#!/bin/bash

python3 create_stdan_gif.py output STDAN_Stack_BSD_3ms24ms_best-ckpt

python3 calc_ssim.py 20231129_STDAN_Stack_night_blur_ckpt-epoch-0905 --gt_path ../../dataset/night_blur/test