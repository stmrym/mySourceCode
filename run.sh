#!/bin/bash

python3 make_diff_heatmap.py \
    --img1_dir ../dataset/night_blur/test/Long \
    --img1_is_input \
    --img2_dir ../STDAN_modified/exp_log/test/20231129_STDAN_Stack_night_blur_ckpt-epoch-0100/output \
    --save_path ../STDAN_modified/exp_log/test/20231129_STDAN_Stack_night_blur_ckpt-epoch-0100/diff