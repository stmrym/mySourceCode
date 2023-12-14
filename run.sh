#!/bin/bash

python3 make_diff_heatmap.py \
    --img1_dir ../dataset/real_night_blur/test/input \
    --img1_is_input \
    --img2_dir ../STDAN_modified/exp_log/test/20231129_STDAN_Stack_real_night_blur_ckpt-epoch-0500/output \
    --save_path ../STDAN_modified/exp_log/test/20231129_STDAN_Stack_real_night_blur_ckpt-epoch-0500/diff