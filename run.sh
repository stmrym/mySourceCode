#!/bin/bash

python3 make_diff_heatmap.py \
    --img1_dir ../dataset/BSD_3ms24ms/test/%s/Blur/RGB \
    --img2_dir ../dataset/BSD_3ms24ms/test/%s/Sharp/RGB \
    --save_path ../dataset/BSD_3ms24ms/diff