import argparse
import cv2
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib as mpl

parser = argparse.ArgumentParser(description='make difference heatmap between input(blur) and output(deblurred).')

parser.add_argument('--img1_dir', default=None, help='e.g., ../dataset/night_blur/test/Long')
parser.add_argument('--img1_is_input', default=False, action='store_true', help='True or False. if True, the first and laset two inputs are ignored.')
parser.add_argument('--img2_dir', default=None, help='e.g., ../STDAN_modified/exp_log/test/20231129_STDAN_Stack_night_blur_ckpt-epoch-0905/output')
parser.add_argument('--img2_is_input', default=False, action='store_true', help='True or False. if True, the first and laset two inputs are ignored.')
parser.add_argument('--save_path', default=None, help='e.g., ../STDAN_modified/exp_log/test/20231129_STDAN_Stack_night_blur_ckpt-epoch-0905/diff')

args = parser.parse_args()


seq_list = [f for f in os.listdir(args.img1_dir) if os.path.isdir(os.path.join(args.img1_dir, f))]

for seq in seq_list:

    files_1 = sorted(glob.glob(os.path.join(args.img1_dir, seq, '*.png')))
    files_2 = sorted(glob.glob(os.path.join(args.img2_dir, seq, '*.png')))
    
    
    if args.img1_is_input == True:
        files_1 = files_1[2:-2]
    if args.img2_is_input == True:
        files_2 = files_2[2:-2]

    assert len(files_1) == len(files_2), f"len(img1_dir)={len(files_1)}, len(img2_dir)={len(files_2)} don't match."

    for file_1, file_2 in zip(files_1, files_2):
        print(f'file_1={file_1} file_2={file_2}')
        basename = os.path.splitext(os.path.basename(file_1))[0]
        img1 = cv2.imread(file_1)
        img2 = cv2.imread(file_2)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        H, W, _ = img1.shape
        img1 = cv2.resize(img1, (W//4, H//4))
        img2 = cv2.resize(img2, (W//4, H//4))

        diff = img1.astype(int) - img2.astype(int)

        diff = np.abs(diff)
        diff_sum = np.sum(diff, axis=2)//3

        save_path = os.path.join(args.save_path, seq)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(diff_sum, vmin=0, vmax=210, cmap='jet')
        ax.axis("off")
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', '5%', pad='3%')

        fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, basename + '.png'))
        print(f'saved {os.path.join(save_path, basename + ".png")}')
        plt.close('all')
