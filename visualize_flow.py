import glob
import os
import numpy as np
from tqdm import tqdm
from graph_util import plot_heatmap


if __name__ == '__main__':

    npz_path = '../dataset/BSD_3ms24ms/flow_sharp/%s.npz'
    seq_select = 'all'
    save_base_dir = '../dataset/BSD_3ms24ms/flow_sharp_cmap_norm'
    scale_h = 0.25
    scale_w = 0.25

    if seq_select == 'all': # for all sequences
        seq_list = sorted([os.path.splitext(os.path.basename(file))[0] for file in glob.glob(npz_path.replace('%s', '*'))])
    else:                   # for a sequence
        seq_list = [seq_select]

    for seq in seq_list:
        print(seq)
        flow_npz = np.load(npz_path % seq)
        save_dir = os.path.join(save_base_dir, seq)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        for basename in tqdm(flow_npz.files):
            flow = flow_npz[basename]
            save_name = os.path.join(save_dir, basename)

            # flow_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
            
            # normalized
            H, W, _ = flow.shape
            flow_mag = np.sqrt((flow[:,:,0]/(scale_w * W))**2 + (flow[:,:,1]/(scale_h * H))**2)
            print(flow_mag.max(), flow_mag.min())
            flow_mag = np.clip(flow_mag, None, 1)

            plot_heatmap(plot_data=flow_mag, save_name=save_name, cmap='jet', vmin=0, vmax=1)

    

