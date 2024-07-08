import cv2
import glob
import os
import numpy as np
from tqdm import tqdm
from graph_util import plot_heatmap, cv2_heatmap, cv2_alpha_heatmap
from skimage.util.arraycrop import crop
from skimage.metrics import structural_similarity as ssim


def visualize_flow( npz_path: str, seq_select: str, save_base_dir: str, scale_h: float, scale_w : float, 
                    mode: str, **kwargs):
    

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

        if 'alpha' in kwargs and 'beta' in kwargs and 'alpha_img_path' in kwargs:
            
            alpha_img_base_path = kwargs['alpha_img_path'] % seq
            alpha_file_list = sorted(glob.glob(os.path.join(alpha_img_base_path, '*.png')))
        
            for (basename, alpha_file) in zip(tqdm(flow_npz.files), alpha_file_list):
                assert basename == os.path.basename(alpha_file), 'flow npz name and alpha img path do not match.'
                flow = flow_npz[basename]
                alpha_img = cv2.imread(alpha_file)
                save_name = os.path.join(save_dir, basename)

                # flow_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
                
                # normalized
                H, W, _ = flow.shape
                flow_mag = np.sqrt((flow[:,:,0]/(scale_w * W))**2 + (flow[:,:,1]/(scale_h * H))**2)
                # print(flow_mag.max(), flow_mag.min())
                flow_mag = np.clip(flow_mag, None, 1)

                if mode == 'img':
                    cv2_alpha_heatmap(map=flow_mag, image=alpha_img, save_name=save_name, alpha=kwargs['alpha'], beta=kwargs['beta'], cmap=cv2.COLORMAP_JET)
                else:
                    print('alpha blend only supports mode == "img"')
                    exit()

        else:
            for basename in tqdm(flow_npz.files):
                flow = flow_npz[basename]
                save_name = os.path.join(save_dir, basename)

                # flow_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
                
                # normalized
                H, W, _ = flow.shape
                flow_mag = np.sqrt((flow[:,:,0]/(scale_w * W))**2 + (flow[:,:,1]/(scale_h * H))**2)
                # print(flow_mag.max(), flow_mag.min())
                flow_mag = np.clip(flow_mag, None, 1)

                if mode == 'fig':
                    plot_heatmap(plot_data=flow_mag, save_name=save_name, cmap='jet', vmin=0, vmax=1)
                elif mode == 'img':
                    cv2_heatmap(image=flow_mag, save_name=save_name, cmap=cv2.COLORMAP_JET)



def visualize_weighted_ssim_map(npz_path: str, output_path: str, gt_path: str, seq_select: str, save_base_dir: str, scale_k: float, **kwargs):
    
    flow_npz = np.load(npz_path % seq_select)
    for basename in tqdm(flow_npz.files):

        output_img_path = os.path.join(output_path, seq_select, basename)
        gt_img_path = os.path.join(gt_path % seq_select, basename)

        if os.path.isfile(output_img_path):
            print(basename)
            img = cv2.imread(output_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

            _, ssim_map = ssim(img, gt, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                    data_range=255.0, gradient=False, full=True)

            # flow_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
            flow = flow_npz[basename]
            # normalized
            H, W, _ = flow.shape
            flow_mag = np.sqrt((flow[:,:,0]/(scale_k * W))**2 + (flow[:,:,1]/(scale_k * H))**2)
            # print(flow_mag.max(), flow_mag.min())
            flow_mag = np.clip(flow_mag, None, 1)  

            mask = flow_mag * ssim_map  

            ssim_map_save_name = os.path.join(save_base_dir, 'ssim_map')
            mask_save_name = os.path.join(save_base_dir, 'mask')

            os.makedirs(ssim_map_save_name, exist_ok=True)
            os.makedirs(mask_save_name, exist_ok=True)

            cv2_heatmap(image=ssim_map, save_name=os.path.join(ssim_map_save_name, basename), cmap=cv2.COLORMAP_JET)
            cv2_heatmap(image=mask, save_name=os.path.join(mask_save_name, basename), cmap=cv2.COLORMAP_JET)



if __name__ == '__main__':

    kwargs = {   
        'npz_path' : '../dataset/BSD_3ms24ms/flow_sharp/%s.npz',
        'seq_select' : '128',
        'save_base_dir' : '../STDAN_modified/exp_log/train/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization',
        'output_path' : '../STDAN_modified/exp_log/train/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/epoch-1200_output',
        'gt_path' : '../dataset/BSD_3ms24ms/test/%s/Sharp/RGB',
        'scale_k' : 0.10,
        'mode' : 'img',
        'alpha': 0.5,
        'beta' : 64, 
        'alpha_img_path': '../dataset/GOPRO_Large/test/%s/blur' 
    }

    # visualize_flow(**kwargs)
    visualize_weighted_ssim_map(**kwargs)
    

