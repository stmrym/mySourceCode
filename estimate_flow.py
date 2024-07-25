import cv2
import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from ..STDAN_modified.mmflow.mmflow.apis import init_model, inference_model
from mmflow.apis import init_model, inference_model
from typing import List, Optional, Sequence, Union


class Flow_estimator:
    def __init__(self, 
                config_file: str = '../STDAN_modified/mmflow/configs/raft/raft_8x2_100k_mixed_368x768.py',
                checkpoint_file: str = '../STDAN_modified/mmflow/checkpoints/raft_8x2_100k_mixed_368x768.pth',
                device: str = 'cuda:0'
                ) -> None:
        
        self.estimator = init_model(config_file, checkpoint_file, device)

    def estimate(self, 
                img1s: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
                img2s: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
                valids: Optional[Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]] = None
                ) -> Union[List[np.ndarray], np.ndarray]:
        
        return inference_model(self.estimator, img1s, img2s, valids)


def flow_estimate_batch(flow_estimator: Flow_estimator,
                        frame_list: Sequence[str],
                        save_base_dir: str,
                        seq: int,
                        batch_size: int = 1,
                        ) -> None:
    # make batch and save estimated flow to .npy file
    # flow_estimator: Instance of Flow_estimator
    # frame_list: List of image files [img1, img2, ...]    
    flows_forward_dict = {}
    flows_backward_dict = {}

    # flows_forward['000.png'] : flow 000 -> 001
    # flows_backward['000.png'] : flow 001 -> 000

    for i in tqdm(range(0, len(frame_list) - 1, batch_size)):
        frame_batch_1 = frame_list[i : i+batch_size]
        frame_batch_2 = frame_list[i+1 : i+batch_size+1]
        valids = [None for _ in range(len(frame_batch_1))]

        flow_forward_batch = flow_estimator.estimate(frame_batch_1, frame_batch_2, valids)
        flow_backward_batch = flow_estimator.estimate(frame_batch_2, frame_batch_1, valids)

        basename_list = [os.path.basename(batch) for batch in frame_batch_1]
        
        # Add flow to dict
        flows_forward_dict = dict(**flows_forward_dict, **dict(zip(basename_list, [flow['flow'] for flow in flow_forward_batch])))
        flows_backward_dict = dict(**flows_backward_dict, **dict(zip(basename_list, [flow['flow'] for flow in flow_backward_batch])))

        save_name_forward = os.path.join(save_base_dir + '_forward', seq) + '.npz'  
        save_name_backward = os.path.join(save_base_dir + '_backward', seq) + '.npz'  

    np.savez(save_name_forward, **flows_forward_dict)
    np.savez(save_name_backward, **flows_backward_dict)
    


if __name__ == '__main__':
    
    ####### 
    # args
    config_file = '../STDAN_modified/mmflow/configs/raft/raft_8x2_100k_mixed_368x768.py'
    checkpoint_file = '../STDAN_modified/mmflow/checkpoints/raft_8x2_100k_mixed_368x768.pth'
    device = 'cuda:0'
    path = '../dataset/BSD_3ms24ms/test/%s/Blur/RGB'
    # seq_select = '128'
    seq_select = '000'
    save_base_dir = '../dataset/BSD_3ms24ms/flow_blur'
    batch_size = 8
    #######


    if seq_select == 'all': # for all sequences
        seq_list = [f for f in sorted(os.listdir(path.split('%s')[0]))]
    else:                   # for a sequence
        seq_list = [seq_select]

    # Initialize flow estimator
    flow_estimator = Flow_estimator(config_file, checkpoint_file, device=device) 

    forward_dir = save_base_dir + '_forward'
    backward_dir = save_base_dir + '_backward'
    if not os.path.isdir(forward_dir):
        os.makedirs(forward_dir, exist_ok=True)
    if not os.path.isdir(backward_dir):
        os.makedirs(backward_dir, exist_ok=True)

    for seq in seq_list:
        seq_path = path % seq
        print(seq_path)    
        
        frame_list = sorted(glob.glob(os.path.join(seq_path, '*.png')))

        flow_estimate_batch(flow_estimator, frame_list, save_base_dir, seq, batch_size)
    

