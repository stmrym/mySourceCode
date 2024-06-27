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
                        save_name: str,
                        batch_size: int = 1,
                        ) -> None:
    # make batch and save estimated flow to .npy file
    # flow_estimator: Instance of Flow_estimator
    # frame_list: List of image files [img1, img2, ...]
    flows_dict = {}

    for i in tqdm(range(0, len(frame_list) - 1, batch_size)):
        frame_batch_1 = frame_list[i : i+batch_size]
        frame_batch_2 = frame_list[i+1 : i+batch_size+1]
        valids = [None for _ in range(len(frame_batch_1))]

        flow_batch = flow_estimator.estimate(frame_batch_1, frame_batch_2, valids)
        basename_list = [os.path.basename(batch) for batch in frame_batch_1]
        
        flows_dict = dict(**flows_dict, **dict(zip(basename_list, [flow['flow'] for flow in flow_batch])))
                
    np.savez(save_name, **flows_dict)
    


if __name__ == '__main__':
    
    ####### 
    # args
    config_file = '../STDAN_modified/mmflow/configs/raft/raft_8x2_100k_mixed_368x768.py'
    checkpoint_file = '../STDAN_modified/mmflow/checkpoints/raft_8x2_100k_mixed_368x768.pth'
    device = 'cuda:0'
    path = '../dataset/BSD_3ms24ms/test/%s/Sharp/RGB'
    # seq_select = '128'
    seq_select = 'all'
    save_base_dir = '../dataset/BSD_3ms24ms/flow_sharp'
    #######


    if seq_select == 'all': # for all sequences
        seq_list = [f for f in sorted(os.listdir(path.split('%s')[0]))]
    else:                   # for a sequence
        seq_list = [seq_select]

    # Initialize flow estimator
    flow_estimator = Flow_estimator(config_file, checkpoint_file, device=device) 

    for seq in seq_list:
        seq_path = path % seq
        print(seq_path)    

        if not os.path.isdir(save_base_dir):
            os.makedirs(save_base_dir, exist_ok=True)
        
        save_name = os.path.join(save_base_dir, seq) + '.npz'
        frame_list = sorted(glob.glob(os.path.join(seq_path, '*.png')))

        flow_estimate_batch(flow_estimator, frame_list, save_name, batch_size=32)
    

