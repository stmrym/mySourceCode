import glob
import os
# from ..STDAN_modified.mmflow.mmflow.apis import init_model, inference_model
import mmflow

# class Flow_estimator():
#     def __init__(self, config_file, checkpoint_file, device='cuda:0'):
#         self.estimator = init_model(config_file, checkpoint_file, device=device)

#     def estimate(self, frame1, frame2):
#         return inference_model(self.estimator, frame1, frame2)




def flow_estimate(seq_dict, config_file, checkpoint_file):
    pass
    # flow_estimator = Flow_estimator(config_file, checkpoint_file, device='cuda:0')  

    # for seq, (output_path_list, gt_path_list) in seq_dict.items():
    #     print(seq)
    #     for output_path, gt_path in zip(tqdm(output_path_list), gt_path_list):
    #         assert os.path.basename(output_path) == os.path.basename(gt_path), f"basenames gt_file={os.path.basename(gt_path)} don't match"
    #         gt = cv2.imread(gt_path).astype(np.float32)
    #         output = cv2.imread(output_path).astype(np.float32)

    #         flow = flow_estimator.estimate(output, gt)


if __name__ == '__main__':
    
    path = '../dataset/GOPRO_Large/test/%s/blur_gamma'

    new_path = path % ('tess')
    print(new_path)
    exit()
    savedir = '../dataset/GOPR0854_11_00.gif'
    file_list = sorted(glob.glob(os.path.join(path, '*.png')))