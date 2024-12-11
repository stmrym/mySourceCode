import cv2
import numpy as np
import torch
import lpips
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from metrics.utils.stop_watch import stop_watch


class LPIPS:
    def __init__(self, val_range=255.0):
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.val_range = val_range

    @stop_watch
    def calculate(self, img1, img2, **kwargs):
        img1 = self._convert_to_tensor(img1, self.val_range) 
        img2 = self._convert_to_tensor(img2, self.val_range) 
        loss = self.loss_fn_alex(img1.permute(2,0,1).unsqueeze(0), img2.permute(2,0,1).unsqueeze(0)).mean().detach().cpu()
        return loss

    def _convert_to_tensor(self, img, val_range):
        if not isinstance(img, torch.Tensor):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.astype(np.float32) / val_range).to(self.device)
            else:
                print('unsupported format')
                exit()
        return img


if __name__ == '__main__':
    
    recons_l = [
        '/mnt/d/results/20241210/074_00000034_output.png'
    ]

    gt_l = [
        '/mnt/d/results/20241210/074_00000034_gt.png'
    ]


    metric = LPIPS()

    for recons_path, gt_path in zip(recons_l, gt_l):

        recons = cv2.imread(recons_path)
        gt = cv2.imread(gt_path)

        recons = cv2.cvtColor(recons, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        result = metric.calculate(img1=recons, img2=gt)
        print(f'{recons_path}, {gt_path}, LPIPS: {result:.3f}\n')