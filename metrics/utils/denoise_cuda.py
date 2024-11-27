import cv2
import torch
from torch import nn
import numpy as np
from bm3d import bm3d_rgb
from pytorch_bm3d import BM3D
from utils.stop_watch import stop_watch


class Denoise:
    def __init__(self):
        self.threshold = 0.01
        self.low = 0.0
        self.high = 0.5
        self.min_step = 0.0005

        self.result = []
        self.patch_size = (5, 5)
        self.margin = (self.patch_size[0] // 2, self.patch_size[1] // 2)
        self.unfold = nn.Unfold(kernel_size=self.patch_size)
        self.bm3d = BM3D(two_step=True)

    def denoise(self, img):

        # numpy (BGR) -> tensor (RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img, device='cuda:0')
        img = img.permute(2, 0, 1)

        denoised, err = self._bm3d_twocolor(img, self.low)
        self.result.append([self.low, err])

        cont = False if err <= self.threshold else True
        
        if cont:
            denoised, err = self._bm3d_twocolor(img, self.high)
            self.result.append([self.high, err])
            if err > self.threshold:
                cont = False

        cur_low = self.low
        cur_high = self.high
        while cont:
            cur = (cur_low + cur_high) * 0.5
            print(cont, cur)
            denoised, err = self._bm3d_twocolor(img, cur)
            print(err)
            self.result.append([cur, err])

            if err <= self.threshold:
                cur_high = cur
            elif err > self.threshold:
                cur_low = cur

            if (cur_low + self.min_step >= cur_high):
                idx = np.abs(np.array(self.result)[:, 0] - cur_high).argmin()
                assert idx is not None

                denoised, err = self._bm3d_twocolor(img, cur_high)
                self.result.append([cur_high, err])
                cont = False
        
        denoised = denoised.cpu().numpy()
        denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        return denoised



    @stop_watch
    def _bm3d_twocolor(self, img, noise_level):
        if noise_level > 1e-6:
            print(f'True {noise_level}')
            # denoised = bm3d_rgb(img, sigma_psd=noise_level * 255).astype(np.float32)
            denoised = self.bm3d((img*255).int(),variants=0)

            print('end denoise')
        else:
            print(f'False {noise_level}')
            denoised = img

        _, _, err = self._two_color(denoised, self.margin)
        err = (np.mean(err**0.8))**(1 / 0.8)

        return denoised, err

    # Initialize cluster centers
    @stop_watch
    def _init_centers(self, img_col, patch_size):
        '''
        img_col: (3, p*p, (H-p+1)(W-p+1) )
        '''
        idx = torch.randint(0, patch_size[0] * patch_size[1], (1, img_col.shape[-1]), device=img_col.device)

        rgb_centers = [None] * 2
        rgb_centers[0] = img_col[:, idx[0], torch.arange(img_col.shape[-1])].unsqueeze(1)

        diff = torch.sum((img_col - rgb_centers[0])**2, dim=0)
        nonzero_num = torch.sum(diff > 1e-12, dim=0)
        
        # (p*p, (H-p+1)(W-p+1))
        s_idx = torch.argsort(diff, dim=0, descending=True)
        # ( (H-p+1)(W-p+1) )
        half_max = torch.max(torch.ceil(nonzero_num * 0.5), torch.tensor(1.0)).long()
        # ( (H-p+1)(W-p+1) )
        idx = s_idx[half_max, torch.arange(img_col.shape[-1])]

        rgb_centers[1] = img_col[:, idx, torch.arange(img_col.shape[-1])].unsqueeze(1)

        return rgb_centers


    # Two-color clustering
    @stop_watch
    def _two_color(self, img, margin):
        # img: (3, H, W)
        # L = (H-p+1)(W-p+1)

        # (3, p*p, L)
        img_col = torch.squeeze(self.unfold(img.unsqueeze(0))).reshape(3, self.patch_size[0]*self.patch_size[1], -1)

        img = img[:, margin[0]: -margin[0], margin[1]: -margin[1]]

        # rgb_centers: [(3, 1, L), (3, 1, L)]
        rgb_centers = self._init_centers(img_col, self.patch_size)
        
        max_iter = 10
        for _ in range(max_iter):

            diff = [torch.sum((img_col - rgb_centers[k])**2, dim=0) for k in range(2)]
            map = (diff[0] <= diff[1]).float()
            
            for k in range(2):
                map_sum = torch.sum(map, axis=0)
                map_sum[map_sum < 1e-10] = 1e+10

                norm_coef = 1.0 / map_sum
                
                # (3, p*p, L) * (1, p*p, L) -> (3, L)
                rgb_centers[k] = torch.sum(img_col * map.unsqueeze(0), dim=1) * norm_coef
                # (3, L) -> (3, 1, L)
                map = 1.0 - map

            # diff1, diff2: (L,)
            # (3, L) - (3, L) 
            diff1 = torch.sum((rgb_centers[0] - img.transpose(1, 2).reshape(3, -1))**2, dim=0)
            diff2 = torch.sum((rgb_centers[1] - img.transpose(1, 2).reshape(3, -1))**2, dim=0)

            # (L, )
            map = diff1 > diff2

            rgb_centers[0][:,map], rgb_centers[1][:,map] = rgb_centers[1][:,map], rgb_centers[0][:,map]
            
            rgb_centers[0] = rgb_centers[0].unsqueeze(1)
            rgb_centers[1] = rgb_centers[1].unsqueeze(1)


        center1 = rgb_centers[0].reshape(img.shape)
        center2 = rgb_centers[1].reshape(img.shape)

        diff = center2 - center1
        diff_len = torch.sqrt(torch.sum(diff**2, dim=0))

        dir = diff / (diff_len.unsqueeze(0) + 1e-12)

        diff = img - center1
        proj = torch.sum(diff * dir, dim=0)

        dist = diff - dir * proj.unsqueeze(0)
        err = torch.sqrt(torch.sum(dist**2, axis=0))

        return center1, center2, err





