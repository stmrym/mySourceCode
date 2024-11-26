import cv2
import torch
from torch import nn
import numpy as np
from bm3d import bm3d_rgb
from utils.stop_watch import stop_watch



class Denoise:
    def __init__(self):
        self.threshold = 0.01
        self.low = 0.0
        self.high = 0.5
        self.min_step = 0.0005

        self.cont = True
        self.result = []
        self.patch_size = (5, 5)
        self.margin = (self.patch_size[0] // 2, self.patch_size[1] // 2)
        self.unfold = nn.Unfold(kernel_size=self.patch_size)

    def denoise(self, img):
        denoised, err = self._bm3d_twocolor(img, self.low)
        self.result.append([self.low, err])
        if err <= self.threshold:
            cont = False

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
        
        denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        return denoised



    @stop_watch
    def _bm3d_twocolor(self, img, noise_level):
        if noise_level > 1e-6:
            print(f'True {noise_level}')
            denoised = bm3d_rgb(img, sigma_psd=noise_level * 255).astype(np.float32)
            print('end denoise')
        else:
            print(f'False {noise_level}')
            denoised = img

        print('begin twocolor')
        denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        denoised = torch.tensor(denoised, device='cuda:0')
        denoised = denoised.permute(2, 0, 1)
        _, _, err = self._two_color(denoised, self.margin)
        err = (np.mean(err**0.8))**(1 / 0.8)

        return denoised, err

    # Initialize cluster centers
    @stop_watch
    def _init_centers(self, r_col, g_col, b_col):
        idx = torch.randint(0, self.patch_size[0] * self.patch_size[1], (1, r_col.shape[1]), device=r_col.device)

        rc = [None] * 2
        gc = [None] * 2
        bc = [None] * 2

        c_idx = torch.ravel(torch.arange(len(idx[0]), device=r_col.device), idx[0])

        rc[0] = r_col.flatten()[c_idx]
        gc[0] = g_col.flatten()[c_idx]
        bc[0] = b_col.flatten()[c_idx]

        diff = (r_col - rc[0])**2 + (g_col - gc[0])**2 + (b_col - bc[0])**2
        nonzero_num = torch.sum(diff > 1e-12, axis=0)
        s_idx = torch.argsort(diff, axis=0, descending=True)

        s_sub2ind = torch.ravel_multi_index((torch.arange(len(nonzero_num)), torch.maximum(torch.ceil(nonzero_num * 0.5).to(torch.int), torch.tensor(1, device=r_col.device))), s_idx.T.shape)
        idx = s_idx.flatten()[s_sub2ind]

        c_idx = torch.ravel_multi_index((torch.arange(len(idx)), idx), r_col.T.shape)

        rc[1] = r_col.flatten()[c_idx]
        gc[1] = g_col.flatten()[c_idx]
        bc[1] = b_col.flatten()[c_idx]

        return rc, gc, bc


    # Two-color clustering
    @stop_watch
    def _two_color(self, img, margin):
        # img: (3, H, W)

        # (3, p*p, (H-p+1)(W-p+1))
        img_col = torch.squeeze(self.unfold(img.unsqueeze(0))).reshape(3, self.patch_size[0]*self.patch_size[1], -1)

        img = img[:, margin[0]: -margin[0], margin[1]: -margin[1]]

        r_col = img_col[0]
        g_col = img_col[1]
        b_col = img_col[2]

        r = img[0]
        g = img[1]
        b = img[2]

        r_centers, g_centers, b_centers = self._init_centers(r_col, g_col, b_col, self.patch_size)

        diff = [None, None]
        max_iter = 10
        for iter in range(max_iter):
            print(iter)
            for k in range(2):
                diff[k] = (r_col - r_centers[k])**2 + (g_col - g_centers[k])**2 + (b_col - b_centers[k])**2

            map = (diff[0] <= diff[1]).float()

            for k in range(2):
                map_sum = torch.sum(map, axis=0)
                map_sum[map_sum < 1e-10] = 1e+10

                norm_coef = 1.0 / map_sum
                r_centers[k] = torch.sum(r_col * map, axis=0) * norm_coef
                g_centers[k] = torch.sum(g_col * map, axis=0) * norm_coef
                b_centers[k] = torch.sum(b_col * map, axis=0) * norm_coef

                map = 1.0 - map

            diff1 = (r_centers[0] - r.t().reshape(-1))**2 + (g_centers[0] - g.t().reshape(-1))**2 + (b_centers[0] - b.t().reshape(-1))**2
            diff2 = (r_centers[1] - r.t().reshape(-1))**2 + (g_centers[1] - g.t().reshape(-1))**2 + (b_centers[1] - b.t().reshape(-1))**2
            map = diff1 > diff2

            tmp = r_centers[0][map]
            r_centers[0][map] = r_centers[1][map]
            r_centers[1][map] = tmp

            tmp = g_centers[0][map]
            g_centers[0][map] = g_centers[1][map]
            g_centers[1][map] = tmp

            tmp = b_centers[0][map]
            b_centers[0][map] = b_centers[1][map]
            b_centers[1][map] = tmp

        center1 = torch.zeros_like(img)
        center1[:, :, 0] = r_centers[0].reshape(r.shape)
        center1[:, :, 1] = g_centers[0].reshape(g.shape)
        center1[:, :, 2] = b_centers[0].reshape(b.shape)

        center2 = torch.zeros_like(img)
        center2[:, :, 0] = r_centers[1].reshape(r.shape)
        center2[:, :, 1] = g_centers[1].reshape(g.shape)
        center2[:, :, 2] = b_centers[1].reshape(b.shape)

        diff = center2 - center1
        len = torch.sqrt(torch.sum(diff**2, axis=2))
        dir = diff / (len.unsqueeze(-1) + 1e-12)

        diff = img - center1
        proj = torch.sum(diff * dir, axis=2)
        dist = diff - dir * proj.unsqueeze(-1)
        err = torch.sqrt(torch.sum(dist**2, axis=2))
        
        return center1, center2, err





