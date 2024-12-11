import cv2
import numpy as np




class PSNR:
    def __init__(self, crop_border=0, max_order=255.0):
        self.crop_border = crop_border
        self.max_order = max_order

    def calculate(self, img1, img2, **kwargs):
        """Calculate PSNR (Peak Signal-to-Noise Ratio).

        Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: PSNR result.
        """

        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')

        if self.crop_border != 0:
            img1 = img1[self.crop_border:-self.crop_border, self.crop_border:-self.crop_border, ...]
            img2 = img2[self.crop_border:-self.crop_border, self.crop_border:-self.crop_border, ...]

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 10. * np.log10(self.max_order * self.max_order / mse)
    

if __name__ == '__main__':

    params = {'crop_border': 0,
            'max_order': 255.0
            }
    
    recons_l = [
        '/mnt/d/results/20241210/074_00000034_output.png'
    ]

    gt_l = [
        '/mnt/d/results/20241210/074_00000034_gt.png'
    ]


    metric = PSNR(**params)


    for recons_path, gt_path in zip(recons_l, gt_l):

        recons = cv2.imread(recons_path)
        gt = cv2.imread(gt_path)

        result = metric.calculate(img1=recons, img2=gt)
        print(f'{recons_path}, {gt_path}, PSNR: {result:.3f}\n')