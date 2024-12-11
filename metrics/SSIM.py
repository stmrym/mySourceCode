import cv2
from skimage.metrics import structural_similarity as compare_ssim

class SSIM:
    def __init__(self, data_range=255.0, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False):
        self.data_range = data_range
        self.channel_axis = channel_axis
        self.gaussian_weights = gaussian_weights
        self.sigma = sigma
        self.use_sample_covariance = use_sample_covariance

    def calculate(self, img1, img2, **kwargs):
        ssim = compare_ssim(img2, img1, **vars(self))
        return ssim



if __name__ == '__main__':
    
    recons_l = [
        '/mnt/d/results/20241210/074_00000034_output.png'
    ]

    gt_l = [
        '/mnt/d/results/20241210/074_00000034_gt.png'
    ]


    metric = SSIM()

    for recons_path, gt_path in zip(recons_l, gt_l):

        recons = cv2.imread(recons_path)
        gt = cv2.imread(gt_path)

        result = metric.calculate(img1=recons, img2=gt)
        print(f'{recons_path}, {gt_path}, SSIM: {result:.3f}\n')