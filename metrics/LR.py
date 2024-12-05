import cv2
import cpbd
import numpy as np
from skimage.transform import resize
from utils import util
from utils.AnisoSetEst import AnisoSetEst, MetricQ
from utils.denoise import denoise
from utils.denoise_cuda import Denoise
from utils.compute_ncc import compute_ncc
from utils.pyr_ring import align, grad_ring
from utils.stop_watch import stop_watch

class LR:
    def __init__(self, device, **kwargs):
        self.device = device


    def calculate(self, img1, img2, **kwargs):
        '''
        img1: deblurred image [0, 255]
        img2: blurred image [0, 255]
        '''

        img1 = (img1/255).astype(np.float32)
        img2 = (img2/255).astype(np.float32)

        score, features = self._measure(deblurred=img1, blurred=img2)
        print(score, features)
        return score
    
    @stop_watch
    def _measure(self, deblurred, blurred):

        features = {}

        # features['sparsity'] = self._sparsity(deblurred)
        # features['smallgrad'] = self._smallgrad(deblurred)
        features['metric_q'] = self._metric_q(deblurred)
 
        denoised = denoise(deblurred)

        print('finish denoise_cuda')

        # denoised = deblurred
        cv2.imwrite('denoised_cuda.png', np.clip(denoised*255, 0, 255).astype(np.uint8))
        np.save('denoised_cuda.npy', denoised)
        # denoised = np.load('denoised.npy')

        features['auto_corr'] = self._auto_corr(denoised)
        features['norm_sps'] = self._norm_sparsity(denoised)
        features['cpbd'] = self._calc_cpbd(denoised)
        features['pyr_ring'] = self._pyr_ring(denoised, blurred)
        features['saturation'] = self._saturation(deblurred)
        
        score = (features['sparsity']   * -8.70515   +
                features['smallgrad']  * -62.23820  +
                features['metric_q']   * -0.04109   +
                features['auto_corr']  * -0.82738   +
                features['norm_sps']   * -13.90913  +
                features['cpbd']       * -2.20373   +
                features['pyr_ring']   * -149.19139 +
                features['saturation'] * -6.62421)

        return score, features

    @stop_watch
    def _sparsity(self, img):
        d = [None] * 3
        for c in range(3):
            dx, dy = np.gradient(img[:, :, c])
            d[c] = (np.sqrt(dx**2 + dy**2)).ravel('F')
        
        result = 0
        for c in range(3):
            result = result + util.mean_norm(d[c], 0.66)
        return result
    
    @stop_watch
    def _smallgrad(self, img):
        d = np.zeros(img[:, :, 0].shape)
        for c in range(3):
            dx, dy = np.gradient(img[:, :, c])
            d += np.sqrt(dx**2 + dy**2)
        d /= 3
        
        sorted_d = np.sort(d.ravel('F'))
        n = max(int(len(sorted_d) * 0.3), 10)
        result = util.my_sd(sorted_d[:n], 0.1)
        
        return result
    
    @stop_watch
    def _metric_q(self, img):
        PATCH_SIZE = 8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 255
        aniso_set = AnisoSetEst(img, PATCH_SIZE)
        result = -MetricQ(img, PATCH_SIZE, aniso_set)
        return result
    
    @stop_watch
    def _auto_corr(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        MARGIN = 50

        ncc_orig = compute_ncc(img, img, MARGIN)

        sizes = ncc_orig.shape 
        assert sizes[0] == sizes[1] 
        assert sizes[0] % 2 == 1 
        radius = sizes[0] // 2 
        y_dists, x_dists = np.meshgrid(np.arange(sizes[0]), np.arange(sizes[1]), indexing='ij') 
        dists = np.sqrt((y_dists - radius) ** 2 + (x_dists - radius) ** 2) 
        ncc = np.abs(ncc_orig) 
        max_m = np.zeros(radius + 1) 
        for r in range(radius + 1): 
            w = np.abs(dists - r) 
            w = np.minimum(w, 1) 
            w = 1 - w 
            max_m[r] = np.max(ncc[w > 0]) 

        max_m[0] = 0 
        result = np.sum(max_m)

        return result

    @stop_watch
    def _norm_sparsity(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dx, dy = np.gradient(img)
        d = np.sqrt(dx**2 + dy**2)

        result = util.mean_norm(d, 1.0) / util.mean_norm(d, 2.0)
        return result

    @stop_watch
    def _calc_cpbd(self, img):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return -cpbd.compute(img)

    @stop_watch
    def _pyr_ring(self, img, blurred):

        img, blurred = align(img, blurred, True)
        height, width, color_count = img.shape

        result = 0.0
        sizes = []
        j = 0
        while True:
            coef = 0.5 ** j
            cur_height = round(height * coef)
            cur_width = round(width * coef)
            if min(cur_height, cur_width) < 16:
                break
            sizes.append([j, cur_width, cur_height])

            cur_img = resize(img, (cur_height, cur_width), order=1)
            cur_blurred = resize(blurred, (cur_height, cur_width), order=1)

            diff = grad_ring(cur_img, cur_blurred)
            if j > 0:
                result += np.mean(diff)

            j += 1

        return result

    @stop_watch
    def _saturation(self, img):
        # 各ピクセルの最大値を計算
        max_values = np.max(img, axis=2)
        
        # 最大値が10/255以下のマスクを作成
        mask_low = (max_values <= 10.0 / 255.0)
        result_low = np.sum(mask_low.astype(np.float64)) / max_values.size

        # 各ピクセルの最小値を計算
        min_values = np.min(img, axis=2)
        
        # 最小値が1 - 10/255以上のマスクを作成
        mask_high = (min_values >= 1.0 - (10.0 / 255.0))
        result_high = np.sum(mask_high.astype(np.float64)) / min_values.size

        # 結果を計算
        result = result_low + result_high
        
        return result



if __name__ == '__main__':

    params = {'device': 'cpu'}

    deblurred = cv2.imread('./source_code_m/deblurred.png')
    blurred = cv2.imread('./source_code_m/blurry.png')

    metric = LR(**params)

    result = metric.calculate(img1=deblurred, img2=blurred)
