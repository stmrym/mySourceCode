import cv2
import numpy as np
from inc.AnisoSetEst import AnisoSetEst, MetricQ
from inc.denoise import denoise
from inc.mask_lines import mask_lines
from scipy.signal import fftconvolve
import cpbd
from inc.align import align
from skimage.transform import resize
from inc.grad_ring import grad_ring

def mean_norm(v, p):
    return np.mean(np.abs(v)**p)**(1/p)

def sparsity(img):
    d = [None] * 3
    for c in range(3):
        dx, dy = np.gradient(img[:, :, c])
        d[c] = (np.sqrt(dx**2 + dy**2)).ravel('F')
    
    result = 0
    for c in range(3):
        result = result + mean_norm(d[c], 0.66)
    return result


def my_sd(x, p):
    avg = np.mean(x)
    sd = np.mean(np.abs(x - avg)**p) ** (1.0/p)
    return sd

def smallgrad(img):
    d = np.zeros(img[:, :, 0].shape)
    for c in range(3):
        dx, dy = np.gradient(img[:, :, c])
        d += np.sqrt(dx**2 + dy**2)
    d /= 3
    
    sorted_d = np.sort(d.ravel('F'))
    n = max(int(len(sorted_d) * 0.3), 10)
    result = my_sd(sorted_d[:n], 0.1)
    
    return result


def metric_q(img):
    PATCH_SIZE = 8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 255
    aniso_set = AnisoSetEst(img, PATCH_SIZE)
    result = -MetricQ(img, PATCH_SIZE, aniso_set)
    return result


def auto_corr(img):

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


def compute_ncc(img, ref, img_margin):

    assert len(img.shape) == 2, 'Image should have a single channel.'

    template = ref[img_margin:-img_margin, img_margin:-img_margin]
    ncc = np.ones((img_margin * 2 + 1, img_margin * 2 + 1)) * 100
    ncc_abs = np.ones((img_margin * 2 + 1, img_margin * 2 + 1))

    img_mask = mask_lines(img)
    ref_mask = mask_lines(ref)
    t_mask = ref_mask[img_margin:-img_margin, img_margin:-img_margin]

    dx, dy = np.gradient(img, axis=(0, 1)) 
    tdx, tdy = np.gradient(template, axis=(0, 1))

    dx[img_mask] = 0 
    dy[img_mask] = 0 
    tdx[t_mask] = 0 
    tdy[t_mask] = 0

    ncc_dx = xcorr2_fft(tdx, dx)
    ncc_dy = xcorr2_fft(tdy, dy)

    ncc_dx = ncc_dx[tdx.shape[0]-1:, tdx.shape[1]-1:]
    ncc_dy = ncc_dy[tdy.shape[0]-1:, tdy.shape[1]-1:]

    ncc_dx = ncc_dx[:img_margin * 2 + 1, :img_margin * 2 + 1]
    ncc_dy = ncc_dy[:img_margin * 2 + 1, :img_margin * 2 + 1]

    ncc_dx = ncc_dx / ncc_dx[img_margin, img_margin]
    ncc_dy = ncc_dy / ncc_dy[img_margin, img_margin]

    ncc_dx_abs = np.abs(ncc_dx)
    ncc_dy_abs = np.abs(ncc_dy)

    mask = ncc_dx_abs < ncc_abs
    ncc[mask] = ncc_dx[mask]
    ncc_abs[mask] = ncc_dx_abs[mask]

    mask = ncc_dy_abs < ncc_abs
    ncc[mask] = ncc_dy[mask]
    ncc_abs[mask] = ncc_dy_abs[mask]

    return ncc


def xcorr2_fft(a, b):
    b_rot = np.rot90(np.conj(b), 2) 
    result = fftconvolve(a, b_rot, mode='full') 
    return result


def norm_sparsity(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx, dy = np.gradient(img)
    d = np.sqrt(dx**2 + dy**2)

    result = mean_norm(d, 1.0) / mean_norm(d, 2.0)
    return result


def calc_cpbd(img):

    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return -cpbd.compute(img)


def pyr_ring(img, blurred):

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


def saturation(img):
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



def measure(deblurred, blurred):

    features = {}

    features['sparsity'] = sparsity(deblurred)
    print('sparsity')
    features['smallgrad'] = smallgrad(deblurred)
    print('smallgrad')
    features['metric_q'] = metric_q(deblurred)
    print('metric_q')
    print('before')
    denoised = denoise(deblurred)
    print('after')
    # denoised = deblurred
    cv2.imwrite('denoised.png', np.clip(denoised*255, 0, 255).astype(np.uint8))
    # np.save('denoised.npy', denoised)
    # denoised = np.load('denoised.npy')

    features['auto_corr'] = auto_corr(denoised)
    print('auto_corr')
    features['norm_sps'] = norm_sparsity(denoised)
    print('norm_sps')
    features['cpbd'] = calc_cpbd(denoised)
    print('cpbd')
    features['pyr_ring'] = pyr_ring(denoised, blurred)
    print('pyr_ring')
    features['saturation'] = saturation(deblurred)
    print('saturation')
    # 各特徴量に対応する係数を掛けて合計したスコアを計算
    score = (features['sparsity']   * -8.70515   +
             features['smallgrad']  * -62.23820  +
             features['metric_q']   * -0.04109   +
             features['auto_corr']  * -0.82738   +
             features['norm_sps']   * -13.90913  +
             features['cpbd']       * -2.20373   +
             features['pyr_ring']   * -149.19139 +
             features['saturation'] * -6.62421)

    return score, features


if __name__ == '__main__':

    deblurred = cv2.imread('deblurred.png').astype(np.float32) / 255.0
    blurred   = cv2.imread('blurry.png').astype(np.float32) / 255.0

    score, features = measure(deblurred, blurred)
    print(score, features)