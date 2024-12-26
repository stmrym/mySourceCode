import numpy as np
from scipy.signal import correlate2d
from pymlfunc import normxcorr2

def align(image, ref, return_images=False):

    margin = 75
    
    # テンプレートの抽出
    template = image[margin:-margin, margin:-margin, :]
    template_size = template.shape
    template_size = np.array(template_size) + (np.mod(template_size, 2) - 1)
    template_size = template_size[0:2]
    template = template[:template_size[0], :template_size[1], :]

    # NCCの計算
    ref_size = ref.shape
    ncc_size = tuple(np.array(ref_size[0:2]) + np.array(template_size) - 1)
    ncc = np.zeros(ncc_size)
    for k in range(3):
        ncc += normxcorr2(template[:, :, k], ref[:, :, k])

    ncc_margin = np.floor(np.array(template_size) / 2).astype(int)
    ncc = ncc[ncc_margin[0]:-ncc_margin[0], ncc_margin[1]:-ncc_margin[1]]

    # dy: row, dx: col
    dx, dy = np.unravel_index(np.argmax(ncc), ncc.T.shape)
    dy = dy - np.floor(template_size[0] / 2) - margin
    dx = dx - np.floor(template_size[1] / 2) - margin
    dy, dx = int(dy), int(dx)

    if return_images:
        if dy < 0:
            height = min(image.shape[0] + dy, ref.shape[0])
            image = image[-dy:height-dy, :, :]
            ref = ref[:height, :, :]
        else:
            height = min(image.shape[0], ref.shape[0] - dy)
            image = image[:height, :, :]
            ref = ref[dy:height+dy, :, :]
        if dx < 0:
            width = min(image.shape[1] + dx, ref.shape[1])
            image = image[:, -dx:width-dx, :]
            ref = ref[:, :width, :]
        else:
            width = min(image.shape[1], ref.shape[1] - dx)
            image = image[:, :width, :]
            ref = ref[:, dx:width+dx, :]
    else:
        return dx, dy

    return image, ref
