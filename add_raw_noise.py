import cv2
import numpy as np
import torch
from unprocess_torch import unprocess, random_noise_levels, add_noise
from process_torch import process


def add_raw_noise(np_image, random_ccm_tensor=None, random_gains_list=None):
    # Input: np array normalized [0,1] of shape (b, h, w, c)
    # batch :0 becomes GT
    # batch :1 becomes noisy
    image = torch.tensor(np_image, dtype=torch.float32)
    raw_image, metadata = unprocess(image, random_ccm_tensor, random_gains_list)
    lambda_shot, lambda_read = random_noise_levels()
    
    batch = raw_image.shape[0]
    if batch == 1:
        noise_image = add_noise(raw_image[0,:,:,:], lambda_shot, lambda_read)
        raw_image = torch.concat([raw_image, noise_image.unsqueeze(0)], dim=0)
    elif batch == 2:
        raw_image[1,:,:,:] = add_noise(raw_image[1,:,:,:], lambda_shot, lambda_read)
    # raw_np = (noise_raw_image.numpy()*255).astype(np.uint8)
    # cv2.imwrite(f'/mnt/d/results/20240417/{fname}_raw.png', cv2.cvtColor(raw_np[:,:,:], cv2.COLOR_RGB2BGR))
    # for item in metadata.items():
        # print(item)
    # for channel in range(0, raw_np.shape[-1]):
        # print(np.max(raw_np[:,:,channel]), np.min(raw_np[:,:,channel]))
        # cv2.imwrite(f'/mnt/d/results/20240417/{fname}_{channel}_raw.png', raw_np[:,:,channel])
    red_gains = torch.cat([metadata['red_gain'], metadata['red_gain']], dim=0)
    blue_gains = torch.cat([metadata['blue_gain'], metadata['blue_gain']], dim=0)
    cam2rgbs = torch.stack([metadata['cam2rgb'], metadata['cam2rgb']], dim=0)

    output = process(raw_image, red_gains, blue_gains, cam2rgbs)
    output = output.numpy()
    redemosaiced_image = output[0]
    noise_image = output[1]
    return redemosaiced_image, noise_image



if __name__ == '__main__':
    fname = '00097'
    ext = 'png'
    path = f'/mnt/d/results/20240417/{fname}.{ext}'
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(np.array(image/255, dtype=np.float32), cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    
    redemosaiced_image, noise_image = add_raw_noise(image)
    
    redemosaiced_image = cv2.cvtColor((redemosaiced_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    noise_image = cv2.cvtColor((noise_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'/mnt/d/results/20240417/{fname}_raw_noise_GT.png', redemosaiced_image)
    cv2.imwrite(f'/mnt/d/results/20240417/{fname}_raw_noise.png', noise_image)

    # for channel in range(0, processed_np.shape[-1]):
    #     cv2.imwrite(f'/mnt/d/results/20240417/{fname}_{channel}_processed.png', cv2.cvtColor(processed_np[0,:,:,channel], cv2.COLOR_RGB2BGR))