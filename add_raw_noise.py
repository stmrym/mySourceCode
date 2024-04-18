import cv2
import numpy as np
import torch
from unprocess_torch import unprocess, random_noise_levels, add_noise
from process_torch import process


if __name__ == '__main__':
    fname = '00097'
    ext = 'png'
    path = f'/mnt/d/results/20240417/{fname}.{ext}'
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(np.array(image/255, dtype=np.float32), cv2.COLOR_BGR2RGB)
    image = torch.tensor(image, dtype=torch.float32)
    raw_image, metadata = unprocess(image)
    lambda_shot, lambda_read = random_noise_levels()
    noise_raw_image = add_noise(raw_image, lambda_shot, lambda_read)
    
    # raw_np = (noise_raw_image.numpy()*255).astype(np.uint8)
    # cv2.imwrite(f'/mnt/d/results/20240417/{fname}_raw.png', cv2.cvtColor(raw_np[:,:,:], cv2.COLOR_RGB2BGR))
    # for item in metadata.items():
        # print(item)

    # for channel in range(0, raw_np.shape[-1]):
        # print(np.max(raw_np[:,:,channel]), np.min(raw_np[:,:,channel]))
        # cv2.imwrite(f'/mnt/d/results/20240417/{fname}_{channel}_raw.png', raw_np[:,:,channel])

    concat_raw_image = torch.stack([raw_image, noise_raw_image], dim=0)
    red_gains = torch.cat([metadata['red_gain'], metadata['red_gain']], dim=0)
    blue_gains = torch.cat([metadata['blue_gain'], metadata['blue_gain']], dim=0)
    cam2rgbs = torch.stack([metadata['cam2rgb'], metadata['cam2rgb']], dim=0)

    processed = process(concat_raw_image, red_gains, blue_gains, cam2rgbs)
    print(processed.shape)
    processed_np = (processed.numpy()*255).astype(np.uint8)
    for batch in range(0, processed_np.shape[0]):
        cv2.imwrite(f'/mnt/d/results/20240417/{fname}_{batch}_processed.png', cv2.cvtColor(processed_np[batch, :,:,:], cv2.COLOR_RGB2BGR))
    
    # for channel in range(0, processed_np.shape[-1]):
    #     cv2.imwrite(f'/mnt/d/results/20240417/{fname}_{channel}_processed.png', cv2.cvtColor(processed_np[0,:,:,channel], cv2.COLOR_RGB2BGR))