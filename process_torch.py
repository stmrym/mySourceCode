# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def apply_gains(bayer_images, red_gains, blue_gains):
    """Applies white balance gains to a batch of Bayer images."""
    # (None, None, None, 4)
    assert bayer_images.shape[3] == 4, 'bayer_images shape does not match'
    green_gains = torch.ones_like(red_gains)
    gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], axis=-1)
    gains = gains[:, None, None, :]
    return bayer_images * gains


def demosaic(bayer_images):
    """Bilinearly demosaics a batch of RGGB Bayer images."""
    # (B, H, W, 4)
    assert bayer_images.shape[3] == 4, 'bayer_images shape does not match'
    # This implementation exploits how edges are aligned when upsampling with
    # tf.image.resize_bilinear().

    shape = bayer_images.shape
    shape = [shape[1] * 2, shape[2] * 2]

    red = bayer_images[Ellipsis, 0:1]

    red = F.interpolate(red.permute(0,3,1,2), size=shape, align_corners=True, mode='bilinear')
    pixel_unshuffle = torch.nn.PixelUnshuffle(2)
    pixel_shuffle = torch.nn.PixelShuffle(2)
    green_red = bayer_images[Ellipsis, 1:2]
    green_red = torch.fliplr(green_red.permute(0,3,1,2))
    green_red = F.interpolate(green_red, size=shape, align_corners=True, mode='bilinear')
    green_red = torch.fliplr(green_red)
    green_red = pixel_unshuffle(green_red)

    green_blue = bayer_images[Ellipsis, 2:3]
    green_blue = torch.flipud(green_blue.permute(0,3,1,2))
    green_blue = F.interpolate(green_blue, size=shape, align_corners=True, mode='bilinear')
    green_blue = torch.flipud(green_blue)
    green_blue = pixel_unshuffle(green_blue)

    green_at_red = (green_red[:, 0, ...] + green_blue[:, 0, ...]) / 2
    green_at_green_red = green_red[:, 1, ...]
    green_at_green_blue = green_blue[:, 2, ...]
    green_at_blue = (green_red[:, 3, ...] + green_blue[:, 3, ...]) / 2

    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = pixel_shuffle(torch.stack(green_planes, dim=1))

    blue = bayer_images[Ellipsis, 3:4]
    blue = torch.flipud(torch.fliplr(blue.permute(0,3,1,2)))
    blue = F.interpolate(blue, size=shape, align_corners=True, mode='bilinear')
    blue = torch.flipud(torch.fliplr(blue))
    
    rgb_images = torch.concat([red, green, blue], dim=1).permute(0,2,3,1)

    return rgb_images

def demosaic_opencv(bayer_images):
    pixel_shuffle = torch.nn.PixelShuffle(2)
    bayer_images_up = pixel_shuffle(bayer_images.permute(0,3,1,2))
    bayer_images_np = (bayer_images_up.permute(0,2,3,1).numpy()*255).astype(np.uint8)
    rgb_images_np = [cv2.cvtColor(bayer_images_np[b], cv2.COLOR_BayerRGGB2RGB_VNG) for b in range(0, bayer_images_np.shape[0])]
    rgb_images_tensor = torch.from_numpy(np.stack(rgb_images_np))/255
    return rgb_images_tensor

def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    assert len(images.shape) == 4, 'bayer_images shape does not match'
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]

    return torch.sum(images * ccms, dim=-1)


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return torch.maximum(images, torch.tensor(1e-8)) ** (1.0 / gamma)

def smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = torch.clamp(image, min=0.0, max=1.0)
    return 3.0 * image**2 - 2.0 * image**3

def process(bayer_images, red_gains, blue_gains, cam2rgbs):
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # White balance.
    bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = demosaic_opencv(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images)
    images = smoothstep(images)
    return images
