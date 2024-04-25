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

"""Unprocesses sRGB images into realistic raw data.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import torch

def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = torch.tensor([[[1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181]]], dtype=torch.float32)
    num_ccms = len(xyz2cams)
    weights = (1e8 - 1e-8)*torch.rand((num_ccms, 1, 1)) + 1e-8
    weights_sum = torch.sum(weights, dim=0)
    xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)
    rgb2cam = torch.matmul(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    return rgb2cam


def random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / (0.1 * torch.randn(1) + 0.8)
    # Red and blue gains represent white balance.
    # X = sigma * Z + mu
    red_gain = (2.4 - 1.9) * torch.rand(1) + 1.9
    blue_gain = (1.9 - 1.5) * torch.rand(1) + 1.5
    return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = torch.clamp(image, min=0.0, max=1.0)
    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)


def gamma_expansion(image, gamma=2.2):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return torch.maximum(image, torch.tensor(1e-8)) ** gamma


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    shape = image.shape
    image = torch.reshape(image, (-1, 3))
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    return torch.reshape(image, shape)

def change_constast(image, constast=1.0, brightness=0.0):
    return (constast*image + brightness)

def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    gains = torch.tensor([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
    gains = gains[None, None, None, :]
    # gains = torch.view(gains)
    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = torch.mean(image, dim=-1, keepdim=True)
    inflection = 0.9
    mask = (torch.maximum(gray - inflection, torch.tensor(0.0)) / (1.0 - inflection)) ** 2.0
    safe_gains = torch.maximum(mask + (1.0 - mask) * gains, gains)

    return image * safe_gains


def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    red = image[:, 0::2, 0::2, 0]
    green_red = image[:, 0::2, 1::2, 1]
    green_blue = image[:, 1::2, 0::2, 1]
    blue = image[:, 1::2, 1::2, 2]
    image = torch.stack((red, green_red, green_blue, blue), axis=-1)
    image = torch.reshape(image, (-1, shape[1] // 2, shape[2] // 2, 4))
    return image


def unprocess(image, random_ccm_tensor=None, random_gains_list=None, contrast=1.0, brightness=0.0):
    """Unprocesses an image from sRGB to realistic raw data."""
    # Randomly creates image metadata.
    if random_ccm_tensor == None:
        rgb2cam = random_ccm()
    else:
        rgb2cam = random_ccm_tensor

    cam2rgb = torch.linalg.inv(rgb2cam)
    
    if random_gains_list == None:
        rgb_gain, red_gain, blue_gain = random_gains()
    else:
        rgb_gain, red_gain, blue_gain = random_gains_list
    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Change constrast before inverts WB
    image = change_constast(image, contrast, brightness)
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = torch.clamp(image, min=0.0, max=1.0)

    # Applies a Bayer mosaic.
    image = mosaic(image)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return image, metadata


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = torch.log(torch.tensor(0.0001))
    log_max_shot_noise = torch.log(torch.tensor(0.012))

    log_shot_noise = (log_max_shot_noise - log_min_shot_noise) * torch.rand(1) + log_min_shot_noise
    shot_noise = torch.exp(log_shot_noise[0])

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + 0.26 * torch.randn(1)
    read_noise = torch.exp(log_read_noise[0])

    return shot_noise, read_noise

def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    noise = torch.normal(mean=0, std=torch.sqrt(variance))
    return image + noise