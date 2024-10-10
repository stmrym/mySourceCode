from functools import wraps
import time
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
import re
import torch
import torchvision
import tensorboardX
from torchvision import transforms as transforms

def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print("{} min in {}".format(elapsed_time/60, func.__name__))
        return result
    return wrapper


class Imagedata():
    def __init__(self, image_path, crop_size=480):

        self.image_path = image_path
        self.pil_image = Image.open(self.image_path)
        self.image = self.transform(self.pil_image)
        
        if 'GOPRO' in str(self.image_path):
            self.label = 'GoPro'
        elif 'BSD' in str(image_path):
            self.label = 'BSD_3ms24ms'
        else:
            self.label = 'Unknown'


class Mydatasets():
    def __init__(self, dir_path_list, crop_size=480):
        self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor()
            ])  
        self.label_img_transform = transforms.Resize(50)
        
        self.image_paths = []
        for dir_path in dir_path_list:
            self.image_paths += sorted([p for p in Path(dir_path).glob('**/*.png') if re.search('blur_gamma|Blur', str(p))])
        
        self.image_dict_list = []
        for path in self.image_paths:
            self.image_dict = {}
            self.image_dict['image'] = Image.open(path)
            if 'GOPRO' in str(path):
                self.image_dict['label'] = 'GoPro'
            elif 'BSD' in str(path):
                self.image_dict['label'] = 'BSD_3ms24ms'
            else:
                self.image_dict['label'] = 'Unknown'
            self.image_dict_list.append(self.image_dict)

    def random_sample(self, n_sample, seed=0):
            random.seed(seed)
            self.image_dict_list = random.sample(self.image_dict_list, n_sample)

    def crop_center(self, pil_img, crop_size=480):
        w, h = pil_img.size
        return pil_img.crop(((w - crop_size) // 2, (h - crop_size) // 2,
                            (w + crop_size) // 2, (h + crop_size) // 2))

    @stop_watch
    def concat_image_list(self):
        self.label_img = torch.stack([self.transform(image['image']) for image in tqdm(self.image_dict_list)], dim=0)
        self.feat = self.label_img.flatten(1)
        self.label_img = self.label_img_transform(self.label_img)
        self.label_list = [image['label']for image in tqdm(self.image_dict_list)]
        
        print(self.label_img.shape)
        print(self.feat.shape)
        print(len(self.label_list))

    @stop_watch
    def run_tensorboard(self):
        print('TensorboardX started...')
        writer = tensorboardX.SummaryWriter()
        writer.add_embedding(self.feat, metadata=self.label_list, label_img=self.label_img)  
        writer.close()      
        print('Wrote.')


if __name__ == '__main__':

    dir_path_list = [
        '../dataset/BSD_3ms24ms/test',
        '../dataset/GOPRO_Large/test'
    ]

    dataset = Mydatasets(dir_path_list)
    dataset.random_sample(500)
    dataset.concat_image_list()
    dataset.run_tensorboard()

