import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils.image_utils import is_png_file, load_img
from utils.GaussianBlur import get_gaussian_kernel

import torch.nn.functional as F


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'clean')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'noisy')))


        self.clean_filenames = [os.path.join(rgb_dir, 'clean', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'noisy', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'noisy')))


        self.noisy_filenames = [os.path.join(rgb_dir, 'noisy', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.noisy_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2,0,1)

        return noisy, noisy_filename


##################################################################################################

MAX_SIZE = 512    

def divisible_by(img, factor=16):
    h, w, _ = img.shape
    img = img[:int(np.floor(h/factor)*factor),:int(np.floor(w/factor)*factor),:]
    return img

class DataLoader_NoisyData(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoader_NoisyData, self).__init__()

        rgb_files=sorted(os.listdir(rgb_dir))
        
        #print("number of images:", len(rgb_files))
        self.target_filenames = [os.path.join(rgb_dir, x) for x in rgb_files if is_png_file(x)]
        
        self.tar_size = len(self.target_filenames)  # get the size of target
        self.blur, self.pad = get_gaussian_kernel(kernel_size=5, sigma=1)   ### preprocessing to remove noise from the input rgb image

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        target = np.float32(load_img(self.target_filenames[tar_index]))
        
        target = divisible_by(target, 16)

        tar_filename = os.path.split(self.target_filenames[tar_index])[-1]


        target = torch.Tensor(target)
        target = target.permute(2,0,1)
        
        target = F.pad(target.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        target = self.blur(target).squeeze(0)

        padh = (MAX_SIZE - target.shape[1])//2
        padw = (MAX_SIZE - target.shape[2])//2
        target = F.pad(target.unsqueeze(0), (padw, padw, padh, padh), mode='constant').squeeze(0)

        return target, tar_filename, padh, padw