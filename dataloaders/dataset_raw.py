import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils.image_utils import  is_pkl_file, load_pkl, pack_raw_torch



class DataLoaderVal(Dataset):
    def __init__(self, gt_dir, img_options=None, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_files=sorted(os.listdir(gt_dir))

        self.target_filenames = [os.path.join(gt_dir, x) for x in gt_files if is_pkl_file(x)]
        
        self.img_options=img_options
        self.tar_size = len(self.target_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        target = load_pkl(self.target_filenames[tar_index])

        clean, noisy, variance = target['clean'], target['noisy'], target['variance']

        tar_filename = os.path.split(self.target_filenames[tar_index])[-1]

        clean = torch.from_numpy(clean).permute(2,0,1)
        noisy = torch.from_numpy(noisy).permute(2,0,1)
        variance = torch.from_numpy(variance).permute(2,0,1)

        return clean, noisy, variance, tar_filename



class DataLoaderTest(Dataset):
    def __init__(self, inp_dir,input_transform=None):
        super(DataLoaderTest, self).__init__()

        self.input_transform = input_transform

        inp_files=sorted(os.listdir(inp_dir))

        self.input_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_pkl_file(x)]

        self.inp_size = len(self.input_filenames)  # get the size of input

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        input = load_pkl(self.input_filenames[index])

        # if self.input_transform:
        #     input= self.input_transform(input)

        input_filename = os.path.split(self.input_filenames[index])[-1]
        raw_noisy = input['image']
        # raw_noisy = raw_noisy[:128,:128,:]
        raw_noisy = torch.Tensor(raw_noisy)
        raw_noisy_orig = raw_noisy.clone()
        raw_noisy = pack_raw_torch(raw_noisy)
        raw_noisy = raw_noisy.permute(2,0,1)
        raw_noisy_orig = raw_noisy_orig.permute(2,0,1)
        bayer = input['pattern']
        return raw_noisy, bayer, input_filename, raw_noisy_orig