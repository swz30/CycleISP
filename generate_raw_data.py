"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from networks.cycleisp import Rgb2Raw
from dataloaders.data_rgb import get_rgb_data
from utils.noise_sampling import random_noise_levels_dnd, random_noise_levels_sidd, add_noise
import utils
import lycon
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='RGB2RAW Network: From clean RGB images, generate {RAW_clean, RAW_noisy} pairs')
parser.add_argument('--input_dir', default='./datasets/sample_rgb_images/',
    type=str, help='Directory of clean RGB images')
parser.add_argument('--result_dir', default='./results/synthesized_data/raw/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/isp/rgb2raw.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save synthesized images in result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir+'pkl')
utils.mkdir(args.result_dir+'png/clean')
utils.mkdir(args.result_dir+'png/noisy')

test_dataset = get_rgb_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

model_rgb2raw = Rgb2Raw()

utils.load_checkpoint(model_rgb2raw,args.weights)
print("===>Testing using weights: ", args.weights)

model_rgb2raw.cuda()

model_rgb2raw=nn.DataParallel(model_rgb2raw)

model_rgb2raw.eval()

with torch.no_grad():
    for ii, data in enumerate(tqdm(test_loader), 0):
        rgb_gt    = data[0].cuda()
        filenames = data[1]
        padh = data[2]
        padw = data[3]
        ## Convert clean rgb image to clean raw image
        raw_gt = model_rgb2raw(rgb_gt)       ## raw_gt is in RGGB format
        raw_gt = torch.clamp(raw_gt,0,1)
        
        ########## Add noise to clean raw images ##########
        for j in range(raw_gt.shape[0]):
            filename = filenames[j]
            shot_noise, read_noise = random_noise_levels_dnd() 
            shot_noise, read_noise = shot_noise.cuda(), read_noise.cuda()
            raw_noisy = add_noise(raw_gt[j], shot_noise, read_noise, use_cuda=True)
            raw_noisy = torch.clamp(raw_noisy,0,1)  ### CLIP NOISE
            variance = shot_noise * raw_noisy + read_noise

            #### Unpadding and saving
            clean_packed = raw_gt[j]
            clean_packed = clean_packed[:,padh[j]//2:-padh[j]//2,padw[j]//2:-padw[j]//2]   ## RGGB channels  (4 x H/2 x W/2)
            clean_unpacked = utils.unpack_raw(clean_packed.unsqueeze(0))                   ## Rearrange RGGB channels into Bayer pattern
            clean_unpacked = clean_unpacked.squeeze().cpu().detach().numpy()
            lycon.save(args.result_dir+'png/clean/'+filename[:-4]+'.png',img_as_ubyte(clean_unpacked))

            noisy_packed = raw_noisy
            noisy_packed = noisy_packed[:,padh[j]//2:-padh[j]//2,padw[j]//2:-padw[j]//2]   ## RGGB channels
            noisy_unpacked = utils.unpack_raw(noisy_packed.unsqueeze(0))                   ## Rearrange RGGB channels into Bayer pattern
            noisy_unpacked = noisy_unpacked.squeeze().cpu().detach().numpy()
            lycon.save(args.result_dir+'png/noisy/'+filename[:-4]+'.png',img_as_ubyte(noisy_unpacked))

            variance_packed = variance[:,padh[j]//2:-padh[j]//2,padw[j]//2:-padw[j]//2]   ## RGGB channels

            dict_ = {}
            dict_['clean'] = clean_packed.cpu().detach().numpy()       ## (4 x H/2 x W/2)
            dict_['noisy'] = noisy_packed.cpu().detach().numpy()       ## (4 x H/2 x W/2)
            dict_['variance'] = variance_packed.cpu().detach().numpy() ## (4 x H/2 x W/2)
            dict_['shot_noise'] = shot_noise.cpu().detach().numpy()
            dict_['read_noise'] = read_noise.cpu().detach().numpy()
            utils.save_dict(dict_, args.result_dir+'pkl/'+filename[:-4]+'.pkl')
