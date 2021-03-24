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
from networks.cycleisp import Rgb2Raw, Raw2Rgb, CCM
from dataloaders.data_rgb import get_rgb_data
from utils.noise_sampling import random_noise_levels_dnd, random_noise_levels_sidd, add_noise
import utils
import lycon
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='From clean RGB images, generate {RGB_clean, RGB_noisy} pairs')
parser.add_argument('--input_dir', default='./datasets/sample_rgb_images/',
    type=str, help='Directory of clean RGB images')
parser.add_argument('--result_dir', default='./results/synthesized_data/rgb/',
    type=str, help='Directory for results')
parser.add_argument('--weights_rgb2raw', default='./pretrained_models/isp/rgb2raw_joint.pth', type=str, help='weights rgb2raw branch')
parser.add_argument('--weights_raw2rgb', default='./pretrained_models/isp/raw2rgb_joint.pth', type=str, help='weights raw2rgb branch')
parser.add_argument('--weights_ccm', default='./pretrained_models/isp/ccm_joint.pth', type=str, help='weights ccm branch')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save synthesized images in result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir+'clean')
utils.mkdir(args.result_dir+'noisy')

test_dataset = get_rgb_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

model_rgb2raw = Rgb2Raw()
model_ccm     = CCM()
model_raw2rgb = Raw2Rgb()

utils.load_checkpoint(model_rgb2raw,args.weights_rgb2raw)
utils.load_checkpoint(model_ccm,args.weights_ccm)
utils.load_checkpoint(model_raw2rgb,args.weights_raw2rgb)

model_rgb2raw.cuda()
model_ccm.cuda()
model_raw2rgb.cuda()

model_rgb2raw = nn.DataParallel(model_rgb2raw)
model_ccm = nn.DataParallel(model_ccm)
model_raw2rgb = nn.DataParallel(model_raw2rgb)

model_rgb2raw.eval()
model_ccm.eval()
model_raw2rgb.eval()

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
        for j in range(raw_gt.shape[0]):   ## Use loop to add different noise to different images.
            filename = filenames[j]
            shot_noise, read_noise = random_noise_levels_dnd() 
            shot_noise, read_noise = shot_noise.cuda(), read_noise.cuda()
            raw_noisy = add_noise(raw_gt[j], shot_noise, read_noise, use_cuda=True)
            raw_noisy = torch.clamp(raw_noisy,0,1)  ### CLIP NOISE
            
            #### Convert raw noisy to rgb noisy ####
            ccm_tensor = model_ccm(rgb_gt[j].unsqueeze(0))
            rgb_noisy = model_raw2rgb(raw_noisy.unsqueeze(0),ccm_tensor) 
            rgb_noisy = torch.clamp(rgb_noisy,0,1)

            rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy()

            rgb_clean = rgb_gt[j].permute(1,2,0).cpu().detach().numpy()

            ## Unpadding
            rgb_clean = rgb_clean[padh[j]:-padh[j],padw[j]:-padw[j],:].copy()
            rgb_noisy = rgb_noisy[padh[j]:-padh[j],padw[j]:-padw[j],:].copy()   

            lycon.save(args.result_dir+'clean/'+filename[:-4]+'.png',img_as_ubyte(rgb_clean))
            lycon.save(args.result_dir+'noisy/'+filename[:-4]+'.png',img_as_ubyte(rgb_noisy))
