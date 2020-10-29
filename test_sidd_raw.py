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
from networks.denoising_raw import DenoiseNet
from dataloaders.data_raw import get_validation_data 

import utils
import lycon
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='RAW denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='./datasets/sidd/sidd_raw/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/sidd/sidd_raw/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/denoising/sidd_raw.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

model_restoration = DenoiseNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()

with torch.no_grad():
    psnr_val_raw = []
    for ii, data_val in enumerate(tqdm(test_loader), 0):
        raw_gt = data_val[0].cuda()
        raw_noisy = data_val[1].cuda()
        variance = data_val[2].cuda()       ##variance = shot_noise * raw_noisy + read_noise  (Shot and Read noise comes from images' metadata)
        filenames = data_val[3]
        raw_restored = model_restoration(raw_noisy, variance)
        raw_restored = torch.clamp(raw_restored,0,1)                
        psnr_val_raw.append(utils.batch_PSNR(raw_restored, raw_gt, 1.))

        if args.save_images:
            for batch in range(len(raw_gt)):
                denoised_img = utils.unpack_raw(raw_restored[batch,:,:,:].unsqueeze(0))
                denoised_img = denoised_img.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
                denoised_img = np.squeeze(np.stack((denoised_img,) * 3, -1))
                lycon.save(args.result_dir + filenames[batch][:-4] + '.png', img_as_ubyte(denoised_img))
                

psnr_val_raw = sum(psnr_val_raw)/len(psnr_val_raw)
print("PSNR: %.2f " %(psnr_val_raw))
