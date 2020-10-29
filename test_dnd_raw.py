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
import h5py
import numpy as np
from networks.denoising_raw import DenoiseNet

import utils
import lycon
from utils.bundle_submissions import bundle_submissions_raw
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='RAW denoising evaluation on the DND dataset')
parser.add_argument('--input_dir', default='./datasets/dnd/dnd_raw/',
    type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='./results/dnd/dnd_raw/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/denoising/dnd_raw.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir+'matfile')
utils.mkdir(args.result_dir+'png')


model_restoration = DenoiseNet()


utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()


def denoiser(raw_noisy,variance):
    raw_noisy = torch.Tensor(raw_noisy).unsqueeze(0).permute(0,3,1,2).cuda()
    variance = torch.Tensor(variance).unsqueeze(0).permute(0,3,1,2).cuda()
    # Predict  
    with torch.no_grad():          
        raw_restored  = model_restoration(raw_noisy, variance)
    raw_restored = torch.clamp(raw_restored,0,1)                
    
    raw_restored = raw_restored.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    return raw_restored


# info = h5py.File(os.path.join(data_dir, 'info.mat'), 'r')['info']
info = h5py.File(args.input_dir +'info.mat', 'r')['info']
bb = info['boundingboxes']

# Denoise each image.
for i in tqdm(range(50)):
    # Loads the noisy image.
    filename = os.path.join(args.input_dir, '%04d.mat' % (i + 1))
    img = h5py.File(filename, 'r')
    noisy = np.float32(np.array(img['Inoisy']).T)

    # Loads raw Bayer color pattern.
    bayer_pattern = np.asarray(info[info['camera'][0][i]]['pattern']).tolist()

    # Denoises each bounding box in this image.
    boxes = np.array(info[bb[0][i]]).T
    metadata = {}
    for k in range(20):
        # Crops the image to this bounding box.
        idx = [
          int(boxes[k, 0] - 1),
          int(boxes[k, 2]),
          int(boxes[k, 1] - 1),
          int(boxes[k, 3])
        ]
        noisy_crop = noisy[idx[0]:idx[1], idx[2]:idx[3]].copy()
        noisy_crop_orig = noisy_crop.copy()

        # Flips the raw image to ensure RGGB Bayer color pattern.
        if (bayer_pattern == [[1, 2], [2, 3]]):
            pass
        elif (bayer_pattern == [[2, 1], [3, 2]]):
            noisy_crop = np.fliplr(noisy_crop)
        elif (bayer_pattern == [[2, 3], [1, 2]]):
            noisy_crop = np.flipud(noisy_crop)
        else:
            print('Warning: assuming unknown Bayer pattern is RGGB.')

        # Loads shot and read noise factors.
        nlf_h5 = info[info['nlf'][0][i]]
        shot_noise = nlf_h5['a'][0][0]
        read_noise = nlf_h5['b'][0][0]

        # Extracts each Bayer image plane.
        denoised_crop = noisy_crop.copy()
        height, width = noisy_crop.shape
        channels = []
        for yy in range(2):
            for xx in range(2):
                noisy_crop_c = noisy_crop[yy:height:2, xx:width:2].copy()
                channels.append(noisy_crop_c)
        channels = np.stack(channels, axis=-1)
        variance = shot_noise * channels + read_noise
        
        # Denoises this crop of the image.
        output = denoiser(channels, variance)

        # Copies denoised results to output denoised array.
        for yy in range(2):
            for xx in range(2):
                denoised_crop[yy:height:2, xx:width:2] = output[:, :, 2 * yy + xx]

        # Flips denoised image back to original Bayer color pattern.
        if (bayer_pattern == [[1, 2], [2, 3]]):
            pass
        elif (bayer_pattern == [[2, 1], [3, 2]]):
            denoised_crop = np.fliplr(denoised_crop)
        elif (bayer_pattern == [[2, 3], [1, 2]]):
            denoised_crop = np.flipud(denoised_crop)

        Idenoised_crop = np.clip(np.float32(denoised_crop), 0.0, 1.0)
        
        # Saves denoised image crop.
        save_file = os.path.join(args.result_dir+ 'matfile/', '%04d_%02d.mat' % (i + 1, k + 1))
        sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
        
        if args.save_images:
            denoised_img = img_as_ubyte(Idenoised_crop)
            save_file = os.path.join(args.result_dir+ 'png/', '%04d_%02d.png' % (i + 1, k + 1))
            lycon.save(save_file, denoised_img)

bundle_submissions_raw(args.result_dir+'matfile/', 'raw_results_for_server_submission/')
os.system("rm {}".format(args.result_dir+'matfile/*.mat'))
