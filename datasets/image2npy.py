

## Convert images into npys and normalize them in range [0,1]

from __future__ import division
import os
import numpy as np
import rawpy
from glob import glob
from tqdm import tqdm
import lycon
from natsort import natsorted

from joblib import Parallel, delayed
import multiprocessing

dir_ = './fivek_dataset/'
img_type = 'jpg'          # change 'jpg' to 'raw' for converting raw images to npys

if img_type == 'raw':
    input_dir = dir_  + 'RAW'
    output_dir = dir_ + 'RAW_npy'
    files =  glob(input_dir+'/*.NEF') + glob(input_dir+'/*.DNG') + glob(input_dir+'/*.dng')

if img_type == 'jpg':
    input_dir = dir_  + 'RGB_jpg'
    output_dir = dir_ + 'RGB_npy'
    files =  glob(input_dir+'/*.jpg')

os.makedirs(output_dir, exist_ok = True)

files = natsorted(files)

def bit_depth(x):
    return np.ceil(np.log(x.max())/np.log(2))

def raw2npy(inp_path):
    filename = os.path.splitext(os.path.split(inp_path)[-1])[0] + '.npy'
    filepath = os.path.join(output_dir,filename)
    raw = rawpy.imread(inp_path)
    im_raw = raw.raw_image_visible.astype(np.float32)
    # normalize values in range [0,1]
    norm_factor = 2 ** bit_depth(im_raw)
    im_raw = im_raw/norm_factor
    im_raw = im_raw[..., np.newaxis]
    np.save(filepath,im_raw)

def jpg2npy(inp_path):
    filename = os.path.splitext(os.path.split(inp_path)[-1])[0] + '.npy'
    filepath = os.path.join(output_dir,filename)
    jpg = lycon.load(inp_path)
    im_jpg = jpg.astype(np.float16)
     # normalize values in range [0,1]
    norm_factor = 255
    im_jpg = im_jpg/norm_factor
    np.save(filepath,im_jpg)

num_cores = multiprocessing.cpu_count()

from concurrent.futures import ProcessPoolExecutor

if img_type == 'raw':
    with ProcessPoolExecutor(num_cores) as e: e.map(raw2npy, files)
if img_type == 'jpg':
    with ProcessPoolExecutor(num_cores) as e: e.map(jpg2npy, files)