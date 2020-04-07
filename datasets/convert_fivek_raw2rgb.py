

## Convert MIT-Adobe Fivek RAW images to RGB using Rawpy library


from __future__ import division
import os
import numpy as np
import rawpy
from glob import glob
from tqdm import tqdm
import lycon
from natsort import natsorted
import shutil

from joblib import Parallel, delayed
import multiprocessing


input_dir = './fivek_dataset/RAW/'
output_dir = './fivek_dataset/RGB_jpg/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

files = natsorted(glob(input_dir+'/*.NEF') + glob(input_dir+'/*.DNG') + glob(input_dir+'/*.dng'))

def raw2rgb(inp_path,out_path):
    filename = inp_path.split('/')[-1].rsplit('.', 1)[0]+'.jpg'
    filepath = out_path+'/'+filename
   
    raw = rawpy.imread(inp_path)
    im = raw.postprocess(use_camera_wb=True, half_size=False, user_flip = 0, no_auto_bright=True, output_bps=8)
    
    im_raw = raw.raw_image_visible.astype(np.float32)    

    if im_raw.shape[0]!= im.shape[0] or im_raw.shape[1]!= im.shape[1]:
        print('Dimension Mismatch in Image: ', filename)
        print('Moving: ', filename[:-4]+'.dng')
        shutil.move(inp_path,'./corrupt/FinePixS2Pro/')
    else:
        lycon.save(filepath,im)

num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(raw2rgb)(path,output_dir) for path in tqdm(files))