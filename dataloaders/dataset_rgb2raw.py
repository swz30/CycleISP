import numpy as np
import os
from torch.utils.data import Dataset
import torch

from utils.image_utils import is_numpy_file, load_npy, pack_raw, load_dict
from utils.dataset_utils import Augment_Bayer, bayer_unify

augment   = Augment_Bayer()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


class DataLoaderTrain(Dataset):
    def __init__(self, raw_dir, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        self.pkl_bayer_patterns = load_dict('./datasets/fivek_bayer.pkl')

        rgb_files=sorted(os.listdir(rgb_dir))
        raw_files=sorted(os.listdir(raw_dir))

        self.rgb_filenames = [os.path.join(rgb_dir, x) for x in rgb_files if is_numpy_file(x)]
        self.raw_filenames = [os.path.join(raw_dir, x) for x in raw_files if is_numpy_file(x)]
        
        self.img_options=img_options
        self.rgb_size = len(self.rgb_filenames)  # get the size of input
        self.raw_size = len(self.raw_filenames)  # get the size of target

    def __len__(self):
        return max(self.rgb_size, self.raw_size)

    def __getitem__(self, index):
        rgb_index = index % self.rgb_size
        raw_index   = index % self.raw_size
        
        filename = os.path.splitext(os.path.split(self.rgb_filenames[rgb_index])[-1])[0]
        bayer_pattern = self.pkl_bayer_patterns[filename]

        ## Load Images
        rgb_image  = load_npy(self.rgb_filenames[rgb_index])
        raw_image = load_npy(self.raw_filenames[raw_index])


        #Extract random crops from rgb and raw images
        ps = self.img_options['patch_size']
        ps_temp = ps*2 + 16
        H = raw_image.shape[0]
        W = raw_image.shape[1]
        r = np.random.randint(0, H - ps_temp)
        c = np.random.randint(0, W - ps_temp)
        if r%2!=0: r = r-1
        if c%2!=0: c = c-1
        rgb_patch = rgb_image[r:r + ps_temp, c:c + ps_temp, :]
        raw_patch = raw_image[r:r + ps_temp, c:c + ps_temp, :]


        raw_patch, rgb_patch = bayer_unify(raw_patch.squeeze(), rgb_patch, bayer_pattern, "RGGB", "crop")

        #Apply Bayer Augmentation
        indx = np.random.randint(0,len(transforms_aug))
        apply_trans = transforms_aug[indx]

        raw_patch, rgb_patch = getattr(augment, apply_trans)(raw_patch[...,np.newaxis], rgb_patch)

        #Pack Target
        raw_patch = pack_raw(raw_patch)

        # Extract crops of desired patch size
        H = raw_patch.shape[0]
        W = raw_patch.shape[1]
        r = (H - ps) // 2
        c = (W - ps) // 2
        PS, R, C = ps*2, r*2, c*2
        rgb_patch = rgb_patch[R:R + PS, C:C + PS, :]
        raw_patch = raw_patch[r:r + ps, c:c + ps, :]
        
        rgb_patch = torch.Tensor(rgb_patch).permute(2,0,1)
        raw_patch = torch.Tensor(raw_patch).permute(2,0,1)

        return rgb_patch,raw_patch 


