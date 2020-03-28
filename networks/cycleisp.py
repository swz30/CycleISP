"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.GaussianBlur import get_gaussian_kernel

##########################################################################ssss

def mosaic(images):
  """Extracts RGGB Bayer planes from RGB image."""
  # import pdb;pdb.set_trace()
  shape = images.shape
  red = images[:, 0, 0::2, 0::2]
  green_red = images[:, 1, 0::2, 1::2]
  green_blue = images[:, 1, 1::2, 0::2]
  blue = images[:, 2, 1::2, 1::2]
  images = torch.stack((red, green_red, green_blue, blue), dim=1)
  # images = tf.reshape(images, (shape[0] // 2, shape[1] // 2, 4))
  return images



##########################################################################

def conv(in_channels, out_channels, kernel_size, bias=True, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)



##########################################################################

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
##########################################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

##########################################################################


## Dual Attention Block (DAB)
class DAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(DAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        
        self.SA = spatial_attn_layer()            ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)     ## Channel Attention
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1)


    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

##########################################################################


## Recursive Residual Group (RRG)
class RRG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act,  num_dab):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [
            DAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act) \
            for _ in range(num_dab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
##########################   RGB2RAW Network  ############################
##########################################################################

class Rgb2Raw(nn.Module):
    def __init__(self, conv=conv):
        super(Rgb2Raw, self).__init__()        
        input_nc  = 3
        output_nc = 4

        num_rrg = 3
        num_dab = 5
        n_feats = 96
        kernel_size = 3
        reduction = 8


        act =nn.PReLU(n_feats)

        modules_head = [conv(input_nc, n_feats, kernel_size = kernel_size, stride = 1)]

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act) 

        modules_tail = [conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)


    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = mosaic(x)
        return x 


##########################################################################
################ Color Correction Network  #############################
##########################################################################

class CCM(nn.Module):
    def __init__(self,  conv=conv):
        super(CCM, self).__init__()        
        input_nc  = 3
        output_nc = 96

        num_rrg = 2
        num_dab = 2
        n_feats = 96
        kernel_size = 3
        reduction = 8

        sigma = 12 ## GAUSSIAN_SIGMA

        act =nn.PReLU(n_feats)


        modules_head = [conv(input_nc, n_feats, kernel_size = kernel_size, stride = 1)]

        modules_downsample = [nn.MaxPool2d(kernel_size=2)] 
        self.downsample = nn.Sequential(*modules_downsample)

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act) 

        modules_tail = [conv(n_feats, output_nc, kernel_size),nn.Sigmoid()]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.blur, self.pad = get_gaussian_kernel(sigma=sigma)


    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x = self.blur(x)
        x = self.head(x)
        x = self.downsample(x)  
        x = self.body(x)
        x = self.tail(x)
        return x


##########################################################################
##########################   RAW2RGB Network  ############################
##########################################################################
class Raw2Rgb(nn.Module):
    def __init__(self, conv=conv):
        super(Raw2Rgb, self).__init__()        
        input_nc  = 4
        output_nc = 3

        num_rrg = 3
        num_dab =5
        n_feats = 96
        kernel_size = 3
        reduction = 8

        act =nn.PReLU(n_feats)

        modules_head = [conv(input_nc, n_feats, kernel_size = kernel_size, stride = 1)]

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act) 

        modules_tail = [conv(n_feats, n_feats, kernel_size), act]
        modules_tail_rgb = [conv(n_feats, output_nc*4, kernel_size=1)]#, nn.Sigmoid()]        

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.tail_rgb = nn.Sequential(*modules_tail_rgb)

        conv1x1 = [conv(n_feats*2, n_feats, kernel_size=1)]
        self.conv1x1 = nn.Sequential(*conv1x1)


    def forward(self, x, ccm_feat):
        x = self.head(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
        body_out = x.clone()
        x = x * ccm_feat          ## Attention 
        x = x + body_out       
        x = self.body[-1](x)
        x = self.tail(x)
        x = self.tail_rgb(x)
        x = nn.functional.pixel_shuffle(x, 2)
        return x
