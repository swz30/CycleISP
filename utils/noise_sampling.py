"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

## We adopt the same noise sampling procedure as in "Unprocessing Images for Learned Raw Denoising" by Brooks et al. CVPR2019

import torch
import torch.distributions as dist
import numpy as np




################ If the target dataset is DND, use this function #####################
def random_noise_levels_dnd():
  """Generates random noise levels from a log-log linear distribution."""
  log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
  log_max_shot_noise = torch.log10(torch.Tensor([0.012]))
  distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)

  log_shot_noise = distribution.sample()
  shot_noise = torch.pow(10,log_shot_noise)
  distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.26]))
  read_noise = distribution.sample()
  line = lambda x: 2.18 * x + 1.20
  log_read_noise = line(log_shot_noise) + read_noise
  read_noise = torch.pow(10,log_read_noise)
  return shot_noise, read_noise

################ If the target dataset is SIDD, use this function #####################
def random_noise_levels_sidd():
  """ Where read_noise in SIDD is not 0 """
  log_min_shot_noise = torch.log10(torch.Tensor([0.00068674]))
  log_max_shot_noise = torch.log10(torch.Tensor([0.02194856]))
  distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)

  log_shot_noise = distribution.sample()
  shot_noise = torch.pow(10,log_shot_noise)

  distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.20]))
  read_noise = distribution.sample()
  line = lambda x: 1.85 * x + 0.30  ### Line SIDD test set
  log_read_noise = line(log_shot_noise) + read_noise
  read_noise = torch.pow(10,log_read_noise)
  return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005, use_cuda=False):
  """Adds random shot (proportional to image) and read (independent) noise."""
  variance = image * shot_noise + read_noise
  mean = torch.Tensor([0.0])
  if use_cuda:
    mean = mean.cuda()
  distribution = dist.normal.Normal(mean, torch.sqrt(variance))
  noise = distribution.sample()
  return image + noise

