# CycleISP: Real Image Restoration via Improved Data Synthesis (CVPR 2020 -- Oral)

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)

**Paper**: https://arxiv.org/abs/2003.07761

**Supplementary**: [pdf](https://drive.google.com/file/d/1FHOPCrmgYcez2ZTI0-roKEbyeXJIulSJ/view?usp=sharing)

> **Abstract:** *The availability of large-scale datasets has helped unleash the true potential of deep convolutional neural networks (CNNs). However, for the single-image denoising problem, capturing a real dataset is an unacceptably expensive and cumbersome procedure. Consequently, image denoising algorithms are mostly developed and evaluated on synthetic data that is usually generated with a widespread assumption of additive white Gaussian noise (AWGN). While the CNNs achieve impressive results on these synthetic datasets, they do not perform well when applied on real camera images, as reported in recent benchmark datasets. This is mainly because the AWGN is not adequate for modeling the real camera noise which is signal-dependent and heavily transformed by the camera imaging pipeline. In this paper, we present a framework that models camera imaging pipeline in forward and reverse directions. It allows us to produce any number of realistic image pairs for denoising both in RAW and sRGB spaces. By training a new image denoising network on realistic synthetic data, we achieve the state-of-the-art performance on real camera benchmark datasets. The parameters in our model are ~5 times lesser than the previous best method for RAW denoising. Furthermore, we demonstrate that the proposed framework generalizes beyond image denoising problem e.g., for color matching in stereoscopic cinema.* 

## CycleISP for Synthesizing Data for Image Denoising
The proposed CycleISP framework allows converting sRGB images to RAW data, and then back to sRGB images. It has (a) RGB2RAW network branch, and (b) RAW2RGB network branch.  

<p align="center">
  <img src = "https://i.imgur.com/owqujSm.png" width="800">
  <br/>
  <b> Overall Framework of CycleISP </b>
</p>

<p align="center">
  <img src = "https://i.imgur.com/gQ4JLyO.png" width="500">
  <br/>
  <b> Recursive Residual Group (RRG) </b>
</p>




### Generating Data for Raw Denoising
The RGB2RAW network branch takes as input a clean sRGB image and converts it to a clean RAW image. The noise injection module adds shot and read noise of different levels to the (RAW) output  RGB2RAW network branch.  Thereby, we can generate clean and its corresponding noisy image pairs {RAW_clean, RAW_noisy} from any sRGB image.


### Generating Data for sRGB Denoising
Given a synthetic RAW noisy image as input, the RAW2RGB network branch maps it to a noisy sRGB image; hence we are able to generate an image pair {sRGB_clean, sRGB_noisy} for the sRGB  denoising  problem.  


### Proposed Denoising Network
<p align="center">
  <img src = "https://i.imgur.com/DFLTDZI.png" width="620">
  <br/>
</p>

## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
sudo apt-get install cmake build-essential libjpeg-dev libpng-dev
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image yacs lycon natsort h5py tqdm
```

## Evaluation
#### Denoising RAW images of DND
- Download the [model](https://drive.google.com/file/d/1yjI3JtfC1IluGB0LRlnpPTAY_UyY9mG8/view?usp=sharing) and place it in ./pretrained_models/denoising/
- Download RAW [images](https://drive.google.com/drive/folders/15Bay1UJURlbP7kpS10_fJ96MxQ6B03Xv?usp=sharing) of DND and place them in ./datasets/dnd/dnd_raw/
- Run
```
python test_dnd_raw.py --save_images
```
#### Denoising RAW images of SIDD
- Download the [model](https://drive.google.com/file/d/1m2A4goZENg_kKV1-FJoeg1rBqLj6WRIm/view?usp=sharing) and place it in ./pretrained_models/denoising/
- Download RAW [images](https://drive.google.com/drive/folders/1invur2uE-QXHh-btHTZQetFcgjUAfbwc?usp=sharing) of SIDD and place them in ./datasets/sidd/sidd_raw/
- Run
```
python test_sidd_raw.py --save_images
```
#### Denoising sRGB images of DND
- Download the [model](https://drive.google.com/file/d/1740sYH7bG-c-jL5wc3e1_NOpxwGTXS9c/view?usp=sharing) and place it in ./pretrained_models/denoising/
- Download sRGB [images](https://drive.google.com/drive/folders/101AfVtkfizl20-XQ3leNt2cE_rI5ypeu?usp=sharing) of DND and place them in ./datasets/dnd/dnd_rgb/noisy/
- Run
```
python test_dnd_rgb.py --save_images
```
#### Denoising sRGB images of SIDD
- Download the [model](https://drive.google.com/file/d/1sraG9JKmp0ieLjntRL7Jj2FXBrPr-YVp/view?usp=sharing) and place it in ./pretrained_models/denoising/
- Download sRGB [images](https://drive.google.com/drive/folders/1JGoCXqHESBIocpDabk74mnee6RA3oCoP?usp=sharing) of SIDD and place them in ./datasets/sidd/sidd_rgb/
- Run
```
python test_sidd_rgb.py --save_images
```
## Results on Real Image Datasets
Experiments are performed for denoising images in RAW and sRGB spaces.

### Results for RAW Denoising

<table>
  <tr>
    <td> <img src = "https://i.imgur.com/NaUy7L8.png" width="550"> </td>
    <td> <img src = "https://i.imgur.com/srCnA7a.png" width="600"> </td>
  </tr>
</table>

### Results for sRGB Denoising

<img src = "https://i.imgur.com/h57wFn9.png" >

## Citation
If you use CycleISP, please consider citing:

    @inproceedings{Zamir2020CycleISP,
        title={CycleISP: Real Image Restoration via Improved Data Synthesis},
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
                and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
        booktitle={CVPR},
        year={2020}
    }
    
## Contact
Should you have any question, please contact waqas.zamir@inceptioniai.org
