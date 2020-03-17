# CycleISP

This repository is for CycleISP introduced in the following paper

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en) "CycleISP: Real Image Restoration via Improved Data Synthesis" CVPR (Oral) 2020

**Paper**: 

**Supplementary**: 

## Codes and Pre-trained Models Releasing Soon! 

## Abstract

The availability of large-scale datasets has helped unleash the true potential of deep convolutional neural networks (CNNs).
However, for the single-image denoising problem, capturing a real dataset is an unacceptably expensive and cumbersome procedure. Consequently, image denoising algorithms are mostly developed and evaluated on synthetic data that is usually generated with a widespread assumption of additive white Gaussian noise (AWGN). While the CNNs achieve impressive results on these synthetic datasets, they do not perform well when applied on real camera images, as reported in recent benchmark datasets. This is mainly because the AWGN is not adequate for modeling the real camera noise which is signal-dependent and heavily transformed by the camera imaging pipeline. In this paper, we present a framework that models camera imaging pipeline in forward and reverse directions. It allows us to produce any number of realistic image pairs for denoising both in RAW and sRGB spaces. By training a new image denoising network on realistic synthetic data, we achieve the state-of-the-art performance on real camera benchmark datasets. The parameters in our models are ~5 times lesser than the previous best method for RAW denoising. Furthermore, we demonstrate that the proposed framework generalizes beyond image denoising problem e.g., for color matching in stereoscopic cinema.

## Network Architecture
<p align="center">
  <img src = "https://i.imgur.com/owqujSm.png" width="800">
  <br/>
  <b> Overall Framework of CycleISP </b>
</p>

<p align="center">
  <img src = "https://i.imgur.com/gQ4JLyO.png" width="500">
  <br/>
  <b> Recursive Residual Group </b>
</p>

<p align="center">
  <img src = "https://i.imgur.com/DFLTDZI.png" width="620">
  <br/>
  <b> Proposed denoising network </b>
</p>

## Results
Experiments are performed on two image datasets for image denoising in RAW and sRGB space.

<table>
  <tr>
    <td> <img src = "https://i.imgur.com/NaUy7L8.png" width="550"> </td>
    <td> <img src = "https://i.imgur.com/srCnA7a.png" width="600"> </td>
  </tr>
</table>

<img src = "https://i.imgur.com/h57wFn9.png" >

## Contact
Should you have any question, please contact waqas.zamir@inceptioniai.org
