## Revisiting Image Fusion for Multi-Illuminant White-Balance Correction.<br>In ICCV, 2025.


<p align="center">
  <a href="https://davidserra9.github.io/">David Serrano-Lozano</a><sup>1,2</sup>, 
  <a href="https://adityac8.github.io/">Aditya Arora</a><sup>3,4,5</sup>, 
  <a href="https://www.lherranz.org/">Luis Herranz</a><sup>6</sup>,<br>
  <a href="https://csprofkgd.github.io/">Konstantinos G. Derpanis</a><sup>3,4</sup>, 
  <a href="https://www.cse.yorku.ca/~mbrown/">Michael S. Brown</a><sup>3</sup> and 
  <a href="https://jvazquezcorral.github.io/">Javier Vazquez-Corral</a><sup>1,2</sup>
  <br><br>
  <sup>1</sup>Computer Vision Center, 
  <sup>2</sup>Universitat Autònoma de Barcelona, 
  <sup>3</sup>York University,<br>
  <sup>4</sup>Vector Institute, 
  <sup>5</sup>TU Darmstadt and 
  <sup>6</sup>Universidad Politécnica de Madrid
  <br><br>
  
  <a href="https://arxiv.org/abs/2503.14774">
    <img src="https://img.shields.io/badge/ArXiv-Paper-B31B1B">
  </a>
  <a href="https://revisitingmiwb.github.io">
    <img src="https://img.shields.io/badge/Project-Page-black">
  </a>
  <a href="https://color.cvc.uab.cat/revisitingmiwb">
    <img src="https://img.shields.io/badge/Dataset-Page-yellow">
  </a>
</p>

***

## TODOs (In Progress)

✅ Upload models for both splits of our dataset and RenderedWB.

✅ Upload the dataset. 

 · Upload the large version of the dataset. We are currently exploring the best way to host it.

## Method

We propose a lightweight Transformer block to blend five white balance (WB) presets and produce a white-balanced image. Our model contains only 7.9K parameters.

## Data

While the original [LSMI dataset](https://www.dykim.me/projects/lsmi) was designed for illumination estimation from RAW images, we repurpose it to compute ground-truth white-balanced images from multi-illuminant scenes. Please check the original dataset for data acquisition and other details.

To download the dataset, please check the following website: 
<a href="https://color.cvc.uab.cat/revisitingmiwb">
  <img src="https://img.shields.io/badge/Dataset-Page-yellow">
</a>

## Getting Started

Clone the repository and install the required dependencies.

## Train and Inference

Pre-trained models are available in the `weights` folder. Each checkpoint is only ~38KB.

Both training and inference use the `config.yaml` file to specify all parameters and configurations. Please adapt it accordingly.

To run inference on our dataset:
```bash
python inference.py
```

To train a model:
```bash
python train.py
```
