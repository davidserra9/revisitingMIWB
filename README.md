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
  <a href="https://davidserra9.github.io/revisitingmiwb">
    <img src="https://img.shields.io/badge/Project-Page-black">
  </a>
  <img src="https://img.shields.io/badge/Dataset-Soon-yellow">
</p>

***

## TODOs (In Progress)

- Upload models for both splits of our dataset and RenderedWB.
- Upload the dataset. We are currently exploring the best way to host it.

## Method

We propose a lightweight Transformer block to blend five white balance (WB) presets and produce a white-balanced image. Our model contains only 7.9K parameters.

## Data

We repurpose the [LSMI dataset](https://www.dykim.me/projects/lsmi) to compute ground-truth white-balanced images from multi-illuminant scenes.

We are currently evaluating the best options for dataset hosting. It will be available for download soon.

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
