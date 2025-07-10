## Revisiting Image Fusion for Multi-Illuminant White-Balance Correction. In ICCV, 2025.

[David Serrano-Lozano](https://davidserra9.github.io/)<sup>1,2</sup>, [Aditya Arora](https://adityac8.github.io/)<sup>3,4,5</sup>, [Luis Herranz](https://www.lherranz.org/)<sup>6</sup>, [Konstantinos G. Derpanis](https://csprofkgd.github.io/)<sup>3,4</sup>, [Michael S. Brown](https://www.cse.yorku.ca/~mbrown/)<sup>3</sup> and [Javier Vazquez-Corral](https://jvazquezcorral.github.io/) <sup>1,2</sup>

<sup>1</sup>Computer Vision Center,
<sup>2</sup>Universitat Autònoma de Barcelona,
<sup>3</sup>York University,
<sup>4</sup>Vector Institute,
<sup>5</sup>TU Darmstadt and
<sup>6</sup>Universidad Autónoma de Madrid

[![arXiv](https://img.shields.io/badge/ArXiv-Paper-B31B1B)](https://arxiv.org/abs/2503.14774)
[![web](https://img.shields.io/badge/Project-Page-black)](https://davidserra9.github.io/revisitingmiwb)
[![Dataset](https://img.shields.io/badge/Dataset-Soon-yellow)]()

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