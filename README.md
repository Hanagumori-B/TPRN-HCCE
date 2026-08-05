# TRPN-HCCE

TRPN-HCCE is a research project for monocular RGB 6D object pose estimation.

This project is developed based on the excellent **HCCEPose** framework and extends it with a fully differentiable end-to-end pose estimation pipeline. We sincerely thank the original authors for making their code publicly available.

**Original HCCEPose repository:**
https://github.com/WangYuLin-SEU/HCCEPose

## Overview

Compared with the original HCCEPose framework, TRPN-HCCE introduces:

- A learnable differentiable HCCE decoder for dense coordinate reconstruction.
- A Transformer-PnP pose regression network for direct SE(3) estimation.
- A fully GPU-based inference pipeline without conventional iterative RANSAC-PnP optimization.
- End-to-end optimization from dual-surface dense geometric representations to object poses.

## Environment

- Python 3.10
- Pytorch 2.6

## Installation

Clone this repository:

```bash
git clone https://github.com/Hanagumori-B/TRPN-HCCE.git
cd TRPN-HCCE
```

The project follows the data organization and preprocessing pipeline of HCCEPose.
After `git clone` this repository, please download the BOP toolkit provided by the original HCCEPose repository: [bop_toolkit](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/bop_toolkit.zip), and unzip it to the project root directory.

```
TRPN-HCCE
│
├── bop_toolkit/
├── datasets/
├── HccePose/
├── output/
├── scripts/
├── tools/
├── yolo_train/
└── ...
```

For more detailed environment configuration information and dataset preprocessing, please refer to [HCCEPose](https://github.com/WangYuLin-SEU/HCCEPose)

## Training

### Training the 2D Detector

Script `scripts/s3_p1_prepare_yolo_label.py` convert BOP PBR data to YOLO format. 

After Converting, you can train the YOLO detector using the script `scripts/s3_p2_train_yolo.py`.

### Training TRPN-HCCE
Run the script `scripts/s4_p1_gen_bf_labels.py` to generate the front and back 3D coordinate label maps.

Then, you can train a TRPN-HCCE model using `scripts/s4_p2_train_pnpnet_bf_pbr_by_epoch_with_smooth_decode_loss.py`.

## Inference

After training, configure the path to the trained `.pt` file in `scripts/s5_inference_pipeline.py`. Then run the inference pipeline to estimate object poses and generate visualization results.
