# TRPN-HCCE

TRPN-HCCE is a research project for monocular RGB 6D object pose estimation.

This project is developed based on the excellent **HCCEPose** framework and extends it with a fully differentiable end-to-end pose estimation pipeline. We sincerely thank the original authors for making their code publicly available.

**Original HCCEPose repository:**
https://github.com/WangYuLin-SEU/HCCEPose

## Overview

Compared with the original HCCEPose framework, TRPN-HCCE introduces:

- A learnable differentiable HCCE decoder for dense coordinate reconstruction.
- A Transformer-PnP pose regression network for direct SE(3) estimation.
- A fully GPU-resident inference pipeline without iterative RANSAC-PnP optimization.
- End-to-end optimization from dual-surface dense geometric representations to object poses.

## Environment

- Python 3.10
- Pytorch 2.6

After `git clone` this repositoriy, please download [bop_toolkit](https://github.com/WangYuLin-SEU/HCCEPose/blob/main/bop_toolkit.zip), and unzip it to the project root directory.
For more detailed environment configuration information and training preprocessing, please refer to [HCCEPose](https://github.com/WangYuLin-SEU/HCCEPose)

You can train a TRPN-HCCE model using `scripts/s4_p2_train_pnpnet_bf_pbr_by_epoch_with_smooth_decode_loss.py`.


