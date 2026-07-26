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
