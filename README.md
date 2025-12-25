# NLCG-Net: A Model-Based Zero-Shot Learning Framework for Undersampled Quantitative MRI Reconstruction

Official implementation of **NLCG-Net** for undersampled quantitative MRI (qMRI) reconstruction,  
as described in:

> Xinrui Jiang, Yohan Jun, Jaejin Cho, Mengze Gao, Xingwang Yong, Berkin Bilgic,  
> *NLCG-Net: A Model-Based Zero-Shot Learning Framework for Undersampled Quantitative MRI Reconstruction*,  
> accepted by ISMRM 2024.

Paper (arXiv): [https://arxiv.org/abs/2401.12004](https://arxiv.org/abs/2401.12004) [file:1]

The repository currently provides implementations for both **T2 mapping** (`t2`) and **T1 mapping** (`t1`).

---

## Overview

<p align="center">
  <img src="images/framework.png" alt="NLCG-Net framework" width="80%">
</p>

---

## Status

- **1/2024 update:** the codebase is being reorganized to be more reader-friendly.  
  - `t1/` contains code for T1 mapping.  
  - `t2/` contains code for T2 mapping.  
  - The main differences lie in `data_consistency.py`, since T1 and T2 use different signal models and raw data.

If you encounter any issues when running the code, please feel free to open an issue or submit a pull request.
