# HistDiST: Histopathological Diffusion-based Stain Transfer

Official pytorch implementation of HistDiST.
This repository contains scripts for fine-tuning pretrained Stable Diffusion models for the task of stain transfer, along with the corresponding inference script.

Erik Großkopf,
[Valay Bundele](https://scholar.google.com/citations?user=xWvW9_UAAAAJ&hl=en&oi=ao),
[Mehran Hossienzadeh](https://scholar.google.com/citations?user=V5yInfUAAAAJ&hl=en), 
[Hendrik P.A. Lensch](https://scholar.google.de/citations?user=2R22h84AAAAJ&hl=en)

## Abstract 
Hematoxylin and Eosin (H&E) staining is the cornerstone of histopathology but lacks molecular specificity. While Immunohistochemistry (IHC) provides molecular insights, it is costly and complex, motivating H&E-to-IHC translation as a cost-effective alternative. Existing translation methods are mainly GAN-based, often struggling with training instability and limited structural fidelity, while diffusion-based approaches remain underexplored. We propose HistDiST, a Latent Diffusion Model (LDM) based framework for high-fidelity H&E-to-IHC translation. HistDiST introduces a dual-conditioning strategy, utilizing Phikon-extracted morphological embeddings alongside VAE-encoded H&E representations to ensure pathology-relevant context and structural consistency. To overcome brightness biases, we incorporate a rescaled noise schedule, v-prediction, and trailing timesteps, enforcing a zero-SNR condition at the final timestep. During inference, DDIM inversion preserves the morphological structure, while an η-cosine noise schedule introduces controlled stochasticity, balancing structural consistency and molecular fidelity. Moreover, we propose Molecular Retrieval Accuracy (MRA), a novel pathology-aware metric leveraging GigaPath embeddings to assess molecular relevance. Extensive evaluations on MIST and BCI datasets demonstrate that HistDiST significantly outperforms existing methods, achieving a 28% improvement in MRA on the H&E-to-Ki67 translation task, highlighting its effectiveness in capturing true IHC semantics.
<p>
<img src="assets/results.jpg" width="100%"/>  
<br>

## Overview
### HistDiST training pipeline
<div align="center">
  <img src="assets/training.jpg" width="98%"/>  
</div>
H&E generation (red arrows, label (2)) conditioned on CLIP text embeddings, and H&E-to-IHC translation (green arrows, label (1)) guided by Phikon embeddings and VAE-encoded H&E features. The VAE encoder maps images to latent space, where noise is added and later denoised by the U-Net. The Ge/Tr switch selects between generation and translation tasks, with each(numbered input, color-coded pathway) independently followed.

### HistDiST inference pipeline
<div align="center">
  <img src="assets/inference.jpg" width="98%"/>  
</div>
VAE encoder maps H&E image to latent space, where DDIM inversion derives noise latent and η-noise scheduling injects noise at different timesteps during denoising. The U-Net, conditioned on Phikon embeddings, refines the features, and the VAE decoder generates the final IHC output.

## Setup
### Models
All trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1onzQ05SVYCxsGMYhMis25Bsm3nwFbiyi?usp=sharing).

### Datasets
[MIST](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE) and [BCI](https://github.com/bupt-ai-cz/BCI) datasets are used to finetune our models from H&E to ER/HER2/Ki67/PR transfer. Images are cropped randomly from size 1024x1024 to 512x512.

### Requirements
```shell
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install diffusers==0.16.1 
pip install transformers==4.32.1 
```

### Training
```bash
python inference/inference.py --model_folder_path path/to/er_model_folder --img_path inference/example_images/er.jpg
```

### Inference
```bash
python inference/inference.py --model_folder_path path/to/er_model_folder --img_path inference/example_images/er.jpg
```

### TODO
- [ ] Update README
- [ ] Requirements.txt (export from python venv)
- [ ] Add arguments to inferece.py script for modelpath
- [ ] Training: Resolve/Refactor imports
- [ ] Training: Compute Metrics (optional) Copy metrics folder from ASP Repo
- [ ] Validate repository instructions (Emty venv/installation/model download)
- [ ] Quote ASP Paper
- [ ] Quote Diffusers library
- [ ] Used Dataset links
- [ ] Figures
  - [ ] Examples
  - [ ] Training
  - [ ] Inference
- [ ] Acknowledgement/Citation
