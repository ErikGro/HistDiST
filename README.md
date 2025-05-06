# HistDiST: Histopathological Diffusion-based Stain Transfer

Official pytorch implementation of HistDiST.

Erik Großkopf,
[Valay Bundele](https://scholar.google.com/citations?user=xWvW9_UAAAAJ&hl=en&oi=ao),
[Mehran Hossienzadeh](https://scholar.google.com/citations?user=V5yInfUAAAAJ&hl=en), 
[Hendrik P.A. Lensch](https://scholar.google.de/citations?user=2R22h84AAAAJ&hl=en)

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](TODO)

## Overview
This repository contains scripts for finetuning pretrained stable diffusion models for the task stain transfer.


## Models
All trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1onzQ05SVYCxsGMYhMis25Bsm3nwFbiyi?usp=sharing).

## Setup
### Requirements
```shell
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install diffusers==0.16.1 
pip install transformers==4.32.1 
```

### Usage
```bash
python run.py --img_path sample/cat1.png --prompt "a cat" --trg_prompt "a pig" --w_cut 3.0 --patch_size 1 2 --n_patches 256
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
