# Evaluating the Effectiveness of Denoising Using Machine Learning in SPECT

**Author**: Wei Miao  
**Supervisors**: Kris Thielemans, Efstathios Varzakis, Cate Gascoigne  
**Affiliation**: MSc Scientific and Data Intensive Computing, University College London (UCL)  
**Project type**: MSc Dissertation (2025)  

This repository evaluates SPECT denoising using three model families:
- **3D U-Net** (image domain)
- **Swin UNETR** (image domain)
- **Single-Step PDPM (Poisson diffusion)** in the **projection/sinogram domain**, followed by **OSEM** reconstruction

We assess five **count levels** (α = 1.0, 0.5, 0.25, 0.125, 0.05) using MSE, PSNR, and SSIM, plus qualitative visualisation on a small clinical set.  
> Note: α denotes **count level** (not only “dose”); reductions may come from lower injected activity and/or shorter acquisition time.

---

## 1) Environment

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Repository layout (key folders)
```
.
├── core/                       # data I/O, transforms, metrics, utilities (project core)
├── experiments/                # runnable entry points (python -m ...)
│   ├── Unet/
│   │   ├── pre_training/train.py
│   │   ├── fine_tuning/train.py
│   │   ├── test.py
│   │   └── test_clinical.py
│   ├── Swin_UNETR/
│   │   ├── pre_training/train.py
│   │   ├── fine_tuning/train.py
│   │   ├── test.py
│   │   └── test_clinical.py
│   └── Single_step_PDPM/
│       ├── pre_training/train.py
│       ├── fine_tuning/train.py
│       ├── test_sino.py        # denoise sinograms
│       ├── test_osem.py        # OSEM of denoised sinograms
        ├── test_clinical.py 
│       └── test_clinical_osem.py
├── models/                     # network definitions, checkpoints
├── requirements.txt
├── LICENSE                     # Apache 2.0 license
├── NOTICE                      # Copyright and attribution info
├── general.samp.par            # XCAT generation template
└── README.md

```

## 3) Data

Due to storage limitations, only a **test dataset** is provided here to allow quick replication of `test*.py` results.  
The full training/validation datasets can be generated using **XCAT (license required)** + **SIRF/STIR simulation scripts** (consistent with the methodology described in the dissertation).

- **Test dataset (download):**  
  [Google Drive link](https://drive.google.com/file/d/1_9qmfDLPg6ccOFm6nz91kM-gz0KYC8aL/view?usp=sharing)

- **Extraction location:**  
  Please unzip directly into the **repository root directory**, resulting in the following structure:

- **XCAT generation template:**  
  A sample parameter file is provided at the repo root:  
  `general.samp.par`  
  This can be used to generate additional XCAT-based phantoms following the same configuration as in the dissertation.


```
data/
  dataset/
    test/
        input/
        label/
    train/
        input/
        label/
    val/
        input/
        label/
  dataset_xcat/
    test/
        input/
        label/
    train/
        input/
        label/
    val/
        input/
        label/
  pdpm_dataset/
    test/
        images/
        sinograms/
    train/
        images/
        sinograms/
            pred_sino/
    val/
        images/
        sinograms/
  pdpm_dataset_xcat/
    test/
        images/
        sinograms/
    train/
        images/
        sinograms/
            pred_sino/
    val/
        images/
        sinograms/
    volume/
        pred_sino/
    segmentations/
```
## 4) Pre-trained Weights

Download the pre-trained checkpoints:
- [Model checkpoints (3D U-Net, Swin UNETR, Single-step PDPM)](https://drive.google.com/file/d/1KXPZZx3yf0yzLfm0KC5sN1jHLogu_7NQ/view?usp=sharing)

### Setup
Extract **model_checkpoints.zip** and copy the following subfolders into the repo’s `models/` directory:
- `3d_unet_checkpoints/`
- `swin_unetr_checkpoints/`
- `Single_step_PDPM_checkpoints/`

## 5) How to run

All commands are executed from the repo root.

### 5.1 3D U-Net (image domain)
``` bash
python3 -m experiments.Unet.pre_training.train
python3 -m experiments.Unet.fine_tuning.train
python3 -m experiments.Unet.test
python3 -m experiments.Unet.test_clinical
```

### 5.2 Swin UNETR (image domain)
``` bash
python3 -m experiments.Swin_UNETR.pre_training.train
python3 -m experiments.Swin_UNETR.fine_tuning.train
python3 -m experiments.Swin_UNETR.test
python3 -m experiments.Swin_UNETR.test_clinical
```

### 5.3 Single-Step PDPM (projection / sinogram domain)
``` bash
python3 -m experiments.Single_step_PDPM.pre_training.train
python3 -m experiments.Single_step_PDPM.fine_tuning.train
python3 -m experiments.Single_step_PDPM.test_sino
python3 -m experiments.Single_step_PDPM.test_osem
python3 -m experiments.Single_step_PDPM.test_clinical
python3 -m experiments.Single_step_PDPM.test_clinical_osem
```

---

## 6) Acknowledgements

This project builds upon prior work by Cate Gascoigne (MRes thesis, UCL, 2024) on 2D CNNs for SPECT phantom denoising, extending it to 3D architectures, anatomically realistic XCAT phantoms, and transformer-based and diffusion-based models.  

I would like to thank my supervisors Kris Thielemans, Efstathios Varzakis, and Cate Gascoigne for their guidance and support throughout this project.


