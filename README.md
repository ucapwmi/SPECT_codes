# Evaluating the Effectiveness of Denoising Using Machine Learning in SPECT

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
├── core/                # data I/O, transforms, metrics, utilities (project core)
├── experiments/         # runnable entry points (python -m ...)
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
│       ├── test_sino.py   # denoise sinograms
│       ├── test_osem.py   # OSEM of denoised sinograms
│       └── test_clinical.py
├── models/              # network definitions, checkpoints
├── requirements.txt
└── README.md
```

## 3) Data
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
  dateset_xcat/
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

## 4) How to run

All commands are executed from the repo root.

### 4.1 3D U-Net (image domain)
``` bash
python3 -m experiments.Unet.pre_training.train
python3 -m experiments.Unet.fine_tuning.train
python3 -m experiments.Unet.test
python3 -m experiments.Unet.test_clinical
```

### 4.2 Swin UNETR (image domain)
``` bash
python3 -m experiments.Swin_UNETR.pre_training.train
python3 -m experiments.Swin_UNETR.fine_tuning.train
python3 -m experiments.Swin_UNETR.test
python3 -m experiments.Swin_UNETR.test_clinical
```

### Single-Step PDPM (projection / sinogram domain)
``` bash
python3 -m experiments.Single_step_PDPM.pre_training.train
python3 -m experiments.Single_step_PDPM.fine_tuning.train
python3 -m experiments.Single_step_PDPM.test_sino
python3 -m experiments.Single_step_PDPM.test_osem
python3 -m experiments.Single_step_PDPM.test_clinical
python3 -m experiments.Single_step_PDPM.test_clinical_osem
```
