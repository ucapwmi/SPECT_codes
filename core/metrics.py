import torch
import torch.nn as nn
from monai.losses import SSIMLoss
import numpy as np


# PSNR, normalized
def compute_psnr_volume(pred_cnt: torch.Tensor,
                        tgt_cnt: torch.Tensor,
                        eps: float = 1e-8) -> float:
    """
    Compute PSNR on *count domain* tensors. Internally divides both
    volumes by tgt_cnt.max() so that MAX_I = 1 for every sample.
    """
    pred_np   = pred_cnt.squeeze().cpu().numpy()
    tgt_np    = tgt_cnt.squeeze().cpu().numpy()
    peak = tgt_np.max() + eps
    pred_norm, tgt_norm = pred_np / peak, tgt_np / peak
    mse = np.mean((pred_norm - tgt_norm) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)

mse_loss  = nn.MSELoss()

# SSIM (Structural Similarity Index), window size = 7, used for validation
ssim_metric = SSIMLoss(spatial_dims=3, data_range=1.0,
                       win_size=7, reduction="mean")

# SSIM (Structural Similarity Index), window size = 5, used for training
ssim_unit = SSIMLoss(spatial_dims=3, data_range=1.0,
                     win_size=5, reduction="mean")

# Combined loss function
def combined_loss(pred_cnt, gt_cnt, alpha=0.5, eps=1e-8):
    peak = gt_cnt.view(gt_cnt.size(0), -1).amax(dim=1)        # (B,)
    peak = peak.view(-1, 1, 1, 1, 1) + eps                    # 5-D reshape
    pred_norm = pred_cnt / peak
    gt_norm   = gt_cnt  / peak
    mse_part  = mse_loss(pred_norm, gt_norm)
    ssim_part = ssim_unit(pred_norm, gt_norm)
    return alpha * mse_part + (1 - alpha) * ssim_part

mse_loss_pdpm = nn.MSELoss(reduction="none")
def per_img_mse(pred, target):
    """Per-image 3D volume mean MSE → [B]"""
    return mse_loss_pdpm(pred, target).mean(dim=(1, 2, 3, 4))

def loss_fn(pred, target, w):
    """batch weighted MSE"""
    return (per_img_mse(pred, target) * w).mean()