import os
from glob import glob
from core.metrics import mse_loss, ssim_metric
import numpy as np
import torch
from core.visualization import visualize_triplet


data_root = "data/pdpm_dataset_xcat/test/images"
save_root = "Single_step_PDPM_visualizations_osem"
os.makedirs(save_root, exist_ok=True)
axis = 1
slice_index = None

noise_map = {
    0: "1.0",
    1: "0.5",
    2: "0.25",
    3: "0.125",
    4: "0.05",
}

def take_slice(vol, axis=0, idx=None):
    if idx is None:
        idx = vol.shape[axis] // 2
    if axis == 0:   return vol[idx, :, :]
    elif axis == 1: return vol[:, idx, :]
    else:           return vol[:, :, idx]

pred_files  = sorted(glob(os.path.join(data_root, "*pred*.npy")))
gt_files    = sorted(glob(os.path.join(data_root, "*gt*.npy")))
noisy_files = sorted(glob(os.path.join(data_root, "*noisy*.npy")))

assert len(pred_files) == len(gt_files) == len(noisy_files), "Mismatch in number of files"
results = {lvl: {"mse": [], "psnr": [], "ssim": []} for lvl in noise_map.values()}

for idx, (pred_path, gt_path, noisy_path) in enumerate(zip(pred_files, gt_files, noisy_files)):
    group_id = idx // 10
    noise_level = noise_map[group_id]

    pred_np  = np.load(pred_path).astype(np.float32)
    gt_np    = np.load(gt_path).astype(np.float32)
    noisy_np = np.load(noisy_path).astype(np.float32)

    # scale（和原脚本一致）
    scale = max(gt_np.mean(), 1e-8)
    pred_cnt = pred_np * scale
    gt_cnt   = gt_np * scale

    peak = gt_cnt.max()
    pred_norm = torch.tensor(pred_cnt / (peak + 1e-8)).unsqueeze(0).unsqueeze(0)
    gt_norm   = torch.tensor(gt_cnt   / (peak + 1e-8)).unsqueeze(0).unsqueeze(0)

    # MSE
    mse_val = mse_loss(pred_norm, gt_norm).item()
    results[noise_level]["mse"].append(mse_val)

    # PSNR
    mse_np = np.mean((pred_cnt/peak - gt_cnt/peak) ** 2)
    psnr_val = float('inf') if mse_np == 0 else 10 * np.log10(1.0 / mse_np)
    results[noise_level]["psnr"].append(psnr_val)

    # SSIM
    ssim_val = 1.0 - ssim_metric(pred_norm, gt_norm).item()
    results[noise_level]["ssim"].append(ssim_val)

    # 保存可视化
    save_dir = os.path.join(save_root, noise_level)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sample{idx}_axis{axis}.png")
    visualize_triplet(pred_np, gt_np, noisy_np, save_path, axis=axis, slice_index=slice_index)

# ===================== 打印结果表格 =====================
print("\n=== Test Results (mean ± std) ===")
for noise in ["1.0", "0.5", "0.25", "0.125", "0.05"]:
    mse_mean = np.mean(results[noise]["mse"])
    mse_std  = np.std(results[noise]["mse"])
    psnr_mean = np.mean(results[noise]["psnr"])
    psnr_std  = np.std(results[noise]["psnr"])
    ssim_mean = np.mean(results[noise]["ssim"])
    ssim_std  = np.std(results[noise]["ssim"])
    print(f"Noise={noise:>5} | "
          f"MSE={mse_mean:.6f}±{mse_std:.6f} | "
          f"PSNR={psnr_mean:.2f}±{psnr_std:.2f} dB | "
          f"SSIM={ssim_mean:.4f}±{ssim_std:.4f}")
