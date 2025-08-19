import os, re
from glob import glob
import numpy as np
import torch
from core.metrics import mse_loss, ssim_metric
from core.visualization import visualize_triplet

# Paths and visualization parameters
data_root = "data/volume"  
save_root = "Single_step_PDPM_visualizations_osem_clinical"
os.makedirs(save_root, exist_ok=True)
axis = 1
slice_index = None

# pred_volumeX_n{level}.npy
pred_files = sorted(glob(os.path.join(data_root, "pred_volume*_n*.npy")))
pat = re.compile(r'^pred_(?P<stem>volume\d+)_n(?P<lvl>(?:1|0\.5|0\.25|0\.125|0\.05))\.npy$')

# Normalize "1" to "1.0" in the report
lvl_label = {"1": "1.0", "0.5": "0.5", "0.25": "0.25", "0.125": "0.125", "0.05": "0.05"}
results = {lab: {"mse": [], "psnr": [], "ssim": []} for lab in lvl_label.values()}

paired_cnt = 0
for pred_path in pred_files:
    name = os.path.basename(pred_path)
    m = pat.match(name)
    if not m:
        print(f"[skip] Filename does not match pattern: {name}")
        continue

    stem = m.group("stem")      # e.g. volume1
    lvl  = m.group("lvl")       # e.g. 0.5 / 0.05 / 1
    noise_label = lvl_label[lvl]

    gt_path    = os.path.join(data_root, f"osem_{stem}_n{lvl}_clean.npy")
    noisy_path = os.path.join(data_root, f"osem_{stem}_n{lvl}.npy")

    if not (os.path.exists(gt_path) and os.path.exists(noisy_path)):
        print(f"[skip] Missing paired files: {name} needs {os.path.basename(gt_path)} and {os.path.basename(noisy_path)}")
        continue

    pred_np  = np.load(pred_path).astype(np.float32)   # (128,128,128)
    gt_np    = np.load(gt_path).astype(np.float32)
    noisy_np = np.load(noisy_path).astype(np.float32)

    # Normalize to [0,1] before calculating metrics (using GT's peak value as reference)
    peak = float(max(gt_np.max(), 1e-8))
    pred_norm = torch.tensor(pred_np / peak).unsqueeze(0).unsqueeze(0)
    gt_norm   = torch.tensor(gt_np   / peak).unsqueeze(0).unsqueeze(0)

    # MSE
    mse_val = mse_loss(pred_norm, gt_norm).item()
    results[noise_label]["mse"].append(mse_val)

    # PSNR
    mse_np = float(np.mean((pred_np/peak - gt_np/peak) ** 2))
    psnr_val = float("inf") if mse_np == 0 else 10.0 * np.log10(1.0 / mse_np)
    results[noise_label]["psnr"].append(psnr_val)

    # SSIM
    ssim_val = 1.0 - ssim_metric(pred_norm, gt_norm).item()
    results[noise_label]["ssim"].append(ssim_val)

    # Visualize triplet: pred / gt(clean) / noisy
    save_dir = os.path.join(save_root, noise_label)
    os.makedirs(save_dir, exist_ok=True)
    out_png = os.path.join(save_dir, f"{stem}_n{lvl}_axis{axis}.png")
    visualize_triplet(pred_np, gt_np, noisy_np, out_png, axis=axis, slice_index=slice_index)
    paired_cnt += 1

print(f"\nSuccessfully paired and visualized: {paired_cnt} sets\n")

print("=== Test Results (mean ± std) ===")
for noise in ["1.0", "0.5", "0.25", "0.125", "0.05"]:
    if len(results[noise]["mse"]) == 0:
        print(f"Noise={noise:>5} | (no samples)")
        continue
    mse_mean  = np.mean(results[noise]["mse"]);  mse_std  = np.std(results[noise]["mse"])
    psnr_mean = np.mean(results[noise]["psnr"]); psnr_std = np.std(results[noise]["psnr"])
    ssim_mean = np.mean(results[noise]["ssim"]); ssim_std = np.std(results[noise]["ssim"])
    print(f"Noise={noise:>5} | MSE={mse_mean:.6f}±{mse_std:.6f} | "
          f"PSNR={psnr_mean:.2f}±{psnr_std:.2f} dB | SSIM={ssim_mean:.4f}±{ssim_std:.4f}")
