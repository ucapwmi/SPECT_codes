from models.Unet.model import CustomUNet3D
import torch
from torch.utils.data import DataLoader
from core.metrics import ssim_metric, mse_loss
from core.visualization import visualize_predictions
from core.transforms import TestNPYDataset
import numpy as np
import os
from glob import glob
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CustomUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
model.load_state_dict(torch.load("models/3d_unet_checkpoints/best_model_3D_unet_xcat.pth"))
model.eval()

data_root = "data/dataset_xcat/test"
input_paths = sorted(glob(os.path.join(data_root, "input_*.npy")))
print(f"Found {len(input_paths)} input files.")
label_paths = [p.replace("input_", "label_", 1) for p in input_paths]

noise_levels = {
    "1.0": r"_n1\.npy$",
    "0.5": r"_n0\.5\.npy$",
    "0.25": r"_n0\.25\.npy$",
    "0.125": r"_n0\.125\.npy$",
    "0.05": r"_n0\.05\.npy$",
}
alpha_map = {"1.0": 1.0, "0.5": 0.5, "0.25": 0.25, "0.125": 0.125, "0.05": 0.05}

groups = {lvl: [] for lvl in noise_levels}
for inp, lab in zip(input_paths, label_paths):
    if not os.path.exists(lab): 
        continue
    for lvl, pattern in noise_levels.items():
        if re.search(pattern, inp):
            groups[lvl].append({"input": inp, "label": lab})
            break

# -------- Pre-scanning phase: calculate the global peak ref_peak for alpha=1.0 --------
print("Pre-scanning to compute reference peak (alpha=1.0 group)...")
ref_peak = 0.0
if groups["1.0"]:
    ds_ref = TestNPYDataset(groups["1.0"])
    loader_ref = DataLoader(ds_ref, batch_size=1, num_workers=4)
    with torch.no_grad():
        for batch in loader_ref:
            lbl = batch["label"]  # [B,1,D,H,W], torch tensor
            scale = batch["scale"].view(-1,1,1,1,1)  # [B,1,1,1,1]
            lbl_cnt = (lbl * scale).float()          # counts domain
            # Note: Get the global maximum across all samples
            cur_max = float(lbl_cnt.max().cpu().numpy())
            if cur_max > ref_peak:
                ref_peak = cur_max

if ref_peak <= 0:
    raise RuntimeError("No valid reference peak found from alpha=1.0 group. Check your test set.")

print(f"Reference peak (alpha=1.0) = {ref_peak:.6f}")

# Calculate fixed vmax = alpha * ref_peak for each noise group
fixed_vmax = {lvl: alpha_map[lvl] * ref_peak for lvl in alpha_map}
print("Per-group fixed vmax (max*alpha):")
for k,v in fixed_vmax.items():
    print(f"  {k}: {v:.6f}")

# -------------------- Inference --------------------
results = {}
with torch.no_grad():
    for noise, file_list in groups.items():
        if not file_list: 
            continue
        ds = TestNPYDataset(file_list)
        loader = DataLoader(ds, batch_size=1, num_workers=4)

        mses, psnrs, ssims = [], [], []
        for i, batch in enumerate(loader):
            inp = batch["input"].to(device)         # normalised domain
            lbl = batch["label"].to(device)
            scale = batch["scale"].to(device).view(-1,1,1,1,1)

            out = model(inp).clamp_min_(0)
            out_cnt = out * scale       # counts domain for visual
            lbl_cnt = lbl * scale
            inp_cnt = inp * scale

            peak = lbl_cnt.max()
            pred_norm = out_cnt / (peak + 1e-8)
            gt_norm   = lbl_cnt / (peak + 1e-8)

            mses.append(mse_loss(pred_norm, gt_norm).item())
            mse_val = np.mean((pred_norm.cpu().numpy() - gt_norm.cpu().numpy())**2)
            psnrs.append(float('inf') if mse_val == 0 else 10*np.log10(1.0 / mse_val))
            ssims.append(1.0 - ssim_metric(pred_norm, gt_norm).item())

            # --- Visualization: Fixed gray range vmin=0, vmax=fixed_vmax[noise] ---
            visualize_predictions(
                pred_np=out_cnt.cpu().squeeze().numpy(),
                gt_np=lbl_cnt.cpu().squeeze().numpy(),
                inp_np=inp_cnt.cpu().squeeze().numpy(),
                save_dir=f"./visuals_xcat_unet3d/{noise}",
                sample_name=f"sample{i}",
                axis=1,
                vmin=0.0,
                vmax=float(fixed_vmax[noise])
            )

        results[noise] = {
            "mse_mean": np.mean(mses), "mse_std": np.std(mses),
            "psnr_mean": np.mean(psnrs), "psnr_std": np.std(psnrs),
            "ssim_mean": np.mean(ssims), "ssim_std": np.std(ssims)
        }

# ---------------------- Print Results ----------------------
print("\n=== Test Results (mean ± std) ===")
for noise in ["1.0","0.5","0.25","0.125","0.05"]:
    if noise in results:
        r = results[noise]
        print(f"Noise={noise:>5} | "
              f"MSE={r['mse_mean']:.6f}±{r['mse_std']:.6f} | "
              f"PSNR={r['psnr_mean']:.2f}±{r['psnr_std']:.2f} dB | "
              f"SSIM={r['ssim_mean']:.4f}±{r['ssim_std']:.4f}")
