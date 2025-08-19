import torch
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from models.Swin_UNETR.model import get_swin_unetr
from core.metrics import ssim_metric, mse_loss
from core.visualization import visualize_predictions
from core.transforms import TestNPYDataset
from core.data import make_groups_from_osem, load_organ_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = get_swin_unetr(device)
model.load_state_dict(torch.load("models/swin_unetr_checkpoints/best_model_swin_unetr_xcat.pth"))
model.eval()

# Prepare data
data_root = "data/volume"  
groups = make_groups_from_osem(data_root)

# load organ masks
organ_masks = load_organ_masks("data/segmentations")

print("=== Pair counts (noisy+clean) ===")
for k in ["1.0", "0.5", "0.25", "0.125", "0.05"]:
    print(f"Level {k:>5}: {len(groups[k])} pairs")

# inference
results = {}
with torch.no_grad():
    for noise, file_list in groups.items():
        if not file_list:
            continue

        ds = TestNPYDataset(file_list)        
        loader = DataLoader(ds, batch_size=1, num_workers=4)

        mses, psnrs, ssims = [], [], []
        sums_gt, sums_noisy, sums_pred = [], [], []
        for i, batch in enumerate(loader):
            inp  = batch["input"].to(device)             
            lbl  = batch["label"].to(device)
            scale = batch["scale"].to(device).view(-1, 1, 1, 1, 1)

            out = model(inp).clamp_min_(0)
            out_cnt = out * scale                    
            lbl_cnt = lbl * scale

            pred_np=out_cnt.cpu().squeeze().numpy()
            gt_np=lbl_cnt.cpu().squeeze().numpy()
            inp_np=(inp * scale).cpu().squeeze().numpy()

            for organ, m in organ_masks.items(): 
                pred_m = np.where(m, pred_np, 0.0)
                gt_m   = np.where(m, gt_np,   0.0)
                inp_m  = np.where(m, inp_np,  0.0)
                if m.sum() == 0:
                    continue
                visualize_predictions(
                    pred_np=pred_m,
                    gt_np=gt_m,
                    inp_np=inp_m,
                    save_dir=f"./visuals_clinical_swin_unetr/{noise}/{organ}",
                    sample_name=f"sample{i}_{organ}",
                    axis=1,
                )

            peak = lbl_cnt.max()
            pred_norm = out_cnt / (peak + 1e-8)
            gt_norm   = lbl_cnt / (peak + 1e-8)

            mses.append(mse_loss(pred_norm, gt_norm).item())
            mse_val = np.mean((pred_norm.cpu().numpy() - gt_norm.cpu().numpy()) ** 2)
            psnrs.append(float("inf") if mse_val == 0 else 10 * np.log10(1.0 / mse_val))
            ssims.append(1.0 - ssim_metric(pred_norm, gt_norm).item())

            s_gt = float(np.sum(gt_np, dtype=np.float64))
            s_ny = float(np.sum(inp_np, dtype=np.float64))
            s_pr = float(np.sum(pred_np, dtype=np.float64))
            sums_gt.append(s_gt)
            sums_noisy.append(s_ny)
            sums_pred.append(s_pr)
            print(f"[sum] noise={noise} sample{i}: GT={s_gt:.6e}, noisy={s_ny:.6e}, pred={s_pr:.6e}")

            visualize_predictions(
                pred_np=pred_np,
                gt_np=gt_np,
                inp_np=inp_np,
                save_dir=f"./visuals_clinical_swin_unetr/{noise}",
                sample_name=f"sample{i}",
                axis=1,
            )

        results[noise] = {
            "mse_mean": np.mean(mses), "mse_std": np.std(mses),
            "psnr_mean": np.mean(psnrs), "psnr_std": np.std(psnrs),
            "ssim_mean": np.mean(ssims), "ssim_std": np.std(ssims),
            "sum_gt_mean": np.mean(sums_gt), "sum_gt_std": np.std(sums_gt),
            "sum_noisy_mean": np.mean(sums_noisy), "sum_noisy_std": np.std(sums_noisy),
            "sum_pred_mean": np.mean(sums_pred), "sum_pred_std": np.std(sums_pred),
        }

print("\n=== Test Results (mean ± std) ===")
for noise in ["1.0", "0.5", "0.25", "0.125", "0.05"]:
    if noise in results:
        r = results[noise]
        print(
            f"Noise={noise:>5} | "
            f"MSE={r['mse_mean']:.6f}±{r['mse_std']:.6f} | "
            f"PSNR={r['psnr_mean']:.2f}±{r['psnr_std']:.2f} dB | "
            f"SSIM={r['ssim_mean']:.4f}±{r['ssim_std']:.4f} | "
            f"GTsum={r['sum_gt_mean']:.3e}±{r['sum_gt_std']:.3e} | "
            f"Noisysum={r['sum_noisy_mean']:.3e}±{r['sum_noisy_std']:.3e} | "
            f"Predsum={r['sum_pred_mean']:.3e}±{r['sum_pred_std']:.3e}"
        )
