import torch
import numpy as np
import random
import os
from core.data import make_list_sspdpm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.data import CacheDataset
from tqdm import tqdm
from core.transforms import transform_pdpm
from models.Single_step_PDPM.model import UNet3D_PDPM
from core.metrics import ssim_unit, loss_fn, per_img_mse
from datetime import datetime
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_root = "data/pdpm_dataset_xcat"
# weight of each count level
count_levels  = torch.tensor([1.0, 0.5, 0.25, 0.125, 0.05])
count_weights = torch.tensor([1.0, 1.0, 1.2, 1.6, 2.0])  

def collate_fn(batch):
    x0_u8 = torch.stack([b["img"].to(torch.uint8) for b in batch])  # [B,1,D,H,W]

    B   = x0_u8.size(0)
    idx = torch.randint(0, 5, (B,))                                # count index
    r   = count_levels[idx].view(B, 1, 1, 1, 1)                     # broadcast
    w   = count_weights[idx].float()                                # [B]

    # ---- Poisson degradation (expectation unchanged) ----
    lam   = (x0_u8.float() * r).clamp(min=0)
    xt_u8 = torch.poisson(lam)                                     # noisy uint8

    # ---- [0,1] ----
    x0 = x0_u8.float().div(255.0)
    xt = xt_u8.float().div(255.0)

    return {"xt": xt, "x0": x0, "r": count_levels[idx].float(), "w": w}

train_ds = CacheDataset(make_list_sspdpm("train", data_root), transform_pdpm, cache_rate=1.0, num_workers=4)
val_ds   = CacheDataset(make_list_sspdpm("val",   data_root), transform_pdpm, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=8,
                          collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=8,
                          collate_fn=collate_fn)

model = UNet3D_PDPM().to(device)
model.load_state_dict(torch.load("models/Single_step_PDPM/best_sspdpm_pre_train.pth"))
optimizer  = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
lr_scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)
history = {"train": [], "val": [], "ssim": [], "mse": []}
best_val, stall, patience, num_epochs = np.inf, 0, 5, 60
ckpt_dir = "models/Single_step_PDPM_checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

for epoch in range(1, num_epochs + 1):
    # ---------- Train ----------
    model.train()
    train_loss = 0.0
    for b in tqdm(train_loader, desc=f"[E{epoch:02d}] Train"):
        xt, x0, r, w = (b[k].to(device) for k in ("xt", "x0", "r", "w"))
        optimizer.zero_grad()
        pred = model(xt, r)
        loss = loss_fn(pred, x0, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    history["train"].append(train_loss)

    # ---------- Val ----------
    model.eval()
    val_loss, val_ssim, tot_mse, n_img = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for b in val_loader:
            xt, x0, r, w = (b[k].to(device) for k in ("xt", "x0", "r", "w"))
            pred = model(xt, r)

            batch_loss = loss_fn(pred, x0, w)
            val_loss   += batch_loss.item()

            val_ssim   += (1 - ssim_unit(pred, x0)).item()

            tot_mse    += per_img_mse(pred, x0).sum().item()
            n_img      += pred.size(0)

    val_loss /= len(val_loader)
    val_ssim /= len(val_loader)
    val_mse   = tot_mse / n_img

    lr_scheduler.step(val_loss)

    history["val" ].append(val_loss)
    history["ssim"].append(val_ssim)
    history["mse" ].append(val_mse)
    print(f"E{epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
          f"SSIM {val_ssim:.3f} | MSE {val_mse:.4f}")

    # ---------- Early-stop & save ----------
    if val_loss < best_val:
        best_val, stall = val_loss, 0
        torch.save(model.state_dict(), f"{ckpt_dir}/best_sspdpm_xcat.pth")  # 
    else:
        stall += 1
        print(f"Stall {stall}/{patience}")
        if stall >= patience:
            print("Early stop.")
            break

# save final-epoch model
torch.save(model.state_dict(),
           f"{ckpt_dir}/final_{datetime.now():%Y%m%d_%H%M%S}_sspdpm_xcat.pth")


ep = np.arange(1, len(history["train"]) + 1)

plt.figure(figsize=(10, 4))
plt.plot(ep, history["train"], label="Train")
plt.plot(ep, history["val"],   label="Val")
plt.ylabel("Weighted MSE"); plt.legend(); plt.title("Loss"); plt.show()
plt.savefig(f"{ckpt_dir}/loss_plot_xcat.png")

plt.figure(figsize=(10, 4))
plt.plot(ep, history["ssim"], label="Val SSIM")
plt.ylabel("SSIM"); plt.legend(); plt.title("SSIM"); plt.show()
plt.savefig(f"{ckpt_dir}/ssim_plot_xcat.png")

plt.figure(figsize=(10, 4))
plt.plot(ep, history["mse"], label="Val MSE")
plt.ylabel("MSE"); plt.legend(); plt.title("Mean MSE"); plt.show()
plt.savefig(f"{ckpt_dir}/mse_plot_xcat.png")
