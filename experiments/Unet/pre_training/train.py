from experiments.Unet.pre_training.data import make_data_list
from core.transforms import build_transforms
from models.Unet.model import CustomUNet3D
from core.metrics import compute_psnr_volume, ssim_metric, ssim_unit, combined_loss, mse_loss
from monai.data import CacheDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 150
patience = 6
batch_size = 4

data_root = "data/dataset"
train_files = make_data_list(data_root, "train")
val_files = make_data_list(data_root, "val")

train_ds = CacheDataset(
    data=train_files,
    transform=build_transforms("train"),
    cache_rate=1.0,
    num_workers=4
)
val_ds = CacheDataset(
    data=val_files,
    transform=build_transforms("val"),
    cache_rate=1.0,
    num_workers=4
)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8
)
model = CustomUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)
lr_scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

train_losses, val_losses = [], []
train_mses,  val_mses    = [] , []    
val_psnrs,   val_ssims   = [], []

best_val_loss = float("inf")
epochs_no_improve = 0

checkpoint_dir = "models/3d_unet_checkpoints" # save checkpoints dir
os.makedirs(checkpoint_dir, exist_ok=True)

# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
for epoch in range(1, num_epochs + 1):
    # --------------------------- Train ---------------------------
    model.train()
    running_train_loss = 0.0
    running_train_mse  = 0.0       

    for batch in tqdm(train_loader, desc=f"[Epoch {epoch:02d}] Train"):
        inp, lbl = batch["input"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = combined_loss(out, lbl)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()


        scale = batch["scale"].to(device).view(-1,1,1,1,1)  # mean value of each sample
        out_cnt = out * scale
        lbl_cnt = lbl * scale
        running_train_mse += mse_loss(out_cnt, lbl_cnt).item()


    avg_train_loss = running_train_loss / len(train_loader)
    avg_train_mse  = running_train_mse  / len(train_loader)   

    train_losses.append(avg_train_loss)
    train_mses.append(avg_train_mse)                         

    print(f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.6f}, "
          f"MSE: {avg_train_mse:.6f}")                     

    # --------------------------- Val ----------------------------
    model.eval()
    running_val_loss = 0.0
    running_val_mse  = 0.0                                  
    epoch_psnr_list, epoch_ssim_list = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"[Epoch {epoch:02d}] Val"):
            inp_v, lbl_v = batch["input"].to(device), batch["label"].to(device)
            out_v = model(inp_v)

            loss_v = combined_loss(out_v, lbl_v)
            running_val_loss += loss_v.item()

            scale = batch["scale"].to(device).view(-1,1,1,1,1)
            out_cnt = out_v * scale
            lbl_cnt = lbl_v * scale

            # per-sample PSNR & SSIM -------------------
            for i in range(out_v.shape[0]):
                pred_i = out_cnt[i:i+1]
                tgt_i  = lbl_cnt[i:i+1]

                # PSNR 
                psnr_i = compute_psnr_volume(pred_i, tgt_i)
                epoch_psnr_list.append(psnr_i)

                # SSIM
                peak_i = tgt_i.max()
                pred_norm = pred_i / (peak_i + 1e-8)
                tgt_norm  = tgt_i / (peak_i + 1e-8)
                ssim_i = 1.0 - ssim_metric(pred_norm, tgt_norm)
                epoch_ssim_list.append(ssim_i.item())

            # MSE 
            running_val_mse += mse_loss(out_cnt, lbl_cnt).item()


    avg_val_loss = running_val_loss / len(val_loader)
    avg_val_mse  = running_val_mse  / len(val_loader)         
    avg_val_psnr = float(np.mean(epoch_psnr_list))
    avg_val_ssim = float(np.mean(epoch_ssim_list))

    lr_scheduler.step(avg_val_loss)

    val_losses.append(avg_val_loss)
    val_mses.append(avg_val_mse)                            
    val_psnrs.append(avg_val_psnr)
    val_ssims.append(avg_val_ssim)

    print(f"[Epoch {epoch:02d}] Val Loss: {avg_val_loss:.6f}, "
          f"MSE: {avg_val_mse:.6f}, "                       
          f"PSNR: {avg_val_psnr:.4f}, SSIM: {avg_val_ssim:.4f}")

    # ------------------- Early stopping ------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # save best model
        best_model_path = os.path.join(
            checkpoint_dir, f"best_model_3D_unet_pretrained.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model → {best_model_path}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs.")
        if epochs_no_improve >= patience:
            print(f"Early stopping: no improvement in {patience} epochs.")
            break

# --------------------------------------------------------------------------
# Save model
# --------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_path = os.path.join(
    checkpoint_dir, f"final_model_{timestamp}_3D_unet_pretraining.pth")
torch.save(model.state_dict(), final_path)
print(f"Saved final model → {final_path}")

# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------
epochs_ran = np.arange(1, len(train_losses) + 1)

plt.figure(figsize=(10,4))
plt.plot(epochs_ran, train_losses, label="Train Loss")
plt.plot(epochs_ran, val_losses,   label="Val Loss")
plt.legend(); plt.title("Loss"); plt.show()

# MSE curve
plt.figure(figsize=(10,4))
plt.plot(epochs_ran, train_mses, label="Train MSE")
plt.plot(epochs_ran, val_mses,   label="Val MSE")
plt.legend(); plt.title("MSE"); plt.show()

# PSNR curve
plt.figure(figsize=(10,4))
plt.plot(epochs_ran, val_psnrs, label="Val PSNR")
plt.legend(); plt.title("PSNR"); plt.show()

# SSIM curve
plt.figure(figsize=(10,4))
plt.plot(epochs_ran, val_ssims, label="Val SSIM")
plt.legend(); plt.title("SSIM"); plt.show()