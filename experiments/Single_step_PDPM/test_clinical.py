import torch
import numpy as np
import random
import os
from models.Single_step_PDPM.model import UNet3D_PDPM
from core.visualization import visualize_sspdpm
import re
from tqdm import tqdm
from core.data import make_groups_from_sino 

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ckpt_path = "models/Single_step_PDPM/best_sspdpm_xcat.pth"
data_root = "data/volume" 
save_dir = "Single_step_PDPM_visualizations_sino"
batch_size = 1
SLICE_IDX = None      #  None == mid slice
SAVE_PRED_NPY = False
PRED_DIR = os.path.join(data_root, "pred_sino")  # save pred results to do osem in SIRF
os.makedirs(save_dir, exist_ok=True)

if SAVE_PRED_NPY:
    os.makedirs(PRED_DIR, exist_ok=True)

Count_VALUES = [1.0, 0.5, 0.25, 0.125, 0.05]  

def arr_to_u8(arr: np.ndarray, max_cnt: float) -> np.ndarray:        
    """Normalize to 0-255 range using given max_cnt, then convert to uint8"""
    return np.clip(arr / max_cnt * 255.0, 0, 255).round().astype(np.uint8)

def add_channel(x: np.ndarray) -> np.ndarray:
    """Add channel dimension: converts (D,H,W) to (1,D,H,W); returns unchanged if already 4D"""
    return x[np.newaxis] if x.ndim == 3 else x

groups = make_groups_from_sino(data_root)  # {'1.0':[{'input':..., 'label':...}], ...}

pairs = []
for lvl_str in ["1.0", "0.5", "0.25", "0.125", "0.05"]:
    for d in groups.get(lvl_str, []):
        pairs.append({"clean": d["label"], "noisy": d["input"], "r": float(lvl_str)})

print(f"Found {len(pairs)} test pairs.")

model = UNet3D_PDPM().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Loaded model from {ckpt_path}")

idx_re = re.compile(r"_(\d{4})\.npy$")  

with torch.no_grad():
    for i in tqdm(range(0, len(pairs), batch_size), desc="Testing"):
        batch = pairs[i : i + batch_size]

        # ----------- Load and normalize -----------
        noisy_u8, clean_u8 = [], []
        noisy_raw, clean_raw, max_cnts = [], [], []
        r_vals, stems = [], []

        for item in batch:
            cln = np.load(item["clean"]).astype(np.float32)     # (seg,D,H,W) or (D,H,W)
            nsy = np.load(item["noisy"]).astype(np.float32)
            max_cnt = cln.max() + 1e-6                          # per-volume max
            max_cnts.append(max_cnt)

            clean_raw.append(cln)
            noisy_raw.append(nsy)

            clean_u8.append(add_channel(arr_to_u8(cln, max_cnt)))
            noisy_u8.append(add_channel(arr_to_u8(nsy, max_cnt)))

            r_vals.append(item["r"])

            base = os.path.basename(item["noisy"])
            stems.append(os.path.splitext(base)[0])

        # ----------- inference -----------
        noisy_t = torch.tensor(np.stack(noisy_u8), dtype=torch.float32,
                               device=device) / 255.0           # (B,1,D,H,W)
        r_t     = torch.tensor(r_vals, dtype=torch.float32, device=device)
        pred = model(noisy_t, r_t).cpu().numpy()                 # (B,1,D,H,W)

        # ----------- save and visualize -----------
        for j in range(pred.shape[0]):
            stem    = stems[j]
            max_cnt = max_cnts[j]
            pred_raw = pred[j, 0] * max_cnt                   

            if SAVE_PRED_NPY:
                np.save(os.path.join(PRED_DIR, f"pred_{stem.replace('sino_', '')}.npy"),
                        pred_raw.astype(np.float32))

        
            cln_vis = clean_raw[j][0] if clean_raw[j].ndim == 4 else clean_raw[j]
            nsy_vis = noisy_raw[j][0] if noisy_raw[j].ndim == 4 else noisy_raw[j]
            prd_vis = pred_raw        if pred_raw.ndim    == 3 else pred_raw[0]

            D = prd_vis.shape[0]
            z = SLICE_IDX if SLICE_IDX is not None else D // 2
            visualize_sspdpm(nsy_vis, prd_vis, cln_vis, z, stem, vmax=max_cnt, SAVE_DIR=save_dir)  

print("Inference finished. Visuals saved to", save_dir)
