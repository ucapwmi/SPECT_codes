import os
import matplotlib.pyplot as plt
from matplotlib import gridspec

def visualize_predictions(pred_np, gt_np, inp_np,
                          save_dir, sample_name,
                          axis=0, slice_index=None,
                          vmin=None, vmax=None): 
    os.makedirs(save_dir, exist_ok=True)
    if slice_index is None:
        slice_index = inp_np.shape[axis] // 2

    def take_slice(vol):
        if axis == 0:   return vol[slice_index, :, :]
        elif axis == 1: return vol[:, slice_index, :]
        else:           return vol[:, :, slice_index]

    slices = [take_slice(arr) for arr in (inp_np, gt_np, pred_np)]
    titles = ["Input", "Ground Truth", "Prediction"]

    if vmin is None or vmax is None:
        vmin_use = min(sl.min() for sl in slices)
        vmax_use = max(sl.max() for sl in slices)
    else:
        vmin_use = vmin
        vmax_use = vmax

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ims = []
    for ax, img, title in zip(axs, slices, titles):
        im = ax.imshow(img, cmap='gray', vmin=vmin_use, vmax=vmax_use)
        ax.set_title(title)
        ax.axis('off')
        ims.append(im)

    cbar = fig.colorbar(ims[0], ax=axs, shrink=0.85,
                        location='right', aspect=35, pad=0.02)
    cbar.ax.set_ylabel('Counts', rotation=270, labelpad=15)

    plt.savefig(os.path.join(save_dir,
                f"{sample_name}_axis{axis}_slice{slice_index}.png"),
                dpi=300)
    plt.close()

def visualize_sspdpm(noisy, pred, clean, z, stem, vmax, SAVE_DIR):             
    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,0.04], wspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    ims  = []
    for ax, img, title in zip(
            axes, [noisy[z], clean[z], pred[z]], ["Noisy", "Ground Truth", "Predict"]):
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)   
        ims.append(im)
        ax.set_title(title); ax.axis("off")
    cax = fig.add_subplot(gs[3])
    fig.colorbar(ims[0], cax=cax)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{stem}_cmp.png"), dpi=200)
    plt.close()
    
# osem sspdpm
def take_slice(vol, axis=0, idx=None):
    if idx is None:
        idx = vol.shape[axis] // 2
    if axis == 0:   return vol[idx, :, :]
    elif axis == 1: return vol[:, idx, :]
    else:           return vol[:, :, idx]

def visualize_triplet(pred_np, gt_np, noisy_np, save_path, axis=0, slice_index=None):
    slices = [
        take_slice(noisy_np, axis, slice_index),
        take_slice(gt_np, axis, slice_index),
        take_slice(pred_np, axis, slice_index)
    ]
    titles = ["Noisy Input", "Ground Truth", "Prediction"]

    vmin = min(sl.min() for sl in slices)
    vmax = max(sl.max() for sl in slices)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ims = []
    for ax, img, title in zip(axs, slices, titles):
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        ims.append(im)

    cbar = fig.colorbar(ims[0], ax=axs, shrink=0.85, location='right', aspect=35, pad=0.02)
    cbar.ax.set_ylabel('Counts', rotation=270, labelpad=15)

    plt.savefig(save_path, dpi=300)
    plt.close()