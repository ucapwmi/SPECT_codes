import os
import matplotlib.pyplot as plt

def visualize_predictions(pred_np, gt_np, inp_np,
                          save_dir, sample_name,
                          axis=0, slice_index=None):
    os.makedirs(save_dir, exist_ok=True)
    if slice_index is None:
        slice_index = inp_np.shape[axis] // 2
    def take_slice(vol):
        if axis == 0:   return vol[slice_index, :, :]
        elif axis == 1: return vol[:, slice_index, :]
        else:           return vol[:, :, slice_index]
    slices = [take_slice(arr) for arr in (inp_np, gt_np, pred_np)]
    titles = ["Input", "Ground Truth", "Prediction"]
    vmin = min(sl.min() for sl in slices)
    vmax = max(sl.max() for sl in slices)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ims = []
    for ax, img, title in zip(axs, slices, titles):
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
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