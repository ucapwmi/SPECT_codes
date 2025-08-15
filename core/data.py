import os
from glob import glob
import re
import numpy as np


def make_data_list_pre_training(root_dir, phase):
    """
    Create a list of file paths for the given phase (train/val/test).
    root_dir: str, the root directory containing the data.
    phase: str, one of 'train', 'val', or 'test'.
    Returns [{'input': input_path.npy, 'label': label_path.npy}, ...]
    """
    data_list = []

    # Validate the phase
    if phase not in ['train', 'val', 'test']:
        raise ValueError("Phase must be one of 'train', 'val', or 'test'.")
    
    phase_dir = os.path.join(root_dir, phase)
    input_paths = sorted(glob(os.path.join(phase_dir, 'input', '*.npy'))) # input files, e.g., input_000.npy
    for input_path in input_paths:
        file_name = os.path.basename(input_path)
        index = file_name.replace('input_', '').replace('.npy', '')
        label_path = os.path.join(phase_dir, 'label', f'label_{index}.npy') # corresponding label file
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} does not exist.")
        data_list.append({'input': input_path, 'label': label_path}) 
    return data_list

def make_data_list_fine_tuning(root_dir, phase):
    if phase not in ('train','val','test'):
        raise ValueError("phase must be 'train', 'val' or 'test'")
    
    phase_dir = os.path.join(root_dir, phase)
    all_npy = sorted(glob(os.path.join(phase_dir, '*.npy')))
    
    data_list = []
    for inp in all_npy:
        fn = os.path.basename(inp)
        # only process files starting with input_
        if not fn.startswith('input_'):  
            continue
        # replace input_ with label_, keep the rest
        label_fn = 'label_' + fn[len('input_'):]
        label_path = os.path.join(phase_dir, label_fn)
        if not os.path.exists(label_path):
            print(f"✗ missing label: expected {label_fn}")
            continue
        data_list.append({'input': inp, 'label': label_path})
    
    print(f"[{phase}] found {len(data_list)} pairs.")
    if len(data_list)==0:
        raise RuntimeError(f"No samples found in {phase_dir}!")
    return data_list

def make_list_sspdpm(split,ROOT):
    files = sorted(glob(os.path.join(ROOT, split, "clean_sino", "clean_*.npy")))
    return [{"path": p} for p in files]  

def make_groups_from_osem(root_dir):
    files = sorted(glob(os.path.join(root_dir, "osem_*.npy")))
    # \osem_volumeX_nLEVEL[ _clean].npy
    pat = re.compile(
        r"^osem_volume(?P<vid>\d+)_n(?P<lvl>(?:1|0\.5|0\.25|0\.125|0\.05))(?:_clean)?\.npy$"
    )
    groups = {"1.0": [], "0.5": [], "0.25": [], "0.125": [], "0.05": []}

    for p in files:
        name = os.path.basename(p)
        m = pat.match(name)
        if not m:
            continue
        if name.endswith("_clean.npy"):
            continue

        lvl = m.group("lvl")
        lvl = "1.0" if lvl == "1" else lvl
        clean = p[:-4] + "_clean.npy"  
        if os.path.exists(clean):
            groups[lvl].append({"input": p, "label": clean})

    vid_pat = re.compile(r"osem_volume(\d+)_")
    for lvl in groups:
        groups[lvl].sort(
            key=lambda d: int(vid_pat.search(os.path.basename(d["input"])).group(1))
        )
    return groups

def make_groups_from_sino(root_dir):
    files = sorted(glob(os.path.join(root_dir, "sino_*.npy")))
    pat = re.compile(
        r"^sino_volume(?P<vid>\d+)_n(?P<lvl>(?:1|0\.5|0\.25|0\.125|0\.05))(?:_clean)?\.npy$"
    )

    groups = {"1.0": [], "0.5": [], "0.25": [], "0.125": [], "0.05": []}

    for p in files:
        name = os.path.basename(p)
        m = pat.match(name)
        if not m:
            continue
        if name.endswith("_clean.npy"):
            continue

        lvl = m.group("lvl")
        lvl = "1.0" if lvl == "1" else lvl

        clean = p[:-4] + "_clean.npy"  
        if os.path.exists(clean):
            groups[lvl].append({"input": p, "label": clean})

    vid_pat = re.compile(r"sino_volume(\d+)_")
    for lvl in groups:
        groups[lvl].sort(
            key=lambda d: int(vid_pat.search(os.path.basename(d["input"])).group(1))
        )
    return groups

def center_crop_first_axis(vol: np.ndarray, target_d=128) -> np.ndarray:
    """(256,128,128)->(128,128,128), organ files are (256,128,128), need center crop"""
    d = vol.shape[0]
    if d == target_d:
        return vol
    if d < target_d:
        raise ValueError(f"Cannot center-crop: depth {d} < target {target_d}")
    start = (d - target_d) // 2
    return vol[start:start + target_d, ...]

def load_organ_masks(seg_dir: str, target_shape=(128, 128, 128)) -> dict:

    paths = sorted(glob(os.path.join(seg_dir, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"No *.npy found in {seg_dir}")
    masks = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0] 
        arr = np.load(p) 
        arr = center_crop_first_axis(arr, target_d=target_shape[0])
        if arr.shape != target_shape:
            raise ValueError(f"Mask {name} has shape {arr.shape}, expected {target_shape}")
        mask = (arr > 0).astype(np.uint8)
        if mask.sum() == 0:
            print(f"[WARN] mask {name} is empty after crop.")
        masks[name] = mask
    print(f"Loaded {len(masks)} masks: {list(masks.keys())}")
    return masks