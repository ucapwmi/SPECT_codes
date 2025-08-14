import numpy as np
from monai.transforms import (
    Compose, ToTensord, MapTransform, Lambdad,
    RandFlipd, RandRotate90d, Rand3DElasticd,
)
from torch.utils.data import Dataset
class LoadNPYd(MapTransform):
    """
    read one .npy file and return a numpy.ndarray (D, H, W)
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        out = dict(data)
        for key in self.keys:
            npy_path = data[key]
            arr = np.load(npy_path)   # load the numpy array
            if arr.ndim != 3:
                raise ValueError(f"Expected a 3D numpy array for key '{key}', but got shape {arr.shape}.")
            # ensure the array is float32
            out[key] = arr.astype(np.float32)
        return out

class SaveMeand(MapTransform):
    "save noisy(input) mean to data['scale']"
    def __call__(self, data):
        d = dict(data)
        scale = float(d["input"].mean())
        # avoid division by zero
        if scale < 1e-8:
            scale = 1e-8
        d["scale"] = scale
        return d

class DivideByScaled(MapTransform):
    """use data['scale'] to normalize input / label"""
    def __call__(self, data):
        d = dict(data)
        s = d["scale"]
        for k in self.keys:          # self.keys = ["input","label"]
            d[k] = d[k] / s
        return d

add_channel = Lambdad(keys=["input", "label"], func=lambda x: x[np.newaxis, ...])  # (1,D,H,W)

def build_transforms(split: str) -> Compose:
    """
    split: 'train' | 'val'
    """
    split = split.lower()
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    base = [
        LoadNPYd(keys=["input", "label"]),
        add_channel,
        SaveMeand(keys=["input"]),
        DivideByScaled(keys=["input", "label"]),
    ]

    if split == "train":
        aug = [
            RandFlipd(keys=["input","label"], spatial_axis=[0], prob=0.5),
            RandRotate90d(keys=["input","label"], spatial_axes=[1,2], prob=0.5, max_k=3),
            Rand3DElasticd(keys=["input","label"], sigma_range=(3,5), magnitude_range=(3,5), prob=0.2),
        ]
        base += aug

    base.append(ToTensord(keys=["input","label","scale"]))
    return Compose(base)

class TestNPYDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
        self.transform = Compose([
            LoadNPYd(keys=["input", "label"]),
            Lambdad(keys=["input", "label"], func=lambda x: x[np.newaxis, ...]),
            SaveMeand(keys=["input"]),
            DivideByScaled(keys=["input", "label"]),
            ToTensord(keys=["input", "label", "scale"]),
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        return self.transform(self.files[idx])
    
# transforms for single-step pdpm

class LoadNPY_pdpm(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d   = dict(data)
        arr = np.load(d[self.keys[0]]).astype(np.float32)
        m   = max(arr.max(), 1e-6)               # max-norm
        arr = np.clip(arr / m * 255.0, 0, 255).astype(np.uint8)
        d["img"] = arr                          # uint8 → Tensor
        return d
    
def add_channel_pdpm(x):                 # -> [1,D,H,W]
    return x[np.newaxis] if x.ndim == 3 else x

transform_pdpm = Compose([
    LoadNPY_pdpm(keys=["path"]),
    Lambdad(keys=["img"], func=add_channel_pdpm),
    ToTensord(keys=["img"])
])