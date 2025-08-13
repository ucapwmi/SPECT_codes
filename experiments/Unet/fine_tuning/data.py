import os
from glob import glob

def make_data_list(root_dir, phase):
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