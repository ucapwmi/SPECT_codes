import os
from glob import glob


def make_data_list(root_dir, phase):
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