import os
import sys
import glob
import pandas as pd
import numpy as np
import mne
from tqdm import tqdm

# Add stroke experiments dir to path for hw_guard
sys.path.append("/home/mashfiq/mwi_stroke_experiments")
sys.path.append("/home/mashfiq/mwi_stroke_experiments/lib")  # hw_guard moved to lib/
from hw_guard import cool_gate, cap_threads

# Enforce hardware guard limits on CPU threads before importing torch/scipy implicitly via other libs
cap_threads(4)

# Config
DATA_DIR = "/home/mashfiq/eeg_vjepa/data/NMT-Scalp-EEG"
OUT_DIR = "/home/mashfiq/eeg_vjepa/data/NMT_preprocessed_100hz"
LABELS_CSV = os.path.join(DATA_DIR, "Labels.csv")
# Paper's EEG-VJEPA checkpoints assume 100 Hz frames (500 samples = 5 s). NMT is
# native 200 Hz, so resample to match -- otherwise each frame is 2.5 s and the
# pretrained encoder's patch filters see the wrong time/frequency scale.
TARGET_FS = 100

# We want 19 channels. 
# The ViT PatchEmbed has stride 4 on 19 rows -> truncates after 16 rows.
# We put the 3 most expendable channels at the end (FZ, CZ, PZ).
CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
    'FZ', 'CZ', 'PZ'
]
WINDOW_SIZE = 500      # 5 seconds at 100 Hz (matches paper checkpoints)
# Paper preprocessing (arXiv:2507.03633 §4.1), reproduced exactly:
CROP_SECONDS = 300     # crop to a fixed 5-minute length
BANDPASS = (1.0, 40.0) # bandpass 1-40 Hz (encoder was pretrained on this band)
WINDOW_STRIDE = 250    # 50% overlap -> ~118 windows per 5 min (paper's 118x19x500)

def preprocess_edf(edf_path, out_path, ch_map=None, crop_start=0.0):
    try:
        # Load header only; data is read after the 5-min crop (same output, far less I/O)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        if ch_map:   # e.g. TUAB 'EEG FP1-REF' -> 'FP1'
            raw.rename_channels({k: v for k, v in ch_map.items() if k in raw.ch_names})

        # Pick channels (ignore missing if any, but they should be there)
        raw.pick_channels(CHANNELS)
        
        # Reorder to match CHANNELS exactly
        raw.reorder_channels(CHANNELS)

        # Paper §4.1, in order: crop to fixed 5 min -> bandpass 1-40 Hz ->
        # downsample to 100 Hz -> channel-wise z-score -> overlapping 5 s windows.
        t0 = min(crop_start, max(0.0, raw.times[-1] - CROP_SECONDS))   # rec too short for the offset: keep its last 5 min
        raw.crop(tmin=t0, tmax=min(t0 + CROP_SECONDS, raw.times[-1]))
        raw.load_data(verbose=False)
        raw.filter(BANDPASS[0], BANDPASS[1], verbose=False)
        if round(raw.info['sfreq']) != TARGET_FS:
            raw.resample(TARGET_FS)

        # Get data: shape (n_channels, n_times)
        data = raw.get_data()

        # Per-channel z-score normalization
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1.0
        data = (data - mean) / std

        # Overlapping 5 s windows -> (num_frames, 19, 500)
        n_channels, n_times = data.shape
        if n_times < WINDOW_SIZE:
            return False   # under one window after cropping -> skip
        num_frames = (n_times - WINDOW_SIZE) // WINDOW_STRIDE + 1
        frames = np.stack([
            data[:, i * WINDOW_STRIDE : i * WINDOW_STRIDE + WINDOW_SIZE]
            for i in range(num_frames)
        ]).astype(np.float32)   # (num_frames, 19, 500)

        tmp = out_path[:-4] + ".tmp.npy"          # atomic: a crash never leaves a half-written .npy
        np.save(tmp, frames)
        os.replace(tmp, out_path)
        return True
    except Exception as e:
        print(f"Failed to process {edf_path}: {e}")
        return False

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(LABELS_CSV)
    
    # Create required subdirectories
    for split in ['train', 'eval']:
        for label in ['normal', 'abnormal']:
            os.makedirs(os.path.join(OUT_DIR, split, label), exist_ok=True)
            
    # Iterate over all files
    success_count = 0
    
    # We will search for all edf files and match them with Labels.csv
    # The recordname in Labels.csv usually matches the basename without extension
    # eeg_vjepa uses train/eval splits. The NMT loc is 'train' or 'eval'.
    
    # Pre-build a dict for faster lookup
    # loc in NMT is 'train' or 'eval'
    record_info = {}
    for _, row in df.iterrows():
        record_info[row['recordname']] = {
            'label': row['label'],
            'loc': row['loc']
        }
        
    edf_files = glob.glob(os.path.join(DATA_DIR, "**/*.edf"), recursive=True)
    
    for edf_path in tqdm(edf_files):
        # Apply hardware thermal guard before each heavy file
        cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=True)
        
        basename = os.path.basename(edf_path)
        recordname = basename
        
        if recordname not in record_info:
            continue
            
        info = record_info[recordname]
        label = info['label']
        split = info['loc']
        
        out_path = os.path.join(OUT_DIR, split, label, f"{recordname.replace('.edf', '')}.npy")
        
        if not os.path.exists(out_path):
            if preprocess_edf(edf_path, out_path):
                success_count += 1
        else:
            success_count += 1
            
    print(f"Successfully processed {success_count} / {len(edf_files)} files.")

if __name__ == "__main__":
    main()
