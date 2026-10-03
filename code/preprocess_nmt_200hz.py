"""The 200 Hz NMT input FEI and GTJ read (data/NMT_preprocessed/{train,eval}/{normal,abnormal}/<id>.npy), recovered
2026-10-03: the script that first wrote it (2026-08-10) was never kept. This recipe reproduces those files byte for
byte: the 19 CHANNELS of preprocess_nmt.py in that order, NMT's native 200 Hz, no filter, no crop, no resample,
per-channel z-score over the whole recording, then non-overlapping 500-sample (2.5 s) frames, float32 (F, 19, 500);
a trailing partial frame is dropped.
  python preprocess_nmt_200hz.py            # verify every existing file against the recipe, write nothing
  python preprocess_nmt_200hz.py --out DIR  # write a fresh copy to DIR (never overwrites data/NMT_preprocessed)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads, cool_gate
cap_threads(1)
import mne
import numpy as np
import pandas as pd

ROOT = "/home/mashfiq/eeg_vjepa/data"
RAW, REF = f"{ROOT}/NMT-Scalp-EEG", f"{ROOT}/NMT_preprocessed"
CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']   # = preprocess_nmt.CHANNELS


def frames(edf):
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
    assert round(raw.info["sfreq"]) == 200, raw.info["sfreq"]
    d = raw.pick(CHANNELS).reorder_channels(CHANNELS).get_data()
    z = (d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)
    return np.stack([z[:, i * 500:(i + 1) * 500] for i in range(z.shape[1] // 500)]).astype(np.float32)


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    assert out is None or os.path.abspath(out) != REF, "write a fresh copy, not over the original"
    lab = pd.read_csv(f"{RAW}/Labels.csv")
    bad = []
    for i, r in enumerate(lab.itertuples()):
        cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=True)
        rel = f"{r.loc}/{r.label}/{r.recordname[:-4]}.npy"
        x = frames(f"{RAW}/{r.label}/{r.loc}/{r.recordname}")
        if out:
            os.makedirs(os.path.dirname(f"{out}/{rel}"), exist_ok=True); np.save(f"{out}/{rel}", x)
        elif not (os.path.exists(f"{REF}/{rel}") and np.array_equal(np.load(f"{REF}/{rel}"), x)):
            bad.append(rel)
        if i % 200 == 0:
            print(f"{i}/{len(lab)}  mismatches so far: {len(bad)}", flush=True)
    print(f"done: {len(lab)} recordings" + ("" if out else f", {len(lab) - len(bad)} byte-identical, mismatches: {bad}"), flush=True)


if __name__ == "__main__":
    main()
