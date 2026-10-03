"""Collapse + linear-separability check of EEG-VJEPA ViT-M (released vs random-init) on TUAB eval.
One deterministic clip per recording (first 96 frames, step 3 = the loader's non-random indices),
CPU only so it can run beside the GPU sweep.
  python tuab_collapse_check.py [--n 276]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append('/home/mashfiq/mwi_stroke_experiments/lib')
from hw_guard import cap_threads, cool_gate
cap_threads(4)
from evals.eeg_classification_frozen.eval import init_model

CKPT = "/home/mashfiq/eeg_vjepa/pretrained/eeg_vjepa_ViT-M_4×30×4.pth.tar"
SFX = f"_c{os.environ['TUAB_CROP']}" if os.environ.get("TUAB_CROP") else ""   # TUAB_CROP=60 -> 60-360 s crop variant
EVAL = f"/home/mashfiq/eeg_vjepa/data/TUAB_preprocessed_100hz{SFX}/eval"


def feats(ckpt, files, seed=0):
    torch.manual_seed(seed)
    enc = init_model("cpu", ckpt, "vit_small", patch_size=(4, 30), frames_per_clip=32, tubelet_size=4,
                     use_sdpa=True, use_SiLU=False, tight_SiLU=False, uniform_power=True).eval()
    out = []
    with torch.no_grad():
        for i, f in enumerate(files):
            if i % 25 == 0:
                cool_gate(pause=88.0, resume=78.0, abort=92.0)
            x = np.load(f)[np.linspace(0, 95, 32).astype(int)]           # (32,19,500)
            tok = enc(torch.from_numpy(x).float()[None, None])[0]      # (512,384)
            out.append(tok.numpy())
    return np.stack(out)                                               # (N,512,384)


def report(name, T, y):
    pooled = T.mean(1)
    tok_std = T.std(1).mean()                         # within-recording token spread
    rec_std = pooled.std(0).mean()                    # across-recording spread of pooled emb
    z = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    cos = (z @ z.T)[np.triu_indices(len(z), 1)].mean()
    X = np.concatenate([pooled, T.std(1)], 1)         # mean+std readout, standardized
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
                          X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"{name:10s} token-std {tok_std:.4f}  across-rec std {rec_std:.4f}  mean cos {cos:.4f}  "
          f"5-fold LogReg AUROC {roc_auc_score(y, p):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=276); a = ap.parse_args()
    files = sorted(glob.glob(f"{EVAL}/normal/*.npy")) + sorted(glob.glob(f"{EVAL}/abnormal/*.npy"))
    y = np.array([0 if "/normal/" in f else 1 for f in files])
    idx = np.random.RandomState(0).permutation(len(files))[:a.n]
    files, y = [files[i] for i in idx], y[idx]
    print(f"{len(files)} TUAB-eval recordings ({y.sum()} abnormal)")
    report("released", feats(CKPT, files), y)
    report("random", feats("none", files), y)
