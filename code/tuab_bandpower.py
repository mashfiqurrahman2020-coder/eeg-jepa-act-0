"""P1.3 — eval-pipeline positive control on TUAB: relative band power (19 ch x 5 bands) -> LogReg.
Sanity only (not an arm): if our TUAB prep/splits are sound this must land near the feature-based
literature (Gemein 2020: ~0.80-0.86 acc). Same data the ViT sees (§4.1 100 Hz npy), same splits:
the 5 paper-style 546-rec subsets + the full 2717-rec train set, all tested on the official eval set.
  python tuab_bandpower.py      -> code/runs/act0/p13_bandpower/{results.json,preds_*.npz}
"""
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads, cool_gate
cap_threads(4)
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from act0_metrics import fmt, metrics, save

ROOT = "/home/mashfiq/eeg_vjepa"
SFX = f"_c{os.environ['TUAB_CROP']}" if os.environ.get("TUAB_CROP") else ""   # TUAB_CROP=60 -> 60-360 s crop variant
PREP, SUBS = f"{ROOT}/data/TUAB_preprocessed_100hz{SFX}", f"{ROOT}/data/TUAB_subsets{SFX}"
OUT = f"{ROOT}/code/runs/act0/p13_bandpower{SFX}"
FS = 100
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]   # data are 1-40 Hz band-passed (§4.1)


def rbp(path):
    x = np.load(path)                                       # (119, 19, 500) 5 s frames, 50% overlap
    f, p = welch(x, fs=FS, nperseg=250, axis=-1)            # overlap doesn't bias a frame-averaged PSD
    p = p.mean(0)                                           # (19, freqs)
    bp = np.stack([p[:, (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], -1)
    return (bp / bp.sum(-1, keepdims=True)).ravel()         # (95,) relative band power


def feats(files, cache):
    if os.path.exists(cache):
        return np.load(cache)
    X = []
    for i, f in enumerate(files):
        if i % 100 == 0:
            cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)
        X.append(rbp(f))
    X = np.array(X, np.float32); np.save(cache, X)
    return X


def split(d):
    files = sorted(glob.glob(f"{d}/normal/*.npy")) + sorted(glob.glob(f"{d}/abnormal/*.npy"))
    return files, np.array([int("/abnormal/" in f) for f in files])   # abnormal = positive = 1


def main():
    os.makedirs(OUT, exist_ok=True)
    fe, ye = split(f"{PREP}/eval")
    ft, yt = split(f"{PREP}/train")
    Xe, Xt = feats(fe, f"{OUT}/X_eval.npy"), feats(ft, f"{OUT}/X_train.npy")
    idx = {os.path.basename(f): i for i, f in enumerate(ft)}
    runs = {f"s{s}": [idx[os.path.basename(f)] for f in split(f"{SUBS}/s{s}/train")[0]] for s in range(5)}
    runs["full_train"] = list(range(len(ft)))
    res = {}
    for name, tr in runs.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(Xt[tr], yt[tr])
        p = clf.predict_proba(Xe)[:, 1]
        res[name] = metrics(ye, p) | {"n_train": len(tr)}
        np.savez(f"{OUT}/preds_{name}.npz", prob_abnormal=p, label=ye, rec=np.array(fe))
        print(f"{name:10s} n_train {len(tr):4d}  {fmt(res[name])}", flush=True)
    sub = np.array([[res[f"s{s}"][k] for k in ("acc", "f1", "auroc", "bal_acc")] for s in range(5)])
    res["subsets_mean"] = dict(zip(("acc", "f1", "auroc", "bal_acc"), sub.mean(0).tolist()))
    res["subsets_sd"] = dict(zip(("acc", "f1", "auroc", "bal_acc"), sub.std(0).tolist()))
    # pre-declared gate: pipeline is sound if full-train band power lands in the feature-based range
    res["gate_pass"] = bool(res["full_train"]["auroc"] >= 0.80 and res["full_train"]["bal_acc"] >= 0.72)
    save(f"{OUT}/results.json", res)
    m, s = res["subsets_mean"], res["subsets_sd"]
    print(f"546-subsets mean: Acc {m['acc']:.3f}±{s['acc']:.3f}  F1 {m['f1']:.3f}  AUROC {m['auroc']:.3f}±{s['auroc']:.3f}")
    print("GATE P1.3:", "PASS" if res["gate_pass"] else "FAIL")


if __name__ == "__main__":
    main()
