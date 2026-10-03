"""Act-0 reproducibility manifests: which recordings sit in which fold/subset/cohort, the
SHA-256 of every checkpoint the report's numbers come from, the Python environment, and the SHA-256 of every saved
result file under code/runs/act0. Read-only over data and checkpoints.
  python act0_repro_manifest.py   -> docs/act0_repro/*
"""
import csv
import glob
import hashlib
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads
cap_threads(1)
import numpy as np
from sklearn.model_selection import StratifiedKFold

import fei_pretrain as F
from stroke_phaseB import COHORTS
from tuab_released_eval import N_TRAIN, SEEDS, SUBS

ROOT = "/home/mashfiq/eeg_vjepa"
OUT = f"{ROOT}/docs/act0_repro"
PY = sys.executable


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(name, header, rows):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main():
    os.makedirs(OUT, exist_ok=True)
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    fold = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
        fold[te] = k
    write("nmt_folds_seed0.csv", ["recording", "label_abnormal", "test_fold"],
          [(os.path.relpath(p, ROOT), c, k) for (p, c), k in zip(files, fold)])
    write("tuab_subsets.csv", ["subset", "label", "file", "subject"],
          [(f"s{s}", lab, f, f.split("_")[0]) for s in SEEDS for lab in N_TRAIN
           for f in sorted(os.listdir(f"{SUBS}/s{s}/train/{lab}"))])
    write("stroke_cohorts.csv", ["cohort", "subject", "label_stroke", "rest_seconds"],
          [(c, s, int(s.startswith("PAC")), secs) for c, (subs, secs) in COHORTS.items() for s in subs])
    ck = sorted(glob.glob(f"{ROOT}/pretrained/eeg_vjepa_*.pth.tar")) + sorted(   # the two released checkpoints only
        p for pat in ("fei_enc_cv_s0_f*", "branchC_h20_enc_cv_s0_f*", "fei_joint_enc_cv_s0_f*",
                      "branchC_h20_joint_enc_cv_s0_f*", "fei_rep*_enc_cv_s0_f*", "branchC_h20_rep*_enc_cv_s0_f*",
                      "fei_joint_rep*_enc_cv_s0_f*", "branchC_h20_joint_rep*_enc_cv_s0_f*")
        for p in glob.glob(f"{ROOT}/code/{pat}.pt"))
    write("checkpoints_sha256.csv", ["file", "bytes", "sha256"], [(os.path.relpath(p, ROOT), os.path.getsize(p), sha(p)) for p in ck])
    res = sorted(glob.glob(f"{ROOT}/code/runs/act0/**/*.json", recursive=True)) + sorted(glob.glob(f"{ROOT}/code/runs/act0/*.json"))
    write("results_sha256.csv", ["file", "bytes", "sha256"], [(os.path.relpath(p, ROOT), os.path.getsize(p), sha(p)) for p in dict.fromkeys(res) if "/liu2024_feic/" not in p])   # Act-2 run, not Act 0
    freeze = subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    open(f"{OUT}/requirements-act0.txt", "w").write(freeze)
    import torch
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    up = subprocess.run(["git", "-C", f"{ROOT}/upstream/eeg-vjepa", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    hf = subprocess.run(["git", "-C", f"{ROOT}/pretrained", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = [f"python {platform.python_version()}", f"torch {torch.__version__} (CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()})",
           f"GPU {gpu}", f"OS {platform.platform()}", f"upstream eeg-vjepa commit {up}", f"pretrained (HF amir-hlp/EEG-VJEPA) commit {hf}"]
    open(f"{OUT}/environment.txt", "w").write("\n".join(env) + "\n")
    print("\n".join(env))


if __name__ == "__main__":
    main()
