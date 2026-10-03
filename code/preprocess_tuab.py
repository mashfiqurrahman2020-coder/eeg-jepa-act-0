"""TUAB v3.0.1 -> EEG-VJEPA §4.1 npy (same path as NMT: preprocess_nmt.preprocess_edf).

  python preprocess_tuab.py            # resumable: existing .npy are skipped
  python preprocess_tuab.py --check    # gate: counts, shapes, channels, subject-disjoint split

Out: data/TUAB_preprocessed_100hz/{train,eval}/{normal,abnormal}/<stem>.npy, (F,19,500) float32.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

import preprocess_nmt as P   # also applies cap_threads(4)
from hw_guard import cool_gate

SRC = "/home/mashfiq/eeg_vjepa/data/TUAB/edf"
SFX = f"_c{os.environ['TUAB_CROP']}" if os.environ.get("TUAB_CROP") else ""   # TUAB_CROP=60 -> 60-360 s crop variant
OUT = "/home/mashfiq/eeg_vjepa/data/TUAB_preprocessed_100hz" + SFX
CH_MAP = {f"EEG {c}-REF": c for c in P.CHANNELS}   # 01_tcp_ar naming
EXPECTED = {("train", "normal"): 1371, ("train", "abnormal"): 1346,
            ("eval", "normal"): 150, ("eval", "abnormal"): 126}   # official TUAB v3.0.1 counts
FAIL_LOG = os.path.join(OUT, "failures.txt")


def edfs():
    for (split, label) in EXPECTED:
        for f in sorted(glob.glob(f"{SRC}/{split}/{label}/01_tcp_ar/*.edf")):
            yield split, label, f


def run():
    for split, label in EXPECTED:
        os.makedirs(f"{OUT}/{split}/{label}", exist_ok=True)
    todo = list(edfs())
    done = fail = 0
    for i, (split, label, f) in enumerate(todo):
        out = f"{OUT}/{split}/{label}/{os.path.basename(f)[:-4]}.npy"
        if os.path.exists(out):
            done += 1; continue
        cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=True)
        if P.preprocess_edf(f, out, ch_map=CH_MAP, crop_start=float(os.environ.get("TUAB_CROP", 0))):
            done += 1
        else:
            fail += 1
            with open(FAIL_LOG, "a") as fh: fh.write(f + "\n")
        if i % 50 == 0:
            print(f"[{i+1}/{len(todo)}] ok={done} fail={fail}", flush=True)
    print(f"DONE ok={done} fail={fail} of {len(todo)}", flush=True)


def check():
    ok = True
    subj = {"train": set(), "eval": set()}
    nfr = []
    for (split, label), n in EXPECTED.items():
        files = sorted(glob.glob(f"{OUT}/{split}/{label}/*.npy"))
        files = [f for f in files if not f.endswith(".tmp.npy")]
        print(f"{split}/{label}: {len(files)} (expected {n})")
        ok &= len(files) == n
        for f in files:
            x = np.load(f, mmap_mode="r")
            if x.ndim != 3 or x.shape[1:] != (19, 500) or x.dtype != np.float32:
                print("BAD SHAPE", f, x.shape, x.dtype); ok = False
            nfr.append(x.shape[0])
            subj[split].add(os.path.basename(f).split("_")[0])
    both = subj["train"] & subj["eval"]
    print(f"subjects train={len(subj['train'])} eval={len(subj['eval'])} overlap={len(both)}")
    ok &= not both
    nfr = np.array(nfr)
    print(f"frames/rec: min={nfr.min()} median={int(np.median(nfr))} max={nfr.max()} "
          f"(<118 = rec shorter than 5 min: {(nfr < 118).sum()})")
    x = np.load(f, mmap_mode="r")
    print(f"last file per-ch mean|max|={np.abs(x.mean((0, 2))).max():.3f}")
    json.dump({"ok": bool(ok), "n_frames_min": int(nfr.min()),
               "subjects": {k: len(v) for k, v in subj.items()}},
              open(os.path.join(OUT, "check.json"), "w"), indent=2)
    print("GATE", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if check() else 1) if a.check else run()
