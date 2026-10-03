"""P1.4 — released EEG-VJEPA checkpoints (HF amir-hlp/EEG-VJEPA, linked from the authors' GitHub)
on TUAB with the AUTHORS' eval protocol (configs/evals/cluster_vitt16_EEG.yaml):
500 ep, batch 2, lr 1e-3 const, wd 1e-3, fp32, upstream AttentionClassifier, unweighted CE.
Paper §4.2 subset: train 276N+270A drawn from TUAB train (5 seeded draws = the "5 runs"),
eval = official TUAB eval 150N+126A.

  python tuab_released_eval.py --subsets          # build data/TUAB_subsets/s{0..4} (symlinks)
  python tuab_released_eval.py --run [--only TAG] [--epochs N]   # sequential, resumable
  python tuab_released_eval.py --summary
"""
import argparse
import csv
import glob
import os
import random
import subprocess
import sys

import yaml

ROOT = "/home/mashfiq/eeg_vjepa"
PREP = f"{ROOT}/data/TUAB_preprocessed_100hz"
SUBS = f"{ROOT}/data/TUAB_subsets"
RUNS = f"{ROOT}/code/runs/tuab_released"
N_TRAIN = {"normal": 276, "abnormal": 270}
SEEDS = range(5)
MODELS = {  # README + paper Table 2 (v9.2 for ViT-M; ViT-B 4x30x2 = frames 16, sampling 2, tubelet 2)
    "vitm": dict(ckpt="eeg_vjepa_ViT-M_4×30×4.pth.tar", model_name="vit_small",
                 frames_per_clip=32, frame_step=3, tubelet_size=4),
    "vitb": dict(ckpt="eeg_vjepa_ViT-B_4×30×2.pth.tar", model_name="vit_base",
                 frames_per_clip=16, frame_step=2, tubelet_size=2),
}


def build_subsets():
    recs = [(lab, f) for lab in N_TRAIN for f in sorted(glob.glob(f"{PREP}/train/{lab}/*.npy"))]
    for s in SEEDS:
        rng = random.Random(s)
        order = recs[:]
        rng.shuffle(order)
        used, take = set(), {lab: [] for lab in N_TRAIN}
        for lab, f in order:   # <=1 recording per subject, class quotas as in the paper
            subj = os.path.basename(f).split("_")[0]
            if subj in used or len(take[lab]) >= N_TRAIN[lab]:
                continue
            used.add(subj); take[lab].append(f)
        d = f"{SUBS}/s{s}"
        for lab, fs in take.items():
            assert len(fs) == N_TRAIN[lab], (s, lab, len(fs))
            os.makedirs(f"{d}/train/{lab}", exist_ok=True)
            for f in fs:
                link = f"{d}/train/{lab}/{os.path.basename(f)}"
                if not os.path.lexists(link):
                    os.symlink(f, link)
        if not os.path.lexists(f"{d}/eval"):
            os.symlink(f"{PREP}/eval", f"{d}/eval")
        n = {lab: len(os.listdir(f"{d}/train/{lab}")) for lab in N_TRAIN}
        assert n == N_TRAIN, n
        print(f"s{s}: {n} + eval -> {PREP}/eval")


def runs(epochs):
    # order = headline first: ViT-M, its random-init control (same arch/probe, no load), then ViT-B
    # head: 'attention' = §4.2 text / upstream default; 'attentive' = Fig. 3 caption (paper is ambiguous)
    out = [(f"vitm_frozen_s{s}", "vitm", s, MODELS["vitm"]["ckpt"], "attention") for s in SEEDS]
    out.insert(1, ("vitm_attentive_s0", "vitm", 0, MODELS["vitm"]["ckpt"], "attentive"))
    # batch-256 sensitivity check; the paper's protocol uses batch 2
    out.insert(2, ("vitm_frozen_b256_s0", "vitm", 0, MODELS["vitm"]["ckpt"], "attention", 256))
    out = [r if len(r) == 6 else r + (2,) for r in out]
    out += [(f"vitm_rand_s{s}", "vitm", s, "none", "attention", 2) for s in range(3)]
    out += [(f"vitb_frozen_s{s}", "vitb", s, MODELS["vitb"]["ckpt"], "attention", 2) for s in SEEDS]
    return out


def write_cfg(tag, m, s, ckpt, head, bs, epochs):
    M = MODELS[m]
    os.makedirs(RUNS, exist_ok=True)
    link = f"{RUNS}/{M['ckpt']}"
    if not os.path.lexists(link):
        os.symlink(f"{ROOT}/pretrained/{M['ckpt']}", link)
    cfg = {
        "nodes": 1, "tasks_per_node": 1, "tag": tag, "eval_name": "eeg_classification_frozen",
        "resume_checkpoint": True, "seed": s, "classifier": head, "class_weights": None,
        "data": {"dataset_type": "VideoDataset", "datasets": [f"{SUBS}/s{s}"],
                 "dataset_train": f"{SUBS}/s{s}", "dataset_val": f"{SUBS}/s{s}",
                 "num_classes": 2, "num_segments": 1, "num_views_per_segment": 1,
                 "resolution": 224, "num_workers": 4, "supervised": False},
        "optimization": {"num_epochs": epochs, "batch_size": bs, "weight_decay": 0.001,
                         "lr": 0.001, "start_lr": 0.001, "final_lr": 0.0, "warmup": 0.0,
                         "use_bfloat16": False, "resolution": 224},
        "pretrain": {"model_name": M["model_name"], "checkpoint_key": "target_encoder",
                     "clip_duration": None, "frames_per_clip": M["frames_per_clip"],
                     "frame_step": M["frame_step"], "tubelet_size": M["tubelet_size"],
                     "uniform_power": True, "use_sdpa": True, "use_silu": False,
                     "tight_silu": False, "patch_size": [4, 30], "folder": RUNS,
                     "checkpoint": ckpt, "write_tag": "probe"},
    }
    p = f"{RUNS}/{tag}.yaml"
    yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
    return p


def epochs_done(tag):
    p = f"{RUNS}/{tag}/probe_r0.csv"
    if not os.path.exists(p):
        return 0
    rows = list(csv.reader(open(p)))[1:]
    return len(rows)


def run(only, epochs):
    for tag, m, s, ckpt, head, bs in runs(epochs):
        if only and only not in tag:
            continue
        if epochs_done(tag) >= epochs:
            print(f"[{tag}] done -> skip", flush=True); continue
        cfg = write_cfg(tag, m, s, ckpt, head, bs, epochs)
        print(f"[{tag}] start (resume from ep {epochs_done(tag)})", flush=True)
        with open(f"{RUNS}/{tag}.log", "a") as log:
            r = subprocess.run([sys.executable, "-m", "evals.main", "--fname", cfg,
                                "--devices", "cuda:0"], cwd=f"{ROOT}/code", stdout=log,
                               stderr=subprocess.STDOUT)
        print(f"[{tag}] exit {r.returncode}, epochs {epochs_done(tag)}", flush=True)
        if r.returncode or epochs_done(tag) < epochs:   # evals.main can crash yet exit 0
            sys.exit(f"[{tag}] incomplete -> driver stopped")


def summary():
    import numpy as np
    print("targets frozen: ViT-M Acc 83.30 F1 82.4 AUROC 87.7 | ViT-B 81.20 / 81.0 / 87.9")
    for group in ["vitm_frozen", "vitm_frozen_b256", "vitm_attentive", "vitb_frozen", "vitm_rand"]:
        fin, best = [], []
        for p in sorted(glob.glob(f"{RUNS}/{group}_s*/probe_r0.csv")):
            rows = list(csv.reader(open(p)))
            hdr, rows = rows[0], [list(map(float, r)) for r in rows[1:]]
            if not rows:
                continue
            H = {h: i for i, h in enumerate(hdr)}
            last = rows[-1]
            fin.append((last[H["val_acc"]], last[H["val_f1"]] * 100, last[H["val_auroc"]] * 100, len(rows)))
            b = max(rows, key=lambda r: r[H["val_auroc"]])
            best.append((b[H["val_acc"]], b[H["val_f1"]] * 100, b[H["val_auroc"]] * 100))
        if not fin:
            continue
        f, b = np.array(fin), np.array(best)
        print(f"{group:12s} n={len(f)} epochs={f[:, 3].astype(int).tolist()}")
        print(f"  final-epoch  Acc {f[:,0].mean():.2f}±{f[:,0].std():.2f}  F1 {f[:,1].mean():.2f}±{f[:,1].std():.2f}  AUROC {f[:,2].mean():.2f}±{f[:,2].std():.2f}")
        print(f"  best-val*    Acc {b[:,0].mean():.2f}  F1 {b[:,1].mean():.2f}  AUROC {b[:,2].mean():.2f}   (*optimistic: selected on eval)")


def status(epochs):
    import time
    print(f"{'run':15s} {'epoch':>9s}  {'val acc':>7s} {'AUROC':>6s} {'F1':>5s}   {'best AUROC':>10s}  ETA")
    for tag, *_ in runs(epochs):
        p = f"{RUNS}/{tag}/probe_r0.csv"
        if not os.path.exists(p):
            print(f"{tag:15s} {'queued':>9s}"); continue
        rows = [list(map(float, r)) for r in list(csv.reader(open(p)))[1:]]
        if not rows:
            print(f"{tag:15s} {'starting':>9s}"); continue
        n, last = len(rows), rows[-1]
        best = max(rows, key=lambda r: r[4])
        eta = ""
        if n < epochs:   # epoch time from CSV mtime vs. creation of run dir
            spe = (os.path.getmtime(p) - os.path.getctime(f"{RUNS}/{tag}.yaml")) / n
            eta = f"{(epochs - n) * spe / 60:.0f} min"
        print(f"{tag:15s} {n:>4d}/{epochs:<4d}  {last[2]:7.2f} {last[4]*100:6.1f} {last[8]*100:5.1f}"
              f"   {best[4]*100:6.1f} @{int(best[0]):<3d}  {eta}")
    print(f"\ntarget ViT-M: Acc 83.3 AUROC 87.7 F1 82.4 | ViT-B: 81.2 / 87.9 / 81.0   ({time.strftime('%H:%M')})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--epochs", type=int, default=500)
    a = ap.parse_args()
    if a.subsets: build_subsets()
    if a.run: run(a.only, a.epochs)
    if a.summary: summary()
    if a.status: status(a.epochs)
