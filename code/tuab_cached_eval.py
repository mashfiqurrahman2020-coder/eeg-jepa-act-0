"""Cached-feature version of the frozen TUAB eval (evals/eeg_classification_frozen/eval.py).

Why it is exact: the encoder is frozen, the eval transform is the identity, and every recording
has 119 frames, so a clip can only start at P discrete positions (ViT-M: 23). The encoder output is
therefore a pure function of (recording, position) -> compute each once (fp32), then train the SAME
head with the SAME init_opt / schedulers / clip-grad / calculate_metrics imported from eval.py.
Recording order + labels come from the upstream VideoDataset and the batch order from the same
DistributedSampler(shuffle=True, never set_epoch) as the live run. Only the random clip draws differ
(a different random stream, same distribution).

  python tuab_cached_eval.py --verify                       # cached tokens == live loader+encoder
  python tuab_cached_eval.py --tag c_vitm_frozen_s0 [--subset 0 --seed 0 --ckpt released|none
                             --head attention|attentive --bs 2 --epochs 500 --fp16]
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('/home/mashfiq/mwi_stroke_experiments/lib')
from hw_guard import cap_threads, cool_gate
cap_threads(4)
import src.datasets.video_dataset as vd
from evals.eeg_classification_frozen.eval import (init_model, init_opt, calculate_metrics,
                                                   AttentionClassifier, AttentiveClassifier)
from evals.video_classification_frozen.utils import EEGTransformEval

ROOT = "/home/mashfiq/eeg_vjepa"
SFX = f"_c{os.environ['TUAB_CROP']}" if os.environ.get("TUAB_CROP") else ""   # TUAB_CROP=60 -> 60-360 s crop variant
SUBS = f"{ROOT}/data/TUAB_subsets" + SFX
CACHE = f"{ROOT}/data/TUAB_cache" + SFX
OUT = f"{ROOT}/code/runs/tuab_cached"
MODELS = {"vitm": dict(ckpt=f"{ROOT}/pretrained/eeg_vjepa_ViT-M_4×30×4.pth.tar", model_name="vit_small",
                       frames_per_clip=32, frame_step=3, tubelet_size=4),
          "vitb": dict(ckpt=f"{ROOT}/pretrained/eeg_vjepa_ViT-B_4×30×2.pth.tar", model_name="vit_base",
                       frames_per_clip=16, frame_step=2, tubelet_size=2)}
# remaining P1.4 queue (same runs as tuab_released_eval.py; b256 already done as a test)
QUEUE = ([dict(tag="c_vitm_frozen_s0"), dict(tag="c_vitm_attentive_s0", head="attentive")] +
         [dict(tag=f"c_vitm_frozen_s{s}", subset=s, seed=s) for s in range(1, 5)] +
         [dict(tag=f"c_vitm_attentive_s{s}", subset=s, seed=s, head="attentive") for s in range(1, 5)] +
         [dict(tag=f"c_vitm_rand_s{s}", subset=s, seed=s, ckpt="none") for s in range(3)] +
         [dict(tag=f"c_vitm_rand_attentive_s{s}", subset=s, seed=s, ckpt="none", head="attentive")
          for s in range(3)] +
         # ViT-B: both heads per subset back-to-back -> the ~74 GB train cache is built once per subset
         [dict(tag=f"c_vitb_{n}_s{s}", model="vitb", subset=s, seed=s, head=h) for s in range(5)
          for n, h in [("frozen", "attention"), ("attentive", "attentive")]])
DEFAULTS = dict(model="vitm", subset=0, seed=0, ckpt="released", head="attention", bs=2, epochs=500,
                lr=1e-3, fp16=False)
DEV = "cuda:0"
HDR = ["epoch", "train_acc", "val_acc", "train_auroc", "val_auroc", "train_bal_acc", "val_bal_acc",
       "train_f1", "val_f1"]


def dataset(subset, split, M):
    # the upstream dataset -> identical sample order + labels (abnormal=0, normal=1)
    return vd.VideoDataset(data_paths=[f"{SUBS}/s{subset}"], frames_per_clip=M["frames_per_clip"],
                           frame_step=M["frame_step"], num_clips=1, random_clip_sampling=True,
                           allow_clip_overlap=True, transform=None, mode=split)


def clip(frames, end, M):   # = VideoDataset.loadvideo_decord for partition_len > clip_len
    fpc, clip_len = M["frames_per_clip"], M["frames_per_clip"] * M["frame_step"]
    start = end - clip_len
    idx = np.clip(np.linspace(start, end, num=fpc), start, end - 1).astype(np.int64)
    return frames[idx]


def encoder(M, ckpt, seed):
    torch.manual_seed(seed)
    return init_model(DEV, ckpt, M["model_name"], patch_size=(4, 30), crop_size=224,
                      frames_per_clip=M["frames_per_clip"], tubelet_size=M["tubelet_size"],
                      use_sdpa=True, use_SiLU=False, tight_SiLU=False, uniform_power=True).eval()


def build_cache(ds, M, ckpt, seed, path, fp16=False):
    """(N, P, tokens, D) fp32 memmap of encoder outputs for every (recording, clip position)."""
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    enc = encoder(M, ckpt, seed)
    clip_len = M["frames_per_clip"] * M["frame_step"]
    T = np.load(ds.samples[0], mmap_mode="r").shape[0]
    ends = range(clip_len, T)                         # np.random.randint(clip_len, T) support
    tmp = path[:-4] + ".tmp.npy"
    X = None
    with torch.no_grad():
        for i, f in enumerate(ds.samples):
            if i % 25 == 0:
                cool_gate(pause=88.0, resume=78.0, abort=92.0)
                print(f"  cache {os.path.basename(path)} {i}/{len(ds.samples)}", flush=True)
            fr = np.load(f).astype("float32")
            assert fr.shape[0] == T, (f, fr.shape)
            x = torch.from_numpy(np.stack([clip(fr, e, M) for e in ends]))[:, None].to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                tok = enc(x).float().cpu().numpy()
            if X is None:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                X = np.lib.format.open_memmap(tmp, "w+", np.float32, (len(ds.samples), len(ends)) + tok.shape[1:])
            X[i] = tok
    X.flush(); del X
    os.replace(tmp, path)                              # atomic: a crash leaves only .tmp
    return np.load(path, mmap_mode="r")


def caches(a, M):
    ck = "released" if a.ckpt == "released" else f"rand{a.seed}"
    ck += "_fp16" if a.fp16 else ""
    ckpt = M["ckpt"] if a.ckpt == "released" else "none"
    out = {}
    for split, name in [("train", f"s{a.subset}_train"), ("eval", "eval")]:
        ds = dataset(a.subset, split, M)
        X = build_cache(ds, M, ckpt, a.seed, f"{CACHE}/{a.model}_{ck}/{name}.npy", a.fp16)
        out[split] = (X, np.array(ds.labels))
    return out


def epoch_pass(head, X, y, order, bs, clip_len, train, opt=None, sched=None, wd_sched=None):
    head.train(mode=train)
    accs, P, L = [], [], []
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        ends = np.random.randint(clip_len, clip_len + X.shape[1], size=len(idx))
        x = torch.from_numpy(np.stack([X[j, e - clip_len] for j, e in zip(idx, ends)])).to(DEV)
        lab = torch.as_tensor(y[idx], device=DEV)
        if train:
            sched.step(); wd_sched.step()
        with torch.set_grad_enabled(train):
            out = head(x)
            loss = F.cross_entropy(out, lab)
        with torch.no_grad():
            p = F.softmax(out, dim=1)
            P.append(p); L.append(lab)
            accs.append(float(100. * p.max(dim=1).indices.eq(lab).sum() / len(idx)))
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); opt.zero_grad()
    return float(np.mean(accs)), torch.cat(P), torch.cat(L)


def train(a):
    M = MODELS[a.model]
    C = caches(a, M)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    D = C["train"][0].shape[-1]
    head = (AttentiveClassifier(embed_dim=D, num_heads=D // 64, depth=1, num_classes=2) if a.head == "attentive"
            else AttentionClassifier(embed_dim=D, num_classes=2)).to(DEV)
    Xtr, ytr = C["train"]; Xev, yev = C["eval"]
    ipe = math.ceil(len(ytr) / a.bs)
    opt, _, sched, wd_sched = init_opt(encoder=None, classifier=head, wd=1e-3, start_lr=a.lr, ref_lr=a.lr,
                                       final_lr=0.0, iterations_per_epoch=ipe, warmup=0.0,
                                       num_epochs=a.epochs, use_bfloat16=False, supervised=False)
    samp = lambda n: np.array(list(torch.utils.data.distributed.DistributedSampler(
        range(n), num_replicas=1, rank=0, shuffle=True)))
    otr, oev = samp(len(ytr)), samp(len(yev))          # fixed across epochs, as upstream
    clip_len = M["frames_per_clip"] * M["frame_step"]
    os.makedirs(f"{OUT}/{a.tag}", exist_ok=True)
    p = f"{OUT}/{a.tag}/probe_r0.csv"
    w = csv.writer(open(p, "w", buffering=1)); w.writerow(HDR)   # a run is minutes; no resume needed
    t0, hist = time.time(), []
    for ep in range(a.epochs):
        if ep % 10 == 0:
            cool_gate(pause=88.0, resume=78.0, abort=92.0)
        tr_acc, tp, tl = epoch_pass(head, Xtr, ytr, otr, a.bs, clip_len, True, opt, sched, wd_sched)
        ev_acc, vp, vl = epoch_pass(head, Xev, yev, oev, a.bs, clip_len, False)
        ta, _, tf, tb = calculate_metrics(tp, tl)
        va, _, vf, vb = calculate_metrics(vp, vl)
        hist.append(vp[:, 0].float().cpu().numpy())
        w.writerow([ep + 1, f"{tr_acc:.5f}", f"{ev_acc:.5f}"] +
                   [f"{float(v):.5f}" for v in (ta, va, tb, vb, tf, vf)])
        if (ep + 1) % 50 == 0:
            print(f"[{a.tag}] ep {ep+1} train_acc {tr_acc:.1f} val_acc {ev_acc:.1f} val_auroc {float(va):.3f} "
                  f"({(time.time()-t0)/(ep+1):.2f} s/ep)", flush=True)
    # per-recording final-epoch outputs + head weights -> confusion/ROC/sens-spec/error analysis later
    files = np.array([str(f) for f in dataset(a.subset, "eval", M).samples])
    np.savez(f"{OUT}/{a.tag}/final_preds.npz", prob=vp.cpu().numpy(), label=vl.cpu().numpy(),
             rec=files[oev], prob_abnormal_by_epoch=np.array(hist, dtype=np.float32))
    # prob[:, 0] = P(abnormal); labels abnormal=0, normal=1; prob_abnormal_by_epoch = (epochs, n_eval), same rec order
    torch.save(head.state_dict(), f"{OUT}/{a.tag}/head.pt")
    print(f"[{a.tag}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def verify(model="vitm"):
    """Cached tokens must equal upstream loader clip -> EEGTransformEval -> encoder."""
    M = MODELS[model]
    a = argparse.Namespace(model=model, ckpt="released", seed=0, subset=0, fp16=False)
    C = caches(a, M)
    enc = encoder(M, M["ckpt"], 0)
    clip_len = M["frames_per_clip"] * M["frame_step"]
    worst = 0.0
    for split, recs in [("train", [0, 300]), ("eval", [5, 200])]:
        ds = dataset(0, split, M)
        X = C[split][0]
        for r in recs:
            for e in (clip_len, clip_len + 11, clip_len + X.shape[1] - 1):
                real = vd.np.random.randint
                vd.np.random.randint = lambda lo, hi, _e=e: _e         # force the loader's draw
                try:
                    buf, _ = ds.loadvideo_decord(ds.samples[r])
                finally:
                    vd.np.random.randint = real
                x = EEGTransformEval()(np.expand_dims(buf, 3) if buf.ndim == 3 else buf)[None].to(DEV)
                with torch.no_grad():
                    live = enc(x)[0].cpu().numpy()
                d = np.abs(live - X[r, e - clip_len]).max()
                worst = max(worst, d)
    spread = X[:, 0].mean(1).std(0).mean()
    print(f"max |cached - live| = {worst:.2e}   (between-recording spread {spread:.4f})")
    assert worst < 1e-4, worst
    print(f"VERIFY PASS ({model})")


def rows(tag):
    p = f"{OUT}/{tag}/probe_r0.csv"
    return [list(map(float, r)) for r in list(csv.reader(open(p)))[1:]] if os.path.exists(p) else []


def has_epoch_preds(tag):
    p = f"{OUT}/{tag}/final_preds.npz"
    return os.path.exists(p) and "prob_abnormal_by_epoch" in np.load(p).files


def queue(only=None):   # only = 'vitm' | 'vitb' -> two disjoint lanes can run at once
    import shutil
    # unfinished runs first, then re-run finished ones lacking per-epoch preds (only if the train cache still exists:
    # ViT-B s0's ~74 GB cache is gone -> it keeps final-epoch preds only)
    pending = lambda q: len(rows(q["tag"])) < DEFAULTS["epochs"]
    nopred = lambda q: not has_epoch_preds(q["tag"]) and (q.get("model", "vitm") == "vitm" or
                                                         os.path.exists(f"{CACHE}/vitb_released/s{q['subset']}_train.npy"))
    todo = [q for q in QUEUE if pending(q)] + [q for q in QUEUE if not pending(q) and nopred(q)]
    todo = [q for q in todo if only in (None, q.get("model", "vitm"))]
    print(f"{len(todo)} runs to do: {[q['tag'] for q in todo]}", flush=True)
    for i, q in enumerate(todo):
        if not (pending(q) or nopred(q)):   # done meanwhile (e.g. by the other lane)
            continue
        a = argparse.Namespace(**{**DEFAULTS, **q})
        if a.model == "vitb" and not os.path.exists(f"{CACHE}/vitb_verified"):
            verify("vitb"); open(f"{CACHE}/vitb_verified", "w").close()
        print(f"[{a.tag}] start", flush=True)
        train(a)
        later = [x for x in todo[i + 1:] if x.get("model") == "vitb" and x["subset"] == a.subset]
        if a.model == "vitb" and not later:   # ~74 GB per subset -> keep only the shared eval cache
            os.remove(f"{CACHE}/vitb_released/s{a.subset}_train.npy")
        torch.cuda.empty_cache()
    print("QUEUE DONE", flush=True)


def status():
    print(f"{'run':24s} {'epoch':>9s}  {'val acc':>7s} {'AUROC':>6s} {'F1':>5s}   {'best AUROC':>10s}")
    for q in QUEUE:
        r = rows(q["tag"])
        if not r:
            print(f"{q['tag']:24s} {'queued/caching':>14s}"); continue
        l, b = r[-1], max(r, key=lambda x: x[4])
        note = "" if has_epoch_preds(q["tag"]) else ("  <- running" if len(r) < 500 else
            "  (final preds only)" if os.path.exists(f"{OUT}/{q['tag']}/final_preds.npz") and q.get("model") == "vitb"
            else "  <- rerun queued")
        print(f"{q['tag']:24s} {len(r):>4d}/500   {l[2]:7.2f} {l[4]*100:6.1f} {l[8]*100:5.1f}   {b[4]*100:6.1f} @{int(b[0])}{note}")


def summary():
    print("targets frozen: ViT-M Acc 83.30 F1 82.4 AUROC 87.7 | ViT-B 81.20 / 81.0 / 87.9")
    groups = {"ViT-M frozen (s0-s4)": [f"c_vitm_frozen_s{s}" for s in range(5)],
              "ViT-M Fig-3 head (s0-s4)": [f"c_vitm_attentive_s{s}" for s in range(5)],
              "ViT-M random-init (s0-s2)": [f"c_vitm_rand_s{s}" for s in range(3)],
              "ViT-M random-init Fig-3 head (s0-s2)": [f"c_vitm_rand_attentive_s{s}" for s in range(3)],
              "ViT-B frozen (s0-s4)": [f"c_vitb_frozen_s{s}" for s in range(5)],
              "ViT-B Fig-3 head (s0-s4)": [f"c_vitb_attentive_s{s}" for s in range(5)],
              "test: ViT-M batch 256 (s0)": ["c_vitm_frozen_b256_s0"],
              "test: ViT-M fp16 cache (s0)": ["c_vitm_frozen_fp16_s0"]}
    for g, tags in groups.items():
        R = [np.array(rows(t)) for t in tags if len(rows(t)) >= 500]
        if not R:
            continue
        f = np.array([[r[-1, 2], r[-1, 8] * 100, r[-1, 4] * 100] for r in R])
        b = np.array([r[:, 4].max() * 100 for r in R])
        print(f"{g:28s} n={len(R)}  final: Acc {f[:,0].mean():.1f}±{f[:,0].std():.1f}  F1 {f[:,1].mean():.1f}±{f[:,1].std():.1f}"
              f"  AUROC {f[:,2].mean():.1f}±{f[:,2].std():.1f}   best-epoch AUROC* {b.mean():.1f}")
    print("* optimistic: epoch picked on the eval set")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true"); ap.add_argument("--queue", action="store_true")
    ap.add_argument("--only", choices=["vitm", "vitb"])
    ap.add_argument("--status", action="store_true"); ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tag"); ap.add_argument("--model", default="vitm")
    ap.add_argument("--subset", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="released"); ap.add_argument("--head", default="attention")
    ap.add_argument("--bs", type=int, default=2); ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    if a.verify: verify(a.model)
    elif a.queue: queue(a.only)
    elif a.status: status()
    elif a.summary: summary()
    else: train(a)
