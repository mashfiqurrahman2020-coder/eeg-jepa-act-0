"""Act-0: like-for-like LINEAR probe on frozen ViT-M features (same probe as FEI+C in tuab_feic.py), so the
FEI+C vs ViT-M gap isn't confounded by the head (P1.4 used the paper's attentive heads).
Feature per recording = token mean ++ token std (768), averaged over ALL clip positions (deterministic; same
definition as stroke Phase B). Tokens come from the exact P1.4 caches (data/TUAB_cache/vitm_*), so no
re-encoding except released full-train (GPU, never cached). Probe = StandardScaler + LogReg(C=1, balanced).
Runs: released x {s0..s4, full_train}; random-init seed s x subset s (s = 0..2, as in P1.4). Test = eval (276).
Then paired vs FEI+C (tuab_feic preds, same eval recordings): delta AUROC per train set, bootstrap on full_train.
  python tuab_vitm_linear.py   -> code/runs/act0/tuab_vitm_linear/{results.json, preds_*.npz, feats_*.npz}
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuab_cached_eval as T   # applies cap_threads(4); upstream dataset = the caches' exact order
from hw_guard import cool_gate
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from act0_metrics import bootstrap_ci, fmt, metrics, save

ROOT = "/home/mashfiq/eeg_vjepa"
TAG = os.environ.get("FEIC_TAG", "")   # "_joint" -> NMT+TUAB-train pretrained FEI/C encoders, separate outputs
OUT = f"{ROOT}/code/runs/act0/tuab_vitm_linear{TAG}"
FEIC = f"{ROOT}/code/runs/act0/tuab_feic{TAG}"
M = T.MODELS["vitm"]


def pool(tok):   # (P, tokens, D) -> (2D,)
    return np.concatenate([tok.mean(1), tok.std(1)], -1).mean(0)


def from_cache(ck, name, subset, split):
    """Pooled features + abnormal=1 labels + basenames for a cached split; also token std (collapse)."""
    path = f"{OUT}/feats_{ck}_{name}.npz"
    if not os.path.exists(path):
        X = np.load(f"{T.CACHE}/vitm_{ck}/{name}.npy", mmap_mode="r")
        ds = T.dataset(subset, split, M)
        F, ts = [], []
        for i in range(len(X)):
            if i % 50 == 0:
                cool_gate(pause=88.0, resume=78.0, abort=92.0)
            t = np.asarray(X[i]); F.append(pool(t)); ts.append(float(t.std(1).mean()))
        np.savez(path, X=np.array(F), y=1 - np.array(ds.labels), rec=np.array([os.path.basename(f) for f in ds.samples]),
                 token_std=np.array(ts))
    d = np.load(path)
    return d["X"], d["y"], list(d["rec"]), d["token_std"]


def full_train():
    path = f"{OUT}/feats_released_full_train.npz"
    if not os.path.exists(path):
        files = sorted(glob.glob(f"{ROOT}/data/TUAB_preprocessed_100hz/train/*/*.npy"))
        enc, clip_len, F = T.encoder(M, M["ckpt"], 0), M["frames_per_clip"] * M["frame_step"], []
        with torch.no_grad():
            for i, f in enumerate(files):
                if i % 25 == 0:
                    cool_gate(pause=88.0, resume=78.0, abort=92.0)
                    print(f"  encode full_train {i}/{len(files)}", flush=True)
                fr = np.load(f)
                x = torch.from_numpy(np.stack([T.clip(fr, e, M) for e in range(clip_len, len(fr))]))[:, None].to(T.DEV)
                F.append(pool(enc(x).float().cpu().numpy()))
        np.savez(path, X=np.array(F), y=np.array([int("/abnormal/" in f) for f in files]),
                 rec=np.array([os.path.basename(f) for f in files]))
    d = np.load(path)
    return d["X"], d["y"], list(d["rec"])


def proba(Xtr, ytr, Xte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def paired_boot(y, pa, pb, n=2000, seed=0):
    rng, d = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(metrics(y[i], pa[i])["auroc"] - metrics(y[i], pb[i])["auroc"])
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float((np.array(d) <= 0).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = [("released", s, s) for s in range(5)] + [(f"rand{s}", s, s) for s in range(3)]
    res, col, P = {}, {}, {}
    for ck, subset, _ in runs:
        Xtr, ytr, _, _ = from_cache(ck, f"s{subset}_train", subset, "train")
        Xev, yev, rev, ts = from_cache(ck, "eval", subset, "eval")
        tag = f"{'rand' if ck != 'released' else 'released'}_s{subset}"
        p = proba(Xtr, ytr, Xev); P[tag] = (p, yev, rev)
        res[tag] = metrics(yev, p) | {"n_train": len(ytr)}
        Xn = Xev / np.linalg.norm(Xev, axis=1, keepdims=True)
        col[ck] = dict(token_std=float(ts.mean()), rec_std=float(Xev.std(0).mean()),
                       rec_cos=float((Xn @ Xn.T)[np.triu_indices(len(Xev), 1)].mean()))
        np.savez(f"{OUT}/preds_{tag}.npz", prob_abnormal=p, label=yev, rec=np.array(rev))
        print(f"{tag:12s} {fmt(res[tag])}", flush=True)
    Xtr, ytr, _ = full_train()
    Xev, yev, rev, _ = from_cache("released", "eval", 0, "eval")
    p = proba(Xtr, ytr, Xev); P["released_full_train"] = (p, yev, rev)
    res["released_full_train"] = metrics(yev, p) | dict(n_train=len(ytr), auroc_ci=bootstrap_ci(yev, p, "auroc"),
                                                         bal_acc_ci=bootstrap_ci(yev, p, "bal_acc"))
    np.savez(f"{OUT}/preds_released_full_train.npz", prob_abnormal=p, label=yev, rec=np.array(rev))
    print(f"{'released_full':12s} {fmt(res['released_full_train'])}", flush=True)
    for kind, tags in (("released", [f"released_s{s}" for s in range(5)]), ("rand", [f"rand_s{s}" for s in range(3)])):
        a = np.array([res[t]["auroc"] for t in tags]); b = np.array([res[t]["bal_acc"] for t in tags])
        res[f"{kind}_subsets_mean"] = dict(auroc=[float(a.mean()), float(a.std())], bal_acc=[float(b.mean()), float(b.std())])
        print(f"{kind} subsets AUROC {a.mean():.3f}±{a.std():.3f}  BAcc {b.mean():.3f}", flush=True)
    # paired vs FEI+C (same probe, same eval recordings, matched by basename)
    paired = {}
    for tname in [f"s{s}" for s in range(5)] + ["full_train"]:
        f = f"{FEIC}/preds_{tname}_FEI+C.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f); o = {r: i for i, r in enumerate(d["rec"])}
        pv, yv, rv = P[f"released_{tname}"]
        pf = d["prob_abnormal"][[o[r] for r in rv]]
        assert (d["label"][[o[r] for r in rv]] == yv).all()
        paired[f"FEI+C - ViT-M ({tname})"] = dict(delta=float(metrics(yv, pf)["auroc"] - metrics(yv, pv)["auroc"]))
        if tname == "full_train":
            paired[f"FEI+C - ViT-M ({tname})"] |= dict(zip(("ci", "p_le0"), paired_boot(yv, pf, pv)))
    if paired:
        dl = [v["delta"] for k, v in paired.items() if "full" not in k]
        paired["subsets_mean_delta"] = [float(np.mean(dl)), float(np.std(dl))]
        print("paired FEI+C - ViT-M(linear):", paired, flush=True)
    save(f"{OUT}/results.json", dict(runs=res, collapse=col, paired_vs_feic=paired,
                                     feature="token mean++std, mean over all clip positions"))
    print("TUAB ViT-M LINEAR DONE", flush=True)


if __name__ == "__main__":
    main()
