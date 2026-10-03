"""Act-0 Experiment 6 re-runs of the adapted encoders. Table 12's adapted rows (FEI and GTJ pre-trained
on NMT training fold k + TUAB's 2,717 unlabelled training recordings, feic_joint.py) rest on one pre-training run (torch seed
= fold). This repeats that pre-training twice with feic_joint's data and recipe, seeded like act0_nmt_pretrain_replicate.py
(torch / numpy / random = 100*rep + fold; cuDNN stays non-deterministic), then scores FEI, GTJ and FEI+GTJ on TUAB exactly
like tuab_feic.py's full_train probe (5 fold encoders ensembled) against the same random-init twins and the released ViT-M
linear probe; rep0 = the original joint run (tuab_feic_joint embeddings), re-scored here as a consistency check.
Resumable per checkpoint and per embedding file; results.json is rewritten after each finished replicate.
Probe only, from the saved embeddings: python -c "import tuab_feic_joint_reps as J; J.probe((1, 2))".

  python tuab_feic_joint_reps.py   -> code/{fei,branchC_h20}_joint_rep{1,2}_enc_cv_s0_f{k}.pt,
                                      code/runs/act0/tuab_feic_joint_reps/{emb_*.npy, results.json, collapse.json}
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feic_joint import CODE, recipe, save_pt, tuab_train   # cap_threads(4) before numpy/torch
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import fei_pretrain as F
import fei_branchC as C
import tuab_feic as TF
from act0_metrics import metrics, save
from head_to_head_cv import h20_args
from hw_guard import cool_gate
from tuab_fei_reps import paired, pooled

OUT = f"{CODE}/runs/act0/tuab_feic_joint_reps"
OLD = f"{CODE}/runs/act0/tuab_feic_joint"
REPS = (1, 2)
ARMS = ("FEI", "C", "FEI+C")


def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def ckpts(r, k):
    return f"{CODE}/fei_joint_rep{r}_enc_cv_s0_f{k}.pt", f"{CODE}/branchC_h20_joint_rep{r}_enc_cv_s0_f{k}.pt"


def pretrain(r):
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    for k, (tr, _) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
        pa, pc = ckpts(r, k)
        data = [files[i] for i in tr] + tuab_train()
        if not os.path.exists(pa):
            seed(100 * r + k); save_pt(F.pretrain(data, recipe(L=1000), tag=f"[rep{r} f{k} FEI] ").state_dict(), pa)
        if not os.path.exists(pc):
            seed(100 * r + k); h = h20_args(); C._configure(h)
            save_pt(C.pretrain_C(data, recipe(**vars(h)), tag=f"[rep{r} f{k} GTJ] ").state_dict(), pc)


def emb(fn, enc, fl):   # guard every 10 recordings (embed_set's own check every 100 once let the CPU reach 97C)
    out = []
    for i in range(0, len(fl), 10):
        cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=False); out.append(fn(enc, fl[i:i + 10])[0])
    return np.vstack(out)


def embed(r):
    args = h20_args(); C._configure(args)
    fl = [(p, y) for p, y, _ in TF.files()]
    for k in range(5):
        pa, pc = ckpts(r, k)
        for name, path, make, fn in (("fei", pa, lambda: F.Encoder(args.d), lambda e, f: F.embed_set(e, f, C.L)),
                                     ("c", pc, lambda: C._enc(args)(args.d), C.embed_dyn)):
            out = f"{OUT}/emb_{name}_rep{r}_f{k}.npy"
            if os.path.exists(out):
                continue
            enc = make().to(F.DEV); enc.load_state_dict(torch.load(path, map_location=F.DEV)); enc.eval()
            print(f"  embed {name} rep{r} f{k}", flush=True)
            np.save(out + ".tmp.npy", emb(fn, enc, fl)); os.replace(out + ".tmp.npy", out)


def load(r, name, k):
    return np.load(f"{OLD}/emb_{name}_{k}.npy" if r in (0, "rand") else f"{OUT}/emb_{name}_rep{r}_{k}.npy")


def probe(reps):
    fl = TF.files()
    y = np.array([l for _, l, _ in fl]); split = np.array([s for _, _, s in fl])
    ev, tr = np.where(split == "eval")[0], np.where(split == "train")[0]
    ye, rec = y[ev], np.array([os.path.basename(p) for p, _, _ in fl])[ev]
    v = np.load(f"{CODE}/runs/act0/tuab_vitm_linear_joint/preds_released_full_train.npz")
    o = {q: i for i, q in enumerate(v["rec"])}
    vit = v["prob_abnormal"][[o[q] for q in rec]]; assert (v["label"][[o[q] for q in rec]] == ye).all()
    runs = [0] + list(reps)
    keys = {"rand": [f"rand{s}" for s in range(3)]} | {r: [f"f{k}" for k in range(5)] for r in runs}
    feat = lambda r, arm, k: {"FEI": lambda: load(r, "fei", k), "C": lambda: load(r, "c", k),
                              "FEI+C": lambda: np.hstack([load(r, "fei", k), load(r, "c", k)])}[arm]()
    P, res, col = {}, {"vit_m_auroc": metrics(ye, vit)["auroc"]}, {}
    for r, ks in keys.items():
        for arm in ARMS:
            P[r, arm] = np.array([(cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=False),
                                   TF.proba(feat(r, arm, k)[tr], y[tr], feat(r, arm, k)[ev]))[1] for k in ks])
        if r not in (0, "rand"):
            for name in ("fei", "c"):
                for k in ks:
                    X = load(r, name, k)[ev]; Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
                    col[f"{name}_rep{r}_{k}"] = dict(rec_std=float(X.std(0).mean()),
                                                    rec_cos=float((Xn @ Xn.T)[np.triu_indices(len(X), 1)].mean()))
    for arm in ARMS:
        a = res[arm] = dict(rand=dict(ensemble_auroc=metrics(ye, P["rand", arm].mean(0))["auroc"],
                                      member_auroc=[metrics(ye, q)["auroc"] for q in P["rand", arm]]))
        for r in runs:
            p = P[r, arm].mean(0)
            a[f"rep{r}"] = metrics(ye, p) | dict(member_auroc=[metrics(ye, q)["auroc"] for q in P[r, arm]],
                                                vs_rand=paired(ye, p, P["rand", arm].mean(0)), vs_vit_m=paired(ye, p, vit))
            print(f"{arm:6s} rep{r}: AUROC {a[f'rep{r}']['auroc']:.3f}  rand {a['rand']['ensemble_auroc']:.3f}  "
                  f"vs rand {a[f'rep{r}']['vs_rand']['delta']:+.3f} (p {a[f'rep{r}']['vs_rand']['p_le0']:.3f})  "
                  f"vs ViT-M {a[f'rep{r}']['vs_vit_m']['delta']:+.3f} (p {a[f'rep{r}']['vs_vit_m']['p_le0']:.3f})",
                  flush=True)
        if len(runs) > 1:
            ens = [P[r, arm].mean(0) for r in runs]
            a["pooled_vs_rand"] = pooled(ye, ens, P["rand", arm]) | dict(n_rand_seeds=len(P["rand", arm]))   # seeds resampled too
            a["pooled_vs_vit_m"] = pooled(ye, ens, vit)
            print(f"{arm:6s} pooled over {len(runs)} runs: vs rand {a['pooled_vs_rand']}  vs ViT-M {a['pooled_vs_vit_m']}",
                  flush=True)
    save(f"{OUT}/collapse.json", col)
    save(f"{OUT}/results.json", res | dict(runs=runs))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    done = []
    for r in REPS:
        pretrain(r); embed(r); done.append(r)
        probe(done)
    print("TUAB FEIC JOINT REPS DONE", flush=True)
