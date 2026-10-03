"""Act-0 Experiment 5 pre-training replicates. The FEI and GTJ encoders behind Table 11 come
from one unseeded pre-training run per fold. This repeats that pre-training (fei_pretrain.pretrain with its defaults =
plain FEI; fei_branchC.pretrain_C with the h20 config) on the same seed-0 training folds, now seeded
(torch / numpy / random = 100*rep + fold; cuDNN kernels stay non-deterministic), and scores FEI@4000, GTJ and FEI+GTJ
exactly as feic_nmt_anchor.py does. Resumable per (rep, fold).
  HW_THREADS=2 python act0_nmt_pretrain_replicate.py [--reps 1 2]
    -> code/runs/act0/a0_feic_nmt_rep/{results.json, preds_rep*_f*.npz}, code/{fei,branchC_h20}_rep*_enc_cv_s0_f*.pt
"""
import json
import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feic_nmt_anchor import CODE, proba   # cap_threads before numpy/torch
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import fei_pretrain as F
import fei_branchC as C
from act0_metrics import metrics, save
from head_to_head_cv import h20_args
from hw_guard import cool_gate

OUT = f"{CODE}/runs/act0/a0_feic_nmt_rep"
COMMON = dict(epochs=60, batch=128, wpe=4, lr=2e-4, ema=0.996)   # both scripts' argparse defaults
FEI_ARGS = SimpleNamespace(**COMMON, L=1000, d=256, h=128, mask_bias=0.0, mask_ema=0.98)
GTJ_ARGS = SimpleNamespace(**vars(h20_args()), **COMMON)


def emb(fn, enc, fl):   # guard every 10 recordings: embed_set's own check (every 100) once let the CPU reach 97C
    out = []
    for i in range(0, len(fl), 10):
        cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=True); out.append(fn(enc, fl[i:i + 10])[0])
    return np.vstack(out)


def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def main():
    reps = [int(r) for r in sys.argv[sys.argv.index("--reps") + 1:]] if "--reps" in sys.argv else [1, 2]
    os.makedirs(OUT, exist_ok=True)
    C._configure(GTJ_ARGS)
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    res = json.load(open(f"{OUT}/results.json")) if os.path.exists(f"{OUT}/results.json") else {}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    for r in reps:
        for k, (tr, te) in enumerate(folds):
            key = f"rep{r}_f{k}"
            if key in res:
                continue
            trf, tef = [files[i] for i in tr], [files[i] for i in te]
            fa, fc = f"{CODE}/fei_rep{r}_enc_cv_s0_f{k}.pt", f"{CODE}/branchC_h20_rep{r}_enc_cv_s0_f{k}.pt"
            if not os.path.exists(fa):
                seed(100 * r + k); torch.save(F.pretrain(trf, FEI_ARGS, tag=f"[{key} FEI] ").state_dict(), fa)
            if not os.path.exists(fc):
                seed(100 * r + k); torch.save(C.pretrain_C(trf, GTJ_ARGS, tag=f"[{key} GTJ] ").state_dict(), fc)
            a = F.Encoder(FEI_ARGS.d).to(F.DEV); a.load_state_dict(torch.load(fa, map_location=F.DEV)); a.eval()
            c = C._enc(GTJ_ARGS)(GTJ_ARGS.d).to(F.DEV); c.load_state_dict(torch.load(fc, map_location=F.DEV)); c.eval()
            fe = lambda e, fl: F.embed_set(e, fl, C.L)
            A, G = (emb(fe, a, trf), emb(fe, a, tef)), (emb(C.embed_dyn, c, trf), emb(C.embed_dyn, c, tef))
            P = {"FEI@4000": proba(A[0], y[tr], A[1]), "C": proba(G[0], y[tr], G[1]),
                 "FEI+C": proba(np.hstack([A[0], G[0]]), y[tr], np.hstack([A[1], G[1]]))}
            np.savez(f"{OUT}/preds_{key}.npz", rec=np.array([p for p, _ in tef]), label=y[te], **P)
            res[key] = {m: metrics(y[te], p) for m, p in P.items()}
            save(f"{OUT}/results.json", res)
            print(f"[{key}] AUROC FEI@4000 {res[key]['FEI@4000']['auroc']:.3f}  GTJ {res[key]['C']['auroc']:.3f}  "
                  f"FEI+GTJ {res[key]['FEI+C']['auroc']:.3f}", flush=True)
    for r in reps:
        print(f"rep{r}: " + "  ".join(f"{m} {np.mean([res[f'rep{r}_f{k}'][m]['auroc'] for k in range(5)]):.3f}"
                                      for m in ("FEI@4000", "C", "FEI+C")), flush=True)


if __name__ == "__main__":
    main()
