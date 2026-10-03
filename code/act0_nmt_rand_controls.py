"""Act-0 Experiment 5 random-init controls: rand-FEI@4000, rand-GTJ and rand-FEI+GTJ on NMT,
the missing twins of feic_nmt_anchor.py's FEI@4000 / C / FEI+C. Same files, same seed-0 5-fold split, same embedding
(F.embed_set at L=4000, C.embed_dyn h20), same probe (feic_nmt_anchor.proba), same metrics. Random encoders need no
training, so this is forward passes only: seeds 0-4, built exactly like tuab_feic.py / stroke_phaseB_clean.py
(torch.manual_seed(s); F.Encoder, then C encoder). Resumable: one embedding file per seed.
  HW_THREADS=2 python act0_nmt_rand_controls.py   -> code/runs/act0/a0_feic_nmt_rand/{emb_rand*.npz, results.json}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feic_nmt_anchor import CODE, proba   # cap_threads(4) before numpy/torch
import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold

import fei_pretrain as F
import fei_branchC as C
from act0_metrics import metrics, save
from head_to_head_cv import h20_args
from hw_guard import cool_gate

OUT = f"{CODE}/runs/act0/a0_feic_nmt_rand"
SEEDS = range(5)
cool = lambda: cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=True)
ARMS = {"FEI@4000": lambda e: e["fei"], "C": lambda e: e["c"], "FEI+C": lambda e: np.hstack([e["fei"], e["c"]])}


def main():
    os.makedirs(OUT, exist_ok=True)
    args = h20_args(); C._configure(args)
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    E = {}
    for s in SEEDS:
        path = f"{OUT}/emb_rand{s}.npz"
        if not os.path.exists(path):
            torch.manual_seed(s)
            a, c = F.Encoder(args.d).to(F.DEV).eval(), C._enc(args)(args.d).to(F.DEV).eval()
            fei, gtj = [], []
            for i in range(0, len(files), 10):   # guard every 10 recordings: embed_set's own check (every 100) let the CPU hit 97C
                cool()
                fei.append(F.embed_set(a, files[i:i + 10], C.L)[0]); gtj.append(C.embed_dyn(c, files[i:i + 10])[0])
            np.savez(path, rec=np.array([p for p, _ in files]), y=y, fei=np.vstack(fei), c=np.vstack(gtj))
            print(f"rand{s}: embedded {len(files)} recordings", flush=True)
        E[s] = dict(np.load(path))
        assert (E[s]["y"] == y).all()
    pre = json.load(open(f"{CODE}/runs/act0/a0_feic_nmt/results.json"))
    res = {"seeds": list(SEEDS), "folds": {}}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    for k, (tr, te) in enumerate(folds):
        res["folds"][str(k)] = {}
        for m, f in ARMS.items():
            res["folds"][str(k)][f"rand-{m}"] = {}
            for s in SEEDS:
                cool(); res["folds"][str(k)][f"rand-{m}"][f"rand{s}"] = metrics(y[te], proba(f(E[s])[tr], y[tr], f(E[s])[te]))
    summ = {}
    for m in ARMS:
        A = np.array([[res["folds"][str(k)][f"rand-{m}"][f"rand{s}"]["auroc"] for s in SEEDS] for k in range(5)])   # (fold, seed)
        P = np.array([pre[str(k)][m]["auroc"] for k in range(5)])
        R = A.mean(1)   # per fold, mean over random seeds
        summ[f"rand-{m}"] = dict(auroc_mean=float(A.mean()), auroc_sd_over_folds=float(R.std()),
                                 auroc_per_seed=[float(v) for v in A.mean(0)], auroc_per_fold=[float(v) for v in R])
        summ[f"{m} vs rand-{m}"] = dict(pretrained=float(P.mean()), random=float(R.mean()), mean_diff=float((P - R).mean()),
                                        folds_pretrained_better=int((P > R).sum()), ttest_rel_p=float(ttest_rel(P, R).pvalue),
                                        seed_fold_cells_pretrained_better=int((P[:, None] > A).sum()), cells=int(A.size))
    for m in ARMS:   # full metric set for the seed-mean random arm (mean over folds of the per-seed mean)
        summ[f"rand-{m}"] |= {q: float(np.mean([[res["folds"][str(k)][f"rand-{m}"][f"rand{s}"][q] for s in SEEDS] for k in range(5)]))
                              for q in ("auc_pr", "bal_acc", "acc", "f1", "sens", "spec")}
    res["summary"] = summ
    save(f"{OUT}/results.json", res)
    for m in ARMS:
        d = summ[f"{m} vs rand-{m}"]
        print(f"{m:9s} pretrained {d['pretrained']:.3f}  rand {d['random']:.3f} ± {summ[f'rand-{m}']['auroc_sd_over_folds']:.3f} "
              f"(seeds {np.round(summ[f'rand-{m}']['auroc_per_seed'], 3)})  Δ {d['mean_diff']:+.3f}  "
              f"{d['folds_pretrained_better']}/5 folds  p={d['ttest_rel_p']:.4f}", flush=True)


if __name__ == "__main__":
    main()
