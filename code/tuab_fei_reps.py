"""Act-0 Experiment 6 re-audit (2026-10-03): does FEI's zero-shot NMT->TUAB gain survive independent pre-training runs?
Scores the two seeded NMT-only FEI replicates (fei_rep{1,2}_enc_cv_s0_f{k}.pt from act0_nmt_pretrain_replicate.py)
exactly like tuab_feic.py's FEI arm (FEI@4000 mean-pooled embedding, StandardScaler + balanced LogReg, 5 fold encoders
ensembled), next to the original run (rep0 = tuab_feic emb_fei_f{k}) and the same 3 random-init twins;
also one bootstrap that resamples the pre-training runs, the three random-init seeds and the test recordings together
(probe only, from the saved embeddings: python -c "import tuab_fei_reps as R; R.probe(R.embed(), R.embed(R.EXTRA))").
Reps 3 and 4 (added 2026-10-03, for Experiment 6 only) are FEI-only NMT pre-training runs with the replicate script's
recipe and seeding (100*rep + fold); they are not part of Experiment 5. Each TUAB file is read once per rep pair and
embedded by its 10 encoders (51 GB of input per pass).

  python tuab_fei_reps.py      # pre-train reps 3-4 (GPU) -> embed (GPU, resumable every 500 files) -> probe (CPU)
"""
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

import fei_pretrain as F
import fei_branchC as C
import tuab_feic as TF
from act0_metrics import metrics, save
from head_to_head_cv import h20_args
from hw_guard import cool_gate

OUT = f"{TF.ROOT}/code/runs/act0/tuab_fei_reps"
EXTRA = (3, 4)


def pretrain(reps):   # FEI only, exactly as act0_nmt_pretrain_replicate.py trains its FEI encoders
    from act0_nmt_pretrain_replicate import FEI_ARGS, seed
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    for r in reps:
        for k, (tr, _) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
            fa = f"{TF.ROOT}/code/fei_rep{r}_enc_cv_s0_f{k}.pt"
            if not os.path.exists(fa):
                seed(100 * r + k)
                torch.save(F.pretrain([files[i] for i in tr], FEI_ARGS, tag=f"[rep{r}_f{k} FEI] ").state_dict(), fa + ".tmp")
                os.replace(fa + ".tmp", fa)


def embed(reps=(1, 2)):
    tag = "" if reps == (1, 2) else "".join(map(str, reps))
    path = f"{OUT}/emb_reps{tag}.npz"
    if os.path.exists(path):
        return np.load(path)["X"]
    args = h20_args(); C._configure(args)
    encs, pairs = [], [(r, k) for r in reps for k in range(5)]
    for r, k in pairs:
        a = F.Encoder(args.d).to(F.DEV)
        a.load_state_dict(torch.load(f"{TF.ROOT}/code/fei_rep{r}_enc_cv_s0_f{k}.pt", map_location=F.DEV)); a.eval()
        encs.append(a)
    fl = TF.files()
    part = f"{OUT}/emb_reps{tag}_partial.npz"
    X = np.load(part)["X"] if os.path.exists(part) else np.zeros((len(pairs), len(fl), args.d), np.float32)
    start = int(np.load(part)["done"]) if os.path.exists(part) else 0
    for i in range(start, len(fl)):
        if i % 10 == 0:
            cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=False)
        if i % 500 == 0 and i > start:
            np.savez(part, X=X, done=i); print(f"  embed {i}/{len(fl)}", flush=True)
        sig = F.load_continuous(fl[i][0]); T, L = sig.shape[1], C.L
        if T < L:
            sig = np.tile(sig, (1, L // T + 1))[:, :L]; T = L
        w = torch.from_numpy(np.stack([sig[:, s:s + L] for s in range(0, T - L + 1, L)])).to(F.DEV)
        with torch.no_grad():
            for j, a in enumerate(encs):
                X[j, i] = a(w).mean(0).cpu().numpy()
    np.savez(path, X=X)
    return X


def paired(y, pa, pb, n=2000, seed=0):   # same resamples for both models; p = share of resamples with delta <= 0
    rng, d = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(metrics(y[i], pa[i])["auroc"] - metrics(y[i], pb[i])["auroc"])
    d = np.array(d)
    return dict(delta=float(metrics(y, pa)["auroc"] - metrics(y, pb)["auroc"]),
                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))


def pooled(y, runs, pb, n=2000, seed=0):   # resample pre-training runs AND test recordings; p = share with delta <= 0.
    # pb 2-D = the random-init twin's member predictions: its seeds are resampled too and averaged like the twin ensemble
    # (third level, re-audit 2026-10-03); pb 1-D = one fixed comparator, the same draws as before that change
    pb = np.atleast_2d(pb)
    rng, d = np.random.RandomState(seed), []
    for b in range(n):
        if b % 200 == 0:
            cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=False)
        r, i = rng.randint(0, len(runs), len(runs)), rng.randint(0, len(y), len(y))
        s = rng.randint(0, len(pb), len(pb)) if len(pb) > 1 else [0]
        if 0 < y[i].sum() < len(i):
            d.append(np.mean([metrics(y[i], runs[k][i])["auroc"] for k in r]) - metrics(y[i], pb[s].mean(0)[i])["auroc"])
    d = np.array(d)
    return dict(delta=float(np.mean([metrics(y, q)["auroc"] for q in runs]) - metrics(y, pb.mean(0))["auroc"]),
                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))


def probe(X, X34):
    fl = TF.files()
    y = np.array([l for _, l, _ in fl]); split = np.array([s for _, _, s in fl])
    ev, tr = np.where(split == "eval")[0], np.where(split == "train")[0]
    old = f"{TF.ROOT}/code/runs/act0/tuab_feic"
    E = {0: [np.load(f"{old}/emb_fei_f{k}.npy") for k in range(5)],
         1: list(X[:5]), 2: list(X[5:]), 3: list(X34[:5]), 4: list(X34[5:]),
         "rand": [np.load(f"{old}/emb_fei_rand{s}.npy") for s in range(3)]}
    runs = [r for r in E if r != "rand"]
    P = {}
    for key, embs in E.items():
        cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=False)
        P[key] = np.array([TF.proba(e[tr], y[tr], e[ev]) for e in embs])
    ye = y[ev]
    res = dict(rand=dict(ensemble_auroc=metrics(ye, P["rand"].mean(0))["auroc"],
                         member_auroc=[metrics(ye, q)["auroc"] for q in P["rand"]]))
    for r in runs:
        res[f"rep{r}"] = dict(ensemble_auroc=metrics(ye, P[r].mean(0))["auroc"],
                              member_auroc=[metrics(ye, q)["auroc"] for q in P[r]],
                              vs_rand=paired(ye, P[r].mean(0), P["rand"].mean(0)))
        print(f"rep{r}: FEI {res[f'rep{r}']['ensemble_auroc']:.3f}  rand {res['rand']['ensemble_auroc']:.3f}  "
              f"delta {res[f'rep{r}']['vs_rand']['delta']:+.3f} CI {np.round(res[f'rep{r}']['vs_rand']['ci'], 3)} "
              f"p {res[f'rep{r}']['vs_rand']['p_le0']:.3f}  members {np.round(res[f'rep{r}']['member_auroc'], 3)}",
              flush=True)
    m = np.array([a for r in runs for a in res[f"rep{r}"]["member_auroc"]])
    res["members_above_rand_ensemble"] = f"{int((m > res['rand']['ensemble_auroc']).sum())}/{len(m)}"
    res["members_above_best_rand_member"] = f"{int((m > max(res['rand']['member_auroc'])).sum())}/{len(m)}"
    print(res["members_above_rand_ensemble"], res["members_above_best_rand_member"], flush=True)
    res["collapse"] = {}   # same collapse check as tuab_feic_joint_reps.py: per fold encoder, on the 276 eval recordings
    for r in runs:
        for k, e in enumerate(E[r]):
            X = e[ev]; Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            res["collapse"][f"rep{r}_f{k}"] = dict(rec_std=float(X.std(0).mean()),
                                                   rec_cos=float((Xn @ Xn.T)[np.triu_indices(len(X), 1)].mean()))
    res["pooled_over_runs"] = pooled(ye, [P[r].mean(0) for r in runs], P["rand"]) | dict(n_runs=len(runs), n_rand_seeds=len(P["rand"]))
    print("pooled over runs and recordings:", res["pooled_over_runs"], flush=True)
    save(f"{OUT}/results.json", res)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pretrain(EXTRA)
    probe(embed(), embed(EXTRA))
