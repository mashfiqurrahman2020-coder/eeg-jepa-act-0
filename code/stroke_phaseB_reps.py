"""Act-0 Experiment 8 with Experiment 5's two seeded pre-training re-runs. Experiment 8 (stroke, post-hoc
spectrally matched input, stroke_phaseB_clean.py) uses the original fold encoders only. This embeds the same 15-subject
primary cohort input with the re-run encoders (fei_rep{1,2} / branchC_h20_rep{1,2}, act0_nmt_pretrain_replicate.py) and
scores FEI+GTJ, FEI and GTJ exactly as stroke_phaseB_clean.run_loso does (5 fold members ensembled, StandardScaler +
LogReg, LOSO, 2000 bootstrap CIs, 100x stratified 5-fold, leave-pair-out) against the same random-init twins and BSI.
No permutation tests (the slow part; the original run's p stays the reference). rep0 = the original encoders, re-scored
from the saved embeddings as a consistency check. Pooled bootstrap over runs and subjects as in tuab_fei_reps.py.

  CUDA_VISIBLE_DEVICES="" python stroke_phaseB_reps.py   -> code/runs/act0/phaseB_reps/{emb_primary.npz, results.json}
"""
import glob
import json
import os

import stroke_phaseB_clean as K   # cap_threads(4) before numpy
import numpy as np

B, V = K.B, K.V
V.GATE = K.cool = lambda: B.cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=False)   # stricter than the originals' 88/78
OUT = f"{B.ROOT}/code/runs/act0/phaseB_reps"
REPS = (1, 2)


def embed():
    path = f"{OUT}/emb_primary.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    import torch
    import fei_pretrain as F
    import fei_branchC as C
    from head_to_head_cv import h20_args
    assert F.DEV == "cpu", "run with CUDA_VISIBLE_DEVICES='' (CPU only)"
    subs, _ = B.COHORTS["primary"]
    args = h20_args(); C._configure(args)
    files = [(f"{K.PREP}/primary/feic/{s}.npy", 0) for s in subs]
    ref = sorted(glob.glob(f"{K.NMT}/eval/*/*.npy")); ref = [(ref[i], 0) for i in np.random.RandomState(0).choice(len(ref), 15, replace=False)]
    E = {}

    def bnz(c, fl):   # stroke_phaseB_clean's BatchNorm gate: channel-averaged input z vs the checkpoint's NMT running stats
        rm, rv = c.bn.running_mean.numpy(), c.bn.running_var.numpy()
        z = []
        for p, _ in fl:
            sig = F.load_continuous(p); w = np.stack([sig[:, i:i + C.L] for i in range(0, sig.shape[1] - C.L + 1, C.L)])
            z.append(((C.spectrogram(torch.from_numpy(w)).numpy() - rm) / np.sqrt(rv + 1e-5)).mean((0, 1)))
        return np.mean(z, 0).reshape(19, C.NF).mean(0)

    for r in REPS:
        for k in range(5):
            K.cool()
            a, c = F.Encoder(args.d), C._enc(args)(args.d)
            a.load_state_dict(torch.load(f"{B.ROOT}/code/fei_rep{r}_enc_cv_s0_f{k}.pt", map_location="cpu"))
            c.load_state_dict(torch.load(f"{B.ROOT}/code/branchC_h20_rep{r}_enc_cv_s0_f{k}.pt", map_location="cpu"))
            a.eval(); c.eval()
            E[f"fei_rep{r}_f{k}"], E[f"c_rep{r}_f{k}"] = F.embed_set(a, files, C.L)[0], C.embed_dyn(c, files)[0]
            E[f"nmt_c_rep{r}_f{k}"] = C.embed_dyn(c, ref)[0]   # same 15 NMT reference recordings as stroke_phaseB_clean
            with torch.no_grad():
                E[f"bnz_c_rep{r}_f{k}"] = bnz(c, files)
            print(f"  embed rep{r} f{k}", flush=True)
    np.savez(path, **E)
    return E


def collapse(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return dict(rec_std=float(X.std(0).mean()), rec_cos=float((Xn @ Xn.T)[np.triu_indices(len(X), 1)].mean()),
                const_frac=float((X.std(0) < 1e-5).mean()))


def main():
    from tuab_fei_reps import pooled
    os.makedirs(OUT, exist_ok=True)
    E0, E = dict(np.load(f"{K.OUT}/emb_primary.npz")), embed()
    names, y = list(E0["names"]), E0["y"]
    pr = json.load(open(f"{B.OUT}/results_primary.json")); assert pr["preds"]["subject"] == names
    bsi = np.array(pr["preds"]["BSI"])
    emb = lambda r, br, k: E0[f"{br}_rand{k}"] if r == "rand" else E0[f"{br}_f{k}"] if r == 0 else E[f"{br}_rep{r}_f{k}"]
    feats = {"FEI+C": lambda r, k: np.hstack([emb(r, "fei", k), emb(r, "c", k)]),
             "FEI": lambda r, k: emb(r, "fei", k), "C": lambda r, k: emb(r, "c", k)}
    n0, B.N_PERM = B.N_PERM, 0
    col = {k: collapse(v) for k, v in E.items() if not k.startswith(("nmt_", "bnz_"))}
    for k in col:   # stroke_phaseB_clean's gates: GTJ spread on stroke / on NMT >= 0.5, BatchNorm max|z| <= 1.5
        if k.startswith("c_"):
            col[k]["std_ratio_vs_nmt"] = col[k]["rec_std"] / collapse(E[f"nmt_{k}"])["rec_std"]
            col[k]["bnz_max"] = float(np.abs(E[f"bnz_{k}"]).max())
    res, P = {"collapse": col}, {}
    for arm, f in feats.items():
        rmem = [f("rand", s) for s in range(3)]
        P["rand", arm] = B.arm(rmem, y, np.random.RandomState(0))[0]
        kr = V.kf5_aurocs(rmem, y, V.R)   # same seeds -> same partitions as each run's kf5, so the win rate is paired
        a = res[arm] = dict(rand_auroc=B.metrics(y, P["rand", arm])["auroc"], rand_kf5_mean=float(kr.mean()))
        for r in (0,) + REPS:
            K.cool(); mem = [f(r, k) for k in range(5)]
            P[r, arm], _, _, m = B.arm(mem, y, np.random.RandomState(0)); m.pop("perm_p")
            kf = V.kf5_aurocs(mem, y, V.R)
            a[f"rep{r}"] = m | dict(kf5_mean=float(kf.mean()), kf5_sd=float(kf.std()), lpo=float(V.lpo(mem, y).mean()),
                                    kf5_frac_better_than_rand=float((kf > kr).mean()),
                                    vs_rand=B.paired(y, P[r, arm], P["rand", arm]), vs_bsi=B.paired(y, P[r, arm], bsi))
            print(f"{arm:6s} rep{r}: LOSO AUROC {m['auroc']:.3f}  kf5 {kf.mean():.3f}  rand {a['rand_auroc']:.3f}  "
                  f"vs rand {a[f'rep{r}']['vs_rand']['delta']:+.3f} (kf5 wins {(kf > kr).mean():.0%})  vs BSI {a[f'rep{r}']['vs_bsi']['delta']:+.3f}",
                  flush=True)
        ens = [P[r, arm] for r in (0,) + REPS]
        a["pooled_vs_rand"], a["pooled_vs_bsi"] = pooled(y, ens, P["rand", arm]), pooled(y, ens, bsi)
        print(f"{arm:6s} pooled over runs: vs rand {a['pooled_vs_rand']}  vs BSI {a['pooled_vs_bsi']}", flush=True)
    B.N_PERM = n0
    B.save(f"{OUT}/results.json", res | dict(bsi_auroc=B.metrics(y, bsi)["auroc"], n=len(y), n_stroke=int(y.sum())))
    print("STROKE REPS DONE", flush=True)


if __name__ == "__main__":
    main()
