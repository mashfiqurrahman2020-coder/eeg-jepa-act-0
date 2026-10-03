"""Phase B clean prep, fold-local sensitivity analysis (post-hoc).
stroke_phaseB_clean.py fits its one spectral-matching filter on all 15 subjects pooled (label-free but transductive:
every held-out subject shapes the filter applied to it). Here the same filter (same 4-iteration log-PSD match onto the
same NMT reference) is re-fitted inside every LOSO fold on the 14 training subjects only, then applied unchanged to
all 15, so the held-out subject contributes to nothing it is scored with. Everything else is stroke_phaseB_clean.py:
same base prep, encoders (5 NMT-only FEI+GTJ members, 3 random-init), probe, ensemble, 1000-perm p, 2000-bootstrap CIs.
Check: the filter fitted on all 15 must equal runs/act0/phaseB_clean/filter_primary.npz.
  CUDA_VISIBLE_DEVICES="" HW_THREADS=2 python stroke_phaseB_foldlocal.py   -> code/runs/act0/phaseB_foldlocal/*
"""
import json
import os
import sys

import stroke_phaseB as B   # cap_threads(4) before numpy
import stroke_phaseB_clean as K
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = f"{B.ROOT}/code/runs/act0/phaseB_foldlocal"
COHORT = "primary"
MEMBERS = [f"f{k}" for k in range(5)] + [f"rand{s}" for s in range(3)]
PERM_ARMS = ["FEI+C", "rand-FEI+C"]
cool = lambda: B.cool_gate(pause=80.0, resume=70.0, abort=92.0, verbose=True)   # stricter than stroke_phaseB_clean's 88


def fit_filter(X):   # stroke_phaseB_clean.prep's matching loop, on the given subjects only
    R = K.nmt_ref(); f, ref = R["f"], R["mean"]
    logh = np.zeros_like(ref)
    for _ in range(K.N_ITER):
        lp = np.stack([K.logpsd(K.zs(K.shape(x, f, logh)))[1] for x in X])
        logh += (ref - lp.mean(0)) / 2
    return f, logh


def base_signals(subs, secs):
    path = f"{OUT}/base_{COHORT}.npz"
    if not os.path.exists(path):
        S = {}
        for s in subs:
            cool(); S[s] = K.base(B.load(s, secs)[0])
        np.savez(path, **S)
    return dict(np.load(path))


def encoders():
    import fei_pretrain as F
    import fei_branchC as C
    from head_to_head_cv import h20_args
    assert F.DEV == "cpu", "run with CUDA_VISIBLE_DEVICES=''"
    args = h20_args(); C._configure(args)
    enc = {}
    for m in MEMBERS:   # same construction + seeds as stroke_phaseB_clean.embed
        torch.manual_seed(int(m[1:]) if m[0] == "f" else int(m[-1]))
        a, c = F.Encoder(args.d), C._enc(args)(args.d)
        if m[0] == "f":
            a.load_state_dict(torch.load(f"{B.ROOT}/code/fei_enc_cv_s0_{m}.pt", map_location="cpu"))
            c.load_state_dict(torch.load(f"{B.ROOT}/code/branchC_h20_enc_cv_s0_{m}.pt", map_location="cpu"))
        enc[m] = (a.eval(), c.eval())
    return F, C, enc


def embed_folds(subs, S):
    F, C, enc = encoders()
    f_all, logh_all = fit_filter(list(S.values()))
    ref = np.load(f"{K.OUT}/filter_{COHORT}.npz")["log10_amp_gain"]
    assert np.allclose(logh_all, ref, atol=1e-6), f"pooled filter != stroke_phaseB_clean's ({np.abs(logh_all - ref).max()})"
    band = (f_all >= 0.5) & (f_all <= 99)
    E = {}
    for h in subs:
        path = f"{OUT}/emb_{COHORT}_heldout_{h}.npz"
        if not os.path.exists(path):
            cool()
            f, logh = fit_filter([S[s] for s in subs if s != h])
            mem = {s: K.frames(K.zs(K.shape(S[s], f, logh))) for s in subs}
            F.load_continuous = lambda p: mem[p].transpose(1, 0, 2).reshape(19, -1)   # embed straight from memory
            files = [(s, 0) for s in subs]
            out = dict(logh=logh, max_dev_vs_pooled=np.abs(logh - logh_all)[:, band].max())
            for m, (a, c) in enc.items():
                cool()
                out[f"fei_{m}"], out[f"c_{m}"] = F.embed_set(a, files, C.L)[0], C.embed_dyn(c, files)[0]
            np.savez(path, **out)
            print(f"  held-out {h}: filter max|Δ log10 gain| vs pooled {float(out['max_dev_vs_pooled']):.3f}", flush=True)
        E[h] = dict(np.load(path))
    return E


def loso_fl(Xs, y):   # Xs[i] = (n,d) features from fold i's filter; fit on the others, score i
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
        p[i] = clf.fit(Xs[i][tr], y[tr]).predict_proba(Xs[i][i:i + 1])[0, 1]
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    subs, secs = B.COHORTS[COHORT]
    y = np.array([int(s.startswith("PAC")) for s in subs])
    E = embed_folds(subs, base_signals(subs, secs))
    g = json.load(open(f"{B.OUT}/gate_{COHORT}.json"))
    bsi = np.array([[g["subjects"][s][k] for k in B.BSI_KEYS] for s in subs])
    feat = {"FEI+C": lambda e, m: np.hstack([e[f"fei_{m}"], e[f"c_{m}"]]),
            "FEI+C+BSI": lambda e, m: np.hstack([e[f"fei_{m}"], e[f"c_{m}"], bsi]),
            "FEI": lambda e, m: e[f"fei_{m}"], "C": lambda e, m: e[f"c_{m}"]}
    res, P, rng = {}, {}, np.random.RandomState(0)
    for kind, mems in (("", MEMBERS[:5]), ("rand-", MEMBERS[5:])):
        for name, fn in feat.items():
            arm = kind + name
            M = [[fn(E[h], m) for h in subs] for m in mems]
            Pm = np.array([loso_fl(Xs, y) for Xs in M]); P[arm] = Pm.mean(0)
            r = B.metrics(y, P[arm]) | dict(auroc_ci=B.bootstrap_ci(y, P[arm], "auroc", B.N_BOOT),
                                             member_auroc=[B.metrics(y, q)["auroc"] for q in Pm])
            if arm in PERM_ARMS:
                null = []
                for _ in range(B.N_PERM):
                    cool(); yp = rng.permutation(y)
                    null.append(B.metrics(yp, np.mean([loso_fl(Xs, yp) for Xs in M], 0))["auroc"])
                r["perm_p"] = float((1 + np.sum(np.array(null) >= r["auroc"])) / (1 + B.N_PERM))
            res[arm] = r
            print(f"  {arm:16s} LOSO AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]"
                  + (f"  perm p {r['perm_p']:.3f}" if "perm_p" in r else ""), flush=True)
    tr = json.load(open(f"{K.OUT}/results_{COHORT}.json"))   # the transductive (pooled-filter) run
    assert tr["preds"]["subject"] == subs
    P["FEI+C (pooled filter)"], P["BSI"] = np.array(tr["preds"]["FEI+C"]), np.array(tr["preds"]["BSI"])
    comps = [("FEI+C", "FEI+C (pooled filter)"), ("FEI+C", "rand-FEI+C"), ("FEI+C", "BSI"), ("FEI+C+BSI", "BSI"),
             ("FEI", "rand-FEI"), ("C", "rand-C")]
    paired = {f"{a} vs {b}": B.paired(y, P[a], P[b]) for a, b in comps}
    out = dict(cohort=COHORT, n=len(y), n_stroke=int(y.sum()), arms=res, paired=paired,
               filter_max_dev_vs_pooled={h: float(E[h]["max_dev_vs_pooled"]) for h in subs},
               preds={"subject": subs, "y": y.tolist()} | {k: v.tolist() for k, v in P.items()})
    B.save(f"{OUT}/results_{COHORT}.json", out)
    for k, v in paired.items():
        print(f"  {k:30s} Δ {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  p(Δ≤0) {v['p_le0']:.3f}", flush=True)
    print("PHASE B FOLD-LOCAL DONE", flush=True)


if __name__ == "__main__":
    main()
