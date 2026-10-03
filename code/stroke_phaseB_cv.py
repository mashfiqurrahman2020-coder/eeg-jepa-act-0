"""Phase B CV robustness (post-hoc five-fold sensitivity analysis).
Same cached embeddings (runs/act0/phaseB{,_joint}/emb_*.npz) and the same classifier as the preregistered LOSO in
stroke_phaseB.py; only the cross-validation scheme changes:
  loso = the preregistered leave-one-subject-out, recomputed (asserted equal to results_*.json)
  kf5  = repeated stratified 5-fold, 100 repeats, pooled out-of-fold AUROC per repeat (mean, 2.5-97.5 %)
         + label-permutation p (200 perms, 10 repeats each) for the key arms
  lpo  = leave-pair-out (Airola et al. 2011): each stroke x control pair held out together, AUROC =
         P(score_stroke > score_control). Unbiased for small n, unlike pooled leave-one-out.
Exploratory arms (not preregistered): FEI, C, rand-FEI, rand-C alone. "(joint)" = NMT+TUAB-train pretrained encoders.
  python stroke_phaseB_cv.py   -> runs/act0/phaseB_cv/results_{cohort}.json (resumable per cohort) + table
"""
import json
import os

import stroke_phaseB as B   # cap_threads(4) before numpy; classifier + LOSO definitions
from hw_guard import cool_gate
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = f"{B.ROOT}/code/runs/act0/phaseB_cv"
R, N_PERM, R_PERM, N_BOOT = 100, 200, 10, 2000
PERM_ARMS = ["BSI", "VJEPA", "rand-VJEPA", "FEI+C", "FEI+C+BSI", "rand-FEI+C", "FEI+C (joint)", "FEI+C+BSI (joint)"]
COMPS = [("VJEPA+BSI", "BSI"), ("VJEPA", "rand-VJEPA"), ("FEI+C", "VJEPA"),              # preregistered
         ("FEI+C", "rand-FEI+C"), ("FEI+C", "BSI"), ("FEI+C+BSI", "BSI"), ("FEI+C (joint)", "rand-FEI+C"),
         ("FEI", "rand-FEI"), ("C", "rand-C")]
GATE = lambda: cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)
clf = lambda: B.make_pipeline(B.StandardScaler(), B.LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))


def arms(E, Ej, bsi):
    fc = lambda E, s: np.hstack([E[f"fei_{s}"], E[f"c_{s}"]])
    A = {"BSI": [bsi], "VJEPA": [E["vitm_released"]], "VJEPA+BSI": [np.hstack([E["vitm_released"], bsi])],
         "rand-VJEPA": [E[f"vitm_rand{s}"] for s in range(3)],
         "rand-VJEPA+BSI": [np.hstack([E[f"vitm_rand{s}"], bsi]) for s in range(3)],
         "rand-FEI+C": [fc(E, f"rand{s}") for s in range(3)],
         "rand-FEI+C+BSI": [np.hstack([fc(E, f"rand{s}"), bsi]) for s in range(3)],
         "rand-FEI": [E[f"fei_rand{s}"] for s in range(3)], "rand-C": [E[f"c_rand{s}"] for s in range(3)]}
    for sfx, e in [("", E), (" (joint)", Ej)]:
        A |= {f"FEI+C{sfx}": [fc(e, f"f{k}") for k in range(5)],
              f"FEI+C+BSI{sfx}": [np.hstack([fc(e, f"f{k}"), bsi]) for k in range(5)],
              f"FEI{sfx}": [e[f"fei_f{k}"] for k in range(5)], f"C{sfx}": [e[f"c_f{k}"] for k in range(5)]}
    return A


def kf5(members, y, seed):   # pooled out-of-fold ensemble P(stroke) for one repeat
    p = np.zeros((len(members), len(y)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y):
        for m, X in enumerate(members):
            p[m, te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p.mean(0)


def kf5_aurocs(members, y, n):
    out = []
    for s in range(n):
        GATE()
        out.append(roc_auc_score(y, kf5(members, y, s)))
    return np.array(out)


def lpo(members, y):   # W[a, b] = 1 if stroke a outscored control b with both held out (0.5 on a tie)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    W = np.zeros((len(pos), len(neg)))
    for a, i in enumerate(pos):
        GATE()
        for b, j in enumerate(neg):
            tr = np.setdiff1d(np.arange(len(y)), [i, j])
            s = np.mean([clf().fit(X[tr], y[tr]).predict_proba(X[[i, j]])[:, 1] for X in members], 0)
            W[a, b] = (s[0] > s[1]) + 0.5 * (s[0] == s[1])
    return W


def lpo_paired(Wa, Wb, seed=0):   # stratified bootstrap over subjects on the precomputed pair matrices
    rng, d = np.random.RandomState(seed), []
    for _ in range(N_BOOT):
        i, j = rng.randint(0, Wa.shape[0], Wa.shape[0]), rng.randint(0, Wa.shape[1], Wa.shape[1])
        d.append(Wa[i][:, j].mean() - Wb[i][:, j].mean())
    d = np.array(d)
    return dict(delta=float(Wa.mean() - Wb.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p_le0=float((d <= 0).mean()))


def cohort(c):
    path = f"{OUT}/results_{c}.json"
    if os.path.exists(path):
        return json.load(open(path))
    E, Ej = np.load(f"{B.OUT}/emb_{c}.npz"), np.load(f"{B.OUT}_joint/emb_{c}.npz")
    names, y = list(E["names"]), E["y"]
    assert list(Ej["names"]) == names
    g = json.load(open(f"{B.OUT}/gate_{c}.json"))
    A = arms(E, Ej, np.array([[g["subjects"][s][k] for k in B.BSI_KEYS] for s in names]))
    pre = {"": json.load(open(f"{B.OUT}/results_{c}.json")), " (joint)": json.load(open(f"{B.OUT}_joint/results_{c}.json"))}
    res, K, W = {}, {}, {}
    for name, mem in A.items():
        p_loso = np.mean([B.loso(X, y) for X in mem], 0)
        r = dict(loso=float(roc_auc_score(y, p_loso)), n_members=len(mem), dim=int(mem[0].shape[1]))
        base = name.replace(" (joint)", ""); stored = pre[name[len(base):]]["arms"].get(base)
        if stored:   # the preregistered numbers must reproduce exactly
            assert abs(stored["auroc"] - r["loso"]) < 1e-9, (name, stored["auroc"], r["loso"])
        K[name] = kf5_aurocs(mem, y, R)
        W[name] = lpo(mem, y)
        r |= dict(kf5_mean=float(K[name].mean()), kf5_sd=float(K[name].std()),
                  kf5_pi=[float(np.percentile(K[name], 2.5)), float(np.percentile(K[name], 97.5))], lpo=float(W[name].mean()))
        if name in PERM_ARMS:   # null = same statistic (mean over the first R_PERM repeats) on shuffled labels
            rng, obs = np.random.RandomState(0), K[name][:R_PERM].mean()
            null = np.array([kf5_aurocs(mem, rng.permutation(y), R_PERM).mean() for _ in range(N_PERM)])
            r["kf5_perm_p"] = float((1 + (null >= obs).sum()) / (1 + N_PERM))
        res[name] = r
        print(f"  {c:11s} {name:18s} loso {r['loso']:.3f}  kf5 {r['kf5_mean']:.3f}±{r['kf5_sd']:.3f}  lpo {r['lpo']:.3f}"
              + (f"  kf5 perm p {r['kf5_perm_p']:.3f}" if "kf5_perm_p" in r else ""), flush=True)
    paired = {f"{a} vs {b}": dict(kf5_delta=float((K[a] - K[b]).mean()), kf5_frac_a_better=float((K[a] > K[b]).mean()),
                                  kf5_frac_tie=float((K[a] == K[b]).mean()), lpo=lpo_paired(W[a], W[b]))
              for a, b in COMPS}
    out = dict(cohort=c, n=len(y), n_stroke=int(y.sum()), repeats=R, arms=res, paired=paired)
    B.save(path, out)
    np.savez(f"{OUT}/kf5_lpo_{c}.npz", **{f"kf5__{k}": v for k, v in K.items()}, **{f"lpo__{k}": v for k, v in W.items()})
    return out


def selftest():   # LPO on separable data = 1, on noise ~0.5; kf5 perm machinery sane
    rng = np.random.RandomState(0); y = np.r_[np.ones(9), np.zeros(6)].astype(int)
    X = rng.randn(15, 4) + 3 * y[:, None]
    assert lpo([X], y).mean() == 1.0 and kf5_aurocs([X], y, 3).min() == 1.0
    assert 0.2 < lpo([rng.randn(15, 4)], y).mean() < 0.8
    Wa = np.ones((9, 6)); assert lpo_paired(Wa, Wa)["delta"] == 0
    print("stroke_phaseB_cv self-check OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    for c in B.COHORTS:
        r = cohort(c)
        print(f"\n== {c} (n={r['n']}, {r['n_stroke']} stroke) ==")
        for k, v in r["paired"].items():
            l = v["lpo"]
            print(f"  {k:30s} kf5 Δ {v['kf5_delta']:+.3f} (A better in {v['kf5_frac_a_better']:.0%} of repeats)   "
                  f"lpo Δ {l['delta']:+.3f} [{l['ci'][0]:+.3f}, {l['ci'][1]:+.3f}] p(Δ≤0) {l['p_le0']:.2f}")
    print("PHASE B CV DONE")
