"""Act-0 Experiment 5 significance, revised for overlapping CV folds. The report's fold-level ttest_rel treats
the 5 folds as independent, but CV training sets overlap (each pair shares 3/4 of the data), so it understates the
variance. This adds the Nadeau-Bengio corrected resampled t-test (Mach. Learn. 52:239, 2003): var x (1/k + n_test/n_train),
df = k-1, and Holm's step-down adjustment over the whole family below. Reads saved fold AUROCs only; no refitting.
If act0_nmt_pretrain_replicate.py has run, "pretrained" per fold is the mean over the original run and its replicates.
  python act0_nmt_stats.py   -> code/runs/act0/a0_feic_nmt_stats.json
"""
import json
import os

import numpy as np
from scipy.stats import t as tdist, ttest_rel

R = f"{os.path.dirname(os.path.abspath(__file__))}/runs/act0"
ARMS = ("FEI@4000", "C", "FEI+C")


def nb(a, b, n_tr, n_te):   # Nadeau-Bengio corrected resampled t-test, two-sided
    d = np.asarray(a) - np.asarray(b); k = len(d)
    t = d.mean() / np.sqrt((1 / k + n_te / n_tr) * d.var(ddof=1))
    return float(t), float(2 * tdist.sf(abs(t), k - 1))


def holm(p):
    o = np.argsort(p); adj = np.maximum.accumulate([min(1, (len(p) - i) * p[j]) for i, j in enumerate(o)])
    out = np.empty(len(p)); out[o] = adj
    return out


def main():
    pre = json.load(open(f"{R}/a0_feic_nmt/results.json"))
    bp = json.load(open(f"{R}/a0_feic_nmt/bandpower.json"))["folds"]
    rnd = json.load(open(f"{R}/a0_feic_nmt_rand/results.json"))["summary"]
    rp = f"{R}/a0_feic_nmt_rep/results.json"
    rep = json.load(open(rp)) if os.path.exists(rp) else {}
    reps = sorted({k.split("_")[0] for k in rep if all(f"{k.split('_')[0]}_f{f}" in rep for f in range(5))})
    n_te = np.array([bp[str(k)]["n"] for k in range(5)]); n_tr = n_te.sum() - n_te
    ratio_n_tr, ratio_n_te = n_tr.mean(), n_te.mean()
    A = {}
    for m in ARMS:
        draws = [[pre[str(k)][m]["auroc"] for k in range(5)]] + [[rep[f"{r}_f{k}"][m]["auroc"] for k in range(5)] for r in reps]
        A[m] = np.mean(draws, 0)
        A[f"rand-{m}"] = np.array(rnd[f"rand-{m}"]["auroc_per_fold"])
    A["band power"] = np.array([bp[str(k)]["auroc"] for k in range(5)])
    comps = [(m, "band power") for m in ARMS] + [(f"rand-{m}", "band power") for m in ARMS] + [(m, f"rand-{m}") for m in ARMS] \
        + [("FEI+C", "FEI@4000"), ("FEI+C", "C")]
    rows = []
    for a, b in comps:
        tc, pc = nb(A[a], A[b], ratio_n_tr, ratio_n_te)
        rows.append(dict(a=a, b=b, mean_a=float(A[a].mean()), mean_b=float(A[b].mean()), diff=float((A[a] - A[b]).mean()),
                         folds_a_better=int((A[a] > A[b]).sum()), p_naive=float(ttest_rel(A[a], A[b]).pvalue), t_nb=tc, p_nb=pc))
    for r, h in zip(rows, holm(np.array([r["p_nb"] for r in rows]))):
        r["p_nb_holm"] = float(h)
    out = dict(pretraining_draws=["original"] + reps, n_train_mean=float(ratio_n_tr), n_test_mean=float(ratio_n_te),
               correction_factor=float(1 / 5 + ratio_n_te / ratio_n_tr), comparisons=rows)
    json.dump(out, open(f"{R}/a0_feic_nmt_stats.json", "w"), indent=1)
    print(f"pretraining draws: {out['pretraining_draws']}   n_train {ratio_n_tr:.0f}  n_test {ratio_n_te:.0f}")
    for r in rows:
        print(f"{r['a']:14s} vs {r['b']:14s} {r['mean_a']:.3f} vs {r['mean_b']:.3f}  Δ {r['diff']:+.3f}  {r['folds_a_better']}/5  "
              f"p naive {r['p_naive']:.4f}  p NB {r['p_nb']:.4f}  Holm {r['p_nb_holm']:.4f}")


if __name__ == "__main__":
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    main()
