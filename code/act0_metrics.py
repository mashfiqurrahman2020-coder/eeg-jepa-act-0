"""Shared detailed metrics for every Act-0 step (positive class = abnormal / stroke = 1).
  python act0_metrics.py   # self-check
"""
import json

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)


def metrics(y, p, thr=0.5):
    """y: 0/1 labels, p: P(positive). Threshold 0.5 unless stated (pre-declared, never tuned on test)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(n=int(len(y)), n_pos=int(y.sum()), auroc=float(roc_auc_score(y, p)),
                auc_pr=float(average_precision_score(y, p)), acc=float((yh == y).mean()),
                bal_acc=float(balanced_accuracy_score(y, yh)), f1=float(f1_score(y, yh, zero_division=0)),
                sens=float(tp / max(tp + fn, 1)), spec=float(tn / max(tn + fp, 1)),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def bootstrap_ci(y, p, key="auroc", n=2000, seed=0):
    """95% percentile CI of a metric over resampled subjects (resamples lacking a class are skipped)."""
    y, p = np.asarray(y), np.asarray(p)
    rng, vals = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(metrics(y[i], p[i])[key])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fmt(m):
    return (f"AUROC {m['auroc']:.3f}  AUC-PR {m['auc_pr']:.3f}  Acc {m['acc']:.3f}  BAcc {m['bal_acc']:.3f}  "
            f"F1 {m['f1']:.3f}  Sens {m['sens']:.3f}  Spec {m['spec']:.3f}  "
            f"[TP {m['tp']} FP {m['fp']} TN {m['tn']} FN {m['fn']}]")


def save(path, obj):
    json.dump(obj, open(path, "w"), indent=1)


if __name__ == "__main__":
    m = metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1) and m["auroc"] == 0.75 and m["bal_acc"] == 0.5
    lo, hi = bootstrap_ci([0, 1] * 10, np.r_[[0.2, 0.8] * 10])
    assert lo == hi == 1.0
    print("act0_metrics self-check OK:", fmt(m))
