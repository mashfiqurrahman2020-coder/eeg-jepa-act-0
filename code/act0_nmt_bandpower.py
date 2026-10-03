"""Act-0 Experiment 5 band-power arm: NMT relative band power on the identical seed-0 5-fold split
and probe as code/runs/act0/a0_feic_nmt/results.json, full Act-0 metric set, from the cached
95-dim features of baseline_bandpower.py (200 Hz, 5 bands up to 45 Hz x 19 ch). CPU only.

    ~/.eeg_vjepa_venv/bin/python code/act0_nmt_bandpower.py   ->  code/runs/act0/a0_feic_nmt/bandpower.json

Run it with the project venv (sklearn 1.5.2): a newer sklearn shifts the thresholded metrics in the 3rd decimal.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads, cool_gate
cap_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from act0_metrics import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs/act0/a0_feic_nmt/bandpower.json")
d = np.load(os.path.join(HERE, "bandpower_features.npz"))
X = np.vstack([d["Xb_tr"], d["Xb_ev"]]); y = np.concatenate([d["y_tr"], d["y_ev"]]).astype(int)
ref = json.load(open(os.path.join(HERE, "head_to_head_cv_results.json")))["folds"]
a0 = json.load(open(os.path.join(HERE, "runs/act0/a0_feic_nmt/results.json")))

folds = {}
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
    cool_gate()
    p = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
                      ).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    folds[str(k)] = metrics(y[te], p)
    # same folds as the FEI/GTJ arms (sizes match) and as head_to_head_cv's band-power arm (AUROC matches;
    # the 6e-4 residual there is its different solver settings)
    assert len(te) == a0[str(k)]["FEI+C"]["n"] and int(y[te].sum()) == a0[str(k)]["FEI+C"]["n_pos"]
    assert abs(folds[str(k)]["auroc"] - ref[str(k)]["band-power"]["auroc"]) < 2e-3

def paired(a, b):
    return {"mean_diff": float(np.mean(a - b)), "wins": int((a > b).sum()), "ttest_rel_p": float(ttest_rel(a, b).pvalue)}

bp = np.array([folds[str(k)]["auroc"] for k in range(5)])
auc = {arm: np.array([a0[str(k)][arm]["auroc"] for k in range(5)]) for arm in ("FEI@4000", "C", "FEI+C")}
out = {"folds": folds,
       "summary": {q: [float(np.mean([f[q] for f in folds.values()])), float(np.std([f[q] for f in folds.values()]))]
                   for q in ("auroc", "bal_acc", "auc_pr", "f1", "sens", "spec")},
       "paired_vs_bandpower_auroc": {arm: paired(v, bp) for arm, v in auc.items()},
       # Experiment 5's fusion gain over each branch alone, same folds
       "paired_fused_vs_branch_auroc": {arm: paired(auc["FEI+C"], auc[arm]) for arm in ("FEI@4000", "C")}}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("summary", "paired_vs_bandpower_auroc", "paired_fused_vs_branch_auroc")}, indent=1))
print("wrote", OUT)
