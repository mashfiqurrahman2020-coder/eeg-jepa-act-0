"""Failure-axis analysis of FEI and Branch C (GTJ) from SAVED predictions only (no training).

Sources (all already on disk):
  NMT : code/runs/act0/a0_feic_nmt/preds_f*.npz  -> out-of-fold P(abnormal), seed-0 5-fold CV, 2417 recs
  TUAB: code/runs/act0/tuab_feic/preds_full_train_*.npz -> 276 official eval recs, full-train probe
Metadata: NMT Labels.csv (age, gender); TUAB EDF header (sex, Age:, duration).
Axes: class, age, sex, recording length, confidence, FEI-vs-C error overlap, age-as-shortcut.
Writes code/runs/act0/failure_axes/results.json.
"""
import csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from hw_guard import cap_threads, cool_gate
cap_threads(4)
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = "/home/mashfiq/eeg_vjepa"
OUT = f"{ROOT}/code/runs/act0/failure_axes"
RNG = np.random.default_rng(0)
NBOOT = 1000


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def boot_ci(y, s):
    """95% percentile bootstrap CI of AUROC (stratified resampling keeps both classes)."""
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    if len(i0) < 2 or len(i1) < 2:
        return None
    v = []
    for b in range(NBOOT):
        if b % 250 == 0:
            cool_gate()
        j = np.r_[RNG.choice(i0, len(i0)), RNG.choice(i1, len(i1))]
        v.append(roc_auc_score(y[j], s[j]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_diff(y, a, b):
    """Paired bootstrap of AUROC(a) - AUROC(b) on the same recordings; two-sided p."""
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    d = []
    for k in range(NBOOT):
        if k % 250 == 0:
            cool_gate()
        j = np.r_[RNG.choice(i0, len(i0)), RNG.choice(i1, len(i1))]
        d.append(roc_auc_score(y[j], a[j]) - roc_auc_score(y[j], b[j]))
    d = np.array(d)
    return {"diff": auc(y, a) - auc(y, b), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def cm(y, s, t=0.5):
    p = s >= t
    tp, fn = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum())
    fp, tn = int((p & (y == 0)).sum()), int((~p & (y == 0)).sum())
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1)}


def by_bins(y, scores, key, edges, names):
    """Per-bin n, prevalence, AUROC + CI, sens/spec for every arm."""
    rows = []
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (key >= lo) & (key < hi)
        r = {"bin": nm, "n": int(m.sum()), "n_abn": int(y[m].sum())}
        for arm, s in scores.items():
            r[arm] = {"auroc": auc(y[m], s[m]), "ci": boot_ci(y[m], s[m]), **cm(y[m], s[m])}
        rows.append(r)
    return rows


def by_cat(y, scores, key):
    rows = []
    for c in sorted(set(key)):
        m = key == c
        r = {"bin": str(c), "n": int(m.sum()), "n_abn": int(y[m].sum())}
        for arm, s in scores.items():
            r[arm] = {"auroc": auc(y[m], s[m]), "ci": boot_ci(y[m], s[m]), **cm(y[m], s[m])}
        rows.append(r)
    return rows


def overlap(y, a, b, t=0.5):
    """Error overlap between two arms at threshold t."""
    ea, eb = (a >= t) != (y == 1), (b >= t) != (y == 1)
    return {"err_a": int(ea.sum()), "err_b": int(eb.sum()), "both": int((ea & eb).sum()),
            "only_a": int((ea & ~eb).sum()), "only_b": int((~ea & eb).sum()),
            "score_corr": float(np.corrcoef(a, b)[0, 1])}


def confidence(y, s):
    """Errors split by how far from 0.5 the score was."""
    err = (s >= 0.5) != (y == 1)
    d = np.abs(s - 0.5)
    return {"n_err": int(err.sum()),
            "err_near_(|p-0.5|<0.1)": int((err & (d < 0.1)).sum()),
            "err_confident_(|p-0.5|>=0.3)": int((err & (d >= 0.3)).sum()),
            "median_|p-0.5|_errors": float(np.median(d[err])) if err.any() else None,
            "median_|p-0.5|_correct": float(np.median(d[~err]))}


def shortcut(y, s, age, male):
    """Is the score partly an age/sex detector? age-alone AUROC + within-class score~age corr."""
    ok = ~np.isnan(age)
    r = {"age_alone_auroc": auc(y[ok], age[ok]), "age_alone_auroc_neg": auc(y[ok], -age[ok])}
    for c, nm in ((0, "normal"), (1, "abnormal")):
        m = ok & (y == c)
        r[f"corr_score_age_{nm}"] = float(np.corrcoef(s[m], age[m])[0, 1])
    return r


def summarize(y, scores, age, male, length, age_edges, age_names, len_edges, len_names):
    res = {"n": int(len(y)), "n_abn": int(y.sum()),
           "overall": {a: {"auroc": auc(y, s), "ci": boot_ci(y, s), **cm(y, s),
                           "confidence": confidence(y, s)} for a, s in scores.items()}}
    res["age"] = by_bins(y, scores, np.nan_to_num(age, nan=-1), age_edges, age_names)
    res["sex"] = by_cat(y, scores, np.where(male == 1, "male", np.where(male == 0, "female", "unknown")))
    res["length"] = by_bins(y, scores, length, len_edges, len_names)
    res["shortcut"] = {a: shortcut(y, s, age, male) for a, s in scores.items()}
    return res


# ---------------- NMT ----------------
lab = {os.path.splitext(r["recordname"])[0]: r for r in csv.DictReader(open(f"{ROOT}/data/NMT-Scalp-EEG/Labels.csv"))}
rec, y, S = [], [], {"FEI": [], "C": [], "FEI+C": []}
for f in sorted(glob.glob(f"{ROOT}/code/runs/act0/a0_feic_nmt/preds_f*.npz")):
    d = np.load(f)
    rec += list(d["rec"]); y += list(d["label"])
    S["FEI"] += list(d["FEI@4000"]); S["C"] += list(d["C"]); S["FEI+C"] += list(d["FEI+C"])
y = np.array(y); S = {k: np.array(v) for k, v in S.items()}
rid = [os.path.splitext(os.path.basename(r))[0] for r in rec]
age = np.array([float(lab[i]["age"]) if lab[i]["age"].strip().replace(".", "").isdigit() else np.nan for i in rid])
male = np.array([1 if lab[i]["gender"] == "male" else 0 if lab[i]["gender"] == "female" else -1 for i in rid])
mins = np.array([np.load(r, mmap_mode="r").shape[0] * 2.5 / 60 for r in rec])  # 500-sample frames @ 200 Hz
assert len(set(rid)) == len(rid) == 2417
nmt = summarize(y, S, age, male, mins,
                [0, 18, 40, 60, 200], ["<18", "18-39", "40-59", "60+"],
                [0, 10, 15, 20, 1e9], ["<10 min", "10-15 min", "15-20 min", ">=20 min"])
nmt["age_missing"] = int(np.isnan(age).sum())
nmt["overlap_FEI_vs_C"] = overlap(y, S["FEI"], S["C"])
nmt["FEI+C_fixes"] = {  # errors of FEI and of C that the fusion gets right
    "FEI_errors_fixed_by_fusion": int((((S["FEI"] >= .5) != (y == 1)) & ((S["FEI+C"] >= .5) == (y == 1))).sum()),
    "C_errors_fixed_by_fusion": int((((S["C"] >= .5) != (y == 1)) & ((S["FEI+C"] >= .5) == (y == 1))).sum()),
    "new_errors_from_fusion": int((((S["FEI"] >= .5) == (y == 1)) & ((S["C"] >= .5) == (y == 1))
                                   & ((S["FEI+C"] >= .5) != (y == 1))).sum())}
kids = ~np.isnan(age) & (age < 18)
nmt["paired_C_minus_FEI"] = {"all": boot_diff(y, S["C"], S["FEI"]),
                             "<18": boot_diff(y[kids], S["C"][kids], S["FEI"][kids]),
                             ">=18": boot_diff(y[~kids & ~np.isnan(age)], S["C"][~kids & ~np.isnan(age)],
                                               S["FEI"][~kids & ~np.isnan(age)])}

# ---------------- TUAB ----------------
def edf_meta(stem):
    f = glob.glob(f"{ROOT}/data/TUAB/edf/eval/*/01_tcp_ar/{stem}.edf")[0]
    h = open(f, "rb").read(256)
    pid = h[8:88].decode("latin1").split()
    sex = pid[1] if len(pid) > 1 else "X"
    a = [t for t in pid if t.startswith("Age:")]
    ag = float(a[0][4:]) if a and a[0][4:].isdigit() else np.nan
    dur = int(h[236:244]) * float(h[244:252]) / 60
    return sex, ag, dur

T = {}
for arm in ["FEI", "C", "FEI+C", "rand-FEI", "rand-C", "rand-FEI+C"]:
    d = np.load(f"{ROOT}/code/runs/act0/tuab_feic/preds_full_train_{arm}.npz")
    T[arm] = d["prob_abnormal"]; ty = d["label"]; trec = list(d["rec"])
meta = [edf_meta(os.path.splitext(r)[0]) for r in trec]
tsex = np.array([1 if m[0] == "M" else 0 if m[0] == "F" else -1 for m in meta])
tage = np.array([m[1] for m in meta]); tdur = np.array([m[2] for m in meta])
tuab = summarize(ty, T, tage, tsex, tdur,
                 [0, 18, 40, 60, 200], ["<18", "18-39", "40-59", "60+"],
                 [0, 15, 20, 25, 1e9], ["<15 min", "15-20 min", "20-25 min", ">=25 min"])
tuab["age_missing"] = int(np.isnan(tage).sum())
tuab["overlap_FEI_vs_C"] = overlap(ty, T["FEI"], T["C"])
tuab["paired"] = {"C_minus_FEI": boot_diff(ty, T["C"], T["FEI"]),
                  "FEI+C_minus_FEI": boot_diff(ty, T["FEI+C"], T["FEI"]),
                  "FEI_minus_randFEI": boot_diff(ty, T["FEI"], T["rand-FEI"]),
                  "C_minus_randC": boot_diff(ty, T["C"], T["rand-C"])}

# normal-recording false-positive rate by fine age (NMT): the developmental-slowing check
FINE = [(0, 6), (6, 12), (12, 18), (18, 40), (40, 60), (60, 200)]
nmt["fp_rate_normals_by_age"] = [{"age": f"{lo}-{hi}", "n_normal": int(((age >= lo) & (age < hi) & (y == 0)).sum()),
    **{a: float((S[a][(age >= lo) & (age < hi) & (y == 0)] >= 0.5).mean()) for a in S}} for lo, hi in FINE]
# confound checks: sex and length within adults / children
adult = age >= 18
nmt["within_age_group"] = {g: {"sex": {sx: {a: auc(y[m & (male == v)], S[a][m & (male == v)]) for a in S}
                                       for sx, v in (("female", 0), ("male", 1))},
                               "length": {ln: {a: auc(y[m & k], S[a][m & k]) for a in S}
                                          for ln, k in (("<10 min", mins < 10), (">=10 min", mins >= 10))}}
                           for g, m in (("adults", adult), ("children", ~adult))}
long_ = tdur >= 25
tuab["long_>=25min"] = {"n": int(long_.sum()), "n_abn": int(ty[long_].sum()), **{a: auc(ty[long_], T[a][long_]) for a in T}}

os.makedirs(OUT, exist_ok=True)
json.dump({"NMT": nmt, "TUAB": tuab}, open(f"{OUT}/results.json", "w"), indent=1)
np.savez(f"{OUT}/per_recording.npz", nmt_rec=np.array(rid), nmt_y=y, nmt_age=age, nmt_male=male, nmt_min=mins,
         **{f"nmt_{k}": v for k, v in S.items()}, tuab_rec=np.array(trec), tuab_y=ty, tuab_age=tage,
         tuab_male=tsex, tuab_min=tdur, **{f"tuab_{k}": v for k, v in T.items()})
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"FEI": "#1f77b4", "C": "#2ca02c", "FEI+C": "#ff7f0e"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, a in enumerate(S):
    xs = np.arange(len(nmt["age"])) + (k - 1) * 0.22
    v = [b[a]["auroc"] for b in nmt["age"]]; ci = np.array([b[a]["ci"] for b in nmt["age"]])
    ax[0].errorbar(xs, v, yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], fmt="o", capsize=3, color=COL[a], label=a)
    ax[1].plot(range(len(FINE)), [r[a] for r in nmt["fp_rate_normals_by_age"]], "o-", color=COL[a], label=a)
ax[0].set_xticks(range(len(nmt["age"]))); ax[0].set_xticklabels([f'{b["bin"]}\n(n={b["n"]}, {b["n_abn"]} abn)' for b in nmt["age"]], fontsize=8)
ax[0].set_ylabel("AUROC (95% bootstrap CI)"); ax[0].set_title("(a) NMT out-of-fold AUROC by age"); ax[0].legend()
ax[1].set_xticks(range(len(FINE))); ax[1].set_xticklabels([f'{r["age"]}\n(n={r["n_normal"]})' for r in nmt["fp_rate_normals_by_age"]], fontsize=8)
ax[1].set_ylabel("share of NORMAL recordings called abnormal"); ax[1].set_title("(b) NMT false-positive rate by age (threshold 0.5)")
ax[1].set_xlabel("age (years); n = normal recordings"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/nmt_age_axis.png", dpi=130)
print("wrote", OUT)
