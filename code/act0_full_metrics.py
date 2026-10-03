"""Round-9 report enrichment: compute the FULL metric set (Acc, BAcc, F1, Sens, Spec, AUC-PR,
confusion matrix), not just AUROC, for every Act-0 experiment/arm, from the raw predictions
already saved on disk by each experiment's own script. No new training, no new numbers invented
-- this only re-reads saved (y, p) pairs through act0_metrics.metrics(), the same formula already
used (and spot-checked) for Experiments 5 and 7b in the submitted report.
  python act0_full_metrics.py > act0_full_metrics_out.txt
"""
import json

import numpy as np

from act0_metrics import metrics, fmt

ROOT = "/home/mashfiq/eeg_vjepa"


def agg(dicts, keys=("auroc", "auc_pr", "acc", "bal_acc", "f1", "sens", "spec")):
    return {k: (float(np.mean([d[k] for d in dicts])), float(np.std([d[k] for d in dicts]))) for k in keys}


def pr(label, dicts):
    a = agg(dicts)
    print(f"  {label:42s} " + "  ".join(f"{k} {m:.3f}±{s:.3f}" for k, (m, s) in a.items()))


def load(path, prob_key="prob_abnormal", label_key="label"):
    d = np.load(path, allow_pickle=True)
    y = d[label_key]
    p = d[prob_key] if prob_key in d.files else d["prob"][:, 1]
    return y, p


print("=" * 100)
print("EXPERIMENT 1 -- released-checkpoint TUAB, 5 subsets each (3 seeds for random-init)")
R1 = f"{ROOT}/code/runs/tuab_cached"
for name, pattern, n in [
    ("ViT-M, §4.2 head", "c_vitm_frozen_s{i}", 5),
    ("ViT-M, Fig-3 head", "c_vitm_attentive_s{i}", 5),
    ("Random ViT-M, §4.2 head", "c_vitm_rand_s{i}", 3),
    ("Random ViT-M, Fig-3 head", "c_vitm_rand_attentive_s{i}", 3),
    ("ViT-B, §4.2 head", "c_vitb_frozen_s{i}", 5),
    ("ViT-B, Fig-3 head", "c_vitb_attentive_s{i}", 5),
]:
    ds = [metrics(*load(f"{R1}/{pattern.format(i=i)}/final_preds.npz")) for i in range(n)]
    pr(name, ds)

print()
print("=" * 100)
print("EXPERIMENT 2 -- fine-tune vs from-scratch, final epoch, 5 subsets")
R2 = f"{ROOT}/code/runs/tuab_finetune"
for name, pattern in [("Fine-tune (released)", "ft_vitm_s{i}_bf16"), ("From scratch (random)", "scratch_vitm_s{i}_bf16")]:
    ds = [metrics(*load(f"{R2}/{pattern.format(i=i)}/final_preds.npz")) for i in range(5)]
    pr(name, ds)

print()
print("=" * 100)
print("EXPERIMENT 4 -- simplest linear probe (band power / random ViT-M / released ViT-M)")
R4a = f"{ROOT}/code/runs/act0/p13_bandpower"
ds = [metrics(*load(f"{R4a}/preds_s{i}.npz")) for i in range(5)]
pr("Band power (hand-crafted)", ds)
print("  full-train:", fmt(metrics(*load(f"{R4a}/preds_full_train.npz"))))
R4b = f"{ROOT}/code/runs/act0/tuab_vitm_linear"
ds = [metrics(*load(f"{R4b}/preds_released_s{i}.npz")) for i in range(5)]
pr("Released ViT-M (linear probe)", ds)
print("  full-train:", fmt(metrics(*load(f"{R4b}/preds_released_full_train.npz"))))
ds = [metrics(*load(f"{R4b}/preds_rand_s{i}.npz")) for i in range(3)]
pr("Random-init ViT-M (linear probe)", ds)

print()
print("=" * 100)
print("EXPERIMENT 5 -- FEI+C on NMT, seed-0 5-fold CV")
R5 = f"{ROOT}/code/runs/act0/a0_feic_nmt"
for branch, key in [("FEI", "FEI@4000"), ("C", "C"), ("FEI+C", "FEI+C")]:
    ds = []
    for f in range(5):
        d = np.load(f"{R5}/preds_f{f}.npz", allow_pickle=True)
        ds.append(metrics(d["label"], d[key]))
    pr(branch, ds)

print()
print("=" * 100)
print("EXPERIMENT 6 -- TUAB transfer, full-train and 5-subset mean, every arm")
print("  (FEI/rand-FEI standalone = NMT-only pretrain dir 'tuab_feic'; C/FEI+C arms = NMT+TUAB-joint dir 'tuab_feic_joint',")
print("   matching the report's own stated methodology -- the FEI inside the fused row is a separately-pretrained encoder)")
R6 = f"{ROOT}/code/runs/act0/tuab_feic_joint"
R6_solo = f"{ROOT}/code/runs/act0/tuab_feic"
SRC = {"FEI": R6_solo, "rand-FEI": R6_solo, "C": R6, "rand-C": R6, "FEI+C": R6, "rand-FEI+C": R6}
for arm, src in SRC.items():
    full = metrics(*load(f"{src}/preds_full_train_{arm}.npz"))
    ds = [metrics(*load(f"{src}/preds_s{i}_{arm}.npz")) for i in range(5)]
    sub = agg(ds)
    print(f"  {arm:14s} [{'solo' if src == R6_solo else 'joint'}] full-train: {fmt(full)}")
    print(f"  {'':14s}          subset mean: " + "  ".join(f"{k} {m:.3f}±{s:.3f}" for k, (m, s) in sub.items()))
print(f"  {'Band power':14s} full-train: {fmt(metrics(*load(f'{R4a}/preds_full_train.npz')))}")
print(f"  {'Released ViT-M':14s} full-train: {fmt(metrics(*load(f'{R4b}/preds_released_full_train.npz')))}")

print()
print("=" * 100)
print("EXPERIMENT 8 -- stroke LOSO, every arm (primary cohort, clean/spectrum-matched prep)")
d = json.load(open(f"{ROOT}/code/runs/act0/phaseB_clean/results_primary.json"))
y = np.array(d["preds"]["y"])
for arm in d["preds"]:
    if arm in ("subject", "y"):
        continue
    p = np.array(d["preds"][arm])
    print(f"  {arm:22s} " + fmt(metrics(y, p)))

print("\nDONE -- all numbers above are read from existing saved predictions, no retraining.")
