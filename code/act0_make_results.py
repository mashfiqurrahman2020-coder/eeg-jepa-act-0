"""Act-0 results summary: re-scores the saved predictions, embeddings and JSONs (no model is run) and computes Experiments
1, 2 and 4's Mann-Whitney, paired t- and Wilcoxon tests.
  python act0_make_results.py   # -> code/runs/act0/results_summary/{figs/*.png, numbers.json, tables.json}
"""
import sys; sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code"); from hw_guard import cap_threads; cap_threads(1)
import filecmp, glob, hashlib, json, os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from act0_metrics import bootstrap_ci, metrics

ROOT = "/home/mashfiq/eeg_vjepa"
R = f"{ROOT}/code/runs"
A = f"{R}/act0"
OUT = f"{A}/results_summary"
FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
N, T = {}, {}                       # numbers (pre-formatted strings) and markdown tables
PAPER = dict(vitm=0.877, vitb=0.879, ft=0.885)   # EEG-VJEPA paper, TUAB AUROC
COL = dict(rel="#c0392b", rnd="#7f8c8d", bp="#8e44ad", fei="#2471a3", c="#17a589", feic="#d68910", bsi="#6c3483",
           vj="#c0392b")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def J(p): return json.load(open(p))
def f3(x): return f"{x:.3f}"
def ms(a): return f"{np.mean(a):.3f} ± {np.std(a):.3f}"     # population sd, as in every saved summary
def ci(c): return f"[{c[0]:.3f}, {c[1]:.3f}]"
def pv(p): return "< 0.001" if p < 0.001 else f"{p:.3f}"
def auc(y, p): return float(roc_auc_score(y, p))
def table(head, rows): return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                                        + ["| " + " | ".join(map(str, r)) + " |" for r in rows])
def save(name): plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", bbox_inches="tight"); plt.close()


def paired_boot(y, pa, pb, n=2000, seed=0):
    """Same as tuab_vitm_linear.paired_boot (copied to avoid importing torch): recording bootstrap of AUROC(a)-AUROC(b)."""
    rng, d = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(auc(y[i], pa[i]) - auc(y[i], pb[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float((np.array(d) <= 0).mean())


# ───────────────────────── 1. Frozen reproduction on TUAB (P1.4) ─────────────────────────
RUNS = {"ViT-M, §4.2 head": ("c_vitm_frozen_s{}", 5), "ViT-M, Fig-3 head": ("c_vitm_attentive_s{}", 5),
        "Random ViT-M, §4.2 head": ("c_vitm_rand_s{}", 3), "Random ViT-M, Fig-3 head": ("c_vitm_rand_attentive_s{}", 3),
        "ViT-B, §4.2 head": ("c_vitb_frozen_s{}", 5), "ViT-B, Fig-3 head": ("c_vitb_attentive_s{}", 5)}
P14, CURVES = {}, {}
for name, (pat, n) in RUNS.items():
    fin, best, acc, f1, cur = [], [], [], [], []
    for s in range(n):
        d = f"{R}/tuab_cached/{pat.format(s)}"
        z, c = np.load(f"{d}/final_preds.npz"), pd.read_csv(f"{d}/probe_r0.csv")
        a = auc(z["label"], z["prob"][:, 1])
        assert abs(a - c.val_auroc.iloc[-1]) < 1e-3 and len(c) == 500, d      # final preds == last CSV row (≤3e-4: float32 preds)
        fin.append(a); best.append(c.val_auroc.max()); acc.append(c.val_acc.iloc[-1]); f1.append(c.val_f1.iloc[-1])
        cur.append(c.val_auroc.values)
    P14[name] = dict(fin=fin, best=best, acc=acc, f1=f1); CURVES[name] = np.array(cur)
rel = P14["ViT-M, §4.2 head"]["fin"] + P14["ViT-M, Fig-3 head"]["fin"]
rnd = P14["Random ViT-M, §4.2 head"]["fin"] + P14["Random ViT-M, Fig-3 head"]["fin"]
N["p14_rel_max"], N["p14_rnd_min"] = f3(max(rel)), f3(min(rnd))
for k, key in [("ViT-M, §4.2 head", "m42"), ("ViT-M, Fig-3 head", "m3"), ("Random ViT-M, §4.2 head", "r42"),
               ("Random ViT-M, Fig-3 head", "r3"), ("ViT-B, §4.2 head", "b42"), ("ViT-B, Fig-3 head", "b3")]:
    N[f"p14_{key}"] = ms(P14[k]["fin"]); N[f"p14_{key}_best"] = f3(np.mean(P14[k]["best"]))
for h, (a, b) in {"42": ("ViT-M, §4.2 head", "Random ViT-M, §4.2 head"), "3": ("ViT-M, Fig-3 head", "Random ViT-M, Fig-3 head")}.items():
    N[f"p14_mw{h}"] = pv(stats.mannwhitneyu(P14[a]["fin"], P14[b]["fin"], alternative="less").pvalue)
    N[f"p14_gap{h}"] = f3(np.mean(P14[b]["fin"]) - np.mean(P14[a]["fin"]))
N["p14_short_m"] = f"{PAPER['vitm'] - np.mean(P14['ViT-M, §4.2 head']['fin']):.3f}"
T["p14"] = table(["Encoder and head", "Runs", "Acc", "F1", "AUROC (final epoch)", "Best-epoch AUROC*", "Paper AUROC"],
                 [[k, len(v["fin"]), ms(np.asarray(v["acc"]) / 100), ms(v["f1"]), f"**{ms(v['fin'])}**", f3(np.mean(v["best"])),
                   f3(PAPER["vitb"] if "ViT-B" in k else PAPER["vitm"]) if "Random" not in k else "—"]
                  for k, v in P14.items()])
T["p14_runs"] = table(["Encoder and head"] + [f"s{i}" for i in range(5)],
                      [[k] + [f3(x) for x in v["fin"]] + ["—"] * (5 - len(v["fin"])) for k, v in P14.items()])

fig, ax = plt.subplots(figsize=(7.2, 3.2))
for i, (k, v) in enumerate(P14.items()):
    c = COL["rnd"] if "Random" in k else COL["rel"]
    ax.bar(i, np.mean(v["fin"]), color=c, alpha=.35, width=.6)
    ax.scatter([i] * len(v["fin"]), v["fin"], color=c, s=14, zorder=3)
ax.axhline(PAPER["vitm"], ls="--", c="k", lw=1); ax.text(5.45, PAPER["vitm"] + .004, "paper 0.877 / 0.879", ha="right", fontsize=8)
ax.axhline(.5, ls=":", c="grey", lw=1); ax.text(5.45, .505, "chance", ha="right", fontsize=8, color="grey")
ax.set_xticks(range(len(P14))); ax.set_xticklabels([k.replace(", ", "\n") for k in P14], fontsize=8)
ax.set_ylim(.45, .92); ax.set_ylabel("TUAB eval AUROC (final epoch)")
ax.set_title("Released checkpoints (red) vs random-init weights (grey); dots = training subsets")
save("p14_bars")

fig, ax = plt.subplots(figsize=(7.2, 3.0))
ep = np.arange(1, 501)
for k, c in [("ViT-M, §4.2 head", COL["rel"]), ("Random ViT-M, §4.2 head", COL["rnd"])]:
    m = CURVES[k]; ax.plot(ep, m.mean(0), c=c, label=f"{k} (mean of {len(m)})")
    ax.fill_between(ep, m.min(0), m.max(0), color=c, alpha=.15)
ax.axhline(PAPER["vitm"], ls="--", c="k", lw=1, label="paper ViT-M 0.877")
ax.set_xlabel("Epoch"); ax.set_ylabel("Eval AUROC"); ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(.45, .92)
ax.set_title("Training curves, authors' recipe (500 epochs, batch 2, lr 1e-3); band = min–max over subsets")
save("p14_curves")

# faithfulness evidence, recomputed from disk
up = f"{ROOT}/upstream/eeg-vjepa"
ups = [l.strip() for l in os.popen(f"git -C {up} ls-files").read().splitlines()]
same = [f for f in ups if os.path.exists(f"{ROOT}/code/{f}") and filecmp.cmp(f"{up}/{f}", f"{ROOT}/code/{f}", shallow=False)]
N["up_commit"] = os.popen(f"git -C {up} log -1 --format=%h").read().strip()
N["up_n"], N["up_same"] = str(len(ups)), str(len(same))
N["up_diff"] = ", ".join(f"`{f}`" for f in ups if f not in same)
for tag, fn in [("vitm", "eeg_vjepa_ViT-M_4×30×4.pth.tar"), ("vitb", "eeg_vjepa_ViT-B_4×30×2.pth.tar")]:
    h = hashlib.sha256(open(f"{ROOT}/pretrained/{fn}", "rb").read()).hexdigest()
    N[f"sha_{tag}"] = f"`{h[:16]}…`"

# ───────────────────────── 2. Fine-tune vs from scratch (bf16) ─────────────────────────
FT = {}
for kind in ("ft", "scratch"):
    fin, best, late, allmaj, cur = [], [], [], [], []
    for s in range(5):
        d = f"{R}/tuab_finetune/{kind}_vitm_s{s}_bf16"
        z, c = np.load(f"{d}/final_preds.npz"), pd.read_csv(f"{d}/probe_r0.csv")
        a = auc(z["label"], z["prob"][:, 1]); assert abs(a - c.val_auroc.iloc[-1]) < 1e-3, d
        fin.append(a); best.append(c.val_auroc.max()); late.append(c.val_auroc.iloc[100:].mean())
        allmaj.append(int((z["prob_abnormal_by_epoch"] < .5).all(1).sum()))   # epochs predicting every eval rec "normal"
        cur.append(c.val_auroc.values)
    FT[kind] = dict(fin=np.array(fin), best=np.array(best), late=np.array(late), maj=allmaj, cur=np.array(cur))
dd = FT["ft"]["fin"] - FT["scratch"]["fin"]
N["ft_fin"], N["sc_fin"] = ms(FT["ft"]["fin"]), ms(FT["scratch"]["fin"])
N["ft_best"], N["sc_best"] = ms(FT["ft"]["best"]), ms(FT["scratch"]["best"])
N["ft_delta"], N["ft_wins"] = f"{dd.mean():+.3f}", f"{(dd > 0).sum()}/5"
N["ft_t_p"] = f"{stats.ttest_rel(FT['ft']['fin'], FT['scratch']['fin']).pvalue:.2f}"
N["ft_w_p"] = f"{stats.wilcoxon(FT['ft']['fin'], FT['scratch']['fin']).pvalue:.2f}"
N["ft_max"] = f3(max(FT["ft"]["fin"].max(), FT["scratch"]["fin"].max()))
N["ft_maj_max"] = str(max(FT["ft"]["maj"] + FT["scratch"]["maj"]))
T["ft"] = table(["Run"] + [f"s{i}" for i in range(5)] + ["Mean ± sd"],
                [["Fine-tune (released start), final AUROC"] + [f3(x) for x in FT["ft"]["fin"]] + [f"**{N['ft_fin']}**"],
                 ["From scratch (random start), final AUROC"] + [f3(x) for x in FT["scratch"]["fin"]] + [f"**{N['sc_fin']}**"],
                 ["Difference (fine-tune − scratch)"] + [f"{x:+.3f}" for x in dd] + [N["ft_delta"]],
                 ["Fine-tune, best-epoch AUROC*"] + [f3(x) for x in FT["ft"]["best"]] + [N["ft_best"]],
                 ["From scratch, best-epoch AUROC*"] + [f3(x) for x in FT["scratch"]["best"]] + [N["sc_best"]],
                 ["Fine-tune: epochs with every eval recording called normal"] + FT["ft"]["maj"] + ["—"],
                 ["Scratch: epochs with every eval recording called normal"] + FT["scratch"]["maj"] + ["—"]])

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1, 2.2]))
for s in range(5):
    ax[0].plot([0, 1], [FT["scratch"]["fin"][s], FT["ft"]["fin"][s]], "-o", c="#555", ms=4, lw=1)
    ax[0].text(1.06, FT["ft"]["fin"][s], f"s{s}", fontsize=7, va="center")
ax[0].axhline(PAPER["ft"], ls="--", c="k", lw=1); ax[0].text(.5, PAPER["ft"] + .006, "paper FT 0.885", ha="center", fontsize=8)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["from\nscratch", "fine-tune\n(released)"]); ax[0].set_xlim(-.3, 1.3)
ax[0].set_ylim(.5, .92); ax[0].set_ylabel("TUAB eval AUROC (final epoch)"); ax[0].set_title("Paired by subset")
for kind, c in [("ft", COL["rel"]), ("scratch", COL["rnd"])]:
    for s, cu in enumerate(FT[kind]["cur"]):
        ax[1].plot(ep, cu, c=c, lw=.7, alpha=.8, label=("fine-tune" if kind == "ft" else "from scratch") if s == 0 else None)
ax[1].axhline(PAPER["ft"], ls="--", c="k", lw=1); ax[1].set_ylim(.35, .95)
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Eval AUROC"); ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_title("All 10 runs over 500 epochs")
save("ft")

# ───────────────────────── 3. Collapse ─────────────────────────
VL = J(f"{A}/tuab_vitm_linear/results.json")["collapse"]
PB = J(f"{A}/phaseB/collapse_primary.json")
fe = {k: np.load(f"{A}/tuab_vitm_linear/feats_{k}_eval.npz") for k in ("released", "rand0", "rand1", "rand2")}
def pcos(X): Xn = X / np.linalg.norm(X, axis=1, keepdims=True); return (Xn @ Xn.T)[np.triu_indices(len(X), 1)]
for k, z in fe.items():   # recomputed == stored
    key = "released" if k == "released" else k
    assert abs(pcos(z["X"]).mean() - VL[key]["rec_cos"]) < 1e-4 and abs(z["token_std"].mean() - VL[key]["token_std"]) < 1e-6
eb = np.load(f"{A}/phaseB/emb_primary.npz")
for k in ("vitm_released", "vitm_rand0"): assert abs(pcos(eb[k]).mean() - PB[k]["rec_cos"]) < 1e-4
rt = [VL[f"rand{i}"]["token_std"] for i in range(3)]; rs = [PB[f"vitm_rand{i}"]["token_std"] for i in range(3)]
N["col_t_rel"], N["col_t_rnd"] = f"{VL['released']['token_std']:.4f}", f"{np.mean(rt):.3f}"
N["col_t_cos_rel"], N["col_t_cos_rnd"] = f"{VL['released']['rec_cos']:.6f}", f"{np.mean([VL[f'rand{i}']['rec_cos'] for i in range(3)]):.3f}"
N["col_s_rel"], N["col_s_rnd"] = f"{PB['vitm_released']['token_std']:.4f}", f"{np.mean(rs):.3f}"
N["col_s_cos_rel"], N["col_s_cos_rnd"] = f"{PB['vitm_released']['rec_cos']:.6f}", f"{np.mean([PB[f'vitm_rand{i}']['rec_cos'] for i in range(3)]):.4f}"
N["col_ratio_t"] = f"{np.mean(rt) / VL['released']['token_std']:.0f}"
N["col_ratio_s"] = f"{np.mean(rs) / PB['vitm_released']['token_std']:.0f}"
N["col_rel_tmax"] = f"{fe['released']['token_std'].max():.4f}"; N["col_rnd_tmin"] = f"{fe['rand0']['token_std'].min():.3f}"
T["collapse"] = table(["Data set", "Encoder", "Token std (mean)", "Mean cosine between recordings", "Recordings"],
                      [["TUAB eval", "Released ViT-M", f"**{N['col_t_rel']}**", f"**{N['col_t_cos_rel']}**", 276],
                       ["TUAB eval", "Random-init ViT-M (3 seeds)", N["col_t_rnd"], N["col_t_cos_rnd"], 276],
                       ["Stroke (primary)", "Released ViT-M", f"**{N['col_s_rel']}**", f"**{N['col_s_cos_rel']}**", 15],
                       ["Stroke (primary)", "Random-init ViT-M (3 seeds)", N["col_s_rnd"], N["col_s_cos_rnd"], 15]])

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
b = np.logspace(-3, 0.2, 50)
ax[0].hist(fe["released"]["token_std"], bins=b, color=COL["rel"], label="released"); ax[0].hist(fe["rand0"]["token_std"], bins=b, color=COL["rnd"], label="random-init")
ax[0].set_xscale("log"); ax[0].set_xlabel("token std of one recording (log)"); ax[0].set_ylabel("recordings"); ax[0].legend(fontsize=7)
ax[0].set_title("(a) TUAB: token spread", fontsize=9)
bc = np.linspace(.9, 1, 60)
ax[1].hist(pcos(fe["rand0"]["X"]), bins=bc, color=COL["rnd"], density=True, label="random-init")
ax[1].hist(pcos(fe["released"]["X"]), bins=bc, color=COL["rel"], density=True, label="released")
ax[1].set_xlabel("cosine between two recordings"); ax[1].set_yticks([]); ax[1].legend(fontsize=7, loc="upper left")
ax[1].set_title("(b) TUAB: all 37,950 pairs", fontsize=9)
x = np.arange(2); w = .35
ax[2].bar(x - w / 2, [VL["released"]["token_std"], PB["vitm_released"]["token_std"]], w, color=COL["rel"], label="released")
ax[2].bar(x + w / 2, [np.mean(rt), np.mean(rs)], w, color=COL["rnd"], label="random-init")
ax[2].set_yscale("log"); ax[2].set_xticks(x); ax[2].set_xticklabels(["TUAB", "stroke"]); ax[2].set_ylabel("token std (log)")
ax[2].set_ylim(top=8); ax[2].legend(fontsize=7, ncol=2, loc="upper center"); ax[2].set_title("(c) both data sets", fontsize=9)
save("collapse")

# ───────────────────────── 4. Linear probe on ViT-M (+ band power) ─────────────────────────
VLr = J(f"{A}/tuab_vitm_linear/results.json")["runs"]; BP = J(f"{A}/p13_bandpower/results.json")
lp = dict(rel=[VLr[f"released_s{s}"]["auroc"] for s in range(5)], rnd=[VLr[f"rand_s{s}"]["auroc"] for s in range(3)],
          bp=[BP[f"s{s}"]["auroc"] for s in range(5)])
for k, v in lp.items(): N[f"lp_{k}"] = ms(v)
zr, zb = np.load(f"{A}/tuab_vitm_linear/preds_released_full_train.npz"), np.load(f"{A}/p13_bandpower/preds_full_train.npz")
assert sorted(map(str, zr["rec"])) == sorted(os.path.basename(str(a)) for a in zb["rec"])   # same 276 eval recs (different order)
yT = zr["label"]
N["lp_rel_full"], N["lp_bp_full"] = f3(auc(yT, zr["prob_abnormal"])), f3(auc(zb["label"], zb["prob_abnormal"]))
mrel = metrics(yT, zr["prob_abnormal"]); N["lp_rel_full_bacc"] = f3(mrel["bal_acc"])
N["lp_rel_full_ci"] = ci(bootstrap_ci(yT, zr["prob_abnormal"]))
N["lp_rel_rnd_p"] = pv(stats.mannwhitneyu(lp["rel"], lp["rnd"], alternative="greater").pvalue)
N["lp_short"] = f3(PAPER["vitm"] - auc(yT, zr["prob_abnormal"]))
N["lp_vs_p14"] = f3(np.mean(lp["rel"]) - np.mean(P14["ViT-M, §4.2 head"]["fin"]))
T["lp"] = table(["Features", "Subsets (546 recordings each)", "Mean ± sd", "Full train (2,717)"],
                [["Band power (hand-crafted)", " / ".join(f3(x) for x in lp["bp"]), N["lp_bp"], N["lp_bp_full"]],
                 ["Random-init ViT-M", " / ".join(f3(x) for x in lp["rnd"]), N["lp_rnd"], "—"],
                 ["Released ViT-M", " / ".join(f3(x) for x in lp["rel"]), f"**{N['lp_rel']}**", f"**{N['lp_rel_full']}** {N['lp_rel_full_ci']}"]])

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
for i, (k, lab, c) in enumerate([("bp", "band power", COL["bp"]), ("rnd", "random ViT-M", COL["rnd"]), ("rel", "released ViT-M", COL["rel"])]):
    ax[0].bar(i, np.mean(lp[k]), color=c, alpha=.35, width=.6); ax[0].scatter([i] * len(lp[k]), lp[k], c=c, s=14, zorder=3)
ax[0].scatter([0, 2], [auc(zb["label"], zb["prob_abnormal"]), auc(yT, zr["prob_abnormal"])], marker="D", c="k", s=22, zorder=4, label="full train")
ax[0].axhline(PAPER["vitm"], ls="--", c="k", lw=1); ax[0].text(2.4, PAPER["vitm"] + .004, "paper 0.877", ha="right", fontsize=8)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["band\npower", "random\nViT-M", "released\nViT-M"]); ax[0].set_ylim(.65, .9)
ax[0].set_ylabel("TUAB eval AUROC"); ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_title("Same linear probe; dots = subsets")
for z, lab, c in [(zr, "released ViT-M", COL["rel"]), (zb, "band power", COL["bp"])]:
    fp, tp, _ = roc_curve(z["label"], z["prob_abnormal"]); ax[1].plot(fp, tp, c=c, label=f"{lab} ({auc(z['label'], z['prob_abnormal']):.3f})")
ax[1].plot([0, 1], [0, 1], ":", c="grey"); ax[1].set_xlabel("False-positive rate"); ax[1].set_ylabel("True-positive rate")
ax[1].legend(fontsize=7, loc="lower right"); ax[1].set_title("ROC, full train set")
save("linear_probe")

# ───────────────────────── 5. FEI+C on NMT (A0) ─────────────────────────
A0 = J(f"{A}/a0_feic_nmt/results.json")
arms0 = ["FEI@4000", "C", "FEI+C"]
fold = {a: [A0[str(k)][a]["auroc"] for k in range(5)] for a in arms0}
for a, key in zip(arms0, ["fei", "c", "feic"]):
    N[f"a0_{key}"] = ms(fold[a]); N[f"a0_{key}_bacc"] = f3(A0["summary"][a]["bal_acc"][0]); N[f"a0_{key}_pr"] = f3(A0["summary"][a]["auc_pr"][0])
    assert abs(np.mean(fold[a]) - A0["summary"][a]["auroc"][0]) < 1e-9
for a, key in [("FEI@4000", "fei"), ("C", "c")]:
    d = np.array(fold["FEI+C"]) - np.array(fold[a]); N[f"a0_d_{key}"] = f"{d.mean():+.3f}"; N[f"a0_w_{key}"] = f"{(d > 0).sum()}/5"
gate = [(A0[str(k)]["gate_fei_L1000_auroc"], A0[str(k)]["gate_ref"]) for k in range(5)]
N["a0_gate_maxdev"] = f"{max(abs(a - b) for a, b in gate):.1e}"
T["a0_gate"] = table(["Fold", "Re-computed now", "Recorded in Act-1", "Absolute difference"],
                     [[f"f{k}", f"{a:.4f}", f"{b:.4f}", f"{abs(a - b):.1e}"] for k, (a, b) in enumerate(gate)])
T["a0"] = table(["Arm", "f0", "f1", "f2", "f3", "f4", "AUROC mean ± sd", "BAcc", "AUC-PR"],
                [[lab] + [f3(x) for x in fold[a]] + [("**%s**" if a == "FEI+C" else "%s") % N[f"a0_{k}"], N[f"a0_{k}_bacc"], N[f"a0_{k}_pr"]]
                 for a, lab, k in [("FEI@4000", "FEI", "fei"), ("C", "C", "c"), ("FEI+C", "FEI+C", "feic")]])
pz = [np.load(f"{A}/a0_feic_nmt/preds_f{k}.npz") for k in range(5)]
N["a0_n"] = str(sum(len(z["label"]) for z in pz)); N["a0_npos"] = str(sum(int(z["label"].sum()) for z in pz))
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
for k in range(5): ax[0].plot(range(3), [fold[a][k] for a in arms0], "-o", c="#555", ms=4, lw=1)
ax[0].plot(range(3), [np.mean(fold[a]) for a in arms0], "-s", c=COL["feic"], lw=2.5, ms=6, label="mean of 5 folds")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["FEI", "C", "FEI+C"]); ax[0].set_ylabel("NMT test-fold AUROC")
ax[0].legend(fontsize=7); ax[0].set_title("Each grey line = one test fold")
for a, lab, c in [("FEI@4000", "FEI", COL["fei"]), ("C", "C", COL["c"]), ("FEI+C", "FEI+C", COL["feic"])]:
    y = np.concatenate([z["label"] for z in pz]); p = np.concatenate([z[a] for z in pz])
    fp, tp, _ = roc_curve(y, p); ax[1].plot(fp, tp, c=c, label=f"{lab} (pooled {auc(y, p):.3f})")
ax[1].plot([0, 1], [0, 1], ":", c="grey"); ax[1].set_xlabel("False-positive rate"); ax[1].set_ylabel("True-positive rate")
ax[1].legend(fontsize=7, loc="lower right"); ax[1].set_title(f"ROC, all {N['a0_n']} test predictions pooled")
save("nmt_a0")

# ───────────────────────── 6. FEI+C on TUAB vs ViT-M (same probe) ─────────────────────────
FN, FJ = J(f"{A}/tuab_feic/results.json"), J(f"{A}/tuab_feic_joint/results.json")
SRC = {"FEI": ("tuab_feic", FN), "rand-FEI": ("tuab_feic", FN), "C": ("tuab_feic_joint", FJ), "rand-C": ("tuab_feic_joint", FJ),
       "FEI+C": ("tuab_feic_joint", FJ), "rand-FEI+C": ("tuab_feic_joint", FJ)}
for a in ("rand-FEI", "rand-C", "rand-FEI+C"):     # random-init arms do not depend on the pretraining corpus
    for s in ["s0", "s1", "s2", "s3", "s4", "full_train"]:
        assert FN["runs"][f"{s}/{a}"]["auroc"] == FJ["runs"][f"{s}/{a}"]["auroc"]
TF = {}
for a, (d, r) in SRC.items():
    z = np.load(f"{A}/{d}/preds_full_train_{a}.npz")
    TF[a] = dict(sub=[r["runs"][f"s{s}/{a}"]["auroc"] for s in range(5)], full=auc(z["label"], z["prob_abnormal"]),
                 mem=[auc(z["label"], m) for m in z["members"]], p=z["prob_abnormal"], y=z["label"], rec=z["rec"],
                 m=r["runs"][f"full_train/{a}"])
    assert abs(TF[a]["full"] - r["runs"][f"full_train/{a}"]["auroc"]) < 1e-9
o_ = {r: i for i, r in enumerate(TF["FEI"]["rec"])}; jx = [o_[r] for r in zr["rec"]]   # re-order FEI+C preds to ViT-M's order, as tuab_vitm_linear did
assert all((TF[a]["rec"] == TF["FEI"]["rec"]).all() for a in TF) and (TF["FEI"]["y"][jx] == yT).all()
for a in TF: TF[a]["p"] = TF[a]["p"][jx]
yT_, pV = yT, zr["prob_abnormal"]
LAB6 = {"FEI": "FEI (NMT-pretrained)", "rand-FEI": "rand-FEI", "C": "C (NMT+TUAB-train pretrained)", "rand-C": "rand-C",
        "FEI+C": "FEI+C (NMT+TUAB-train pretrained)", "rand-FEI+C": "rand-FEI+C"}
rows = [["Band power (hand-crafted)", N["lp_bp"], N["lp_bp_full"], "—"], ["Released ViT-M", N["lp_rel"], N["lp_rel_full"], "—"]]
for a in SRC:
    b = "**%s**" if a == "FEI" else "%s"
    rows.append([LAB6[a], ms(TF[a]["sub"]), b % f3(TF[a]["full"]), ms(TF[a]["mem"])])
    N[f"t6_{a}_full"] = f3(TF[a]["full"]); N[f"t6_{a}_sub"] = ms(TF[a]["sub"]); N[f"t6_{a}_mem"] = ms(TF[a]["mem"])
T["t6"] = table(["Features (all: same linear probe)", "Subsets s0–s4", "Full train (2,717)", "Full train, single encoders"], rows)
for a in ("FEI", "FEI+C"):
    c_, p_ = paired_boot(yT_, TF[a]["p"], pV)
    N[f"t6_{a}_vs_vit"] = f"{TF[a]['full'] - auc(yT_, pV):+.3f}"
    N[f"t6_{a}_vs_vit_ci"], N[f"t6_{a}_vs_vit_p"] = ci(c_), pv(p_)
for a, b in [("FEI", "rand-FEI"), ("FEI+C", "rand-FEI+C"), ("C", "rand-C")]:
    c_, p_ = paired_boot(yT_, TF[a]["p"], TF[b]["p"])
    N[f"t6_{a}_vs_rand"] = f"{TF[a]['full'] - TF[b]['full']:+.3f}"; N[f"t6_{a}_vs_rand_ci"], N[f"t6_{a}_vs_rand_p"] = ci(c_), pv(p_)
    N[f"t6_{a}_vs_rand_mem"] = f"{np.mean(TF[a]['mem']) - np.mean(TF[b]['mem']):+.3f}"
pj = J(f"{A}/tuab_vitm_linear_joint/results.json")["paired_vs_feic"]["FEI+C - ViT-M (full_train)"]   # stored by the original run
assert abs(TF["FEI+C"]["full"] - auc(yT, pV) - pj["delta"]) < 1e-9 and np.allclose(paired_boot(yT, TF["FEI+C"]["p"], pV)[0], pj["ci"])
d6 = np.array(TF["FEI"]["sub"]) - np.array(lp["rel"]); N["t6_FEI_vs_vit_sub"], N["t6_FEI_vs_vit_subw"] = f"{d6.mean():+.3f}", f"{(d6 > 0).sum()}/5"
d6 = np.array(TF["FEI+C"]["sub"]) - np.array(lp["rel"]); N["t6_FEIC_vs_vit_sub"], N["t6_FEIC_vs_vit_subw"] = f"{d6.mean():+.3f}", f"{(d6 > 0).sum()}/5"
mf = TF["FEI"]["m"]; N["t6_fei_bacc"], N["t6_fei_sens"], N["t6_fei_spec"] = f3(mf["bal_acc"]), f3(mf["sens"]), f3(mf["spec"])
CN, CJ = J(f"{A}/tuab_feic/collapse.json"), J(f"{A}/tuab_feic_joint/collapse.json")
rng_ = lambda d, p, k: f"{min(d[f'{p}_f{i}'][k] for i in range(5)):.2f}–{max(d[f'{p}_f{i}'][k] for i in range(5)):.2f}"
T["t6_col"] = table(["Encoder", "Embedding std across recordings", "Mean cosine between recordings"],
                    [["FEI (NMT-pretrained), 5 fold encoders", rng_(CN, "fei", "rec_std"), rng_(CN, "fei", "rec_cos")],
                     ["C (NMT+TUAB-train pretrained), 5 fold encoders", rng_(CJ, "c", "rec_std"), rng_(CJ, "c", "rec_cos")],
                     ["Released ViT-M (for comparison)", f"{VL['released']['rec_std']:.4f}", f"{VL['released']['rec_cos']:.6f}"]])

fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
order = [("bp", "band\npower", COL["bp"]), ("vit", "released\nViT-M", COL["rel"]), ("rand-FEI", "rand-\nFEI", COL["rnd"]),
         ("FEI", "FEI", COL["fei"]), ("rand-C", "rand-\nC", COL["rnd"]), ("C", "C", COL["c"]),
         ("rand-FEI+C", "rand-\nFEI+C", COL["rnd"]), ("FEI+C", "FEI+C", COL["feic"])]
for i, (k, lab, c) in enumerate(order):
    sub, full = (lp["bp"], auc(zb["label"], zb["prob_abnormal"])) if k == "bp" else (lp["rel"], auc(yT_, pV)) if k == "vit" else (TF[k]["sub"], TF[k]["full"])
    ax[0].bar(i, full, color=c, alpha=.35, width=.65); ax[0].scatter([i] * 5, sub, c=c, s=10, zorder=3)
ax[0].axhline(PAPER["vitm"], ls="--", c="k", lw=1); ax[0].text(7.4, PAPER["vitm"] + .003, "paper ViT-M 0.877", ha="right", fontsize=7)
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([o[1] for o in order], fontsize=7); ax[0].set_ylim(.68, .9)
ax[0].set_ylabel("TUAB eval AUROC"); ax[0].set_title("Bars = full train; dots = the 5 subsets", fontsize=9)
for k, lab, c in [("FEI", "FEI", COL["fei"]), ("FEI+C", "FEI+C", COL["feic"])]:
    fp, tp, _ = roc_curve(yT_, TF[k]["p"]); ax[1].plot(fp, tp, c=c, label=f"{lab} ({TF[k]['full']:.3f})")
fp, tp, _ = roc_curve(yT_, pV); ax[1].plot(fp, tp, c=COL["rel"], label=f"released ViT-M ({auc(yT_, pV):.3f})")
ax[1].plot([0, 1], [0, 1], ":", c="grey"); ax[1].set_xlabel("False-positive rate"); ax[1].set_ylabel("True-positive rate")
ax[1].legend(fontsize=7, loc="lower right"); ax[1].set_title("ROC, full train set", fontsize=9)
save("tuab_feic")

# ───────────────────────── 7. Stroke: gate + pre-registered LOSO ─────────────────────────
G = {c: J(f"{A}/phaseB/gate_{c}.json") for c in ("primary", "sensitivity")}
FEAT = [("bsi_1_30", "BSI 1–30 Hz"), ("bsi_delta", "BSI delta"), ("bsi_theta", "BSI theta"), ("bsi_alpha", "BSI alpha"),
        ("bsi_beta", "BSI beta"), ("rel_delta", "rel. delta"), ("rel_theta", "rel. theta"), ("rel_alpha", "rel. alpha"),
        ("rel_beta", "rel. beta"), ("dar", "DAR"), ("dtabr", "DTABR")]
arrow = lambda t: "↑" if t["direction"] > 0 else "↓"
T["gate"] = table(["Feature", "Stroke vs control", "Primary: median stroke / control", "Primary p", "Sensitivity p"],
                  [[lab, arrow(G["primary"]["tests"][k]),
                    f"{G['primary']['tests'][k]['stroke_median']:.3f} / {G['primary']['tests'][k]['control_median']:.3f}",
                    pv(G["primary"]["tests"][k]["p"]), pv(G["sensitivity"]["tests"][k]["p"])] for k, lab in FEAT])
for c in G: N[f"gate_{c}_p"] = pv(G[c]["tests"]["bsi_1_30"]["p"]); N[f"gate_{c}_pass"] = "PASS" if G[c]["gate_pass"] else "FAIL"
N["gate_n_sig"] = str(sum(G["primary"]["tests"][k]["p"] < .05 for k, _ in FEAT)); N["gate_n_feat"] = str(len(FEAT))
N["gate_rest_s"] = f"{G['primary']['rest_s']:.0f}"
RB = {c: J(f"{A}/phaseB/results_{c}.json") for c in ("primary", "sensitivity")}
OB = {c: np.load(f"{A}/phaseB/oof_{c}.npz") for c in ("primary", "sensitivity")}
ARMS7 = ["BSI"]   # stroke part reports no EEG-VJEPA arms
for c in RB:
    T[f"loso_{c}"] = table(["Arm", "LOSO AUROC", "95 % CI", "Permutation p", "Sens", "Spec"],
                           [[a, ("**%s**" % f3(RB[c]["arms"][a]["auroc"])), ci(RB[c]["arms"][a]["auroc_ci"]), pv(RB[c]["arms"][a]["perm_p"]),
                             f3(RB[c]["arms"][a]["sens"]), f3(RB[c]["arms"][a]["spec"])] for a in ARMS7])
    for a in ARMS7:
        k = a.replace("+", "p").replace("-", "_"); N[f"b_{c}_{k}"] = f3(RB[c]["arms"][a]["auroc"])
        N[f"b_{c}_{k}_ci"], N[f"b_{c}_{k}_p"] = ci(RB[c]["arms"][a]["auroc_ci"]), pv(RB[c]["arms"][a]["perm_p"])
        nul = OB[c][f"{a}__null_auroc"]; N[f"b_{c}_{k}_nullmean"] = f3(nul.mean())

sub = G["primary"]["subjects"]; ids = list(sub); ys = np.array([sub[s]["y"] for s in ids])
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1, 1.7]))
for yv, c, lab in [(1, COL["rel"], "stroke"), (0, COL["fei"], "control")]:
    v = [sub[s]["bsi_1_30"] for s in ids if sub[s]["y"] == yv]
    ax[0].scatter(np.full(len(v), 1 - yv) + np.linspace(-.12, .12, len(v)), v, c=c, s=18); ax[0].hlines(np.median(v), .7 - yv + .0, 1.3 - yv, colors=c)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["stroke (9)", "control (6)"]); ax[0].set_ylabel("broadband BSI (1–30 Hz)")
ax[0].set_title(f"(a) Primary cohort, p = {N['gate_primary_p']}", fontsize=9)
x = np.arange(len(FEAT)); w = .38
for j, (c, col) in enumerate([("primary", "#34495e"), ("sensitivity", "#95a5a6")]):
    ax[1].bar(x + (j - .5) * w, [-np.log10(G[c]["tests"][k]["p"]) for k, _ in FEAT], w, color=col, label=c)
ax[1].axhline(-np.log10(.05), ls="--", c="k", lw=1, label="p = 0.05")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{l} {arrow(G['primary']['tests'][k])}" for k, l in FEAT], rotation=60, ha="right", fontsize=7)
ax[1].set_ylabel("−log10 p (Mann-Whitney)"); ax[1].legend(fontsize=7); ax[1].set_title("(b) All gate features, both cohorts", fontsize=9)
save("stroke_gate")

RC = J(f"{A}/phaseB_clean/results_primary.json")
assert RC["preds"]["subject"] == RB["primary"]["preds"]["subject"] and RC["preds"]["BSI"] == RB["primary"]["preds"]["BSI"]
PR = [("BSI", RB["primary"]["preds"]["BSI"], COL["bsi"]), ("FEI+C", RC["preds"]["FEI+C"], COL["feic"]),
      ("rand-FEI+C", RC["preds"]["rand-FEI+C"], "#b2babb")]
yP = np.array(RB["primary"]["preds"]["y"]); subj = RB["primary"]["preds"]["subject"]
fig, ax = plt.subplots(figsize=(7.4, 2.2))
for i, (lab, p, c) in enumerate(PR):
    p = np.array(p)
    ax.scatter(p[yP == 1], np.full((yP == 1).sum(), i) + .12, marker="o", c=COL["rel"], s=22, label="stroke" if i == 0 else None)
    ax.scatter(p[yP == 0], np.full((yP == 0).sum(), i) - .12, marker="s", c=COL["fei"], s=22, label="control" if i == 0 else None)
    ax.text(1.03, i, f"AUROC {auc(yP, p):.3f}", va="center", fontsize=8)
ax.axvline(.5, ls=":", c="grey"); ax.set_yticks(range(len(PR))); ax.set_yticklabels([p[0] for p in PR]); ax.set_xlim(-.02, 1.2)
ax.set_xlabel("held-out P(stroke) for each subject (leave-one-subject-out)"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.42), ncol=2)
ax.set_title("Primary cohort: each dot is one subject predicted by a model that never saw them", fontsize=9)
save("stroke_subjects")

fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.5)); ax = [ax]
for j, a in enumerate(["BSI"]):
    nul = OB["primary"][f"{a}__null_auroc"]; obs = RB["primary"]["arms"][a]["auroc"]
    ax[j].hist(nul, bins=np.linspace(0, 1, 31), color="#bbb"); ax[j].axvline(obs, c=COL["rel"], lw=2)
    ax[j].set_title(f"{a}: {obs:.3f}, p = {pv(RB['primary']['arms'][a]['perm_p'])}", fontsize=8); ax[j].set_xlabel("LOSO AUROC")
ax[0].set_ylabel("shuffled-label runs (of 1000)")
save("stroke_perm")

# ───────────────────────── 8. Clean FEI+C prep + CV robustness ─────────────────────────
KC = np.load(f"{A}/phaseB_clean/kf5_lpo_primary.npz"); KV = np.load(f"{A}/phaseB_cv/kf5_lpo_primary.npz")
for a in ("BSI", "VJEPA"): assert np.allclose(KC[f"kf5__{a}"], KV[f"kf5__{a}"])
RV = J(f"{A}/phaseB_cv/results_primary.json")["arms"]
AR8 = ["BSI", "rand-FEI+C", "FEI+C", "FEI+C+BSI"]
def arm8(a):
    if a in ("BSI", "VJEPA"): return dict(loso=RV[a]["loso"], kf=KC[f"kf5__{a}"], lpo=RV[a]["lpo"], perm=RB["primary"]["arms"][a]["perm_p"])
    r = RC["arms"][a]; return dict(loso=r["auroc"], kf=KC[f"kf5__{a}"], lpo=r["lpo"], perm=r.get("perm_p"))
S8 = {a: arm8(a) for a in AR8}
for a in AR8: assert abs(S8[a]["kf"].mean() - (RV[a]["kf5_mean"] if a in RV and a in ("BSI", "VJEPA") else RC["arms"][a]["kf5_mean"])) < 1e-9
T["clean"] = table(["Arm", "LOSO", "5-fold ×100: mean ± sd [2.5–97.5 %]", "Leave-pair-out", "Permutation p (LOSO)"],
                   [[("**%s**" if a == "FEI+C" else "%s") % a, f3(S8[a]["loso"]),
                     f"{S8[a]['kf'].mean():.3f} ± {S8[a]['kf'].std():.3f} [{np.percentile(S8[a]['kf'], 2.5):.3f}, {np.percentile(S8[a]['kf'], 97.5):.3f}]",
                     f3(S8[a]["lpo"]), pv(S8[a]["perm"]) if S8[a]["perm"] is not None else "—"] for a in AR8])
for a in AR8:
    k = a.replace("+", "p").replace("-", "_"); N[f"c_{k}"], N[f"c_{k}_kf"], N[f"c_{k}_lpo"] = f3(S8[a]["loso"]), f3(S8[a]["kf"].mean()), f3(S8[a]["lpo"])
fc = RC["arms"]["FEI+C"]; N["c_FEIpC_ci"], N["c_FEIpC_p"] = ci(fc["auroc_ci"]), pv(fc["perm_p"])
N["c_FEIpC_op"] = f"Acc {fc['acc']:.3f}, BAcc {fc['bal_acc']:.3f}, Sens {fc['sens']:.3f}, Spec {fc['spec']:.3f} (TP {fc['tp']}, FP {fc['fp']}, TN {fc['tn']}, FN {fc['fn']})"
N["c_rand_p"] = pv(RC["arms"]["rand-FEI+C"]["perm_p"])
PAIRS8 = ["FEI+C vs rand-FEI+C", "FEI+C vs BSI", "FEI+C+BSI vs BSI"]
rows = []
for pr in PAIRS8:
    q = RC["paired"][pr]; k = pr.replace(" vs ", "__").replace("+", "p").replace("-", "_")
    N[f"cp_{k}"], N[f"cp_{k}_ci"], N[f"cp_{k}_p"] = f"{q['delta']:+.3f}", ci(q["ci"]), pv(q["p_le0"])
    N[f"cp_{k}_kf"], N[f"cp_{k}_kfw"] = f"{q['kf5_delta']:+.3f}", f"{q['kf5_frac_a_better'] * 100:.0f} %"
    N[f"cp_{k}_lpo"], N[f"cp_{k}_lpo_ci"], N[f"cp_{k}_lpo_p"] = f"{q['lpo']['delta']:+.3f}", ci(q["lpo"]["ci"]), pv(q["lpo"]["p_le0"])
    rows.append([pr, f"{N[f'cp_{k}']} {N[f'cp_{k}_ci']}, p = {N[f'cp_{k}_p']}", f"{N[f'cp_{k}_kf']} ({N[f'cp_{k}_kfw']})",
                 f"{N[f'cp_{k}_lpo']} {N[f'cp_{k}_lpo_ci']}, p = {N[f'cp_{k}_lpo_p']}"])
T["clean_pairs"] = table(["Comparison (A vs B)", "LOSO Δ [95 % CI], p(Δ ≤ 0)", "5-fold ×100 Δ (A better in % of repeats)",
                          "Leave-pair-out Δ [95 % CI], p(Δ ≤ 0)"], rows)
EX = ["FEI", "rand-FEI", "C", "rand-C"]
T["clean_branch"] = table(["Branch (exploratory)", "LOSO", "5-fold ×100 mean ± sd", "Leave-pair-out"],
                          [[a, f3(RC["arms"][a]["auroc"]), f"{RC['arms'][a]['kf5_mean']:.3f} ± {RC['arms'][a]['kf5_sd']:.3f}", f3(RC["arms"][a]["lpo"])] for a in EX])
for pr in ("FEI vs rand-FEI", "C vs rand-C"):
    q = RC["paired"][pr]; k = pr.split(" ")[0]
    N[f"cb_{k}_kf"], N[f"cb_{k}_kfw"] = f"{q['kf5_delta']:+.3f}", f"{q['kf5_frac_a_better'] * 100:.0f} %"
    N[f"cb_{k}_lpo"], N[f"cb_{k}_lpo_ci"], N[f"cb_{k}_lpo_p"] = f"{q['lpo']['delta']:+.3f}", ci(q["lpo"]["ci"]), pv(q["lpo"]["p_le0"])
for a in EX: N[f"cb_{a.replace('-', '_')}"] = f3(RC["arms"][a]["auroc"])

PP, CC = J(f"{A}/phaseB_clean/prep_primary.json"), J(f"{A}/phaseB_clean/collapse_primary.json")
g = CC["gate"]; N["g_psd"], N["g_psd_mean"] = f"{PP['matched']['max_abs_dev_0p5_99']:.3f}", f"{PP['matched']['mean_abs_dev_0p5_99']:.3f}"
N["g_bnz"], N["g_cstd"], N["g_const"] = f"{g['bnz_max']:.2f}", f"{g['c_std_ratio_min']:.2f}", f"{g['c_const_frac_max'] * 100:.0f} %"
N["g_bnz_nmt"] = f"{max(np.abs(CC[f'c_f{i}']['bnz_nmt']).max() for i in range(5)):.2f}"
th = g["thresholds"]; N["g_th_psd"], N["g_th_bnz"], N["g_th_c"], N["g_th_const"] = str(th["psd_max_dev"]), str(th["bnz_max"]), str(th["c_std_ratio_min"]), f"{th['c_const_frac_max'] * 100:.0f} %"
cs = [CC[f"c_f{i}"]["rec_std"] for i in range(5)]; cn = [CC[f"c_f{i}"]["nmt"]["rec_std"] for i in range(5)]; cr = [CC[f"c_rand{i}"]["rec_std"] for i in range(3)]
N["g_c_stroke"], N["g_c_nmt"], N["g_c_rand"] = f"{min(cs):.2f}–{max(cs):.2f}", f"{min(cn):.2f}–{max(cn):.2f}", f"{min(cr):.3f}–{max(cr):.3f}"
N["g_nref"] = str(PP["n_nmt_ref"])
T["gates"] = table(["Check (clean prep, primary cohort)", "Value", "Threshold (fixed in advance)", "Result"],
                   [["Largest gap between stroke and NMT spectrum, 0.5–99 Hz (log10 units)", N["g_psd"], f"≤ {N['g_th_psd']}", "PASS" if PP["psd_gate_pass"] else "FAIL"],
                    ["Largest Branch-C input z-score after its NMT BatchNorm", N["g_bnz"], f"≤ {N['g_th_bnz']}", "PASS"],
                    ["C embedding spread on stroke ÷ spread on NMT (worst fold)", N["g_cstd"], f"≥ {N['g_th_c']}", "PASS"],
                    ["Constant C embedding dimensions", N["g_const"], f"≤ {N['g_th_const']}", "PASS" if g["pass"] else "FAIL"]])

ref = np.load(f"{A}/phaseB_clean/nmt_ref_psd.npz"); fq, nm = ref["f"], ref["mean"].mean(0)
hz = [float(k[:-2]) for k in PP["matched"]["log10psd"]]
assert np.allclose([nm[np.argmin(abs(fq - h))] for h in hz], list(PP["nmt"]["log10psd"].values()), atol=1e-3)
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.8))
ax[0].plot(fq, nm, c=COL["fei"], label=f"NMT reference ({N['g_nref']} recs)")
ax[0].scatter(hz, list(PP["matched"]["log10psd"].values()), c=COL["rel"], s=16, zorder=3, label="stroke, clean prep")
ax[0].set_xlabel("Hz"); ax[0].set_ylabel("log10 relative power"); ax[0].legend(fontsize=6.5); ax[0].set_title("(a) spectrum matches NMT", fontsize=9)
bins = np.arange(len(CC["c_f0"]["bnz_stroke"]))
for i in range(5):
    ax[1].plot(bins, CC[f"c_f{i}"]["bnz_stroke"], c=COL["rel"], lw=.8, label="stroke" if i == 0 else None)
    ax[1].plot(bins, CC[f"c_f{i}"]["bnz_nmt"], c=COL["fei"], lw=.8, label="NMT" if i == 0 else None)
for s in (1, -1): ax[1].axhline(s * th["bnz_max"], ls="--", c="k", lw=.8)
ax[1].set_xlabel("spectrogram bin (Hz)"); ax[1].set_ylabel("z after C's BatchNorm"); ax[1].legend(fontsize=6.5)
ax[1].set_title("(b) C input stays in range", fontsize=9)
x = np.arange(5); w = .38
ax[2].bar(x - w / 2, cn, w, color=COL["fei"], label="on NMT"); ax[2].bar(x + w / 2, cs, w, color=COL["rel"], label="on stroke")
ax[2].axhline(np.mean(cr), ls=":", c="grey", label="random-init C"); ax[2].set_ylim(0, .5)
ax[2].set_xticks(x); ax[2].set_xticklabels([f"f{i}" for i in range(5)]); ax[2].set_ylabel("C embedding std"); ax[2].legend(fontsize=6.5)
ax[2].set_title("(c) C is not collapsed", fontsize=9)
save("stroke_clean_gates")

fig, ax = plt.subplots(figsize=(6.4, 3.0))
cols8 = [COL["bsi"], COL["rnd"], COL["feic"], "#a04000"]
bp = ax.boxplot([S8[a]["kf"] for a in AR8], widths=.5, patch_artist=True, showfliers=False)
for p_, c in zip(bp["boxes"], cols8): p_.set_facecolor(c); p_.set_alpha(.35)
ax.scatter(np.arange(1, len(AR8) + 1), [S8[a]["loso"] for a in AR8], marker="x", c="k", s=30, zorder=3, label="LOSO")
ax.scatter(np.arange(1, len(AR8) + 1), [S8[a]["lpo"] for a in AR8], marker="^", facecolors="none", edgecolors="k", s=30, zorder=3, label="leave-pair-out")
ax.set_xticks(range(1, len(AR8) + 1)); ax.set_xticklabels(AR8); ax.axhline(.5, ls=":", c="grey"); ax.set_ylim(.45, 1.03)
ax.set_ylabel("AUROC"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("Primary cohort: boxes = 100 repeats of 5-fold CV (the same 100 splits for every arm)", fontsize=9)
save("stroke_clean_cv")

# ───────────────────────── V. t-SNE / UMAP of the saved TUAB eval embeddings (proposal deliverable) ─────────────────────────
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap

VR, VN = np.load(f"{A}/tuab_vitm_linear/feats_released_eval.npz"), np.load(f"{A}/tuab_vitm_linear/feats_rand0_eval.npz")
assert (VR["rec"] == VN["rec"]).all() and (VR["y"] == VN["y"]).all()
JI = np.load(f"{A}/tuab_feic_joint/emb_index.npz"); jev = np.where(JI["split"] == "eval")[0]
o_ = {r: i for i, r in enumerate(JI["rec"][jev])}; jx = jev[[o_[r] for r in VR["rec"]]]     # FEI+C rows in ViT-M's order
assert (JI["label_abnormal"][jx] == VR["y"]).all()
JE = {k: np.load(f"{A}/tuab_feic_joint/emb_{k}.npy") for k in ("fei_f0", "c_f0", "fei_rand0", "c_rand0")}
jc = J(f"{A}/tuab_feic_joint/collapse.json")
for k, X in JE.items(): assert abs(X[jev].std(0).mean() - jc[k]["rec_std"]) < 1e-6   # same eval features as the Exp-6 run
yV = VR["y"]
EMB = {"Released ViT-M": VR["X"], "Random-init ViT-M": VN["X"],
       "FEI+C": np.hstack([JE["fei_f0"], JE["c_f0"]])[jx], "rand-FEI+C": np.hstack([JE["fei_rand0"], JE["c_rand0"]])[jx]}
K = 10
def knn_agree(Z):   # mean share of each recording's K nearest neighbours that carry the same label
    _, nb = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Z)
    return float((yV[nb[:, 1:]] == yV[:, None]).mean())
cnt = np.bincount(yV); N["v_chance"] = f3(float((cnt * (cnt - 1)).sum() / (len(yV) * (len(yV) - 1))))
N["v_n"], N["v_nab"], N["v_k"] = str(len(yV)), str(int(yV.sum())), str(K)
MAP, rows = {}, []
for a, X in EMB.items():
    Xs = StandardScaler().fit_transform(X)                                       # same scaling as the linear probe
    MAP[a] = dict(tsne=TSNE(perplexity=30, init="pca", random_state=0).fit_transform(Xs),
                  umap=umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xs))
    ag = [knn_agree(Xs), knn_agree(MAP[a]["tsne"]), knn_agree(MAP[a]["umap"])]
    key = {"Released ViT-M": "rel", "Random-init ViT-M": "rnd", "FEI+C": "feic", "rand-FEI+C": "rfeic"}[a]
    N[f"v_{key}_orig"], N[f"v_{key}_tsne"], N[f"v_{key}_umap"] = map(f3, ag)
    rows.append([a, X.shape[1]] + [f3(x) for x in ag])
T["vmap"] = table(["Features (TUAB eval, one encoder each)", "Dims", f"{K}-NN agreement: original features", "t-SNE map", "UMAP map"], rows)
fig, ax = plt.subplots(2, 4, figsize=(7.6, 3.4))
for j, a in enumerate(EMB):
    for i, m in enumerate(("tsne", "umap")):
        Z = MAP[a][m]
        for lab, c, nm in ((0, COL["fei"], "normal"), (1, COL["rel"], "abnormal")):
            ax[i, j].scatter(Z[yV == lab, 0], Z[yV == lab, 1], s=4, c=c, alpha=.7, lw=0, label=nm)
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, j].set_xlabel(f"{K}-NN agreement {N['v_' + {'Released ViT-M': 'rel', 'Random-init ViT-M': 'rnd', 'FEI+C': 'feic', 'rand-FEI+C': 'rfeic'}[a] + '_' + m]}", fontsize=7)
        if i == 0: ax[i, j].set_title(a, fontsize=8.5)
    ax[0, 0].set_ylabel("t-SNE"); ax[1, 0].set_ylabel("UMAP")
ax[0, 0].legend(fontsize=6.5, markerscale=2.5, loc="upper left", frameon=True)
save("tuab_maps")

json.dump(N, open(f"{OUT}/numbers.json", "w"), indent=1, ensure_ascii=False)
json.dump(T, open(f"{OUT}/tables.json", "w"), indent=1, ensure_ascii=False)
print(f"{len(N)} numbers, {len(T)} tables, figs: {sorted(os.listdir(FIG))}")
