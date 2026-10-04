"""Act-0 Part 2 (Phase B): healthy vs stroke on Zenodo 19599466 (resting EEG, BrainVision 63 ch, 1000 Hz, ref A2).
The original design is recorded in runs/act0/phaseB/PREREGISTRATION.md; its sha256 is stored when present.

  stages (serial, resumable -- each stage skips if its output exists):
    --prereg   write the preregistration (never overwritten)
    --prep     CPU: bad-channel interp -> BSI/DAR/DTABR/rel-power gate (A2 ref) + encoder preps (CAR)
    --embed    GPU: released/random ViT-M, FEI+C (5 NMT encoders) + random FEI+C, collapse metrics
    --loso     CPU: LOSO arms, full metrics, per-subject preds, permutation p, bootstrap CIs, paired deltas
    (no flag = prep, embed and loso for both cohorts)      --selftest
  python stroke_phaseB.py   -> code/runs/act0/phaseB/{gate,emb,results}_{primary,sensitivity}.*
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads, cool_gate
cap_threads(4)
import numpy as np
from scipy.signal import welch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from act0_metrics import bootstrap_ci, fmt, metrics, save

ROOT = "/home/mashfiq/eeg_vjepa"
RAW = f"{ROOT}/data/zenodo_stroke"
PREP = f"{ROOT}/data/zenodo_stroke_prep"
TAG = os.environ.get("FEIC_TAG", "")   # "_joint" -> NMT+TUAB-train pretrained FEI/C encoders, separate outputs
OUT = f"{ROOT}/code/runs/act0/phaseB{TAG}"
BASE = f"{ROOT}/code/runs/act0/phaseB"   # NMT-only run (reused arms in joint mode)
PREREG = f"{OUT}/PREREGISTRATION.md"
CONTROLS6 = ["C03", "CONTROL04", "CONTROL05", "CONTROL06", "CONTROL07", "CONTROL08"]
COHORTS = {  # name -> (subjects, rest seconds). y: stroke = 1
    "primary": ([f"PAC{i:02d}" for i in range(2, 11)] + CONTROLS6, 285.0),        # pure rest block, equal length
    "sensitivity": ([f"PAC{i:02d}" for i in range(1, 11)] + ["02", "C01"] + CONTROLS6, 300.0),  # authors' first 300 s
}
# authors' bad-channel lists (1-based indices into the file's channel order; qEEG_5min_previos.m)
BADS = {"C03": [27], "C04": [40], "C05": [22, 34, 40], "C06": [40], "C07": [40], "C08": [40],
        "PAC01": ["VEOGn", 42, 63], "PAC02": [32], "PAC03": [11, 15, 27, 33, 37, 47, 51, 62], "PAC04": [42],
        "PAC05": [57], "PAC06": [57], "PAC08": [42], "PAC09": [40], "PAC10": [40]}
CENTRAL21 = ["FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCZ", "C1", "C2", "C3", "C4", "C5", "C6", "CZ",
             "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ"]
AH, UH = ["FC1", "FC3", "C1", "C3", "C5", "CP1"], ["FC2", "FC4", "C2", "C4", "C6", "CP2"]   # preprint pairs
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
BSI_KEYS = ["bsi_1_30", "bsi_delta", "bsi_theta", "bsi_alpha", "bsi_beta"]   # the BSI classifier features
RENAME = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
N_PERM, N_BOOT = 1000, 2000


def bad_key(s):   # CONTROL04 -> C04 (authors' naming)
    return s.replace("CONTROL", "C")


# ------------------------------------------------------------------ prep (CPU)
def load(subj, secs):
    import mne
    raw = mne.io.read_raw_brainvision(f"{RAW}/{subj}.vhdr", preload=False, verbose=False)
    raw.crop(tmax=secs - 1 / raw.info["sfreq"]).load_data(verbose=False)
    raw.set_channel_types({c: "eog" for c in raw.ch_names if "EOG" in c} | {"A1": "misc"})
    raw.set_montage("standard_1005", match_case=False, on_missing="ignore", verbose=False)
    bads = [b if isinstance(b, str) else raw.ch_names[b - 1] for b in BADS.get(bad_key(subj), [])]
    raw.info["bads"] = [b for b in bads if raw.get_channel_types([b])[0] == "eeg"]
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw, bads


def gate_feats(raw):
    """Preprint pipeline (minus ICA): notch 50, 0.5-100 Hz, 2 s epochs, +-100 uV reject, Welch, 21 central ch."""
    r = raw.copy().notch_filter(50, verbose=False).filter(0.5, 100, verbose=False)
    fs = int(r.info["sfreq"])
    x = r.get_data(picks=CENTRAL21) * 1e6                                       # (21, T) uV
    ep = x[:, :x.shape[1] // (2 * fs) * 2 * fs].reshape(21, -1, 2 * fs).transpose(1, 0, 2)
    keep = np.abs(ep).max((1, 2)) <= 100
    f, p = welch(ep[keep], fs=fs, nperseg=fs, axis=-1)                           # 1 Hz bins
    p = p.mean(0)                                                                # (21, F)
    ch = {c: p[i] for i, c in enumerate(CENTRAL21)}
    bp = {b: p[:, (f >= lo) & (f < hi)].sum(-1).mean() for b, (lo, hi) in BANDS.items()}
    tot = sum(bp.values())
    out = {f"rel_{b}": float(v / tot) for b, v in bp.items()}
    out |= {"dar": float(bp["delta"] / bp["alpha"]),
            "dtabr": float((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"]))}
    for name, (lo, hi) in [("1_30", (1, 30))] + list(BANDS.items()):
        sel = (f >= lo) & (f < hi)
        out[f"bsi_{name}"] = float(pdbsi(np.stack([ch[c][sel] for c in AH]), np.stack([ch[c][sel] for c in UH])))
    return out | {"n_epochs": int(len(keep)), "kept_frac": float(keep.mean())}


def pdbsi(L, R):   # pairwise-derived BSI (Sheorajpanday 2009): mean over pairs & bins of |(R-L)/(R+L)|, in [0,1]
    return np.abs((R - L) / (R + L)).mean()


def encoder_preps(raw):
    """CAR over EEG ch -> 19 ch (CHANNELS order). vitm = §4.1 (1-40 Hz, 100 Hz, z, 5 s stride 2.5 s);
    feic = NMT-matched (0.5-40 Hz = NMT device band, 200 Hz, z, non-overlap 500)."""
    from preprocess_nmt import CHANNELS
    r = raw.copy().pick("eeg").set_eeg_reference("average", verbose=False).rename_channels(RENAME)
    r.pick(CHANNELS).reorder_channels(CHANNELS)
    out = {}
    for name, (lo, fs, stride) in {"vitm": (1.0, 100, 250), "feic": (0.5, 200, 500)}.items():
        d = r.copy().filter(lo, 40.0, verbose=False).resample(fs, verbose=False).get_data()
        d = (d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)
        out[name] = np.stack([d[:, i:i + 500] for i in range(0, d.shape[1] - 500 + 1, stride)]).astype(np.float32)
    return out


def prep(cohort):
    path = f"{OUT}/gate_{cohort}.json"
    if os.path.exists(path):
        return json.load(open(path))
    subs, secs = COHORTS[cohort]
    rows = {}
    for s in subs:
        cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)
        raw, bads = load(s, secs)
        rows[s] = gate_feats(raw) | {"y": int(s.startswith("PAC")), "bads": [str(b) for b in bads]}
        for name, fr in encoder_preps(raw).items():
            os.makedirs(f"{PREP}/{cohort}/{name}", exist_ok=True)
            np.save(f"{PREP}/{cohort}/{name}/{s}.npy", fr)
        print(f"  prep {cohort} {s}: bsi {rows[s]['bsi_1_30']:.3f} kept {rows[s]['kept_frac']:.2f}", flush=True)
    y = np.array([r["y"] for r in rows.values()])
    expect = {"bsi_1_30": 1, "dar": 1, "dtabr": 1, "rel_delta": 1, "rel_alpha": -1, "rel_beta": -1}   # preprint
    tests = {}
    for k in [k for k in next(iter(rows.values())) if k.startswith(("bsi", "rel", "dar", "dtabr"))]:
        v = np.array([r[k] for r in rows.values()])
        a, b = v[y == 1], v[y == 0]
        tests[k] = dict(stroke_median=float(np.median(a)), control_median=float(np.median(b)),
                        p=float(mannwhitneyu(a, b, alternative="two-sided").pvalue),
                        direction=int(np.sign(np.median(a) - np.median(b))), preprint_direction=expect.get(k))
    t = tests["bsi_1_30"]
    res = dict(cohort=cohort, n_stroke=int(y.sum()), n_control=int((1 - y).sum()), rest_s=secs, subjects=rows,
               tests=tests, gate_pass=bool(t["p"] < 0.05 and t["direction"] == 1),
               direction_agreement={k: tests[k]["direction"] == d for k, d in expect.items()})
    save(path, res)
    return res


# ------------------------------------------------------------------ embed (GPU)
def embed(cohort):
    path = f"{OUT}/emb_{cohort}.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    import torch
    import tuab_cached_eval as T
    import fei_pretrain as F
    import fei_branchC as C
    from head_to_head_cv import h20_args
    subs, _ = COHORTS[cohort]
    E, col = {"names": np.array(subs), "y": np.array([int(s.startswith("PAC")) for s in subs])}, {}
    M = T.MODELS["vitm"]; clip_len = M["frames_per_clip"] * M["frame_step"]
    for tag, ck, seed in [("vitm_released", M["ckpt"], 0)] + [(f"vitm_rand{s}", "none", s) for s in range(3)]:
        enc, feats, tstd = T.encoder(M, ck, seed), [], []
        with torch.no_grad():
            for s in subs:
                cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)
                fr = np.load(f"{PREP}/{cohort}/vitm/{s}.npy")
                x = torch.from_numpy(np.stack([T.clip(fr, e, M) for e in range(clip_len, len(fr))]))[:, None].to(T.DEV)
                tok = enc(x).float()                                   # (P, tokens, D)
                feats.append(torch.cat([tok.mean(1), tok.std(1)], -1).mean(0).cpu().numpy())
                tstd.append(float(tok.std(1).mean()))
        E[tag], col[tag] = np.array(feats), {"token_std": float(np.mean(tstd))}
        del enc; torch.cuda.empty_cache()
    args = h20_args(); C._configure(args)
    files = [(f"{PREP}/{cohort}/feic/{s}.npy", 0) for s in subs]
    for k in list(range(5)) + [f"rand{s}" for s in range(3)]:
        torch.manual_seed(k if isinstance(k, int) else int(k[-1]))
        a, c = F.Encoder(args.d).to(F.DEV), C._enc(args)(args.d).to(F.DEV)
        if isinstance(k, int):   # PLAIN FEI (never Ada-FEI in Act-0) + Branch-C h20, NMT seed-0 fold-k encoders
            a.load_state_dict(torch.load(f"{ROOT}/code/fei{TAG}_enc_cv_s0_f{k}.pt", map_location=F.DEV))
            c.load_state_dict(torch.load(f"{ROOT}/code/branchC_h20{TAG}_enc_cv_s0_f{k}.pt", map_location=F.DEV))
        a.eval(); c.eval()
        sfx = f"f{k}" if isinstance(k, int) else k
        E[f"fei_{sfx}"], E[f"c_{sfx}"] = F.embed_set(a, files, C.L)[0], C.embed_dyn(c, files)[0]
    for k in [k for k in E if k not in ("names", "y")]:
        X = E[k]; Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        cos = Xn @ Xn.T
        col.setdefault(k, {}).update(rec_std=float(X.std(0).mean()),
                                     rec_cos=float(cos[np.triu_indices(len(X), 1)].mean()))
    np.savez(path, **E)
    save(f"{OUT}/collapse_{cohort}.json", col)
    return E


# ------------------------------------------------------------------ LOSO (CPU)
def loso(X, y):
    """Leave-one-subject-out out-of-fold P(stroke). One recording per subject -> LOO."""
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
        p[i] = clf.fit(X[tr], y[tr]).predict_proba(X[i:i + 1])[0, 1]
    return p


def arm(members, y, rng):
    """members: list of feature matrices (one per encoder/seed). Ensemble = mean member probability."""
    P = np.array([loso(X, y) for X in members]); p = P.mean(0)
    null = []
    for _ in range(N_PERM):
        cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)
        yp = rng.permutation(y)
        null.append(metrics(yp, np.mean([loso(X, yp) for X in members], 0))["auroc"])
    m = metrics(y, p)
    au = [metrics(y, q)["auroc"] for q in P]
    return p, P, np.array(null), m | dict(auroc_ci=bootstrap_ci(y, p, "auroc", N_BOOT), bal_acc_ci=bootstrap_ci(y, p, "bal_acc", N_BOOT),
                       perm_p=float((1 + np.sum(np.array(null) >= m["auroc"])) / (1 + N_PERM)),
                       member_auroc=au, member_auroc_mean=float(np.mean(au)), member_auroc_sd=float(np.std(au)))


def paired(y, pa, pb, seed=0):
    """Bootstrap over subjects of AUROC(a) - AUROC(b) on the SAME resamples."""
    rng, d = np.random.RandomState(seed), []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(metrics(y[i], pa[i])["auroc"] - metrics(y[i], pb[i])["auroc"])
    d = np.array(d)
    return dict(delta=float(metrics(y, pa)["auroc"] - metrics(y, pb)["auroc"]),
                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))


def run_loso(cohort, E, gate):
    path = f"{OUT}/results_{cohort}.json"
    if os.path.exists(path):
        return json.load(open(path))
    names, y = list(E["names"]), E["y"]
    B = np.array([[gate["subjects"][s][k] for k in BSI_KEYS] for s in names])
    fc = lambda sfx: np.hstack([E[f"fei_{sfx}"], E[f"c_{sfx}"]])
    ARMS = {"BSI": [B],
            "VJEPA": [E["vitm_released"]], "VJEPA+BSI": [np.hstack([E["vitm_released"], B])],
            "rand-VJEPA": [E[f"vitm_rand{s}"] for s in range(3)],
            "rand-VJEPA+BSI": [np.hstack([E[f"vitm_rand{s}"], B]) for s in range(3)],
            "FEI+C": [fc(f"f{k}") for k in range(5)], "FEI+C+BSI": [np.hstack([fc(f"f{k}"), B]) for k in range(5)],
            "rand-FEI+C": [fc(f"rand{s}") for s in range(3)],
            "rand-FEI+C+BSI": [np.hstack([fc(f"rand{s}"), B]) for s in range(3)]}
    res, P, raw, reuse = {}, {}, {}, {}
    if TAG:   # joint encoders change only the FEI+C arms; the rest see identical inputs -> reuse the NMT-only run
        bo, be = np.load(f"{BASE}/oof_{cohort}.npz"), np.load(f"{BASE}/emb_{cohort}.npz")
        br = json.load(open(f"{BASE}/results_{cohort}.json"))
        assert br["preds"]["subject"] == names
        same = all(np.allclose(E[k], be[k], atol=1e-5) for k in E if k.startswith(("vitm_", "fei_rand", "c_rand")))
        print(f"  reuse NMT-only non-FEI+C arms: {'yes' if same else 'NO (inputs differ) -> recompute'}", flush=True)
        if same:
            reuse = {n: (np.array(br["preds"][n]), bo[f"{n}__members"], bo[f"{n}__null_auroc"], br["arms"][n])
                     for n in ARMS if not n.startswith("FEI+C")}
    for name, mem in ARMS.items():
        P[name], raw[f"{name}__members"], raw[f"{name}__null_auroc"], res[name] = \
            reuse.get(name) or arm(mem, y, np.random.RandomState(0))
        print(f"  {cohort:11s} {name:15s} {fmt(res[name])}  perm p {res[name]['perm_p']:.3f}", flush=True)
    comps = {"VJEPA+BSI vs BSI": ("VJEPA+BSI", "BSI"), "VJEPA vs rand-VJEPA": ("VJEPA", "rand-VJEPA"),
             "FEI+C vs VJEPA": ("FEI+C", "VJEPA")}
    out = dict(cohort=cohort, prereg_sha256=sha(PREREG) if os.path.isfile(PREREG) else None,
               n=len(y), n_stroke=int(y.sum()), arms=res,
               paired={k: paired(y, P[a], P[b]) for k, (a, b) in comps.items()},
               preds={"subject": names, "y": y.tolist()} | {k: v.tolist() for k, v in P.items()})
    # member-level out-of-fold P(stroke) (members x subjects) + permutation null AUROCs per arm -> later analysis
    np.savez(f"{OUT}/oof_{cohort}.npz", subject=np.array(names), y=y, **raw)
    save(path, out)
    return out


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


PREREG_TEXT = """# Act-0 Part 2 (Phase B) preregistration: healthy vs stroke (Zenodo 19599466)
Written before the original LOSO runs; future runs record its sha256 when the file is present.

## Cohorts (y: stroke = 1; POST-intervention files never used)
- **primary** (pure rest block, before any task marker / segment break), fixed 285 s from onset for every subject
  (shortest rest block = PAC05 285.0 s, so no duration confound): stroke PAC02-PAC10 (9) vs controls C03,
  CONTROL04-08 (6) = the authors' own control list. PAC01, 02, C01 excluded (<= 52 s of rest before the task).
- **sensitivity**: authors' definition = first 300 s of every file (qEEG_5min_previos.m), PAC01-10 (10) vs
  02, C01, C03, CONTROL04-08 (8). For PAC01/02/C01 this includes task data -- reported as sensitivity only.

## Preprocessing
- Authors' bad-channel lists (1-based file-order indices) -> spherical-spline interpolation (EEG ch only). No ICA
  (deviation from preprint; stated).
- BSI gate: recorded A2 reference, notch 50 Hz, 0.5-100 Hz, 2 s non-overlapping epochs, reject epochs > +-100 uV
  on the 21 central channels, Welch (1 s Hann, 50%), bands delta 1-4 theta 4-8 alpha 8-12 beta 12-30.
  Relative power = band / sum of 4 (channel-averaged); DAR = d/a; DTABR = (d+t)/(a+b).
  pdBSI = mean over 6 pairs (FC1/FC2, FC3/FC4, C1/C2, C3/C4, C5/C6, CP1/CP2) and bins of |(R-L)/(R+L)|; broadband
  1-30 Hz + per band.
- Encoder inputs: CAR over EEG channels, 19 ch (T7/T8/P7/P8 -> T3/T4/T5/T6).
  ViT-M: paper §4.1 (1-40 Hz, 100 Hz, per-channel z, 5 s frames stride 2.5 s).
  FEI+C: NMT-matched (0.5-40 Hz = NMT device band, 200 Hz, per-channel z, 2.5 s non-overlapping frames).

## Gate (pre-declared)
Pipeline soundness: broadband pdBSI higher in stroke, two-sided Mann-Whitney p < 0.05 (primary cohort).
Direction agreement with the preprint also reported for DAR, DTABR, rel-delta (up), rel-alpha, rel-beta (down).
LOSO runs regardless; a failed gate is reported as a caveat, never used to change the design.

## Arms (features)
- BSI = [pdBSI 1-30, delta, theta, alpha, beta] (5).
- VJEPA = released ViT-M (4x30x4) frozen; per clip position token mean ++ token std, averaged over all positions.
- rand-VJEPA = same architecture random-init, seeds 0-2 (collapse control).
- FEI+C = PLAIN FEI (fei_enc_cv_s0_f{k}.pt, L=4000) ++ Branch-C h20 (branchC_h20_enc_cv_s0_f{k}.pt), k = 0..4 NMT
  seed-0 fold encoders -> 5 members. rand-FEI+C = random-init seeds 0-2.
- each also + BSI (hstack).
- Collapse metrics per encoder: token std, across-subject feature std, mean pairwise cosine.

## Classifier / evaluation
StandardScaler + LogisticRegression(C=1, class_weight=balanced), leave-one-subject-out. Multi-member arms:
ensemble = mean member P(stroke) (member AUROC mean +- sd also reported). Threshold 0.5 (never tuned).
Metrics: AUROC (primary), AUC-PR, Acc, BAcc, F1, Sens, Spec, confusion; bootstrap 95% CI (2000, over subjects);
label-permutation p (1000, full LOSO re-run per permutation).
Pre-declared paired comparisons (bootstrap delta-AUROC, same resamples): VJEPA+BSI vs BSI; VJEPA vs rand-VJEPA;
FEI+C vs VJEPA. Our-pretrained ViT-M arm is added after P1.5 as a separate, labelled addition.
"""


def selftest():
    L = np.ones((6, 10)); assert pdbsi(L, L) == 0 and np.isclose(pdbsi(L, 2 * L), 1 / 3)
    rng = np.random.RandomState(0); y = np.r_[np.ones(8), np.zeros(7)].astype(int)
    X = rng.randn(15, 4) + 3 * y[:, None]
    p = loso(X, y); assert metrics(y, p)["auroc"] == 1.0
    global N_PERM, N_BOOT; N_PERM, N_BOOT = 50, 200
    _, Pm, nl, m = arm([X, X + 0.1 * rng.randn(*X.shape)], y, np.random.RandomState(0))
    assert m["perm_p"] < 0.05 and m["auroc"] == 1.0 and Pm.shape == (2, 15) and nl.shape == (50,)
    _, _, _, m0 = arm([rng.randn(15, 4)], y, np.random.RandomState(0)); assert m0["perm_p"] > 0.05
    print("stroke_phaseB self-check OK")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    os.makedirs(OUT, exist_ok=True)
    stages = [s for s in ("--prereg", "--prep", "--embed", "--loso") if s in sys.argv] or \
             ["--prep", "--embed", "--loso"]
    if "--prereg" in stages and not os.path.exists(PREREG):
        open(PREREG, "w").write(PREREG_TEXT)
    if os.path.isfile(PREREG):
        print("preregistration sha256:", sha(PREREG), flush=True)
    for cohort in COHORTS:
        gate = prep(cohort) if {"--prep", "--loso"} & set(stages) else None
        if "--prep" in stages:
            t = gate["tests"]
            print(f"GATE {cohort}: {'PASS' if gate['gate_pass'] else 'FAIL'}  " +
                  "  ".join(f"{k} p={v['p']:.3f}{'+' if v['direction'] > 0 else '-'}" for k, v in t.items()), flush=True)
        E = embed(cohort) if {"--embed", "--loso"} & set(stages) else None
        if "--loso" in stages:
            run_loso(cohort, E, gate)
    print("PHASE B DONE" if "--loso" in stages else "stages done: " + " ".join(stages), flush=True)


if __name__ == "__main__":
    main()
