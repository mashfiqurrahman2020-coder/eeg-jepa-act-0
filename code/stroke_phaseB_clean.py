"""Phase B clean FEI+C re-prep (post-hoc, after the CV run).
The preregistered FEI+C input (0.5-40 Hz FIR) does NOT reproduce NMT's spectrum: NMT is unfiltered (device roll-off
~35-40 Hz onto a floor, little <0.5 Hz power), so the stroke input is out of distribution for Branch-C's input BN
(z ~ +3..4 at 36-45 Hz, ~ -3 above 46 Hz) and pretrained C collapses on stroke (rec_std 0.03 vs 0.38 on NMT).
Fix = label-free spectral matching: CAR -> 19 ch -> 200 Hz, NO band-pass, then one fixed zero-phase per-channel
filter per cohort that maps the cohort's mean log-PSD (all subjects pooled, labels unused) onto NMT-train's mean
log-PSD; per-channel z; 2.5 s frames. Everything else (encoders, classifier, LOSO, perm, CV schemes) unchanged.
Preregistered outputs untouched; new dirs only. Design + gate fixed in runs/act0/phaseB_clean/POSTHOC.md.
  stages: --prep (CPU, spectral match + PSD gate) --embed (CPU, collapse + BN-z gate) --loso   (none = all)
  CUDA_VISIBLE_DEVICES="" python stroke_phaseB_clean.py   -> runs/act0/phaseB_clean/*        --selftest
"""
import glob
import hashlib
import json
import os
import sys

import stroke_phaseB as B   # cap_threads(4) before numpy
import stroke_phaseB_cv as V
import numpy as np
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

PREP = f"{B.ROOT}/data/zenodo_stroke_prep_nmtmatch"
OUT = f"{B.ROOT}/code/runs/act0/phaseB_clean"
NMT = f"{B.ROOT}/data/NMT_preprocessed"
FS, NPER, N_NMT, N_ITER = 200, 800, 100, 4
GATE = dict(psd_max_dev=0.25, bnz_max=1.5, c_std_ratio_min=0.5, c_const_frac_max=0.05)
PERM_ARMS = ["FEI+C", "rand-FEI+C"]   # the rest: LOSO + CI + kf5 + LPO, no perm
COHORTS = ["primary"]   # preregistered primary cohort
COMPS = [("FEI+C", "rand-FEI+C"), ("FEI+C", "BSI"), ("FEI+C+BSI", "BSI"), ("FEI+C", "FEI+C (orig prep)"),
         ("FEI+C", "VJEPA"), ("C", "rand-C"), ("FEI", "rand-FEI")]
cool = lambda: B.cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)
zs = lambda d: (d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)


def logpsd(sig):   # (19,T) -> (f, (19,F) log10 Welch PSD, 0.25 Hz bins)
    f, p = welch(sig, fs=FS, nperseg=NPER, axis=-1)
    return f, np.log10(p + 1e-20)


def shape(sig, f, logh):   # zero-phase per-channel spectral shaping; logh = log10 AMPLITUDE gain on grid f
    pad = NPER
    x = np.pad(sig, ((0, 0), (pad, pad)), mode="reflect")
    g = np.fft.rfftfreq(x.shape[1], 1 / FS)
    H = 10 ** np.stack([np.interp(g, f, h) for h in logh])
    return np.fft.irfft(np.fft.rfft(x) * H, x.shape[1])[:, pad:-pad]


def frames(d):
    return np.stack([d[:, i:i + 500] for i in range(0, d.shape[1] - 500 + 1, 500)]).astype(np.float32)


def nmt_ref():   # mean log-PSD of N_NMT fixed-random NMT-train recordings (first 300 s), the encoders' training input
    path = f"{OUT}/nmt_ref_psd.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    fl = sorted(glob.glob(f"{NMT}/train/*/*.npy"))
    fl = [fl[i] for i in np.random.RandomState(0).choice(len(fl), N_NMT, replace=False)]
    P = []
    for i, p in enumerate(fl):
        if i % 10 == 0: cool()
        x = np.asarray(np.load(p, mmap_mode="r")[:120])
        f, lp = logpsd(x.transpose(1, 0, 2).reshape(19, -1).astype(np.float64)); P.append(lp)
    np.savez(path, f=f, mean=np.mean(P, 0), sd=np.std(P, 0), files=np.array(fl))
    return dict(np.load(path))


def base(raw):   # preregistered channel handling, minus the band-pass: CAR -> 19 ch (CHANNELS) -> 200 Hz
    from preprocess_nmt import CHANNELS
    r = raw.copy().pick("eeg").set_eeg_reference("average", verbose=False).rename_channels(B.RENAME)
    r.pick(CHANNELS).reorder_channels(CHANNELS)
    return zs(r.resample(FS, verbose=False).get_data())


def summary(f, lp, ref):   # group-mean log-PSD vs NMT at the audit frequencies + variance fractions
    m, d = lp.mean(0), lp.mean(0) - ref
    at = lambda hz: float(np.interp(hz, f, m.mean(0)))
    lin = (10 ** lp).mean((0, 1)); vf = lambda sel: float(lin[sel].sum() / lin.sum())
    band = (f >= 0.5) & (f <= 99)
    return dict(log10psd={f"{hz}Hz": at(hz) for hz in (0.25, 2, 10, 20, 35, 40, 45, 60, 80)},
                var_frac_lt0p5=vf(f < 0.5), var_frac_gt40=vf(f > 40),
                max_abs_dev_0p5_99=float(np.abs(d[:, band]).max()), mean_abs_dev_0p5_99=float(np.abs(d[:, band]).mean()))


def prep(cohort):
    path = f"{OUT}/prep_{cohort}.json"
    if os.path.exists(path):
        return json.load(open(path))
    R = nmt_ref(); f, ref = R["f"], R["mean"]
    subs, secs = B.COHORTS[cohort]
    S = {}
    for s in subs:
        cool(); S[s] = base(B.load(s, secs)[0])
    logh, lp0 = np.zeros_like(ref), np.stack([logpsd(x)[1] for x in S.values()])
    for it in range(N_ITER):   # iterate: re-z after shaping shifts the level slightly
        lp = np.stack([logpsd(zs(shape(x, f, logh)))[1] for x in S.values()])
        logh += (ref - lp.mean(0)) / 2
    D = {s: zs(shape(x, f, logh)) for s, x in S.items()}
    lp1 = np.stack([logpsd(x)[1] for x in D.values()])
    os.makedirs(f"{PREP}/{cohort}/feic", exist_ok=True)
    for s, d in D.items():
        np.save(f"{PREP}/{cohort}/feic/{s}.npy", frames(d))
    old = np.stack([logpsd(np.load(f"{B.PREP}/{cohort}/feic/{s}.npy").transpose(1, 0, 2).reshape(19, -1))[1] for s in subs])
    nm = np.stack([ref])   # the NMT mean itself, same summary for reference
    res = dict(cohort=cohort, n_nmt_ref=N_NMT, filter_log10_gain_range=[float(logh.min()), float(logh.max())],
               nmt=summary(f, nm, ref), prereg_prep=summary(f, old, ref), unfiltered=summary(f, lp0, ref),
               matched=summary(f, lp1, ref))
    res["psd_gate_pass"] = bool(res["matched"]["max_abs_dev_0p5_99"] <= GATE["psd_max_dev"])
    np.savez(f"{OUT}/filter_{cohort}.npz", f=f, log10_amp_gain=logh)
    B.save(path, res)
    return res


def embed(cohort):
    path = f"{OUT}/emb_{cohort}.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    import torch
    import fei_pretrain as F
    import fei_branchC as C
    from head_to_head_cv import h20_args
    assert F.DEV == "cpu", "run with CUDA_VISIBLE_DEVICES='' (GPU belongs to the TUAB lanes)"
    subs, _ = B.COHORTS[cohort]
    args = h20_args(); C._configure(args)
    files = [(f"{PREP}/{cohort}/feic/{s}.npy", 0) for s in subs]
    ref = sorted(glob.glob(f"{NMT}/eval/*/*.npy")); ref = [(ref[i], 0) for i in np.random.RandomState(0).choice(len(ref), 15, replace=False)]
    E, col = {"names": np.array(subs), "y": np.array([int(s.startswith("PAC")) for s in subs])}, {}

    def colm(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        return dict(rec_std=float(X.std(0).mean()), rec_cos=float((Xn @ Xn.T)[np.triu_indices(len(X), 1)].mean()),
                    const_frac=float((X.std(0) < 1e-5).mean()))

    def bnz(c, fl):   # channel-averaged z of the input spectrogram vs the checkpoint's NMT running stats, per bin
        rm, rv = c.bn.running_mean.numpy(), c.bn.running_var.numpy()
        z = []
        for p, _ in fl:
            sig = F.load_continuous(p); w = np.stack([sig[:, i:i + C.L] for i in range(0, sig.shape[1] - C.L + 1, C.L)])
            z.append(((C.spectrogram(torch.from_numpy(w)).numpy() - rm) / np.sqrt(rv + 1e-5)).mean((0, 1)))
        return np.mean(z, 0).reshape(19, C.NF).mean(0)   # (101,) 1 Hz bins

    for k in list(range(5)) + [f"rand{s}" for s in range(3)]:
        cool()
        torch.manual_seed(k if isinstance(k, int) else int(k[-1]))
        a, c = F.Encoder(args.d), C._enc(args)(args.d)
        if isinstance(k, int):   # PLAIN FEI + Branch-C h20, NMT-only seed-0 fold-k encoders (never Ada-FEI in Act-0)
            a.load_state_dict(torch.load(f"{B.ROOT}/code/fei_enc_cv_s0_f{k}.pt", map_location="cpu"))
            c.load_state_dict(torch.load(f"{B.ROOT}/code/branchC_h20_enc_cv_s0_f{k}.pt", map_location="cpu"))
        a.eval(); c.eval()
        sfx = f"f{k}" if isinstance(k, int) else k
        E[f"fei_{sfx}"], E[f"c_{sfx}"] = F.embed_set(a, files, C.L)[0], C.embed_dyn(c, files)[0]
        col[f"fei_{sfx}"], col[f"c_{sfx}"] = colm(E[f"fei_{sfx}"]), colm(E[f"c_{sfx}"])
        if isinstance(k, int):
            col[f"c_{sfx}"] |= dict(nmt=colm(C.embed_dyn(c, ref)[0]), bnz_stroke=bnz(c, files).round(3).tolist(),
                                    bnz_nmt=bnz(c, ref).round(3).tolist(),
                                    bnz_stroke_prereg=bnz(c, [(f"{B.PREP}/{cohort}/feic/{s}.npy", 0) for s in subs]).round(3).tolist())
        print(f"  embed {cohort} {sfx}: C rec_std {col[f'c_{sfx}']['rec_std']:.3f} const {col[f'c_{sfx}']['const_frac']:.2f}"
              + (f" (NMT {col[f'c_{sfx}']['nmt']['rec_std']:.3f})  BN max|z| {np.abs(col[f'c_{sfx}']['bnz_stroke']).max():.2f}"
                 f" (prereg prep {np.abs(col[f'c_{sfx}']['bnz_stroke_prereg']).max():.2f})" if isinstance(k, int) else ""), flush=True)
    pre = [col[f"c_f{k}"] for k in range(5)]
    col["gate"] = dict(thresholds=GATE,
                       bnz_max=float(max(np.abs(c["bnz_stroke"]).max() for c in pre)),
                       c_std_ratio_min=float(min(c["rec_std"] / c["nmt"]["rec_std"] for c in pre)),
                       c_const_frac_max=float(max(c["const_frac"] for c in pre)))
    g = col["gate"]
    g["pass"] = bool(g["bnz_max"] <= GATE["bnz_max"] and g["c_std_ratio_min"] >= GATE["c_std_ratio_min"]
                     and g["c_const_frac_max"] <= GATE["c_const_frac_max"])
    np.savez(path, **E)
    B.save(f"{OUT}/collapse_{cohort}.json", col)
    return E


def arms(E, bsi):
    fc = lambda s: np.hstack([E[f"fei_{s}"], E[f"c_{s}"]])
    f5, r3 = [f"f{k}" for k in range(5)], [f"rand{s}" for s in range(3)]
    return {"FEI+C": [fc(s) for s in f5], "FEI+C+BSI": [np.hstack([fc(s), bsi]) for s in f5],
            "rand-FEI+C": [fc(s) for s in r3], "rand-FEI+C+BSI": [np.hstack([fc(s), bsi]) for s in r3],
            "FEI": [E[f"fei_{s}"] for s in f5], "C": [E[f"c_{s}"] for s in f5],
            "rand-FEI": [E[f"fei_{s}"] for s in r3], "rand-C": [E[f"c_{s}"] for s in r3]}


def run_loso(cohort, E):
    path = f"{OUT}/results_{cohort}.json"
    if os.path.exists(path):
        return json.load(open(path))
    names, y = list(E["names"]), E["y"]
    g = json.load(open(f"{B.OUT}/gate_{cohort}.json"))
    A = arms(E, np.array([[g["subjects"][s][k] for k in B.BSI_KEYS] for s in names]))
    res, P, K, W = {}, {}, {}, {}
    for name, mem in A.items():
        n0 = B.N_PERM
        B.N_PERM = n0 if name in PERM_ARMS else 0
        P[name], _, _, r = B.arm(mem, y, np.random.RandomState(0))
        B.N_PERM = n0
        if name not in PERM_ARMS: r.pop("perm_p")
        K[name], W[name] = V.kf5_aurocs(mem, y, V.R), V.lpo(mem, y)
        res[name] = r | dict(kf5_mean=float(K[name].mean()), kf5_sd=float(K[name].std()), lpo=float(W[name].mean()))
        print(f"  {cohort:11s} {name:15s} {B.fmt(r)}" + (f"  perm p {r['perm_p']:.3f}" if "perm_p" in r else "")
              + f"  kf5 {K[name].mean():.3f}±{K[name].std():.3f}  lpo {W[name].mean():.3f}", flush=True)
    # reference arms: the preregistered LOSO preds + the CV run's kf5/LPO (same seeds -> same partitions, paired)
    pr, cvz = json.load(open(f"{B.OUT}/results_{cohort}.json")), np.load(f"{V.OUT}/kf5_lpo_{cohort}.npz")
    assert pr["preds"]["subject"] == names
    for name, src in [("BSI", "BSI"), ("VJEPA", "VJEPA"), ("FEI+C (orig prep)", "FEI+C")]:
        P[name], K[name], W[name] = np.array(pr["preds"][src]), cvz[f"kf5__{src}"], cvz[f"lpo__{src}"]
    paired = {f"{a} vs {b}": B.paired(y, P[a], P[b]) | dict(kf5_delta=float((K[a] - K[b]).mean()),
              kf5_frac_a_better=float((K[a] > K[b]).mean()), lpo=V.lpo_paired(W[a], W[b])) for a, b in COMPS}
    out = dict(cohort=cohort, posthoc_sha256=hashlib.sha256(open(f"{OUT}/POSTHOC.md", "rb").read()).hexdigest(),
               n=len(y), n_stroke=int(y.sum()), arms=res, paired=paired,
               preds={"subject": names, "y": y.tolist()} | {k: v.tolist() for k, v in P.items()})
    np.savez(f"{OUT}/kf5_lpo_{cohort}.npz", **{f"kf5__{k}": v for k, v in K.items()}, **{f"lpo__{k}": v for k, v in W.items()})
    B.save(path, out)
    return out


POSTHOC = """# Phase B clean FEI+C re-prep -- POST-HOC analysis (not preregistered), fixed before any clean LOSO run
Why: audit 2026-09-26 found the preregistered FEI+C input band (0.5-40 Hz FIR, labelled "NMT device band") does not
reproduce NMT's spectrum (NMT is unfiltered: roll-off ~35-40 Hz onto a floor, little <0.5 Hz power), putting stroke
inputs out of distribution for Branch-C's input BN and collapsing pretrained C on stroke.
Prep: authors' bad-channel interpolation -> CAR -> 19 ch (T7/T8/P7/P8 -> T3/T4/T5/T6) -> resample 200 Hz, no
band-pass -> per-channel z -> ONE fixed zero-phase per-channel spectral-shaping filter per cohort, estimated from the
cohort's mean log10 Welch PSD (0.25 Hz bins, all subjects pooled, labels never used) to NMT-train's mean log10 PSD
(100 fixed-random NMT-train recordings, first 300 s), 4 fixed-point iterations -> per-channel z -> 2.5 s frames.
Transductive but label-free: the same linear filter is applied to every subject.
Gate (reported; LOSO runs regardless, a failed gate is a caveat): matched group-mean log10 PSD within 0.25 decades
of NMT at every channel x bin in 0.5-99 Hz; Branch-C input-BN |z| (channel-averaged, per 1 Hz bin, vs each NMT
checkpoint's running stats) <= 1.5 everywhere; pretrained C across-subject std >= 0.5 x its std on 15 NMT-eval
recordings; <= 5 % constant dims (std < 1e-5).
Encoders/classifier/evaluation identical to the preregistration (5 NMT-only PLAIN FEI + Branch-C h20 members,
3 random-init, StandardScaler + LogReg C=1 balanced, LOSO, 1000-perm p for FEI+C and rand-FEI+C;
2000 bootstrap CIs) plus the CV-robustness schemes (100x stratified 5-fold, leave-pair-out).
Comparisons: FEI+C vs rand-FEI+C; FEI+C vs BSI; FEI+C+BSI vs BSI; FEI+C vs FEI+C (preregistered prep); FEI+C vs
VJEPA; C vs rand-C; FEI vs rand-FEI. Primary cohort only (15 subjects; trimmed for time, decided before any run).
Nothing else in Phase B changes; preregistered outputs are never overwritten.
"""


def selftest():   # shaping by a flat gain is exact; matching drives a toy spectrum onto the target
    rng = np.random.RandomState(0)
    x = rng.randn(19, 20000)
    f = np.fft.rfftfreq(NPER, 1 / FS)
    assert np.allclose(shape(x, f, np.full((19, len(f)), np.log10(2.0))), 2 * x)
    y = zs(shape(x, f, np.tile(-1.5 / (1 + np.exp(-(f - 40) / 2)), (19, 1))))   # toy "NMT": smooth roll-off to a floor
    ref = logpsd(y)[1]; logh = np.zeros_like(ref)
    for _ in range(N_ITER):
        logh += (ref - logpsd(zs(shape(x, f, logh)))[1]) / 2
    m = logpsd(zs(shape(x, f, logh)))[1]
    assert np.abs(m - ref)[:, (f >= 0.5) & (f <= 99)].max() < 0.25, np.abs(m - ref).max()
    print("stroke_phaseB_clean self-check OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(f"{OUT}/POSTHOC.md"):
        open(f"{OUT}/POSTHOC.md", "w").write(POSTHOC)
    stages = [s for s in ("--prep", "--embed", "--loso") if s in sys.argv] or ["--prep", "--embed", "--loso"]
    for c in COHORTS:
        if "--prep" in stages:
            r = prep(c); m = r["matched"]
            print(f"PSD GATE {c}: {'PASS' if r['psd_gate_pass'] else 'FAIL'}  max|dev| {m['max_abs_dev_0p5_99']:.3f} "
                  f"(prereg prep {r['prereg_prep']['max_abs_dev_0p5_99']:.3f})  var<0.5Hz {m['var_frac_lt0p5']:.4f} "
                  f">40Hz {m['var_frac_gt40']:.4f}  NMT {r['nmt']['var_frac_lt0p5']:.4f}/{r['nmt']['var_frac_gt40']:.4f}", flush=True)
        if "--embed" in stages or "--loso" in stages:
            E = embed(c); g = json.load(open(f"{OUT}/collapse_{c}.json"))["gate"]
            print(f"EMBED GATE {c}: {'PASS' if g['pass'] else 'FAIL'}  BN max|z| {g['bnz_max']:.2f}  C std ratio "
                  f"{g['c_std_ratio_min']:.2f}  const {g['c_const_frac_max']:.2f}", flush=True)
        if "--loso" in stages:
            r = run_loso(c, E)
            print(f"\n== {c} (n={r['n']}, {r['n_stroke']} stroke) ==")
            for k, v in r["paired"].items():
                print(f"  {k:32s} loso Δ {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  kf5 Δ {v['kf5_delta']:+.3f} "
                      f"({v['kf5_frac_a_better']:.0%})  lpo Δ {v['lpo']['delta']:+.3f} p(Δ≤0) {v['lpo']['p_le0']:.2f}", flush=True)
    print("PHASE B CLEAN DONE" if "--loso" in stages else "stages done: " + " ".join(stages), flush=True)
