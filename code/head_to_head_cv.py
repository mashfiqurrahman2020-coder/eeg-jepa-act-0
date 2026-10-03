"""
Unified head-to-head on the ADOPTED protocol (2026-09-10, replaces Khan's fixed 185-split):
ONE canonical stratified subject-independent 5-fold CV -- StratifiedKFold(5, shuffle=True,
random_state=0) over the pooled 2417 NMT recs = the IDENTICAL folds Branch-C used for A+C=0.870
-- recording-level, metrics {AUROC, Balanced Acc, AUC-PR}. Every method on the SAME folds and
the SAME StandardScaler+LogReg probe (fei_pretrain.fit_probe, pr=True).

Frozen FM features are FAITHFUL to the CBraMod/LaBraM code (verified 2026-09-10 against their
repos: bipolar/unipolar montage, filter, notch, get_data(uV), /100, model arch, mean-pool over
(ch,patch), input_chans) -- see the cbramod_on_nmt.py / labram_on_nmt.py headers.

Methods: band-power | A (Ada-FEI) | C (spectral-dynamics) | A+C | A+C+BP |
         CBraMod frozen (+rand-init) | LaBraM frozen (+rand-init).
A/C reuse the per-fold cached encoders (no leakage). FM features from the cached pkls.

Run:      python head_to_head_cv.py       (GPU for A/C embedding; hw-guarded, resumable)
Selftest: python head_to_head_cv.py --selftest
"""
import os, sys, json, pickle, argparse
from types import SimpleNamespace
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "/home/mashfiq/eeg_vjepa/code")
import fei_pretrain as F
import fei_branchC as C
import baseline_bandpower as B
from hw_guard import cap_threads, cool_gate

OUT = "/home/mashfiq/eeg_vjepa/code/head_to_head_cv_results.json"
CBRAMOD_PKL = "/home/mashfiq/eeg_vjepa/code/cbramod_nmt_emb.pkl"
LABRAM_PKL = "/home/mashfiq/eeg_vjepa/code/labram_nmt_emb.pkl"
A_CKPT = "/home/mashfiq/eeg_vjepa/code/adafei_neg1_enc_cv_s0_f{}.pt"
C_CKPT = "/home/mashfiq/eeg_vjepa/code/branchC_h20_enc_cv_s0_f{}.pt"   # 20s harder-horizon C (the A+C=0.870 encoders)
METHODS = ["band-power", "A", "C", "A+C", "A+C+BP", "CBraMod", "CBraMod(rand)", "LaBraM", "LaBraM(rand)"]


def h20_args():   # Branch-C 20s config = the A+C=0.870 setting (--L 4000 --hop 100, rest default)
    return SimpleNamespace(d=256, h=128, L=4000, nfft=200, hop=100, tc_frac=0.6, enc="gru", tag="h20")


def rid_of(p):
    return os.path.basename(p).split(".")[0]


def fm_feat(d, key, sel):
    return np.stack([d[r][key] for r in sel])


def selftest():
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    seen = np.zeros(len(y), bool)
    for _, te in skf.split(np.zeros(len(y)), y):
        assert not seen[te].any(), "folds overlap"
        seen[te] = True
    assert seen.all(), "folds don't cover all recs"
    rng = np.random.default_rng(0); X = rng.standard_normal((200, 20)); yy = (X[:, 0] > 0).astype(int)
    out = F.fit_probe(X, yy, X, yy, pr=True)
    assert len(out) == 3 and all(np.isfinite(out)), out
    print(f"selftest ok: 5 folds partition all {len(y)} recs disjointly; fit_probe(pr=True)->3 finite metrics {tuple(round(v,3) for v in out)}")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    cap_threads()
    args = h20_args(); C._configure(args)
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    rids = [rid_of(p) for p, _ in files]

    print(f"band-power for {len(files)} recs...", flush=True)
    bp = {}
    for i, (p, _) in enumerate(files):
        if i % 300 == 0: cool_gate(verbose=False)
        bp[p] = B.features_for_recording(p)[0]

    cbr = pickle.load(open(CBRAMOD_PKL, "rb"))
    lab = pickle.load(open(LABRAM_PKL, "rb"))
    for name, d in [("CBraMod", cbr), ("LaBraM", lab)]:
        miss = [r for r in rids if r not in d]
        assert not miss, f"{name} pkl missing {len(miss)} rids e.g. {miss[:3]} -> re-run the frozen script"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    loaded = json.load(open(OUT)) if os.path.exists(OUT) else {}
    state = {k: v for k, v in loaded.items() if k.isdigit()}       # keep only fold entries for resume

    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        if str(k) in state:
            print(f"[f{k+1}] cached -> skip", flush=True); continue
        cool_gate(verbose=False)
        trf = [files[i] for i in tr]; tef = [files[i] for i in te]; ytr, yte = y[tr], y[te]
        rtr = [rids[i] for i in tr]; rte = [rids[i] for i in te]

        encA = F.Encoder(args.d).to(F.DEV); encA.load_state_dict(torch.load(A_CKPT.format(k), map_location=F.DEV)); encA.eval()
        XA_tr, _ = F.embed_set(encA, trf, C.L); XA_te, _ = F.embed_set(encA, tef, C.L)
        encC = C._enc(args)(args.d).to(F.DEV); encC.load_state_dict(torch.load(C_CKPT.format(k), map_location=F.DEV)); encC.eval()
        XC_tr, _ = C.embed_dyn(encC, trf); XC_te, _ = C.embed_dyn(encC, tef)
        Xbp_tr = np.array([bp[p] for p, _ in trf]); Xbp_te = np.array([bp[p] for p, _ in tef])

        feats = {
            "band-power": (Xbp_tr, Xbp_te),
            "A": (XA_tr, XA_te), "C": (XC_tr, XC_te),
            "A+C": (np.hstack([XA_tr, XC_tr]), np.hstack([XA_te, XC_te])),
            "A+C+BP": (np.hstack([XA_tr, XC_tr, Xbp_tr]), np.hstack([XA_te, XC_te, Xbp_te])),
            "CBraMod": (fm_feat(cbr, "pretrained", rtr), fm_feat(cbr, "pretrained", rte)),
            "LaBraM": (fm_feat(lab, "pretrained", rtr), fm_feat(lab, "pretrained", rte)),
        }
        fold = {}
        for m, (Xtr, Xte) in feats.items():
            au, ba, pr = F.fit_probe(Xtr, ytr, Xte, yte, pr=True)
            fold[m] = {"auroc": float(au), "bacc": float(ba), "aucpr": float(pr)}
        for name, d in [("CBraMod(rand)", cbr), ("LaBraM(rand)", lab)]:      # rand-init control = mean over 3 seeds
            vals = np.array([F.fit_probe(fm_feat(d, f"rand{s}", rtr), ytr, fm_feat(d, f"rand{s}", rte), yte, pr=True) for s in range(3)])
            fold[name] = {"auroc": float(vals[:, 0].mean()), "bacc": float(vals[:, 1].mean()), "aucpr": float(vals[:, 2].mean())}

        state[str(k)] = fold
        json.dump(state, open(OUT + ".tmp", "w"), indent=2); os.replace(OUT + ".tmp", OUT)
        print(f"[f{k+1}] " + "  ".join(f"{m}={fold[m]['auroc']:.3f}" for m in METHODS), flush=True)

    print("\n===== HEAD-TO-HEAD (seed-0 5-fold CV, recording-level, ADOPTED protocol) =====")
    print(f"{'method':16s} {'AUROC':>15s} {'BAcc':>15s} {'AUC-PR':>15s}")
    summary = {}
    for m in METHODS:
        A = np.array([state[str(k)][m]["auroc"] for k in range(5)])
        Ba = np.array([state[str(k)][m]["bacc"] for k in range(5)])
        Pr = np.array([state[str(k)][m]["aucpr"] for k in range(5)])
        summary[m] = {"auroc": [float(A.mean()), float(A.std())], "bacc": [float(Ba.mean()), float(Ba.std())], "aucpr": [float(Pr.mean()), float(Pr.std())]}
        print(f"{m:16s} {A.mean():.3f}+/-{A.std():.3f}  {Ba.mean():.3f}+/-{Ba.std():.3f}  {Pr.mean():.3f}+/-{Pr.std():.3f}")
    json.dump({"folds": state, "summary": summary, "protocol": "StratifiedKFold(5,shuffle,seed0) recording-level; probe=StandardScaler+LogReg(balanced,C=1); metrics AUROC/BAcc/AUC-PR"},
              open(OUT, "w"), indent=2)
    print(f"\nrefs: Khan fixed-split DROPPED. bars = FMs on THIS identical protocol + NeuroAtlas REVE ~0.845.\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
