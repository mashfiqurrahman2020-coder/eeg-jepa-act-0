"""Act-0: FEI+C on TUAB (first time) -- frozen NMT encoders (PLAIN FEI fei_enc_cv_s0_f{k} + Branch-C h20
branchC_h20_enc_cv_s0_f{k}, k=0..4; never Ada-FEI) + random-init controls (seeds 0-2), LogReg probe.
Same train sets as P1.4 (the 5 paper-style 546-rec subsets + full 2717 train), test = official eval (276).
Prep = the NMT FEI+C recipe mapped to TUAB: 19 ch (CHANNELS order, -REF as stored), first 20 min (NMT p90 = 16 min),
0.5-40 Hz (= NMT device band), 200 Hz, per-channel z-score, (F,19,500) non-overlapping frames.

  python tuab_feic.py   # prep (CPU) -> embed (GPU) -> probe (CPU); every stage resumable
  -> data/TUAB_feic_200hz/..., code/runs/act0/tuab_feic/{emb_*.npy, results.json, preds_*.npz, collapse.json}
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess_nmt as P   # applies cap_threads(4) before numpy
from hw_guard import cool_gate
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from act0_metrics import bootstrap_ci, fmt, metrics, save
from preprocess_tuab import CH_MAP, edfs

ROOT = "/home/mashfiq/eeg_vjepa"
PREP = f"{ROOT}/data/TUAB_feic_200hz"
TAG = os.environ.get("FEIC_TAG", "")   # "_joint" -> NMT+TUAB-train pretrained FEI/C encoders, separate outputs
OUT = f"{ROOT}/code/runs/act0/tuab_feic{TAG}"
CROP_S = 1200
ENCODERS = [f"f{k}" for k in range(5)] + [f"rand{s}" for s in range(3)]


def prep(rev=False):
    import fcntl
    import mne
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)   # single-threaded FFT/BLAS: 4-thread bursts spiked this box to ~93 C
    os.makedirs(PREP, exist_ok=True)
    outp = lambda t: f"{PREP}/{t[0]}/{t[1]}/{os.path.basename(t[2])[:-4]}.npy"
    todo = list(edfs())
    if not rev:
        lock = open(f"{PREP}/.lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)   # a 2nd prep waits, then skips all done
    # rev = helper worker from the END, no lock; stops 3 files before the forward worker's contiguous prefix
    # Meeting-point heuristic; a rare 1-file overlap is harmless (deterministic output, own tmp name).
    order = range(len(todo) - 1, -1, -1) if rev else range(len(todo))
    for i in order:
        split, label, f = todo[i]
        out = outp(todo[i])
        if rev and (os.path.exists(out) or any(os.path.exists(outp(todo[j])) for j in range(max(0, i - 3), i))):
            print(f"  rev prep met forward worker at {i}", flush=True); break
        if os.path.exists(out):
            continue
        cool_gate(pause=82.0, resume=74.0, abort=92.0)   # every file, stricter: prep spikes this box to ~93 C
        if i % 25 == 0:
            print(f"  prep {i}/{len(todo)}", flush=True)
        raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
        raw.rename_channels({k: v for k, v in CH_MAP.items() if k in raw.ch_names})
        raw.pick(P.CHANNELS).reorder_channels(P.CHANNELS)
        raw.crop(tmax=min(CROP_S, raw.times[-1])).load_data(verbose=False)
        d = raw.filter(0.5, 40.0, verbose=False).resample(200, verbose=False).get_data()
        d = (d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)
        fr = d[:, :d.shape[1] // 500 * 500].reshape(19, -1, 500).transpose(1, 0, 2).astype(np.float32)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out[:-4] + (".tmprev.npy" if rev else ".tmp.npy")
        np.save(tmp, fr); os.replace(tmp, out)


def files():   # label: abnormal = 1 (act0 convention)
    return [(f"{PREP}/{s}/{l}/{os.path.basename(f)[:-4]}.npy", int(l == "abnormal"), s) for s, l, f in edfs()]


def embed():
    import torch
    import fei_pretrain as F
    import fei_branchC as C
    from head_to_head_cv import h20_args
    args = h20_args(); C._configure(args)
    fl = [(p, y) for p, y, _ in files()]
    fx = files()   # row index for every emb_*.npy (same order)
    np.savez(f"{OUT}/emb_index.npz", rec=np.array([os.path.basename(p) for p, _, _ in fx]),
             label_abnormal=np.array([y for _, y, _ in fx]), split=np.array([s for _, _, s in fx]))
    for k in ENCODERS:
        pa, pc = f"{OUT}/emb_fei_{k}.npy", f"{OUT}/emb_c_{k}.npy"
        if os.path.exists(pa) and os.path.exists(pc):
            continue
        torch.manual_seed(int(k[-1]))
        a, c = F.Encoder(args.d).to(F.DEV), C._enc(args)(args.d).to(F.DEV)
        if k.startswith("f"):
            a.load_state_dict(torch.load(f"{ROOT}/code/fei{TAG}_enc_cv_s0_{k}.pt", map_location=F.DEV))
            c.load_state_dict(torch.load(f"{ROOT}/code/branchC_h20{TAG}_enc_cv_s0_{k}.pt", map_location=F.DEV))
        a.eval(); c.eval()
        print(f"  embed {k}", flush=True)
        np.save(pa, F.embed_set(a, fl, C.L)[0]); np.save(pc, C.embed_dyn(c, fl)[0])


def proba(Xtr, ytr, Xte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def probe():
    fl = files()
    y = np.array([l for _, l, _ in fl]); split = np.array([s for _, _, s in fl])
    base = [os.path.basename(p) for p, _, _ in fl]
    ev = np.where(split == "eval")[0]
    idx = {b: i for i, b in enumerate(base)}
    trains = {f"s{s}": [idx[os.path.basename(f)] for f in sorted(glob.glob(f"{ROOT}/data/TUAB_subsets/s{s}/train/*/*.npy"))]
              for s in range(5)} | {"full_train": list(np.where(split == "train")[0])}
    E = {f"{b}_{k}": np.load(f"{OUT}/emb_{b}_{k}.npy") for b in ("fei", "c") for k in ENCODERS}
    feat = lambda arm, k: {"FEI": E[f"fei_{k}"], "C": E[f"c_{k}"], "FEI+C": np.hstack([E[f"fei_{k}"], E[f"c_{k}"]])}[arm]
    col = {}
    for k, X in E.items():
        Xe = X[ev]; Xn = Xe / np.linalg.norm(Xe, axis=1, keepdims=True)
        col[k] = dict(rec_std=float(Xe.std(0).mean()), rec_cos=float((Xn @ Xn.T)[np.triu_indices(len(Xe), 1)].mean()))
    save(f"{OUT}/collapse.json", col)
    res = {}
    for tname, tr in trains.items():
        cool_gate(pause=88.0, resume=78.0, abort=92.0)
        for arm in ("FEI", "C", "FEI+C"):
            for kind, ks in (("", ENCODERS[:5]), ("rand-", ENCODERS[5:])):
                P_ = np.array([proba(feat(arm, k)[tr], y[tr], feat(arm, k)[ev]) for k in ks])
                p = P_.mean(0); au = [metrics(y[ev], q)["auroc"] for q in P_]
                r = metrics(y[ev], p) | dict(member_auroc=au, member_auroc_mean=float(np.mean(au)),
                                             member_auroc_sd=float(np.std(au)), n_train=len(tr))
                if tname == "full_train":
                    r |= dict(auroc_ci=bootstrap_ci(y[ev], p, "auroc"), bal_acc_ci=bootstrap_ci(y[ev], p, "bal_acc"))
                res[f"{tname}/{kind}{arm}"] = r
                np.savez(f"{OUT}/preds_{tname}_{kind}{arm}.npz", prob_abnormal=p, members=P_, label=y[ev],
                         rec=np.array(base)[ev])
                print(f"{tname:10s} {kind + arm:11s} {fmt(r)}  members {np.mean(au):.3f}±{np.std(au):.3f}", flush=True)
    summ = {}
    for arm in ("FEI", "C", "FEI+C", "rand-FEI", "rand-C", "rand-FEI+C"):
        v = np.array([[res[f"s{s}/{arm}"][q] for q in ("auroc", "auc_pr", "acc", "bal_acc", "f1")] for s in range(5)])
        summ[arm] = {q: [float(m), float(sd)] for q, m, sd in zip(("auroc", "auc_pr", "acc", "bal_acc", "f1"), v.mean(0), v.std(0))}
        print(f"546-subsets {arm:11s} AUROC {v[:, 0].mean():.3f}±{v[:, 0].std():.3f}  Acc {v[:, 2].mean():.3f}  "
              f"BAcc {v[:, 3].mean():.3f}  F1 {v[:, 4].mean():.3f}", flush=True)
    save(f"{OUT}/results.json", dict(runs=res, subsets_summary=summ, crop_s=CROP_S))


def main():
    os.makedirs(OUT, exist_ok=True)
    prep(rev="--prep-rev" in sys.argv)
    if "--prep" in sys.argv or "--prep-rev" in sys.argv:   # CPU-only lane
        return print("TUAB FEI+C PREP DONE", flush=True)
    embed(); probe()
    print("TUAB FEI+C DONE", flush=True)


if __name__ == "__main__":
    main()
