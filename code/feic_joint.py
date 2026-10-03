"""Act-0: FEI+C pretrained on NMT + TUAB, replacing NMT-only pretraining as a second encoder set.
Per NMT seed-0 fold k: pretrain PLAIN FEI (never Ada-FEI; fei_pretrain defaults, L=1000, 60 ep) and Branch-C h20
(L=4000 hop=100, 60 ep) on NMT train-fold-k  +  TUAB TRAIN (2717). Held out, never seen even unlabeled:
NMT test fold k, TUAB official eval, all stroke data. Same recipes/epochs as the NMT-only encoders.
  -> code/fei_joint_enc_cv_s0_f{k}.pt, code/branchC_h20_joint_enc_cv_s0_f{k}.pt
Then NMT CV eval identical to feic_nmt_anchor (A0): FEI@4000, C, FEI+C, paired vs the NMT-only A0 preds.
  -> code/runs/act0/a0_feic_nmt_joint/{results.json, preds_f*.npz, emb_f*.npz}
TUAB + stroke evals reuse tuab_feic.py / stroke_phaseB.py / tuab_vitm_linear.py with FEIC_TAG=_joint.
  python feic_joint.py            # resumable per fold (ckpts + results written atomically)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads
cap_threads(4)
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import fei_pretrain as F
import fei_branchC as C
from act0_metrics import fmt, metrics, save
from feic_nmt_anchor import proba
from head_to_head_cv import h20_args
import json

CODE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{CODE}/runs/act0/a0_feic_nmt_joint"
A0 = f"{CODE}/runs/act0/a0_feic_nmt"
TUAB = "/home/mashfiq/eeg_vjepa/data/TUAB_feic_200hz"


def tuab_train():
    fs = sorted(f"{TUAB}/train/{l}/{f}" for l in ("normal", "abnormal") for f in os.listdir(f"{TUAB}/train/{l}")
                if f.endswith(".npy") and "tmp" not in f)
    assert len(fs) == 2717, len(fs)   # official train only; eval never enters pretraining
    return [(f, 0) for f in fs]       # SSL: labels unused


def recipe(**kw):   # the fei_pretrain / fei_branchC argparse defaults (what the NMT-only encoders used)
    from types import SimpleNamespace
    return SimpleNamespace(**(dict(epochs=60, batch=128, wpe=4, d=256, h=128, lr=2e-4, ema=0.996,
                                   mask_bias=0.0, mask_ema=0.98) | kw))


def save_pt(sd, path):
    torch.save(sd, path + ".tmp"); os.replace(path + ".tmp", path)


def pretrain(k, trf):
    pa, pc = f"{CODE}/fei_joint_enc_cv_s0_f{k}.pt", f"{CODE}/branchC_h20_joint_enc_cv_s0_f{k}.pt"
    files = trf + tuab_train()
    if not os.path.exists(pa):
        torch.manual_seed(k)
        save_pt(F.pretrain(files, recipe(L=1000), tag=f"[f{k} FEI] ").state_dict(), pa)
    if not os.path.exists(pc):
        torch.manual_seed(k)
        h = h20_args(); C._configure(h)
        save_pt(C.pretrain_C(files, recipe(**vars(h)), tag=f"[f{k} C] ").state_dict(), pc)
    return pa, pc


def main():
    os.makedirs(OUT, exist_ok=True)
    args = h20_args(); C._configure(args)
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    res = json.load(open(f"{OUT}/results.json")) if os.path.exists(f"{OUT}/results.json") else {}
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
        trf, tef = [files[i] for i in tr], [files[i] for i in te]
        pa, pc = pretrain(k, trf)
        if str(k) in res:
            continue
        a = F.Encoder(args.d).to(F.DEV); a.load_state_dict(torch.load(pa, map_location=F.DEV))
        c = C._enc(args)(args.d).to(F.DEV); c.load_state_dict(torch.load(pc, map_location=F.DEV))
        a.eval(); c.eval()
        A = F.embed_set(a, trf, C.L)[0], F.embed_set(a, tef, C.L)[0]
        Cc = C.embed_dyn(c, trf)[0], C.embed_dyn(c, tef)[0]
        P = {"FEI@4000": proba(A[0], y[tr], A[1]), "C": proba(Cc[0], y[tr], Cc[1]),
             "FEI+C": proba(np.hstack([A[0], Cc[0]]), y[tr], np.hstack([A[1], Cc[1]]))}
        ref = np.load(f"{A0}/preds_f{k}.npz")
        assert list(ref["rec"]) == [p for p, _ in tef]   # same fold, same order -> paired
        np.savez(f"{OUT}/emb_f{k}.npz", rec_tr=np.array([p for p, _ in trf]), rec_te=np.array([p for p, _ in tef]),
                 y_tr=y[tr], y_te=y[te], fei4000_tr=A[0], fei4000_te=A[1], c_tr=Cc[0], c_te=Cc[1])
        np.savez(f"{OUT}/preds_f{k}.npz", rec=np.array([p for p, _ in tef]), label=y[te], **P)
        res[str(k)] = {m: metrics(y[te], p) | {"nmt_only_auroc": metrics(y[te], ref[m])["auroc"]} for m, p in P.items()}
        res[str(k)]["C_emb_std"] = float(Cc[1].std(0).mean()); res[str(k)]["FEI_emb_std"] = float(A[1].std(0).mean())
        save(f"{OUT}/results.json", res)
        print(f"[f{k}] joint FEI+C {fmt(res[str(k)]['FEI+C'])}  (NMT-only {res[str(k)]['FEI+C']['nmt_only_auroc']:.3f})",
              flush=True)
    for m in ("FEI@4000", "C", "FEI+C"):
        j = np.array([res[str(k)][m]["auroc"] for k in range(5)])
        n = np.array([res[str(k)][m]["nmt_only_auroc"] for k in range(5)])
        res.setdefault("summary", {})[m] = dict(joint=[float(j.mean()), float(j.std())], nmt_only=[float(n.mean()), float(n.std())],
                                                delta=float((j - n).mean()), wins=int((j > n).sum()))
        print(f"NMT CV {m:9s} joint {j.mean():.3f}±{j.std():.3f}  NMT-only {n.mean():.3f}±{n.std():.3f}  "
              f"delta {(j - n).mean():+.3f} ({int((j > n).sum())}/5)", flush=True)
    save(f"{OUT}/results.json", res)
    print("FEIC JOINT DONE", flush=True)


if __name__ == "__main__":
    main()
