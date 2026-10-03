"""Act-0 A0: FEI+C load-path anchor on NMT (the gate before FEI+C is used on stroke / TUAB).
Same seed-0 5-fold CV + the SAME per-fold encoders as the records. GATE: plain FEI (L=1000, fit_probe) must
reproduce fei_enc_cv_s0_state.json fold AUROCs to 1e-3 (0.858/.795/.847/.787/.804). Then report the Act-0
FEI+C config (FEI at L=4000 as used downstream, C = Branch-C h20, fusion = hstack) with full act0 metrics.
  python feic_nmt_anchor.py  -> code/runs/act0/a0_feic_nmt/{results.json, preds_f*.npz}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads
cap_threads(4)
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fei_pretrain as F
import fei_branchC as C
from act0_metrics import fmt, metrics, save
from head_to_head_cv import h20_args

CODE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{CODE}/runs/act0/a0_feic_nmt"


def proba(Xtr, ytr, Xte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = json.load(open(f"{CODE}/fei_enc_cv_s0_state.json"))
    args = h20_args(); C._configure(args)
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    res = json.load(open(f"{OUT}/results.json")) if os.path.exists(f"{OUT}/results.json") else {}
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
        if str(k) in res:
            continue
        trf, tef = [files[i] for i in tr], [files[i] for i in te]
        a = F.Encoder(args.d).to(F.DEV); a.load_state_dict(torch.load(f"{CODE}/fei_enc_cv_s0_f{k}.pt", map_location=F.DEV))
        c = C._enc(args)(args.d).to(F.DEV)
        c.load_state_dict(torch.load(f"{CODE}/branchC_h20_enc_cv_s0_f{k}.pt", map_location=F.DEV))
        a.eval(); c.eval()
        G = F.embed_set(a, trf, 1000)[0], F.embed_set(a, tef, 1000)[0]
        gate = F.fit_probe(G[0], y[tr], G[1], y[te])[0]
        A = F.embed_set(a, trf, C.L)[0], F.embed_set(a, tef, C.L)[0]
        Cc = C.embed_dyn(c, trf)[0], C.embed_dyn(c, tef)[0]
        P = {"FEI@4000": proba(A[0], y[tr], A[1]), "C": proba(Cc[0], y[tr], Cc[1]),
             "FEI+C": proba(np.hstack([A[0], Cc[0]]), y[tr], np.hstack([A[1], Cc[1]]))}
        np.savez(f"{OUT}/emb_f{k}.npz", rec_tr=np.array([p for p, _ in trf]), rec_te=np.array([p for p, _ in tef]),
                 y_tr=y[tr], y_te=y[te], fei1000_tr=G[0], fei1000_te=G[1], fei4000_tr=A[0], fei4000_te=A[1],
                 c_tr=Cc[0], c_te=Cc[1])   # embeddings -> any later analysis without the GPU
        np.savez(f"{OUT}/preds_f{k}.npz", rec=np.array([p for p, _ in tef]), label=y[te], **P)
        res[str(k)] = {"gate_fei_L1000_auroc": float(gate), "gate_ref": ref[str(k)]["auc"],
                       "gate_ok": bool(abs(gate - ref[str(k)]["auc"]) < 1e-3)} | {m: metrics(y[te], p) for m, p in P.items()}
        save(f"{OUT}/results.json", res)
        print(f"[f{k}] gate FEI@1000 {gate:.4f} (ref {ref[str(k)]['auc']:.4f}) "
              f"{'OK' if res[str(k)]['gate_ok'] else 'MISMATCH'}  FEI+C {fmt(res[str(k)]['FEI+C'])}", flush=True)
    summ = {m: {q: [float(np.mean([res[str(k)][m][q] for k in range(5)])), float(np.std([res[str(k)][m][q] for k in range(5)]))]
                for q in ("auroc", "auc_pr", "bal_acc", "acc", "f1", "sens", "spec")} for m in ("FEI@4000", "C", "FEI+C")}
    res["summary"] = summ | {"gate_pass": all(res[str(k)]["gate_ok"] for k in range(5))}
    save(f"{OUT}/results.json", res)
    for m, s in summ.items():
        print(f"{m:9s} AUROC {s['auroc'][0]:.3f}±{s['auroc'][1]:.3f}  BAcc {s['bal_acc'][0]:.3f}  AUC-PR {s['auc_pr'][0]:.3f}")
    print("GATE A0:", "PASS" if res["summary"]["gate_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
