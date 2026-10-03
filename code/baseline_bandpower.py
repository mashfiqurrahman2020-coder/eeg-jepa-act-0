"""
Cheap spectral baseline for NMT normal/abnormal: relative band power (+ optional
interhemispheric asymmetry) -> class-weighted LogReg / SVM. Pure CPU.

The real bar any SSL/JEPA model must beat is hand-crafted spectral features, NOT
random init. This measures that bar. Data is already per-channel z-scored, so
"power" here is spectral SHAPE (relative band power), independent of amplitude.

Run:  python baseline_bandpower.py            # full baseline
      python baseline_bandpower.py --selftest # 10 Hz sine -> alpha must dominate
"""
import os, sys, glob, argparse
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Reuse the project's hardware guard (same as preprocess_nmt.py) -- do not reinvent.
sys.path.append("/home/mashfiq/mwi_stroke_experiments")
sys.path.append("/home/mashfiq/mwi_stroke_experiments/lib")  # hw_guard moved to lib/
from hw_guard import cool_gate, cap_threads
cap_threads(4)

FS = 200
NPERSEG = 500                       # 2.5 s -> 0.4 Hz resolution
DATA = "/home/mashfiq/eeg_vjepa/data/NMT_preprocessed"
CACHE = "/home/mashfiq/eeg_vjepa/code/bandpower_features.npz"
BANDS = [("delta",0.5,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,45)]
# 8 bilateral pairs by channel index (order fixed in preprocess_nmt.py CHANNELS)
PAIRS = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]


def relative_bandpower(sig):
    """sig: (n_ch, n_samples) -> (n_ch, n_bands) relative band power (rows sum ~1)."""
    f, pxx = welch(sig, fs=FS, nperseg=NPERSEG, axis=-1)   # (n_ch, n_freqs)
    total = np.trapz(pxx[:, (f >= 0.5) & (f <= 45)], axis=-1) + 1e-12
    bp = np.stack([np.trapz(pxx[:, (f >= lo) & (f < hi)], axis=-1) for _, lo, hi in BANDS], axis=-1)
    return bp / total[:, None]


def features_for_recording(path):
    x = np.load(path)                       # (F, 19, 500) z-scored frames
    sig = x.transpose(1, 0, 2).reshape(x.shape[1], -1)   # back to continuous (19, F*500)
    rbp = relative_bandpower(sig)           # (19, 5)
    asym = np.array([(rbp[l] - rbp[r]) / (rbp[l] + rbp[r] + 1e-9) for l, r in PAIRS])  # (8, 5)
    return rbp.ravel(), asym.ravel()        # 95, 40


def load_split(split):
    Xb, Xa, y = [], [], []
    files = []
    for lab, cls in (("normal", 0), ("abnormal", 1)):
        for p in sorted(glob.glob(f"{DATA}/{split}/{lab}/*.npy")):
            files.append((p, cls))
    for i, (p, cls) in enumerate(files):
        if i % 100 == 0:
            cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)
            print(f"  {split}: {i}/{len(files)}", flush=True)
        b, a = features_for_recording(p)
        Xb.append(b); Xa.append(a); y.append(cls)
    return np.array(Xb), np.array(Xa), np.array(y)


def build_features():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return (d["Xb_tr"], d["Xa_tr"], d["y_tr"], d["Xb_ev"], d["Xa_ev"], d["y_ev"])
    Xb_tr, Xa_tr, y_tr = load_split("train")
    Xb_ev, Xa_ev, y_ev = load_split("eval")
    np.savez(CACHE, Xb_tr=Xb_tr, Xa_tr=Xa_tr, y_tr=y_tr, Xb_ev=Xb_ev, Xa_ev=Xa_ev, y_ev=y_ev)
    return Xb_tr, Xa_tr, y_tr, Xb_ev, Xa_ev, y_ev


def evaluate(name, Xtr, ytr, Xev, yev, clf):
    sc = StandardScaler().fit(Xtr)
    clf.fit(sc.transform(Xtr), ytr)
    def scores(X):
        return clf.decision_function(X) if hasattr(clf, "decision_function") else clf.predict_proba(X)[:, 1]
    str_, sev = scores(sc.transform(Xtr)), scores(sc.transform(Xev))
    print(f"{name:38s} train_auroc={roc_auc_score(ytr,str_):.3f}  "
          f"eval_auroc={roc_auc_score(yev,sev):.3f}  "
          f"eval_balacc={balanced_accuracy_score(yev, (sev>np.median(str_)).astype(int)):.3f}")


def main():
    Xb_tr, Xa_tr, y_tr, Xb_ev, Xa_ev, y_ev = build_features()
    maj = max(y_ev.mean(), 1 - y_ev.mean())
    print(f"\ntrain n={len(y_tr)} ({int(y_tr.sum())} abn)  eval n={len(y_ev)} "
          f"({int(y_ev.sum())} abn)  eval majority-acc={maj:.3f}\n")
    lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    evaluate("bandpower(95)         LogReg",  Xb_tr, y_tr, Xb_ev, y_ev, lr())
    evaluate("bandpower(95)         SVM-rbf", Xb_tr, y_tr, Xb_ev, y_ev,
             SVC(class_weight="balanced", C=1.0, gamma="scale"))
    Xba_tr = np.hstack([Xb_tr, Xa_tr]); Xba_ev = np.hstack([Xb_ev, Xa_ev])
    evaluate("bandpower+asymmetry(135) LogReg", Xba_tr, y_tr, Xba_ev, y_ev, lr())
    evaluate("asymmetry-only(40)    LogReg",  Xa_tr, y_tr, Xa_ev, y_ev, lr())


def selftest():
    t = np.arange(4000) / FS
    sig = np.vstack([np.sin(2*np.pi*10*t) for _ in range(19)])   # pure 10 Hz -> alpha
    rbp = relative_bandpower(sig)
    assert rbp.shape == (19, 5)
    assert np.argmax(rbp[0]) == 2, f"alpha band should dominate, got band {np.argmax(rbp[0])}"
    assert abs(rbp[0].sum() - 1.0) < 0.05, rbp[0].sum()
    print("selftest ok: 10 Hz sine -> alpha dominates, rows sum to 1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main()
