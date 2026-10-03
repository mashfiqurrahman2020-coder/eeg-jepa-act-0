"""
FEI (Frequency-masked Embedding Inference) on NMT normal/abnormal -- option "B".

A frequency-aware JEPA: single time-domain encoder, no negatives. The pretext
task removes random frequency bands (via rFFT) from a window and asks the online
encoder to predict the *momentum* encoder's embedding of the frequency-masked
window, given the original embedding + a prompt saying which bands were removed.
The idea (Frequency-Masked Embedding Inference, AAAI'25): force the latent to be
sensitive to spectral content -- the exact thing plain V-JEPA on NMT ignored,
and the reason the band-power baseline beat it.

Bar to beat: baseline_bandpower.py -> AUC ~0.79-0.86 on the eval split.

Run:
  python fei_pretrain.py --selftest              # fast sanity check, no data
  python fei_pretrain.py --epochs 60             # pretrain, then linear-probe
  python fei_pretrain.py --probe-only --ckpt fei_enc.pt   # just re-probe a ckpt
"""
import os, sys, glob, time, json, argparse, random

# hw guard FIRST -- cap_threads sets BLAS env vars before numpy/torch import.
sys.path.append("/home/mashfiq/mwi_stroke_experiments")
sys.path.append("/home/mashfiq/mwi_stroke_experiments/lib")  # hw_guard moved to lib/
from hw_guard import cool_gate, cap_threads
cap_threads(4)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score

DATA = "/home/mashfiq/eeg_vjepa/data/NMT_preprocessed"
FS, NCH = 200, 19
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- data -------------------------
def list_split(split):
    files = []
    for lab, cls in (("normal", 0), ("abnormal", 1)):
        for p in sorted(glob.glob(f"{DATA}/{split}/{lab}/*.npy")):
            files.append((p, cls))
    return files


def load_continuous(path):
    """(F,19,500) z-scored frames -> full continuous (19, F*500) float32.
    Used only by the probe (one recording at a time -> bounded memory)."""
    x = np.load(path)                              # (F, 19, 500)
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1).astype(np.float32)


def load_window(path, L):
    """Random length-L continuous window, reading ONLY the 2-3 frames it spans
    via mmap (~114KB), not the whole recording. This is the speedup + the OOM fix:
    per-item memory is bounded regardless of recording length."""
    x = np.load(path, mmap_mode="r")               # (F,19,500), no full read
    Fr, C, W = x.shape
    T = Fr * W
    if T < L:                                      # short -> wrap-pad (rare)
        cont = np.asarray(x).transpose(1, 0, 2).reshape(C, -1)
        return np.ascontiguousarray(np.tile(cont, (1, L // T + 1))[:, :L]).astype(np.float32)
    s = random.randint(0, T - L)
    f0, f1 = s // W, (s + L - 1) // W              # frames covering [s, s+L)
    chunk = np.asarray(x[f0:f1 + 1]).transpose(1, 0, 2).reshape(C, -1)  # (19, nf*500)
    off = s - f0 * W
    return np.ascontiguousarray(chunk[:, off:off + L]).astype(np.float32)


class Windows(Dataset):
    """One random L-window per (recording x wpe). Labels ignored (SSL)."""
    def __init__(self, files, L, wpe=4):
        self.files = [p for p, _ in files]
        self.L, self.wpe = L, wpe

    def __len__(self):
        return len(self.files) * self.wpe

    def __getitem__(self, i):
        return torch.from_numpy(load_window(self.files[i % len(self.files)], self.L))


# ------------------------- model -------------------------
class Encoder(nn.Module):
    """(B,19,L) -> (B,d). Small strided 1D conv stack + global average pool."""
    def __init__(self, d=256, ch=NCH):
        super().__init__()
        w = [ch, 64, 128, 128, 256]
        blocks = []
        for a, b in zip(w[:-1], w[1:]):
            blocks += [nn.Conv1d(a, b, 7, stride=2, padding=3), nn.BatchNorm1d(b), nn.GELU()]
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Linear(w[-1], d)

    def forward(self, x):
        h = self.conv(x).mean(-1)          # global avg pool over time
        return self.head(h)


def mlp(h):
    return nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))


class FEI(nn.Module):
    def __init__(self, n_freq, d=256, h=128, in_ch=NCH, enc_factory=None):
        # n_freq = mask-prompt dim (freq bins for Branch A, time positions for Branch B);
        # in_ch = encoder input channels (19 time / 38 complex-spectrum for Branch B).
        # enc_factory: 0-arg callable -> encoder module (default = conv Encoder); lets
        # Branch B swap in a frequency-token transformer without touching the recipe.
        super().__init__()
        mk = enc_factory or (lambda: Encoder(d, ch=in_ch))
        self.enc, self.proj = mk(), nn.Linear(d, h)                          # online
        self.enc_t, self.proj_t = mk(), nn.Linear(d, h)                      # momentum target
        self._sync(); [p.requires_grad_(False) for p in self._tgt_params()]
        # mask prompt encoder: (M in {0,1}^n) -> h,  W ~ N(0,1)/sqrt(n)
        self.mask_enc = nn.Linear(n_freq, h, bias=False)
        nn.init.normal_(self.mask_enc.weight, std=1.0 / np.sqrt(n_freq))
        self.z1, self.z2 = mlp(h), mlp(h)                          # predictors

    def _tgt_params(self):
        return list(self.enc_t.parameters()) + list(self.proj_t.parameters())

    def _src_params(self):
        return list(self.enc.parameters()) + list(self.proj.parameters())

    @torch.no_grad()
    def _sync(self):
        for s, t in zip(self._src_params(), self._tgt_params()):
            t.copy_(s)

    @torch.no_grad()
    def ema(self, a=0.996):
        for s, t in zip(self._src_params(), self._tgt_params()):
            t.mul_(a).add_(s, alpha=1 - a)

    def forward(self, x, xp, M):
        e = self.enc(x); u = self.proj(e)                 # online, original window
        with torch.no_grad():
            up = self.proj_t(self.enc_t(xp))              # momentum, masked window
        m = self.mask_enc(M)
        uhat = self.z1(u + m.detach())                    # predict masked embedding
        mhat = self.z2(u.detach() - up)                   # predict which bands removed
        # L2-normalize pred & target before MSE (BYOL-style) -> bounded loss, no
        # scale collapse. Equivalent to 2 - 2*cos(pred, target).
        nrm = lambda a: F.normalize(a, dim=-1)
        l_embed = F.mse_loss(nrm(uhat), nrm(up), reduction="none").mean(-1)  # (B,) per-sample
        loss = l_embed.mean() + F.mse_loss(nrm(mhat), nrm(m))
        return loss, e, l_embed.detach()                  # psl feeds the Ada difficulty map


def freq_mask(x, d=None, bias=0.0, b1=0.0, b2=0.7):
    """Zero a random U(b1,b2) fraction of rFFT bins (shared across channels).
    Ada-FEI: when bias>0, re-weight per-bin masking toward high-difficulty bins
    using the EMA difficulty map d (n,) -- the adaptive/curriculum frequency
    masking that plain FEI (bias=0, uniform) lacks. bias=0 -> IDENTICAL to plain
    FEI (clean ablation). E[fraction masked] stays ~ratio regardless of bias, so
    the knob changes WHERE bands are masked, not how many.
    Returns masked signal x' (B,19,L) and mask M (B,n) in {0,1}."""
    B, _, L = x.shape
    Xf = torch.fft.rfft(x, dim=-1)                        # (B,19,n) complex
    n = Xf.shape[-1]
    ratio = torch.empty(B, 1, device=x.device).uniform_(b1, b2)
    if bias and d is not None and float(d.std()) > 0:
        z = ((d - d.mean()) / (d.std() + 1e-6)).clamp(-5, 5)   # standardized difficulty, capped
        w = torch.exp(bias * z)                                # emphasize hard/informative bins
        p = (ratio * w / w.mean()).clamp(0, 1)                 # (B,n), keeps E[frac]~ratio
    else:
        p = ratio                                             # uniform == plain FEI
    M = (torch.rand(B, n, device=x.device) < p).float()   # 1 = masked
    xp = torch.fft.irfft(Xf * (1 - M).unsqueeze(1), n=L, dim=-1)
    return xp.float(), M


# ------------------------- train + probe -------------------------
def embed_recording(enc, path, L):
    """Mean-pooled encoder embedding over non-overlapping L-windows."""
    sig = load_continuous(path)
    T = sig.shape[1]
    if T < L:
        sig = np.tile(sig, (1, L // T + 1))[:, :L]; T = L
    starts = range(0, T - L + 1, L)
    w = np.stack([sig[:, s:s + L] for s in starts])      # (W,19,L)
    with torch.no_grad():
        e = enc(torch.from_numpy(w).to(DEV)).mean(0)      # (d,)
    return e.cpu().numpy()


def embed_set(enc, files, L):
    """files: list of (path,label) -> (X, y) mean-pooled embeddings."""
    X, y = [], []
    for i, (p, c) in enumerate(files):
        if i % 100 == 0: cool_gate(verbose=False)
        X.append(embed_recording(enc, p, L)); y.append(c)
    return np.array(X), np.array(y)


def fit_probe(Xtr, ytr, Xte, yte, pr=False):
    """StandardScaler + balanced LogReg C=1.0 -> (auc, bal-acc). pr=True appends AUC-PR
    (sklearn average_precision_score) = the SOTA binary metric set {AUROC, BAcc, AUC-PR}.
    Default 2-tuple kept so existing callers are unchanged."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(sc.transform(Xtr), ytr)
    s = clf.decision_function(sc.transform(Xte))
    auc = roc_auc_score(yte, s); bacc = balanced_accuracy_score(yte, (s > 0).astype(int))
    return (auc, bacc, average_precision_score(yte, s)) if pr else (auc, bacc)


def probe(enc, L):
    enc.eval()
    Xtr, ytr = embed_set(enc, list_split("train"), L)
    Xev, yev = embed_set(enc, list_split("eval"), L)
    auc, bacc = fit_probe(Xtr, ytr, Xev, yev)
    print(f"[probe] eval AUC={auc:.3f}  balanced-acc={bacc:.3f}   (baseline bar: 0.79-0.86 AUC)")
    return auc


def pretrain(files, args, tag=""):
    """Train a fresh FEI on `files` (SSL, labels ignored); return the online encoder."""
    # per-item is now ~114KB (mmap, few frames) -> 4 workers safe (~0.1GB in flight).
    dl = DataLoader(Windows(files, args.L, args.wpe), batch_size=args.batch, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    n_freq = args.L // 2 + 1
    model = FEI(n_freq, d=args.d, h=args.h).to(DEV)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    d = torch.zeros(n_freq, device=DEV)          # Ada-FEI EMA per-band difficulty (uniform at start)
    mode = f"Ada-FEI bias={args.mask_bias}" if args.mask_bias else "FEI (uniform mask)"
    print(f"{tag}{mode} on {len(files)} recs: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"params L={args.L} n_freq={n_freq} steps/epoch={len(dl)} dev={DEV}", flush=True)
    for ep in range(args.epochs):
        model.train(); t0, tot = time.time(), 0.0
        for step, x in enumerate(dl):
            if step % 20 == 0: cool_gate(verbose=False)
            x = x.to(DEV, non_blocking=True)
            xp, M = freq_mask(x, d, args.mask_bias)
            loss, _, psl = model(x, xp, M)
            opt.zero_grad(); loss.backward(); opt.step(); model.ema(args.ema)
            if args.mask_bias:                    # update difficulty map = E[loss | bin masked], no extra fwd
                cnt = M.sum(0)                     # (n,) times each bin was masked this batch
                upd = (M * psl.unsqueeze(1)).sum(0) / cnt.clamp(min=1)
                hit = cnt > 0
                d = torch.where(hit, args.mask_ema * d + (1 - args.mask_ema) * upd, d)
            tot += loss.item()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"{tag}  epoch {ep+1}/{args.epochs}  loss={tot/len(dl):.4f}  {time.time()-t0:.0f}s", flush=True)
    return model.enc


def train(args):
    enc = pretrain(list_split("train"), args)
    torch.save(enc.state_dict(), args.ckpt)
    print("saved encoder ->", args.ckpt)
    probe(enc, args.L)


def run_cv(args):
    """Subject-independent 5-fold CV matching baseline_ci.py: pooled 2417 recs,
    same seed/order -> identical folds. Pretrain FEI per-fold on TRAIN folds only
    (no test recording seen, even unlabeled) -> honest vs band-power 0.786.
    RESUMABLE: per-fold results persisted atomically; completed folds skipped on restart."""
    from sklearn.model_selection import StratifiedKFold
    state_path = os.path.splitext(args.ckpt)[0] + f"_cv_s{args.cv_seed}_state.json"  # per-ckpt+seed -> runs never collide
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    if state:
        print(f"resuming CV: folds already done = {sorted(int(k)+1 for k in state)}", flush=True)
    files = list_split("train") + list_split("eval")     # pooled, same order as band-power
    y = np.array([c for _, c in files])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.cv_seed)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        if str(k) in state:
            print(f"[f{k+1}] done (AUC={state[str(k)]['auc']:.3f}) -> skip", flush=True); continue
        tr_files = [files[i] for i in tr]; te_files = [files[i] for i in te]
        print(f"\n===== FOLD {k+1}/5  (train {len(tr)} / test {len(te)}) =====", flush=True)
        enc = pretrain(tr_files, args, tag=f"[f{k+1}] "); enc.eval()
        torch.save(enc.state_dict(), args.ckpt.replace(".pt", f"_cv_s{args.cv_seed}_f{k}.pt"))
        Xtr, ytr = embed_set(enc, tr_files, args.L)
        Xte, yte = embed_set(enc, te_files, args.L)
        auc, bacc = fit_probe(Xtr, ytr, Xte, yte)
        print(f"[f{k+1}] test AUC={auc:.3f}  bal-acc={bacc:.3f}", flush=True)
        state[str(k)] = {"auc": float(auc), "bacc": float(bacc)}
        tmp = state_path + ".tmp"                          # atomic write -> crash can't corrupt state
        json.dump(state, open(tmp, "w"), indent=2); os.replace(tmp, state_path)
    aucs = np.array([state[str(k)]["auc"] for k in range(5)])
    baccs = np.array([state[str(k)]["bacc"] for k in range(5)])
    print(f"\n===== FEI 5-fold CV (n={len(y)}) =====")
    print(f"  AUC     = {aucs.mean():.3f} +/- {aucs.std():.3f}   folds {np.round(aucs,3)}")
    print(f"  bal-acc = {baccs.mean():.3f} +/- {baccs.std():.3f}")
    print(f"  >>> band-power CV bar = 0.786 +/- 0.009  (same folds, paired) <<<")
    print(f"  state: {state_path}", flush=True)


# ------------------------- selftest -------------------------
def selftest():
    torch.manual_seed(0)
    # freq_mask must remove energy at the masked band: 10 Hz sine, mask its bin.
    L = 400; t = torch.arange(L) / FS
    x = torch.sin(2 * np.pi * 10 * t).repeat(NCH, 1)[None]        # (1,19,L)
    Xf = torch.fft.rfft(x, dim=-1); bin10 = int(round(10 * L / FS))
    M = torch.zeros(1, Xf.shape[-1]); M[0, bin10] = 1
    xp = torch.fft.irfft(Xf * (1 - M).unsqueeze(1), n=L, dim=-1)
    assert xp.abs().max() < 1e-4, f"masking 10Hz bin should null the sine, got {xp.abs().max():.3e}"
    # model + loss step decreases on a fixed tiny batch.
    m = FEI(L // 2 + 1, d=32, h=16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    xb = torch.randn(8, NCH, L)
    xpb, Mb = freq_mask(xb)
    l0 = m(xb, xpb, Mb)[0].item()
    for _ in range(40):
        l, _, _ = m(xb, xpb, Mb); opt.zero_grad(); l.backward(); opt.step(); m.ema(0.9)
    l1 = m(xb, xpb, Mb)[0].item()
    assert l1 < l0, f"loss did not drop: {l0:.4f} -> {l1:.4f}"
    assert np.isfinite(l1)
    # Ada-FEI: a high-difficulty bin must get masked more than under uniform sampling.
    xbig = torch.randn(500, NCH, L)
    diff = torch.zeros(L // 2 + 1); diff[bin10] = 10.0
    _, Mu = freq_mask(xbig, diff, bias=0.0)
    _, Ma = freq_mask(xbig, diff, bias=4.0)
    assert Ma[:, bin10].mean() > Mu[:, bin10].mean() + 0.1, \
        f"biased mask should hit the hard bin more: {Mu[:,bin10].mean():.2f} vs {Ma[:,bin10].mean():.2f}"
    print(f"selftest OK: mask nulls band; loss {l0:.4f} -> {l1:.4f}; "
          f"Ada bias {Mu[:,bin10].mean():.2f}->{Ma[:,bin10].mean():.2f} on hard bin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--L", type=int, default=1000)      # 5 s @ 200 Hz -> 0.2 Hz bins
    ap.add_argument("--wpe", type=int, default=4)        # random windows per recording / epoch
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--mask_bias", type=float, default=0.0,
                    help="Ada-FEI: >0 biases masking toward high-difficulty bands (EMA map); 0 = plain FEI")
    ap.add_argument("--mask_ema", type=float, default=0.98, help="EMA rate for the per-band difficulty map")
    ap.add_argument("--ckpt", default="/home/mashfiq/eeg_vjepa/code/fei_enc.pt")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--cv", action="store_true", help="subject-independent 5-fold CV vs band-power")
    ap.add_argument("--cv_seed", type=int, default=0, help="StratifiedKFold seed; run 0/1/2 for the 15-fold power-up")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    elif args.cv:
        run_cv(args)
    elif args.probe_only:
        enc = Encoder(args.d).to(DEV)
        enc.load_state_dict(torch.load(args.ckpt, map_location=DEV))
        probe(enc, args.L)
    else:
        train(args)
