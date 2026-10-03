"""Branch C — spectral-DYNAMICS JEPA: the one view band-power can't see.

Band-power AVERAGES the spectrum over time. Branch C keeps the STFT spectrogram
(frequency x TIME) and predicts the FUTURE-span embedding from the PAST span =>
temporal-spectral dynamics (bursts, slowing progression, rhythmic modulation).
Structurally NOT shortcut-able (the future frames are never in the context input) =>
fixes the triviality that made Branch B (masked-input) converge to band-power.

Design choices, each earned from a Branch-B failure:
  - past->future prediction (not masked-input) => can't be shortcut, loss stays real.
  - GRU over frames (not a transformer) => param-light, right for ~1900 recs/fold.
  - random-init control + embedding-std in the CV => never trust a probe number alone.

Recipe otherwise = FEI/BYOL: EMA target, stop-grad, L2-normalized MSE (anti-collapse).
Late fusion: freeze A (Ada-FEI) + C (+ band-power) -> LogReg, seed-0 5-fold (same folds).
Reuses fei_pretrain (Encoder/Windows/probe/data) + baseline_bandpower. Resumable per fold.

Run:
  python fei_branchC.py --selftest
  python fei_branchC.py --cv --epochs 60
"""
import os, time, json, argparse
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as NF_
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import fei_pretrain as F
import baseline_bandpower as B

DEV = F.DEV
L = 1000                            # window samples (5 s @200Hz default; longer = more non-stationarity)
NFFT, HOP = 200, 50                 # 1 s STFT window, 0.25 s hop
NT = 1 + (L - NFFT) // HOP          # 17 time-frames
NF = NFFT // 2 + 1                  # 101 freq bins (1 Hz)
TC = 11                            # context frames; predict the last NT-TC future frames
IN_DIM = NF * F.NCH                 # per-frame dim
A_CKPT = "/home/mashfiq/eeg_vjepa/code/adafei_neg1_enc_cv_s0_f{}.pt"   # Branch A = Ada-FEI(-1.0)
STATE = "/home/mashfiq/eeg_vjepa/code/branchC_cv_results.json"
_WIN = torch.hann_window(NFFT)


def _configure(args):
    """Set the window/STFT globals from args (the 'harder horizon' knob: longer L = real dynamics)."""
    global L, NFFT, HOP, NT, NF, TC, IN_DIM, _WIN
    L, NFFT, HOP = args.L, args.nfft, args.hop
    NT = 1 + (L - NFFT) // HOP
    NF = NFFT // 2 + 1
    TC = max(1, min(NT - 1, round(NT * args.tc_frac)))
    IN_DIM = NF * F.NCH
    _WIN = torch.hann_window(NFFT)
    print(f"config: L={L}({L/F.FS:.0f}s) nfft={NFFT} hop={HOP} -> {NT} frames x {NF} bins; "
          f"ctx={TC} predict {NT-TC}", flush=True)


def spectrogram(x):
    """(B,19,L) -> (B, NT, 19*NF) log-power frames: a temporal SEQUENCE of spectra."""
    B_, C, _ = x.shape
    z = torch.stft(x.reshape(B_ * C, L), n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                   window=_WIN.to(x.device), center=False, return_complex=True)   # (B*C, NF, T)
    p = (z.real ** 2 + z.imag ** 2 + 1e-6).log()                                  # log-power
    p = p.reshape(B_, C, NF, -1).permute(0, 3, 1, 2)                              # (B, T, C, NF)
    return p.reshape(B_, p.shape[1], C * NF)                                      # (B, T, C*NF)


class DynEncoder(nn.Module):
    """Per-frame spectral embed + GRU over time -> summary of a frame span. (B,T,IN_DIM)->(B,d)."""
    def __init__(self, d=256, in_dim=None):
        super().__init__()
        in_dim = in_dim or IN_DIM          # read the (possibly reconfigured) global at build time
        self.bn = nn.BatchNorm1d(in_dim)
        self.embed = nn.Linear(in_dim, d)
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, z):                       # z: (B,T,in_dim)
        B_, T, Din = z.shape
        h = self.bn(z.reshape(B_ * T, Din)).reshape(B_, T, Din)
        h = torch.relu(self.embed(h))
        out, _ = self.gru(h)                    # (B,T,d)
        return out[:, -1]                       # last hidden = summary of the span


class LRU(nn.Module):
    """Minimal diagonal Linear Recurrent Unit (Orvieto et al. 2023) = the lazy REAL core of
    Google's RG-LRU/Griffin: stable diagonal linear recurrence h_t = λ⊙h_{t-1} + γ⊙(B x_t),
    λ∈(0,1) per-channel init NEAR 1 (=long memory, the whole point vs a GRU that forgets);
    γ=√(1−λ²) keeps state variance stable. Sequential scan (T<=79 here => no parallel scan needed).
    Upgrade path if it wins: complex-diagonal λ (magnitude+phase) to model oscillations directly."""
    def __init__(self, d):
        super().__init__()
        self.log_a = nn.Parameter(torch.linspace(2.0, 5.0, d))   # sigmoid -> λ in ~[0.88, 0.993]
        self.B = nn.Linear(d, d)
        self.C = nn.Linear(d, d)

    def forward(self, x):                        # (B,T,d)
        lam = torch.sigmoid(self.log_a)
        u = self.B(x) * torch.sqrt(1 - lam ** 2 + 1e-6)          # variance-normalized input
        h = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        outs = []
        for t in range(x.size(1)):
            h = lam * h + u[:, t]                                # long-memory linear recurrence
            outs.append(h)
        return self.C(torch.stack(outs, 1))      # (B,T,d)


class LRUEncoder(nn.Module):
    """Branch-C encoder with a long-memory LRU instead of a GRU. (B,T,in_dim)->(B,d)."""
    def __init__(self, d=256, in_dim=None):
        super().__init__()
        in_dim = in_dim or IN_DIM
        self.bn = nn.BatchNorm1d(in_dim)
        self.embed = nn.Linear(in_dim, d)
        self.lru = LRU(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, z):
        B_, T, Din = z.shape
        h = torch.relu(self.embed(self.bn(z.reshape(B_ * T, Din)).reshape(B_, T, Din)))
        y = self.norm(self.lru(h) + h)           # residual recurrent block (Griffin-style)
        return y[:, -1]                          # last state = long-memory summary of the span


def _enc(args):
    return LRUEncoder if getattr(args, "enc", "gru") == "lru" else DynEncoder


class DynJEPA(nn.Module):
    """Predict FUTURE-span embedding from PAST-span embedding. BYOL-style, EMA target."""
    def __init__(self, d=256, h=128, enc_cls=DynEncoder):
        super().__init__()
        self.enc, self.proj = enc_cls(d), nn.Linear(d, h)            # online  (past/context)
        self.enc_t, self.proj_t = enc_cls(d), nn.Linear(d, h)        # EMA target (future)
        self._sync(); [p.requires_grad_(False) for p in self._tgt()]
        self.pred = F.mlp(h)

    def _tgt(self): return list(self.enc_t.parameters()) + list(self.proj_t.parameters())
    def _src(self): return list(self.enc.parameters()) + list(self.proj.parameters())

    @torch.no_grad()
    def _sync(self):
        for s, t in zip(self._src(), self._tgt()): t.copy_(s)

    @torch.no_grad()
    def ema(self, a=0.996):
        for s, t in zip(self._src(), self._tgt()): t.mul_(a).add_(s, alpha=1 - a)

    def forward(self, z):
        past, future = z[:, :TC], z[:, TC:]
        u = self.proj(self.enc(past))
        with torch.no_grad():
            v = self.proj_t(self.enc_t(future))
        nrm = lambda a: NF_.normalize(a, dim=-1)
        return NF_.mse_loss(nrm(self.pred(u)), nrm(v))


def pretrain_C(files, args, tag=""):
    dl = DataLoader(F.Windows(files, L, args.wpe), batch_size=args.batch, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    model = DynJEPA(d=args.d, h=args.h, enc_cls=_enc(args)).to(DEV)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    print(f"{tag}Branch-C (spectral-dynamics past->future, enc={args.enc}) on {len(files)} recs: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params steps/ep={len(dl)} T={NT} ctx={TC}", flush=True)
    for ep in range(args.epochs):
        model.train(); t0, tot = time.time(), 0.0
        for step, x in enumerate(dl):
            if step % 20 == 0: F.cool_gate(verbose=False)
            x = x.to(DEV, non_blocking=True)
            loss = model(spectrogram(x))
            opt.zero_grad(); loss.backward(); opt.step(); model.ema(args.ema)
            tot += loss.item()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"{tag}  ep {ep+1}/{args.epochs} loss={tot/len(dl):.4f} {time.time()-t0:.0f}s", flush=True)
    return model.enc


def embed_dyn(enc, files):
    """Branch-C recording embedding: online encoder over the FULL spectrogram of each window, mean-pooled."""
    X, y = [], []
    for i, (p, c) in enumerate(files):
        if i % 100 == 0: F.cool_gate(verbose=False)
        sig = F.load_continuous(p); T = sig.shape[1]
        if T < L: sig = np.tile(sig, (1, L // T + 1))[:, :L]; T = L
        w = np.stack([sig[:, s:s + L] for s in range(0, T - L + 1, L)])   # (W,19,L)
        with torch.no_grad():
            e = enc(spectrogram(torch.from_numpy(w).to(DEV))).mean(0)
        X.append(e.cpu().numpy()); y.append(c)
    return np.array(X), np.array(y)


def cv(args):
    _configure(args)
    sp = STATE if not args.tag else STATE.replace(".json", f"_{args.tag}.json")
    ck = f"branchC{'_'+args.tag if args.tag else ''}_enc_cv_s0_f{{}}.pt"
    files = F.list_split("train") + F.list_split("eval")
    y = np.array([c for _, c in files])
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    print(f"band-power feats for {len(files)} recs...", flush=True)
    bp = {}
    for i, (p, _) in enumerate(files):
        if i % 300 == 0: F.cool_gate(verbose=False)
        bp[p] = B.features_for_recording(p)[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    variants = ["A", "C", "C(rand)", "A+C", "A+C+BP"]
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        if str(k) in state:
            print(f"[f{k+1}] cached -> skip", flush=True); continue
        trf = [files[i] for i in tr]; tef = [files[i] for i in te]; ytr, yte = y[tr], y[te]
        encC = pretrain_C(trf, args, tag=f"[f{k+1}] "); encC.eval()
        torch.save(encC.state_dict(), f"/home/mashfiq/eeg_vjepa/code/{ck.format(k)}")
        XC_tr, _ = embed_dyn(encC, trf); XC_te, _ = embed_dyn(encC, tef)
        rnd = _enc(args)(args.d).to(DEV).eval()                       # random-init control, SAME arch (guardrail)
        XR_tr, _ = embed_dyn(rnd, trf); XR_te, _ = embed_dyn(rnd, tef)
        encA = F.Encoder(args.d).to(DEV)
        encA.load_state_dict(torch.load(A_CKPT.format(k), map_location=DEV)); encA.eval()
        XA_tr, _ = F.embed_set(encA, trf, L); XA_te, _ = F.embed_set(encA, tef, L)
        Xbp_tr = np.array([bp[p] for p, _ in trf]); Xbp_te = np.array([bp[p] for p, _ in tef])
        feats = {
            "A": (XA_tr, XA_te), "C": (XC_tr, XC_te), "C(rand)": (XR_tr, XR_te),
            "A+C": (np.hstack([XA_tr, XC_tr]), np.hstack([XA_te, XC_te])),
            "A+C+BP": (np.hstack([XA_tr, XC_tr, Xbp_tr]), np.hstack([XA_te, XC_te, Xbp_te])),
        }
        fold = {"C_std": float(XC_te.std(0).mean())}
        line = [f"fold{k+1}"]
        for v in variants:
            a, _ = F.fit_probe(feats[v][0], ytr, feats[v][1], yte); fold[v] = float(a); line.append(f"{v}={a:.3f}")
        line.append(f"Cstd={fold['C_std']:.3f}")
        print("  ".join(line), flush=True)
        state[str(k)] = fold
        tmp = sp + ".tmp"; json.dump(state, open(tmp, "w"), indent=2); os.replace(tmp, sp)
    print("\n===== Branch-C spectral-dynamics seed-0 5-fold =====")
    for v in variants:
        a = np.array([state[str(k)][v] for k in range(5)])
        print(f"  {v:10s} {a.mean():.3f} +/- {a.std():.3f}  folds {np.round(a,3)}")
    cst = np.array([state[str(k)]["C_std"] for k in range(5)])
    print(f"  C emb-std {cst.mean():.3f} (near 0 = collapse)   refs: A=0.833  A+B+BP=0.846  band-power=0.786")


def selftest():
    torch.manual_seed(0)
    x = torch.randn(8, F.NCH, L)
    z = spectrogram(x)
    assert z.shape == (8, NT, IN_DIM), z.shape
    m = DynJEPA(d=32, h=16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    l0 = m(z).item()
    for _ in range(80):
        l = m(z); opt.zero_grad(); l.backward(); opt.step(); m.ema(0.9)
    l1 = m(z).item()
    assert l1 < l0, f"loss did not drop: {l0:.4f}->{l1:.4f}"
    # future must actually differ from past (else the task is degenerate): shift a burst in time
    xb = torch.zeros(4, F.NCH, L); xb[:, :, :400] += torch.randn(4, F.NCH, 400)  # energy only in the past
    zb = spectrogram(xb)
    assert not torch.allclose(zb[:, :TC].mean(1), zb[:, TC:].mean(1), atol=1e-2), "past/future spectra identical"
    e = m.enc(z)
    # LRU encoder: same (B,T,in_dim)->(B,d) interface, loss drops under the same DynJEPA recipe
    assert LRUEncoder(32)(z).shape == (8, 32)
    ml = DynJEPA(d=32, h=16, enc_cls=LRUEncoder)
    ol = torch.optim.AdamW(ml.parameters(), lr=1e-3); q0 = ml(z).item()
    for _ in range(80):
        l = ml(z); ol.zero_grad(); l.backward(); ol.step(); ml.ema(0.9)
    q1 = ml(z).item()
    assert q1 < q0, f"LRU loss did not drop: {q0:.4f}->{q1:.4f}"
    print(f"branchC selftest OK: spectrogram{tuple(z.shape)}; GRU-loss {l0:.4f}->{l1:.4f}; "
          f"LRU-loss {q0:.4f}->{q1:.4f}; emb-std {float(e.std(0).mean()):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--wpe", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--L", type=int, default=1000, help="window samples; 4000 = 20s = harder-horizon (real non-stationarity)")
    ap.add_argument("--nfft", type=int, default=200)
    ap.add_argument("--hop", type=int, default=50, help="STFT hop; scale with L to keep enough frames")
    ap.add_argument("--tc_frac", type=float, default=0.6, help="context fraction; predict the last (1-frac) frames")
    ap.add_argument("--tag", default="", help="suffix for state/ckpt files; keeps runs separate (no clobber)")
    ap.add_argument("--enc", choices=["gru", "lru"], default="gru",
                    help="temporal encoder: gru (default) or lru (Google-style long-memory linear recurrence)")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest: selftest()
    elif args.cv: cv(args)
    else: print("use --cv or --selftest")
