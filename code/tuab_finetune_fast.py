"""Fast fine-tune / supervised-from-scratch ViT-M on TUAB: a port of evals/eeg_classification_frozen/eval.py with
data.supervised = true (the authors' fine-tune mode). Kept from upstream: init_model / AttentionClassifier /
init_opt(supervised=True) / WarmupCosine + CosineWD schedulers stepped per iteration / unweighted CE /
clip_grad_norm_ on the CLASSIFIER only / fp32 / batch 2 / lr 1e-3 / wd 1e-3 / 500 epochs / DistributedSampler(shuffle,
never set_epoch) batch order / EEGTransformEval (= identity, verified in tuab_cached_eval.py --verify).
Changed (numerically equivalent): recordings live on the GPU and clips are gathered there (same np.random.randint
end draw, different random stream); eval forward in batches of 32 (no batch-norm -> same outputs up to fp order);
no single-rank DDP wrapper. Speed options (each measured by --bench): --tf32, --compile, --bf16 (a real deviation).

  python tuab_finetune_fast.py --gate            # 50 steps: upstream run_one_epoch vs our step, same init + clips
  python tuab_finetune_fast.py --bench           # s/epoch: upstream loop vs each speed variant
  python tuab_finetune_fast.py --tag ft_vitm_s0 [--subset 0 --seed 0 --ckpt released|none ...]   # resumable
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_guard import cap_threads, cool_gate
cap_threads(4)
import numpy as np
import torch
import torch.nn.functional as F

import evals.eeg_classification_frozen.eval as E
from tuab_cached_eval import MODELS, HDR, dataset, clip, encoder

ROOT = "/home/mashfiq/eeg_vjepa"
OUT = f"{ROOT}/code/runs/tuab_finetune"
DEV = "cuda:0"
GATE = lambda *a: cool_gate(pause=88.0, resume=78.0, abort=92.0, verbose=False)


def load_split(subset, split, M):
    ds = dataset(subset, split, M)
    R = torch.from_numpy(np.stack([np.load(f).astype("float32") for f in ds.samples])).to(DEV)   # (N, T, 19, 500)
    return ds, R, np.array(ds.labels)


def clip_table(M, T):   # IDX[end - clip_len] = frame indices of the clip ending at `end` (= clip())
    L = M["frames_per_clip"] * M["frame_step"]
    return torch.from_numpy(np.stack([clip(np.arange(T), e, M) for e in range(L, T)])).to(DEV), L


def gather(R, IDX, L, idx, ends):
    return R[torch.as_tensor(idx, device=DEV)[:, None], IDX[torch.as_tensor(ends - L, device=DEV)]][:, None]


def build(M, ckpt, seed, epochs, ipe, a):
    enc = encoder(M, M["ckpt"] if ckpt == "released" else "none", seed)
    torch.manual_seed(seed)
    head = E.AttentionClassifier(embed_dim=enc.embed_dim, num_classes=2).to(DEV)
    adamw = torch.optim.AdamW   # fused AdamW = the same update in one kernel (~3 ms/step saved; rounding-level diff)
    torch.optim.AdamW = lambda groups: adamw(groups, fused=getattr(a, "fused", True))
    try:
        opt, _, sched, wd_sched = E.init_opt(encoder=enc, classifier=head, wd=1e-3, start_lr=a.lr, ref_lr=a.lr,
                                             final_lr=0.0, iterations_per_epoch=ipe, warmup=0.0, num_epochs=epochs,
                                             use_bfloat16=False, supervised=True)
    finally:
        torch.optim.AdamW = adamw
    return enc, head, opt, sched, wd_sched


class Net(torch.nn.Module):
    def __init__(self, enc, head):
        super().__init__(); self.enc, self.head = enc, head

    def forward(self, x):
        return self.head(self.enc(x))


def make_fwd(enc, head, a):
    net = Net(enc, head)
    f = torch.compile(net, mode=a.compile) if a.compile else net
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16)

    def fwd(x):
        with ac():
            return f(x).float()
    return net, fwd


def train_step(fwd, head, x, lab, opt, sched, wd_sched):
    sched.step(); wd_sched.step()
    out = fwd(x)
    loss = F.cross_entropy(out, lab)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)   # upstream clips the classifier only
    opt.step(); opt.zero_grad()
    return F.softmax(out.detach(), 1), float(loss)


def epoch(net, fwd, head, R, IDX, L, y, order, train, bs, opt=None, sched=None, wd_sched=None):
    net.train(mode=train)
    P, accs, T = [], [], R.shape[1]
    step = bs if train else 32
    for i in range(0, len(order), step):
        if i % (100 * step) == 0:
            GATE()
        idx = order[i:i + step]
        ends = np.random.randint(L, T, size=len(idx))
        x, lab = gather(R, IDX, L, idx, ends), torch.as_tensor(y[idx], device=DEV)
        if train:
            p, _ = train_step(fwd, head, x, lab, opt, sched, wd_sched)
        else:
            with torch.no_grad():
                p = F.softmax(fwd(x), 1)
        P.append(p)
        if train:   # upstream: mean of per-batch accuracy
            accs.append(float(100. * p.argmax(1).eq(lab).sum() / len(idx)))
    P = torch.cat(P); lab = torch.as_tensor(y[order], device=DEV)
    acc = float(np.mean(accs)) if train else float(100. * P.argmax(1).eq(lab).float().mean())
    return acc, P, lab


def sampler_order(n):
    return np.array(list(torch.utils.data.distributed.DistributedSampler(range(n), num_replicas=1, rank=0, shuffle=True)))


def ckpt_save(path, **kw):
    torch.save(kw, path + ".tmp"); os.replace(path + ".tmp", path)


def run(a):
    M = MODELS["vitm"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = a.tf32
    _, Rtr, ytr = load_split(a.subset, "train", M)
    ev, Rev, yev = load_split(a.subset, "eval", M)
    IDX, L = clip_table(M, Rtr.shape[1])
    ipe = math.ceil(len(ytr) / a.bs)
    np.random.seed(a.seed)
    enc, head, opt, sched, wd_sched = build(M, a.ckpt, a.seed, a.epochs, ipe, a)
    net, fwd = make_fwd(enc, head, a)
    otr, oev = sampler_order(len(ytr)), sampler_order(len(yev))
    d = f"{OUT}/{a.tag}"; os.makedirs(d, exist_ok=True)
    json.dump(vars(a), open(f"{d}/args.json", "w"), indent=1)
    ck, csvp, start, hist = f"{d}/state.pt", f"{d}/probe_r0.csv", 0, []
    if os.path.exists(ck):   # resume: model + optimizer + RNG + per-epoch preds; schedulers re-stepped as upstream
        s = torch.load(ck, map_location=DEV, weights_only=False)
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); start, hist = s["epoch"], s["hist"]
        np.random.set_state(s["np_rng"]); torch.set_rng_state(s["torch_rng"].cpu())
        for _ in range(start * ipe):
            sched.step(); wd_sched.step()
        rows = list(csv.reader(open(csvp)))[:start + 1]
        csv.writer(open(csvp, "w")).writerows(rows)
        print(f"[{a.tag}] resumed at epoch {start}", flush=True)
    else:
        csv.writer(open(csvp, "w")).writerow(HDR)
    w = csv.writer(open(csvp, "a", buffering=1))
    t0 = time.time()
    for ep in range(start, a.epochs):
        tr_acc, tp, tl = epoch(net, fwd, head, Rtr, IDX, L, ytr, otr, True, a.bs, opt, sched, wd_sched)
        ev_acc, vp, vl = epoch(net, fwd, head, Rev, IDX, L, yev, oev, False, a.bs)
        ta, _, tf, tb = E.calculate_metrics(tp, tl)
        va, _, vf, vb = E.calculate_metrics(vp, vl)
        hist.append(vp[:, 0].float().cpu().numpy())
        w.writerow([ep + 1, f"{tr_acc:.5f}", f"{ev_acc:.5f}"] + [f"{float(v):.5f}" for v in (ta, va, tb, vb, tf, vf)])
        spe = (time.time() - t0) / (ep + 1 - start)
        print(f"[{a.tag}] ep {ep+1}/{a.epochs} train_acc {tr_acc:.1f} val_acc {ev_acc:.1f} val_auroc {float(va):.3f} "
              f"({spe:.1f} s/ep, ETA {(a.epochs-ep-1)*spe/3600:.1f} h)", flush=True)
        if (ep + 1) % a.save_every == 0 or ep + 1 == a.epochs:
            ckpt_save(ck, net=net.state_dict(), opt=opt.state_dict(), epoch=ep + 1, hist=hist,
                      np_rng=np.random.get_state(), torch_rng=torch.get_rng_state())
    files = np.array([str(f) for f in ev.samples])
    np.savez(f"{d}/final_preds.npz", prob=vp.cpu().numpy(), label=vl.cpu().numpy(), rec=files[oev],
             prob_abnormal_by_epoch=np.array(hist, dtype=np.float32))   # prob[:,0] = P(abnormal); abnormal=0
    print(f"[{a.tag}] DONE", flush=True)


def gate(a, steps=50):
    """Upstream run_one_epoch (training, supervised) vs our train_step, same init + same clips + same order."""
    M = MODELS["vitm"]
    _, R, y = load_split(0, "train", M)
    IDX, L = clip_table(M, R.shape[1])
    rng = np.random.RandomState(0)
    batches = [(rng.choice(len(y), 2, replace=False), rng.randint(L, R.shape[1], size=2)) for _ in range(steps)]
    X = [(gather(R, IDX, L, i, e), torch.as_tensor(y[i], device=DEV)) for i, e in batches]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False   # reference = upstream fp32
    ref = build(M, a.ckpt, 0, 500, 273, a)
    enc2, head2 = copy.deepcopy(ref[0]), copy.deepcopy(ref[1])
    opt2, _, s2, w2 = E.init_opt(encoder=enc2, classifier=head2, wd=1e-3, start_lr=a.lr, ref_lr=a.lr, final_lr=0.0,
                                 iterations_per_epoch=273, warmup=0.0, num_epochs=500, use_bfloat16=False, supervised=True)
    loader = [([x.cpu()], [l.cpu()]) for x, l in X]
    _, Pref, _ = E.run_one_epoch(device=DEV, training=True, encoder=ref[0], classifier=ref[1], scaler=None,
                                 optimizer=ref[2], scheduler=ref[3], wd_scheduler=ref[4], data_loader=loader,
                                 use_bfloat16=False, num_spatial_views=1, num_temporal_views=1,
                                 attend_across_segments=False, supervised=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = a.tf32
    net, fwd = make_fwd(enc2, head2, a)
    net.train()
    Pour = torch.cat([train_step(fwd, head2, x, l, opt2, s2, w2)[0] for x, l in X])
    dp = (Pref - Pour).abs().max().item()
    dw = max((p - q).abs().max().item() for p, q in zip(ref[0].parameters(), enc2.parameters()))
    print(f"GATE {a.label}: max|softmax diff| over {steps} steps = {dp:.2e}, max|encoder weight diff| = {dw:.2e}", flush=True)
    return dp, dw


def bench(a):
    """s/epoch (train 273 steps + eval 276 recs) for the upstream loop and each speed variant, s0 released ckpt."""
    res = {}
    M = MODELS["vitm"]
    GATE()
    if a.variants:   # re-bench named variants only; upstream time comes from the earlier full bench
        res = json.load(open(f"{OUT}/bench.json"))
    else:
        # upstream: real DataLoader (4 workers) + run_one_epoch, batch 2 train + batch 2 eval
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        enc, head, opt, sched, wd = build(M, "released", 0, 500, 273, a)
        mk = lambda tr: E.make_dataloader(root_path=[f"{ROOT}/data/TUAB_subsets/s0"], batch_size=2, world_size=1, rank=0,
                                          frames_per_clip=32, frame_step=3, num_segments=1, training=tr, num_workers=4)
        kw = dict(device=DEV, encoder=enc, classifier=head, scaler=None, optimizer=opt, scheduler=sched,
                  wd_scheduler=wd, use_bfloat16=False, num_spatial_views=1, num_temporal_views=1,
                  attend_across_segments=False, supervised=True)
        tl, vl = mk(True), mk(False)
        E.logger.disabled = True
        torch.cuda.synchronize(); t = time.time()
        E.run_one_epoch(training=True, data_loader=tl, **kw); E.run_one_epoch(training=False, data_loader=vl, **kw)
        torch.cuda.synchronize(); res["upstream"] = time.time() - t
        print(f"BENCH upstream  {res['upstream']:.1f} s/epoch", flush=True)
        del enc, head, opt, tl, vl; torch.cuda.empty_cache()
    _, Rtr, ytr = load_split(0, "train", M); _, Rev, yev = load_split(0, "eval", M)
    IDX, L = clip_table(M, Rtr.shape[1])
    variants = [("fp32", {}), ("tf32", dict(tf32=True)), ("fp32+compile", dict(compile="default")),
                ("fp32+cudagraphs", dict(compile="reduce-overhead")), ("tf32+cudagraphs", dict(tf32=True, compile="reduce-overhead")),
                ("bf16+cudagraphs", dict(bf16=True, compile="reduce-overhead")), ("bf16", dict(bf16=True)),
                ("bf16+compile", dict(bf16=True, compile="default")),
                ("bf16+autotune", dict(bf16=True, compile="max-autotune-no-cudagraphs"))]
    variants = [x for x in variants if x[0] in a.variants] if a.variants else variants[:6]
    for name, v in variants:
        b = argparse.Namespace(**{**vars(a), "tf32": False, "compile": None, "bf16": False, **v})
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = b.tf32
        enc = head = opt = None
        try:
            enc, head, opt, sched, wd = build(M, "released", 0, 500, 273, b)
            net, fwd = make_fwd(enc, head, b)
            otr, oev = sampler_order(len(ytr)), sampler_order(len(yev))
            args = lambda tr: (net, fwd, head, Rtr if tr else Rev, IDX, L, ytr if tr else yev, otr if tr else oev, tr, 2)
            epoch(*args(True), opt, sched, wd); epoch(*args(False))          # warm-up epoch (compile/graph capture)
            torch.cuda.synchronize(); t = time.time()
            epoch(*args(True), opt, sched, wd); epoch(*args(False))
            torch.cuda.synchronize(); res[name] = time.time() - t
            print(f"BENCH {name:16s} {res[name]:.1f} s/epoch  ({res['upstream']/res[name]:.2f}x, "
                  f"500 ep = {res[name]*500/3600:.2f} h)", flush=True)
        except Exception as e:   # a failing variant is a result, not a crash
            res[name] = f"FAILED: {type(e).__name__}: {e}"[:300]
            print(f"BENCH {name:16s} {res[name]}", flush=True)
        del enc, head, opt; torch._dynamo.reset(); torch.cuda.empty_cache()
        json.dump(res, open(f"{OUT}/bench.json", "w"), indent=1)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true"); ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tag"); ap.add_argument("--subset", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="released", choices=["released", "none"])
    ap.add_argument("--bs", type=int, default=2); ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--tf32", action="store_true"); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--compile", choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"])
    ap.add_argument("--variants", nargs="*")   # --bench: only these variants (names as in bench())
    ap.add_argument("--no_fused", dest="fused", action="store_false")   # fused AdamW is the default from scratch_vitm_s0 on
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.gate:
        res = {}
        for label, v in [("fp32 eager", {}), ("fp32 cudagraphs", dict(compile="reduce-overhead")),
                         ("tf32 cudagraphs", dict(tf32=True, compile="reduce-overhead"))]:
            b = argparse.Namespace(**{**vars(a), **v, "label": label})
            res[label] = gate(b); torch._dynamo.reset()
        json.dump(res, open(f"{OUT}/gate.json", "w"), indent=1)
    elif a.bench:
        bench(a)
    else:
        run(a)
