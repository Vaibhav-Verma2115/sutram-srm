#!/usr/bin/env python
"""Fine-tune the SR model on packed SEN2VENuS shards.

Designed for a free Colab/Kaggle session: checkpoints every epoch so a
disconnect costs at most one epoch, and --resume picks up exactly where it left
off (model, optimiser, scaler and epoch counter).

    python scripts/train.py --data data/interim/sen2venus_x4 --epochs 30
    python scripts/train.py --data ... --resume checkpoints/last.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from srm.metrics.core import evaluate  # noqa: E402
from srm.train.dataset import Sen2VenusShards  # noqa: E402
from srm.train.losses import SRLoss  # noqa: E402
from srm.train.model import build_model, load_pretrained_weights  # noqa: E402


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def validate(model, loader, device, scale, limit=None):
    model.eval()
    import numpy as np
    acc = []
    for i, (lr, hr) in enumerate(loader):
        if limit and i >= limit:
            break
        sr = model(lr.to(device)).clamp(0, 1).cpu().numpy()
        hr = hr.numpy()
        for b in range(sr.shape[0]):
            acc.append(evaluate(sr[b], hr[b], scale=scale))
    model.train()
    if not acc:
        return {}
    return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scratch", action="store_true", help="skip pretrained warm start")
    ap.add_argument("--arch", default="lite", choices=["lite", "wide"],
                    help="lite = released SEN2SRLite shape (full warm start); "
                         "wide = 48ch/10blk (partial warm start)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    args = ap.parse_args()

    device = pick_device(args.device)
    data = pathlib.Path(args.data)
    scale = json.loads((data / "manifest.json").read_text())["scale"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = Sen2VenusShards(data, "train", augment=True)
    val_ds = Sen2VenusShards(data, "val", augment=False)
    # num_workers must be 0 for the shard cache to be effective (each worker
    # would otherwise hold its own copy and thrash).
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    overrides = {"feature_channels": 48, "num_blocks": 10} if args.arch == "wide" else {}
    model = build_model(scale=scale, train_mode=True, **overrides).to(device)
    if not args.scratch:
        try:
            report = load_pretrained_weights(model)
            print(f"warm start: {report['loaded']}/{report['total']} tensors loaded")
        except FileNotFoundError:
            print("no pretrained weights found; training from scratch")

    crit = SRLoss(scale=scale)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, -1e9
    if args.resume and pathlib.Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", best)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(f"device={device} scale=x{scale} train={len(train_ds)} val={len(val_ds)} "
          f"params={sum(p.numel() for p in model.parameters())/1e3:.0f}k\n")

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        agg, n = {}, 0
        for lr, hr in train_dl:
            lr, hr = lr.to(device), hr.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                sr = model(lr)
                loss, parts = crit(sr, hr, lr)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        sched.step()
        agg = {k: v / max(n, 1) for k, v in agg.items()}

        vm = validate(model, val_dl, device, scale, limit=args.val_batches)
        row = {"epoch": epoch, "seconds": round(time.time() - t0, 1),
               "lr": sched.get_last_lr()[0], **agg, **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)
        print(f"epoch {epoch:>3}  loss {agg['loss']:.5f}  l1 {agg['l1']:.5f}  "
              f"cons {agg['consistency']:.5f}  | val PSNR {vm.get('psnr', 0):.2f}  "
              f"SSIM {vm.get('ssim', 0):.4f}  SAM {vm.get('sam_deg', 0):.3f}  "
              f"[{row['seconds']:.0f}s]")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "scale": scale, "best": best, "history": history}
        torch.save(ck, out / "last.pt")
        if vm.get("psnr", -1e9) > best:
            best = vm["psnr"]
            ck["best"] = best
            torch.save(ck, out / "best.pt")
            print(f"           new best PSNR {best:.2f} -> {out/'best.pt'}")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\ndone. best val PSNR {best:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
