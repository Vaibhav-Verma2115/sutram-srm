#!/usr/bin/env python
"""Fine-tune the SR model on packed SEN2VENuS shards.

Designed for a free Colab/Kaggle session: checkpoints every epoch so a
disconnect costs at most one epoch, and --resume picks up exactly where it left
off (model, optimiser, scaler and epoch counter).

    python scripts/train.py --data data/interim/sen2venus_x4 --epochs 30
    python scripts/train.py --data ... --resume checkpoints/last.pt

MODEL SELECTION
---------------
`--select` decides which checkpoint becomes best.pt. It defaults to SSIM, not
PSNR. Our v2 model was selected on PSNR and then benchmarked at 31.57 dB
against bicubic's 32.68 -- we were picking weights by the one metric we lose,
while the case we actually make rests on SSIM, spectral angle and downstream
task accuracy. `composite` (SSIM penalised by spectral angle) selects on both
at once.

The validation split must be built by scripts/build_dataset.py at --split-by
location; a patch-level split shares ground with training and its scores are
terrain recall, not generalisation.
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


def selection_score(vm: dict, mode: str, s2m: dict | None = None) -> float:
    """Scalar that `best.pt` maximises.

    composite keeps SSIM as the primary term and subtracts a spectral-angle
    penalty, so a checkpoint cannot buy structural score with colour drift.
    SAM is in degrees and typically ~2, so 0.01/deg makes it worth ~0.02 SSIM.

    The `s2_*` modes score against the same-sensor S2->S2 validation set, which
    is the operator we actually deploy. The SEN2VENuS split scores a
    cross-sensor task, and the two rank models differently -- in the
    w_consistency ablation the arm that won on SEN2VENuS came last on real
    Sentinel-2 -- so prefer an s2_* mode whenever --val-s2s2 is on.
    """
    src = vm
    if mode.startswith("s2_"):
        if not s2m:
            return -1e9
        src, mode = s2m, mode[3:]
    if mode == "psnr":
        return src.get("psnr", -1e9)
    if mode == "ssim":
        return src.get("ssim", -1e9)
    return src.get("ssim", -1e9) - 0.01 * src.get("sam_deg", 0.0)


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=0,
                    help="random LR crop size for augmentation (0 = use full patches)")
    ap.add_argument("--select", default="ssim",
                    choices=["psnr", "ssim", "composite",
                             "s2_psnr", "s2_ssim", "s2_composite"],
                    help="metric that decides best.pt; composite = SSIM - 0.01*SAM deg. "
                         "s2_* scores the same-sensor S2->S2 val set (needs --val-s2s2) "
                         "and is the one that matches deployment")
    ap.add_argument("--val-s2s2", action="store_true",
                    help="also validate on same-sensor S2->S2 pairs from the held-out "
                         "locations -- the deployment operator, unlike the cross-sensor "
                         "SEN2VENuS split")
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-consistency", type=float, default=0.5,
                    help="weight on the LR-consistency term. It supplies only ~3%% of the "
                         "total loss, but the ablation shows dropping it costs S2->S2 SSIM "
                         "and spectral accuracy, and that it saturates by 0.5.")
    ap.add_argument("--w-sam", type=float, default=0.1)
    ap.add_argument("--w-gradient", type=float, default=0.1)
    ap.add_argument("--mix-s2s2", type=float, default=0.0,
                    help="fraction of training samples drawn as same-sensor S2->S2 pairs "
                         "(needs a dataset built with --with-s2, and --crop). The VENuS "
                         "pairs teach S2->VENuS; deployment is S2->S2, so mixing removes "
                         "the cross-sensor transfer from what the model has to learn.")
    ap.add_argument("--tag", default="", help="free-text note stored in the checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    data = pathlib.Path(args.data)
    scale = json.loads((data / "manifest.json").read_text())["scale"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data / "manifest.json").read_text())
    split_by = manifest.get("split_by")
    if split_by != "location":
        print(f"WARNING: dataset was split by {split_by!r}, not 'location'. Validation "
              f"scores will be inflated by shared ground between train and val. "
              f"Rebuild with: scripts/build_dataset.py --split-by location",
              file=sys.stderr)
    else:
        g = manifest.get("groups", {})
        print(f"split by location: {g.get('train','?')} train / {g.get('val','?')} val "
              f"locations, 0 shared")

    train_ds = Sen2VenusShards(data, "train", augment=True,
                               crop=args.crop or None,
                               mix_s2s2=args.mix_s2s2, seed=args.seed)
    val_ds = Sen2VenusShards(data, "val", augment=False)
    val_s2_ds = None
    if args.val_s2s2 or args.select.startswith("s2_"):
        val_s2_ds = Sen2VenusShards(data, "val", augment=False, mix_s2s2=1.0, seed=args.seed)
    # num_workers must be 0 for the shard cache to be effective (each worker
    # would otherwise hold its own copy and thrash).
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    val_s2_dl = (DataLoader(val_s2_ds, batch_size=args.batch_size, num_workers=0)
                 if val_s2_ds is not None else None)

    overrides = {"feature_channels": 48, "num_blocks": 10} if args.arch == "wide" else {}
    model = build_model(scale=scale, train_mode=True, **overrides).to(device)
    if not args.scratch:
        try:
            report = load_pretrained_weights(model)
            print(f"warm start: {report['loaded']}/{report['total']} tensors loaded")
        except FileNotFoundError:
            print("no pretrained weights found; training from scratch")

    crit = SRLoss(scale=scale, w_l1=args.w_l1, w_consistency=args.w_consistency,
                  w_sam=args.w_sam, w_gradient=args.w_gradient)
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
          f"params={sum(p.numel() for p in model.parameters())/1e3:.0f}k "
          f"select={args.select} crop={args.crop or 'full'}")
    print(f"loss weights: l1={args.w_l1} consistency={args.w_consistency} "
          f"sam={args.w_sam} gradient={args.w_gradient}")
    if args.mix_s2s2:
        print(f"mixing {args.mix_s2s2:.0%} same-sensor S2->S2 pairs into training")
    print()

    config = {"select": args.select, "crop": args.crop, "seed": args.seed,
              "mix_s2s2": args.mix_s2s2,
              "arch": args.arch, "epochs": args.epochs, "batch_size": args.batch_size,
              "lr": args.lr, "scratch": args.scratch, "tag": args.tag,
              "weights": {"l1": args.w_l1, "consistency": args.w_consistency,
                          "sam": args.w_sam, "gradient": args.w_gradient},
              "data": str(data), "split_by": split_by}

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        # New order every epoch, without breaking the shard cache.
        train_ds.reshuffle(args.seed * 100003 + epoch)
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
        s2m = (validate(model, val_s2_dl, device, scale, limit=args.val_batches)
               if val_s2_dl is not None else {})
        row = {"epoch": epoch, "seconds": round(time.time() - t0, 1),
               "lr": sched.get_last_lr()[0], **agg,
               **{f"val_{k}": v for k, v in vm.items()},
               **{f"s2_{k}": v for k, v in s2m.items()}}
        history.append(row)
        line = (f"epoch {epoch:>3}  loss {agg['loss']:.5f}  l1 {agg['l1']:.5f}  "
                f"cons {agg['consistency']:.5f}  | val PSNR {vm.get('psnr', 0):.2f}  "
                f"SSIM {vm.get('ssim', 0):.4f}  SAM {vm.get('sam_deg', 0):.3f}")
        if s2m:
            line += (f"  | S2 PSNR {s2m.get('psnr', 0):.2f}  "
                     f"SSIM {s2m.get('ssim', 0):.4f}")
        print(line + f"  [{row['seconds']:.0f}s]")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "scale": scale, "best": best, "history": history,
              "config": config, "select": args.select}
        torch.save(ck, out / "last.pt")
        score = selection_score(vm, args.select, s2m)
        row["select_score"] = score
        if score > best:
            best = score
            ck["best"] = best
            torch.save(ck, out / "best.pt")
            print(f"           new best {args.select} {best:.4f} -> {out/'best.pt'}")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\ndone. best val {args.select} {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
