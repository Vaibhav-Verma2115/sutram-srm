#!/usr/bin/env python
"""Ablate the LR-consistency weight and report what it actually buys.

    python scripts/ablate.py --data data/interim/sen2venus_x4_loc \
        --values 0,0.5,5,10 --epochs 15 --device mps

The README claims LR-consistency is the term that stops the model inventing
radiometry. That claim was never measured: at the shipped weight of 0.5 the
term supplied 2.8% of the training loss. This runs the same recipe at several
weights and prints val metrics side by side.

DECISION METRIC IS THE VALIDATION SPLIT, NOT THE TEST TILE. The Wald test tile
(data/raw/test_ref_2p5m.tif) is reported for information only -- choosing a
hyper-parameter on it would make every later number on that tile meaningless.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def wald_on_tile(ckpt: pathlib.Path, tile: pathlib.Path, device: str) -> dict:
    """Score one checkpoint on a real Sentinel-2 tile via Wald's protocol."""
    import numpy as np  # noqa: F401

    from srm.io.raster import read_bands
    from srm.models.ours_branch import OursBranch
    from srm.validate.wald import wald_protocol

    hr, _ = read_bands(str(tile))
    branch = OursBranch(ckpt=ckpt, device=device)
    return wald_protocol(branch, hr, scale=4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/interim/sen2venus_x4_loc")
    ap.add_argument("--values", default="0,0.5,5,10",
                    help="comma list of w_consistency values to try")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--select", default="ssim")
    ap.add_argument("--arch", default="lite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="checkpoints/ablation")
    ap.add_argument("--test-tile", default="data/raw/test_ref_2p5m.tif")
    ap.add_argument("--out", default="data/outputs/ablation_consistency.json")
    args = ap.parse_args()

    values = [float(v) for v in args.values.split(",")]
    root = pathlib.Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for w in values:
        tag = f"wcons{w:g}".replace(".", "p")
        out = root / tag
        print(f"\n{'='*70}\n  w_consistency = {w}   ->  {out}\n{'='*70}")
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--data", args.data, "--out", str(out),
            "--epochs", str(args.epochs), "--device", args.device,
            "--batch-size", str(args.batch_size), "--select", args.select,
            "--arch", args.arch, "--seed", str(args.seed),
            "--w-consistency", str(w), "--tag", f"ablation {tag}",
        ]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"training failed for w={w}", file=sys.stderr)
            return r.returncode

        history = json.loads((out / "history.json").read_text())
        best_row = max(history, key=lambda h: h.get("select_score", -1e9))
        row = {
            "w_consistency": w,
            "epoch": best_row["epoch"],
            "val_psnr": best_row.get("val_psnr"),
            "val_ssim": best_row.get("val_ssim"),
            "val_sam_deg": best_row.get("val_sam_deg"),
            "val_ergas": best_row.get("val_ergas"),
            "train_consistency": best_row.get("consistency"),
        }

        tile = pathlib.Path(args.test_tile)
        if tile.exists():
            try:
                m = wald_on_tile(out / "best.pt", tile, args.device)
                row["tile_psnr"] = m["psnr"]
                row["tile_ssim"] = m["ssim"]
                row["tile_sam_deg"] = m["sam_deg"]
            except Exception as e:  # a failed side-report must not kill the sweep
                print(f"  (tile report skipped: {e})", file=sys.stderr)
        results.append(row)

    print(f"\n\n{'='*94}")
    print("LR-consistency ablation  --  decision metric is val (held-out locations)")
    print(f"{'='*94}")
    head = (f"{'w_cons':>7} {'epoch':>6} | {'val PSNR':>9} {'val SSIM':>9} {'val SAM':>8} "
            f"{'val ERGAS':>10} {'train cons':>11} | {'tile PSNR':>10} {'tile SSIM':>10}")
    print(head)
    print("-" * len(head))
    for r in results:
        print(f"{r['w_consistency']:>7g} {r['epoch']:>6} | "
              f"{r.get('val_psnr', 0):>9.3f} {r.get('val_ssim', 0):>9.4f} "
              f"{r.get('val_sam_deg', 0):>8.3f} {r.get('val_ergas', 0):>10.3f} "
              f"{r.get('train_consistency', 0):>11.6f} | "
              f"{r.get('tile_psnr', float('nan')):>10.3f} "
              f"{r.get('tile_ssim', float('nan')):>10.4f}")
    print("\n(tile columns are reported, NOT used to choose w_consistency)")

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(
        {"epochs": args.epochs, "select": args.select, "arch": args.arch,
         "data": args.data, "decision_metric": f"val_{args.select}",
         "results": results}, indent=2))
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
