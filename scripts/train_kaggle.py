#!/usr/bin/env python
"""Kaggle GPU training driver: two experiments, best model wins.

Runs on a Kaggle P100/T4 against the sutram-sen2venus dataset. Trains two
candidates sequentially and reports both, so one kernel session answers the
capacity question empirically instead of us guessing:

  A. "lite-long"  -- released SEN2SRLite architecture (24ch/6blk, full warm
                     start), long schedule. Bet: v1 plateaued on schedule/data,
                     not capacity.
  B. "wide"       -- 48ch/10blk (~1.9M params, partial warm start). Bet: v1
                     plateaued on capacity.

Outputs land in /kaggle/working: ckpt_lite/ and ckpt_wide/ with best.pt each.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

DATA = pathlib.Path("/kaggle/input")
WORK = pathlib.Path("/kaggle/working")


def find_shards() -> pathlib.Path:
    hits = list(DATA.glob("*/sen2venus_x4/manifest.json")) + list(DATA.glob("*/manifest.json"))
    if not hits:
        sys.exit(f"no manifest.json under {DATA}")
    return hits[0].parent


def run(args: list[str]) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.call(args)


def main() -> None:
    shards = find_shards()
    print("shards:", shards, flush=True)

    code = next(DATA.glob("*/code"), None)
    if code is None:
        sys.exit("code/ directory not found in the dataset")
    sys.path.insert(0, str(code / "src"))

    # train.py warm-starts from a path relative to CWD; stage the weights there.
    weights = next(DATA.glob("*/weights/model.safetensor"), None)
    if weights:
        dst = WORK / "models" / "SEN2SRLite_NonReference_RGBN_x4"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "model.safetensor").write_bytes(weights.read_bytes())
        import os
        os.chdir(WORK)

    base = [sys.executable, str(code / "scripts" / "train.py"),
            "--data", str(shards), "--device", "cuda", "--amp",
            "--batch-size", "16", "--lr", "2e-4"]

    # Experiment A: same arch as v1, longer schedule.
    run(base + ["--out", str(WORK / "ckpt_lite"), "--epochs", "120",
                "--resume", str(WORK / "ckpt_lite" / "last.pt")])

    # Experiment B: wider+deeper, partial warm start.
    run(base + ["--out", str(WORK / "ckpt_wide"), "--epochs", "120",
                "--arch", "wide",
                "--resume", str(WORK / "ckpt_wide" / "last.pt")])

    import json
    for name in ["ckpt_lite", "ckpt_wide"]:
        h = WORK / name / "history.json"
        if h.exists():
            hist = json.loads(h.read_text())
            best = max(hist, key=lambda r: r.get("val_psnr", -1e9))
            print(f"{name}: best epoch {best['epoch']} "
                  f"PSNR {best['val_psnr']:.2f} SSIM {best['val_ssim']:.4f} "
                  f"SAM {best['val_sam_deg']:.3f}", flush=True)


if __name__ == "__main__":
    main()
