#!/usr/bin/env python
"""Benchmark every SR branch with Wald's protocol and write a results table.

    python scripts/evaluate.py --input data/raw/test_scene_10m.tif

Degrades the input 4x, super-resolves it back, and scores each branch against
the original -- the only way to get true reference metrics without owning
2.5 m ground truth.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from srm.io.raster import read_bands, to_reflectance  # noqa: E402
from srm.metrics.core import evaluate as eval_metrics  # noqa: E402
from srm.trust.layer import confidence_map  # noqa: E402
from srm.validate.wald import calibration_curve, degrade  # noqa: E402

COLS = [("psnr", "PSNR", "{:.2f}"), ("ssim", "SSIM", "{:.4f}"),
        ("sam_deg", "SAM°", "{:.3f}"), ("ergas", "ERGAS", "{:.3f}"),
        ("rmse", "RMSE", "{:.5f}"), ("seconds", "sec", "{:.2f}")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--branches", default="bicubic,sen2sr")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.0, help="sensor noise std in degradation")
    ap.add_argument("--reflectance", action="store_true")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="data/outputs/benchmark.json")
    args = ap.parse_args()

    hr, _ = read_bands(args.input)
    if args.reflectance:
        hr = to_reflectance(hr)
    lr = degrade(hr, scale=args.scale, noise_std=args.noise)
    print(f"Wald protocol: {hr.shape[1]}x{hr.shape[2]} -> degraded {lr.shape[1]}x{lr.shape[2]} "
          f"-> SR back to {hr.shape[1]}x{hr.shape[2]}  (noise std={args.noise})\n")

    names = [n.strip().lower() for n in args.branches.split(",")]
    branches: dict = {}
    if "bicubic" in names:
        from srm.models.bicubic_branch import BicubicBranch
        branches["bicubic"] = BicubicBranch()
    if "sen2sr" in names:
        from srm.models.sen2sr_branch import Sen2SRBranch
        branches["SEN2SR"] = Sen2SRBranch(device=args.device)
    if "ldsr" in names:
        from srm.models.ldsr_branch import LdsrBranch
        branches["LDSR-S2"] = LdsrBranch(device=args.device, n_samples=args.n_samples,
                                         steps=args.steps)

    rows, preds = {}, {}
    for name, br in branches.items():
        p = br.predict(lr)
        sr = p.sr[:, : hr.shape[1], : hr.shape[2]]
        preds[name] = p
        m = eval_metrics(sr, hr, scale=args.scale)
        m["seconds"] = p.seconds
        rows[name] = m

    header = f"{'branch':<12}" + "".join(f"{c[1]:>9}" for c in COLS)
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        print(f"{name:<12}" + "".join(f"{c[2].format(m[c[0]]):>9}" for c in COLS))

    # Does the confidence map predict where the error actually is?
    calib = {}
    if "SEN2SR" in preds:
        fid = preds["SEN2SR"]
        other = preds.get("LDSR-S2")
        t = confidence_map(fid.sr, lr, scale=args.scale,
                           sigma=other.sigma if other else None,
                           other_branch=other.sr if other else None)
        err = np.abs(fid.sr[:, : hr.shape[1], : hr.shape[2]] - hr).mean(axis=0)
        calib = calibration_curve(t["confidence"][: hr.shape[1], : hr.shape[2]], err)
        print(f"\ntrust-layer calibration: corr(confidence, error) = "
              f"{calib['confidence_error_corr']:+.4f}  (negative = confidence predicts error)")
        for b in calib["bins"]:
            bar = "#" * max(1, int(b["mean_error"] * 4000))
            print(f"  conf {b['conf_lo']:.1f}-{b['conf_hi']:.1f}  "
                  f"n={b['n_pixels']:>7}  mean|err| {b['mean_error']:.5f} {bar}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"protocol": "wald", "scale": args.scale, "noise_std": args.noise,
         "metrics": rows, "calibration": calib}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
