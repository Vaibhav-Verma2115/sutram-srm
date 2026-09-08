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
    ap.add_argument("--ours-ckpt", default="checkpoints/best.pt",
                    help="checkpoint for the 'ours' branch")
    ap.add_argument("--calibrate", default="",
                    help="comma list of branches to build a trust-layer calibration curve "
                         "for; default = every branch that ran")
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
    if "ours" in names:
        from srm.models.ours_branch import OursBranch
        branches["Ours"] = OursBranch(ckpt=args.ours_ckpt, device=args.device)
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
    #
    # This used to be hardcoded to SEN2SR, so a run that benchmarked "Ours"
    # still wrote out SEN2SR's calibration curve under the Ours filename -- the
    # branch we actually promote was never calibrated. Every branch gets its own
    # curve now, and `calibration` keys them by branch name.
    which = [n.strip() for n in args.calibrate.split(",")] if args.calibrate else list(preds)
    other = preds.get("LDSR-S2")
    calib: dict = {}
    for name in which:
        if name not in preds:
            continue
        p_ = preds[name]
        # Don't hand a branch its own output as the disagreement reference.
        ref = other if (other is not None and other.name != p_.name) else None
        t = confidence_map(p_.sr, lr, scale=args.scale,
                           sigma=ref.sigma if ref else None,
                           other_branch=ref.sr if ref else None)
        err = np.abs(p_.sr[:, : hr.shape[1], : hr.shape[2]] - hr).mean(axis=0)
        c = calibration_curve(t["confidence"][: hr.shape[1], : hr.shape[2]], err)

        # Monotonicity is the actual claim, so measure it instead of eyeballing
        # a few bins: Spearman rank correlation between bin midpoint and bin
        # mean error, weighted by nothing -- plus how many adjacent bin pairs
        # actually descend.
        bins = c["bins"]
        mids = [(b["conf_lo"] + b["conf_hi"]) / 2 for b in bins]
        errs = [b["mean_error"] for b in bins]
        drops = sum(1 for a, b in zip(errs, errs[1:]) if b <= a)
        c["monotonic_bin_pairs"] = f"{drops}/{max(len(errs) - 1, 1)}"
        c["strictly_monotonic"] = drops == len(errs) - 1
        if len(mids) > 2:
            r = np.corrcoef(np.argsort(np.argsort(mids)),
                            np.argsort(np.argsort(errs)))[0, 1]
            c["bin_spearman"] = float(r)
        calib[name] = c

        print(f"\ntrust-layer calibration [{name}]: corr(confidence, error) = "
              f"{c['confidence_error_corr']:+.4f}  (negative = confidence predicts error)")
        print(f"  bin-mean monotonicity: {c['monotonic_bin_pairs']} adjacent pairs descend"
              f"{'' if c['strictly_monotonic'] else '  <- NOT strictly monotonic'}")
        for b in bins:
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
