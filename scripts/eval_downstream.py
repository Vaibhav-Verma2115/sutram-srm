#!/usr/bin/env python
"""Downstream-task benchmark: is analysis on SR output closer to analysis on
real HR than analysis on plain upsampling is?

    python scripts/eval_downstream.py --data data/interim/sen2venus_x4 \
        --branches bicubic,sen2sr,ours --n 100 --device mps

Uses the SEN2VENuS validation split, where the HR side is *real* VENuS 5 m
imagery -- so "truth analysis" means edges of a real high-resolution scene,
not of a synthetic degradation round trip.
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
from tqdm import tqdm  # noqa: E402

from srm.train.dataset import Sen2VenusShards  # noqa: E402
from srm.validate.downstream import TASKS, evaluate_tasks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/interim/sen2venus_x4")
    ap.add_argument("--branches", default="bicubic,sen2sr,ours")
    ap.add_argument("--n", type=int, default=100, help="validation patches to use")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=int, default=2, help="edge match tolerance (px)")
    ap.add_argument("--out", default="data/outputs/downstream.json")
    args = ap.parse_args()

    names = [x.strip().lower() for x in args.branches.split(",")]
    branches = {}
    if "bicubic" in names:
        from srm.models.bicubic_branch import BicubicBranch
        branches["bicubic"] = BicubicBranch()
    if "sen2sr" in names:
        from srm.models.sen2sr_branch import Sen2SRBranch
        branches["SEN2SR"] = Sen2SRBranch(device=args.device)
    if "ours" in names:
        from srm.models.ours_branch import OursBranch
        branches["Ours"] = OursBranch(device=args.device)

    ds = Sen2VenusShards(args.data, "val", augment=False)
    n = min(args.n, len(ds))
    print(f"{n} validation patches, branches: {list(branches)}\n")

    # Aggregate F1 across patches per task per branch.
    agg: dict = {t: {b: [] for b in branches} for t in TASKS}
    for i in tqdm(range(n), unit="patch"):
        lr, hr = ds[i]
        lr, hr = lr.numpy(), hr.numpy()
        candidates = {name: br.predict(lr).sr for name, br in branches.items()}
        res = evaluate_tasks(hr, candidates, tol=args.tol)
        for task, v in res.items():
            for bname, s in v["scores"].items():
                agg[task][bname].append(s["f1"])

    summary = {}
    for task, per_branch in agg.items():
        summary[task] = {b: {"mean_f1": float(np.mean(v)), "std": float(np.std(v))}
                         for b, v in per_branch.items()}

    print()
    for task, rows in summary.items():
        print(f"== {task} (boundary F1 vs real VENuS 5 m analysis, tol={args.tol}px)")
        base = rows.get("bicubic", {}).get("mean_f1")
        for b, v in sorted(rows.items(), key=lambda kv: -kv[1]["mean_f1"]):
            delta = f"  ({v['mean_f1']-base:+.3f} vs bicubic)" if base and b != "bicubic" else ""
            print(f"   {b:<10} F1 {v['mean_f1']:.3f} ± {v['std']:.3f}{delta}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_patches": n, "tol_px": args.tol, "tasks": summary}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
