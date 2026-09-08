#!/usr/bin/env python
"""Score checkpoints on same-sensor S2->S2 pairs from held-out locations.

    python scripts/eval_s2s2.py --ckpts checkpoints/ablation/*/best.pt --device mps

Why this exists. We have two validation instruments and they disagree:

  * the SEN2VENuS val split scores S2 -> VENuS, a *cross-sensor* task, and
  * the Wald tile scores S2 -> S2 on real Sentinel-2, which is deployment,

and in the consistency ablation the arm that won on the first came last on the
second. Selecting on the SEN2VENuS split therefore optimises the wrong operator,
but selecting on the Wald tile would tune a hyper-parameter on the one tile
every headline number is later measured against.

This is the third instrument that resolves it: real Sentinel-2 on both sides
(the native 10 m patch degraded to 40 m and reconstructed back), drawn only from
locations held out of training, and 1428 patches rather than one tile. Same
operator as deployment, no leak, and not the test tile.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from srm.metrics.core import evaluate as eval_metrics  # noqa: E402
from srm.train.dataset import Sen2VenusShards  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/interim/sen2venus_x4_s2")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=400, help="val patches to score")
    ap.add_argument("--baselines", action="store_true",
                    help="also score bicubic and SEN2SR for context")
    ap.add_argument("--out", default="data/outputs/s2s2_val.json")
    args = ap.parse_args()

    ds = Sen2VenusShards(args.data, "val", augment=False, mix_s2s2=1.0, seed=0)
    n = min(args.n, len(ds))
    pairs = [ds[i] for i in range(n)]
    print(f"{n} same-sensor S2->S2 pairs from held-out locations "
          f"({pairs[0][0].shape[-1]} -> {pairs[0][1].shape[-1]} px)\n")

    def score(predict) -> dict:
        acc = []
        for lr, hr in tqdm(pairs, unit="patch", leave=False):
            lr, hr = lr.numpy(), hr.numpy()
            sr = predict(lr)[:, : hr.shape[1], : hr.shape[2]]
            acc.append(eval_metrics(sr, hr, scale=4))
        return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}

    rows: dict[str, dict] = {}

    if args.baselines:
        from srm.models.bicubic_branch import BicubicBranch
        from srm.models.sen2sr_branch import Sen2SRBranch
        for name, br in [("bicubic", BicubicBranch()),
                         ("SEN2SR", Sen2SRBranch(device=args.device))]:
            print(f"scoring {name} ...")
            rows[name] = score(lambda x, b=br: b.predict(x).sr)

    from srm.models.ours_branch import OursBranch
    for c in args.ckpts:
        path = pathlib.Path(c)
        if not path.exists():
            print(f"  skip missing {path}", file=sys.stderr)
            continue
        name = path.parent.name
        print(f"scoring {name} ...")
        br = OursBranch(ckpt=path, device=args.device)
        rows[name] = score(lambda x, b=br: b.predict(x).sr)

    head = (f"{'model':<16}{'PSNR':>9}{'SSIM':>9}{'SAM deg':>9}{'ERGAS':>9}{'RMSE':>10}")
    print(f"\n{head}\n{'-'*len(head)}")
    for name, m in sorted(rows.items(), key=lambda kv: -kv[1]["ssim"]):
        print(f"{name:<16}{m['psnr']:>9.3f}{m['ssim']:>9.4f}{m['sam_deg']:>9.3f}"
              f"{m['ergas']:>9.3f}{m['rmse']:>10.5f}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"protocol": "s2s2_wald_heldout_locations", "n_patches": n,
         "data": args.data, "metrics": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
