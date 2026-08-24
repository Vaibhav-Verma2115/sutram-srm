#!/usr/bin/env python
"""Run the SRM pipeline on a Sentinel-2 GeoTIFF and write a 2.5 m COG product.

    python scripts/run_inference.py --input data/raw/scene.tif --out data/outputs/scene_sr

Writes <out>.tif (4 SR bands + sigma + confidence) and <out>_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from srm.io.raster import check_footprint, read_bands, to_reflectance  # noqa: E402
from srm.models.bicubic_branch import BicubicBranch  # noqa: E402
from srm.models.sen2sr_branch import Sen2SRBranch  # noqa: E402
from srm.pipeline import run, write_metrics, write_product  # noqa: E402


def build_branches(names: list[str], device: str, n_samples: int, steps: int) -> dict:
    branches: dict = {}
    if "bicubic" in names:
        branches["bicubic"] = BicubicBranch()
    if "sen2sr" in names:
        branches["SEN2SR"] = Sen2SRBranch(device=device)
    if "ldsr" in names:
        from srm.models.ldsr_branch import LdsrBranch

        branches["LDSR-S2"] = LdsrBranch(device=device, n_samples=n_samples, steps=steps)
    return branches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Sentinel-2 GeoTIFF, bands B04 B03 B02 B08")
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    ap.add_argument("--branches", default="bicubic,sen2sr",
                    help="comma list of: bicubic, sen2sr, ldsr")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--reflectance", action="store_true",
                    help="input is integer L2A DN and must be divided by 10000")
    ap.add_argument("--n-samples", type=int, default=8, help="LDSR stochastic samples")
    ap.add_argument("--steps", type=int, default=100, help="LDSR diffusion steps")
    args = ap.parse_args()

    arr, profile = read_bands(args.input)
    if args.reflectance:
        arr = to_reflectance(arr)
    print(f"input {arr.shape} {profile['crs']} res={profile['transform'].a:g}m")

    names = [n.strip().lower() for n in args.branches.split(",")]
    branches = build_branches(names, args.device, args.n_samples, args.steps)
    if not branches:
        print("no branches selected", file=sys.stderr)
        return 2

    primary = "SEN2SR" if "SEN2SR" in branches else next(iter(branches))
    result = run(arr, branches, scale=args.scale, fidelity=primary)

    out = pathlib.Path(args.out)
    info = write_product(out.with_suffix(".tif"), result, profile, scale=args.scale)
    mpath = write_metrics(out.parent / f"{out.name}_metrics.json", result)

    fp = check_footprint(args.input, info["path"])
    payload = json.loads(mpath.read_text())
    payload["footprint"] = fp
    mpath.write_text(json.dumps(payload, indent=2))

    print(f"\nwrote {info['path']}  bands={info['bands']}")
    for name, p in result["predictions"].items():
        c = result["consistency"][name]
        print(f"  {name:<10} {p.seconds*1000:7.0f} ms  "
              f"LR-consistency MAE {c['consistency_mae']:.5f}  SAM {c['consistency_sam_deg']:.3f}deg")
    t = result["trust"]["confidence"]
    print(f"  confidence mean {t.mean():.3f}  low-confidence area {float((t<0.5).mean()):.2%}")
    print(f"  footprint preserved: {fp['ok']}  (max drift {fp['max_bound_drift_m']:.2e} m, "
          f"res {fp['src_res'][0]:g}m -> {fp['dst_res'][0]:g}m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
