#!/usr/bin/env python
"""Pack SEN2VENuS sites into compact float16 shards for Colab/Kaggle training.

Reading thousands of small GeoTIFFs out of nested zips is far too slow to do
inside a training loop on a free-tier VM, and Drive/Kaggle both punish many
small files. We therefore pre-pack into a handful of .npz shards, held as
float16 (reflectance needs ~3 decimal places; float16 gives ~3 and halves both
the upload and the epoch time).

    python scripts/build_dataset.py --sites KUDALIAR --scale 4 --max-per-site 6000

Output: data/interim/sen2venus_x{scale}/{split}_{NNN}.npz with arrays lr, hr.
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

from srm.data.sen2venus import load_pair, make_x4_input, read_index, usable  # noqa: E402


def build_site(site_dir: pathlib.Path, scale: int, max_patches: int, rng) -> list:
    df = read_index(site_dir)
    order = rng.permutation(len(df))[:max_patches]
    pairs = []
    for i in tqdm(order, desc=site_dir.name, leave=False):
        try:
            lr, hr = load_pair(site_dir, df.iloc[int(i)])
        except Exception:
            continue
        if not usable(lr, hr):
            continue
        # x4 mode degrades the real 10 m input to 20 m so the pair spans x4
        # against the real 5 m VENuS target. See srm/data/sen2venus.py.
        if scale == 4:
            lr = make_x4_input(lr)
        pairs.append((lr.astype(np.float16), hr.astype(np.float16)))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw/sen2venus")
    ap.add_argument("--sites", default="", help="comma list; default = all downloaded")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4])
    ap.add_argument("--max-per-site", type=int, default=6000)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if args.sites:
        sites = [root / s.strip() for s in args.sites.split(",") if s.strip()]
    else:
        sites = sorted(p for p in root.iterdir() if p.is_dir() and (p / "index.csv").exists())
    if not sites:
        print(f"no sites under {root}; run scripts/fetch_sen2venus.py first", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out or f"data/interim/sen2venus_x{args.scale}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_pairs, provenance = [], {}
    for s in sites:
        got = build_site(s, args.scale, args.max_per_site, rng)
        provenance[s.name] = len(got)
        all_pairs.extend(got)
        print(f"{s.name:<12} {len(got):>6} usable patches")

    if not all_pairs:
        print("no usable patches found", file=sys.stderr)
        return 1

    # Shuffle across sites so a shard is never single-biome -- otherwise the
    # last batches of an epoch would all come from one landscape.
    idx = rng.permutation(len(all_pairs))
    n_val = int(len(idx) * args.val_frac)
    splits = {"val": idx[:n_val], "train": idx[n_val:]}

    manifest = {"scale": args.scale, "sites": provenance, "shards": {},
                "n_train": len(splits["train"]), "n_val": len(splits["val"])}

    for split, ids in splits.items():
        shards = []
        for k in range(0, len(ids), args.shard_size):
            chunk = ids[k : k + args.shard_size]
            lr = np.stack([all_pairs[j][0] for j in chunk])
            hr = np.stack([all_pairs[j][1] for j in chunk])
            path = out / f"{split}_{k // args.shard_size:03d}.npz"
            np.savez_compressed(path, lr=lr, hr=hr)
            shards.append(path.name)
            print(f"  {path.name}  lr{lr.shape} hr{hr.shape}  {path.stat().st_size/1e6:.1f} MB")
        manifest["shards"][split] = shards

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e9
    print(f"\n{manifest['n_train']} train / {manifest['n_val']} val, {total:.2f} GB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
