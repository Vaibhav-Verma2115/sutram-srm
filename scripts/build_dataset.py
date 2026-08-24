#!/usr/bin/env python
"""Pack SEN2VENuS sites into compact float16 shards for Colab/Kaggle training.

Reading thousands of small GeoTIFFs out of nested zips is far too slow to do
inside a training loop on a free-tier VM, and Drive/Kaggle both punish many
small files. We therefore pre-pack into a handful of .npz shards, held as
float16 (reflectance needs ~3 decimal places; float16 gives ~3 and halves both
the upload and the epoch time).

Shards are written incrementally rather than accumulated: KUDALIAR alone is
~4 GB of patches, which would sit dangerously close to a free Colab VM's 12 GB
ceiling if buffered whole. We stream a shard-sized buffer and flush it.

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


class ShardWriter:
    """Buffers patches and flushes a .npz once `size` of them accumulate."""

    def __init__(self, out: pathlib.Path, split: str, size: int):
        self.out = out
        self.split = split
        self.size = size
        self.buf: list[tuple[np.ndarray, np.ndarray]] = []
        self.names: list[str] = []
        self.count = 0

    def add(self, lr: np.ndarray, hr: np.ndarray) -> None:
        self.buf.append((lr, hr))
        self.count += 1
        if len(self.buf) >= self.size:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        path = self.out / f"{self.split}_{len(self.names):03d}.npz"
        lr = np.stack([b[0] for b in self.buf])
        hr = np.stack([b[1] for b in self.buf])
        np.savez_compressed(path, lr=lr, hr=hr)
        print(f"  {path.name}  lr{lr.shape} hr{hr.shape}  {path.stat().st_size/1e6:.1f} MB")
        self.names.append(path.name)
        self.buf.clear()


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
    missing = [s for s in sites if not (s / "index.csv").exists()]
    if missing:
        print(f"missing index.csv for: {[m.name for m in missing]}\n"
              f"run scripts/fetch_sen2venus.py first", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out or f"data/interim/sen2venus_x{args.scale}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Build a global, site-interleaved work list first, so a shard is never
    # single-biome -- otherwise the last batches of an epoch would all come from
    # one landscape and the model would see a non-stationary distribution.
    work = []
    for s in sites:
        df = read_index(s)
        picks = rng.permutation(len(df))[: args.max_per_site]
        work.extend((s, int(i)) for i in picks)
        print(f"{s.name:<12} {len(df):>6} indexed, {len(picks):>6} selected")
    rng.shuffle(work)

    n_val = int(len(work) * args.val_frac)
    writers = {
        "val": ShardWriter(out, "val", args.shard_size),
        "train": ShardWriter(out, "train", args.shard_size),
    }

    kept = {s.name: 0 for s in sites}
    skipped = 0
    index_cache: dict[pathlib.Path, object] = {}

    print(f"\npacking {len(work)} candidate patches (scale x{args.scale})")
    for n, (site, i) in enumerate(tqdm(work, unit="patch")):
        if site not in index_cache:
            index_cache[site] = read_index(site)
        try:
            lr, hr = load_pair(site, index_cache[site].iloc[i])
        except Exception:
            skipped += 1
            continue
        if not usable(lr, hr):
            skipped += 1
            continue
        # x4 mode degrades the real 10 m input to 20 m so the pair spans x4
        # against the real 5 m VENuS target. See srm/data/sen2venus.py.
        if args.scale == 4:
            lr = make_x4_input(lr)
        split = "val" if n < n_val else "train"
        writers[split].add(lr.astype(np.float16), hr.astype(np.float16))
        kept[site.name] += 1

    for w in writers.values():
        w.flush()

    manifest = {
        "scale": args.scale,
        "sites": kept,
        "skipped": skipped,
        "n_train": writers["train"].count,
        "n_val": writers["val"].count,
        "shards": {k: w.names for k, w in writers.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e9
    print(f"\n{manifest['n_train']} train / {manifest['n_val']} val "
          f"({skipped} skipped), {total:.2f} GB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
