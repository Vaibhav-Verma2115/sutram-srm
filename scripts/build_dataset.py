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

THE SPLIT IS BY GROUND LOCATION, NOT BY PATCH
---------------------------------------------
A SEN2VENuS site is a fixed grid of locations imaged repeatedly: KUDALIAR is
927 locations (across two MGRS tiles) seen on up to 20 dates each, 7269 patches
in all. Splitting those 7269 patches at random puts the *same ground* in train
and val, separated only by acquisition date, so the val score rewards a model
for memorising terrain. Our first model scored 38.65 dB on such a split and
31.57 dB on a genuinely unseen scene.

So we hold out whole locations: every patch of a held-out location goes to val,
whatever date it was taken, and no held-out ground appears in training. The
holdout is drawn per site, so val still spans every biome.

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

from srm.data.sen2venus import (  # noqa: E402
    LR_COL,
    acquisition_group,
    load_pair,
    location_group,
    make_x4_input,
    read_index,
    usable,
)


class ShardWriter:
    """Buffers patches and flushes a .npz once `size` of them accumulate."""

    def __init__(self, out: pathlib.Path, split: str, size: int):
        self.out = out
        self.split = split
        self.size = size
        self.buf: list[tuple[np.ndarray, ...]] = []
        self.names: list[str] = []
        self.count = 0

    def add(self, lr: np.ndarray, hr: np.ndarray, s2: np.ndarray | None = None) -> None:
        self.buf.append((lr, hr) if s2 is None else (lr, hr, s2))
        self.count += 1
        if len(self.buf) >= self.size:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        path = self.out / f"{self.split}_{len(self.names):03d}.npz"
        arrays = {"lr": np.stack([b[0] for b in self.buf]),
                  "hr": np.stack([b[1] for b in self.buf])}
        if len(self.buf[0]) == 3:
            arrays["s2"] = np.stack([b[2] for b in self.buf])
        np.savez_compressed(path, **arrays)
        shapes = " ".join(f"{k}{v.shape}" for k, v in arrays.items())
        print(f"  {path.name}  {shapes}  {path.stat().st_size/1e6:.1f} MB")
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
    ap.add_argument("--split-by", default="location",
                    choices=["location", "acquisition"],
                    help="group held out for validation. 'location' (default) holds out "
                         "ground locations so no terrain is shared with training; "
                         "'acquisition' holds out whole overpass dates instead, which "
                         "still shares terrain and is therefore weaker.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-s2", action="store_true",
                    help="also store the native 10 m Sentinel-2 patch, enabling "
                         "same-sensor S2->S2 training pairs (see train.py --mix-s2s2)")
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

    keyfn = location_group if args.split_by == "location" else acquisition_group

    # Build a global, site-interleaved work list first, so a shard is never
    # single-biome -- otherwise the last batches of an epoch would all come from
    # one landscape and the model would see a non-stationary distribution.
    #
    # The val holdout is chosen per site, over groups rather than patches, so
    # that (a) no group straddles the split and (b) val still covers every
    # biome instead of whichever site happened to land in the first 10%.
    work: list[tuple[pathlib.Path, int, str]] = []
    val_groups: set[str] = set()
    group_stats = {}
    index_cache: dict[pathlib.Path, object] = {}

    for s in sites:
        df = read_index(s)
        index_cache[s] = df
        groups = df[LR_COL].map(keyfn).to_numpy()

        uniq = np.array(sorted(set(groups)))
        perm = rng.permutation(len(uniq))
        n_hold = max(1, int(round(len(uniq) * args.val_frac)))
        held = set(uniq[perm[:n_hold]])
        val_groups |= held

        picks = rng.permutation(len(df))[: args.max_per_site]
        work.extend((s, int(i), groups[i]) for i in picks)
        group_stats[s.name] = {
            "patches_indexed": int(len(df)),
            "patches_selected": int(len(picks)),
            "groups_total": int(len(uniq)),
            "groups_held_out": int(n_hold),
        }
        print(f"{s.name:<12} {len(df):>6} indexed, {len(picks):>6} selected, "
              f"{len(uniq):>4} {args.split_by}s ({n_hold} held out for val)")

    rng.shuffle(work)

    writers = {
        "val": ShardWriter(out, "val", args.shard_size),
        "train": ShardWriter(out, "train", args.shard_size),
    }

    kept = {s.name: 0 for s in sites}
    seen_groups = {"train": set(), "val": set()}
    skipped = 0

    print(f"\npacking {len(work)} candidate patches (scale x{args.scale}, "
          f"split by {args.split_by})")
    for site, i, group in tqdm(work, unit="patch"):
        try:
            lr, hr = load_pair(site, index_cache[site].iloc[i])
        except Exception:
            skipped += 1
            continue
        if not usable(lr, hr):
            skipped += 1
            continue
        # Keep the untouched 10 m Sentinel-2 patch before degrading it: it is
        # the HR side of a same-sensor S2->S2 pair, which trains the operator
        # we actually deploy (S2 in, S2 out) rather than a cross-sensor
        # S2 -> VENuS transfer.
        s2 = lr.astype(np.float16) if args.with_s2 else None
        # x4 mode degrades the real 10 m input to 20 m so the pair spans x4
        # against the real 5 m VENuS target. See srm/data/sen2venus.py.
        if args.scale == 4:
            lr = make_x4_input(lr)
        split = "val" if group in val_groups else "train"
        writers[split].add(lr.astype(np.float16), hr.astype(np.float16), s2)
        seen_groups[split].add(group)
        kept[site.name] += 1

    for w in writers.values():
        w.flush()

    # The whole point of the exercise -- assert it rather than trust it.
    overlap = seen_groups["train"] & seen_groups["val"]
    if overlap:
        print(f"\nFATAL: {len(overlap)} {args.split_by}(s) leaked across the split, "
              f"e.g. {sorted(overlap)[:3]}", file=sys.stderr)
        return 1

    manifest = {
        "scale": args.scale,
        "with_s2": bool(args.with_s2),
        "split_by": args.split_by,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "sites": kept,
        "site_groups": group_stats,
        "groups": {k: len(v) for k, v in seen_groups.items()},
        "leak_check": {"shared_groups": 0, "verified": True},
        "skipped": skipped,
        "n_train": writers["train"].count,
        "n_val": writers["val"].count,
        "shards": {k: w.names for k, w in writers.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e9
    print(f"\n{manifest['n_train']} train / {manifest['n_val']} val "
          f"({skipped} skipped), {total:.2f} GB -> {out}")
    print(f"leak check: {len(seen_groups['train'])} train {args.split_by}s, "
          f"{len(seen_groups['val'])} val {args.split_by}s, 0 shared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
