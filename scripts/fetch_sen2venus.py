#!/usr/bin/env python
"""Download SEN2VENuS v2.0.0 site archives from Zenodo.

The full dataset is 139.5 GB across 29 sites, which is far more than a free-tier
training run needs. We pull a small, deliberately chosen subset:

  KUDALIAR  -- Telangana, India. 7269 patches. The reason this dataset was
               chosen at all: it puts real Indian terrain in the training set,
               which is what the SIH jury will ask about.
  ANJI      -- China, mixed agriculture/forest.
  MAD-AMBO  -- Madagascar, tropical.
  SUDOUE-4  -- France, agricultural parcels.
  ESTUAMAR  -- Spain, coastal/estuary.
  FGMANAUS  -- Amazon, tiny; useful as a fast structural smoke test.

    python scripts/fetch_sen2venus.py --sites KUDALIAR,ANJI --out data/raw/sen2venus
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile

import requests
from tqdm import tqdm

RECORD = "14603764"
API = f"https://zenodo.org/api/records/{RECORD}"

# Curated subset: India first, then biome diversity, cheapest useful sites.
DEFAULT_SITES = ["KUDALIAR", "ANJI", "MAD-AMBO", "SUDOUE-4", "ESTUAMAR"]
SMOKE_SITE = "FGMANAUS"


def file_index() -> dict[str, dict]:
    r = requests.get(API, timeout=120)
    r.raise_for_status()
    return {f["key"].removesuffix(".zip"): f for f in r.json()["files"]}


def download(entry: dict, out_dir: pathlib.Path) -> pathlib.Path:
    dst = out_dir / entry["key"]
    size = entry["size"]
    if dst.exists() and dst.stat().st_size == size:
        print(f"skip {dst.name} (complete)")
        return dst

    url = entry["links"]["self"]
    headers = {}
    mode = "wb"
    start = 0
    if dst.exists() and dst.stat().st_size < size:
        start = dst.stat().st_size
        headers["Range"] = f"bytes={start}-"
        mode = "ab"
        print(f"resuming {dst.name} at {start/1e9:.2f} GB")

    out_dir.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        with dst.open(mode) as f, tqdm(
            total=size, initial=start, unit="iB", unit_scale=True, desc=entry["key"]
        ) as bar:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
                bar.update(len(chunk))
    return dst


def extract(zip_path: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    target = out_dir / zip_path.stem
    if target.exists() and any(target.iterdir()):
        print(f"skip extract {zip_path.stem} (present)")
        return target
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES),
                    help="comma-separated site names, or 'smoke' for the tiny one")
    ap.add_argument("--out", default="data/raw/sen2venus")
    ap.add_argument("--list", action="store_true", help="list sites and sizes, then exit")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    idx = file_index()
    if args.list:
        for name, f in sorted(idx.items(), key=lambda kv: kv[1]["size"]):
            print(f"{f['size']/1e9:8.2f} GB  {name}")
        return 0

    sites = [SMOKE_SITE] if args.sites == "smoke" else [
        s.strip() for s in args.sites.split(",") if s.strip()
    ]
    unknown = [s for s in sites if s not in idx]
    if unknown:
        print(f"unknown sites: {unknown}\navailable: {sorted(idx)}", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    total = sum(idx[s]["size"] for s in sites)
    print(f"{len(sites)} site(s), {total/1e9:.2f} GB total -> {out}\n")

    for s in sites:
        z = download(idx[s], out)
        if not args.no_extract:
            d = extract(z, out)
            n = sum(1 for _ in d.rglob("*.tif"))
            print(f"  {s}: {n} tif files under {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
