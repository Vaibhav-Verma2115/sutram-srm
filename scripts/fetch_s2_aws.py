#!/usr/bin/env python
"""Fetch real Sentinel-2 L2A crops from the public AWS archive (no account).

Uses Element84 Earth Search (STAC) over the sentinel-cogs bucket. Downloads a
windowed crop of B04/B03/B02/B08 (+SCL) as an analysis-ready 4-band GeoTIFF in
model band order, plus the SCL band for cloud masking.

    python scripts/fetch_s2_aws.py --bbox 78.30 17.90 78.42 18.00 \
        --date-range 2026-01-01/2026-03-31 --out data/raw/telangana_mar26

Why this source: Copernicus Data Space needs an account and interactive login;
the AWS mirror is anonymous, COG-native, and supports HTTP range reads, so we
can pull a small AOI without downloading a 700 MB granule.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

STAC = "https://earth-search.aws.element84.com/v1"
# Asset keys in Earth Search v1, mapped to our model band order.
ASSETS = ["red", "green", "blue", "nir"]  # -> B04, B03, B02, B08
BAND_NAMES = ["B04_red", "B03_green", "B02_blue", "B08_nir"]


def search(bbox, date_range, max_cloud):
    from pystac_client import Client

    cat = Client.open(STAC)
    items = list(cat.search(
        collections=["sentinel-2-l2a"], bbox=bbox,
        datetime=date_range.replace("/", "/"),
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).items())
    # Least cloudy first; ties broken by newest.
    items.sort(key=lambda it: (it.properties["eo:cloud_cover"],
                               it.properties["datetime"]))
    return items


def crop_asset(href: str, bbox_wgs84, out=None):
    """Range-read just the AOI window from a remote COG."""
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
        win = from_bounds(*b, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        transform = src.window_transform(win)
        return arr, transform, src.crs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("W", "S", "E", "N"), help="WGS84 lon/lat bounds")
    ap.add_argument("--date-range", default="2026-01-01/2026-03-31")
    ap.add_argument("--max-cloud", type=float, default=5.0)
    ap.add_argument("--index", type=int, default=0, help="which search hit (0 = least cloudy)")
    ap.add_argument("--tile", default="", help="restrict to an MGRS tile, e.g. 44QKE")
    ap.add_argument("--out", required=True, help="output basename")
    args = ap.parse_args()

    items = search(args.bbox, args.date_range, args.max_cloud)
    if args.tile:
        items = [it for it in items if args.tile in it.id]
    if not items:
        print("no scenes found; widen the date range or cloud limit", file=sys.stderr)
        return 1
    for i, it in enumerate(items[:5]):
        mark = " <-- selected" if i == args.index else ""
        print(f"[{i}] {it.id}  {it.properties['datetime'][:10]}  "
              f"cloud {it.properties['eo:cloud_cover']:.1f}%{mark}")
    item = items[args.index]

    bands, transform, crs = [], None, None
    for key in ASSETS:
        arr, transform, crs = crop_asset(item.assets[key].href, args.bbox)
        bands.append(arr)
        print(f"  {key}: {arr.shape}")
    stack = np.stack(bands).astype(np.float32)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=stack.shape[1], width=stack.shape[2],
                   count=4, dtype="float32", crs=crs, transform=transform,
                   compress="deflate")
    with rasterio.open(f"{out}.tif", "w", **profile) as dst:
        # AWS COGs are L2A DN (x10000); keep DN here, pipeline scales with --reflectance.
        dst.write(stack)
        for i, n in enumerate(BAND_NAMES, 1):
            dst.set_band_description(i, n)

    scl, scl_t, scl_crs = crop_asset(item.assets["scl"].href, args.bbox)
    with rasterio.open(f"{out}_SCL.tif", "w", driver="GTiff", height=scl.shape[0],
                       width=scl.shape[1], count=1, dtype="uint8", crs=scl_crs,
                       transform=scl_t, compress="deflate") as dst:
        dst.write(scl, 1)

    print(f"wrote {out}.tif ({stack.shape[2]}x{stack.shape[1]} @10m, DN) + {out}_SCL.tif")
    print(f"scene: {item.id}  {item.properties['datetime'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
