#!/usr/bin/env python
"""Application vignettes: crop monitoring, urban analysis, disaster/change.

The PS names three applications. Each vignette is one figure comparing the
analysis at 10 m against the same analysis at 2.5 m on real Telangana imagery
(tile 44QKE, Nov 2025 vs Mar 2026 -- rabi sowing to pre-harvest).
"""
from __future__ import annotations

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

from srm.trust.layer import ndvi  # noqa: E402
from srm.validate.downstream import ndvi_field_boundaries  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"


def load(path, reflectance=False):
    with rasterio.open(path) as s:
        a = s.read().astype(np.float32)
    return np.clip(a / 10000, 0, 1) if reflectance else a


def stretch(rgb, pct=2):
    out = np.zeros_like(rgb)
    for i in range(rgb.shape[-1]):
        lo, hi = np.percentile(rgb[..., i], [pct, 100 - pct])
        out[..., i] = np.clip((rgb[..., i] - lo) / max(hi - lo, 1e-8), 0, 1)
    return out


lr_nov = load(ROOT / "data/raw/telangana_2025-11.tif", reflectance=True)
lr_mar = load(ROOT / "data/raw/telangana_2026-03.tif", reflectance=True)
sr_nov = load(ROOT / "data/outputs/telangana_2025-11_sr.tif")[:4]
sr_mar = load(ROOT / "data/outputs/telangana_2026-03_sr.tif")[:4]

# ---- 1. Disaster / change detection: dNDVI Nov -> Mar
y, x, w = 620, 700, 220  # LR window; SR window is 4x
d_lr = ndvi(lr_mar) - ndvi(lr_nov)
d_sr = ndvi(sr_mar) - ndvi(sr_nov)
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
for a, (d, t, res) in zip(ax, [(d_lr[y:y+w, x:x+w], "Change at 10 m", 10),
                               (d_sr[4*y:4*(y+w), 4*x:4*(x+w)], "Change at 2.5 m (SR)", 2.5)]):
    im = a.imshow(d, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    a.set_title(f"{t}  |  ΔNDVI Nov 2025 → Mar 2026", fontsize=13)
    a.axis("off")
fig.colorbar(im, ax=ax, shrink=0.7, label="ΔNDVI (red = vegetation loss)")
plt.savefig(OUT / "vignette_change_detection.png", dpi=110, bbox_inches="tight")
plt.close()
print("change-detection vignette: mean |dNDVI| lr=%.4f sr=%.4f (should be ~equal: "
      "signal preserved, boundaries sharper)" % (np.abs(d_lr).mean(), np.abs(d_sr).mean()))

# ---- 2. Crop monitoring: NDVI + field boundaries
y, x, w = 760, 850, 150
fig, ax = plt.subplots(2, 2, figsize=(13, 12))
for r, (arr, res) in enumerate([(lr_mar, "10 m"), (sr_mar, "2.5 m SR")]):
    win = (slice(y, y+w), slice(x, x+w)) if r == 0 else (slice(4*y, 4*(y+w)), slice(4*x, 4*(x+w)))
    n = ndvi(arr)[win]
    sub = arr[(slice(None),) + win]
    edges = ndvi_field_boundaries(sub)
    ax[r, 0].imshow(n, cmap="YlGn", vmin=0, vmax=0.8)
    ax[r, 0].set_title(f"NDVI at {res}", fontsize=13)
    rgb = stretch(sub[:3].transpose(1, 2, 0))
    rgb[edges] = [1, 0, 0]
    ax[r, 1].imshow(rgb)
    ax[r, 1].set_title(f"Field boundaries at {res}", fontsize=13)
    for a in ax[r]:
        a.axis("off")
plt.tight_layout()
plt.savefig(OUT / "vignette_crop_monitoring.png", dpi=110, bbox_inches="tight")
plt.close()
print("crop vignette written")

# ---- 3. Urban: settlement structure
y, x, w = 690, 640, 110
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
ax[0].imshow(stretch(np.repeat(np.repeat(lr_mar[:3, y:y+w, x:x+w], 4, 1), 4, 2).transpose(1, 2, 0)),
             interpolation="nearest")
ax[0].set_title("Settlement at 10 m", fontsize=13)
ax[1].imshow(stretch(sr_mar[:3, 4*y:4*(y+w), 4*x:4*(x+w)].transpose(1, 2, 0)))
ax[1].set_title("Settlement at 2.5 m (SR) — structures separable", fontsize=13)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(OUT / "vignette_urban.png", dpi=110, bbox_inches="tight")
plt.close()
print("urban vignette written")
