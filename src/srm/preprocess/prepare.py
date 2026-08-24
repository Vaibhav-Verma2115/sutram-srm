"""Sentinel-2 L2A preprocessing: cloud masking, scaling, tiling.

Implements the "Preprocessing" stage of the pipeline: SCL mask, scale,
tile with overlap.
"""

from __future__ import annotations

import numpy as np

# Scene Classification Layer classes to treat as invalid.
# 0 no-data, 1 saturated, 3 cloud shadow, 8 cloud medium prob,
# 9 cloud high prob, 10 thin cirrus.
SCL_INVALID = (0, 1, 3, 8, 9, 10)

# Model input band order for SEN2SR / LDSR-S2 (verified from mlm.json).
BAND_ORDER = ("B04", "B03", "B02", "B08")


def scl_mask(scl: np.ndarray, invalid: tuple[int, ...] = SCL_INVALID) -> np.ndarray:
    """Boolean mask, True where the pixel is valid (clear land/water)."""
    return ~np.isin(scl, invalid)


def cloud_fraction(scl: np.ndarray) -> float:
    """Fraction of the scene flagged as cloud/shadow/cirrus."""
    return float(1.0 - scl_mask(scl).mean())


def tile_positions(height: int, width: int, tile: int = 128, overlap: int = 32):
    """Top-left (row, col) positions covering the image with `overlap` px stride.

    The last row/column is clamped to the image edge so the whole raster is
    covered without padding.
    """
    step = tile - overlap
    rows = list(range(0, max(height - tile, 0) + 1, step))
    cols = list(range(0, max(width - tile, 0) + 1, step))
    if rows[-1] != height - tile and height > tile:
        rows.append(height - tile)
    if cols[-1] != width - tile and width > tile:
        cols.append(width - tile)
    return [(r, c) for r in rows for c in cols]


def hann_window(size: int, overlap: int) -> np.ndarray:
    """2-D feathering weight for seamless tile blending.

    Ramps up over `overlap` pixels at each edge and stays 1.0 in the middle,
    so overlapping predictions cross-fade instead of producing seam lines.
    """
    ramp = np.hanning(overlap * 2)[:overlap]
    w = np.ones(size, dtype=np.float32)
    w[:overlap] = ramp
    w[-overlap:] = ramp[::-1]
    return np.outer(w, w).astype(np.float32)
