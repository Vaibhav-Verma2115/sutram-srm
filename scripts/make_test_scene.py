"""Write SEN2SR example patches out as georeferenced S2-like GeoTIFFs.

Produces two files:
  test_scene_10m.tif  -- the 128x128 LR patch, for inference smoke tests
  test_ref_2p5m.tif   -- the 512x512 HR reference, used as ground truth by
                         Wald's protocol (degrade -> super-resolve -> compare)

Gives us real CRS/transform to exercise the geospatial path before the
Copernicus scenes are downloaded.
"""
import pathlib
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import mlstac  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANDS = ["B04_red", "B03_green", "B02_blue", "B08_nir"]
# UTM 43N (covers much of northern India); arbitrary but valid origin.
ORIGIN = (700000.0, 3100000.0)
CRS = "EPSG:32643"


def write(path: pathlib.Path, arr: np.ndarray, res: float) -> None:
    profile = dict(
        driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
        dtype="float32", crs=CRS, transform=from_origin(*ORIGIN, res, res),
        compress="deflate",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32))
        for i, n in enumerate(BANDS, 1):
            dst.set_band_description(i, n)
    print(f"wrote {path.name}  {arr.shape}  {CRS} @{res}m")


ml = mlstac.load("models/SEN2SRLite_NonReference_RGBN_x4")
lr, hr = ml.example_data()
write(ROOT / "data" / "raw" / "test_scene_10m.tif", lr[0].numpy(), 10.0)
write(ROOT / "data" / "raw" / "test_ref_2p5m.tif", hr[0].numpy(), 2.5)
