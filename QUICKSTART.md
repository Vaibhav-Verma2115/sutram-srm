# How to run this

Everything runs from the project root:

```bash
cd "/Users/vaibhavverma/Documents/Sutram SIH"
```

There is a virtualenv at `.venv` (Python 3.12). Two ways to use it:

```bash
source .venv/bin/activate     # then just: python scripts/...
# or, without activating, prefix every command:
.venv/bin/python scripts/...
```

The examples below use the prefix form so they work in a fresh terminal.

---

## 1. See the demo (start here)

```bash
.venv/bin/python -m streamlit run app/demo.py
```

Opens at <http://localhost:8501>. Four tabs: **Imagery** (before/after),
**Trust layer** (confidence map + red flags on invented detail), **NDVI**,
**Metrics**. It reads precomputed GeoTIFFs from `data/outputs/`, so it works
offline and cannot fail live during a presentation.

Stop it with `Ctrl+C`.

---

## 2. Super-resolve an image

```bash
.venv/bin/python scripts/run_inference.py \
    --input data/raw/test_scene_10m.tif \
    --out   data/outputs/my_result \
    --branches bicubic,sen2sr
```

Writes `my_result.tif` (6 bands: 4 super-resolved + sigma + confidence) and
`my_result_metrics.json`.

| flag | meaning |
|------|---------|
| `--branches` | any of `bicubic,sen2sr,ldsr,ours` (comma separated) |
| `--device` | `cpu`, `mps` (Apple GPU), or `cuda` |
| `--reflectance` | add this if your input is raw L2A integers, not 0–1 floats |
| `--n-samples` | LDSR only: how many diffusion samples for the sigma map (default 8) |

**Note on `ldsr`:** it is slow (~100 s per tile on CPU). Fine for generating demo
products in advance, not for live use.

**On your own Sentinel-2 data:** the input must be a GeoTIFF with four bands in
the order **B04, B03, B02, B08** (red, green, blue, NIR).

---

## 3. Benchmark the models

```bash
.venv/bin/python scripts/evaluate.py \
    --input data/raw/test_ref_2p5m.tif \
    --branches bicubic,sen2sr,ours
```

Prints the PSNR / SSIM / SAM / ERGAS table plus the trust-layer calibration
curve, and writes `data/outputs/benchmark.json`.

This uses **Wald's protocol**: it degrades the reference image ×4, super-resolves
it back, and scores against the original — the only way to get true reference
metrics without owning 2.5 m ground truth.

---

## 4. Train our own model

Three steps. Step 1 is a large download and only needs doing once.

```bash
# 1. Get the data (7.9 GB; KUDALIAR = Telangana, India). Resumable.
.venv/bin/python scripts/fetch_sen2venus.py --sites KUDALIAR

#    See all 29 available sites and their sizes:
.venv/bin/python scripts/fetch_sen2venus.py --list

# 2. Pack it into training shards
.venv/bin/python scripts/build_dataset.py --sites KUDALIAR --scale 4

# 3. Train (~16 min for 40 epochs on the Mac's GPU)
.venv/bin/python scripts/train.py \
    --data data/interim/sen2venus_x4 \
    --epochs 40 --device mps
```

Use `--device mps` on the Mac — it is about 23× faster than `cpu`.

Checkpoints land in `checkpoints/` every epoch. If training is interrupted:

```bash
.venv/bin/python scripts/train.py --data data/interim/sen2venus_x4 \
    --epochs 40 --device mps --resume checkpoints/last.pt
```

Once `checkpoints/best.pt` exists, `--branches ours` works everywhere.

### Training on Colab instead

`notebooks/train_colab.ipynb` does all of the above on a free T4, checkpointing
to Google Drive so a disconnect costs at most one epoch. You need the repo on
GitHub first, then set `REPO_URL` in cell 3.

---

## 5. Regenerate the PDF documentation

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="docs/Sutram_SRM_Pipeline.pdf" \
    "file://$PWD/docs/pipeline.html"
```

Edit `docs/pipeline.html` and re-run to update. Open with
`open docs/Sutram_SRM_Pipeline.pdf`.

---

## Getting Sentinel-2 data

1. Go to <https://browser.dataspace.copernicus.eu> (free account).
2. Pick an area, filter to **Sentinel-2 L2A**, low cloud cover.
3. Download, then build a 4-band GeoTIFF in B04/B03/B02/B08 order.
4. Run it through step 2 above with `--reflectance`.

Start with a small crop (~2000×2000 px). A full tile is 10980² and will be slow.

---

## If something breaks

| Symptom | Fix |
|---------|-----|
| `command not found: python` | Use `.venv/bin/python`, not `python` |
| `ModuleNotFoundError: srm` | Run from the project root, not from `scripts/` |
| `No such file: checkpoints/best.pt` | Train first (step 4), or drop `ours` from `--branches` |
| Streamlit shows "No products found" | Run step 2 first to create something in `data/outputs/` |
| Shape mismatch in SEN2SR | Input bands must be exactly 4, in B04/B03/B02/B08 order |
| Download died partway | Just re-run the same command — it resumes |
