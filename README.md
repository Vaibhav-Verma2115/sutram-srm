# Sutram SRM — Deep Learning Super-Resolution Mapping

Sentinel-2 **10 m → 2.5 m** super-resolution with **per-pixel trust**, for the SIH problem
statement *"Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution
Satellite Imageries"*.

The problem statement asks for two things that pull against each other: reconstruct
fine-scale detail, *and* be honest that reconstructed detail is inferred rather than
observed. This pipeline resolves that by never shipping a super-resolved pixel without the
map that says how much to trust it.

## Pipeline

```
Sentinel-2 L2A (B04 B03 B02 B08 @ 10 m)
   └─ Preprocessing ......... SCL cloud mask, reflectance scaling, tiling with overlap
   └─ Branches
        ├─ Bicubic ......... baseline control
        ├─ SEN2SR .......... fidelity branch, hard radiometric constraint (ESA OpenSR)
        ├─ LDSR-S2 ......... generative detail + per-pixel sigma from N diffusion samples
        └─ Our ESRGAN ...... team-trained on SEN2VENuS  (in progress, Phase B)
   └─ Trust layer ........... LR-consistency, spectral angle, branch disagreement, sigma
   └─ Reference validation .. Wald protocol, consistency check, calibration
   └─ 2.5 m COG GeoTIFF ..... 4 SR bands + sigma + confidence
   └─ Downstream ............ NDVI, building edges, change detection
```

## Results so far

Wald protocol on the 512×512 reference tile (degrade ×4 → super-resolve → compare):

| branch  | PSNR  | SSIM   | SAM°  | ERGAS | time    |
|---------|-------|--------|-------|-------|---------|
| bicubic | 32.68 | 0.7965 | 2.025 | 3.735 | 1 ms    |
| SEN2SR  | 33.21 | 0.8194 | 1.904 | 3.500 | 40 ms   |

LR-consistency on the real 10 m input — *does the product survive being re-observed by the
sensor?* (lower is better; this needs no ground truth, so it runs on every product):

| branch  | consistency MAE | consistency SAM° |
|---------|-----------------|------------------|
| SEN2SR  | **0.00384**     | **0.880**        |
| bicubic | 0.00488         | 1.032            |
| LDSR-S2 | 0.00586         | 1.577            |

That ordering is the whole argument for a two-branch design: LDSR-S2 produces the sharpest
imagery but drifts furthest from what the sensor actually measured, so it is offered for
visual interpretation while SEN2SR carries the analysis-grade pixels.

**The trust layer is calibrated, not decorative.** Binning pixels by predicted confidence
and measuring actual error against ground truth:

| confidence | 0.0–0.1 | 0.3–0.4 | 0.6–0.7 | 0.9–1.0 |
|------------|---------|---------|---------|---------|
| mean error | 0.0366  | 0.0296  | 0.0223  | 0.0087  |

Error falls monotonically as confidence rises (corr = −0.38): the map genuinely predicts
where the model is wrong.

## Setup

```bash
brew install uv
uv venv --python 3.12
uv pip install -e .
uv pip install sen2sr mlstac opensr-model streamlit
python scripts/fetch_ldsr.py          # 2.1 GB diffusion weights (optional)
python scripts/make_test_scene.py     # georeferenced test tiles
```

## Usage

```bash
# Super-resolve a scene -> COG with sigma + confidence bands
python scripts/run_inference.py \
    --input data/raw/scene.tif --out data/outputs/scene_sr \
    --branches bicubic,sen2sr,ldsr

# Benchmark all branches with Wald's protocol
python scripts/evaluate.py --input data/raw/test_ref_2p5m.tif

# Interactive demo (runs offline on precomputed products)
streamlit run app/demo.py
```

Input rasters must be band order **B04, B03, B02, B08**. Pass `--reflectance` if the input
is integer L2A DN rather than 0–1 float.

## Layout

| path | contents |
|------|----------|
| `src/srm/preprocess/` | SCL masking, tiling, Hann feathering |
| `src/srm/models/` | branch wrappers behind one `predict()` interface |
| `src/srm/trust/` | LR-consistency, SAM, ΔNDVI, confidence fusion |
| `src/srm/validate/` | Wald protocol, consistency check, calibration curve |
| `src/srm/metrics/` | reflectance-aware PSNR/SSIM/SAM/ERGAS |
| `src/srm/io/` | COG writing, CRS/footprint preservation |
| `scripts/` | inference, evaluation, data fetching |
| `app/` | Streamlit demo |

## Design notes

- **Reflectance-aware throughout.** Metrics use max reflectance = 1.0, not 255. Puri &
  Kotze (2022, AGILE) show that 0–255-oriented losses distort results on 16-bit satellite
  data; their SRGAN baseline on this exact task reached PSNR ≈ 29.9 / SSIM ≈ 0.71.
- **Degradation uses the sensor PSF**, not bicubic. A model trained to invert bicubic
  learns the wrong operator and fails on real imagery.
- **Footprint is verified, not assumed.** `check_footprint()` runs after every write; the
  affine transform is rescaled so pixel size shrinks while bounds stay fixed.
- **Custom tiling.** `sen2sr.predict_large` derives output height from input *width*, so it
  corrupts non-square rasters; `src/srm/models/tiling.py` replaces it with Hann-feathered
  blending.

## Credits

Built on [ESA OpenSR](https://opensr.eu/) — SEN2SR (Aybar et al., *RSE* 2025) and LDSR-S2.
Benchmarking follows the [opensr-test](https://github.com/ESAOpenSR/opensr-test)
consistency / synthesis / hallucination framing.
