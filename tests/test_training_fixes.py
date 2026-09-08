"""Tests for the training-integrity fixes.

Each test here pins a bug that was actually shipped, not a hypothetical one.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srm.data.sen2venus import acquisition_group, location_group, parse_ref  # noqa: E402
from srm.train.dataset import Sen2VenusShards  # noqa: E402


# --------------------------------------------------------------------------
# Split grouping
# --------------------------------------------------------------------------

def test_parse_ref_handles_hyphenated_sites():
    """MAD-AMBO and SUDOUE-4 carry hyphens and digits in the site name."""
    p = parse_ref("MAD-AMBO_137_2018-02-25_38KQD_b2b3b4b8_10m.tif")
    assert p == {"site": "MAD-AMBO", "pid": 137, "date": "2018-02-25", "tile": "38KQD"}
    p = parse_ref("SUDOUE-4_0_2019-07-05_31TCH_b2b3b4b8_10m.tif")
    assert p["site"] == "SUDOUE-4" and p["pid"] == 0


def test_location_group_includes_tile():
    """KUDALIAR spans 44QKE and 44QKF, each numbering patches from 0.

    Dropping the tile from the key merges two different places into one group,
    which would put the same key on both sides of a split.
    """
    a = location_group("x/KUDALIAR_7_2020-04-23_44QKE_b2b3b4b8_10m.tif")
    b = location_group("x/KUDALIAR_7_2020-04-23_44QKF_b2b3b4b8_10m.tif")
    assert a != b


def test_location_group_is_date_invariant():
    """The same ground on two dates is one location -- that is the whole point."""
    a = location_group("x/KUDALIAR_7_2020-04-23_44QKE_b2b3b4b8_10m.tif")
    b = location_group("x/KUDALIAR_7_2019-12-15_44QKE_b2b3b4b8_10m.tif")
    assert a == b


def test_acquisition_group_separates_dates():
    a = acquisition_group("x/KUDALIAR_7_2020-04-23_44QKE_b2b3b4b8_10m.tif")
    b = acquisition_group("x/KUDALIAR_7_2019-12-15_44QKE_b2b3b4b8_10m.tif")
    assert a != b


def test_group_helpers_accept_row_or_string():
    ref = "x/ANJI_3_2020-02-18_50RQV_b2b3b4b8_10m.tif"
    assert location_group(ref) == location_group({"b2b3b4b8_10m": ref})


def test_parse_ref_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ref("not-a-sen2venus-name.tif")


# --------------------------------------------------------------------------
# Built dataset: the split must not leak
# --------------------------------------------------------------------------

DATA = ROOT / "data/interim/sen2venus_x4_loc"


@pytest.mark.skipif(not (DATA / "manifest.json").exists(),
                    reason="built dataset not present")
def test_manifest_split_is_leak_free():
    m = json.loads((DATA / "manifest.json").read_text())
    assert m["split_by"] == "location"
    assert m["leak_check"]["shared_groups"] == 0
    assert m["groups"]["train"] > 0 and m["groups"]["val"] > 0


@pytest.mark.skipif(not (DATA / "manifest.json").exists(),
                    reason="built dataset not present")
def test_val_covers_every_site():
    """A holdout drawn globally could miss whole biomes; ours is per site."""
    m = json.loads((DATA / "manifest.json").read_text())
    for site, g in m["site_groups"].items():
        assert g["groups_held_out"] >= 1, f"{site} contributes nothing to val"


# --------------------------------------------------------------------------
# Epoch reshuffling
# --------------------------------------------------------------------------

class _FakeShards(Sen2VenusShards):
    """Shard index without touching disk."""

    def __init__(self, counts):
        self._counts = list(counts)
        self.shards = [f"s{i}.npz" for i in range(len(counts))]
        self.index = [(si, i) for si, n in enumerate(counts) for i in range(n)]
        self.scale = 4
        self.augment = False
        self.crop = None
        self._cache_id = None
        self._cache = None


def test_reshuffle_is_a_permutation():
    """Every patch must appear exactly once -- a reshuffle that drops or
    duplicates samples would quietly change the training set size."""
    ds = _FakeShards([5, 3, 7])
    before = sorted(ds.index)
    ds.reshuffle(seed=1)
    assert len(ds.index) == 15
    assert sorted(ds.index) == before


def test_reshuffle_changes_order_and_is_seed_stable():
    a = _FakeShards([50, 50])
    b = _FakeShards([50, 50])
    a.reshuffle(1)
    b.reshuffle(1)
    assert a.index == b.index          # same seed -> reproducible
    b.reshuffle(2)
    assert a.index != b.index          # different seed -> different order


def test_reshuffle_keeps_shard_locality():
    """Reads must stay grouped by shard or the one-shard cache thrashes."""
    ds = _FakeShards([40, 40, 40])
    ds.reshuffle(3)
    visited = [si for si, _ in ds.index]
    runs = [k for k, nxt in zip(visited, visited[1:]) if k != nxt]
    assert len(runs) == len(ds.shards) - 1, "shards should be visited in blocks"


# --------------------------------------------------------------------------
# Checkpoint selection
# --------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "scripts"))


def _selection_score(vm, mode):
    import importlib.util
    spec = importlib.util.spec_from_file_location("trainmod", ROOT / "scripts/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.selection_score(vm, mode)


def test_selection_prefers_ssim_over_psnr():
    """The v2 model was chosen on PSNR, the one metric it loses on.

    A checkpoint with better SSIM and worse PSNR must win under the new default.
    """
    sharp = {"psnr": 31.5, "ssim": 0.84, "sam_deg": 1.80}
    smooth = {"psnr": 33.0, "ssim": 0.79, "sam_deg": 2.10}
    assert _selection_score(sharp, "ssim") > _selection_score(smooth, "ssim")
    assert _selection_score(smooth, "psnr") > _selection_score(sharp, "psnr")


def test_composite_penalises_spectral_drift():
    """Equal SSIM must not tie when one candidate drifts spectrally."""
    clean = {"psnr": 32.0, "ssim": 0.84, "sam_deg": 1.8}
    drifted = {"psnr": 32.0, "ssim": 0.84, "sam_deg": 4.5}
    assert _selection_score(clean, "composite") > _selection_score(drifted, "composite")


# --------------------------------------------------------------------------
# TTA uncertainty
# --------------------------------------------------------------------------

def test_d4_transforms_round_trip():
    """sigma is built by averaging outputs mapped back through _d4_inv.

    If the inverse is wrong the eight predictions are misaligned, the mean
    blurs and sigma reports disagreement that is really just misregistration --
    a bug that would look like a plausible uncertainty map.
    """
    from srm.models.ours_branch import OursBranch

    x = np.random.default_rng(0).random((4, 12, 20)).astype(np.float32)
    for flip in (False, True):
        for k in range(4):
            back = OursBranch._d4_inv(OursBranch._d4(x, k, flip), k, flip)
            assert back.shape == x.shape
            assert np.allclose(back, x), f"round trip failed for k={k} flip={flip}"


def test_d4_transforms_are_distinct():
    """Eight genuinely different views, or the spread is meaningless."""
    from srm.models.ours_branch import OursBranch

    x = np.random.default_rng(1).random((4, 16, 16)).astype(np.float32)
    seen = {OursBranch._d4(x, k, f).tobytes() for f in (False, True) for k in range(4)}
    assert len(seen) == 8


CKPT = ROOT / "checkpoints/best.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
def test_tta_produces_sigma_and_plain_does_not():
    """The shipped product carried an all-zero sigma band whenever the slow
    diffusion branch was skipped. TTA must fill it in."""
    from srm.models.ours_branch import OursBranch

    lr = np.random.default_rng(2).random((4, 32, 32)).astype(np.float32)
    plain = OursBranch(device="cpu", tta=False).predict(lr)
    assert plain.sigma is None and not plain.has_uncertainty

    tta = OursBranch(device="cpu", tta=True).predict(lr)
    assert tta.has_uncertainty
    assert tta.sigma.shape == tta.sr.shape[1:]
    assert np.isfinite(tta.sigma).all()
    assert tta.sigma.min() >= 0.0
    assert tta.sigma.max() > 0.0, "a constant sigma carries no information"


# --------------------------------------------------------------------------
# Tiled inference
# --------------------------------------------------------------------------

def test_hann_window_is_strictly_positive():
    """np.hanning(2*overlap) starts at exactly 0.

    tiled_predict normalises by accumulated weight, so a zero weight turns the
    prediction into 0 instead of itself. That blacked out the first row and
    column of every product this pipeline ever wrote.
    """
    from srm.preprocess.prepare import hann_window

    w = hann_window(512, 128)
    assert w.min() > 0.0, "a zero weight destroys the pixel it covers"
    assert w[256, 256] == pytest.approx(1.0), "interior must be unattenuated"
    assert np.allclose(w, w[::-1, ::-1]), "window must be symmetric"


@pytest.mark.parametrize("shape", [(4, 32, 32), (4, 128, 128), (4, 150, 97), (4, 300, 260)])
def test_tiled_predict_reconstructs_exactly(shape):
    """With an exact model, tiling must be the identity -- at every size,
    including inputs smaller than one tile and non-square rasters."""
    from srm.models.tiling import tiled_predict

    def exact(p):
        return np.repeat(np.repeat(p, 4, axis=1), 4, axis=2).astype(np.float32)

    lr = np.random.default_rng(0).random(shape).astype(np.float32)
    out = tiled_predict(lr, exact, tile=128, scale=4)
    ref = exact(lr)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-5


def test_tiled_predict_preserves_the_first_row_and_column():
    """The exact regression: the boundary ring must not be zeroed."""
    from srm.models.tiling import tiled_predict

    def exact(p):
        return np.repeat(np.repeat(p, 4, axis=1), 4, axis=2).astype(np.float32)

    lr = np.random.default_rng(1).random((4, 32, 32)).astype(np.float32) * 0.5 + 0.25
    out = tiled_predict(lr, exact, tile=128, scale=4)
    assert out[:, 0, :].min() > 0.0, "first row was blacked out"
    assert out[:, :, 0].min() > 0.0, "first column was blacked out"
