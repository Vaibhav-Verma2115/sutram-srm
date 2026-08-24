"""Torch dataset over the packed SEN2VENuS shards.

Shards are loaded lazily and cached one at a time: a free-tier Colab VM has
~12 GB RAM, so holding the whole set in memory is not an option, but reopening
a .npz per sample is far too slow. Sampling is sequential within a shard and the
shard order is shuffled instead, which keeps I/O linear.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


class Sen2VenusShards(Dataset):
    def __init__(
        self,
        root: str | pathlib.Path,
        split: str = "train",
        augment: bool = True,
        crop: int | None = None,
    ):
        self.root = pathlib.Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.scale = manifest["scale"]
        self.shards = manifest["shards"][split]
        self.augment = augment and split == "train"
        self.crop = crop

        # Index (shard, offset) pairs up front so __len__ is exact.
        self.index: list[tuple[int, int]] = []
        for si, name in enumerate(self.shards):
            with np.load(self.root / name) as z:
                n = z["lr"].shape[0]
            self.index.extend((si, i) for i in range(n))

        self._cache_id: int | None = None
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, si: int):
        if self._cache_id != si:
            with np.load(self.root / self.shards[si]) as z:
                self._cache = (z["lr"], z["hr"])
            self._cache_id = si
        return self._cache

    def __getitem__(self, i: int):
        si, off = self.index[i]
        lr_all, hr_all = self._shard(si)
        lr = lr_all[off].astype(np.float32)
        hr = hr_all[off].astype(np.float32)

        if self.crop and lr.shape[1] > self.crop:
            y = np.random.randint(0, lr.shape[1] - self.crop + 1)
            x = np.random.randint(0, lr.shape[2] - self.crop + 1)
            lr = lr[:, y : y + self.crop, x : x + self.crop]
            s = self.scale
            hr = hr[:, y * s : (y + self.crop) * s, x * s : (x + self.crop) * s]

        if self.augment:
            # Dihedral augmentation only. No colour jitter: shifting reflectance
            # would teach the model that radiometry is negotiable, which is the
            # one thing this pipeline must not learn.
            k = np.random.randint(4)
            if k:
                lr = np.rot90(lr, k, axes=(1, 2))
                hr = np.rot90(hr, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                lr = lr[:, :, ::-1]
                hr = hr[:, :, ::-1]

        return (
            torch.from_numpy(np.ascontiguousarray(lr)),
            torch.from_numpy(np.ascontiguousarray(hr)),
        )
