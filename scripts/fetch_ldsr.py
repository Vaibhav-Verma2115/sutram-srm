"""Download LDSR-S2 (opensr-model) pretrained weights + example scenes.

NOTE: opensr-model 1.1.1 hardcodes 'opensr_10m_v4_v4.ckpt', which no longer
exists in the HF repo. We fetch the current checkpoint explicitly instead.
"""
import pathlib
import requests
from tqdm import tqdm

REPO = "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main"
FILES = ["opensr_10m_v4_v6.ckpt", "example_urban.pt", "example_rural.pt"]
OUT = pathlib.Path(__file__).resolve().parents[1] / "models" / "ldsr"


def fetch(name: str) -> pathlib.Path:
    dst = OUT / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"skip {name} (already present)")
        return dst
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with requests.get(f"{REPO}/{name}", stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(total=total, unit="iB", unit_scale=True, desc=name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dst)
    return dst


if __name__ == "__main__":
    for name in FILES:
        p = fetch(name)
        print("OK", p, f"{p.stat().st_size / 1e6:.1f} MB")
