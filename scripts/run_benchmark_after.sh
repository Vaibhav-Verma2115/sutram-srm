#!/bin/bash
# Waits for the Phase B chain to exit, then benchmarks all four branches.
# Appends to the same log so the existing monitor picks up the markers.
cd "/Users/vaibhavverma/Documents/Sutram SIH"
PY=.venv/bin/python
CHAIN_PID="${1:-}"

if [ -n "$CHAIN_PID" ]; then
  while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 30; done
fi

if [ ! -f checkpoints/best.pt ]; then
  echo "[FAIL] no checkpoints/best.pt -- training did not produce a model"
  exit 1
fi

echo "[STAGE] benchmarking all branches"
$PY scripts/make_test_scene.py 2>&1 | tail -2

# Wald protocol on the held-out reference tile: bicubic vs SEN2SR vs ours.
# LDSR is excluded here -- 100 s/tile on CPU would dominate the runtime; it is
# benchmarked separately.
$PY scripts/evaluate.py --input data/raw/test_ref_2p5m.tif \
    --branches bicubic,sen2sr,ours --device cpu \
    --out data/outputs/benchmark_ours.json 2>&1 | grep -vE "^\s*$"

echo "[STAGE] benchmark complete"
