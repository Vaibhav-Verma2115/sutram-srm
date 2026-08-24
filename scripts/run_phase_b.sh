#!/bin/bash
# Chain: wait for the SEN2VENuS fetch -> build shards -> train.
# Logs clear [STAGE] markers so a monitor can report progress without tailing
# raw output.
set -o pipefail
cd "/Users/vaibhavverma/Documents/Sutram SIH"
PY=.venv/bin/python
FETCH_PID="${1:-}"

if [ -n "$FETCH_PID" ]; then
  echo "[STAGE] waiting for fetch pid $FETCH_PID"
  while kill -0 "$FETCH_PID" 2>/dev/null; do sleep 30; done
fi

if [ ! -f data/raw/sen2venus/KUDALIAR/index.csv ]; then
  echo "[FAIL] KUDALIAR not extracted; index.csv missing"
  exit 1
fi
echo "[STAGE] fetch complete: $(du -sh data/raw/sen2venus/KUDALIAR | cut -f1) extracted"

echo "[STAGE] building dataset"
$PY scripts/build_dataset.py --sites KUDALIAR --scale 4 \
    --max-per-site 6000 --shard-size 1000 2>&1 | grep -v "patch/s\]" || {
  echo "[FAIL] build_dataset failed"; exit 1; }
echo "[STAGE] dataset built"

echo "[STAGE] training start (mps, 40 epochs)"
$PY scripts/train.py --data data/interim/sen2venus_x4 \
    --epochs 40 --batch-size 16 --lr 2e-4 --device mps 2>&1 || {
  echo "[FAIL] training failed"; exit 1; }
echo "[STAGE] training complete"
