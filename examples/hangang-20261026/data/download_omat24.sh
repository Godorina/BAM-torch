#!/bin/bash
# Download Meta's OMat24 (.aselmdb) into $DATA_ROOT/omat24/raw/{train,val}/<subset>/.
# Same releases as the 2026-10-26 run: train 241018, val 241220 (val md5-checked against it).
# Resumable: wget -c, and a subset with a .done marker is skipped.
set -euo pipefail
ROOT=${DATA_ROOT:-/scratch/$USER}/omat24/raw
URL=https://dl.fbaipublicfiles.com/opencatalystproject/data/omat
SUBSETS="rattled-1000 rattled-1000-subsampled rattled-500 rattled-500-subsampled
         rattled-300 rattled-300-subsampled aimd-from-PBE-1000-npt aimd-from-PBE-1000-nvt
         aimd-from-PBE-3000-npt aimd-from-PBE-3000-nvt rattled-relax"

for split_rel in train:241018 val:241220; do
  split=${split_rel%%:*} rel=${split_rel#*:}
  mkdir -p "$ROOT/$split/.downloads"
  for s in $SUBSETS; do
    [ -e "$ROOT/$split/$s/.done" ] && { echo "skip $split/$s"; continue; }
    tgz=$ROOT/$split/.downloads/$s.tar.gz
    wget -c -q --show-progress -O "$tgz" "$URL/$rel/omat/$split/$s.tar.gz"
    tar -xzf "$tgz" -C "$ROOT/$split"
    touch "$ROOT/$split/$s/.done" && rm -f "$tgz"
  done
done
echo "done: $ROOT"
