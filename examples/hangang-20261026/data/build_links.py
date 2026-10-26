#!/usr/bin/env python3
"""Build $DATA_ROOT/omat24/{train_mix,val_mix} (DATA_ROOT defaults to /scratch/$USER): flat symlinks over the 11 omat24 subsets (no data copied).

train_mix: every train shard once, the 11 subsets interleaved evenly (each shard keyed by its
           fractional position in its own subset), named <index>_<subset>__<shard> so name order == order.
val_mix:   every subset's val data.extxyz as <subset>__data.extxyz.
"""
import os, re, sys
ROOT = os.environ.get("DATA_ROOT", f"/scratch/{os.environ['USER']}") + "/omat24"


def shards(d):
    num = lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)]
    return sorted((f for f in os.listdir(d) if f.endswith(".extxyz")), key=num)


for name in ("train_mix", "val_mix"):
    if os.path.exists(os.path.join(ROOT, name)):
        sys.exit(f"{ROOT}/{name} exists; remove it to rebuild")

subsets = sorted(os.listdir(f"{ROOT}/train"))
keyed = []
for k, sub in enumerate(subsets):
    fs = shards(f"{ROOT}/train/{sub}")
    keyed += [((i + 0.5) / len(fs), k, sub, f) for i, f in enumerate(fs)]
keyed.sort()
os.makedirs(f"{ROOT}/train_mix")
w = len(str(len(keyed) - 1))
for i, (_, _, sub, f) in enumerate(keyed):
    os.symlink(f"{ROOT}/train/{sub}/{f}", f"{ROOT}/train_mix/{i:0{w}d}_{sub}__{f}")

os.makedirs(f"{ROOT}/val_mix")
nval = 0
for sub in sorted(os.listdir(f"{ROOT}/val")):
    for f in shards(f"{ROOT}/val/{sub}"):
        os.symlink(f"{ROOT}/val/{sub}/{f}", f"{ROOT}/val_mix/{sub}__{f}")
        nval += 1
print("train_mix", len(keyed), "links from", len(subsets), "subsets; val_mix", nval, "links")
