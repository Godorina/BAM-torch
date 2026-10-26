#!/usr/bin/env python3
"""Build $DATA_ROOT/omol25_opoly26/{train,val} (DATA_ROOT defaults to /scratch/$USER) as interleaved symlinks (no data copied).

train: every omol25 train shard once + every opoly26 train shard OPOLY_REPEAT times, ordered so
       the two sets are spread evenly (each item keyed by its fractional position in its own list).
val:   all opoly26 val shards + every 4th omol25 val shard.
Links are <index>_<shard>, index zero-padded so name order == intended order.
"""
import os, sys
ROOT = os.environ.get("DATA_ROOT", f"/scratch/{os.environ['USER']}")
OMOL, OPOLY = f"{ROOT}/omol25/omol25_extxyz", f"{ROOT}/opoly26/opoly26_extxyz"
OUT = f"{ROOT}/omol25_opoly26"
OPOLY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def shards(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".extxyz"))


def link_all(name, items):
    d = os.path.join(OUT, name)
    if os.path.exists(d):
        sys.exit(f"{d} exists; remove it to rebuild")
    os.makedirs(d)
    w = len(str(len(items) - 1))
    for i, src in enumerate(items):
        os.symlink(src, os.path.join(d, f"{i:0{w}d}_{os.path.basename(src)}"))
    return len(items)


om = [os.path.join(OMOL, "train", f) for f in shards(OMOL + "/train")]
po = [os.path.join(OPOLY, "train", f) for f in shards(OPOLY + "/train")] * OPOLY_REPEAT
keyed = [((i + 0.5) / len(om), 0, s) for i, s in enumerate(om)] + [((i + 0.5) / len(po), 1, s) for i, s in enumerate(po)]
train = [s for _, _, s in sorted(keyed)]
val = [os.path.join(OPOLY, "val", f) for f in shards(OPOLY + "/val")] + \
      [os.path.join(OMOL, "val", f) for f in shards(OMOL + "/val")][::4]
print("train", link_all("train", train), "links (omol", len(om), "+ opoly", len(po), f"= x{OPOLY_REPEAT})")
print("val", link_all("val", val), "links")
