#!/usr/bin/env python
"""OMAT24 aselmdb -> extxyz converter.

Dumps each frame verbatim (arrays + info + calc results) so the output matches
the pre-existing omat24_train_*.extxyz.gz schema. Per-frame Properties lines are
produced by ASE, so subdatasets that carry initial_magmoms keep that column and
those that do not simply omit it.

Shard unit = one input .aselmdb -> one output .extxyz, with a .done marker so an
interrupted run resumes without redoing finished shards.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool

_ROOT = os.environ.get("DATA_ROOT", f"/scratch/{os.environ.get('USER', '')}") + "/omat24"
BASE = _ROOT + "/raw"  # download_omat24.sh: raw/{train,val}/<subset>/*.aselmdb
OUT = _ROOT            # -> {train,val}/<subset>/*.extxyz, what build_links.py reads
NAMES = [
    "rattled-1000", "rattled-1000-subsampled",
    "rattled-500", "rattled-500-subsampled",
    "rattled-300", "rattled-300-subsampled",
    "aimd-from-PBE-1000-npt", "aimd-from-PBE-1000-nvt",
    "aimd-from-PBE-3000-npt", "aimd-from-PBE-3000-nvt",
    "rattled-relax",
]


def shard_tasks(base, out, splits, subs):
    tasks = []
    for split in splits:
        for sub in subs:
            src_dir = os.path.join(base, split, sub)
            if not os.path.isdir(src_dir):
                continue
            for fname in sorted(os.listdir(src_dir)):
                if not fname.endswith(".aselmdb"):
                    continue
                stem = fname[: -len(".aselmdb")]
                tasks.append((
                    os.path.join(src_dir, fname),
                    os.path.join(out, split, sub, stem + ".extxyz"),
                ))
    return tasks


def convert_shard(task):
    src, dst = task
    done = dst + ".done"
    if os.path.exists(done):
        with open(done) as fh:
            return (src, int(fh.read().strip() or 0), "skip", 0.0)
    try:
        from ase.io import write
        from fairchem.core.datasets import AseDBDataset

        t0 = time.time()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        ds = AseDBDataset({"src": src})
        n = len(ds)
        with open(tmp, "w") as fh:
            for i in range(n):
                write(fh, ds.get_atoms(i), format="extxyz")
        os.replace(tmp, dst)
        with open(done, "w") as fh:
            fh.write(str(n))
        return (src, n, "ok", time.time() - t0)
    except Exception:
        return (src, 0, "FAIL:" + traceback.format_exc(limit=3).replace("\n", " | "), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    ap.add_argument("--subs", nargs="+", default=NAMES)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--limit", type=int, default=0, help="convert only first N shards")
    args = ap.parse_args()

    tasks = shard_tasks(args.base, args.out, args.splits, args.subs)
    if args.limit:
        tasks = tasks[: args.limit]
    print("shards: %d | workers: %d" % (len(tasks), args.workers), flush=True)

    t0 = time.time()
    frames = 0
    failed = 0
    with Pool(args.workers) as pool:
        for k, (src, n, status, dt) in enumerate(
            pool.imap_unordered(convert_shard, tasks, chunksize=1), 1
        ):
            if status.startswith("FAIL"):
                failed += 1
                print("FAIL %s :: %s" % (src, status), flush=True)
            else:
                frames += n
            if k % 25 == 0 or k == len(tasks):
                el = time.time() - t0
                print(
                    "progress %d/%d frames=%d failed=%d elapsed=%.0fs rate=%.0f f/s"
                    % (k, len(tasks), frames, failed, el, frames / max(el, 1e-9)),
                    flush=True,
                )
    print("DONE shards=%d frames=%d failed=%d elapsed=%.0fs"
          % (len(tasks), frames, failed, time.time() - t0), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
