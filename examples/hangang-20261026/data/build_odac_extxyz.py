#!/usr/bin/env python3
"""ODAC25 (periodic MOF +/- adsorbate, fairchem .aselmdb) -> extxyz shards for BAM-torch (mp_v1).

Same shard scheme as build_mol_extxyz.py (omol25), but periodic: real cell, pbc="T T T",
positions untouched. Total energy (raw eV) and forces in a SinglePointCalculator; ODAC25 has
no stress, so none is written. info["subset"] = gcmc / mof / mof_plus_adsorbate.

  python build_odac_extxyz.py SRC_DIR OUT_DIR --name odac25 [--per-shard 10000] [-w N] [--split RANK/N]

SRC_DIR is searched recursively. Shards are named {name}_{subset}_{part}_{k:03d}.extxyz,
written atomically and skipped on rerun.
"""
import argparse, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def to_atoms(d, subset):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    pbc = np.asarray(d["pbc"], dtype=bool)
    if not pbc.all():
        raise ValueError(f"expected full pbc, got {pbc}")
    atoms = Atoms(numbers=d["numbers"], positions=np.asarray(d["positions"], dtype=np.float64),
                  cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=float(d["energy"]),
                                       forces=np.asarray(d["forces"], dtype=np.float64))
    atoms.info["subset"] = subset
    return atoms


def live_keys(path):
    """Live row keys of an aselmdb in cursor order."""
    import lmdb
    from ase.io.jsonio import decode
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        raw = txn.get(b"deleted_ids")
        deleted = set(decode(zlib.decompress(raw).decode())) if raw else set()
        keys = [k for k in txn.cursor().iternext(values=False) if k.isdigit() and int(k) not in deleted]
    env.close()
    return keys


def shard_task(args):
    import lmdb
    from ase.io import write
    from ase.io.jsonio import decode
    src, out, name, k, keys = args
    subset = Path(src).parent.name
    p = out / f"{name}_{subset}_{Path(src).stem}_{k:03d}.extxyz"
    if p.exists():  # written atomically, so an existing shard is complete
        return 0
    env = lmdb.open(str(src), subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        frames = [to_atoms(decode(zlib.decompress(txn.get(key)).decode()), subset) for key in keys]
    env.close()
    write(str(p) + ".tmp", frames, format="extxyz")
    os.replace(str(p) + ".tmp", p)
    return len(frames)


def selftest():
    from ase.build import bulk
    from ase.io import read, write
    import tempfile
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    d = {"positions": a.positions, "numbers": a.numbers, "cell": a.cell[:], "pbc": [True] * 3,
         "energy": -12.5, "forces": np.ones((4, 3))}
    at = to_atoms(d, "mof")
    with tempfile.TemporaryDirectory() as t:
        write(f"{t}/x.extxyz", [at, at], format="extxyz")
        b = read(f"{t}/x.extxyz", ":")
    assert len(b) == 2 and b[1].pbc.all() and np.allclose(b[1].cell, a.cell)
    assert b[1].get_potential_energy() == -12.5 and np.allclose(b[1].get_forces(), 1)
    assert b[1].info["subset"] == "mof"
    d["pbc"] = [True, True, False]
    try:
        to_atoms(d, "mof")
        raise AssertionError("partial pbc accepted")
    except ValueError:
        pass
    print("selftest OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest(); sys.exit()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--name", required=True)
    ap.add_argument("--per-shard", type=int, default=10000)
    ap.add_argument("-w", "--workers", type=int, default=64)
    ap.add_argument("--split", default="0/1", metavar="RANK/N", help="multi-node: take every N-th shard")
    a = ap.parse_args()
    rank, nsplit = map(int, a.split.split("/"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.src).rglob("*.aselmdb"))
    tasks = [(f, out, a.name, k, keys[k * a.per_shard:(k + 1) * a.per_shard])
             for f in files for keys in [live_keys(f)] for k in range((len(keys) + a.per_shard - 1) // a.per_shard)]
    total = sum(len(t[4]) for t in tasks)
    mine = tasks[rank::nsplit]
    print(f"extxyz {a.name} split {rank}/{nsplit}: {len(files)} files, {total} frames, "
          f"{len(mine)} of {len(tasks)} shards, {a.workers} workers", flush=True)
    done = written = 0
    t = time.time()
    with Pool(a.workers) as pool:
        for m in pool.imap_unordered(shard_task, mine):
            done += 1; written += m
            if done % 100 == 0 or done == len(mine):
                print(f"[{done}/{len(mine)}] frames written {written} {time.time() - t:.0f}s", flush=True)
    print(f"SPLIT OK {rank}/{nsplit}: shards={len(mine)} frames_written={written} total_frames={total}", flush=True)
