#!/usr/bin/env python3
"""Multi-node E0 fit for ODAC25: build_odac_pkl.py fit-e0, with the id-range tasks split across nodes.

  python fit_e0_split.py part OUT_DIR TRAIN_DIR --split RANK/N [-w N]   # each node: OUT_DIR/R_RANK.npy
  python fit_e0_split.py solve OUT_DIR E0.txt                           # combine R factors -> E0 file
"""
import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_odac_pkl import tasks_of, fit_task, solve_e0, write_e0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["part", "solve"]); ap.add_argument("out"); ap.add_argument("arg")
    ap.add_argument("--split", default="0/1"); ap.add_argument("-w", "--workers", type=int, default=240)
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "part":
        rank, n = map(int, a.split.split("/"))
        _, tasks, expect = tasks_of([a.arg])
        mine = tasks[rank::n]
        Rs, total, t0 = [], 0, time.time()
        with Pool(a.workers) as pool:
            for _, m, R in pool.imap_unordered(fit_task, mine):
                Rs.append(R); total += m
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"R_{rank}.npy", np.linalg.qr(np.vstack(Rs), mode="r"))
        np.save(out / f"n_{rank}.npy", np.array([total, expect]))
        print(f"e0 part {rank}/{n}: {len(mine)} of {len(tasks)} tasks, rows={total} {time.time() - t0:.0f}s", flush=True)
    else:
        parts = sorted(out.glob("n_*.npy"))
        counts = np.array([np.load(p) for p in parts])
        total, expect = int(counts[:, 0].sum()), int(counts[0, 1])
        if total != expect:
            sys.exit(f"E0 fit row count mismatch: {total} != {expect} ({len(parts)} parts)")
        e0, present, rss = solve_e0([np.load(p) for p in sorted(out.glob("R_*.npy"))])
        write_e0(a.arg, e0)
        print(f"E0 written to {a.arg}: frames={total} elements={int(present.sum())} "
              f"rms_residual={np.sqrt(rss / total):.4f} eV/frame", flush=True)
