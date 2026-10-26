#!/usr/bin/env python
"""Least-squares per-element reference energies (E0s) from extxyz data.

Same contract as bam_torch.utils.get_enr_avg_per_element with element="auto":
element = sorted unique Z in the data, uniq_element = {Z: index}, E0s by index.
Streams the data in parallel instead of loading the whole trajectory into memory.
Every file is cut into byte ranges, so many small files and one huge file both
spread across all workers.

usage: fit_element_refs2.py DATA_DIR_OR_FILE [-o OUT.txt] [-w WORKERS]
"""
import argparse, os, re, time
from glob import glob
from multiprocessing import Pool

import numpy as np
from ase.data import atomic_numbers, chemical_symbols

NZ = len(chemical_symbols)  # 0..118
ENERGY_RE = re.compile(rb"(?:^|\s)energy=(\S+)")


def accumulate(task):
    """Fit contributions from frames whose header line starts in [start, end)."""
    path, start, end = task
    A = np.zeros((NZ, NZ)); b = np.zeros(NZ); counts = np.zeros(NZ)
    nf = sk = na = 0; te = 0.0
    with open(path, "rb") as fh:
        if start:  # land mid-frame: drop the partial line, then skip to the next header
            fh.seek(start - 1); fh.readline()
            while fh.tell() < end:
                pos = fh.tell(); line = fh.readline()
                try:
                    int(line); fh.seek(pos); break
                except ValueError:
                    pass
        while fh.tell() < end:
            head = fh.readline()
            if not head:
                break
            if not head.strip():
                continue
            try:
                nat = int(head)
            except ValueError:
                sk += 1; continue
            m = ENERGY_RE.search(fh.readline())
            counts[:] = 0.0
            bad = False
            for _ in range(nat):
                line = fh.readline()
                z = atomic_numbers.get(line.split(None, 1)[0].decode()) if line.strip() else None
                if z is None:
                    bad = True; break
                counts[z] += 1.0
            if bad or m is None:
                sk += 1; continue
            e = float(m.group(1))
            A += np.outer(counts, counts); b += counts * e
            nf += 1; na += nat; te += e
    return A, b, nf, sk, na, te


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", help="folder (recursive *.xyz/*.extxyz) or a single file")
    p.add_argument("-o", "--out", default="get_enr_avg_per_element.txt")
    p.add_argument("-w", "--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    files = [a.data_dir] if os.path.isfile(a.data_dir) else sorted(
        glob(os.path.join(a.data_dir, "**", "*.xyz"), recursive=True) +
        glob(os.path.join(a.data_dir, "**", "*.extxyz"), recursive=True))
    sizes = [os.path.getsize(f) for f in files]
    chunk = max(1 << 25, sum(sizes) // (4 * a.workers))  # >= 32 MB, ~4 chunks per worker
    tasks = [(f, s, min(s + chunk, n)) for f, n in zip(files, sizes) for s in range(0, n, chunk)]
    print(f"files: {len(files)} | chunks: {len(tasks)} ({chunk >> 20} MB) | workers: {a.workers}", flush=True)

    A = np.zeros((NZ, NZ)); b = np.zeros(NZ); tf = ts = ta = 0; te = 0.0
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (a_, b_, nf, sk, na, e) in enumerate(pool.imap_unordered(accumulate, tasks), 1):
            A += a_; b += b_; tf += nf; ts += sk; ta += na; te += e
            if k % 500 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"progress {k}/{len(tasks)} frames={tf} skipped={ts} "
                      f"elapsed={el:.0f}s rate={tf/max(el,1e-9):.0f} f/s", flush=True)

    element = np.flatnonzero(np.diag(A))
    uniq_element = {int(z): i for i, z in enumerate(element)}
    Ae, be = A[np.ix_(element, element)], b[element]
    # Least squares around the mean energy per atom, like BAM-torch's BFGS from that start.
    # When composition never varies (rank < n) this picks the same non-unique solution.
    m0 = te / ta
    d, _, rank, _ = np.linalg.lstsq(Ae, be - Ae.sum(1) * m0, rcond=None)
    e0 = m0 + d
    print(f"\nframes used={tf}  skipped={ts}", flush=True)
    print(f"element ({len(element)}): {[chemical_symbols[z] for z in element]}", flush=True)
    print(f"rank: {rank}/{len(element)}" + ("  (composition never varies; E0s not unique, "
          "BAM-torch convention used)" if rank < len(element) else f"  condition number: {np.linalg.cond(Ae):.3e}"), flush=True)
    for z, i in uniq_element.items():
        print(f"  {chemical_symbols[z]:<3} {z:3d} {e0[i]:14.6f}")

    with open(a.out, "w") as f:
        f.write("({" + ",\n    ".join(f"{i}: {float(e0[i])!r}" for i in range(len(element))) + "},\n")
        f.write(" {" + ",\n    ".join(f"{z}: {i}" for z, i in uniq_element.items()) + "})\n")
    print(f"\nwrote {a.out}  elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
