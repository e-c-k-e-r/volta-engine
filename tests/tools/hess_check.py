#!/usr/bin/env python3
"""Checks the files of a Hessian capture directory: symmetric, positive diagonal, one row count per directory.   hess_check.py DIR [DIR...]"""
import glob, os, sys
import numpy as np

bad = 0
for d in sys.argv[1:]:
    rows_seen = set()
    files = sorted(glob.glob(os.path.join(d, "*.f32")))
    for f in files:
        b = open(f, "rb").read()
        k = int(np.frombuffer(b[4:8], dtype=np.int32)[0])
        if len(b) != 16 + 4 * k * k:       # a write that ran out of disk
            bad += 1
            print(f"BAD {f}: {len(b)} bytes, expected {16 + 4 * k * k}")
            continue
        rows = int(np.frombuffer(b[8:16], dtype=np.int64)[0])
        H = np.frombuffer(b[16:], dtype=np.float32).reshape(k, k)
        dg = np.diag(H)
        sym = float(np.abs(H - H.T).max() / max(np.abs(H).max(), 1e-30))
        rows_seen.add(rows)
        if (dg < 0).any() or sym > 1e-4 or not np.isfinite(H).all():
            bad += 1
            print(f"BAD {f}: neg diag {int((dg < 0).sum())}, asymmetry {sym:.1e}, finite {bool(np.isfinite(H).all())}")
    print(f"{d}: {len(files)} files, row counts {sorted(rows_seen)}")
sys.exit(1 if bad else 0)
