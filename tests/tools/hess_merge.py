#!/usr/bin/env python3
"""Merges Hessian capture directories (ENGINE_HESSIAN_DIR files) with weights: H = sum_i w_i * H_i / rows_i * rows_total, rows_total = sum_i rows_i.

  hess_merge.py --out DIR --src DIR_A:0.7 --src DIR_B:0.3

Each source holds a sum of x^T x over its rows; the weights make the mixture, not the row counts, decide how much a source counts. Files missing in a source are skipped for it.
"""
import argparse, os
import numpy as np


def read(path):
    b = open(path, "rb").read()
    k = int(np.frombuffer(b[4:8], dtype=np.int32)[0])
    rows = int(np.frombuffer(b[8:16], dtype=np.int64)[0])
    return k, rows, np.frombuffer(b[16:], dtype=np.float32).reshape(k, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True, help="DIR:WEIGHT")
    a = ap.parse_args()
    srcs = [(s.rsplit(":", 1)[0], float(s.rsplit(":", 1)[1])) for s in a.src]
    wsum = sum(w for _, w in srcs)
    os.makedirs(a.out, exist_ok=True)
    names = sorted({f for d, _ in srcs for f in os.listdir(d) if f.endswith(".f32")})
    for n in names:
        srcp = [os.path.join(d, n) for d, _ in srcs if os.path.exists(os.path.join(d, n))]
        outp = os.path.join(a.out, n)
        if os.path.exists(outp) and os.path.getmtime(outp) > max(os.path.getmtime(p) for p in srcp):
            continue    # merged before, sources unchanged
        acc, total_rows, k0 = None, 0, None
        parts = []
        for d, w in srcs:
            p = os.path.join(d, n)
            if os.path.exists(p):
                k, rows, H = read(p)
                parts.append((w / wsum, rows, H, k))
        total_rows = sum(r for _, r, _, _ in parts)
        acc = np.zeros((parts[0][3], parts[0][3]), dtype=np.float64)
        wsum_here = sum(w for w, _, _, _ in parts)
        for w, rows, H, k in parts:
            assert k == parts[0][3], n
            acc += (w / wsum_here) * H.astype(np.float64) / rows
        acc *= total_rows
        k = parts[0][3]
        with open(os.path.join(a.out, n), "wb") as f:
            f.write(np.array([0x48455353, k], dtype=np.int32).tobytes())
            f.write(np.array([total_rows], dtype=np.int64).tobytes())
            f.write(acc.astype(np.float32).tobytes())
    print(f"{len(names)} files -> {a.out}")


if __name__ == "__main__":
    main()
