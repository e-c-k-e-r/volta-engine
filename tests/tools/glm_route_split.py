#!/usr/bin/env python3
"""Splits a `route_dump` file (GLM) into one raw FP16 file per layer, route_L<layer>.f16 = the rows that feed the experts of that layer ([rows, hidden]), so that a tool can
memory-map the rows of one layer instead of reading the whole dump. One pass over the dump, bounded memory.

  glm_route_split.py DUMP OUT_DIR --hidden 4096
"""
import argparse, os, struct, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("out"); ap.add_argument("--hidden", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files, rows = {}, {}
    with open(a.dump, "rb") as f:
        while True:
            h = f.read(16)
            if len(h) < 16: break
            layer, m, first, K = struct.unpack("<4i", h)
            f.seek(4 * m * K, 1)
            x = f.read(2 * m * a.hidden)
            if len(x) < 2 * m * a.hidden: break
            if layer not in files: files[layer] = open(os.path.join(a.out, f"route_L{layer:02d}.f16"), "wb")
            files[layer].write(x)
            rows[layer] = rows.get(layer, 0) + m
    for fh in files.values(): fh.close()
    print(f"{len(files)} layers, rows per layer {min(rows.values()) if rows else 0} to {max(rows.values()) if rows else 0} -> {a.out}")


if __name__ == "__main__":
    main()
