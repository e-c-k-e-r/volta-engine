"""Compares a test_dsv41 trace (trace_layers=<file>) with dsv41_ref.py's dumps: per layer, the hyper-connection streams after the FFN
(tag = layer) against h_LNN.npy, over the positions both cover.
  python tests/tools/dsv41/trace_cmp.py <trace file> <dsv41_ref.py --out directory>"""
import struct
import sys

import numpy as np

H = 5120


def main():
    trace, ref = sys.argv[1], sys.argv[2]
    b = open(trace, "rb").read()
    o, recs = 0, {}
    while o < len(b):
        tag, m, p0 = struct.unpack_from("iii", b, o)
        o += 12
        a = np.frombuffer(b, dtype=np.float32, count=m * 4 * H, offset=o).reshape(m, 4, H)
        o += 4 * m * 4 * H
        recs.setdefault(tag, []).append((p0, a))
    for layer in range(40):
        if layer not in recs:
            continue
        a = np.concatenate([x for _, x in sorted(recs[layer], key=lambda t: t[0])])   # the chunks of a prefill in order
        try:
            r = np.load(f"{ref}/h_L{layer:02d}.npy").astype(np.float32)
        except FileNotFoundError:
            break
        n = min(len(a), len(r))
        d = a[:n] - r[:n]
        rel = np.linalg.norm(d) / np.linalg.norm(r[:n])
        per_pos = np.linalg.norm(d, axis=(1, 2)) / np.linalg.norm(r[:n], axis=(1, 2))
        print(f"L{layer:02d}  rel {rel:.3e}  max|d| {np.abs(d).max():.4g}  |ref|max {np.abs(r[:n]).max():.4g}  worst pos {per_pos.argmax()} ({per_pos.max():.3e})")


if __name__ == "__main__":
    main()
