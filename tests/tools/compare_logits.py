#!/usr/bin/env python3
"""Compare two logit dumps of diag_logits_dump: cosine, relative RMS difference and the overlap of the top 1 / 10 / 100 tokens.
  compare_logits.py a.bin b.bin"""
import sys
import numpy as np

a, b = (np.fromfile(p, dtype=np.float16).astype(np.float64) for p in sys.argv[1:3])
assert a.shape == b.shape
print(f"cosine {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):.5f}   relative rms difference {np.linalg.norm(a - b) / np.linalg.norm(a):.4f}")
for k in (1, 10, 100):
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    print(f"top-{k} overlap {len(ta & tb)}/{k}")
