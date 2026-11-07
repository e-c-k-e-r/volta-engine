#!/usr/bin/env python3
"""Fixtures for test_gguf_linear: per GGUF block type, a random weight quantized by ggml itself, gguf-py's dequantization of those blocks,
random FP16 activations and two references for x @ W^T: exact, and with the activations rounded through q8_1 the way the kernels do.

  gguf_fixtures.py --libggml <dir with libggml-base.so> --gguf-py <llama.cpp/gguf-py> --out DIR [--n 384] [--k 2048] [--m 20]

File per type: <name>.bin = int32 type, n, k, m, int64 block bytes; blocks; m*k FP16 x; m*n FP32 exact; m*n FP32 q8_1 reference.
The IQ types that need an importance matrix get a uniform one.
"""
import argparse, ctypes, os, struct, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--libggml", required=True); ap.add_argument("--gguf-py", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=384); ap.add_argument("--k", type=int, default=2048); ap.add_argument("--m", type=int, default=20)
a = ap.parse_args()
sys.path.insert(0, a.gguf_py)
from gguf import quants, GGMLQuantizationType as T

lib = ctypes.CDLL(os.path.join(a.libggml, "libggml-base.so"))
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_quantize_init.argtypes = [ctypes.c_int]
lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
lib.ggml_row_size.restype = ctypes.c_size_t
lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]

TYPES = [T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0, T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K, T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS,
         T.IQ3_S, T.IQ1_S, T.IQ1_M, T.IQ4_NL, T.IQ4_XS, T.Q2_0]

def q8_1(x):
    """llama.cpp's quantize_q8_1 on rows of x: per 32 values d = amax / 127, q = round(x / d)"""
    b = x.reshape(x.shape[0], -1, 32).astype(np.float32)
    d = np.abs(b).max(axis=2, keepdims=True) / 127.0
    q = np.where(d > 0, np.round(b / np.where(d > 0, d, 1)), 0)
    d16 = d.astype(np.float16).astype(np.float32)   # the scale is stored as half
    return (q * d16).reshape(x.shape)

rng = np.random.default_rng(7)
os.makedirs(a.out, exist_ok=True)
n, k, m = a.n, a.k, a.m
for t in TYPES:
    w = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    w[:, :8] *= 4.0   # a few larger columns, as real weights have
    lib.ggml_quantize_init(int(t))
    row = lib.ggml_row_size(int(t), k)
    dst = np.zeros(n * row, dtype=np.uint8)
    imat = np.ones(k, dtype=np.float32) if lib.ggml_quantize_requires_imatrix(int(t)) else None
    got = lib.ggml_quantize_chunk(int(t), w.ctypes.data, dst.ctypes.data, 0, n, k, imat.ctypes.data if imat is not None else None)
    assert got == n * row, (t.name, got, n * row)
    wd = quants.dequantize(dst.reshape(n, row), t).astype(np.float64)   # [n, k]
    x = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)
    exact = (x.astype(np.float64) @ wd.T).astype(np.float32)
    ref_q = (q8_1(x.astype(np.float32)).astype(np.float64) @ wd.T).astype(np.float32)
    with open(os.path.join(a.out, t.name + ".bin"), "wb") as f:
        f.write(struct.pack("<4iq", int(t), n, k, m, dst.nbytes))
        f.write(dst.tobytes()); f.write(x.tobytes()); f.write(exact.tobytes()); f.write(ref_q.tobytes())
    print(f"{t.name:8s} {row * 8 / k:5.2f} bits per weight, dequant rms error {np.sqrt(((wd - w) ** 2).mean()) / np.sqrt((w ** 2).mean()):.4f}")
