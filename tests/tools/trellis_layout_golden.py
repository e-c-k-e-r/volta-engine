#!/usr/bin/env python3
"""Golden files for test_trellis_layout: random trellis words and the row-major FP16 weights they decode to, in the stock or the Volta
tile order (needs the exllamav3 venv). Writes <out>/trellis.bin (uint16 [tk, tn, 16 K]) and <out>/expect.bin (fp16 [16 tk, 16 tn]).
  trellis_layout_golden.py <out_dir> --layout stock|volta [--bits 4] [--tk 6] [--tn 16] [--mul1]
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("EXLLAMAV3_DIR", os.path.expanduser("~/Programs/exllamav3")))
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import tensor_core_perm
import exl3_volta_layout as V

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--layout", default="stock", choices=["stock", "volta"])
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--tk", type=int, default=6)
ap.add_argument("--tn", type=int, default=16)
ap.add_argument("--mcg", action="store_true")
a = ap.parse_args()
dev = torch.device("cuda:0")
K, tk, tn = a.bits, a.tk, a.tn
g = torch.Generator(device="cpu").manual_seed(1234)
words = torch.randint(-32768, 32768, (tk, tn, 16 * K), dtype=torch.int16, generator=g)
unp = torch.zeros((tk, tn, 256), dtype=torch.int16, device=dev)
ext.unpack_trellis(unp, words.to(dev).contiguous(), K)
tiles = torch.zeros((tk * tn, 256), dtype=torch.float, device=dev)
ext.decode(unp.reshape(-1, 256).contiguous(), tiles, a.mcg, not a.mcg)
perm = tensor_core_perm(dev).long() if a.layout == "stock" else V.volta_perm(dev)   # perm[pos] = row-major element at stream position pos
rm = torch.empty_like(tiles)
rm[:, perm] = tiles
rm = rm.half().reshape(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(16 * tk, 16 * tn)
os.makedirs(a.out, exist_ok=True)
words.numpy().view(np.uint16).tofile(os.path.join(a.out, "trellis.bin"))
rm.cpu().numpy().tofile(os.path.join(a.out, "expect.bin"))
print(f"wrote {a.out}: layout {a.layout}, K {K}, tiles {tk} x {tn}, {'mcg' if a.mcg else 'mul1'}")
