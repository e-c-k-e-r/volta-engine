#!/usr/bin/env python3
"""Relative error of EXL3 tensors against the BF16 source, decoded in plain torch (stock or Volta tile order; integer rates; needs the exllamav3 venv).

  exl3_dequant_check.py <exl3 dir> <bf16 dir> [--order stock|volta] [--only SUBSTRING] [--limit N] [--device cuda:0]
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors import safe_open
import exl3_quantize as eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exl3"); ap.add_argument("src")
    ap.add_argument("--order", default="stock", choices=["stock", "volta"])
    ap.add_argument("--only", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    qz = eq.load_upstream(a.order)
    ext = qz.ext
    dev = torch.device(a.device)
    from exllamav3.util.hadamard import get_hadamard_dt
    import math
    srcs = {}
    for fn in glob.glob(a.src + "/*.safetensors"):
        with safe_open(fn, "pt") as f:
            for k in f.keys(): srcs[k] = fn
    n = 0
    for fn in sorted(glob.glob(a.exl3 + "/*.safetensors")):
        with safe_open(fn, "pt") as f:
            keys = list(f.keys())
            for name in keys:
                if not name.endswith(".trellis") or a.only not in name: continue
                base = name[:-8]
                if base + ".weight" not in srcs: continue
                tr = f.get_tensor(name).to(dev)
                suh, svh = f.get_tensor(base + ".suh").to(dev).float(), f.get_tensor(base + ".svh").to(dev).float()
                mcg, mul1 = (base + ".mcg") in keys, (base + ".mul1") in keys
                tk, tn, w = tr.shape
                K = w // 16
                unp = torch.zeros((tk, tn, 256), dtype=torch.int16, device=dev)
                ext.unpack_trellis(unp, tr.contiguous(), K)
                flat = unp.reshape(-1, 256).contiguous()
                tiles = torch.zeros((flat.shape[0], 256), dtype=torch.float, device=dev)
                for c in range(0, flat.shape[0], 32768):
                    ext.decode(flat[c:c + 32768].contiguous(), tiles[c:c + 32768], mcg, mul1 and not mcg)
                tiles = tiles[:, qz.tensor_core_perm_i(dev).long()]
                wq = tiles.reshape(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(tk * 16, tn * 16)
                had = get_hadamard_dt(128, dev, torch.float, 1 / math.sqrt(128))
                wq = (had @ wq.view(-1, 128, wq.shape[1])).view(wq.shape)        # had_l
                wq = wq * suh.unsqueeze(1)
                wq = (wq.view(wq.shape[0], -1, 128) @ had).view(wq.shape)        # had_r
                wq = wq * svh.unsqueeze(0)
                with safe_open(srcs[base + ".weight"], "pt") as g:
                    ref = g.get_tensor(base + ".weight").to(dev).float().t()
                rel = ((wq - ref).square().sum() / ref.square().sum()).sqrt().item()
                print(f"{base:50s} K={K} relative error {rel:.5f}", flush=True)
                n += 1
                if a.limit and n >= a.limit: return


if __name__ == "__main__":
    main()
