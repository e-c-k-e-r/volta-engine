#!/usr/bin/env python3
"""Re-encode an EXL3 checkpoint with the Volta tile order (proof of concept; needs the exllamav3 venv).

The stock order inside every 16x16 weight tile suits Ampere's m16n8k16 fragments: lane t decodes elements t*8..t*8+7 and holds an
Ampere B fragment. The Volta order puts into lane t's eight decoded values exactly what its m8n8k4 operands need, so the kernels
need no shuffles (built with EXL3_VOLTA_LAYOUT=1, see docs/HARDWARE_SM70.md).

  volta order, lane = 4*q + t_ + 16*hh  (q = octet 0..3, t_ = lane & 3, hh = lane >> 4)
    nf  = q & 1            the n8 half of the 16x16 tile the octet works on
    cs  = q >> 1           which pair of k4 chunks
    col = 8*nf + t_ + 4*hh
    value 4*s + i of the lane (s = 0, 1; i = 0..3) is element (k = 4*(2*cs + s) + i, n = col)

Re-encoding already-quantized weights adds about one quantization's worth of error (the stored values are not jointly
representable in another order); the order itself does not change the error. A real conversion from BF16 avoids that.

  exl3_volta_layout.py <in_dir> <out_dir> [--device cuda:0] [--files SHARD,SHARD] [--only SUBSTRING]
"""
import argparse, json, os, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.environ.get("EXLLAMAV3_DIR", os.path.expanduser("~/Programs/exllamav3")))
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles, pack_trellis, tensor_core_perm_i


def volta_perm(device):
    """perm[t] = row-major tile element (k * 16 + n) stored at stream position t."""
    perm = [0] * 256
    for lane in range(32):
        q, t_, hh = (lane >> 2) & 3, lane & 3, lane >> 4
        nf, cs = q & 1, q >> 1
        col = 8 * nf + t_ + 4 * hh
        for s in range(2):
            for i in range(4):
                k = 4 * (2 * cs + s) + i
                perm[lane * 8 + 4 * s + i] = k * 16 + col
    assert sorted(perm) == list(range(256))
    return torch.tensor(perm, dtype=torch.long, device=device)


def transcode(trellis, K, mul1, mcg, device, chunk_tiles=262144):
    """trellis: int16 [tiles_k, tiles_n, 16*K] in the stock order -> same shape in the Volta order."""
    tk, tn, w = trellis.shape
    assert w == 16 * K
    args = {"K": K}
    if mcg: args["mcg"] = True
    if mul1: args["mul1"] = True
    q_ord = volta_perm(device)
    p_inv = tensor_core_perm_i(device).long()
    out = torch.empty_like(trellis)
    rows = max(1, chunk_tiles // tn)
    sq_err = sq_ref = 0.0
    for a in range(0, tk, rows):
        b = min(tk, a + rows)
        tr = trellis[a:b].to(device).contiguous()
        unp = torch.zeros((b - a, tn, 256), dtype=torch.int16, device=device)
        ext.unpack_trellis(unp, tr, K)
        flat = unp.reshape(-1, 256).contiguous()
        tiles = torch.zeros((flat.shape[0], 256), dtype=torch.float, device=device)
        for c in range(0, flat.shape[0], 32768):          # the decode grid's y dimension is limited to 65535
            ext.decode(flat[c:c + 32768].contiguous(), tiles[c:c + 32768], mcg, mul1 and not mcg)
        new_seq = tiles[:, p_inv[q_ord]].contiguous()       # position t of the new stream holds element q_ord[t]
        qv, qi = quantize_tiles(new_seq, args)
        sq_err += (qv - new_seq).square().sum().item()
        sq_ref += new_seq.square().sum().item()
        out[a:b] = pack_trellis(qi.reshape(b - a, tn, 256), args).cpu()
    return out, (sq_err / max(sq_ref, 1e-30)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="re-encode only tensors whose name contains this (others are copied as they are)")
    ap.add_argument("--files", default="", help="only these shards, comma separated (run one process per GPU)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many trellis tensors (testing)")
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(a.dst, exist_ok=True)
    t0 = time.time()
    done = 0
    errs = []
    # a tensor's codebook flag can live in another shard than its trellis (lm_head does), so look at the keys of all of them
    all_keys = set()
    for fn in os.listdir(a.src):
        if fn.endswith(".safetensors") and fn != "cal_trace.safetensors":
            with safe_open(os.path.join(a.src, fn), "pt") as f:
                all_keys.update(f.keys())
    for fn in sorted(os.listdir(a.src)):
        src = os.path.join(a.src, fn)
        if not fn.endswith(".safetensors") or fn == "cal_trace.safetensors":
            if os.path.isfile(src) and not fn.endswith(".safetensors"):
                if fn == "quantization_config.json":
                    cfg = json.load(open(src))
                    cfg["tile_layout"] = "volta"
                    json.dump(cfg, open(os.path.join(a.dst, fn), "w"), indent=4)
                else:
                    shutil.copy(src, os.path.join(a.dst, fn))
            continue
        if a.files and fn not in a.files.split(","):
            continue
        tensors = {}
        with safe_open(src, "pt") as f:
            meta = f.metadata()
            names = list(f.keys())
            for name in names:
                t = f.get_tensor(name)
                if name.endswith(".trellis") and a.only in name and (not a.limit or done < a.limit):
                    base = name[:-len(".trellis")]
                    K = t.shape[2] // 16
                    mul1 = (base + ".mul1") in all_keys
                    mcg = (base + ".mcg") in all_keys
                    t, err = transcode(t, K, mul1, mcg, device)
                    errs.append(err)
                    done += 1
                    print(f"[{time.time() - t0:7.1f}s] {name} K={K} relative re-encode error {err:.4f}", flush=True)
                tensors[name] = t.contiguous()
        save_file(tensors, os.path.join(a.dst, fn), metadata=meta)
        print(f"wrote {fn}", flush=True)
    if errs:
        print(f"{done} tensors, mean relative re-encode error {sum(errs) / len(errs):.4f}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
