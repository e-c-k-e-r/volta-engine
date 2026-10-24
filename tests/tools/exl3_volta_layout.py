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

Half-integer rates (1.5, 2.5, 3.5 bpw: tiles of 16 K + 8 words) go through the fractional-rate quantizer and packer of upstream exllamav3, which the installed
extension predates: tests/tools/exl3_frac_ext.cu is built from the upstream sources on first use (EXLLAMAV3_UPSTREAM_DIR, default ~/Programs/exllamav3-upstream).

Re-encoding already-quantized weights adds about one quantization's worth of error (the stored values are not jointly
representable in another order); the order itself does not change the error. A real conversion from BF16 avoids that.

  exl3_volta_layout.py <in_dir> <out_dir> [--device cuda:0] [--files SHARD,SHARD] [--only SUBSTRING]
"""
import argparse, json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.environ.get("EXLLAMAV3_DIR", os.path.expanduser("~/Programs/exllamav3")))
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles, pack_trellis, tensor_core_perm_i


_frac = None


def frac_ext():
    """The fractional-rate quantizer and tile packer, built once from upstream's sources (JIT, sm_70)."""
    global _frac
    if _frac is None:
        from torch.utils.cpp_extension import load
        up = os.environ.get("EXLLAMAV3_UPSTREAM_DIR", os.path.expanduser("~/Programs/exllamav3-upstream")) + "/exllamav3/exllamav3_ext"
        here = os.path.dirname(os.path.abspath(__file__))
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        build = os.path.expanduser("~/.cache/volta_engine_exl3_frac_ext")
        os.makedirs(build, exist_ok=True)
        _frac = load(name="exl3_frac_ext", sources=[here + "/exl3_frac_ext.cu", up + "/quant/frac.cu", up + "/quant/comp_units/quantize_tiles_frac_inst.cu"],
                     extra_include_paths=[up, up + "/quant"], build_directory=build, verbose=False,
                     extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"], extra_cflags=["-O2", "-std=c++17"])
    return _frac


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
    """trellis: int16 [tiles_k, tiles_n, 16*K (+8 at a half-integer rate)] in the stock order -> same shape in the Volta order."""
    tk, tn, w = trellis.shape
    half = w % 16 == 8
    assert w == 16 * K + (8 if half else 0)
    args = {"K": K}
    if mcg: args["mcg"] = True
    if mul1: args["mul1"] = True
    assert not half or (mul1 and not mcg), "half-integer rates are mul1 only"
    q_ord = volta_perm(device)
    p_inv = tensor_core_perm_i(device).long()
    out = torch.empty_like(trellis)
    rows = max(1, chunk_tiles // tn)
    sq_err = sq_ref = 0.0
    if half:
        fe = frac_ext()
        mask = 0xaaaa
        costs = torch.zeros((128, 2, 65536 >> K), dtype=torch.half, device=device)
        edges = torch.zeros((128, 256, 65536 >> K), dtype=torch.short, device=device)
    for a in range(0, tk, rows):
        b = min(tk, a + rows)
        tr = trellis[a:b].to(device).contiguous()
        unp = torch.zeros((b - a, tn, 256), dtype=torch.int16, device=device)
        if half: fe.unpack_trellis_frac(unp, tr, K, mask)
        else: ext.unpack_trellis(unp, tr, K)
        flat = unp.reshape(-1, 256).contiguous()
        tiles = torch.zeros((flat.shape[0], 256), dtype=torch.float, device=device)
        for c in range(0, flat.shape[0], 32768):          # the decode grid's y dimension is limited to 65535
            ext.decode(flat[c:c + 32768].contiguous(), tiles[c:c + 32768], mcg, mul1 and not mcg)
        new_seq = tiles[:, p_inv[q_ord]].contiguous()       # position t of the new stream holds element q_ord[t]
        if half:
            qv = torch.zeros_like(new_seq)
            qi = torch.zeros_like(new_seq, dtype=torch.short)
            fe.quantize_tiles_frac(new_seq, qv, qi, costs, edges, K, mask)
            packed = torch.zeros((b - a, tn, w), dtype=torch.int16, device=device)
            fe.pack_trellis_frac(packed, qi.reshape(b - a, tn, 256).contiguous(), K, mask)
        else:
            qv, qi = quantize_tiles(new_seq, args)
            packed = pack_trellis(qi.reshape(b - a, tn, 256), args)
        sq_err += (qv - new_seq).square().sum().item()
        sq_ref += new_seq.square().sum().item()
        out[a:b] = packed.cpu()
    return out, (sq_err / max(sq_ref, 1e-30)) ** 0.5


def dropped(name, max_layers):
    m = re.search(r"language_model\.layers\.(\d+)\.", name)
    return bool(max_layers and m and int(m.group(1)) >= max_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="re-encode only tensors whose name contains this (others are copied as they are)")
    ap.add_argument("--files", default="", help="only these shards, comma separated (run one process per GPU)")
    ap.add_argument("--claim", action="store_true", help="take the next shard nobody has claimed (start one process per GPU, add more at any time; a finished shard is skipped on a restart)")
    ap.add_argument("--max-layers", type=int, default=0, help="a checkpoint of the first N decoder layers (drops the tensors of the others, edits num_hidden_layers and the index): the same model cut short, to compare the two orders in an hour instead of a day")
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
                    if a.max_layers and "tensor_storage" in cfg:
                        cfg["tensor_storage"] = {k: v for k, v in cfg["tensor_storage"].items() if not dropped(k + ".", a.max_layers)}
                    json.dump(cfg, open(os.path.join(a.dst, fn), "w"), indent=4)
                elif fn == "config.json" and a.max_layers:
                    cfg = json.load(open(src))
                    (cfg.get("text_config") or cfg)["num_hidden_layers"] = a.max_layers
                    json.dump(cfg, open(os.path.join(a.dst, fn), "w"), indent=2)
                elif fn == "model.safetensors.index.json" and a.max_layers:
                    idx = json.load(open(src))
                    idx["weight_map"] = {k: v for k, v in idx["weight_map"].items() if not dropped(k, a.max_layers)}
                    json.dump(idx, open(os.path.join(a.dst, fn), "w"), indent=2)
                else:
                    shutil.copy(src, os.path.join(a.dst, fn))
            continue
        if a.files and fn not in a.files.split(","):
            continue
        final = os.path.join(a.dst, fn)
        if a.claim:
            if os.path.exists(final):
                continue
            try:
                os.close(os.open(final + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        tensors = {}
        with safe_open(src, "pt") as f:
            meta = f.metadata()
            names = list(f.keys())
            for name in names:
                if dropped(name, a.max_layers): continue
                t = f.get_tensor(name)
                if name.endswith(".trellis") and a.only in name and (not a.limit or done < a.limit):
                    base = name[:-len(".trellis")]
                    K = t.shape[2] // 16   # a half-integer rate K + 0.5 has 16 K + 8 words
                    mul1 = (base + ".mul1") in all_keys
                    mcg = (base + ".mcg") in all_keys
                    t, err = transcode(t, K, mul1, mcg, device)
                    errs.append(err)
                    done += 1
                    print(f"[{time.time() - t0:7.1f}s] {name} K={K} relative re-encode error {err:.4f}", flush=True)
                tensors[name] = t.contiguous()
        if not tensors:   # every tensor of this shard was dropped
            if a.claim: os.unlink(final + ".claim")
            continue
        save_file(tensors, final + ".tmp", metadata=meta)
        os.replace(final + ".tmp", final)
        if a.claim:
            os.unlink(final + ".claim")
        print(f"wrote {fn}", flush=True)
    if errs:
        print(f"{done} tensors, mean relative re-encode error {sum(errs) / len(errs):.4f}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
