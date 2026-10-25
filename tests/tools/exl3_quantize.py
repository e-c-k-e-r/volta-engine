#!/usr/bin/env python3
"""Quantize BF16 weights to EXL3 with upstream exllamav3's quantizer, in the stock or the Volta tile order.

The quantizer is upstream's quantize_exl3 (regularization, global scale search, block-LDL error feedback, trellis encoder, refit), unmodified. Only
the 16x16 tile permutation in front of the encoder is swapped (see exl3_volta_layout.py), so a Volta-order checkpoint is quantized from the
BF16 weights directly and carries no re-encode error. The pieces it needs from upstream's extension are built from upstream's sources on first
use (tests/tools/exl3_quant_ext.cpp, sm_70); upstream's Python is imported from EXLLAMAV3_UPSTREAM_DIR (default ~/Programs/exllamav3-upstream).

A template checkpoint (an existing EXL3 directory of the same model) supplies which tensors are quantized, at which rate (4 bpw, 2.5 bpw, ...),
with which codebook, plus every tensor that stays as it is and the config files. For each <name>.trellis of the template the matching
<name>.weight is read from the source directory, quantized, and written as <name>.trellis / .suh / .svh (+ the codebook marker).

Hessians (x^T x of a projection's input, from the engine's ENGINE_HESSIAN_DIR capture) are optional: a tensor without one is quantized
without calibration (upstream's fallback). --hess takes a JSON file mapping a tensor name (or a name prefix ending in '*') to a Hessian file of the
engine's format; one Hessian may serve several tensors that share an input (q, k, v).

  exl3_quantize.py --src <bf16 dir> --template <exl3 dir> --out <dir> [--order volta|stock] [--hess map.json] [--device cuda:0]
                   [--only SUBSTRING] [--claim] [--limit N] [--seed N]
"""
import argparse, glob, json, os, re, shutil, sys, time

UPSTREAM = os.environ.get("EXLLAMAV3_UPSTREAM_DIR", os.path.expanduser("~/Programs/exllamav3-upstream"))
EXT_BUILD = os.path.expanduser("~/.cache/volta_engine_exl3_quant_ext")


def build_ext():
    """upstream's trellis encoder, packers and codebook decode as the module exllamav3_ext (JIT, sm_70)"""
    import torch
    from torch.utils.cpp_extension import load
    up = UPSTREAM + "/exllamav3/exllamav3_ext"
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    os.makedirs(EXT_BUILD, exist_ok=True)
    srcs = [here + "/exl3_quant_ext.cpp"] + [up + "/quant/" + f for f in ("quantize.cu", "pack.cu", "frac.cu", "exl3_devctx.cu", "util.cu")] \
        + sorted(glob.glob(up + "/quant/comp_units/quantize_tiles_*inst*.cu"))
    return load(name="exllamav3_ext", sources=[s for s in srcs if os.path.exists(s)], extra_include_paths=[up, up + "/quant"], build_directory=EXT_BUILD,
                verbose=False, extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"],
                extra_cflags=["-O2", "-std=c++17"])


def volta_perm_fn(torch):
    """perm[t] = row-major tile element (k * 16 + n) at stream position t, as quantize.tensor_core_perm returns it"""
    cache = {}

    def perm(device):
        key = str(device)
        if key not in cache:
            p = [0] * 256
            for lane in range(32):
                q, t_, hh = (lane >> 2) & 3, lane & 3, lane >> 4
                col = 8 * (q & 1) + t_ + 4 * hh
                for s in range(2):
                    for i in range(4):
                        p[lane * 8 + 4 * s + i] = (4 * (2 * (q >> 1) + s) + i) * 16 + col
            assert sorted(p) == list(range(256))
            cache[key] = torch.tensor(p, dtype=torch.int, device=device)
        return cache[key]
    return perm


def load_upstream(order):
    """import upstream's quantizer with the extension built here, and the tile order selected"""
    import torch
    ext = build_ext()
    # upstream's package __init__ imports the whole model stack (and with it every op of the full extension): load only the quantizer's modules
    import types
    pkg = UPSTREAM + "/exllamav3"
    for name, path in (("exllamav3", pkg), ("exllamav3.modules", pkg + "/modules"), ("exllamav3.modules.quant", pkg + "/modules/quant"),
                       ("exllamav3.modules.quant.exl3_lib", pkg + "/modules/quant/exl3_lib")):
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
    stub = types.ModuleType("exllamav3.ext")
    stub.exllamav3_ext = ext
    sys.modules["exllamav3.ext"] = stub
    import importlib
    qz = importlib.import_module("exllamav3.modules.quant.exl3_lib.quantize")
    # upstream keeps tensors above 5e8 elements on the CPU, where its uncalibrated path (fallback_quant) cannot run: bring the weight to the GPU for it
    orig_fallback = qz.fallback_quant

    def fallback_on_gpu(weight, q_device, quant_args, pb=None):
        if not weight.is_cuda:
            weight = weight.to(quant_args["devices"][0])
        return orig_fallback(weight, q_device, quant_args, pb)
    qz.fallback_quant = fallback_on_gpu
    if order == "volta":
        perm = volta_perm_fn(torch)
        qz.tensor_core_perm = perm
        qz.tensor_core_perm_i = lambda device: torch.argsort(perm(device))
    return qz


def load_hessian(path, device):
    import numpy as np, torch
    with open(path, "rb") as f:
        magic, k = np.frombuffer(f.read(8), dtype=np.int32)
        rows = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        assert magic == 0x48455353, path
        h = np.frombuffer(f.read(), dtype=np.float32).reshape(k, k)
    return torch.from_numpy(h.copy()).to(device), rows


def hessian_path(hmap, name):
    if name in hmap:
        return hmap[name]
    for pat, p in hmap.items():
        if pat.endswith("*") and name.startswith(pat[:-1]):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="volta", choices=["volta", "stock"])
    ap.add_argument("--hess", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="", help="comma separated substrings: matching tensors keep the template's version (the vision tower, whose widths are not multiples of 128)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--files", default="", help="only these template shards, comma separated; the others are linked to the template's (a partial requantization costs one shard of disk)")
    ap.add_argument("--claim", action="store_true", help="work on the next template shard nobody has claimed (one process per GPU)")
    a = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    qz = load_upstream(a.order)
    device = torch.device(a.device)
    hmap = json.load(open(a.hess)) if a.hess else {}
    os.makedirs(a.out, exist_ok=True)

    # source tensors by name (the index file when there is one: opening every shard of a network share only to list its keys is slow)
    src_idx = {}
    idx_path = os.path.join(a.src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        src_idx = dict(json.load(open(idx_path))["weight_map"])
    else:
        for fn in sorted(os.listdir(a.src)):
            if fn.endswith(".safetensors"):
                with safe_open(os.path.join(a.src, fn), "pt") as f:
                    for k in f.keys():
                        src_idx[k] = fn
    handles = {}

    def src_get(name):
        fn = src_idx[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(a.src, fn), "pt")
        return handles[fn].get_tensor(name)

    def src_weight(base, dev):
        """<base>.weight as fp32 [out, in]: BF16/FP16 as it is, FP8 (E4M3) with its block scales (<base>.weight_scale_inv, one per 128 x 128 block) applied"""
        w = src_get(base + ".weight").to(dev)
        if w.dtype == torch.float8_e4m3fn:
            sc = src_get(base + ".weight_scale_inv").to(dev).float()
            bo, bi = w.shape[0] // sc.shape[0] if w.shape[0] % sc.shape[0] == 0 else 128, w.shape[1] // sc.shape[1] if w.shape[1] % sc.shape[1] == 0 else 128
            sc = sc.repeat_interleave(bo, 0)[:w.shape[0]].repeat_interleave(bi, 1)[:, :w.shape[1]]
            return w.float() * sc
        return w.float()

    qcfg = json.load(open(os.path.join(a.template, "quantization_config.json")))
    out_scales = {"always": True, "never": False}.get(qcfg.get("out_scales", "auto"))
    all_keys = set()
    for fn in os.listdir(a.template):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(a.template, fn), "pt") as f:
                all_keys.update(f.keys())

    for fn in sorted(os.listdir(a.template)):
        p = os.path.join(a.template, fn)
        if not fn.endswith(".safetensors"):
            if os.path.isfile(p) and fn != "model.safetensors.index.json" or fn == "model.safetensors.index.json":
                if fn == "quantization_config.json":
                    cfg = json.load(open(p))
                    if a.order == "volta": cfg["tile_layout"] = "volta"
                    json.dump(cfg, open(os.path.join(a.out, fn), "w"), indent=4)
                else:
                    shutil.copy(p, os.path.join(a.out, fn))
            continue
        final = os.path.join(a.out, fn)
        if a.files and fn not in a.files.split(","):
            if not os.path.lexists(final): os.symlink(os.path.abspath(p), final)
            continue
        if a.claim:
            if os.path.exists(final): continue
            try: os.close(os.open(final + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError: continue
        tensors, new = {}, {}
        part_dir = os.path.join(a.out, ".partial", fn)    # every finished tensor is saved here, so a restart of this shard continues where it stopped
        os.makedirs(part_dir, exist_ok=True)
        t0 = time.time()
        done = 0
        errs = []
        with safe_open(p, "pt") as f:
            meta = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                base = name[:-len(".trellis")] if name.endswith(".trellis") else None
                if base and a.only in name and not any(k and k in name for k in a.skip.split(",")) and (not a.limit or done < a.limit) and base + ".weight" in src_idx:
                    part = os.path.join(part_dir, base + ".safetensors")
                    if os.path.exists(part):
                        with safe_open(part, "pt") as pf:
                            for k in pf.keys(): new[k] = pf.get_tensor(k)
                        done += 1
                        print(f"[{time.time() - t0:7.1f}s] {base} (from a previous run)", flush=True)
                        continue
                    w = src_weight(base, device).t().contiguous()     # [in, out]
                    kk, nn = w.shape
                    words = t.shape[2]
                    K = words // 16 if words % 16 == 0 else words // 16 + 0.5
                    qa = {"seed": a.seed + done, "K": K, "devices": [a.device], "device_ratios": None, "apply_out_scales": out_scales,
                          "debug_dir": os.path.join(a.out, "debug")}
                    if base + ".mcg" in all_keys: qa["mcg"] = True
                    if base + ".mul1" in all_keys: qa["mul1"] = True
                    hp = hessian_path(hmap, base)
                    H_data = {"first_key": base, "count": 0, "finalized": False, "num_total": 0, "device": device,
                              "inf_nan": torch.zeros(2, dtype=torch.long, device=device)}
                    if hp:
                        H, rows = load_hessian(hp, device)
                        assert H.shape[0] == kk, f"{base}: Hessian {tuple(H.shape)} for {kk} inputs"
                        H_data.update(H=H * float(rows), count=rows)      # quantize.py divides by the row count
                    else:
                        H_data["H"] = torch.zeros(kk, kk, device="meta")
                    wq, proxy, out = qz.quantize_exl3(w, H_data, qa, False)
                    for k, v in out.items():
                        new[f"{base}.{k}"] = v.cpu().contiguous()
                    assert new[base + ".trellis"].shape == t.shape, (base, new[base + ".trellis"].shape, t.shape)
                    save_file({k: v for k, v in new.items() if k.startswith(base + ".")}, part + ".tmp")
                    os.replace(part + ".tmp", part)
                    errs.append(proxy)
                    done += 1
                    print(f"[{time.time() - t0:7.1f}s] {base} K={K} {kk}x{nn} proxy error {proxy:.5f}{'' if hp else ' (no Hessian)'}", flush=True)
                    del w, wq
                    continue
                tensors[name] = t.contiguous()
        tensors.update(new)         # the template's suh / svh / codebook markers of a requantized tensor give way to the new ones
        save_file(tensors, final + ".tmp", metadata=meta)
        os.replace(final + ".tmp", final)
        shutil.rmtree(part_dir, ignore_errors=True)
        if a.claim: os.unlink(final + ".claim")
        print(f"wrote {fn}: {done} tensors, mean proxy error {sum(errs) / max(len(errs), 1):.5f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
