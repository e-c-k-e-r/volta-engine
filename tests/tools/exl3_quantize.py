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
engine's format, or to a list of them (one per rank of a row-parallel input: block diagonal); one Hessian may serve several tensors that share an input (q, k, v).

  exl3_quantize.py --src <bf16 dir> --template <exl3 dir> --out <dir> [--order volta|stock] [--hess map.json] [--device cuda:0]
                   [--only SUBSTRING] [--claim] [--limit N] [--seed N]
"""
import argparse, collections, glob, json, os, re, shutil, sys, time

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


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_EDGES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]       # midpoints; ties go to the lower magnitude


def nvfp4_dequant(packed, scale, inv_global, dev):
    """NVFP4 weight [N, K] fp32: E2M1 codes (the even element of a pair in the low nibble) times the E4M3 scale of its block of 16 times inv_global"""
    import torch
    p = packed.to(dev)
    codes = torch.stack([p & 15, p >> 4], dim=-1).reshape(p.shape[0], -1)
    mag = torch.tensor(E2M1, device=dev)[(codes & 7).long()]
    val = torch.where((codes & 8) != 0, -mag, mag)
    return val * scale.to(dev).float().repeat_interleave(16, 1) * inv_global


def nvfp4_roundtrip(w):
    """w [N, K] fp32 through NVFP4 as post-training quantizers write it (modelopt): one fp32 scale per tensor, an E4M3 scale per 16 values (amax / 6), round to nearest E2M1"""
    import torch
    N, K = w.shape
    gs = w.abs().max() / (6.0 * 448.0)
    blocks = w.reshape(N, K // 16, 16)
    sc = (blocks.abs().amax(-1) / 6.0 / gs).clamp(min=1e-12).to(torch.float8_e4m3fn).float()      # per block, in units of gs
    scale = (sc * gs).unsqueeze(-1)
    a = (blocks / scale).abs()
    m = torch.bucketize(a, torch.tensor(E2M1_EDGES, device=w.device))
    q = torch.tensor(E2M1, device=w.device)[m] * torch.where(blocks < 0, -1.0, 1.0)
    return (q * scale).reshape(N, K)


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
    ap.add_argument("--nvfp4-roundtrip", action="store_true", help="pass every BF16 source weight through NVFP4 first (as a post-training NVFP4 checkpoint would have it): measures what the detour costs")
    ap.add_argument("--skip", default="", help="comma separated substrings: matching tensors keep the template's version (the vision tower, whose widths are not multiples of 128)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--files", default="", help="only these template shards, comma separated; the others are linked to the template's (a partial requantization costs one shard of disk)")
    ap.add_argument("--plan", action="store_true", help="only list what would be done with every tensor of the template (from the index files: no GPU, no tensor data)")
    ap.add_argument("--passthrough", action="store_true", help="tensors the template does not quantize (norms, gates, embeddings ...) come from the original when it has them, not from the template (needed when the template is another model's checkpoint)")
    ap.add_argument("--redo", action="store_true", help="quantize shards again that already exist in --out")
    ap.add_argument("--claim", action="store_true", help="work on the next template shard nobody has claimed (one process per GPU)")
    a = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    qz = None if a.plan else load_upstream(a.order)
    device = torch.device(a.device)
    hmap = json.load(open(a.hess)) if a.hess else {}
    if not a.plan: os.makedirs(a.out, exist_ok=True)

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

    fused_cache = collections.OrderedDict()      # a few fused expert tensors of the original (BF16 [experts, ...]), the layer being quantized

    def fused_ref(base):
        """('<prefix>.experts', expert, 'gate_proj' | 'up_proj' | 'down_proj') for a per-expert name whose original stores the experts fused (gate_up_proj [E, 2I, H], down_proj [E, H, I]), else None"""
        m = re.match(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)$", base)
        if m and (m.group(1) + ".gate_up_proj") in src_idx and (m.group(1) + ".down_proj") in src_idx:
            return m.group(1), int(m.group(2)), m.group(3)
        return None

    def fused_tensor(name):
        if name not in fused_cache:
            fused_cache[name] = src_get(name)
            while len(fused_cache) > 4: fused_cache.popitem(last=False)
        fused_cache.move_to_end(name)
        return fused_cache[name]

    def has_src(base):
        return base + ".weight" in src_idx or base + ".weight_packed" in src_idx or fused_ref(base) is not None

    def src_weight(base, dev):
        """<base> as fp32 [out, in]: BF16/FP16 as it is; FP8 (E4M3) with its block scales (<base>.weight_scale_inv, one per 128 x 128 block); NVFP4 of either layout
        (compressed-tensors: weight_packed / weight_scale / weight_global_scale, a divisor; modelopt: uint8 weight / weight_scale / weight_scale_2, a multiplier)"""
        fr = fused_ref(base)
        if fr:
            prefix, e, proj = fr
            if proj == "down_proj":
                return fused_tensor(prefix + ".down_proj")[e].to(dev).float()              # [hidden, intermediate]: [out, in]
            gu = fused_tensor(prefix + ".gate_up_proj")[e]                                 # [2 * intermediate, hidden]: gate rows first, then up
            half = gu.shape[0] // 2
            return (gu[:half] if proj == "gate_proj" else gu[half:]).to(dev).float()
        if base + ".weight_packed" in src_idx:
            return nvfp4_dequant(src_get(base + ".weight_packed"), src_get(base + ".weight_scale"), 1.0 / float(src_get(base + ".weight_global_scale").float().item()), dev)
        w = src_get(base + ".weight").to(dev)
        if w.dtype == torch.uint8 and base + ".weight_scale_2" in src_idx:
            return nvfp4_dequant(w, src_get(base + ".weight_scale"), float(src_get(base + ".weight_scale_2").float().item()), dev)
        if w.dtype == torch.float8_e4m3fn:
            sc = src_get(base + ".weight_scale_inv").to(dev).float()
            bo, bi = w.shape[0] // sc.shape[0] if w.shape[0] % sc.shape[0] == 0 else 128, w.shape[1] // sc.shape[1] if w.shape[1] % sc.shape[1] == 0 else 128
            sc = sc.repeat_interleave(bo, 0)[:w.shape[0]].repeat_interleave(bi, 1)[:, :w.shape[1]]
            return w.float() * sc
        return nvfp4_roundtrip(w.float()) if a.nvfp4_roundtrip and not base.endswith("lm_head") else w.float()

    qcfg = json.load(open(os.path.join(a.template, "quantization_config.json")))
    out_scales = {"always": True, "never": False}.get(qcfg.get("out_scales", "auto"))
    all_keys = set()
    for fn in os.listdir(a.template):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(a.template, fn), "pt") as f:
                all_keys.update(f.keys())


    if a.plan:
        skips = [x for x in a.skip.split(",") if x]
        quant_bases = {k[:-len(".trellis")] for k in all_keys if k.endswith(".trellis")}
        counts, missing, kept = collections.Counter(), [], []
        for k in sorted(all_keys):
            stem = k.rsplit(".", 1)[0]
            if k.rsplit(".", 1)[-1] in ("trellis", "suh", "svh", "mul1", "mcg") and stem in quant_bases:
                if k.endswith(".trellis"):
                    if "ngram_embedding" in k: counts["n-gram table (kept from the template)"] += 1
                    elif any(x in k for x in skips): counts["quantized tensor kept from the template (--skip)"] += 1
                    elif has_src(stem): counts["quantize from the original"] += 1
                    else: counts["quantize but the original has no such tensor"] += 1; missing.append(stem)
                continue
            if any(x in k for x in skips) or "ngram_embedding" in k: counts["kept from the template"] += 1; continue
            if a.passthrough and k in src_idx: counts["copied from the original"] += 1
            else: counts["kept from the template"] += 1; kept.append(k)
        for k, v in counts.most_common(): print(f"{v:7d}  {k}")
        for m in missing[:20]: print("   no source for", m)
        print("kept from the template (first 25):", kept[:25])
        return

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
        if os.path.exists(final) and not os.path.islink(final) and fn.startswith("model") and not a.redo:
            print(f"{fn}: already written, skipped (--redo quantizes it again)", flush=True)
            continue
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
                if base and a.only in name and not any(k and k in name for k in a.skip.split(",")) and (not a.limit or done < a.limit) and has_src(base):
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
                        # one file, or one per rank for a row-parallel input: the rank blocks form a block diagonal (the cross-rank covariance is not captured)
                        blocks = []
                        for path in (hp if isinstance(hp, list) else [hp]):
                            Hb, rows = load_hessian(path, device)
                            blocks.append(Hb)
                        H = blocks[0] if len(blocks) == 1 else torch.block_diag(*blocks)
                        assert H.shape[0] == kk, f"{base}: Hessian {tuple(H.shape)} for {kk} inputs"
                        H_data.update(H=H, count=rows)      # the capture files hold sums over the rows; quantize.py divides by the row count
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
                if a.passthrough and name in src_idx and not any(x and x in name for x in a.skip.split(",")):
                    st = src_get(name)
                    if tuple(st.shape) == tuple(t.shape): t = st.to(t.dtype)
                    else: print(f"{name}: shape {tuple(st.shape)} in the original, {tuple(t.shape)} in the template: kept from the template", flush=True)
                tensors[name] = t.contiguous()
        tensors.update(new)         # the template's suh / svh / codebook markers of a requantized tensor give way to the new ones
        save_file(tensors, final + ".tmp", metadata=meta)
        os.replace(final + ".tmp", final)
        shutil.rmtree(part_dir, ignore_errors=True)
        if a.claim: os.unlink(final + ".claim")
        print(f"wrote {fn}: {done} tensors, mean proxy error {sum(errs) / max(len(errs), 1):.5f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
