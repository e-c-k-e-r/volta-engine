"""Quantize the DSpark drafter (mtp.* of the DeepSeek-V4.1 checkpoint) to EXL3, uncalibrated (the converter's own treatment of MTP
layers: no Hessian, fallback path), one worker process per GPU.

  quantize_dspark.py --out DIR [--k-expert 3] [--k-dense 4] [--k-wo-a 5] [--blocks 0,1,2] [--limit N] [--worker i --workers n]
Writes DIR/model-exl3-mtp-w<i>.safetensors with <key>.trellis / .suh / .svh / .mul1 for every quantized linear:
  mtp.K.attn.{wq_a,wq_b,wkv,wo_b}, mtp.K.attn.wo_a.slice.G, mtp.K.ffn.shared_experts.{w1,w2,w3}, mtp.K.ffn.experts.E.{w1,w2,w3},
  mtp.0.main_proj.
Everything else (norms, gate, hyper-connection tensors, markov and confidence heads) stays official in the main checkpoint."""
import argparse, json, os, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("EXLLAMAV3_DIR", os.path.expanduser("~/Programs/exllamav3")))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="the EXL3 checkpoint directory (official mtp.* tensors)")
ap.add_argument("--out", required=True)
ap.add_argument("--k-expert", type=int, default=3)
ap.add_argument("--k-dense", type=int, default=4)
ap.add_argument("--k-wo-a", type=int, default=5)
ap.add_argument("--blocks", default="0,1,2")
ap.add_argument("--limit", type=int, default=0, help="only the first N items (timing test)")
ap.add_argument("--worker", type=int, default=-1)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--aux-only", action="store_true", help="write only model-aux.safetensors (the official tensors the engine needs next to the quantized ones)")
a = ap.parse_args()

NB, NE = 3, 128
blocks = [int(x) for x in a.blocks.split(",")]

def items():
    """(key, source tensor name, row range or None, K)"""
    out = []
    for b in blocks:
        p = f"mtp.{b}"
        for n in ("wq_a", "wq_b", "wkv", "wo_b"):
            out.append((f"{p}.attn.{n}", f"{p}.attn.{n}", None, a.k_dense))
        for g in range(8):
            out.append((f"{p}.attn.wo_a.slice.{g}", f"{p}.attn.wo_a", (g * 1024, (g + 1) * 1024), a.k_wo_a))
        for n in ("w1", "w2", "w3"):
            out.append((f"{p}.ffn.shared_experts.{n}", f"{p}.ffn.shared_experts.{n}", None, a.k_dense))
        for e in range(NE):
            for n in ("w1", "w2", "w3"):
                out.append((f"{p}.ffn.experts.{e}.{n}", f"{p}.ffn.experts.{e}.{n}", None, a.k_expert))
        if b == 0:
            out.append((f"{p}.main_proj", f"{p}.main_proj", None, a.k_dense))
    return out

if a.aux_only:
    # the tensors that stay official (norms, router, hyper-connections, attention sinks, markov and confidence heads) -> DIR/model-aux.safetensors,
    # so that DIR is the whole drafter (the engine reads it as draft_model_dir)
    import dsv41_ref as R  # noqa: E402
    from safetensors.torch import save_file  # noqa: E402
    idx = R.TensorIndex(a.model)
    linear = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "main_proj")
    aux = {}
    for k in sorted(idx.keys("mtp.")):
        if k.endswith(".scale") or ".experts." in k or ".shared_experts." in k or any(f".{n}.weight" in k for n in linear):
            continue
        aux[k] = idx.get(k, "cpu").contiguous()
    os.makedirs(a.out, exist_ok=True)
    save_file(aux, f"{a.out}/model-aux.safetensors")
    print(f"{len(aux)} tensors, {sum(t.numel() * t.element_size() for t in aux.values()) / 2**20:.0f} MiB")
    sys.exit(0)

if a.worker < 0:
    os.makedirs(a.out, exist_ok=True)
    procs = []
    for w in range(a.workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w))
        cmd = [sys.executable] + sys.argv + ["--worker", str(w)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(f"{a.out}/worker{w}.log", "w"), stderr=subprocess.STDOUT))
    rc = [p.wait() for p in procs]
    print("workers exited", rc)
    sys.exit(max(rc))

import dsv41_ref as R  # noqa: E402
from kernel import fp4_dequant_weight  # noqa: E402
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

idx = R.TensorIndex(a.model)
todo = items()
if a.limit:
    todo = todo[:a.limit]
mine = [(i, it) for i, it in enumerate(todo) if i % a.workers == a.worker]
print(f"worker {a.worker}: {len(mine)} of {len(todo)} tensors", flush=True)

def dequant(src, rows):
    w = idx.get(src + ".weight", "cuda")
    s = idx.get(src + ".scale", "cuda")
    if w.dtype == torch.float8_e4m3fn:
        w = w.float()
        sc = s.float().repeat_interleave(32, 0)[:w.shape[0]].repeat_interleave(32, 1)[:, :w.shape[1]]
        w = w * sc
    else:
        w = fp4_dequant_weight(w.view(torch.uint8), s)
    if rows is not None:
        w = w[rows[0]:rows[1]]
    return w.t().contiguous()   # [in, out]

tensors = {}
t0 = time.time()
for n, (i, (key, src, rows, K)) in enumerate(mine):
    w = dequant(src, rows)
    quant_args = {"seed": i, "K": K, "devices": ["cuda:0"], "device_ratios": None, "apply_out_scales": None, "mul1": True,
                  "debug_dir": "/tmp/dspark_dbg"}
    H = {"H": torch.zeros(w.shape[0], w.shape[0], device="meta"), "first_key": key, "count": 0, "finalized": False,
         "num_total": 0, "device": torch.device("cuda:0")}
    t1 = time.time()
    wq, err, out = quantize_exl3(w, H, quant_args, False, None, False, None)
    for name, t in out.items():
        tensors[f"{key}.{name}"] = t.cpu().contiguous()
    if n % 10 == 0 or a.limit:
        print(f"{n + 1}/{len(mine)} {key} K={K} {w.shape[0]}x{w.shape[1]} err {err:.5f} {time.time() - t1:.1f} s (total {time.time() - t0:.0f} s)", flush=True)
os.makedirs(a.out, exist_ok=True)
save_file(tensors, f"{a.out}/model-exl3-mtp-w{a.worker}.safetensors")
print("done", flush=True)
