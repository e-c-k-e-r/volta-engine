"""Golden reference for DeepSeek-V4.1-Flash EXL3 checkpoints (the sudoingX / 0xSero layout: official tensors in
model-000NN-of-000MM.safetensors, EXL3 dense linears and routed experts in model-exl3-LNN.safetensors, the head in
model-exl3-head.safetensors, Engram tables in DeepSeek's shards 47 / 48).

Runs DeepSeek's own inference/model.py (vendored here, MIT) with the torch kernel stand-ins in kernel.py, one layer at a
time on one GPU, so a checkpoint far larger than VRAM works: each Block is built, loaded, applied to the whole prompt and
dropped. EXL3 linears are reconstructed to FP16 weights by exllamav3 and multiplied in FP32 (no activation quantization:
the EXL3 tensors replace the FP8 / MXFP4 GEMMs). Everything else, including the FP8 / FP4 quantize-dequantize of the KV
caches, the indexer and the wo_a input, is the reference's. Engram rows are read from the shards through a memory map.

Outputs (--out): ids.npy, h_LNN.npy (each block's hc stream [S, hc, dim]), premix_LNN.npy ([S, hc] pre-mix handed to the
next block), logits_last.npy, top8 ids / values per position, logsumexp per position, nll of each next token, meta.json.

Run from anywhere with the exllamav3 venv (needs its compiled extension):
  ~/Programs/exllamav3/.venv/bin/python tests/tools/dsv41/dsv41_ref.py --model DIR --chat "What is 17*19? Reply with the integer only." --out results/dsv41_ref
Options: --layers N stops after N blocks (partial downloads), --no-engram skips the Engram lookups, --gen N greedy-decodes N
tokens by re-running the prefill (slow: every step streams the whole model)."""
import argparse
import glob
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel  # noqa: E402,F401  (model.py imports it by name)
import model as M  # noqa: E402

EXLLAMAV3 = os.environ.get("EXLLAMAV3_DIR", os.path.expanduser("~/Programs/exllamav3"))
sys.path.insert(0, EXLLAMAV3)


# ---------------------------------------------------------------------------------------------------------------------
# checkpoint access

class TensorIndex:
    """Name -> file for every .safetensors in the directory (header scan, so a missing index json does not matter)."""

    def __init__(self, directory: str):
        self.dir = directory
        self.where: dict[str, str] = {}
        self.meta: dict[str, dict] = {}
        self.handles = {}
        for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
            try:
                header = read_header(path)
            except (OSError, ValueError, struct.error):
                print(f"skipping unreadable {os.path.basename(path)} (still downloading?)")
                continue
            for k, v in header.items():
                if k != "__metadata__":
                    self.where[k] = path
                    self.meta[k] = v

    def has(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str, device="cuda") -> torch.Tensor:
        from safetensors import safe_open
        path = self.where[name]
        if path not in self.handles:
            self.handles[path] = safe_open(path, framework="pt", device="cpu")
        return self.handles[path].get_tensor(name).to(device)

    def keys(self, prefix: str):
        return [k for k in self.where if k.startswith(prefix)]


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


# ---------------------------------------------------------------------------------------------------------------------
# EXL3 linears

def exl3_linear(idx: TensorIndex, key: str, in_features: int, out_features: int):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    t = {s: idx.get(f"{key}.{s}") for s in ("trellis", "suh", "svh", "mul1") if idx.has(f"{key}.{s}")}
    return LinearEXL3(None, in_features, out_features, suh=t["suh"], svh=t["svh"], trellis=t["trellis"], mul1=t.get("mul1"),
                      key=key)


class Exl3Dense(nn.Module):
    """y = x @ W with W reconstructed once (FP16 -> FP32), for the per-layer dense linears."""

    def __init__(self, lin):
        super().__init__()
        self.w = lin.get_weight_tensor().float()  # [in, out]

    def forward(self, x):
        return (x.float() @ self.w).to(x.dtype)


class Exl3Lazy(nn.Module):
    """Reconstructs on every call: routed experts (1152 per layer would not fit as FP32)."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return (x.float() @ self.lin.get_weight_tensor().float()).to(x.dtype)


class Exl3Expert(nn.Module):
    """model.Expert with EXL3 projections (same clamps and dtype handling)."""

    def __init__(self, w1, w2, w3, swiglu_limit: float):
        super().__init__()
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.swiglu_limit = swiglu_limit

    forward = M.Expert.forward


# ---------------------------------------------------------------------------------------------------------------------
# Engram tables straight from the shard files

class DiskEngramEmbedding(nn.Module):
    """ParallelEngramEmbedding (world_size 1) reading FP8 rows and E8M0 block scales through a memory map."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings, self.dim = num_embeddings, dim
        self.weight = self.scale = None

    def attach(self, idx: TensorIndex, key: str):
        path = idx.where[f"{key}.weight"]
        data_start = 8 + struct.unpack("<Q", open(path, "rb").read(8))[0]
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        w, s = idx.meta[f"{key}.weight"], idx.meta[f"{key}.scale"]
        assert w["dtype"] == "F8_E4M3" and s["dtype"] == "F8_E8M0", (w["dtype"], s["dtype"])
        assert w["shape"] == [self.num_embeddings, self.dim], (w["shape"], self.num_embeddings, self.dim)
        a, b = w["data_offsets"]
        self.weight = mm[data_start + a:data_start + b].reshape(w["shape"])
        a, b = s["data_offsets"]
        self.scale = mm[data_start + a:data_start + b].reshape(s["shape"])
        self.block = self.dim // s["shape"][1]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.flatten().cpu().numpy()
        rows = torch.from_numpy(np.ascontiguousarray(self.weight[flat])).to(indices.device)
        scales = torch.from_numpy(np.ascontiguousarray(self.scale[flat])).to(indices.device)
        v = rows.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, self.block))
        s = scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        return (v * s).flatten(-2).to(torch.bfloat16).view(*indices.shape, self.dim)


# ---------------------------------------------------------------------------------------------------------------------
# attention forward with the grouped wo_a done in FP32 (the reference einsums bf16 x bf16)

def attention_forward(self, x: torch.Tensor, start_pos: int):
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
    rd = self.rope_head_dim
    qr = self.q_norm(self.wq_a(x))
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    M.apply_rotary_emb(q[..., -rd:], freqs_cis)
    kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
    if self.compress_ratio:
        compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))
        kv = torch.cat([kv, compress_kv], dim=1)
        topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
    o = M.sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    M.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    M.act_quant(o, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    o = torch.einsum("bsgd,grd->bsgr", o.float(), self.wo_a_groups).to(x.dtype)
    return self.wo_b(o.flatten(2))


M.Attention.forward = attention_forward


# ---------------------------------------------------------------------------------------------------------------------
# block construction

def build_block(layer_id: int, args: M.ModelArgs, layout, idx: TensorIndex, use_engram: bool) -> M.Block:
    real_expert, real_engram_embed = M.Expert, M.ParallelEngramEmbedding
    M.Expert = lambda *a, **k: None  # routed and shared experts come from the EXL3 file
    M.ParallelEngramEmbedding = DiskEngramEmbedding
    try:
        block = M.Block(layer_id, args, layout if use_engram else None)
    finally:
        M.Expert, M.ParallelEngramEmbedding = real_expert, real_engram_embed
    p = f"layers.{layer_id}"
    dim, q_rank, hd, nh = args.dim, args.q_lora_rank, args.head_dim, args.n_heads
    a = block.attn
    a.wq_a = Exl3Dense(exl3_linear(idx, f"{p}.attn.wq_a", dim, q_rank))
    a.wq_b = Exl3Dense(exl3_linear(idx, f"{p}.attn.wq_b", q_rank, nh * hd))
    a.wkv = Exl3Dense(exl3_linear(idx, f"{p}.attn.wkv", dim, hd))
    a.wo_b = Exl3Dense(exl3_linear(idx, f"{p}.attn.wo_b", args.o_groups * args.o_lora_rank, dim))
    in_g = nh * hd // args.o_groups
    a.wo_a = None
    a.wo_a_groups = torch.stack([exl3_linear(idx, f"{p}.attn.wo_a.slice.{g}", in_g, args.o_lora_rank).get_weight_tensor().float().t()
                                 for g in range(args.o_groups)])  # [groups, o_lora_rank, in_g]
    ffn = block.ffn
    lim = args.swiglu_limit
    def expert(key, lazy):
        wrap = Exl3Lazy if lazy else Exl3Dense
        return Exl3Expert(wrap(exl3_linear(idx, f"{key}.w1", dim, args.moe_inter_dim)),
                          wrap(exl3_linear(idx, f"{key}.w2", args.moe_inter_dim, dim)),
                          wrap(exl3_linear(idx, f"{key}.w3", dim, args.moe_inter_dim)), lim)
    ffn.shared_experts = expert(f"{p}.ffn.shared_experts", False)
    ffn.experts = nn.ModuleList([expert(f"{p}.ffn.experts.{e}", True) for e in range(args.n_routed_experts)])
    # every remaining parameter is an official tensor
    missing = []
    with torch.no_grad():
        for name, param in block.named_parameters():  # the EXL3 wrappers hold plain tensors, not parameters
            key = f"{p}.{name}"
            if not idx.has(key):
                missing.append(key)
                continue
            src = idx.get(key, param.device)
            assert src.shape == param.shape, (key, src.shape, param.shape)
            if src.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                param.view(torch.uint8).copy_(src.view(torch.uint8))
            else:
                param.copy_(src.to(param.dtype))
    if missing:
        raise KeyError(f"layer {layer_id}: tensors not in the checkpoint: {missing}")
    if block.engram is not None:
        block.engram.embed.attach(idx, f"{p}.engram.embed")
    return block


def check_exl3(idx: TensorIndex, args: M.ModelArgs):
    """Reconstruct + FP32 matmul must match exllamav3's own EXL3 forward (orientation and Hadamard handling)."""
    lin = exl3_linear(idx, "layers.0.attn.wq_a", args.dim, args.q_lora_rank)
    x = torch.randn(4, args.dim, device="cuda", dtype=torch.half)
    ref = lin.forward(x, {}).float()
    mine = x.float() @ lin.get_weight_tensor().float()
    rel = ((mine - ref).norm() / ref.norm()).item()
    print(f"EXL3 reconstruct check (layers.0.attn.wq_a): rel {rel:.2e}")
    assert rel < 1e-2, rel


class Exl3Head(nn.Module):
    def __init__(self, idx: TensorIndex, dim: int, vocab: int):
        super().__init__()
        self.w = exl3_linear(idx, "head", dim, vocab).get_weight_tensor().float()  # [dim, vocab]

    def forward(self, x):
        rows = x.reshape(-1, x.size(-1)).float()
        return torch.cat([rows[r:r + 64] @ self.w for r in range(0, rows.size(0), 64)]).view(*x.shape[:-1], -1)


# ---------------------------------------------------------------------------------------------------------------------

def make_args(seqlen: int) -> M.ModelArgs:
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    cfg.update(max_batch_size=1, max_seq_len=max(256, 1 << math.ceil(math.log2(seqlen + 1))), vision_n_layers=0)
    names = set(M.ModelArgs.__dataclass_fields__)
    return M.ModelArgs(**{k: v for k, v in cfg.items() if k in names})


@torch.inference_mode()
def forward(ids: torch.Tensor, args, idx, tokenizer, use_engram: bool, n_layers: int, out: str | None):
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    M.shared_attn = M.SharedAttentionRuntime()
    layout = M.EngramLayout.from_args(args) if use_engram else None
    hashes = None
    if layout is not None:
        if not hasattr(forward, "hash_state"):  # the compressed token map takes a while to build
            forward.hash_state = M.NgramHashState(args, layout, tokenizer)
        hashes = forward.hash_state(ids, 0)
    h = M.F.embedding(ids, idx.get("embed.weight"))
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
    block = None
    for i in range(n_layers):
        t0 = time.time()
        block = build_block(i, args, layout, idx, use_engram)
        t1 = time.time()
        if block.engram is not None:
            h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :])
        h, pre_mix = block(h, 0, pre_mix, None)
        torch.cuda.synchronize()
        print(f"layer {i:2d}: load {t1 - t0:5.1f} s, run {time.time() - t1:5.1f} s, |h| {h.float().norm().item():.4e}", flush=True)
        if out is not None:
            hf = h[0].float().cpu().numpy()
            np.save(f"{out}/h_L{i:02d}.npy", hf.astype(np.float16) if np.abs(hf).max() < 6e4 else hf)
            np.save(f"{out}/premix_L{i:02d}.npy", pre_mix[0].float().cpu().numpy())
        if i < n_layers - 1:
            del block
        torch.cuda.empty_cache()
    if n_layers < args.n_layers:
        return None
    h = block.hc_pre(h, pre_mix)
    norm = M.RMSNorm(args.dim, args.norm_eps)
    norm.weight.copy_(idx.get("norm.weight").to(norm.weight.dtype))
    return Exl3Head(idx, args.dim, args.vocab_size)(norm(h))[0]  # [S, vocab] fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chat", help="one user message, chat mode (no thinking), via the checkpoint's encoding/encoding.py")
    g.add_argument("--think", help="one user message, thinking mode")
    g.add_argument("--text", help="raw text (BOS prepended)")
    g.add_argument("--ids", help=".npy of token ids")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", type=int, default=0, help="stop after this many blocks (0: all)")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--gen", type=int, default=0)
    a = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    tokenizer = None
    if os.path.exists(os.path.join(a.model, "tokenizer.json")):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model)
    elif not (a.ids and a.no_engram and not a.gen):
        sys.exit("tokenizer.json not found: only --ids with --no-engram (and no --gen) works without it")
    if a.ids:
        ids = np.load(a.ids).astype(np.int64).tolist()
    else:
        if a.text is not None:
            prompt = tokenizer.bos_token + a.text
        else:
            sys.path.insert(0, os.path.join(a.model, "encoding"))
            from encoding import encode_messages
            msg = [{"role": "user", "content": a.chat if a.chat is not None else a.think}]
            prompt = encode_messages(msg, thinking_mode="chat" if a.chat is not None else "thinking")
            if isinstance(prompt, tuple):
                prompt = prompt[0]
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"{len(ids)} prompt tokens")

    idx = TensorIndex(a.model)
    use_engram = not a.no_engram
    if use_engram and not idx.has("layers.1.engram.embed.weight"):
        sys.exit("Engram tables not found (DeepSeek-V4.1-Flash-1017 shards 47 / 48); download them or pass --no-engram")
    args = make_args(len(ids) + a.gen)
    n_layers = a.layers or args.n_layers
    check_exl3(idx, args)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        np.save(f"{a.out}/ids.npy", np.array(ids, dtype=np.int64))

    t0 = time.time()
    logits = forward(torch.tensor([ids]), args, idx, tokenizer, use_engram, n_layers, a.out)
    if logits is None:
        print(f"stopped after {n_layers} blocks ({time.time() - t0:.0f} s)")
        return
    lse = torch.logsumexp(logits, -1)
    nxt = torch.tensor(ids[1:])
    nll = (lse[:-1] - logits[:-1].gather(1, nxt.unsqueeze(1)).squeeze(1))
    top = logits.topk(8, -1)
    print(f"forward {time.time() - t0:.0f} s; prompt ppl {math.exp(nll.mean().item()):.4f}; next token "
          f"{top.indices[-1, 0].item()} {tokenizer.decode([top.indices[-1, 0].item()]) if tokenizer else ''!r}")
    if a.out:
        np.save(f"{a.out}/logits_last.npy", logits[-1].cpu().numpy())
        np.save(f"{a.out}/top8_ids.npy", top.indices.cpu().numpy())
        np.save(f"{a.out}/top8_vals.npy", top.values.cpu().numpy())
        np.save(f"{a.out}/lse.npy", lse.cpu().numpy())
        np.save(f"{a.out}/nll.npy", nll.cpu().numpy())
        json.dump({"model": a.model, "n_tokens": len(ids), "engram": use_engram,
                   "act_sim": not kernel.NO_ACT_SIM}, open(f"{a.out}/meta.json", "w"), indent=1)
    seq = list(ids)
    for _ in range(a.gen):
        seq.append(int(logits[-1].argmax()))
        print(repr(tokenizer.decode(seq[len(ids):])), flush=True)
        if seq[-1] == tokenizer.eos_token_id:
            break
        logits = forward(torch.tensor([seq]), args, idx, tokenizer, use_engram, n_layers, None)


if __name__ == "__main__":
    main()
