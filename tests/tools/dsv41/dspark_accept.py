"""DSpark drafter acceptance on a recorded target sequence (teacher forced). With the exllamav3 venv:
  dspark_accept.py --model DIR --ref REFDIR --prompt N --out acc.npz [--quant OUTPUT DIR OF quantize_dspark.py]

Inputs: --ref DIR (dsv41_ref.py --out dump of the whole sequence: ids.npy, h_L36..h_L38.npy), --prompt N (tokens of the prompt; the
rest is the target's own greedy continuation). For every position t >= N the drafter gets the taps of position t and the next token
ids[t+1], drafts 5 tokens by greedy markov-chained argmax, and they are compared with ids[t+2..t+6] (the target's greedy path)."""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsv41_ref as R  # noqa: E402
M = R.M

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="the EXL3 checkpoint directory (official mtp.* tensors)")
ap.add_argument("--ref", required=True)
ap.add_argument("--prompt", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--quant", default="", help="directory with the EXL3 drafter tensors (quantize_dspark.py)")
ap.add_argument("--steps", type=int, default=0, help="limit the number of evaluated positions")
a = ap.parse_args()

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
ids = np.load(f"{a.ref}/ids.npy").tolist()
S, P = len(ids), a.prompt
idx = R.TensorIndex(a.model)
args = R.make_args(S + 16)
args.temperature = 0.0
args.n_mtp_layers = 3
M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
M.shared_attn = M.SharedAttentionRuntime()
tap_layers = list(args.dspark_target_layer_ids)
print("tap layers", tap_layers, "block", args.dspark_block_size, "seq", S, "prompt", P, flush=True)

def load_h(layer):
    return torch.from_numpy(np.load(f"{a.ref}/h_L{layer:02d}.npy")).float()   # [S, hc, dim]
taps = torch.cat([load_h(l - 1).mean(dim=1) for l in tap_layers], dim=-1).to("cuda", torch.bfloat16)   # [S, 3 * dim]: the stream entering layer l

# ----- the drafter blocks

idxq = R.TensorIndex(a.quant) if a.quant else None

def dense_to_bf16(mod, key, in_f, out_f):
    w = R.exl3_linear(idxq, key, in_f, out_f).get_weight_tensor()   # [in, out] fp16
    with torch.no_grad():
        mod.weight = nn.Parameter(w.t().to(torch.bfloat16).contiguous(), requires_grad=False)
    mod.scale = None

def quantize_block(b, k):
    p = f"mtp.{k}"
    d, qr, hd, nh = args.dim, args.q_lora_rank, args.head_dim, args.n_heads
    dense_to_bf16(b.attn.wq_a, f"{p}.attn.wq_a", d, qr)
    dense_to_bf16(b.attn.wq_b, f"{p}.attn.wq_b", qr, nh * hd)
    dense_to_bf16(b.attn.wkv, f"{p}.attn.wkv", d, hd)
    dense_to_bf16(b.attn.wo_b, f"{p}.attn.wo_b", args.o_groups * args.o_lora_rank, d)
    in_g = nh * hd // args.o_groups
    ws = [R.exl3_linear(idxq, f"{p}.attn.wo_a.slice.{g}", in_g, args.o_lora_rank).get_weight_tensor().t() for g in range(args.o_groups)]
    with torch.no_grad():
        b.attn.wo_a.weight = nn.Parameter(torch.cat(ws, 0).to(torch.bfloat16).contiguous(), requires_grad=False)
    if k == 0:
        dense_to_bf16(b.main_proj, f"{p}.main_proj", d * len(args.dspark_target_layer_ids), d)
    ffn = b.ffn
    lim = args.swiglu_limit
    def expert(key, lazy):
        wrap = R.Exl3Lazy if lazy else R.Exl3Dense
        return R.Exl3Expert(wrap(R.exl3_linear(idxq, f"{key}.w1", d, args.moe_inter_dim)),
                            wrap(R.exl3_linear(idxq, f"{key}.w2", args.moe_inter_dim, d)),
                            wrap(R.exl3_linear(idxq, f"{key}.w3", d, args.moe_inter_dim)), lim)
    ffn.shared_experts = expert(f"{p}.ffn.shared_experts", False)
    ffn.experts = nn.ModuleList([expert(f"{p}.ffn.experts.{e}", True) for e in range(args.dspark_n_routed_experts)])
    torch.cuda.empty_cache()

blocks = nn.ModuleList()
t0 = time.time()
for k in range(args.n_mtp_layers):
    b = M.DSparkBlock(args.n_layers + k, args)
    blocks.append(b)
    p = f"mtp.{k}"
    missing = []
    with torch.no_grad():
        for name, param in b.named_parameters():
            if name.startswith(("embed.", "head.")) or name == "attn.wo_a.weight":   # wo_a: dequantized below
                continue
            key = f"{p}.{name}"
            if not idx.has(key):
                missing.append(key); continue
            src = idx.get(key, "cuda")
            if src.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                if param.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                    param.view(torch.uint8).copy_(src.view(torch.uint8))
                else:
                    raise RuntimeError((key, src.dtype, param.dtype))
            elif param.dtype == torch.float4_e2m1fn_x2:
                param.view(torch.uint8).copy_(src.view(torch.uint8))
            else:
                assert src.shape == param.shape, (key, src.shape, param.shape)
                param.copy_(src.to(param.dtype))
    # wo_a is bf16 in the model but FP8 + block scales in this checkpoint
    wk = f"{p}.attn.wo_a.weight"
    if idx.has(wk) and idx.has(f"{p}.attn.wo_a.scale") and idx.meta[wk]["dtype"].startswith("F8"):
        w = idx.get(wk).float()
        s = idx.get(f"{p}.attn.wo_a.scale").float()
        bs = M.fp8_block_size
        s = s.repeat_interleave(bs, 0)[:w.shape[0]].repeat_interleave(bs, 1)[:, :w.shape[1]]
        with torch.no_grad():
            b.attn.wo_a.weight.copy_((w * s).to(torch.bfloat16))
        missing = [m for m in missing if m != wk]
    if a.quant:
        quantize_block(b, k)
    print(f"block {k}: loaded, missing {missing}", flush=True)
print(f"drafter loaded in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

class Emb(nn.Module):
    def __init__(s, w): super().__init__(); s.w = w
    def forward(s, x): return nn.functional.embedding(x, s.w)
class Head(nn.Module):
    def __init__(s, h): super().__init__(); s.h = h
    def forward(s, x, full_logits=False): return s.h(x)
embed_w = idx.get("embed.weight").to(torch.bfloat16)
head = R.Exl3Head(idx, args.dim, args.vocab_size)
for b in blocks:
    object.__setattr__(b, "embed", Emb(embed_w))
    object.__setattr__(b, "head", Head(head))
    b.eval()

@torch.inference_mode()
def spec(input_ids, main_hidden, start_pos):
    h, main_x = blocks[0].forward_embed(main_hidden, input_ids)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
    for layer in blocks:
        h, pre_mix = layer(h, start_pos, pre_mix, main_x)
    if start_pos == 0:
        return None
    return blocks[-1].forward_head(h, pre_mix, input_ids)

@torch.inference_mode()
def run():
    # seed the window caches with the prompt's taps (positions 0..P-1)
    spec(torch.tensor([ids[P]]), taps[None, :P], 0)
    res = []
    last = S - 1 - args.dspark_block_size - 1
    if a.steps:
        last = min(last, P + a.steps - 1)
    t0 = time.time()
    for t in range(P, last + 1):
        out, logits, conf = spec(torch.tensor([ids[t + 1]]), taps[None, t:t + 1], t)
        d = out[0, 1:].tolist()
        truth = ids[t + 2:t + 2 + args.dspark_block_size]
        m = [int(x == y) for x, y in zip(d, truth)]
        res.append((t, m, conf[0].float().tolist(), d))
        if (t - P) % 50 == 0:
            acc = np.mean([sum(np.cumprod(r[1])) for r in res])
            print(f"t {t}: {len(res)} steps, mean accepted {acc:.2f}, {(time.time() - t0) / len(res):.2f} s/step", flush=True)
    return res

res = run()
m = np.array([r[1] for r in res])
conf = np.array([r[2] for r in res])
acc = np.cumprod(m, axis=1)
print(f"steps {len(res)}")
print("per-position match (unconditional):", m.mean(0).round(3).tolist())
print("prefix >= j:", acc.mean(0).round(3).tolist())
print(f"mean accepted drafts per round {acc.sum(1).mean():.3f}; tokens per round {1 + acc.sum(1).mean():.3f}")
np.savez(a.out, m=m, conf=conf, drafts=np.array([r[3] for r in res]), t=np.array([r[0] for r in res]))
