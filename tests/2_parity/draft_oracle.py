# Independent torch oracle for the DFlash2 drafter backbone. Semantics are transcribed from exllamav3
# (architecture/dflash2.py, modules/arch_specific/dflash2.py) and 1Cat-vLLM (qwen3_dflash2.py), NOT from the
# engine, so it can catch errors a self-written reference would share (it found the interleaved-RoPE bug).
#
#   EXL3_DUMP_DRAFT=<round> ./build/bench_accept --prompts 1 --tokens 96     # writes /tmp/draft_dump.bin
#   <exllamav3>/.venv/bin/python tests/2_parity/draft_oracle.py              # per-row cosine vs the engine
#
# Expect cos >= ~0.998 per row against the 5-bit drafter (BF16 oracle weights, FP32 math). VAR= takes
# space-free flags (ropeint, convswap, causal, norow0, plus1) that flip one convention to see which one the
# engine implements. Paths below are this machine's.
# Independent from-scratch torch version of the DFlash2 drafter (semantics from exllamav3 dflash2.py and
# 1Cat-vLLM qwen3_dflash2.py), compared against the engine's dumped draft hidden for the same round.
import json, struct, sys, torch, numpy as np
from safetensors import safe_open
dev = "cuda:0"
import os
VAR = os.environ.get("VAR","")
D = "/home/mrq/Documents/models/Qwen3.8-27B-DFlash2/model.safetensors"
E = "/home/mrq/Documents/models/Qwen3.8-27B-exl3/model-00001-of-00002.safetensors"
raw = open("/tmp/draft_dump.bin", "rb").read()
pos, anchor, H, cap = struct.unpack("4i", raw[:16])
taps = np.frombuffer(raw[16:16 + pos * 5 * H * 2], dtype=np.float16).reshape(pos, 5 * H)
hid = np.frombuffer(raw[16 + pos * 5 * H * 2:], dtype=np.float16).reshape(-1, H)
print("pos", pos, "anchor", anchor, "H", H)
W = {}
with safe_open(D, "pt", device=dev) as f:
    for k in f.keys(): W[k] = f.get_tensor(k).float()
with safe_open(E, "pt", device=dev) as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight")
x_ids = [anchor] + [248070] * 7
x = emb[x_ids].float()
L, NH, NKV, HD, TAPS, GS, EPS, THETA = 5, 32, 8, 128, 2, 16, 1e-6, 1e7
G = H // GS
def rms(v, w): return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS) * ((1 + w) if 'plus1' in VAR else w)
def rope(t, p0):  # t [n, heads, hd]
    n = t.shape[0]
    inv = THETA ** (-torch.arange(0, HD, 2, device=dev).float() / HD)
    ang = (torch.arange(p0, p0 + n, device=dev).float()[:, None]) * inv[None]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    if "ropeint" in VAR:
        a, b = t[..., 0::2], t[..., 1::2]
        o = torch.stack([a * c - b * s, b * c + a * s], -1)
        return o.flatten(-2)
    a, b = t[..., :HD // 2], t[..., HD // 2:]
    return torch.cat([a * c - b * s, b * c + a * s], -1)
def gconv(h, coef, base):  # h [n,H]; coef [n,TAPS,G]; base [TAPS,H]
    n = h.shape[0]
    blocks = h.view(n, G, GS)
    c = base.view(1, TAPS, G, GS) + coef.unsqueeze(-1)
    if 'convswap' in VAR: c = c.flip(1)
    out = c[:, 0] * blocks
    for tp in range(1, TAPS):
        sh = torch.cat([torch.zeros(tp, G, GS, device=dev), blocks[:-tp]], 0)
        pm = (torch.arange(n, device=dev) >= tp).view(-1, 1, 1)
        out = out + c[:, tp] * sh * pm
    return out.reshape(n, H)
# context stage: fc -> hidden_norm -> per layer k_norm + rope, v
ctx = torch.from_numpy(taps.astype(np.float32)).to(dev) @ W["fc.weight"].T
ctxn = rms(ctx, W["hidden_norm.weight"])
KC, VC = [], []
for l in range(L):
    p = f"layers.{l}.self_attn."
    k = (ctxn @ W[p + "k_proj.weight"].T).view(pos, NKV, HD)
    k = rope(rms(k, W[p + "k_norm.weight"]), 0)
    KC.append(k); VC.append((ctxn @ W[p + "v_proj.weight"].T).view(pos, NKV, HD))
hidden, residual = x, None
for l in range(L):
    pre = f"layers.{l}."
    if residual is None:
        residual = hidden; hidden = rms(hidden, W[pre + "input_layernorm.weight"])
    else:
        residual = hidden + residual; hidden = rms(residual, W[pre + "input_layernorm.weight"])
    co = (hidden @ W[pre + "attention_conv.kernel_projection.weight"].T).view(8, 2, TAPS, G)
    hidden = gconv(hidden, co[:, 0], W[pre + "attention_conv.base_kernel"][0])
    p = pre + "self_attn."
    q = (hidden @ W[p + "q_proj.weight"].T).view(8, NH, HD)
    k = (hidden @ W[p + "k_proj.weight"].T).view(8, NKV, HD)
    v = (hidden @ W[p + "v_proj.weight"].T).view(8, NKV, HD)
    q = rope(rms(q, W[p + "q_norm.weight"]), pos)
    k = rope(rms(k, W[p + "k_norm.weight"]), pos)
    Kall = torch.cat([KC[l], k], 0); Vall = torch.cat([VC[l], v], 0)       # [pos+8, NKV, HD]
    rep = NH // NKV
    Kx = Kall.repeat_interleave(rep, dim=1); Vx = Vall.repeat_interleave(rep, dim=1)
    sc = torch.einsum("qhd,khd->hqk", q, Kx) / HD ** 0.5
    if "causal" in VAR:
        m = torch.triu(torch.ones(8, 8, device=dev), 1).bool()
        sc[:, :, pos:] = sc[:, :, pos:].masked_fill(m, float("-inf"))
    if "norow0" in VAR:
        sc[:, :, pos] = float("-inf")
    att = torch.einsum("hqk,khd->qhd", sc.softmax(-1), Vx).reshape(8, NH * HD)
    hidden = att @ W[p + "o_proj.weight"].T
    hidden = gconv(hidden, co[:, 1], W[pre + "attention_conv.base_kernel"][1])
    residual = hidden + residual; hidden = rms(residual, W[pre + "post_attention_layernorm.weight"])
    co2 = (hidden @ W[pre + "mlp_conv.kernel_projection.weight"].T).view(8, 2, TAPS, G)
    hidden = gconv(hidden, co2[:, 0], W[pre + "mlp_conv.base_kernel"][0])
    g = hidden @ W[pre + "mlp.gate_proj.weight"].T; u = hidden @ W[pre + "mlp.up_proj.weight"].T
    hidden = (torch.nn.functional.silu(g) * u) @ W[pre + "mlp.down_proj.weight"].T
    hidden = gconv(hidden, co2[:, 1], W[pre + "mlp_conv.base_kernel"][1])
out = rms(hidden + residual, W["norm.weight"])
ref = out.cpu().numpy(); eng = hid.astype(np.float32)
for r in range(8):
    a, b = ref[r], eng[r]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print("row %d cos %.5f  rel-L2 %.4f  |ref| %.1f |eng| %.1f" % (r, cos, np.linalg.norm(a - b) / np.linalg.norm(a), np.linalg.norm(a), np.linalg.norm(b)))
