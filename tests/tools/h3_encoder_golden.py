"""Golden for the H3 text conditioning: Qwen3-VL text tower, first 50 decoder layers, no final norm, plain prompt tokens (T2VA).
An FP32 layer-streaming reference computes the 50-layer hidden states; the first two layers are cross-checked against HF transformers.
Run in a venv with torch + transformers + safetensors (the exllamav3 venv works), one GPU.

usage: h3_encoder_golden.py OUT.safetensors [--prompt TEXT] [--layers 50]
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dir", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/text_encoder")
ap.add_argument("--tok", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/tokenizer")
ap.add_argument("--prompt", default="A golden retriever runs across a sunny beach while waves crash behind it, cinematic lighting.")
ap.add_argument("--layers", type=int, default=50)
ap.add_argument("--no-hf", action="store_true")
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tok)
ids = torch.tensor(tok(a.prompt, add_special_tokens=False)["input_ids"])
print("ids", ids.shape[0], ids[:8].tolist())

cfg = json.load(open(os.path.join(a.dir, "config.json")))["text_config"]
idx = json.load(open(os.path.join(a.dir, "model.safetensors.index.json")))["weight_map"]
handles = {}

def get(name):
    f = idx[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.dir, f), framework="pt", device="cpu")
    return handles[f].get_tensor(name)

dev = "cuda"
H, nh, nkv, hd, eps, theta = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"], cfg["rope_theta"]

def rms(x, w):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()

def rope_tables(S):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=dev) / hd))
    ang = torch.arange(S, dtype=torch.float32, device=dev)[:, None] * inv[None]
    return torch.cos(ang), torch.sin(ang)

def layer(x, i, cos, sin):
    p = f"model.language_model.layers.{i}."
    w = lambda n: get(p + n).to(dev).float()
    h = rms(x, w("input_layernorm.weight"))
    S = x.shape[0]
    q = (h @ w("self_attn.q_proj.weight").t()).view(S, nh, hd)
    k = (h @ w("self_attn.k_proj.weight").t()).view(S, nkv, hd)
    v = (h @ w("self_attn.v_proj.weight").t()).view(S, nkv, hd)
    q = rms(q, w("self_attn.q_norm.weight"))
    k = rms(k, w("self_attn.k_norm.weight"))
    def rot(t):
        a_, b_ = t[..., : hd // 2], t[..., hd // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat((a_ * c - b_ * s, b_ * c + a_ * s), -1)
    q, k = rot(q), rot(k)
    k = k.repeat_interleave(nh // nkv, dim=1)
    v = v.repeat_interleave(nh // nkv, dim=1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True)
    o = o[0].transpose(0, 1).reshape(S, nh * hd)
    x = x + o @ w("self_attn.o_proj.weight").t()
    h = rms(x, w("post_attention_layernorm.weight"))
    g = h @ w("mlp.gate_proj.weight").t()
    u = h @ w("mlp.up_proj.weight").t()
    return x + (F.silu(g) * u) @ w("mlp.down_proj.weight").t()

emb = get("model.language_model.embed_tokens.weight")
x = emb[ids].to(dev).float()
cos, sin = rope_tables(x.shape[0])
out = {"ids": ids.int()}
with torch.inference_mode():
    for i in range(a.layers):
        x = layer(x, i, cos, sin)
        if i == 1:
            out["hidden_l2"] = x.cpu().clone()
            if not a.no_hf:
                # cross-check the first two layers against transformers' own text tower
                from transformers import AutoConfig
                from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
                tcfg = AutoConfig.from_pretrained(a.dir).text_config
                tcfg.num_hidden_layers = 2
                m = Qwen3VLTextModel(tcfg).to(dev, torch.float32).eval()
                sd = {}
                for n in idx:
                    if n.startswith("model.language_model.") and (".layers." not in n or int(n.split(".layers.")[1].split(".")[0]) < 2) and "lm_head" not in n:
                        sd[n[len("model.language_model."):]] = get(n).float()
                missing = m.load_state_dict(sd, strict=False)
                print("hf missing", missing.missing_keys[:3], "unexpected", missing.unexpected_keys[:3])
                ref = m(input_ids=ids[None].to(dev)).last_hidden_state[0]  # HF applies the final norm
                mine = rms(x, get("model.language_model.norm.weight").to(dev))
                c = F.cosine_similarity(ref.flatten(), mine.flatten(), dim=0).item()
                print(f"HF 2-layer cosine {c:.6f}, max|d| {(ref - mine).abs().max().item():.4g}")
                del m
                torch.cuda.empty_cache()
    print("hidden", tuple(x.shape), "absmax", float(x.abs().max()), "rms", float(x.square().mean().sqrt()))
out["hidden"] = x.cpu().contiguous()
save_file({k: v.contiguous() for k, v in out.items()}, a.out)
print("saved", a.out)
