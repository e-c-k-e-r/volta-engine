"""Golden for the H3 text conditioning with one reference image (FL2VA presentation '<Picture 1>: <vision block> prompt').
FP32: HF vision tower -> image features + DeepStack, Qwen3-VL get_rope_index positions, interleaved mRoPE, 50-layer text tower (streamed), with the
first 3 layers cross-checked against HF's Qwen3VLTextModel. Run in .venv-1cat13 (torchvision for the processor), one GPU.
usage: h3_encoder_mm_golden.py OUT.safetensors [--w 320] [--h 192] [--layers 50] [--prompt TEXT]
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dir", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/text_encoder")
ap.add_argument("--proc", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/processor")
ap.add_argument("--tok", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/tokenizer")
ap.add_argument("--w", type=int, default=320)
ap.add_argument("--h", type=int, default=192)
ap.add_argument("--layers", type=int, default=50)
ap.add_argument("--prompt", default="The camera slowly pushes in as the dog turns its head.")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLTextModel

idx = json.load(open(os.path.join(a.dir, "model.safetensors.index.json")))["weight_map"]
handles = {}
def get(name):
    f = idx[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.dir, f), framework="pt", device="cpu")
    return handles[f].get_tensor(name)

dev = "cuda"
rng = np.random.default_rng(a.seed)
img = Image.fromarray(rng.integers(0, 255, size=(6, 10, 3), dtype=np.uint8)).resize((a.w, a.h), Image.Resampling.BICUBIC)
pix, grid = (lambda o: (o["pixel_values"], o["image_grid_thw"]))(AutoImageProcessor.from_pretrained(a.proc)(images=[img], return_tensors="pt"))
cfg_all = AutoConfig.from_pretrained(a.dir)
vm = Qwen3VLVisionModel(cfg_all.vision_config).to(dev, torch.float32).eval()
vm.load_state_dict({n[len("model.visual."):]: get(n).float() for n in idx if n.startswith("model.visual.")})
with torch.inference_mode():
    out = vm(pix.to(dev).float(), grid_thw=grid.to(dev))
merged = (out[0] if isinstance(out, tuple) else out.pooler_output).float()
ds = [d.float() for d in (out[1] if isinstance(out, tuple) else out.deepstack_features)]
n_img = merged.shape[0]

tok = AutoTokenizer.from_pretrained(a.tok)
t = lambda s: tok(s, add_special_tokens=False)["input_ids"]
VS, VE, PAD = tok.convert_tokens_to_ids("<|vision_start|>"), tok.convert_tokens_to_ids("<|vision_end|>"), tok.convert_tokens_to_ids("<|image_pad|>")
ids = t("<Picture 1>: ") + [VS] + [PAD] * n_img + [VE] + t(a.prompt)
ids_t = torch.tensor(ids)
S = len(ids)
print("tokens", S, "image tokens", n_img, "grid", grid.tolist())

# get_rope_index (single image)
pos = torch.zeros(3, S, dtype=torch.long)
cur, i = 0, 0
mm = (ids_t == PAD).int().tolist()
gh, gw = grid[0, 1].item() // 2, grid[0, 2].item() // 2
while i < S:
    j = i
    while j < S and mm[j] == mm[i]:
        j += 1
    if mm[i] == 0:
        pos[:, i:j] = torch.arange(j - i) + cur
        cur += j - i
    else:
        n = torch.arange(j - i)
        pos[0, i:j] = cur
        pos[1, i:j] = cur + n // gw
        pos[2, i:j] = cur + n % gw
        cur += max(gh, gw)
    i = j

tc = json.load(open(os.path.join(a.dir, "config.json")))["text_config"]  # raw json: transformers versions differ in where rope_theta lives
H, nh, nkv, hd, eps, theta = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"], tc["rms_norm_eps"], tc["rope_theta"]
def rms(x, w): x = x.float(); return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()
inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=dev) / hd))  # 64
axis = torch.zeros(64, dtype=torch.long, device=dev)
for f in range(60):
    axis[f] = 1 if f % 3 == 1 else (2 if f % 3 == 2 else 0)
ang = pos.to(dev).float()[axis, :].T * inv[None]  # [S, 64]: freq f uses the position of its axis
cos, sin = torch.cos(ang), torch.sin(ang)

def layer(x, i):
    p = f"model.language_model.layers.{i}."
    w = lambda n: get(p + n).to(dev).float()
    h = rms(x, w("input_layernorm.weight"))
    q = (h @ w("self_attn.q_proj.weight").t()).view(S, nh, hd); k = (h @ w("self_attn.k_proj.weight").t()).view(S, nkv, hd); v = (h @ w("self_attn.v_proj.weight").t()).view(S, nkv, hd)
    q, k = rms(q, w("self_attn.q_norm.weight")), rms(k, w("self_attn.k_norm.weight"))
    def rot(z):
        a_, b_ = z[..., :hd // 2], z[..., hd // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat((a_ * c - b_ * s, b_ * c + a_ * s), -1)
    q, k = rot(q), rot(k)
    k, v = k.repeat_interleave(nh // nkv, 1), v.repeat_interleave(nh // nkv, 1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1).reshape(S, nh * hd)
    x = x + o @ w("self_attn.o_proj.weight").t()
    h = rms(x, w("post_attention_layernorm.weight"))
    return x + (F.silu(h @ w("mlp.gate_proj.weight").t()) * (h @ w("mlp.up_proj.weight").t())) @ w("mlp.down_proj.weight").t()

emb = get("model.language_model.embed_tokens.weight")
x = emb[ids_t].to(dev).float()
mask = (ids_t == PAD).to(dev)
x[mask] = merged.half().float()  # the reference rounds image embeds to the encoder dtype (FP16)
with torch.inference_mode():
    for li in range(a.layers):
        x = layer(x, li)
        if li < 3:
            x = x.clone(); x[mask] = x[mask] + ds[li].half().float()
        if li == 2:
            # cross-check against HF's text model on the same inputs (3 layers, final norm applied by HF)
            tcfg = cfg_all.text_config; tcfg.num_hidden_layers = 3
            m = Qwen3VLTextModel(tcfg).to(dev, torch.float32).eval()
            m.load_state_dict({n[len("model.language_model."):]: get(n).float() for n in idx if n.startswith("model.language_model.") and (".layers." not in n or int(n.split(".layers.")[1].split(".")[0]) < 3)}, strict=False)
            e = emb[ids_t].to(dev).float()[None]; e[0, mask] = merged.half().float()
            r = m(inputs_embeds=e, position_ids=pos.to(dev)[:, None, :], visual_pos_masks=mask[None], deepstack_visual_embeds=[d.half().float() for d in ds]).last_hidden_state[0]
            mine = rms(x, get("model.language_model.norm.weight").to(dev))
            print(f"HF 3-layer cosine {F.cosine_similarity(r.flatten(), mine.flatten(), dim=0).item():.6f}")
            del m
print("hidden", tuple(x.shape), "absmax", float(x.abs().max()))
save_file({"ids": ids_t.int(), "pixel_values": pix.float().contiguous(), "grid": grid.int().contiguous(), "hidden": x.cpu().contiguous()}, a.out)
print("saved", a.out)
