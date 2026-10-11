"""FP32 reference of the MiniMax-H3 DiT forward (no vLLM dependency).

Loads the Comfy INT8 ConvRot checkpoint block by block, restores effective weights
(INT8 * row scale, then the 256-group Hadamard that the checkpoint folded into W), and runs
the packed forward of 1Cat's MiniMaxH3DiTModel in FP32. It is the oracle for the C++ port.
Semantics follow vllm/model_executor/models/minimax_h3/{transformer,modulation,ops}.py.
"""
import json, math
import torch
import torch.nn.functional as F
from safetensors import safe_open


def convrot(x, group=256):
    """x @ H per 256-channel group (Kronecker power of H4, orthonormal, symmetric)."""
    shape = x.shape
    v = x.float().reshape(-1, group)
    stride = 1
    while stride < group:
        a, b, c, d = v.reshape(-1, group // (4 * stride), 4, stride).unbind(-2)
        v = torch.stack((a + b + c - d, a + b - c + d, a - b + c + d, -a + b + c + d), -2)
        stride *= 4
    return (v.reshape(shape) / 16)


def rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()


class H3Ref:
    def __init__(self, ckpt, cfg, device="cuda"):
        self.f = safe_open(ckpt, framework="pt", device="cpu")
        self.cfg = cfg
        self.dev = device
        self.H = cfg["hidden_size"]
        self.nh = cfg["num_attention_heads"]
        self.hd = cfg["attention_head_dim"]
        self.eps = cfg["norm_eps"]
        self.qkeps = cfg["qk_norm_eps"]
        self.cache = {}

    def t(self, name):
        return self.f.get_tensor(name).to(self.dev)

    def weight(self, prefix):
        """Effective [N, K] FP32 weight of a (possibly INT8 ConvRot) linear."""
        w = self.f.get_tensor(prefix + ".weight")
        if w.dtype != torch.int8:
            return w.to(self.dev).float()
        scale = self.t(prefix + ".weight_scale").float().reshape(-1, 1)
        w = w.to(self.dev).float() * scale
        q = bytes(self.f.get_tensor(prefix + ".comfy_quant").tolist())
        meta = json.loads(q.decode())
        if meta.get("convrot", False):
            w = convrot(w, meta.get("convrot_groupsize", 256))
        return w

    # --- pieces -------------------------------------------------------------------------
    def attention(self, prefix, x, rope, rope_cols=96):
        """x [U, H] fp32. rope: (cos, sin) [U, 48] or None. Full bidirectional attention."""
        U = x.shape[0]
        qkv = x @ self.weight(prefix + ".qkv_proj").t()
        q, k, v = qkv.split(self.nh * self.hd, dim=-1)
        q = q.view(U, self.nh, self.hd)
        k = k.view(U, self.nh, self.hd)
        v = v.view(U, self.nh, self.hd)
        q = rms(q, self.t(prefix + ".q_norm.weight"), self.qkeps)
        k = rms(k, self.t(prefix + ".k_norm.weight"), self.qkeps)
        if rope is not None:
            cos, sin = rope
            half = rope_cols // 2

            def apply(z):
                a, b = z[..., :half], z[..., half:rope_cols]
                c, s = cos.unsqueeze(1), sin.unsqueeze(1)
                return torch.cat((a * c - b * s, b * c + a * s, z[..., rope_cols:]), -1)

            q, k = apply(q), apply(k)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), scale=self.hd ** -0.5)
        o = o.squeeze(0).transpose(0, 1).reshape(U, self.nh * self.hd)
        return o @ self.weight(prefix + ".out_proj").t()

    def mlp(self, prefix, x):
        h = x @ self.weight(prefix + ".fc1").t()
        g, u = h.chunk(2, dim=-1)
        return (F.silu(g) * u) @ self.weight(prefix + ".fc2").t()

    def refiner(self, text):
        x = text
        for i in range(self.cfg["token_refiner_num_layers"]):
            p = f"token_refiner.blocks.{i}"
            x = x + self.attention(p + ".attn", rms(x, self.t(p + ".norm1.weight"), self.eps), None)
            x = x + self.mlp(p + ".mlp", rms(x, self.t(p + ".norm2.weight"), self.eps))
        return rms(x, self.t("token_refiner.final_norm.weight"), self.cfg["final_norm_eps"])

    def time_embed(self, uniq):
        table = self.t("adaln_t_table")
        pos = uniq.float().clamp(0, 1) * (table.shape[0] - 1)
        lo = pos.floor().long().clamp(max=table.shape[0] - 2)
        frac = (pos - lo).unsqueeze(1)
        return torch.lerp(table[lo], table[lo + 1], frac)

    def adaln(self, prefix, temb):
        w = self.t(prefix + ".linear.weight").float()
        b = self.t(prefix + ".linear.bias").float()
        return temb @ w.t() + b

    # --- full forward --------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, inp, dump=None):
        """inp keys: x [S,96], audio_x [S,32], prompt_embeds [T,5120], img_position_ids [S,3],
        img_pos, audio_pos, text_pos (long), update_mask (bool, over img_pos), token_tags [S],
        unique_timesteps [M], inverse_indices [S], used (int). Returns (video[len(img_pos),96],
        audio[len(audio_pos),32])."""
        d = self.dev
        g = lambda k: inp[k].to(d)
        S = g("x").shape[0]
        used = int(inp["used"])
        img_pos, audio_pos, text_pos = g("img_pos").long(), g("audio_pos").long(), g("text_pos").long()
        vemb = g("x")[img_pos].float() @ self.t("video_patch_proj.weight").float().t() + self.t("video_patch_proj.bias").float()
        aemb = g("audio_x")[audio_pos].float() @ self.t("audio_patch_proj.weight").float().t() + self.t("audio_patch_proj.bias").float()
        text = g("prompt_embeds").float() @ self.weight("condition_proj").t() + self.t("condition_proj.bias").float()
        text = self.refiner(text)
        x = torch.zeros(S, self.H, device=d)
        x.index_add_(0, text_pos, text)
        x.index_add_(0, img_pos, vemb)
        x.index_add_(0, audio_pos, aemb)
        if dump is not None:
            dump["embed"] = x[:used].clone()
        # rope table: [t,h,w] x 16 frequencies
        inv = self.t("rope.inv_freq").float()
        per = g("img_position_ids").float().unsqueeze(-1) * inv.view(1, 1, -1)  # [S,3,16]
        half = per.reshape(S, 48)
        cos, sin = torch.cos(half)[:used], torch.sin(half)[:used]
        temb = self.time_embed(g("unique_timesteps"))
        comb = (g("inverse_indices").long() * 3 + g("token_tags").long().clamp(min=0))[:used]
        x = x[:used]
        for i in range(self.cfg["num_layers"]):
            p = f"blocks.{i}"
            a = self.adaln(p + ".adaln_proj", temb).view(-1, 6 * self.H)  # [M*3, 6H]
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = a.chunk(6, -1)
            h = rms(x, self.t(p + ".norm1.weight"), self.eps)
            h = h * (1 + scale_a[comb]) + shift_a[comb]
            h = self.attention(p + ".attn", h, (cos, sin))
            x = x + gate_a[comb] * h
            h = rms(x, self.t(p + ".norm2.weight"), self.eps)
            h = h * (1 + scale_m[comb]) + shift_m[comb]
            h = self.mlp(p + ".mlp", h)
            x = x + gate_m[comb] * h
            if dump is not None and (i in (0, 1, self.cfg["num_layers"] // 2) or i == self.cfg["num_layers"] - 1):
                dump[f"block{i}"] = x.clone()
        fa = self.adaln("final_layer.adaln_proj", temb).view(-1, 2 * self.H)
        shift, scale = fa.chunk(2, -1)
        inv_idx = g("inverse_indices").long()[:used]
        h = rms(x, self.t("final_layer.norm.weight"), self.cfg["final_norm_eps"])
        h = h * (1 + scale[inv_idx]) + shift[inv_idx]
        video = h @ self.t("final_layer.video_out.weight").float().t() + self.t("final_layer.video_out.bias").float()
        audio = h @ self.t("final_layer.audio_out.weight").float().t() + self.t("final_layer.audio_out.bias").float()
        if dump is not None:
            dump["final_hidden"] = h.clone()
        v = video[img_pos] * g("update_mask").float().unsqueeze(-1)
        a = audio[audio_pos]
        return v, a


# --- scheduler pieces (match scheduling_minimax_h3_euler_ancestral / time_request) ----------
def shifted_sigmas(base, shift):
    return [shift * u / (1 + (shift - 1) * u) for u in base]


def euler_step(state, v, t, sigma_curr, sigma_next):
    """rectified-flow x0 = x + (1-t) v ; eta-0 Euler: r*x + (1-r)*x0 (t = 1 - sigma_curr)."""
    x0 = state + (1 - t) * v
    if sigma_curr == 0.0:
        return state
    r = sigma_next / sigma_curr
    return r * state + (1 - r) * x0
