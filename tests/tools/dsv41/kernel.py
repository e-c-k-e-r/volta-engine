"""Plain-torch stand-ins for the tilelang kernels of DeepSeek's V4.1 reference (inference/kernel.py), so model.py runs on
SM70. Same signatures and the same quantization arithmetic: block absmax, power-of-two (UE8M0) or E4M3 scales, E4M3 / E2M1
round-to-nearest-even. GEMMs dequantize and multiply in FP32 (the originals accumulate per K block in FP32, so results agree to
rounding); sparse_attn is the gather + softmax with the per-head sink in the denominator, in FP32 (the original rounds P to
BF16 before the PV product)."""
import os
from typing import Optional

import torch

# DSV41_NO_ACT_SIM=1 turns the in-place quantize-dequantize of activations and caches into identity (for logic checks: the
# 3-bit fp8 mantissa turns one-ulp GEMM differences between prefill and decode into percent-level ones)
NO_ACT_SIM = os.environ.get("DSV41_NO_ACT_SIM", "0") == "1"
FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(v)) for positive normal v, as fast_round_scale computes it from the exponent and mantissa bits."""
    m, e = torch.frexp(v)  # v = m * 2^e, m in [0.5, 1)
    return torch.ldexp(torch.ones_like(v), e - (m == 0.5).to(e.dtype))


def _fp4_round(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value (|x| <= 6 already), ties to the even mantissa."""
    a = x.abs()
    r = torch.where(a <= 0.25, 0.0,
        torch.where(a < 0.75, 0.5,
        torch.where(a <= 1.25, 1.0,
        torch.where(a < 1.75, 1.5,
        torch.where(a <= 2.5, 2.0,
        torch.where(a < 3.5, 3.0,
        torch.where(a <= 5.0, 4.0, 6.0)))))))
    return torch.copysign(r, x)


def _to_scale_dtype(s: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    return s if scale_dtype == torch.float32 else s.to(scale_dtype)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    N = x.size(-1)
    assert N % block_size == 0
    if inplace and NO_ACT_SIM:
        return x
    xb = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), _to_scale_dtype(s.squeeze(-1), scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    if inplace and NO_ACT_SIM:
        return x
    xb = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2**-126) / FP4_MAX)
    q = _fp4_round((xb / s).clamp(-FP4_MAX, FP4_MAX))
    if not inplace:
        raise NotImplementedError("packed fp4 output is not used by model.py")
    x.copy_((q * s).flatten(-2).to(x.dtype))
    return x


def _dequant_act(a: torch.Tensor, a_s: torch.Tensor, block: int) -> torch.Tensor:
    K = a.size(-1)
    return (a.float().unflatten(-1, (K // block, block)) * a_s.float().view(*a.shape[:-1], K // block, 1)).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    N, K = b.shape
    bs = b_s.float().repeat_interleave(block_size, 0)[:N].repeat_interleave(block_size, 1)
    w = b.float() * bs
    return (_dequant_act(a, a_s, block_size) @ w.t()).to(torch.get_default_dtype())


def fp4_dequant_weight(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """[N, K/2] float4_e2m1fn_x2 (low nibble first along K) with [N, K/32] scales -> [N, K] fp32."""
    u = b.view(torch.uint8)
    table = FP4_TABLE.to(u.device)
    vals = torch.stack([table[(u & 0x0F).long()], table[(u >> 4).long()]], dim=-1).flatten(-2)
    return vals * b_s.float().repeat_interleave(32, 1)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    w = fp4_dequant_weight(b, b_s)
    return (_dequant_act(a, a_s, act_block_size) @ w.t()).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk: int = 256):
    b, m, h, d = q.shape
    o = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h)
    for s0 in range(0, m, chunk):
        s1 = min(m, s0 + chunk)
        idx = topk_idxs[:, s0:s1].long()  # [b, c, k]
        valid = idx >= 0
        g = torch.gather(kv.float().unsqueeze(1).expand(b, s1 - s0, -1, -1), 2,
                         idx.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, d))  # [b, c, k, d]
        sc = torch.einsum("bchd,bckd->bchk", q[:, s0:s1].float(), g) * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(2), -torch.inf)
        mx = torch.maximum(sc.amax(-1).clamp_min(-1e30), sink)  # [b, c, h]
        p = torch.exp(sc - mx.unsqueeze(-1))
        den = p.sum(-1) + torch.exp(sink - mx)
        o[:, s0:s1] = (torch.einsum("bchk,bckd->bchd", p, g) / den.unsqueeze(-1)).to(q.dtype)
    return o


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
