"""Self-test for the torch kernel stand-ins: a small random DeepSeek-V4.1 (model.py's default shapes, candidate selection on,
no engram / vision / DSpark) must give the same last-position logits whether a position is reached by a fresh prefill or by
decoding token by token past the sliding window, through partial compression groups and past index_topk.
Run: ~/Programs/exllamav3/.venv/bin/python tests/tools/dsv41/selftest.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel  # noqa: E402
import model as M  # noqa: E402


def quantize_fp8_blocks(w: torch.Tensor, block: int = 32):
    N, K = w.shape
    wb = w.float().view(N // block, block, K // block, block)
    s = kernel._pow2_ceil(wb.abs().amax((1, 3), keepdim=True).clamp_min(1e-12) / kernel.FP8_MAX)
    q = (wb / s).clamp(-kernel.FP8_MAX, kernel.FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(N, K), s.view(N // block, K // block).to(torch.float8_e8m0fnu)


def quantize_fp4(w: torch.Tensor):
    N, K = w.shape
    wb = w.float().view(N, K // 32, 32)
    s = kernel._pow2_ceil(wb.abs().amax(-1, keepdim=True).clamp_min(1e-12) / kernel.FP4_MAX)
    q = kernel._fp4_round((wb / s).clamp(-6, 6)).view(N, K)
    mag =(q.abs().unsqueeze(-1) == kernel.FP4_TABLE.to(q.device)[:8]).float().argmax(-1)
    codes = (mag | ((q < 0).long() << 3)).to(torch.uint8).view(N, K // 2, 2)
    packed = codes[..., 0] | (codes[..., 1] << 4)
    return packed.view(torch.float4_e2m1fn_x2), s.view(N, K // 32).to(torch.float8_e8m0fnu)


@torch.no_grad()
def init_weights(model: torch.nn.Module, gen: torch.Generator):
    for mod in model.modules():
        if isinstance(mod, M.Linear):
            out_f, in_f = mod.out_features, mod.in_features
            w = torch.randn(out_f, in_f, generator=gen, device="cuda") / in_f**0.5
            if mod.weight.dtype == torch.float8_e4m3fn:
                q, s = quantize_fp8_blocks(w)
                mod.weight.copy_(q)
                mod.scale.copy_(s)
            elif mod.weight.dtype == torch.float4_e2m1fn_x2:
                q, s = quantize_fp4(w)
                mod.weight.view(torch.uint8).copy_(q.view(torch.uint8))
                mod.scale.copy_(s)
            else:
                mod.weight.copy_(w.to(mod.weight.dtype))
        elif isinstance(mod, M.RMSNorm):
            mod.weight.copy_(1 + 0.1 * torch.randn(mod.weight.shape, generator=gen, device="cuda"))
    for name, p in model.named_parameters():
        if name.endswith(("hc_attn_fn", "hc_ffn_fn")):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * 0.02)
        elif name.endswith(("hc_attn_base", "hc_ffn_base")):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * 0.5)
        elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
            p.fill_(1.0)
        elif name.endswith("attn_sink"):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda"))
        elif name.endswith("gate.weight"):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") / p.shape[1] ** 0.5)
        elif name.endswith("gate.bias"):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * 0.1)
        elif name.endswith(("embed.weight",)) or name == "head.weight":
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda") * (1.0 if "embed" in name else p.shape[1] ** -0.5))


def unit_checks():
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16) * 3
    y = x.clone()
    kernel.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, True)
    rel = ((y.float() - x.float()).norm() / x.float().norm()).item()
    assert rel < 0.05, rel
    z = x.clone()
    kernel.fp4_act_quant(z, 16, True, scale_dtype=torch.float8_e4m3fn)
    assert ((z.float() - x.float()).norm() / x.float().norm()).item() < 0.2
    v = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.3, 5.1], device="cuda")
    assert kernel._fp4_round(v).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -0.5, 6.0]
    w = torch.randn(64, 128, device="cuda")
    q, s = quantize_fp4(w)
    deq = kernel.fp4_dequant_weight(q, s)
    ref = (kernel._fp4_round((w.view(64, 4, 32) / s.float().unsqueeze(-1)).clamp(-6, 6)) * s.float().unsqueeze(-1)).view(64, 128)
    assert torch.equal(deq, ref)
    print("unit checks ok")


def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    unit_checks()
    variant = os.environ.get("DSV41_SELFTEST", "full")
    extra = dict(index_topk=24, candidate_source_layer=3, candidate_topk_blocks=3, candidate_block_size=8)
    if variant == "window":  # sliding window only
        extra = dict(compress_ratios=(0, 0, 0, 0, 0, 0), kv_source_layers=(), index_source_layers=())
    elif variant == "nocand":  # compression and indexer, no candidate pre-filter
        extra = dict(index_topk=24)
    elif variant == "dense":  # compression with index_topk above every compressed length (selection keeps everything)
        extra = dict(index_topk=512)
    # DSV41_NO_ACT_SIM=1 also switches the weights to bf16, so nothing quantizes and only bf16 rounding separates the paths
    precision = dict(dtype="bf16", expert_dtype=None) if kernel.NO_ACT_SIM else {}
    # every token routes to all experts: top-k flips on rounding noise would otherwise dominate the comparison
    args = M.ModelArgs(max_batch_size=1, max_seq_len=512, vocab_size=4096, dspark_block_size=0, n_mtp_layers=0,
                       n_activated_experts=8, **extra, **precision)
    gen = torch.Generator(device="cuda").manual_seed(0)
    model = M.Transformer(args)
    init_weights(model, gen)
    ids = torch.randint(0, args.vocab_size, (1, 160), generator=gen, device="cuda")
    p0 = 120
    # decode continuously from a prefill of p0 tokens, so state carried across steps (ring, partial groups) is exercised
    model(ids[:, :p0])
    decoded = [model(ids[:, t:t + 1], t)[1].float().clone() for t in range(p0, ids.size(1))]
    worst = 0.0
    for t, dec in zip(range(p0, ids.size(1)), decoded):
        pre = model(ids[:, :t + 1])[1].float()
        rel = ((dec - pre).norm() / pre.norm()).item()
        agree = (dec.argmax(-1) == pre.argmax(-1)).all().item()
        worst = max(worst, rel)
        print(f"pos {t}: rel {rel:.3e} argmax {'same' if agree else 'DIFF'}")
    print(f"worst rel diff {worst:.3e}")
    # bf16 noise through 5 random layers, compounded over consecutive decode steps; an indexing error shows as 15-30%
    limit = 5e-2 if kernel.NO_ACT_SIM else 1e-1
    assert worst < limit, (worst, limit)


if __name__ == "__main__":
    main()
