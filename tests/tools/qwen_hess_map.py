#!/usr/bin/env python3
"""Map file for exl3_quantize.py --hess: which Hessian file(s) of a test_ppl --calibrate capture serve which tensor of a Qwen3.5-family dense model.

  qwen_hess_map.py --dir <ENGINE_HESSIAN_DIR> --out map.json [--layers 64] [--ranks 2]

Sites per layer (see Qwen3_5Pipeline): s0 attention input (q/k/v_proj, or in_proj_qkv and in_proj_z of a linear-attention layer), s1 MLP input (gate/up_proj),
s2 o_proj / out_proj input and s3 down_proj input (one file per rank: the input channels are split over the ranks), (layer 0, s4) the LM head input.
A layer is full attention when (layer + 1) % 4 == 0. Tensors without a capture file are left out and get the uncalibrated quantization.
"""
import argparse, json, os

P = "model.language_model.layers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--ranks", type=int, default=2)
    a = ap.parse_args()
    f = lambda L, s, r: os.path.join(a.dir, f"L{L:02d}_s{s}_r{r}.f32")
    have = lambda L, s, r=0: os.path.exists(f(L, s, r))
    m = {}
    for L in range(a.layers):
        full = (L + 1) % 4 == 0
        if have(L, 0):
            names = ["q_proj", "k_proj", "v_proj"] if full else ["in_proj_qkv", "in_proj_z"]
            for n in names: m[f"{P}.{L}.{'self_attn' if full else 'linear_attn'}.{n}"] = f(L, 0, 0)
        if have(L, 1):
            for n in ("gate_proj", "up_proj"): m[f"{P}.{L}.mlp.{n}"] = f(L, 1, 0)
        if all(have(L, 2, r) for r in range(a.ranks)):
            m[f"{P}.{L}.{'self_attn.o_proj' if full else 'linear_attn.out_proj'}"] = [f(L, 2, r) for r in range(a.ranks)]
        if all(have(L, 3, r) for r in range(a.ranks)):
            m[f"{P}.{L}.mlp.down_proj"] = [f(L, 3, r) for r in range(a.ranks)]
    if have(0, 4): m["lm_head"] = f(0, 4, 0)
    json.dump(m, open(a.out, "w"), indent=1)
    print(f"{len(m)} tensors mapped -> {a.out}")


if __name__ == "__main__":
    main()
