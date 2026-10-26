#!/usr/bin/env python3
"""Map file for exl3_quantize.py --hess: which Hessian file(s) of a Qwen3.8-Flash-Next capture (test_ppl --flash --calibrate with ENGINE_HESSIAN_DIR) serve which tensor.

  flash_hess_map.py --dir <capture dir> --src <original weights> --out map.json [--ranks 4] [--down-dir DIR]

Sites per layer (Qwen4ExpPipeline): s0 attention / GDN input (q/k/v_proj and indexer.index_qk_proj of a full-attention layer, in_proj_qkv and in_proj_z of a linear-attention layer),
s1 MoE input (gate_proj and up_proj of the shared expert and of EVERY routed expert, the recipe of upstream's converter: all experts see all rows and share the block input),
s2 o_proj / out_proj input (one file per rank, block diagonal), s3 shared-expert down input, (layer 0, s4) lm_head input; the MTP head's block (layer 48) and input projections (layer 49) when the capture ran the head. The routed experts' down_proj inputs are the experts' own intermediate
activations: their Hessians come from moe_down_hessians.py (files L<layer>_E<expert>_down.f32 in --down-dir). A layer is full attention when the original has its self_attn.q_proj.
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--down-dir", default="")
    ap.add_argument("--experts", type=int, default=512)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    layers = sorted({int(k.split("layers.")[1].split(".")[0]) for k in idx if k.startswith("model.language_model.layers.") and ".mlp.gate.weight" in k})
    f = lambda L, s, r=0: os.path.join(a.dir, f"L{L:02d}_s{s}_r{r}.f32")
    have = lambda L, s, r=0: os.path.exists(f(L, s, r))
    P = "model.language_model.layers"
    m = {}
    for L in layers:
        full = f"{P}.{L}.self_attn.q_proj.weight" in idx
        if have(L, 0):
            if full:
                for n in ("q_proj", "k_proj", "v_proj"): m[f"{P}.{L}.self_attn.{n}"] = f(L, 0)
                if f"{P}.{L}.self_attn.indexer.index_qk_proj.weight" in idx: m[f"{P}.{L}.self_attn.indexer.index_qk_proj"] = f(L, 0)
            else:
                for n in ("in_proj_qkv", "in_proj_z"): m[f"{P}.{L}.linear_attn.{n}"] = f(L, 0)
        if have(L, 1):
            for n in ("gate_proj", "up_proj"):
                m[f"{P}.{L}.mlp.shared_expert.{n}"] = f(L, 1)
                for e in range(a.experts): m[f"{P}.{L}.mlp.experts.{e}.{n}"] = f(L, 1)
        if all(have(L, 2, r) for r in range(a.ranks)):
            m[f"{P}.{L}.{'self_attn.o_proj' if full else 'linear_attn.out_proj'}"] = [f(L, 2, r) for r in range(a.ranks)]
        if have(L, 3): m[f"{P}.{L}.mlp.shared_expert.down_proj"] = f(L, 3)
        if a.down_dir:
            for e in range(a.experts):
                p = os.path.join(a.down_dir, f"L{L:02d}_E{e:03d}_down.f32")
                if os.path.exists(p): m[f"{P}.{L}.mlp.experts.{e}.down_proj"] = p
    if have(0, 4): m["lm_head"] = f(0, 4)
    # The MTP head (a replay with ENGINE_CALIB_MTP): its block is layer N, its input projections layer N + 1 (site 0 fc_embedding, site 1 fc_hidden), N = the number of decoder layers
    Lm = len(layers)
    Q = "mtp.layers.0"
    if have(Lm, 0) and f"{Q}.self_attn.q_proj.weight" in idx:
        for n in ("q_proj", "k_proj", "v_proj"): m[f"{Q}.self_attn.{n}"] = f(Lm, 0)
        if f"{Q}.self_attn.indexer.index_qk_proj.weight" in idx: m[f"{Q}.self_attn.indexer.index_qk_proj"] = f(Lm, 0)
        if have(Lm, 1):
            for n in ("gate_proj", "up_proj"):
                m[f"{Q}.mlp.shared_expert.{n}"] = f(Lm, 1)
                for e in range(a.experts): m[f"{Q}.mlp.experts.{e}.{n}"] = f(Lm, 1)
        if all(have(Lm, 2, r) for r in range(a.ranks)): m[f"{Q}.self_attn.o_proj"] = [f(Lm, 2, r) for r in range(a.ranks)]
        if have(Lm, 3): m[f"{Q}.mlp.shared_expert.down_proj"] = f(Lm, 3)
        if a.down_dir:
            for e in range(a.experts):
                p = os.path.join(a.down_dir, f"L{Lm:02d}_E{e:03d}_down.f32")
                if os.path.exists(p): m[f"{Q}.mlp.experts.{e}.down_proj"] = p
        if have(Lm + 1, 0): m["mtp.fc_embedding"] = f(Lm + 1, 0)
        if have(Lm + 1, 1): m["mtp.fc_hidden"] = f(Lm + 1, 1)
    json.dump(m, open(a.out, "w"))
    print(f"{len(m)} tensors mapped ({len(layers)} layers) -> {a.out}")


if __name__ == "__main__":
    main()
