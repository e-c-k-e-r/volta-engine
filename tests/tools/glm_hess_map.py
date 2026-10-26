#!/usr/bin/env python3
"""Map file for exl3_quantize.py --hess: which Hessian file(s) of a GLM-5.3-Flash capture (test_ppl --glm --calibrate with ENGINE_HESSIAN_DIR) serve which tensor of the EXL3 checkpoint
(the names of the template checkpoint, not of the original: GLM's linear attention is one fused qkv_proj there).

  glm_hess_map.py --dir <capture dir> --src <original weights> --out map.json [--ranks 4] [--down-dir DIR] [--assume-down]

Sites per layer (Glm5NextPipeline): s0 attention input (qkv_proj of a linear-attention layer; q_a_proj and kv_a_proj_with_mqa of an MLA layer), s1 FFN input (gate_proj and up_proj of a dense
MLP, of the shared expert and of EVERY routed expert: all experts see all rows), s2 o_proj input (one file per rank, block diagonal), s3 shared-expert down input (a dense MLP's down input in
layers 0 to 2, unsliced in a capture), s4 q-lora norm output (MLA: q_b_proj and indexer.wq_b), (layer 0, s5) lm_head input. The routed experts' down_proj inputs are the experts' own
intermediate activations: their Hessians come from moe_down_hessians.py (files L<layer>_E<expert>_down.f32 in --down-dir; --assume-down lists them even when the files do not exist yet, for a
pipeline that writes them just before the layer is quantized).
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--down-dir", default="")
    ap.add_argument("--assume-down", action="store_true")
    ap.add_argument("--experts", type=int, default=288)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    P = "model.language_model.layers"
    layers = sorted({int(k.split("layers.")[1].split(".")[0]) for k in idx if k.startswith(P + ".") and ".self_attn." in k and int(k.split("layers.")[1].split(".")[0]) < 45})
    f = lambda L, s, r=0: os.path.join(os.path.abspath(a.dir), f"L{L:02d}_s{s}_r{r}.f32")
    have = lambda L, s, r=0: os.path.exists(f(L, s, r))
    m = {}
    for L in layers:
        pre = f"{P}.{L}"
        mla = f"{pre}.self_attn.q_a_proj.weight" in idx
        moe = f"{pre}.mlp.gate.weight" in idx
        if have(L, 0):
            if mla:
                for n in ("q_a_proj", "kv_a_proj_with_mqa"): m[f"{pre}.self_attn.{n}"] = f(L, 0)
            else:
                m[f"{pre}.self_attn.qkv_proj"] = f(L, 0)
        if mla and have(L, 4):
            m[f"{pre}.self_attn.q_b_proj"] = f(L, 4)
            if f"{pre}.self_attn.indexer.wq_b.weight" in idx: m[f"{pre}.self_attn.indexer.wq_b"] = f(L, 4)
        if all(have(L, 2, r) for r in range(a.ranks)): m[f"{pre}.self_attn.o_proj"] = [f(L, 2, r) for r in range(a.ranks)]
        if have(L, 1):
            if moe:
                for n in ("gate_proj", "up_proj"):
                    m[f"{pre}.mlp.shared_experts.{n}"] = f(L, 1)
                    for e in range(a.experts): m[f"{pre}.mlp.experts.{e}.{n}"] = f(L, 1)
            else:
                for n in ("gate_proj", "up_proj"): m[f"{pre}.mlp.{n}"] = f(L, 1)
        if have(L, 3): m[f"{pre}.mlp.shared_experts.down_proj" if moe else f"{pre}.mlp.down_proj"] = f(L, 3)
        if moe and a.down_dir:
            for e in range(a.experts):
                p = os.path.join(os.path.abspath(a.down_dir), f"L{L:02d}_E{e:03d}_down.f32")
                if a.assume_down or os.path.exists(p): m[f"{pre}.mlp.experts.{e}.down_proj"] = p
    if have(0, 5): m["lm_head"] = f(0, 5)
    json.dump(m, open(a.out, "w"))
    print(f"{len(m)} tensors mapped ({len(layers)} layers) -> {a.out}")


if __name__ == "__main__":
    main()
