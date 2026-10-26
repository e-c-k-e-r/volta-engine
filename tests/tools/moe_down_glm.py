#!/usr/bin/env python3
"""Down-projection Hessians of GLM-5.3-Flash's routed experts, upstream's recipe (every expert sees every row), for the experts named in a plan. The rows are the router-input rows of one
layer (glm_route_split.py), the activation of expert e is silu(min(gate, limit)) * clamp(up, -limit, limit) with gate / up from the original's FP8 weights (block scales, 128 x 128) and the
model's swiglu limit, and its Hessian is the sum of a a^T over the rows (intermediate x intermediate), written in the engine's format as L<layer>_E<expert>_down.f32 (a file that exists is skipped).

  moe_down_glm.py --rows DIR --src <original dir> --plan plan.json --out DIR [--device cuda:0] [--hidden 4096] [--limit 10]

plan.json: {"<layer>": [experts]}
"""
import argparse, json, os, sys
import numpy as np
import torch
from safetensors import safe_open


def dequant(w, s):
    o, i = w.shape
    return (w.float().view(o // 128, 128, i // 128, 128) * s.float()[:, None, :, None]).view(o, i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--src", required=True); ap.add_argument("--plan", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--hidden", type=int, default=4096); ap.add_argument("--limit", type=float, default=10.0)
    a = ap.parse_args()
    dev = torch.device(a.device)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    plan = json.load(open(a.plan))
    os.makedirs(a.out, exist_ok=True)
    handles = {}

    def get(name):
        fn = idx[name]
        if fn not in handles: handles[fn] = safe_open(os.path.join(a.src, fn), "pt")
        return handles[fn].get_tensor(name)

    P = "model.language_model.layers"
    for L in sorted(plan, key=int):
        todo = [e for e in plan[L] if not os.path.exists(os.path.join(a.out, f"L{int(L):02d}_E{e:03d}_down.f32"))]
        if not todo: continue
        x = torch.from_numpy(np.fromfile(os.path.join(a.rows, f"route_L{int(L):02d}.f16"), dtype=np.float16).reshape(-1, a.hidden)).to(dev)
        n = x.shape[0]
        for e in todo:
            pre = f"{P}.{L}.mlp.experts.{e}"
            wg = dequant(get(pre + ".gate_proj.weight").to(dev), get(pre + ".gate_proj.weight_scale_inv").to(dev)).half()
            wu = dequant(get(pre + ".up_proj.weight").to(dev), get(pre + ".up_proj.weight_scale_inv").to(dev)).half()
            I = wg.shape[0]
            H = torch.zeros((I, I), dtype=torch.float32, device=dev)
            for c in range(0, n, 8192):
                xc = x[c:c + 8192]
                g = (xc @ wg.t()).float().clamp(max=a.limit)
                u = (xc @ wu.t()).float().clamp(-a.limit, a.limit)
                act = torch.nn.functional.silu(g) * u
                H += act.t() @ act
            path = os.path.join(a.out, f"L{int(L):02d}_E{e:03d}_down.f32")
            with open(path + ".tmp", "wb") as f:
                f.write(np.array([0x48455353, I], dtype=np.int32).tobytes())
                f.write(np.array([n], dtype=np.int64).tobytes())
                f.write(H.cpu().numpy().tobytes())
            os.replace(path + ".tmp", path)
        print(f"layer {L}: {n} rows, {len(todo)} experts of {I} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
