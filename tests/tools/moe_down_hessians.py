#!/usr/bin/env python3
"""Down-projection Hessians of the routed experts, upstream's recipe: every expert sees every calibration row. For each MoE layer the rows are the experts' input (the rows that
`route_dump` writes: the router's input, post-norm hidden states, one record per prefill chunk), the activation of expert e is silu(gate_e x) * (up_e x) and its Hessian is the sum of a a^T over
all rows (intermediate x intermediate), written in the engine's Hessian format as L<layer>_E<expert>_down.f32. The original's fused gate_up_proj tensor ([experts, 2 * intermediate, hidden], gate first)
is read per layer.

  moe_down_hessians.py --route FILE --src <original dir> --hidden 2560 --experts 512 --out DIR [--layers 0-47] [--device cuda:0] [--rows-limit N]
  the MTP head's experts (route layer 48, tensors mtp.layers.0): --layers 48-48 --prefix mtp.layers --src-offset 48
"""
import argparse, json, os, struct, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_route import records
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows-limit", type=int, default=0, help="use at most this many rows per layer")
    ap.add_argument("--prefix", default="model.language_model.layers")
    ap.add_argument("--src-offset", type=int, default=0, help="the original's layer index is the route layer minus this")
    a = ap.parse_args()
    dev = torch.device(a.device)
    rows = {}
    for layer, session, pos, ids, x in records(a.route, a.hidden, 0):
        rows.setdefault(layer, []).append(torch.from_numpy(x.copy()))
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    lo, hi = (int(v) for v in a.layers.split("-")) if a.layers else (min(rows), max(rows))
    os.makedirs(a.out, exist_ok=True)
    for L in range(lo, hi + 1):
        if L not in rows: continue
        x = torch.cat(rows.pop(L))
        if a.rows_limit: x = x[:a.rows_limit]
        n = x.shape[0]
        name = f"{a.prefix}.{L - a.src_offset}.mlp.experts.gate_up_proj"
        with safe_open(os.path.join(a.src, idx[name]), "pt") as g:
            gu = g.get_tensor(name)                                   # [E, 2I, H] bf16 (kept on the host)
        I = gu.shape[1] // 2
        xd = x.to(dev)
        for e in range(a.experts):
            w = gu[e].to(dev)
            H = torch.zeros((I, I), dtype=torch.float32, device=dev)
            for c in range(0, n, 8192):
                xc = xd[c:c + 8192]
                gate = (xc @ w[:I].t().to(xc.dtype)).float()
                up = (xc @ w[I:].t().to(xc.dtype)).float()
                act = torch.nn.functional.silu(gate) * up
                H += act.t() @ act
            with open(os.path.join(a.out, f"L{L:02d}_E{e:03d}_down.f32"), "wb") as f:
                f.write(np.array([0x48455353, I], dtype=np.int32).tobytes())
                f.write(np.array([n], dtype=np.int64).tobytes())
                f.write(H.cpu().numpy().tobytes())
        print(f"layer {L}: {n} rows, {a.experts} experts of {I} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
