#!/usr/bin/env python3
"""Reads the files that `route_dump` writes (Flash-Next, GLM): per MoE layer and prefill chunk a header {layer, rows, first position, K} (int32), the rows' expert ids (int32, rows x K) and the
router's input rows (FP16, rows x hidden), which are the rows that feed the experts. Two uses:

  moe_route.py stats  FILE --hidden H --experts E [--shared-prefix N]     rows per expert per layer (how thin the expert Hessians are)
  moe_route.py hess   FILE --hidden H --experts E --layer L --out DIR [--shared-prefix N] [--device cuda:0]
                                                                          the gate / up input Hessians of every expert of one layer (sum of x x^T over its rows), files of the engine's
                                                                          format named E<expert>_gate_up.f32

A replay that sends every session through the model repeats the shared prefix: a chunk that starts at position 0 begins a new session, and with --shared-prefix N the rows below position N
of every session but the first are left out, so each distinct row counts once.
"""
import argparse, collections, os, struct, sys
import numpy as np


def records(path, hidden, shared_prefix=0):
    """yields (layer, session, positions, ids [rows, K], x [rows, hidden] fp16) with the repeated prefix rows removed"""
    session = -1
    last_pos = {}
    with open(path, "rb") as f:
        while True:
            h = f.read(16)
            if len(h) < 16: return
            layer, rows, first, K = struct.unpack("<4i", h)
            ids = np.frombuffer(f.read(4 * rows * K), dtype=np.int32).reshape(rows, K)
            x = np.frombuffer(f.read(2 * rows * hidden), dtype=np.float16).reshape(rows, hidden)
            if first == 0 and (layer not in last_pos or last_pos[layer] != -1):
                if layer == min(last_pos, default=layer) or session < 0: session += 1
            last_pos[layer] = first + rows
            pos = first + np.arange(rows)
            keep = pos >= shared_prefix if session > 0 else np.ones(rows, bool)
            if keep.all(): yield layer, session, pos, ids, x
            elif keep.any(): yield layer, session, pos[keep], ids[keep], x[keep]


def stats(a):
    counts = collections.defaultdict(lambda: np.zeros(a.experts, dtype=np.int64))
    toks = collections.Counter()
    for layer, session, pos, ids, x in records(a.file, a.hidden, a.shared_prefix):
        counts[layer] += np.bincount(ids.ravel(), minlength=a.experts)
        toks[layer] += len(pos)
    print(f"{'layer':>5s} {'rows':>9s} {'min':>7s} {'p5':>7s} {'median':>7s} {'mean':>8s} {'max':>8s}  experts below {a.hidden} / {2 * a.hidden} rows")
    for layer in sorted(counts):
        c = counts[layer]
        print(f"{layer:5d} {toks[layer]:9d} {c.min():7d} {int(np.percentile(c, 5)):7d} {int(np.median(c)):7d} {c.mean():8.0f} {c.max():8d}  {int((c < a.hidden).sum()):4d} / {int((c < 2 * a.hidden).sum()):4d}")


def hess(a):
    import torch
    dev = torch.device(a.device)
    H = torch.zeros((a.experts, a.hidden, a.hidden), dtype=torch.float32, device=dev) if a.experts * a.hidden * a.hidden * 4 < 24e9 else None
    assert H is not None, "the Hessians of one layer do not fit on the device: process a range of experts (--expert-range)"
    rows_per = np.zeros(a.experts, dtype=np.int64)
    for layer, session, pos, ids, x in records(a.file, a.hidden, a.shared_prefix):
        if layer != a.layer: continue
        xt = torch.from_numpy(x.astype(np.float32)).to(dev)
        idt = torch.from_numpy(ids.astype(np.int64)).to(dev)
        for e in range(a.experts):
            sel = (idt == e).any(dim=1)
            n = int(sel.sum())
            if n == 0: continue
            xe = xt[sel]
            H[e].addmm_(xe.T, xe)
            rows_per[e] += n
    os.makedirs(a.out, exist_ok=True)
    for e in range(a.experts):
        if rows_per[e] == 0: continue
        with open(os.path.join(a.out, f"E{e:03d}_gate_up.f32"), "wb") as f:
            f.write(np.array([0x48455353, a.hidden], dtype=np.int32).tobytes())
            f.write(np.array([rows_per[e]], dtype=np.int64).tobytes())
            f.write(H[e].cpu().numpy().tobytes())
    print(f"layer {a.layer}: {int((rows_per > 0).sum())} experts with rows, median {int(np.median(rows_per))} rows -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "hess"])
    ap.add_argument("file")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--shared-prefix", type=int, default=0)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    (stats if a.cmd == "stats" else hess)(a)


if __name__ == "__main__":
    main()
