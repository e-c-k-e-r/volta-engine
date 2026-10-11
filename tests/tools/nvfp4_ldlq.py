#!/usr/bin/env python3
"""Calibrated NVFP4 quantization of Flash-Next's dense projections (block LDLQ, as exllamav3's quantizer without the trellis).

For each matrix W [N, K] and the input Hessian H = X^T X of its projection (written by the engine with ENGINE_HESSIAN_DIR):
  H /= rows; H += sigma_reg * mean(diag) * I; H = L L^T (Cholesky); L <- L * blockdiag(L_ii)^-1 (unit 16x16 diagonal blocks, zeroed)
  for the 16-row blocks i = K/16 - 1 .. 0:   rows_i = W_i + sum_{j>i} L[j, i]^T (W_j - Wq_j);  Wq_i = NVFP4(rows_i)
NVFP4(): one E4M3 scale per (16 K values, output column), searched over the +-2 codes around amax / 6, grid E2M1, one fp32 global scale
per matrix (amax / (6 * 448)) fixed before the loop. Row-parallel inputs (o_proj, out_proj) use the per-rank Hessian blocks as a block diagonal.

Writes modelopt-named tensors (<name>.weight u8 [N, K/2], .weight_scale f8_e4m3 [N, K/16], .weight_scale_2 f32) to <out>/dense-LAA-LBB.safetensors (eight layers per file) and dense-head.safetensors.

  nvfp4_ldlq.py --src <NVFP4 checkpoint> --hess <hessian dir> --out <dir> [--groups gdn,shared,attn,idx,head] [--layers 0-47] [--tp 4] [--rtn]
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
THRESH = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])   # midpoints; ties go to the lower magnitude


def e4m3_table(dev):
    codes = torch.arange(127, dtype=torch.uint8)
    return codes.view(torch.float8_e4m3fn).to(torch.float32).to(dev)   # 0..448 for codes 0..126


class Quantizer:
    def __init__(self, dev):
        self.dev = dev
        self.et = e4m3_table(dev)
        self.grid = GRID.to(dev)
        self.thresh = THRESH.to(dev)

    def codes(self, x):
        """E2M1 code (sign bit 3, magnitude index) of x (any shape), nearest, ties to the lower magnitude."""
        a = x.abs()
        m = torch.bucketize(a, self.thresh, right=False)   # number of thresholds strictly below a
        return (m | ((x < 0).to(torch.int64) << 3)).to(torch.uint8)

    def block(self, v, gs):
        """v [16, N] (K block x columns) -> (dequantized [16, N], codes [16, N] u8, scale codes [N] u8)."""
        amax = v.abs().amax(dim=0)
        ideal = amax / (6.0 * gs)
        c0 = torch.bucketize(ideal, self.et).clamp(max=126)           # first table entry >= ideal
        lo = (c0 - 1).clamp(min=0)
        c0 = torch.where((ideal - self.et[lo]) < (self.et[c0] - ideal), lo, c0).clamp(1, 126)
        best_err = None
        best_c = c0
        best_dq = None
        for d in (-2, -1, 0, 1, 2):
            c = (c0 + d).clamp(1, 126)
            sc = self.et[c] * gs                                       # [N]
            q = self.codes(v / sc)
            dq = self.grid[(q & 7).long()] * torch.where((q & 8) != 0, -1.0, 1.0) * sc
            err = ((dq - v) ** 2).sum(dim=0)
            if best_err is None:
                best_err, best_c, best_dq, best_q = err, c, dq, q
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_c = torch.where(better, c, best_c)
                best_dq = torch.where(better, dq, best_dq)
                best_q = torch.where(better, q, best_q)
        return best_dq, best_q, best_c.to(torch.uint8)


def load_hessian(path, dev):
    with open(path, "rb") as f:
        magic, k = np.frombuffer(f.read(8), dtype=np.int32)
        rows = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        assert magic == 0x48455353
        h = np.frombuffer(f.read(), dtype=np.float32).reshape(k, k)
    return torch.from_numpy(h.copy()).to(dev), rows


def block_ldl(H, sigma_reg=0.025, b=16):
    """H [K, K] (already /rows) -> unit block lower-triangular L with zero diagonal blocks (exllamav3 block_ldl)."""
    K = H.shape[0]
    H = H.clone().double()
    H.diagonal().add_(sigma_reg * H.diagonal().mean())
    for attempt in range(10):
        try:
            L = torch.linalg.cholesky(H)
            break
        except Exception:
            H.diagonal().add_(2.0 * sigma_reg * H.diagonal().mean())
    else:
        raise RuntimeError("Cholesky failed")
    m = K // b
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)   # [m, b, b]
    DL = torch.linalg.inv(DL)
    L = L.view(K, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i]
    L = L.reshape(K, K)
    Lb = L.view(m, b, m, b)
    for i in range(m):
        Lb[i, :, i, :] = 0
    return L.float().contiguous(), H.float()


def ldlq(W, H, q, gs, chunk=16384, use_h=True):
    """W [N, K] float32 (cuda) -> (codes [N, K] u8, scale codes [N, K/16] u8, Wq [N, K], proxy errors (rtn, ldlq))."""
    N, K = W.shape
    L = block_ldl(H)[0] if use_h else None
    codes = torch.empty((N, K), dtype=torch.uint8, device=W.device)
    scs = torch.empty((N, K // 16), dtype=torch.uint8, device=W.device)
    Wq = torch.empty_like(W)
    for n0 in range(0, N, chunk):
        Wt = W[n0:n0 + chunk].t().contiguous()                          # [K, n]
        n = Wt.shape[1]
        err = torch.zeros_like(Wt)                                       # W - Wq per row block, filled from the bottom
        for i in range(K // 16 - 1, -1, -1):
            r0, r1 = i * 16, i * 16 + 16
            rows = Wt[r0:r1]
            if use_h and r1 < K:
                rows = rows + L[r1:, r0:r1].t() @ err[r1:]
            dq, qc, sc = q.block(rows, gs)
            codes[n0:n0 + n, r0:r1] = qc.t()
            scs[n0:n0 + n, i] = sc
            Wq[n0:n0 + n, r0:r1] = dq.t()
            err[r0:r1] = Wt[r0:r1] - dq
    return codes, scs, Wq


def proxy(W, Wq, H):
    d = (W - Wq).double()
    Hd = H.double()
    return float(((d @ Hd) * d).sum() / ((W.double() @ Hd * W.double()).sum() + 1e-30))


def pack(codes, scs, gs):
    N, K = codes.shape
    lo, hi = codes[:, 0::2], codes[:, 1::2]
    return (lo | (hi << 4)).cpu(), scs.cpu().view(torch.float8_e4m3fn), torch.tensor(gs, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--hess", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--model", default="flash", choices=["flash", "glm"], help="flash: Qwen3.8-Flash-Next; glm: GLM-5.3-Flash (groups kda, mla, idx, shared, head)")
    ap.add_argument("--rtn", action="store_true", help="no Hessian (round-to-nearest with the scale search), for comparison")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if a.groups is None: a.groups = "gdn,shared,attn,idx,head" if a.model == "flash" else "kda,mla,idx,shared,head"
    if a.layers is None: a.layers = "0-47" if a.model == "flash" else "0-44"
    groups = set(a.groups.split(","))
    lo, hi = (int(x) for x in a.layers.split("-"))
    os.makedirs(a.out, exist_ok=True)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(a.src, f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    P = "model.language_model"
    q = Quantizer(dev)

    def hess_for(L, site, tp_blocks=False):
        """Hessian of a layer's site: replicated sites come from rank 0, rank-sliced ones as a block diagonal over the ranks."""
        if not tp_blocks:
            H, rows = load_hessian(os.path.join(a.hess, f"L{L:02d}_s{site}_r0.f32"), dev)
            return H / rows
        blocks = []
        for r in range(a.tp):
            Hr, rows = load_hessian(os.path.join(a.hess, f"L{L:02d}_s{site}_r{r}.f32"), dev)
            blocks.append(Hr / rows)
        return torch.block_diag(*blocks)

    def do(out, name, H, label):
        t0 = time.time()
        W = get(name + ".weight").to(torch.float32).to(dev)
        gs = float(W.abs().max()) / (6.0 * 448.0)
        codes, scs, Wq = ldlq(W, H, q, gs, use_h=not a.rtn)
        pe = proxy(W, Wq, H)
        if not a.rtn:
            c2, s2, W2 = ldlq(W, H, q, gs, use_h=False)
            pr = proxy(W, W2, H)
            del c2, s2, W2
        else:
            pr = pe
        w, s, g = pack(codes, scs, gs)
        out[name + ".weight"] = w
        out[name + ".weight_scale"] = s
        out[name + ".weight_scale_2"] = g
        print(f"  {label:34s} N {W.shape[0]:6d} K {W.shape[1]:5d}  proxy err rtn {pr:.5f} -> ldlq {pe:.5f}  ({time.time() - t0:.1f}s)", flush=True)
        del W, Wq, codes, scs
        torch.cuda.empty_cache()

    # The engine's loader opens at most 32 shard files: eight layers per file
    pending, first = {}, lo
    for L in range(lo, hi + 1):
        out = pending
        pre = f"{P}.layers.{L}"
        if a.model == "glm":
            mla = f"{pre}.self_attn.q_a_proj.weight" in idx
            print(f"layer {L} ({'mla' if mla else 'kda'})", flush=True)
            sa = f"{pre}.self_attn"
            if mla:
                if "mla" in groups:
                    H0 = hess_for(L, 0)
                    do(out, f"{sa}.q_a_proj", H0, "q_a_proj")
                    do(out, f"{sa}.kv_a_proj_with_mqa", H0, "kv_a_proj_with_mqa")
                    do(out, f"{sa}.q_b_proj", hess_for(L, 4), "q_b_proj")
                    do(out, f"{sa}.o_proj", hess_for(L, 2, True), "o_proj (mla)")
                if "idx" in groups:
                    do(out, f"{sa}.indexer.wq_b", hess_for(L, 4), "indexer wq_b")
            elif "kda" in groups:
                do_multi(out, [f"{sa}.q_proj", f"{sa}.k_proj", f"{sa}.v_proj"], hess_for(L, 0), "q / k / v")
                do(out, f"{sa}.o_proj", hess_for(L, 2, True), "o_proj (kda)")
            if "shared" in groups and L >= 3:
                H1 = hess_for(L, 1)
                do(out, f"{pre}.mlp.shared_experts.gate_proj", H1, "shared gate")
                do(out, f"{pre}.mlp.shared_experts.up_proj", H1, "shared up")
                do(out, f"{pre}.mlp.shared_experts.down_proj", hess_for(L, 3), "shared down")
        else:
            full = f"{pre}.self_attn.q_proj.weight" in idx
            print(f"layer {L} ({'attn' if full else 'gdn'})", flush=True)
            if full:
                H0 = hess_for(L, 0)
                if "attn" in groups:
                    for n in ("q_proj", "k_proj", "v_proj"):
                        do(out, f"{pre}.self_attn.{n}", H0, n)
                    do(out, f"{pre}.self_attn.o_proj", hess_for(L, 2, True), "o_proj")
                if "idx" in groups:
                    do(out, f"{pre}.self_attn.indexer.index_qk_proj", H0, "index_qk_proj")
            else:
                if "gdn" in groups:
                    H0 = hess_for(L, 0)
                    do(out, f"{pre}.linear_attn.in_proj_qkv", H0, "in_proj_qkv")
                    do(out, f"{pre}.linear_attn.in_proj_z", H0, "in_proj_z")
                    do(out, f"{pre}.linear_attn.out_proj", hess_for(L, 2, True), "out_proj")
            if "shared" in groups:
                H1 = hess_for(L, 1)
                do(out, f"{pre}.mlp.shared_expert.gate_proj", H1, "shared gate")
                do(out, f"{pre}.mlp.shared_expert.up_proj", H1, "shared up")
                do(out, f"{pre}.mlp.shared_expert.down_proj", hess_for(L, 3), "shared down")
        if (L - lo + 1) % 8 == 0 or L == hi:
            if out:
                save_file(out, os.path.join(a.out, f"dense-L{first:02d}-{L:02d}.safetensors"))
            pending, first = {}, L + 1
    if "head" in groups and lo == 0:
        out = {}
        do(out, "lm_head", hess_for(0, 4 if a.model == "flash" else 5), "lm_head")
        save_file(out, os.path.join(a.out, "dense-head.safetensors"))
    write_index(a.out)


def merge_shards(out, per=8):
    """Merges per-layer dense-LNN.safetensors files (an older run) into files of `per` layers."""
    files = sorted(f for f in os.listdir(out) if re.fullmatch(r"dense-L\d\d\.safetensors", f))
    for i in range(0, len(files), per):
        group = files[i:i + per]
        merged = {}
        for fn in group:
            with safe_open(os.path.join(out, fn), framework="pt", device="cpu") as f:
                for k in f.keys():
                    merged[k] = f.get_tensor(k)
        save_file(merged, os.path.join(out, f"dense-{group[0][7:9]}-{group[-1][7:9]}.safetensors"))
        for fn in group:
            os.remove(os.path.join(out, fn))


def write_index(out):
    """model.safetensors.index.json over the dense-*.safetensors files (the engine's loader opens a directory through its index)."""
    wm = {}
    for fn in sorted(os.listdir(out)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(out, fn), framework="pt", device="cpu") as f:
                for k in f.keys():
                    wm[k] = fn
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))


if __name__ == "__main__":
    main()
