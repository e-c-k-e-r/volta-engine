#!/usr/bin/env python3
"""Geometry of the probe's refusal directions per layer from steer_geometry's output.

  steer_geometry.py STATES.bin --layout tech=15,ethic=15,benign=30

Per prompt p and layer: delta_p = mean(declining openings) - mean(accepting openings) (the probe's contrast). Per category c:
  axis_c      mean of delta_p over the category's prompts, minus the benign prompts' mean (the part that depends on the prompt, not on the
              forced words)
  cos         cosine between the tech and ethic axes
  own / cross share of a category's delta (prompt-dependent part) along its own axis / along the other category's axis
  auc         leave-one-out AUC of the prompt-end state's projection on (category mean - benign mean) for category vs benign
Prints a table over layers.
"""
import sys, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("states")
ap.add_argument("--layout", default="tech=15,ethic=15,benign=30")
a = ap.parse_args()
raw = open(a.states, "rb").read()
n, L, H, first = np.frombuffer(raw[:16], dtype=np.int32)
data = np.frombuffer(raw[16:], dtype=np.float32).reshape(n, 3, L, H).astype(np.float64)
end, acc, dec = data[:, 0], data[:, 1], data[:, 2]
delta = dec - acc
sizes = [int(x.split("=")[1]) for x in a.layout.split(",")]
names = [x.split("=")[0] for x in a.layout.split(",")]
idx, o = {}, 0
for nm, sz in zip(names, sizes):
    idx[nm] = slice(o, o + sz); o += sz
assert o == n, (o, n)

def auc(pos, neg):
    s = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    s.sort(key=lambda t: t[0])
    ranks = {}
    for i, (v, _) in enumerate(s): ranks.setdefault(v, []).append(i + 1)
    r = sum(np.mean(ranks[v]) for v, l in s if l == 1)
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def loo_auc(cat, ben, layer):
    """projection of held-out prompt-end states on the mean difference (without that prompt)"""
    pos_scores, neg_scores = [], []
    for i in range(len(cat)):
        rest = np.delete(cat, i, axis=0)
        w = rest.mean(0) - ben.mean(0); w /= np.linalg.norm(w) + 1e-12
        pos_scores.append(cat[i] @ w)
    for j in range(len(ben)):
        rest = np.delete(ben, j, axis=0)
        w = cat.mean(0) - rest.mean(0); w /= np.linalg.norm(w) + 1e-12
        neg_scores.append(ben[j] @ w)
    return auc(pos_scores, neg_scores)

bmean = delta[idx["benign"]].mean(0)   # [L, H]
print(f"{'layer':>5} {'|tech|':>7} {'|eth|':>7} {'cos':>6} {'tech own/cross':>15} {'eth own/cross':>15} {'AUC tech':>9} {'AUC eth':>8} {'AUC eth vs tech':>16}")
for li in range(L):
    dt = delta[idx["tech"], li] - bmean[li]
    de = delta[idx["ethic"], li] - bmean[li]
    at, ae = dt.mean(0), de.mean(0)
    nt, ne = np.linalg.norm(at), np.linalg.norm(ae)
    cos = at @ ae / (nt * ne + 1e-12)
    ut, ue = at / (nt + 1e-12), ae / (ne + 1e-12)
    def share(d, u): return float(np.mean((d @ u) ** 2 / (np.sum(d * d, axis=1) + 1e-12)))
    tt, tx = share(dt, ut), share(dt, ue)
    et, ex = share(de, ue), share(de, ut)
    ben = end[idx["benign"], li]
    at_auc = loo_auc(end[idx["tech"], li], ben, li)
    ae_auc = loo_auc(end[idx["ethic"], li], ben, li)
    ev_auc = loo_auc(end[idx["ethic"], li], end[idx["tech"], li], li)
    print(f"{first + li:5d} {nt:7.2f} {ne:7.2f} {cos:6.2f} {tt:7.2f}/{tx:5.2f}  {et:7.2f}/{ex:5.2f}  {at_auc:9.2f} {ae_auc:8.2f} {ev_auc:16.2f}")
