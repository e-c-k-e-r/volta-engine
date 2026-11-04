#!/usr/bin/env python3
"""Builds a router-steering file (.steer version 3, STEER_ROUTE entries) from a route_dump of two prompt sets: per MoE layer and expert, how often the
expert is chosen at the last prompt token of the `pos` prompts against the `neg` prompts; the N pairs with the largest gap get a bias on their router
selection score: experts the pos prompts favour get -B (picked less), the ones the neg prompts favour +B (picked more).

  steer_route_build.py --dump routes.bin --log log.json --pos target --neg control --out NAME.steer [--top 10] [--bias 0.1] [--mode both|suppress|promote]
                       [--hidden 4096] [--layers 45] [--use even|odd|all] [--skip 0] [--with OTHER.steer]

--log is a JSON list with one {"kind": ...} per request in dump order (one plain prefill each); --use picks the discovery prompts of each kind (even
indices by default, so the odd ones stay held out); --skip drops the first passes of the dump (warm-up). Bias units are those of the router's
selection score (GLM: sigmoid score plus the correction bias, so about 0 to 1; a bias of -1 or +1 always removes or keeps an expert).
--with copies another .steer file's entries (clamps, adds, probes, router entries) into the output, so one request gets both.
"""
import argparse, json, struct
import numpy as np

STEER_ROUTE = 9

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True); ap.add_argument("--log", required=True)
ap.add_argument("--pos", required=True); ap.add_argument("--neg", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=10); ap.add_argument("--bias", type=float, default=0.1)
ap.add_argument("--mode", default="both", choices=["both", "suppress", "promote"])
ap.add_argument("--hidden", type=int, default=4096); ap.add_argument("--layers", type=int, default=45)
ap.add_argument("--use", default="even", choices=["even", "odd", "all"]); ap.add_argument("--skip", type=int, default=2)
ap.add_argument("--with", dest="other", default="")
a = ap.parse_args()

log = json.load(open(a.log))
passes, first_layer = [], None
with open(a.dump, "rb") as f:
    while True:
        h = f.read(16)
        if len(h) < 16: break
        layer, rows, first, K = struct.unpack("<4i", h)
        ids = np.frombuffer(f.read(4 * rows * K), dtype=np.int32).reshape(rows, K)
        f.seek(2 * rows * a.hidden, 1)
        if first_layer is None: first_layer = layer
        if layer == first_layer and first == 0: passes.append({})
        passes[-1][layer] = ids[-1].copy()   # the last prompt row (a later chunk of the prompt overwrites)
passes = passes[a.skip:]
assert len(passes) == len(log), f"{len(passes)} passes for {len(log)} requests"
layers = sorted(passes[0])
E = 1 + max(int(p[l].max()) for p in passes for l in layers)
kind = np.array([e["kind"] for e in log])
pick = {"even": slice(0, None, 2), "odd": slice(1, None, 2), "all": slice(None)}[a.use]
P = np.flatnonzero(kind == a.pos)[pick]; N = np.flatnonzero(kind == a.neg)[pick]
act = np.zeros((len(log), len(layers), E), bool)
for i, p in enumerate(passes):
    for j, l in enumerate(layers): act[i, j, p[l]] = True
gap = act[P].mean(0) - act[N].mean(0)
order = np.argsort(-np.abs(gap).ravel())
entries = []
for o in order:
    if len(entries) >= a.top: break
    j, e = divmod(int(o), E)
    g = gap[j, e]
    if a.mode == "suppress" and g <= 0: continue
    if a.mode == "promote" and g >= 0: continue
    entries.append((layers[j], e, -a.bias if g > 0 else a.bias, g))
other, scope, scale = [], 7, 1.0   # (layer, kind, par, expert, vec bytes) of --with
if a.other:
    with open(a.other, "rb") as f:
        assert f.read(4) == b"VSTR"
        ver, hid, _, scope, n = struct.unpack("<5i", f.read(20))
        assert hid == a.hidden, f"--with: hidden {hid}"
        if ver >= 2: scale = struct.unpack("<f", f.read(4))[0]
        for _ in range(n):
            layer, kind, par = struct.unpack("<iif", f.read(12))
            expert = struct.unpack("<i", f.read(4))[0] if ver >= 3 else -1
            vec = b"" if kind == STEER_ROUTE else f.read(4 * hid)
            other.append((layer, kind, par, expert, vec))
with open(a.out, "wb") as f:
    f.write(b"VSTR"); f.write(struct.pack("<5i", 3, a.hidden, a.layers, scope, len(entries) + len(other))); f.write(struct.pack("<f", scale))
    for layer, kind, par, expert, vec in other: f.write(struct.pack("<iifi", layer, kind, par, expert) + vec)
    for layer, e, b, _ in entries: f.write(struct.pack("<iifi", layer, STEER_ROUTE, b, e))
for layer, e, b, g in entries: print(f"layer {layer:2d} expert {e:3d}: chosen {act[P, layers.index(layer), e].mean():.2f} ({a.pos}) vs {act[N, layers.index(layer), e].mean():.2f} ({a.neg}), bias {b:+.3f}")
print(f"wrote {a.out}: {len(entries)} router entries from {len(P)} + {len(N)} prompts")
