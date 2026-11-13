#!/usr/bin/env python3
"""Flags the cost cliffs in a bench_kernels sweep (CSV: shape,format,rows,us) or a kernel_sweep.sh whole-model sweep (CSV: model,weights,
context,rows,ms): within a curve, a row count that costs more than a larger one (beyond --slack), or a step from one row count to the next
that rises much faster than the curve does on average; across formats, a Volta-order cell slower than the stock cell next to it.
  kernel_cliffs.py SWEEP.csv [--slack 0.03] [--jump 1.6]
"""
import argparse, collections, csv, sys

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--slack", type=float, default=0.03, help="tolerated relative noise")
ap.add_argument("--jump", type=float, default=1.6, help="a step this many times the curve's mean step per row is flagged")
a = ap.parse_args()

rows = list(csv.DictReader(l for l in open(a.csv) if not l.startswith("[")))   # skip the backend's device banner
if not rows:
    sys.exit("empty sweep")
val = "us" if "us" in rows[0] else "ms"
key = ("shape", "format") if "shape" in rows[0] else ("model", "weights", "context")
curves = collections.defaultdict(dict)
for r in rows:
    curves[tuple(r[k] for k in key)][int(r["rows"])] = float(r[val])

flags = []
for c, pts in sorted(curves.items()):
    xs = sorted(pts)
    for i, x in enumerate(xs):
        later = [y for y in xs[i + 1:] if pts[y] < pts[x] * (1 - a.slack)]
        if later:
            flags.append((c, x, "non-monotonic: %d rows %.2f costs more than %d rows %.2f" % (x, pts[x], later[0], pts[later[0]])))
    small = [x for x in xs if x <= 16]
    if len(small) > 3:
        span = (pts[small[-1]] - pts[small[0]]) / max(1, small[-1] - small[0])
        for p, q in zip(small, small[1:]):
            step = (pts[q] - pts[p]) / (q - p)
            if span > 0 and step > a.jump * span and pts[q] - pts[p] > a.slack * pts[p]:
                flags.append((c, q, "jump: %d -> %d rows +%.2f (%.1fx the curve's mean step)" % (p, q, pts[q] - pts[p], step / span)))
if "format" in key:
    for (shape, fmt), pts in curves.items():
        if not fmt.endswith(" volta"):
            continue
        stock = curves.get((shape, fmt.replace(" volta", " stock")))
        if not stock:
            continue
        for x, t in sorted(pts.items()):
            if x in stock and t > stock[x] * (1 + a.slack):
                flags.append(((shape, fmt), x, "volta slower than stock: %.2f against %.2f" % (t, stock[x])))

for c, x, msg in flags:
    print("%-45s rows %4d  %s" % (" / ".join(c), x, msg))
print("%d flags over %d curves" % (len(flags), len(curves)), file=sys.stderr)
