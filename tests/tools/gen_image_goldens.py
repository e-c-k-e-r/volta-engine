#!/usr/bin/env python3
"""Writes tests/data/image_cases.json (+ PNGs): Qwen2-VL smart_resize, PIL bicubic resize and patch layout references."""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw

out = os.path.join(os.path.dirname(__file__), "..", "data")
P, M, T = 16, 2, 2

def smart_resize(h, w, f=32, mn=65536, mx=1048576):
	hb, wb = round(h / f) * f, round(w / f) * f
	if hb * wb > mx:
		b = math.sqrt(h * w / mx)
		hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
	elif hb * wb < mn:
		b = math.sqrt(mn / (h * w))
		hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
	return max(hb, f), max(wb, f)

def patches(img):
	h, w = smart_resize(img.height, img.width)
	if (h, w) != (img.height, img.width):
		img = img.resize((w, h), Image.BICUBIC)
	a = (np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0 - 0.5) / 0.5
	a = a.transpose(2, 0, 1)                     # C H W
	a = np.stack([a] * T)                        # T C H W
	gh, gw = h // P, w // P
	a = a.reshape(1, T, 3, gh // M, M, P, gw // M, M, P)
	a = a.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
	return gh, gw, a.reshape(gh * gw, 3 * T * P * P)

def scene(w, h, seed):
	rng = np.random.default_rng(seed)
	img = Image.new("RGB", (w, h), (30, 60, 90))
	d = ImageDraw.Draw(img)
	for i in range(12):
		x0, y0 = int(rng.integers(0, w - 8)), int(rng.integers(0, h - 8))
		d.rectangle([x0, y0, x0 + int(rng.integers(4, w // 2)), y0 + int(rng.integers(4, h // 2))], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
	d.text((4, 4), "volta", fill=(255, 255, 255))
	return img

cases = []
for name, (w, h) in { "small": (100, 60), "square": (224, 224), "wide": (640, 200), "tiny": (20, 12), "big": (1600, 1200) }.items():
	img = scene(w, h, len(cases) + 1)
	img.save(os.path.join(out, f"image_{name}.png"))
	gh, gw, p = patches(img)
	cases.append({ "file": f"image_{name}.png", "width": w, "height": h, "grid_h": gh, "grid_w": gw,
		"row_sums": [round(float(v), 3) for v in p.sum(axis=1)[:: max(1, len(p) // 64)]], "stride": max(1, len(p) // 64),
		"head": [round(float(v), 4) for v in p[0][:16]], "mean": round(float(p.mean()), 5) })
json.dump(cases, open(os.path.join(out, "image_cases.json"), "w"), indent=1)
print(len(cases), "cases")
