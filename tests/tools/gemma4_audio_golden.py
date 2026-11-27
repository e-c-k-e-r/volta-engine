#!/usr/bin/env python3
"""Writes the golden of test_gemma4_audio (and of test_gemma4_vision --llama): llama.cpp's gemma4a encoder on a clip (gemma4v on an image with
--image), as llama-mtmd-cli reports it with MTMD_DEBUG_EMBEDDINGS (the embeddings' shape, mean / std / min / max, the first and last 16 values
of the first token); an image takes up to 280 soft tokens, as the checkpoint's image processor gives it.
  gemma4_audio_golden.py <llama-mtmd-cli> <model gguf> <mmproj gguf> <clip> <out json> [--image]"""
import json, os, re, subprocess, sys

cli, model, mmproj, clip, out = sys.argv[1:6]
env = dict(os.environ, MTMD_DEBUG_EMBEDDINGS="1")
kind = "--image" if "--image" in sys.argv[6:] else "--audio"
text = subprocess.run([cli, "-m", model, "--mmproj", mmproj, "--jinja", kind, clip, "-p", "Transcribe this audio.", "-n", "1", "--temp", "0",
	"-ngl", "99", "-v"] + ( ["--image-max-tokens", "280"] if kind == "--image" else [] ), env=env, capture_output=True, text=True)
log = text.stdout + text.stderr
block = log[log.index("=== MTMD_DEBUG_EMBEDDINGS ==="):log.index("=== END MTMD_DEBUG_EMBEDDINGS ===")]
nums = lambda s: [float(x) for x in re.findall(r"I (-?\d+\.\d+) ", s + " ")]
shape = re.search(r"Shape: \[(\d+), (\d+)\]", block)
first = block[block.index("first 16 values"):block.index("last 16 values")]
last = block[block.index("last 16 values"):block.index("Stats")]
stats = re.search(r"mean=(-?[\d.]+), std=([\d.]+), min=(-?[\d.]+), max=(-?[\d.]+)", block)
g = {"hidden": int(shape.group(1)), "tokens": int(shape.group(2)), "first": nums(first)[-16:], "last": nums(last)[-16:],
	"mean": float(stats.group(1)), "std": float(stats.group(2)), "min": float(stats.group(3)), "max": float(stats.group(4))}
assert len(g["first"]) == 16 and len(g["last"]) == 16, g
json.dump(g, open(out, "w"), indent=1)
print(g)
