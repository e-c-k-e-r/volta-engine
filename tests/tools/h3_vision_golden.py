"""Golden for the H3 text encoder's vision tower (Qwen3-VL ViT with DeepStack): HF transformers on the checkpoint's own visual weights.
Saves the processor's pixel_values / grid, the merged image features and the three deepstack features (FP32).
Run in the exllamav3 venv (torch + transformers), one GPU.   usage: h3_vision_golden.py OUT.safetensors [--w 320] [--h 192] [--seed 1]
"""
import argparse, json, os
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dir", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/text_encoder")
ap.add_argument("--proc", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/processor")
ap.add_argument("--w", type=int, default=320)
ap.add_argument("--h", type=int, default=192)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

rng = np.random.default_rng(a.seed)
small = rng.integers(0, 255, size=(6, 10, 3), dtype=np.uint8)
img = Image.fromarray(small).resize((a.w, a.h), Image.Resampling.BICUBIC)  # smooth-ish test image
imp = AutoImageProcessor.from_pretrained(a.proc)
inp = imp(images=[img], return_tensors="pt")
pix, grid = inp["pixel_values"], inp["image_grid_thw"]
print("pixel_values", tuple(pix.shape), "grid", grid.tolist())

cfg = AutoConfig.from_pretrained(a.dir).vision_config
m = Qwen3VLVisionModel(cfg).to("cuda", torch.float32).eval()
idx = json.load(open(os.path.join(a.dir, "model.safetensors.index.json")))["weight_map"]
sd, handles = {}, {}
for n, f in idx.items():
    if n.startswith("model.visual."):
        if f not in handles:
            handles[f] = safe_open(os.path.join(a.dir, f), framework="pt", device="cpu")
        sd[n[len("model.visual."):]] = handles[f].get_tensor(n).float()
print(m.load_state_dict(sd, strict=False))
with torch.inference_mode():
    out = m(pix.cuda().float(), grid_thw=grid.cuda())
merged = out[0] if isinstance(out, tuple) else out.pooler_output
ds = out[1] if isinstance(out, tuple) else out.deepstack_features
print("merged", tuple(merged.shape), "deepstack", [tuple(d.shape) for d in ds])
save_file({"pixel_values": pix.float().contiguous(), "grid": grid.int().contiguous(), "merged": merged.float().cpu().contiguous(),
           **{f"deepstack{i}": d.float().cpu().contiguous() for i, d in enumerate(ds)}}, a.out)
print("saved", a.out)
