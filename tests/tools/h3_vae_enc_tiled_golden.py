"""Golden for the H3 video VAE image encoder with the checkpoint's spatial tiling (encoder_tiling: 256 px tiles, >= 64 px overlap, linear blend of the moments).
Saves enc_in [1, 3, H, W] (normalized image) and enc_moments [48, 1, H/16, W/16] from tiled_encode (run in .venv-1cat13, one GPU).

usage: h3_vae_enc_tiled_golden.py OUT.safetensors [--h 38] [--w 28] [--seed 1]   (h, w in latent cells)
"""
import argparse, json, os
import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dir", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/video_vae")
ap.add_argument("--h", type=int, default=38)
ap.add_argument("--w", type=int, default=28)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.dir, "config.json")))
cls = get_class_from_dynamic_module(cfg["auto_map"]["AutoModel"], a.dir)
with torch.device("cpu"):
    vae = cls.from_pretrained(a.dir)
vae.eval().to("cuda", torch.float32)
m = vae.model
m.parallel_tiling = False
print("encoder_tiling", m.encoder_tiling, "tile", m.tile_size, m.tile_overlap_min)
gi = torch.Generator().manual_seed(a.seed)
img = torch.nn.functional.interpolate(torch.randn(1, 3, 12, 20, generator=gi), size=(a.h * 16, a.w * 16), mode="bilinear", align_corners=False)[0].cuda()
with torch.inference_mode():
    mom = m.tiled_encode(img[None, :, None])  # [1, 48, 1, h, w]
print("moments", tuple(mom.shape))
save_file({"enc_in": img.cpu().contiguous(), "enc_moments": mom[0].float().cpu().contiguous()}, a.out)
