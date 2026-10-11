"""Golden for the H3 video VAE decoder from the checkpoint's own remote code (run in .venv-1cat13, one GPU).
Saves: latent z (normalized, as the DiT produces it), the denormalized z, one raw ViT-decoder tile, and the full tiled/temporal decode.

usage: h3_vae_golden.py OUT.safetensors [--t 7] [--h 12] [--w 20] [--seed 1]
"""
import argparse, json, os, sys
import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dir", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/video_vae")
ap.add_argument("--t", type=int, default=7)
ap.add_argument("--h", type=int, default=12)
ap.add_argument("--w", type=int, default=20)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

cfg = json.load(open(os.path.join(a.dir, "config.json")))
cls = get_class_from_dynamic_module(cfg["auto_map"]["AutoModel"], a.dir)
with torch.device("cpu"):
    vae = cls.from_pretrained(a.dir)
vae.eval().to("cuda", torch.float32)
m = vae.model
print("decoder", type(m.decoder).__name__, "tiling", m.decoder_tiling, "tile", m.decoder_tile_size, m.decoder_tile_overlap_min, "clip", m.clip_length, "drop", m.token_drop)
print("isolated", m.isolated_first_frame, m.isolated_last_frame, "cfg keys", {k: cfg[k] for k in cfg if k.startswith("vae_")})
g = torch.Generator().manual_seed(a.seed)
z = torch.randn(1, 24, a.t, a.h, a.w, generator=g)
mean = torch.tensor(cfg["latents_mean"]).view(1, 24, 1, 1, 1)
std = torch.tensor(cfg["latents_std"]).view(1, 24, 1, 1, 1)
zd = (z * std + mean).cuda()
out = {"z": z.contiguous(), "z_denorm": zd.cpu().contiguous()}
with torch.inference_mode():
    # one raw decoder tile: first 5 latent frames, 12 x 16 latent cells
    tile = zd[:, :, :5, :, :16]
    out["tile_in"] = tile.cpu().contiguous()
    out["tile_out"] = m.decode(tile).float().cpu().contiguous()
    dec = m.decode_base(zd)
    print("decode_base", tuple(dec.shape))
    frames = m.processor.revert_tensor(dec)
    print("revert", tuple(frames.shape), frames.dtype, float(frames.min()), float(frames.max()))
    out["decoded"] = dec.float().cpu().contiguous()
    out["frames"] = frames.float().cpu().contiguous()
save_file(out, a.out)
print("saved", a.out, {k: tuple(v.shape) for k, v in out.items()})
