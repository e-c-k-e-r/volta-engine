"""Golden for the H3 video VAE encoder on a clip (reference videos): causal 3-D convs per 17-frame clip, time-isolated GroupNorm, spatial tiling.
Saves vid_in [T, 3, H, W] (ImageNet-normalized frames, trimmed to 17 n + 5) and vid_moments [T', 48, H/16, W/16] from encode_temporal (run in .venv-1cat13, one GPU).

usage: h3_vae_enc_video_golden.py OUT.safetensors [--frames 40] [--h 18] [--w 20] [--seed 1]   (h, w in latent cells)
"""
import argparse, json, os
import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dir", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/video_vae")
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--h", type=int, default=18)
ap.add_argument("--w", type=int, default=20)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.dir, "config.json")))
cls = get_class_from_dynamic_module(cfg["auto_map"]["AutoModel"], a.dir)
with torch.device("cpu"):
    vae = cls.from_pretrained(a.dir)
vae.eval().to("cuda", torch.float32)
m = vae.model
m.parallel_tiling = False
g = torch.Generator().manual_seed(a.seed)
base = torch.randn(1, 3, 6, 12, 20, generator=g)
vid = torch.nn.functional.interpolate(base, size=(a.frames, a.h * 16, a.w * 16), mode="trilinear", align_corners=False)[0].cuda()  # [3, T, H, W]
used = m.processor.get_suitable_video_length(a.frames)
vid = vid[:, :used]
print("frames", a.frames, "used", used)
with torch.inference_mode():
    mom = m.encode_temporal(vid[None])  # [1, 48, T', h, w]
print("moments", tuple(mom.shape))
save_file({"vid_in": vid.permute(1, 0, 2, 3).contiguous().cpu(), "vid_moments": mom[0].permute(1, 0, 2, 3).float().cpu().contiguous()}, a.out)
