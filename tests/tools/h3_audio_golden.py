"""Golden for the H3 audio VAE decoder from the checkpoint's remote code (run in .venv-1cat13, one GPU).
Saves latent rows in the DiT order [2*T, 32] (normalized), and the decoded stereo waveform [2, L] (FP32).

usage: h3_audio_golden.py OUT.safetensors [--t 37] [--seed 1]
"""
import argparse, json, os
import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dir", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/audio_vae")
ap.add_argument("--t", type=int, default=37)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.dir, "config.json")))
cls = get_class_from_dynamic_module(cfg["auto_map"]["AutoModel"], a.dir)
with torch.device("cpu"):
    vae = cls.from_pretrained(a.dir)
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
vae.eval().to("cuda", torch.float32)
g = torch.Generator().manual_seed(a.seed)
rows = torch.randn(2 * a.t, 32, generator=g) * 0.15  # small: random latents at unit scale saturate the output clamp
lat = rows.reshape(2, a.t, 32).permute(0, 2, 1).contiguous().cuda()  # [2, 32, T]
mean = torch.tensor(cfg["latents_mean"]).view(1, 32, 1).cuda()
std = torch.tensor(cfg["latents_std"]).view(1, 32, 1).cuda()
with torch.inference_mode():
    wave = vae.decode(lat * std + mean)  # [2, 1, L]
print("wave", tuple(wave.shape), float(wave.abs().max()))
save_file({"rows": rows.contiguous(), "wave": wave[:, 0, :].float().cpu().contiguous()}, a.out)
