"""Golden for the H3 audio VAE encoder (reference-audio latents) from the checkpoint's remote code (run in .venv-1cat13, one GPU).
Saves a synthetic stereo waveform [2, L] and the normalized latent rows [2 * T, 32] (channel-major, the DiT's packed audio order).
The encode follows MiniMaxH3AudioVAE.encode_waveform in 1Cat-vLLM: right-pad to a multiple of 800, DAC encoder, attention projection, mean_proj, (x - mean) / std.

usage: h3_audio_enc_golden.py OUT.safetensors [--seconds 3.0] [--seed 1]
"""
import argparse, json, math, os
import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dir", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/audio_vae")
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.dir, "config.json")))
cls = get_class_from_dynamic_module(cfg["auto_map"]["AutoModel"], a.dir)
with torch.device("cpu"):
    vae = cls.from_pretrained(a.dir)
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
vae.eval().to("cuda", torch.float32)
m = vae.model

n = int(round(a.seconds * 32000))
g = torch.Generator().manual_seed(a.seed)
t = torch.arange(n, dtype=torch.float32) / 32000
wave = torch.stack([0.3 * torch.sin(2 * math.pi * 220 * t) + 0.05 * torch.randn(n, generator=g),
                    0.25 * torch.sin(2 * math.pi * 330 * t + 1.0) * (0.5 + 0.5 * torch.sin(2 * math.pi * 3 * t)) + 0.05 * torch.randn(n, generator=g)])
with torch.inference_mode():
    audio = m.preprocess(wave.cuda().unsqueeze(1), 32000)
    lat = m.encoder(audio)
    lat = m.pre_block(lat.transpose(1, 2)).transpose(1, 2)
    lat = m.mean_proj(lat).float().cpu()  # [2, 32, T]
T = lat.shape[-1]
mean = torch.tensor(cfg["latents_mean"]).view(1, 32, 1)
std = torch.tensor(cfg["latents_std"]).view(1, 32, 1)
rows = ((lat - mean) / std).permute(0, 2, 1).reshape(-1, 32).contiguous()
print("samples", n, "T", T, "rows", tuple(rows.shape), "rms", float(rows.pow(2).mean().sqrt()))
save_file({"wave": wave.contiguous(), "rows": rows}, a.out)
