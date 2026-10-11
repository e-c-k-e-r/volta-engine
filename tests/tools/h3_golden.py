"""Golden capture for the H3 port: packed-layout inputs from 1Cat's builder, outputs from h3_reference.

Run with ~/Programs/v100-skinny/.venv-1cat13/bin/python (1Cat's packed_sequence builder is imported so the
layout is the reference one). Output: a safetensors file with the layout, inputs and per-stage tensors.

usage: h3_golden.py OUT.safetensors [--ckpt PATH] [--frames 22] [--h 192] [--w 320] [--text 32] [--seed 1]
"""
import argparse, json, os, sys
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/home/mrq/Programs/1Cat-vLLM")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm.model_executor.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
from vllm.model_executor.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER as PL
from h3_reference import H3Ref

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ckpt", default="/home/mrq/Documents/models/Comfy-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
ap.add_argument("--cfg", default="/home/mrq/Documents/models/MiniMax-H3/FL2VA/transformer/config.json")
ap.add_argument("--frames", type=int, default=22)
ap.add_argument("--h", type=int, default=192)
ap.add_argument("--w", type=int, default=320)
ap.add_argument("--text", type=int, default=32)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--keyframe", default="")  # "", "0", "-1", "0,-1"
ap.add_argument("--layers", type=int, default=0, help="limit block count (0 = all)")
a = ap.parse_args()

cfg = json.load(open(a.cfg))
if a.layers:
    cfg["num_layers"] = a.layers
frames = PL.align_frame_count(a.frames)
lt = PL.video_latent_t(frames)
at = PL.audio_latent_t(frames / 24)
lh, lw = a.h // 16, a.w // 16
kf = [int(v) for v in a.keyframe.split(",")] if a.keyframe else []
packed = minimax_h3_packed_sequence(
    text_len=a.text, latent_t=lt, latent_h=lh, latent_w=lw, audio_t=at,
    include_keyframe_cond=bool(kf), keyframe_frame_indices=kf or None, frame_count=frames if kf else None)
S = int(packed["seq_len"])
used = int(packed["cu_seqlens"][1])
print(f"frames {frames} latent_t {lt} audio_t {at} lat {lh}x{lw} seq {S} used {used}")

g = torch.Generator().manual_seed(a.seed)
img_pos, audio_pos, text_pos = packed["img_pos"], packed["audio_pos"], packed["text_pos"]
upd = packed["update_mask"]
x = torch.zeros(S, 96)
x[img_pos] = torch.randn(len(img_pos), 96, generator=g)
ax = torch.zeros(S, 32)
ax[audio_pos] = torch.randn(len(audio_pos), 32, generator=g)
prompt = torch.randn(a.text, 5120, generator=g) * 3.0
tags = packed["token_tags"]

outs = {}
ref = H3Ref(a.ckpt, cfg)
steps = [(0.9, 0.85, 0.01, 0.01), (0.5, 0.45, 0.01, 0.01)]  # t_video, t_audio, imgvid_cond_t, audio_ref_t
for si, (tv, ta, tc, tr) in enumerate(steps):
    ts = torch.full((S,), tv)
    ts[img_pos[upd]] = tv
    ts[img_pos[~upd]] = tc
    ts[audio_pos] = ta
    uniq, inv = torch.unique(ts, sorted=True, return_inverse=True)
    inp = dict(x=x, audio_x=ax, prompt_embeds=prompt, img_position_ids=packed["img_position_ids"].float(),
               img_pos=img_pos, audio_pos=audio_pos, text_pos=text_pos, update_mask=upd, token_tags=tags,
               unique_timesteps=uniq, inverse_indices=inv, used=used)
    dump = {}
    v, au = ref.forward(inp, dump)
    outs[f"s{si}.unique_timesteps"] = uniq
    outs[f"s{si}.inverse_indices"] = inv.int()
    outs[f"s{si}.video_out"] = v.cpu()
    outs[f"s{si}.audio_out"] = au.cpu()
    for k, t in dump.items():
        outs[f"s{si}.{k}"] = t.cpu()
    print(f"step {si}: video |v| {v.abs().mean():.4f} audio |a| {au.abs().mean():.4f}", flush=True)

meta = {"frames": str(frames), "latent_t": str(lt), "audio_t": str(at), "latent_h": str(lh), "latent_w": str(lw),
        "seq_len": str(S), "used": str(used), "text_len": str(a.text), "layers": str(cfg["num_layers"])}
tensors = dict(x=x, audio_x=ax, prompt_embeds=prompt, img_position_ids=packed["img_position_ids"].float(),
               img_pos=img_pos.int(), audio_pos=audio_pos.int(), text_pos=text_pos.int(),
               update_mask=upd.to(torch.uint8), token_tags=tags.int(), **outs)
save_file({k: v.contiguous() for k, v in tensors.items()}, a.out, metadata=meta)
print("saved", a.out)
