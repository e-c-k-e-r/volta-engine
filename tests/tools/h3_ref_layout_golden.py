"""Golden for the Ref2VA packed layout from 1Cat's minimax_h3_packed_sequence_ref2va_blocks (run in .venv-1cat13).
Saves the position ids, tags, row indices and masks for a fixed block configuration; tests/2_parity/test_h3_ref_layout.cpp builds the same one.

usage: h3_ref_layout_golden.py OUT.safetensors
"""
import sys
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/home/mrq/Programs/1Cat-vLLM")
from vllm.model_executor.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks

blocks = [
    {"kind": "image", "latent_h": 24, "latent_w": 32},
    {"kind": "image", "latent_h": 16, "latent_w": 16},
    {"kind": "video_audio", "ref_audio_t": 60, "latent_t": 7, "latent_h": 16, "latent_w": 24},
    {"kind": "video", "ref_audio_t": 0, "latent_t": 2, "latent_h": 12, "latent_w": 20},
    {"kind": "audio", "ref_audio_t": 100},
]
p = minimax_h3_packed_sequence_ref2va_blocks(text_len=20, latent_t=7, latent_h=12, latent_w=20, audio_t=50, ref_blocks=blocks)
print("seq", int(p["seq_len"]), "used", int(p["cu_seqlens"][1]), "img rows", len(p["img_pos"]), "audio rows", len(p["audio_pos"]))
save_file({
    "img_position_ids": p["img_position_ids"].float().contiguous(),
    "token_tags": p["token_tags"].int().contiguous(),
    "img_pos": p["img_pos"].int().contiguous(),
    "audio_pos": p["audio_pos"].int().contiguous(),
    "update_mask": p["update_mask"].to(torch.uint8).contiguous(),
    "audio_update_mask": p["audio_update_mask"].to(torch.uint8).contiguous(),
    "used": p["cu_seqlens"][1:2].int().contiguous(),
}, sys.argv[1])
