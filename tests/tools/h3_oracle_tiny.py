"""Cross-check h3_reference against 1Cat's real MiniMaxH3DiTModel on a tiny random config (run in .venv-1cat13, GPU needed).
Output: cosine of video/audio outputs (expected 1.000000)."""
import sys, os, json, torch
sys.path.insert(0, "/home/mrq/Programs/1Cat-vLLM"); sys.path.insert(0, "/home/mrq/Programs/volta-engine/tests/tools")
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29534")
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
_cm = set_current_vllm_config(VllmConfig()); _cm.__enter__()
initialize_model_parallel(1, 1)
from vllm.model_executor.models.minimax_h3.transformer import MiniMaxH3DiTModel
from vllm.model_executor.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
from safetensors.torch import save_file
from h3_reference import H3Ref
cfg = dict(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=128,
           ffn_hidden_size=512, latents_dim=24, audio_latents_dim=32, patch_size=[1,2,2], text_dim=64,
           adaln_curve_grid=33, adaln_curve_dim=8, adaln_out_features=18*256, final_adaln_out_features=2*256, rope_inv_freq_len=16,
           norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5)
torch.manual_seed(0)
m = MiniMaxH3DiTModel(cfg).cuda()
ck = {}
with torch.no_grad():
    for k, p in m.state_dict().items():
        if k == "rope.inv_freq": v = 1.0 / (10000.0 ** (torch.arange(0, 32, 2).float() / 32))
        elif k.endswith("norm.weight") or "norm1" in k or "norm2" in k or "q_norm" in k or "k_norm" in k: v = 1 + 0.1 * torch.randn(p.shape)
        elif "adaln_proj" in k and k.endswith("weight"): v = 0.3 * torch.randn(p.shape)
        elif k == "adaln_t_table": v = torch.randn(p.shape)
        elif k.endswith("bias"): v = 0.1 * torch.randn(p.shape)
        else: v = (0.5 / p.shape[-1] ** 0.5) * torch.randn(p.shape)
        v = v.to(p.dtype)
        if k.startswith("blocks.") and "adaln" in k and k.endswith("bias"): v = v * 0.5
        p.copy_(v.to(p.device)); ck[k] = v.cpu().contiguous()
save_file(ck, os.path.join(os.environ.get("TMPDIR","/tmp"),"h3_tiny_ckpt.safetensors"))
frames, lt, lh, lw, at, tl = 22, 7, 4, 6, 37, 11
packed = minimax_h3_packed_sequence(text_len=tl, latent_t=lt, latent_h=lh, latent_w=lw, audio_t=at, include_keyframe_cond=False)
S = int(packed["seq_len"]); used = int(packed["cu_seqlens"][1])
g = torch.Generator().manual_seed(3)
img_pos, audio_pos, text_pos, upd, tags = packed["img_pos"], packed["audio_pos"], packed["text_pos"], packed["update_mask"], packed["token_tags"]
x = torch.zeros(S, 96); x[img_pos] = torch.randn(len(img_pos), 96, generator=g)
ax = torch.zeros(S, 32); ax[audio_pos] = torch.randn(len(audio_pos), 32, generator=g)
pe = torch.randn(tl, 64, generator=g)
tv, ta = 0.7, 0.4
ts = torch.full((S,), tv); ts[audio_pos] = ta
uniq, inv = torch.unique(ts, sorted=True, return_inverse=True)
dev = "cuda"
kw = dict(x=x[None].to(dev), audio_x=ax[None].to(dev), img_position_ids=packed["img_position_ids"].float()[None].to(dev),
    unique_timesteps=uniq.to(dev), inverse_indices=inv.to(dev), update_mask=upd.to(dev), token_tags=tags.to(dev),
    prompt_embeds=pe.to(dev), img_pos_info={"position_ids": img_pos.to(dev)}, audio_pos_info={"position_ids": audio_pos.to(dev)},
    text_pos_info={"position_ids": text_pos.to(dev)}, img_pos_for_infer_output_info={"position_ids": img_pos.to(dev)},
    packed_seq_params={"cu_seqlens_q": packed["cu_seqlens"].to(dev), "max_seqlen_q": used, "num_requests": 1},
    refiner_packed_seq_params={"cu_seqlens_q": torch.tensor([0, tl, tl], dtype=torch.int32, device=dev), "max_seqlen_q": tl},
    skip_mask_out_condition=False)
with torch.inference_mode():
    vo, ao = m(**kw)
vo, ao = vo.float().cpu(), ao.float().cpu()
ref = H3Ref(os.path.join(os.environ.get("TMPDIR","/tmp"),"h3_tiny_ckpt.safetensors"), cfg, "cuda")
inp = dict(x=x, audio_x=ax, prompt_embeds=pe, img_position_ids=packed["img_position_ids"].float(), img_pos=img_pos, audio_pos=audio_pos,
           text_pos=text_pos, update_mask=upd, token_tags=tags, unique_timesteps=uniq, inverse_indices=inv, used=used)
rv, ra = ref.forward(inp)
rv, ra = rv.cpu(), ra.cpu()
def cmp(n, a, b):
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"{n}: cos {cos:.6f}  max|d| {(a-b).abs().max():.4g}  mean|ref| {b.abs().mean():.4g}")
cmp("video", vo, rv); cmp("audio", ao, ra)
