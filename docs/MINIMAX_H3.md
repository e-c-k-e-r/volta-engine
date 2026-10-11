# MiniMax-H3 port: inventory and roadmap

Status (2026-10-11): text-to-video-and-audio runs end to end on one GPU (section 9). Written from a read of 1Cat-vLLM (`~/Programs/1Cat-vLLM`, branch
`backup/sm70-skinny-port-20260925`, commit 5efdba966) and the checkpoints under `~/Documents/models`. Numbers marked (est) are
back-of-envelope from the configs, not measurements. Numbers quoted from 1Cat are theirs (TP4, 4x V100, single request).

Scope: CUDA only, SM70, the video+audio generation path (T2VA, FL2VA with first / last / both keyframes, Ref2VA), the distilled
turbo LoRAs, multi-LoRA. Not in scope: ComfyUI integration, 1Cat's web UI (the engine gets its own), FP8/SAGE/Skip-Softmax recipes,
TeaCache/Cache-DiT, FastH3 VSA (fails 1Cat's own numerical diagnostic), CPU and Vulkan.

## 1. What H3 is

A rectified-flow diffusion transformer that denoises video and audio latents jointly. Text, image/video/audio references and the
two latent streams are packed into one token sequence; the model is CFG-distilled, so there is one DiT forward per step and no
guidance batch. There is no KV cache, no causal mask and no token streaming. A request is minutes of batch compute, then an MP4.

Pipeline (`MiniMaxH3Pipeline`, `pipeline.py`):

| Stage | Component | Size on disk | Runs |
|---|---|---|---|
| 1 | Qwen3-VL text encoder (64-layer dense LM, hidden 5120, FFN 25600, 64 heads / 8 KV, head_dim 128, mRoPE interleaved [24,20,20], 27-layer vision tower with deepstack at 8/16/24) | 114 GB BF16, 14 shards | once per request (prompt cache keyed by prompt+media) |
| 2 | Reference conditioning: video VAE encoder (3D conv, causal), audio VAE encoder, image/video preprocessing | in the VAE files | only FL2VA keyframes and Ref2VA |
| 3 | DiT: 50 blocks + 2 token-refiner blocks, hidden 5376, 56 heads x 128, FFN 14336 (fc1 28672 wide, gated), patch (1,2,2), 24 video / 32 audio latent channels | 21 GB INT8 (Comfy "pruned convrot") | 4 or 8 steps with turbo LoRA, 49 without |
| 4 | Euler (eta 0) rectified-flow update, sigma schedule with flow shift (video 12.0, audio 3.0), per-step reset of conditioning rows | none | per step |
| 5 | Video VAE decode: ViT decoder (36 blocks, dim 2048 = 32 heads x 64, gated SiLU FFN, RMSNorm, RoPE), spatial 16x, temporal 4x | 10.4 GB (source file) | once, tiled |
| 6 | Audio VAE decode: DAC / BigVGAN style 1D conv stack (snake activations, alias-free resampling), 32 kHz stereo | 0.6 GB | once |
| 7 | Mux to MP4 (24 fps video + PCM audio) | | once |

The video decoder is a ViT, not a 3D conv stack. The 3D convolutions (35 five-dimensional weights, ch 128, ch_mult 1,2,2,4,4,8,
reflect padding, time-isolated GroupNorm) are in the encoder only, which matters for keyframes and Ref2VA but not for plain T2VA.
This reduces the VAE risk compared with a first read.

### Sequence length and cost (est)

1280x736 canvas, 124 frames at 24 fps: latent 80x46 per frame, patch 2x2 gives 920 tokens/frame; ~31-32 latent frames gives about
29K video tokens, plus audio, text and condition tokens. Per DiT forward: linear layers 2 x 19.3B x 29K = ~1.1 PFLOP, attention
4 x N^2 x 7168 x 50 layers = ~1.2 PFLOP; ~2.3 PFLOP/step. 1Cat measures 58 s for four steps at ~53 TFLOP/s per card on four cards,
so the workload is compute-bound and the target is HMMA efficiency, not bandwidth. Ref2VA sequences are longer (181 s for four steps).

## 2. Checkpoints on this machine

| Path | Content |
|---|---|
| `MiniMax-H3/FL2VA`, `MiniMax-H3/Ref2VA` | diffusers layout: text_encoder (114 GB, shared), video_vae, audio_vae, tokenizer, processor, `transformer/config.json` only (no DiT weights) |
| `Comfy-H3/diffusion_models/minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors` | the only DiT weights on disk: 20.97 GB each |
| `h3-turbo/*.safetensors` | six LightX2V turbo LoRAs (rank 128, BF16, 1.4 GB each; 4-step v1.2 768p, 8-step v1.0 768p and non-768p, Ref2V 4-step v0.1 and 8-step v1.0 768p) |
| `h3-loras/*.safetensors` | five community LoRAs (1.2 GB total) |

DiT checkpoint layout (FL2VA file, 932 tensors): 200 INT8 weight matrices (19.27 GB; qkv_proj [21504,5376], out_proj [5376,7168],
fc1 [28672,5376], fc2 [5376,14336] per block) each with a per-output-row FP32 scale `[N,1]` and a `.comfy_quant` metadata blob
(format `int8_tensorwise`, `convrot`, `convrot_groupsize` 256); the two token-refiner blocks, norms, and q/k norms are BF16; the
patch/condition/time projections are FP32. The AdaLN is "pruned": `blocks.N.adaln_proj.linear` is [96768, 8] (18 x 5376 outputs
from an 8-dim coordinate) fed by `adaln_t_table` [1025, 8], instead of the dense time embedder. The pruned DiT is the only base
available locally, which fixes the reference for every parity gate: W8A16 convrot, not the original floating weights (1Cat's
original-floating runs are not reproducible here without a download).

ConvRot means the stored weight is already Hadamard-rotated (groups of 256 along K) and row-quantized, so the activation must be
rotated with the same Hadamard before the FP16 GEMM. The engine has `hadamard.cu` (EXL3 trellis) which may be reusable for the
256-group rotation; that needs checking, the sign conventions will not match automatically.

Turbo / community LoRAs are PEFT-style A/B pairs on `to_q/to_k/to_v/to_out.0` and `ff.net.0.proj / ff.net.2` for all 50 blocks and
the token refiner (624 tensors, alpha 8, rank 128). They are small (one is 1.4 GB), so unmerged staged application (x A^T B) is
cheap relative to the base GEMM and keeps the INT8 base intact; merging into INT8 weights is not possible. Multi-LoRA is therefore
a sum of staged deltas with per-request scales.

### 2.1 Measured kernel baseline (2026-10-11, one V100 at the 150 W cap, 1Cat's built kernels, single runs)

"W8A16" in 1Cat means the INT8 weights are dequantized to FP16 at load (`dequantize`, `rotate` for the ConvRot butterfly) and the
projections then run as plain FP16 GEMMs (`ops.gemm` is cuBLAS; `ColumnMajorGemmPlan` is a CUTLASS-style plan for column-major
weights). Weights are resident as FP16: 38.6 GB, 9.7 GB per card at TP4. No INT8 compute kernel is needed.

M = 29,440 tokens, TP4 per-card shapes (script: `tests/3_bench/h3_kernel_baseline.py`):

| Op | N x K | cuBLAS | 1Cat column-major plan |
|---|---|---|---|
| qkv | 5376 x 5376 | 42.2 ms, 40.3 TF | 35.9 ms, 47.5 TF |
| out | 5376 x 1792 | 14.2 ms, 39.9 TF | 12.3 ms, 46.3 TF |
| fc1 | 7168 x 5376 | 55.5 ms, 40.9 TF | 48.2 ms, 47.1 TF |
| fc2 | 5376 x 3584 | 28.3 ms, 40.1 TF | 24.4 ms, 46.5 TF |

Non-causal attention, d=128, 14 heads per card: 170 ms (36.5 TF, query tile 128 / key tile 128), 197 ms with key tile 64, 202 ms
with query tile 64. A full-head (56) run gives 689 ms.

Per layer per card: GEMMs ~121 ms (column-major) + attention ~170 ms, so ~14.5 s per step over 50 layers, which matches 1Cat's
14.57 s per step. Attention is about 58% of the step. Consequences: cuBLAS is the baseline for the engine's GEMMs (the column-major
plan is worth ~15% and can come later); attention is where a better kernel pays most; the card is at 40-47 TF of a 125 TF peak,
and the 150 W cap limits what any kernel can reach, so absolute figures here are caps, not kernel limits.

## 3. Inventory of the 1Cat source

Python sources (branch `backup/sm70-skinny-port-20260925`; the SM70 kernels are byte-identical to `main`, the difference is
Python: this branch has the LoRA, protocol and UI work, `main` has `prepared_weights.py`):

| 1Cat file | Lines | Role | Engine disposition |
|---|---|---|---|
| `transformer.py` | 1737 | DiT model, token refiner, final layer, SP/residual-parallel paths | rewrite as C++ module; the block is simple (section 4) |
| `pipeline.py` | 2311 | request lifecycle, shape resolution, conditioning, residency, adapters | rewrite; mostly orchestration |
| `denoise_loop.py` | 399 | Euler loop with per-step condition reset | port directly (short) |
| `scheduling_*euler_ancestral.py`, `sigma_schedule.py`, `time_request.py` | ~410 | sigma schedules, DMD2 turbo schedule, flow shift | port; golden-compare the schedule arrays |
| `encoder.py` | 1751 | Qwen3-VL encoder with TP layers, vision tower, mRoPE index | reuse engine Qwen3 vision tower (`vision.cu`) and a new dense-Qwen3 text stack; mRoPE index is new |
| `vae.py`, `reference_video.py`, `preprocessing.py` | ~1900 | tiled VAE wrappers, reference video/image preprocessing | port logic; the VAE math is in the checkpoint's `video_vae/*.py` |
| `lora.py` | 1040 | adapter formats (LightX2V, FlashGen grouped QKV, FastH3 fusion), staged A/B | port LightX2V + generic PEFT; FlashGen and FastH3 are optional later |
| `quantization.py`, `ops.py`, `weights.py`, `comfy_checkpoint.py` | ~940 | ConvRot INT8 handling, Comfy key mapping | port the key mapping and rotation contract |
| `modulation.py`, `conditioning.py`, `condition_noise.py`, `packed_*.py`, `attention.py`, `qk_norm_rope.py`, `activation.py` | ~1700 | AdaLN modulation, token tagging and packing, 3D RoPE, QK norm | port as small kernels plus host packing |
| `residency.py`, `weight_cache.py`, `collectives.py`, `prepared_weights.py` | ~1300 | CPU/GPU residency, peer reduction | replaced by the engine's arena and NVLink comm layer |
| `vllm/video/{engine,server,protocol,media,gpu,fastpath}.py` | ~900 | request/task API, media IO | replace with a job API in the engine's HTTP server; the 187-line `protocol.py` addition is the field list for multi-LoRA requests |
| `web_ui_html.py`, `tools/minimax_h3/web_ui.py` | ~1440 | web UI | not ported (new UI, section 6) |

SM70 kernels in 1Cat:

| File | Lines | Role |
|---|---|---|
| `flash-attention-v100/kernel/h3/{fmha.h,forward.cu,prefetch_mma.h,default_fmha.h}` | ~1630 | non-causal FlashAttention forward for SM70, head_dim 128; the retained default, +4.4% denoise over FlashInfer |
| `flashinfer-sm70/csrc/h3_noncausal_sm70.cu` | 353 | alternative non-causal kernel (register P); 1Cat keeps it as the fallback |
| `csrc/sm70_turbomind/ops/h3_w8a16.cu`, `h3_column_major_gemm.h` | ~410 | INT8-weight x FP16-activation GEMM for large M (CUTLASS/TurboMind derived) |
| `csrc/minimax_reduce_rms_kernel.cu` | | TP reduce + RMS fusion |
| `forward_sparse.cu`, `vsa_layout.h`, `h3_vsa_fp32.cu` | | FastH3 VSA, out of scope |

Tests worth lifting as parity references (they define expected behavior): `test_h3_numerics.py` (536), `test_h3_lora.py` (611),
`test_h3_workflows.py`, `test_h3_flashattn.py`, `test_h3_qk_norm_rope.py`, `test_h3_column_major.py`.

Licensing: the model code is adapted from Apache-2.0 vLLM-Omni (provenance in `UPSTREAM.md`); keep the SPDX notices on anything
derived and write the C++ from the algorithm, not by transliteration, where practical.

## 4. DiT block (what has to exist)

Per block, tokens T x 5376, tags per token (video / text / audio, padding -1):

1. `adaln_proj(t_emb)` yields shift/scale/gate for attention and MLP per modality (3 modalities, 18 x 5376 outputs; per-token
   selection through `combined_indices = timestep_index * 3 + modality`).
2. RMSNorm(norm1) fused with indexed scale/shift, FP16 output.
3. Hadamard rotation (ConvRot) of the activation, then qkv_proj (INT8 W, FP16 A) -> [T, 3 x 7168]. The grouped-QKV reorder
   (`_reorder_grouped_qkv_to_qkv`) applies to some adapter layouts.
4. Per-head RMSNorm of q and k, 3D RoPE (rope table of `rope_inv_freq_len` 16 over frame/row/col positions), non-causal attention
   (56 heads x 128, full sequence; segment isolation for packed requests only matters if batching, which is out of scope).
5. out_proj (INT8, K = 7168), then residual `indexed_gate_rms_norm_scale_shift` (gate*h + residual, norm2, scale/shift).
6. fc1 (28672 wide, gated SiLU -> 14336), fc2, `indexed_gate` residual.
7. Residual stream in FP32 under sequence-parallel (1Cat found NCCL reduce-scatter's summation order fails their latent gate and
   uses a custom peer reduction that keeps the replicated FP32 order; the engine's own all-reduce is deterministic, which should
   be checked against this).

Outside the blocks: video/audio patch projections and condition projection (FP32), time embedding table lookup, final layer
(AdaLN + linear to 24x4 video / 32 audio channels, FP32), the 2-block token refiner over text tokens, rope table construction.

Numerics warning from the source: H3 has outlier activations (per-element amax in the hundreds), so FP16 QK^T can exceed 65504;
1Cat keeps FP32 for several projections and for the residual. Parity has to be checked with the same precision choices before any
kernel is optimized.

## 5. Gap analysis against the engine

| Need | Engine today | Work |
|---|---|---|
| Large-M projection GEMM | cuBLAS is linked; weights are dequantized INT8 -> FP16 at load (section 2.1) | cuBLAS first; later optionally port the column-major plan (~15%); INT8 dequant + ConvRot at load is a small kernel |
| Non-causal FlashAttention, d=128, ~30K tokens | `attention.cu` is causal/decode-oriented | new: port `flash-attention-v100` h3 forward (the dominant cost, ~58% of a step); keep parity test against an FP32 reference |
| AdaLN indexed norm/scale/shift/gate kernels | `norm.cu`, `gated_residual.cu` are close | small new kernels |
| 3D RoPE + QK norm | `rope.cu` has 1D/mRoPE variants | new rope table + fused qk-norm-rope |
| ConvRot Hadamard (group 256) | `hadamard.cu` (EXL3) | verify reuse; otherwise new |
| TP4 / sequence-parallel, deterministic FP32 residual | NVLink all-reduce exists (decode-sized) | extend to large payloads (T x 5376 FP16/FP32, reduce-scatter/all-gather); correctness before speed |
| Dense Qwen3-VL text stack (64 layers, 5120 hidden) | `qwen3_5` is the hybrid GDN model; no plain dense Qwen3 | new model class or generalize `qwen3_5` attention-only layers; weights via EXL3 |
| Vision tower | `vision.cu`, `qwen3_vision.h` | reuse; check deepstack taps and 16-px patches vs Qwen3 27B tower |
| mRoPE position index for text+image+video tokens | present for Qwen vision input | verify equivalence with `_get_rope_index` |
| Video VAE decoder ViT (36 blocks, 2048) | none | new: reuses the same GEMM/attention/RoPE/RMSNorm primitives; plus unpatchify and tiling with overlap blending |
| Video VAE encoder (3D conv, reflect pad, time-isolated GN) | none (no conv3d at all) | new conv3d kernels (implicit-GEMM); needed only for keyframes and Ref2VA |
| Audio VAE (snake, alias-free resample, conv1d, BigVGAN) | none | new conv1d / transposed-conv1d kernels and activations; decoder first, encoder for audio references |
| LoRA staged delta | none | two skinny-rank GEMMs per target; trivial next to the base GEMM |
| Job/task API, MP4 mux, image/video decode | HTTP server speaks chat protocols | new `engine::video` target; ffmpeg/libav for IO is the pragmatic exception to "no framework", to be decided |
| Memory plan | static arena, one model | DiT and encoder cannot be resident together with activations; load order and offload need a plan (section 7) |

## 6. Architecture proposal

* New namespace `engine::video`, separate from the LM generator: `H3Pipeline` owns components and a request state machine; the
  existing `Model`/slot machinery is not reused. Reuse the arena, comm layer, safetensors loader, tokenizer and vision tower.
* No CUDA graphs for the DiT: at M ~ 30K each kernel runs for milliseconds, so launch overhead is negligible. The static-arena
  rule still applies.
* Parallelism: TP4 with sequence-parallel FP32 residual (as 1Cat measured), unless single-GPU data-parallel wins on throughput:
  the INT8 DiT is 21 GB and fits one 32 GB card with ~8 GiB left, tight for a 30K-token sequence (est: fc1 output alone is
  30K x 28672 x 2 B = 1.7 GB, attention workspace, FP32 residual 0.6 GB). Measure both before committing.
* Text encoder: quantize to EXL3 (about 4 bpw gives ~30 GB) with the exllamav3 pipeline so the existing trellis kernels run it;
  it runs once per request on a few hundred tokens (skinny GEMMs, the engine's strength). Check output drift against the BF16
  hidden states that feed `condition_proj`; the DiT may be sensitive to it, so this is a quality gate, not a given.
* Own web UI: a single static page plus job endpoints (submit, progress via SSE with per-step latents preview optional, list,
  download, delete) with LoRA picker, per-LoRA scale, seed, resolution, frame count, keyframe and reference uploads. Replaces
  1Cat's; designed after the API settles.

## 7. Roadmap

Each phase ends with a gate; phases are ordered so something runs end-to-end as early as possible. Session counts are guesses for
focused work.

| Phase | Deliverable | Gate | Sessions |
|---|---|---|---|
| H0 | Golden capture: run 1Cat's pipeline once at a small shape (e.g. 33 frames, 512x320) and dump fixed-seed tensors: encoder hidden + tags, patch-embedded rows, outputs of block 0, 1, 49, token refiner, final velocity per step, final latents, decoded RGB and PCM; also sigma arrays | files on disk, reproducible from a script in `tests/tools/` | 0.5 (needs the vLLM environment working) |
| H1 | DiT core, single GPU, small shape: loader for the Comfy INT8 file + key map, ConvRot rotation, W8A16 GEMM, AdaLN kernels, QK-norm + 3D RoPE, non-causal attention, token refiner, final layer | per-block cosine >= 0.9999 vs H0 golden; velocity of step 0 within 1Cat's own numeric tolerance | 2-3 |
| H2 | Full-size DiT on 4 GPUs: TP4 + SP, FP32 residual order, memory plan, T2VA 4-step loop with the Euler scheduler fed H0's encoder hidden and decoded by 1Cat's VAE offline | final latents match golden; denoise time vs 1Cat's 58 s at 1280x736x124 (target: within 10%, then beat it) | 2 |
| H3 | Turbo LoRA (LightX2V) + multi-LoRA staging, per-request scales, DMD2 schedule; FL2VA keyframes need H5 for the encoder side | latents match golden with and without adapters; `test_h3_lora.py` cases | 1 |
| H4 | Decoders: video VAE ViT decoder + tiling, audio VAE decoder, mux to MP4 | decoded RGB/PCM match golden (PSNR / cosine gates), full T2VA request produces a watchable file | 1-2 |
| H5 | Encoder side: EXL3 quantization of the text encoder, dense Qwen3-VL stack in the engine, tower + mRoPE, prompt cache; then conv3d and conv1d encoders for keyframes and references | hidden-state cosine vs BF16 reference; FL2VA first/last/both and Ref2VA parity vs 1Cat | 2-3 |
| H6 | Job API + new web UI; benchmark matrix (as for the LM models) at a fixed stated power cap with repeats | end-to-end request via UI; documented numbers (cap stated, repeats) | 1-2 |

Total about 10-14 sessions; H0-H4 (T2VA with turbo LoRA, no references) is the first useful milestone at roughly 6-8.

### Overnight-safe pieces (no judgment needed, scripted, GPUs free)

* H0 golden capture, once the 1Cat environment is verified to run a small shape.
* Microbenchmarks of the large-M INT8 GEMM and non-causal attention on the real shapes (M 29K, K 5376, N 21504 / 28672; 56 heads
  x 128): establishes the achievable TFLOP/s on this machine's power cap before writing the engine versions, and decides whether
  porting 1Cat's kernels is enough or the existing `mma.m8n8k4` building blocks are better.
* EXL3 quantization of the text encoder (long exllamav3 job), followed by a hidden-state drift measurement.

## 8. Risks and open questions

1. Sustained throughput: 1Cat's 53 TFLOP/s per card is under the usual 150 W cap here? Their power setup is unknown; the same
   cap caveat as the Flash-Next numbers applies.
2. Text-encoder quantization: DiT quality sensitivity to a 4-bit encoder is untested. Fallback: 8-bit EXL3 (~60 GB) or keep BF16
   in pinned host RAM and stream layers (114 GB does not fit in 128 GB VRAM alongside the DiT).
3. ConvRot reproducibility: the exact Hadamard layout (group 256, ordering, scaling) must be read from `ops.py` and
   `quantization.py`; getting it wrong gives plausible-looking garbage rather than an error.
4. Memory at full shape: 1Cat reports 20.5 GB/card live at TP4 including IPC; the engine arena adds its own reservation. VAE
   decode memory (ViT over the whole latent grid, tiled) is the likely peak.
5. Determinism requirements of 1Cat's residual path (summation order) may constrain the all-reduce design.
6. Video VAE encoder conv3d: reflect padding in time, GroupNorm isolated per frame, causal; the spec is in
   `video_vae/{klvae,vae_cnn,conv}.py` in the checkpoint directory (remote code, 1258 lines in `klvae.py`).
7. The original floating-point DiT is not on disk; parity can only be against the INT8-convrot checkpoint. If original-weight
   quality ever matters, it is a 38 GB download.
8. Not read in detail yet: `modulation.py`, `condition_noise.py`, `packed_sequence.py` (token tagging and layout), `reference_video.py`,
   the audio VAE sources, and the diff between this branch and `main` for `pipeline.py` (270 lines) and `config.py`.
9. Branch divergence: this branch is 1701 commits behind `main`; the kernels are identical, so port from here, but re-diff the
   Python before treating any behavior as final.

## 9. Implementation status (2026-10-11)

Built and checked against the official code (goldens in `~/Documents/h3`, generated by `tests/tools/h3_*.py`):

| Component | Files | Check |
|---|---|---|
| DiT forward, INT8 ConvRot weights resident, per-matrix dequant+unrotate, FP32 residual, CUTLASS FMHA | `src/video/h3_dit.cu`, `inc/engine/video/h3_dit.h` | 50 layers: video cosine 0.999998 vs the FP32 reference, which matches 1Cat's `MiniMaxH3DiTModel` (cosine 1.000000, tiny config); 34K rows: 79 s per forward on one V100 at the 150 W cap |
| Staged LoRA (diffusers layout) | same | with the adapter the match improves 10x |
| Video VAE decoder (ViT3D) with tiling and temporal chunks | `h3_vae.cu` | tile cosine 1.000000; tiled/chunked video PSNR 59 dB |
| Audio VAE decoder (BigVGAN) | `h3_audio.cu` | cosine 1.000000 |
| Text conditioning (Qwen3 tower, 50 layers, streamed BF16) | `h3_text.cu` | cosine 1.000000 vs the reference, which matches HF; tokenizer ids identical; 9 s per prompt |
| Layout, schedule, Euler loop, MP4 mux, `h3_generate` CLI | `h3_pipeline.cu`, `h3_generate_main.cpp` | layout identical to 1Cat's; 320x192 x 22 frames in 26 s, coherent output |

Corrections to the plan above: the video decoder is a ViT (no conv3d); keyframe conditioning of a single image needs only a 2-D reduction of the causal 3-D encoder (one frame sees only the last temporal tap); the text encoder is streamed from disk instead of quantized (49 GB, 9 s per prompt), quantization stays an option.

Not yet built: FL2VA keyframes (VAE image encoder, vision tower with deepstack and image mRoPE in the encoder, image preprocessing, conditioning noise), Ref2VA (video/audio reference encoders), multi-GPU (sequence parallel with replicated INT8 weights is the plan), job server and UI, FP16-range audits for long clips, the benchmark matrix.
