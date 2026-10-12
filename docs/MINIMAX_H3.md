# MiniMax-H3 (video + audio generation)

MiniMax-H3 is a rectified-flow diffusion transformer that denoises video and audio latents jointly. The engine runs it on SM70 (V100) GPUs through `h3_generate`
(command line) and through the video service of `volta_server` (job queue, HTTP API, web page). CUDA only. Supported tasks: text to video and audio (T2VA),
first / last / both keyframes (FL2VA), and references (Ref2VA: images, videos, audio), with the distilled turbo adapters and community LoRAs.

Not supported: ComfyUI integration, FP8 / sparse-attention recipes, TeaCache / Cache-DiT, CPU and Vulkan.

## 1. How a job runs

Text, reference rows and the two latent streams are packed into one token sequence. The model is CFG-distilled: one transformer forward per step, no guidance batch, no
KV cache, no causal mask. A request is minutes of batch compute, then an MP4.

| Stage | Component | Notes |
|---|---|---|
| plan | layout, sigma schedule, noise | `h3_pipeline.cu`, `h3_image.cpp`; the noise generator is the engine's own (`H3Gaussian`), so a seed does not reproduce 1Cat's torch noise |
| references | video VAE encoder (3-D conv, causal), image encoder, audio VAE encoder | only FL2VA keyframes and Ref2VA; tiles are spread over the GPUs |
| text | Qwen3-VL language model, first 50 decoder layers, plus its vision tower for images and video frames | BF16 layers streamed from disk, or an EXL3 / NVFP4 conversion resident on the GPUs (section 4.3) |
| denoise | the transformer: 50 blocks + 2 token-refiner blocks, hidden 5376, 56 heads x 128, MLP 14336 (fc1 28672 wide, gated), patch (1,2,2), 24 video / 32 audio latent channels | 4 or 8 Euler steps (eta 0), flow shift 12 / 3 (video / audio) with the turbo adapters, per-step reset of the condition rows |
| decode | video VAE decoder (ViT, 36 blocks of dim 2048, spatial 16x, temporal 4x, tiled), audio VAE decoder (BigVGAN-style 1-D conv stack, 32 kHz stereo) | |
| mux | MP4, 24 fps video + PCM audio, through the `ffmpeg` binary | |

At 1280x736 and 124 frames the sequence is about 34,500 rows (80x46 latent per frame, patch 2x2, ~32 latent frames, plus audio, text and condition rows); one forward is
about 2.3 PFLOP, split roughly evenly between the linear layers and attention.

## 2. Files you need

| Path (example) | Content |
|---|---|
| `MiniMax-H3/FL2VA`, `MiniMax-H3/Ref2VA` | diffusers layout: `text_encoder` (BF16, 14 shards, shared), `video_vae`, `audio_vae`, `tokenizer`, `processor`, `model_index.json` of class `MiniMaxH3Pipeline` |
| `Comfy-H3/diffusion_models/minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors` | the transformer, 21 GB each (the only transformer weights the engine reads) |
| `h3-turbo/*.safetensors`, `h3-loras/*.safetensors` | turbo LoRAs (LightX2V, rank 128) and community LoRAs |
| optional: an EXL3 or NVFP4 conversion of Qwen3-VL-32B-Instruct | resident text tower, section 4.3 |

**Transformer format.** INT8 matrices (`qkv_proj`, `out_proj`, `fc1`, `fc2` of every block) with a per-output-row FP32 scale and a ConvRot `.comfy_quant` blob (Hadamard-rotated
in groups of 256 along K, then row-quantized); the token-refiner blocks, norms and q/k norms are BF16, the patch / condition / time projections FP32. The AdaLN is "pruned":
`blocks.N.adaln_proj.linear` maps an 8-dimensional coordinate fed by `adaln_t_table` to the shift / scale / gate of three modalities. Each projection is dequantized and
un-rotated to FP16 right before its cuBLAS GEMM (FP32 accumulation); the residual stream is FP32.

**Adapters.** PEFT-style A/B pairs on `to_q / to_k / to_v / to_out.0` and `ff.net.0.proj / ff.net.2` (diffusers, kohya and ai-toolkit key layouts are read). They are applied
unmerged (`x A^T B` per target, per-adapter scale, several at once), which keeps the INT8 base intact.

## 3. Running it

```
build/h3_generate --root <MiniMax-H3/FL2VA> --dit <fl2va checkpoint> \
    --lora <h3-turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors> --prompt "..." --size 1280x736 --frames 124 --steps 4 \
    --devices 0,1,2,3 --out out.mp4 [--image first.png [--image last.png]] [--lora other.safetensors:0.8]
build/h3_generate --root ... --dit-ref2va <ref2va checkpoint> --lora <h3-turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors> \
    --prompt "The dog from Picture 1 ..." --ref-image a.png [--ref-image b.png] [--ref-video clip.mp4] [--ref-audio voice.wav] --devices 0,1,2,3 --out out.mp4
build/volta_server --model-dir <MiniMax-H3/FL2VA> --dit <fl2va checkpoint> [--dit-ref2va <ref2va checkpoint>] --lora-dir <h3-turbo> --lora-dir <h3-loras> \
    --out-dir <jobs dir> --host 127.0.0.1 --port 8200 [--devices 0,1,2,3] [--tensor-parallel] [--text-encoder <exl3 dir>]
```

`h3_generate` also takes `--resident` (keep the models loaded between `--repeat N` runs), `--resident-decoders`, `--tp`, `--text-encoder DIR`, `--seed`, `--video-shift`,
`--audio-shift`. The configuration is checked before anything loads (section 5).

**Server.** A `model_dir` with a `model_index.json` of class `MiniMaxH3Pipeline` selects the `minimax_h3` model family: no language-model runtime, a job queue with one worker
thread that owns the GPUs, and these routes: `GET /` (the page), `GET /api/info`, `GET` / `POST /api/jobs`, `GET` / `DELETE /api/jobs/<id>`, `GET /api/jobs/<id>/video` (Range)
and `/thumb`, `POST /api/jobs/<id>/cancel`, `/health`, `/metrics`, `/metrics/prometheus`. A job body is
`{prompt, width, height, frames, steps, seed, turbo, loras:[{file,scale}], images:[{data,frame_index}], ref_images, ref_videos, ref_audios}` with base64 uploads. Settings are in the
launch config's `video` object (see [CONFIG.md](CONFIG.md)). There is no authentication; the server binds to localhost by default. Code: `src/server/video/`.

**References (Ref2VA).** Up to 9 reference images (resized to a 2048 px short edge, 32 px multiples, aspect ratio 0.4 to 2.5), 3 reference videos and 3 audio clips (12 in all;
videos and audio 2 to 15 s each and in total), on the Ref2VA transformer. In the prompt they are `Picture i`, `Video k`, `Audio j`. Reference videos are decoded by ffmpeg onto a
768 px canvas (at most 768 x 1344, 24 fps), encoded by the causal 3-D video VAE in 17-frame clips and shown to the language model as every 12th frame with timestamps; a
video's soundtrack becomes an audio reference. Image references sit at timestep 0.999, audio references at 1.0 (clean latents). Layout:
`[text | reference blocks | target audio | target video]`.

## 4. Architecture

### 4.1 The transformer (`src/video/h3_dit.cu`, `h3_dit_group.cpp`)

Per block: AdaLN-indexed RMSNorm (shift / scale per modality and timestep), qkv projection, per-head RMSNorm of q and k, 3-D RoPE, non-causal attention over the whole
sequence, out projection with a gated residual, fc1 (gated SiLU), fc2 with a gated residual. The SiLU product carries a per-row power-of-two scale that keeps the FP16 value
in range; it is undone after the down projection. There are no CUDA graphs: each kernel runs for milliseconds.

**Attention** is the engine's own kernel (`src/kernels/cuda/h3/h3_attn.cu`, WMMA on Volta): a warp owns 16 query rows with Q and the FP32 output in registers, K / V tiles are
double buffered through shared memory with padded rows, the online softmax runs on the accumulators. head_dim 128 (full and causal) and 64 (the VAE decoder). It reaches about
54 TF/s on the DiT shape. A key-range variant carries the running (max, sum, output) between calls. Register use is at the limit of 255, so small source changes can cost
10% or more: check `bench_h3_attn` after touching it.

**Sequence parallelism (default).** Every GPU holds the full INT8 weights and owns a quarter of the rows. Per layer each rank attends to its own keys while the peers'
K / V slices arrive over NVLink (one copy stream per peer, ordered with events, no host waits on the GPU), then to each peer's slice as it lands. Needs about 21 GB of weights
plus 6 GB of scratch per GPU.

**Tensor parallelism** (`video.tensor_parallel`, `--tensor-parallel`, `h3_generate --tp`; needs NCCL at build time, found by CMake as `ENGINE_HAVE_NCCL`). Every GPU holds a
quarter of the heads and of the MLP width and runs all rows. The ConvRot rotation works in groups of 256 along K, so the K-sharded matrices keep whole groups (out
projection 7168 / 4 = 7 groups, MLP down 14336 / 4 = 14). The attention-output and MLP-down partial products are summed over NCCL in FP32 (two all-reduces per layer); the SiLU
row scale is the maximum over the whole MLP width. Adapters are sliced the same way (column-parallel projections keep the B rows of the rank's slice, row-parallel ones the A
columns). Usable GPU counts: 2, 4 or 7. The point is memory: 14 GB per GPU while denoising instead of 27, so that the text tower and the VAEs can stay loaded. Cost: about 1 s
per step more than sequence parallelism. A prompt may have at most about 8.6K text rows (the full-width token refiner runs on them); longer ones fail the job with a message.

### 4.2 VAEs

*Decoder* (`h3_vae.cu`): a ViT, run on 256 px tiles of every temporal chunk (7 latent frames, 28 tiles at 1280x736), the tiles spread over the GPUs (one decoder per GPU), two tiles
per transformer pass when the card has room, blended with the checkpoint's overlap rules and streamed to RGB8. *Image / video encoder* (`h3_vae_enc.cu`): causal 3-D convolutions
as FP16 im2col + GEMM with valid-tap skipping, tiled like the checkpoint (256 px tiles, at least 64 px overlap), tiles spread over the GPUs. *Audio VAE* (`h3_audio.cu`): a
DAC-style encoder and the BigVGAN decoder (snake activations, alias-free resampling) on conv1d kernels. The decoder weights are read once and broadcast to all GPUs through a
shared pinned ring (`h3_stager.h`).

### 4.3 The text tower (`h3_text.cu`, `h3_text_group.cpp`, `h3_vision.cu`)

Qwen3-VL-32B-Instruct's language model (dense, 5120 hidden, 64 heads / 8 KV heads of 128, MLP 25600, interleaved mRoPE), first 50 layers, no final norm; the MiniMax encoder is the
stock model (`test_h3_text_identity` finds its norms and embedding bit-identical to the public EXL3 and NVFP4 conversions). Three ways to run it:

* **Streamed BF16** (default): layers are read from the `text_encoder` shards through pinned chunks and converted to FP16 on the GPU, one layer ahead of the compute. 5 to 7 s for
  a short prompt.
* **Resident EXL3** (`video.text_encoder`, `--text-encoder DIR` with a conversion such as `turboderp/Qwen3-VL-32B-Instruct-exl3`): the trellis projections stay on the GPUs and
  run through `LinearTrellis`; `H3TextGroup` shards them over the GPUs (16 query heads, 2 KV heads and 6400 MLP columns per GPU on four; column-parallel q/k/v/gate/up,
  row-parallel o/down; FP16 all-reduces of the two partial products per layer; the MLP row scale from a max all-reduce). 6.0 bpw is 18.3 GB for 50 layers, 4.6 GB per GPU on four.
  A short prompt takes about 0.2 s. Hidden-state cosine against the BF16 reference: 0.999999 text only, 0.999958 with an image.
* **Resident NVFP4** (a compressed-tensors conversion such as `Qwen3-VL-32B-Instruct-NVFP4`): same path through the engine's W4A16 kernels, 3.4 GB per GPU on four. Text only
  0.99991, but with an image 0.974: at layer 43 the BF16 and EXL3 runs create a very large activation on one token (channel 731 reaches about 7400) that the 4-bit weights do not
  reproduce (`test_h3_text_layers` shows the divergence layer by layer). Prefer EXL3 6 bpw for image or reference prompts.

The resident tower does not fit next to the full-weight sequence-parallel transformer in 32 GB; use it with `tensor_parallel`. The vision tower is the engine's
`Qwen3VisionTower` (`src/models/qwen3_vision.cpp`), loaded from the BF16 `text_encoder` directory as a dense tower (fused qkv, DeepStack taps after blocks 8 / 16 / 24);
`h3_vision.cu` only adapts it to the H3 interface.

### 4.4 Memory and residency

`video.resident` (default on) keeps the transformer and its adapters loaded between jobs; the activation scratch is released while the text and decode stages run
(`trim`) and taken back for the next job. With `tensor_parallel` the resident text tower (when `text_encoder` is set) and, by default, the video decoders and the audio VAE
(`video.resident_decoders`) stay loaded too. Image / video / audio reference encoders load for the stage that needs them.

| Configuration, 4 GPUs | peak per GPU | step | per job (1280x736x124, turbo 4 steps) |
|---|---|---|---|
| sequence parallel, BF16 text streamed, decoders per job | 30 GB | 15.6 s | text 7 s + denoise 62 s + decode 15 s + mux 2 s |
| tensor parallel, EXL3 text resident, resident decoders | 28.6 / 27.3 GB | 16.6 s | text 0.4 s + denoise 66 s + decode 11 s + mux 2 s |

Measured on four V100 at a 225 W power limit (at 150 W the cards run at about 680 MHz and every figure is 30 to 50% slower). The first job also pays the loads (transformer
7 to 9 s, text tower 4 s, decoders 3.6 s). Where a sequence-parallel step goes: attention about two thirds (54 TF/s), the GEMMs (cuBLAS, 85 to 97 TF/s) the rest.

### 4.5 Determinism and numerics

Every run logs fingerprints (prompt, noise, each step's velocity, the final latents), so a change that should not alter results can be checked from the log. H3 has outlier
activations (per-element magnitudes in the hundreds), so FP16 QK^T can exceed the FP16 range in principle: the residual stream and the wide-range projections stay FP32.
Work that touches a non-blocking stream must order itself against default-stream uploads (`cudaStreamSynchronize(nullptr)` first); the quantized text kernels grow per-device
workspaces lazily under a global lock, so they are sized at load, before any collective (a rank growing one mid-forward deadlocks the others in NCCL).

## 5. Configuration checks

`h3_check_paths` (the server at startup, `h3_generate` before it loads anything) refuses configurations that cannot work and says why: a device that does not exist or is listed
twice, a first device that is not the primary one, a tensor-parallel device count that cannot shard the transformer (2, 4 or 7) or the text tower (2, 4 or 8), a `text_encoder`
that is not an EXL3 or NVFP4 conversion of the 5120-wide Qwen3-VL language model with at least 50 layers, and a `text_encoder` without `tensor_parallel` when the transformer
plus the tower exceed the GPU memory. It warns when resident decoders would not fit. Without NCCL the server keeps sequence parallelism and the BF16 text path and says so.

## 6. Validation

Goldens come from the official code (`tests/tools/h3_*.py`, written to `~/Documents/h3`); each test prints a cosine or PSNR and `ok` / `FAIL`.

| Test | Checks | Expected |
|---|---|---|
| `test_h3_dit`, `test_h3_dit_sp`, `test_h3_dit_tp` | transformer forward on 1 GPU, sequence-parallel and tensor-parallel (2 and 4 GPUs, with and without the turbo adapter) | video cosine >= 0.999996, audio >= 0.99997 |
| `test_h3_attn`, `bench_h3_attn` | attention against a CPU softmax (full, causal, key ranges, head_dim 64 / 128), TF/s at the DiT shape | max error 3e-4 |
| `test_h3_vae`, `test_h3_vae_det` | decoder tile and full tiled decode (59 dB), run-to-run determinism | identical hashes |
| `test_h3_vae_enc`, `test_h3_audio`, `test_h3_audio_enc` | image / video / audio encoders and the audio decoder | cosine 1.000000 |
| `test_h3_text`, `test_h3_text_mm`, `test_h3_text_tp` | text tower (BF16, EXL3, NVFP4; text only and with an image; 1 and 4 GPUs) against the BF16 reference | see 4.3 |
| `test_h3_vision` | vision tower, merged rows and the three DeepStack taps | cosine >= 0.99998 |
| `test_h3_layout`, `test_h3_ref_layout`, `test_h3_image` | packed layouts (identical to the reference builder), Lanczos resize, keyframe rows | exact / PSNR gates |
| `test_h3_text_identity`, `test_h3_text_layers` | whether two text checkpoints are the same model; per-layer drift between two quantizations | |
| `bench_h3_dit`, `bench_h3_gemm`, `bench_allreduce`, `p2p_check` | DiT layer timing, cuBLAS algorithm sweep on the DiT and VAE shapes, NCCL all-reduce, NVLink copy integrity | |

## 7. Limits and known issues

* A prompt of more than about 8.6K text rows is rejected under tensor parallelism and with the resident text tower; the packed sequence is limited by `video.max_rows`
  (default 36000).
* Adapters are applied unmerged and cost a little compute per step; more than six per job are rejected.
* NVFP4 text conditioning degrades on image-conditioned prompts (4.3); its effect on generated video is untested.
* MP4 muxing, reference-video decoding and thumbnails call the `ffmpeg` / `ffprobe` binaries.
* The reference noise comes from the engine's generator, not torch's, so a seed does not reproduce 1Cat's output; results are deterministic per configuration.
* The web page has had little browser testing.
