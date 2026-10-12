# `src/video/` (MiniMax-H3)

This section of the project handles MiniMax-H3, a rectified-flow diffusion transformer that denoises video and audio latents jointly. Everything here is CUDA only, and the model runs through either:
* `h3_generate`: a command line tool for a single job (or a few, with `--repeat`)
* the video service of `volta_server`: a job queue, an HTTP API, and a (barely tested) web page

Supported tasks:
* text to video and audio (T2VA)
* first / last / both keyframes (FL2VA)
* references (Ref2VA): images, videos, and audio
* the distilled turbo LoRAs and community LoRAs, several at once

Not supported: ComfyUI integration, FP8 / sparse-attention recipes, TeaCache / Cache-DiT, CPU and Vulkan.

## How a job runs

Text, reference rows, and the two latent streams are packed into one token sequence. The model is CFG-distilled, so there is one transformer forward per step: no guidance batch, no KV cache, no causal mask. A request is minutes of batch compute, then an MP4.

| Stage | Component | Notes |
|---|---|---|
| plan | layout, sigma schedule, noise | `h3_pipeline.cu`, `h3_image.cpp`; the noise generator is the engine's own (`H3Gaussian`), so a seed does not reproduce 1Cat's torch noise |
| references | video VAE encoder (3-D conv, causal), image encoder, audio VAE encoder | only FL2VA keyframes and Ref2VA; tiles are spread over the GPUs |
| text | Qwen3-VL language model (first 50 decoder layers), plus its vision tower for images and video frames | BF16 layers streamed from disk, or an EXL3 / NVFP4 conversion resident on the GPUs |
| denoise | the transformer: 50 blocks + 2 token-refiner blocks, hidden 5376, 56 heads x 128, MLP 14336 (fc1 28672 wide, gated), patch (1,2,2), 24 video / 32 audio latent channels | 4 or 8 Euler steps (eta 0), flow shift 12 / 3 (video / audio) with the turbo adapters, per-step reset of the condition rows |
| decode | video VAE decoder (ViT, 36 blocks of dim 2048, spatial 16x, temporal 4x, tiled), audio VAE decoder (BigVGAN-style 1-D conv stack, 32 kHz stereo) | |
| mux | MP4, 24 fps video + PCM audio | through the `ffmpeg` binary |

At 1280x736 and 124 frames the sequence is about 34,500 rows (80x46 latent per frame, patch 2x2, ~32 latent frames, plus audio, text, and condition rows). One forward is about 2.3 PFLOP, split roughly evenly between the linear layers and attention.

## Files you need

* `MiniMax-H3/FL2VA` and `MiniMax-H3/Ref2VA`: the diffusers layout
  * `text_encoder` (BF16, 14 shards, shared), `video_vae`, `audio_vae`, `tokenizer`, `processor`
  * `model_index.json` of class `MiniMaxH3Pipeline`, which is how `volta_server` recognizes it
* `Comfy-H3/diffusion_models/minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors`: the transformer, 21 GB each. This is the only transformer weights the engine reads.
* `h3-turbo/*.safetensors` and `h3-loras/*.safetensors`: the turbo LoRAs (LightX2V, rank 128) and community LoRAs
* optional: an EXL3 or NVFP4 conversion of Qwen3-VL-32B-Instruct, for the resident text tower

The transformer is INT8 with ConvRot:
* `qkv_proj`, `out_proj`, `fc1`, and `fc2` of every block are INT8 with a per-output-row FP32 scale and a `.comfy_quant` blob
  * the weights are Hadamard-rotated in groups of 256 along K, then row-quantized
* the token-refiner blocks, norms, and q/k norms are BF16, and the patch / condition / time projections are FP32
* the AdaLN is "pruned": `blocks.N.adaln_proj.linear` maps an 8-dimensional coordinate (fed by `adaln_t_table`) to the shift / scale / gate of three modalities
* each projection is dequantized and un-rotated to FP16 right before its cuBLAS GEMM (FP32 accumulation), and the residual stream is FP32

LoRAs are PEFT-style A/B pairs on `to_q / to_k / to_v / to_out.0` and `ff.net.0.proj / ff.net.2` (diffusers, kohya, and ai-toolkit key layouts are read).
* they are applied unmerged (`x A^T B` per target, per-adapter scale), which keeps the INT8 base intact
* more than six per job are rejected by the server

## `h3_generate_main.cpp`

The command line tool.

```
build/h3_generate --root <MiniMax-H3/FL2VA> --dit <fl2va checkpoint> \
    --lora <h3-turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors> --prompt "..." --size 1280x736 --frames 124 --steps 4 \
    --devices 0,1,2,3 --out out.mp4 [--image first.png [--image last.png]] [--lora other.safetensors:0.8]
build/h3_generate --root ... --dit-ref2va <ref2va checkpoint> --lora <h3-turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors> \
    --prompt "The dog from Picture 1 ..." --ref-image a.png [--ref-image b.png] [--ref-video clip.mp4] [--ref-audio voice.wav] --devices 0,1,2,3 --out out.mp4
```

Flags:
* `--root`: the pipeline directory (`FL2VA` or `Ref2VA`)
* `--dit` / `--dit-ref2va`: the transformer checkpoint for text / keyframe jobs, and for reference jobs
* `--devices`: the GPUs for the transformer (the first one is the primary: encoders and decoders run there). `--device` selects just one.
* `--lora PATH[:scale]`: adapter, repeatable
* `--prompt`, `--out`, `--size WxH`, `--frames`, `--steps`, `--seed`, `--video-shift`, `--audio-shift`
* `--image`: a keyframe (up to two, `--frame-index` places them)
* `--ref-image`, `--ref-video`, `--ref-audio`: references (see below)
* `--tp`: tensor parallelism instead of sequence parallelism
* `--text-encoder DIR`: use a resident EXL3 / NVFP4 text tower
* `--resident`: keep the models loaded between `--repeat N` runs, and `--resident-decoders` keeps the VAE decoders as well
* `--no-trim`: don't release the transformer's activation scratch during the text and decode stages

The configuration is checked before anything loads (see `h3_check.cpp`).

### References (Ref2VA)

* up to 9 reference images (resized to a 2048 px short edge, 32 px multiples, aspect ratio 0.4 to 2.5), 3 reference videos, and 3 audio clips (12 in all)
  * videos and audio are 2 to 15 s each and in total
  * they need the Ref2VA transformer and the `ref2v` turbo adapters
* in the prompt, they are `Picture i`, `Video k`, and `Audio j`
* reference videos are decoded by ffmpeg onto a 768 px canvas (at most 768 x 1344, 24 fps), encoded by the causal 3-D video VAE in 17-frame clips, and shown to the language model as every 12th frame with timestamps
  * a video's soundtrack becomes an audio reference
* image references sit at timestep 0.999, and audio references at 1.0 (clean latents)
* the layout is `[text | reference blocks | target audio | target video]`

## `src/server/video/`

The video service of `volta_server`.

A `model_dir` with a `model_index.json` of class `MiniMaxH3Pipeline` selects the `minimax_h3` model family: no language-model runtime, and a job queue with one worker thread that owns the GPUs.

Routes:
* `GET /`: the page
* `GET /api/info`
* `GET` / `POST /api/jobs`, and `GET` / `DELETE /api/jobs/<id>`
* `GET /api/jobs/<id>/video` (supports Range) and `/thumb`
* `POST /api/jobs/<id>/cancel`
* `/health`, `/metrics`, `/metrics/prometheus`

A job body is `{prompt, width, height, frames, steps, seed, turbo, loras:[{file,scale}], images:[{data,frame_index}], ref_images, ref_videos, ref_audios}` with base64 uploads.

Settings live in the launch config's `video` object, documented in [CONFIG.md](CONFIG.md): `dit`, `dit_ref2va`, `lora_dirs`, `out_dir`, `devices`, `max_rows`, `resident`, `resident_idle_s`, `tensor_parallel`, `text_encoder`, and `resident_decoders`.

There is no authentication, and the server binds to localhost by default. The page has had very little browser testing.

## `h3_pipeline.cu`, `h3_engine.cpp`, `h3_stage_*`, `h3_check.cpp`

The orchestration.
* `H3Run` (`h3_run.h`) is one job split into stages (`plan_shape`, `encode_references`, `encode_prompt`, `plan_denoise`, `denoise`, `decode`, mux), each in `h3_stage_*.cpp/.cu`
* `H3Engine` drives a run, and `H3Components` (`h3_components.h`) owns the loaded models and hands them out as leases
  * a one-shot run loads each component when its stage first asks for it and frees it afterwards
  * with `resident`, the transformer and its adapters stay loaded between jobs
  * with `resident` + `tensor_parallel`, the resident text tower (if `text_encoder` is set) and the video decoders + audio VAE (`resident_decoders`) stay loaded too
  * the transformer's activation scratch is released while the text and decode stages run (`trim`), and taken back for the next job
* the reference encoders load for the stage that needs them

`h3_check_paths` is run by the server at startup and by `h3_generate` before it loads anything. It refuses configurations that cannot work, and says why:
* a device that does not exist or is listed twice, or a first device that is not the primary one
* a tensor-parallel device count that cannot shard the transformer (2, 4, or 7) or the text tower (2, 4, or 8)
* a `text_encoder` that is not an EXL3 or NVFP4 conversion of the 5120-wide Qwen3-VL language model with at least 50 layers
* a `text_encoder` without `tensor_parallel`, when the transformer plus the tower exceed the GPU memory
* it only warns when the resident decoders would not fit

Without NCCL, the server keeps sequence parallelism and the BF16 text path, and says so.

## `h3_dit.cu`, `h3_dit_group.cpp`

The transformer. Per block: AdaLN-indexed RMSNorm, qkv projection, per-head RMSNorm of q and k, 3-D RoPE, non-causal attention over the whole sequence, the out projection with a gated residual, then fc1 (gated SiLU) and fc2 with a gated residual.
* the SiLU product carries a per-row power-of-two scale that keeps the FP16 value in range, which is undone after the down projection
* there are no CUDA graphs, as each kernel runs for milliseconds

### Sequence parallelism

The default. Every GPU holds the full INT8 weights and owns a quarter of the rows.
* per layer, each rank attends to its own keys while the peers' K / V slices arrive over NVLink, then to each peer's slice as it lands
  * one copy stream per peer, ordered with events, so nothing waits on the GPU from the host
* needs about 21 GB of weights plus 6 GB of scratch per GPU

### Tensor parallelism

`video.tensor_parallel`, `--tensor-parallel`, or `h3_generate --tp`. Needs NCCL at build time (CMake defines `ENGINE_HAVE_NCCL` when it finds it).
* every GPU holds a quarter of the heads and of the MLP width, and runs all rows
  * the ConvRot rotation works in groups of 256 along K, so K-sharded matrices keep whole groups (out projection 7168 / 4 = 7 groups, MLP down 14336 / 4 = 14)
  * usable GPU counts are 2, 4, or 7
* the attention-output and MLP-down partial products are summed over NCCL in FP32 (two all-reduces per layer)
* the SiLU row scale is the maximum over the whole MLP width (a small max all-reduce)
* adapters are sliced the same way: column-parallel projections keep the B rows of the rank's slice, row-parallel ones keep the A columns
* the point is memory: ~14 GB per GPU while denoising instead of 27, so the text tower and the VAEs can stay loaded
* the cost is about 1 s per step more than sequence parallelism
* a prompt may have at most about 8.6K text rows (the full-width token refiner runs on them), and longer ones fail the job with a message

## `h3_text.cu`, `h3_text_group.cpp`, `h3_vision.cu`

The text tower: Qwen3-VL-32B-Instruct's language model (dense, 5120 hidden, 64 heads / 8 KV heads of 128, MLP 25600, interleaved mRoPE), first 50 layers, no final norm.
* the MiniMax encoder is the stock model: `test_h3_text_identity` finds its norms and embedding bit-identical to the public EXL3 and NVFP4 conversions

Three ways to run it:
* **streamed BF16** (default): layers are read from the `text_encoder` shards through pinned chunks and converted to FP16 on the GPU, one layer ahead of the compute
  * 5 to 7 s for a short prompt
* **resident EXL3** (`video.text_encoder`, `--text-encoder DIR`, e.g. `turboderp/Qwen3-VL-32B-Instruct-exl3`)
  * the trellis projections stay on the GPUs and run through `LinearTrellis`
  * `H3TextGroup` shards them over the GPUs: 16 query heads, 2 KV heads, and 6400 MLP columns per GPU on four (column-parallel q/k/v/gate/up, row-parallel o/down), with FP16 all-reduces of the two partial products per layer and the MLP row scale from a max all-reduce
  * 6.0 bpw is 18.3 GB for 50 layers (4.6 GB per GPU on four), and a short prompt takes about 0.2 s
  * hidden-state cosine against the BF16 reference: 0.999999 text only, 0.999958 with an image
* **resident NVFP4** (a compressed-tensors conversion, e.g. `Qwen3-VL-32B-Instruct-NVFP4`)
  * the same path through the engine's W4A16 kernels, 3.4 GB per GPU on four
  * text only is 0.99991, but with an image it's 0.974
    * at layer 43 the BF16 and EXL3 runs create a very large activation on one token (channel 731 reaches ~7400) that the 4-bit weights don't reproduce (`test_h3_text_layers` shows the divergence layer by layer)
  * prefer EXL3 6 bpw for image or reference prompts

The resident tower does not fit next to the full-weight sequence-parallel transformer in 32 GB, so use it with `tensor_parallel`.

The vision tower is the engine's `Qwen3VisionTower` (`src/models/qwen3_vision.cpp`), loaded from the BF16 `text_encoder` directory as a dense tower (fused qkv, DeepStack taps after blocks 8 / 16 / 24). `h3_vision.cu` only adapts it to the H3 interface (arena sizing, FP32 rows).

## `h3_vae.cu`, `h3_vae_enc.cu`, `h3_audio.cu`, `h3_refs.cpp`, `h3_image.cpp`

* `h3_vae.cu`: the video decoder, a ViT
  * run on 256 px tiles of every temporal chunk (7 latent frames, 28 tiles at 1280x736), with the tiles spread over the GPUs (one decoder per GPU)
  * two tiles go through a transformer pass together when the card has the room
  * tiles are blended with the checkpoint's overlap rules and streamed to RGB8
* `h3_vae_enc.cu`: the image / video encoder
  * causal 3-D convolutions as FP16 im2col + GEMM with valid-tap skipping
  * tiled like the checkpoint (256 px tiles, at least 64 px overlap), with the tiles spread over the GPUs
* `h3_audio.cu`: a DAC-style encoder and the BigVGAN decoder (snake activations, alias-free resampling) on conv1d kernels
* `h3_stager.h`: the decoder weights are read once and broadcast to all GPUs through a shared pinned ring
* `h3_refs.cpp`: reference video / audio loading through `ffmpeg` / `ffprobe`, sample indices and timestamps
* `h3_image.cpp`: layouts, Lanczos resize, keyframe rows, and the noise generator

## `src/kernels/cuda/h3/h3_attn.cu`

The attention kernel, WMMA on Volta.
* a warp owns 16 query rows with Q and the FP32 output in registers
* K / V tiles are double buffered through shared memory with padded rows
* the online softmax runs on the accumulators
* head_dim 128 (full and causal) and 64 (the VAE decoder), about 54 TF/s on the DiT shape
* a key-range variant carries the running (max, sum, output) between calls, which sequence parallelism uses
* register use is at the limit of 255, so small source changes can cost 10% or more: check `bench_h3_attn` after touching it

See [HARDWARE_SM70.md](HARDWARE_SM70.md) for the fragment layouts.

## Memory and timings

Measured on four V100 at a 225 W power limit (at 150 W the cards run at about 680 MHz, and every figure is 30 to 50% slower), 1280x736 and 124 frames, turbo adapter, 4 steps:

| Configuration, 4 GPUs | peak per GPU | step | per job |
|---|---|---|---|
| sequence parallel, BF16 text streamed, decoders per job | 30 GB | 15.6 s | text 7 s + denoise 62 s + decode 15 s + mux 2 s |
| tensor parallel, EXL3 text resident, resident decoders | 28.6 / 27.3 GB | 16.6 s | text 0.4 s + denoise 66 s + decode 11 s + mux 2 s |

* the first job also pays the loads (transformer 7 to 9 s, text tower 4 s, decoders 3.6 s)
* in a sequence-parallel step, attention is about two thirds (54 TF/s), and the GEMMs (cuBLAS, 85 to 97 TF/s) are the rest

## Determinism and numerics

* every run logs fingerprints (prompt, noise, each step's velocity, the final latents), so a change that should not alter results can be checked from the log
* H3 has outlier activations (per-element magnitudes in the hundreds), so the residual stream and the wide-range projections stay FP32
* work on a non-blocking stream must order itself against default-stream uploads (`cudaStreamSynchronize(nullptr)` first), which was the cause of a few "random" results
* the quantized text kernels grow per-device workspaces lazily under a global lock, so they are sized at load, before any collective (a rank growing one mid-forward deadlocks the others in NCCL)

## Tests

The goldens come from the official code (`tests/tools/h3_*.py`, written to `~/Documents/h3`). Each test prints a cosine or PSNR and `ok` / `FAIL`.

* `test_h3_dit`, `test_h3_dit_sp`, `test_h3_dit_tp`: transformer forward on one GPU, sequence-parallel, and tensor-parallel (2 and 4 GPUs, with and without the turbo adapter). Expect video cosine >= 0.999996 and audio >= 0.99997.
* `test_h3_attn`, `bench_h3_attn`: attention against a CPU softmax (full, causal, key ranges, head_dim 64 / 128), and TF/s at the DiT shape
* `test_h3_vae`, `test_h3_vae_det`: decoder tile and the full tiled decode (59 dB), and run-to-run determinism
* `test_h3_vae_enc`, `test_h3_audio`, `test_h3_audio_enc`: the encoders and the audio decoder (cosine 1.000000)
* `test_h3_text`, `test_h3_text_mm`, `test_h3_text_tp`: the text tower (BF16, EXL3, NVFP4; text only and with an image; one and four GPUs)
* `test_h3_vision`: the vision tower, merged rows and the three DeepStack taps (cosine >= 0.99998)
* `test_h3_layout`, `test_h3_ref_layout`, `test_h3_image`: packed layouts (identical to the reference builder), Lanczos resize, keyframe rows
* `test_h3_text_identity`, `test_h3_text_layers`: whether two text checkpoints are the same model, and the per-layer drift between two quantizations
* `bench_h3_dit`, `bench_h3_gemm`, `bench_allreduce`, `p2p_check`: DiT layer timing, a cuBLAS algorithm sweep on the DiT and VAE shapes, NCCL all-reduce, and NVLink copy integrity

## Notes

* a prompt of more than about 8.6K text rows is rejected under tensor parallelism and with the resident text tower, and the packed sequence is limited by `video.max_rows` (default 36000)
* NVFP4 text conditioning degrades on image-conditioned prompts, and its effect on the generated video is untested
* MP4 muxing, reference-video decoding, and thumbnails call the `ffmpeg` / `ffprobe` binaries
* results are deterministic per configuration, but not comparable to 1Cat's for the same seed
