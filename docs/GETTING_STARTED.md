# Getting started

## Build

CUDA 12.x (13+ drops SM_70) and CMake 3.24+ with Ninja. The CUDA build is the primary one:

```
git submodule update --init   # deps/json and deps/propan2ol (the TTS text front-end)
cmake -S . -B build -G Ninja && cmake --build build
```

A full CUDA build takes a while (the trellis kernels are template heavy); incremental builds are quick.

Video files and audio other than WAV as model input need FFmpeg's libraries (libavformat, libavcodec, libswscale; found with pkg-config, in
every configuration). `-DENGINE_FFMPEG=OFF` leaves them out; a video then arrives as its frames, audio as WAV.

### Build time

The EXL3 GEMM kernels are the bulk of a from-scratch CUDA build: each (bit width, codebook) pair is its own generated translation unit, so the
build runs them in parallel. Two options leave out kernels a setup does not use:

* `ENGINE_EXL3_PRUNE` (`AUTO`, `ON`, `OFF`) skips the tile shapes the built architectures never select: shape 1 below SM_90, shape 4 at 7 and 8 bits
  below SM_89. `AUTO`, the default, prunes when SM_70 is the only target (the default target; `native` is resolved to the installed
  GPUs) and builds everything otherwise; `ON` prunes for whatever is built. The selection steps down to a shape that exists
* `ENGINE_EXL3_CODEBOOKS` (default `0;1;2`: 3inst, mcg, mul1; the codebook is in the checkpoint's `config.json`) lists the codebooks to build.
  `-DENGINE_EXL3_CODEBOOKS=2` builds `mul1` only; a checkpoint using a codebook left out stops at its first GEMM with a message naming the option

### Other GPUs (SM_86, SM_89)

The CUDA build targets SM_70 (V100) by default. Ampere (SM_86: RTX 30xx, A10, A40) and Ada (SM_89: RTX 40xx, L4, L40) build from the
same sources; pick the targets with `CMAKE_CUDA_ARCHITECTURES` (several at once give one binary for all of them):

```
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES="70;86;89"   # or 86, 89, native
```

CUDA 12.x builds all three; CUDA 13 builds 86 / 89 only (configuration fails if 70 is in the list). On SM_86 / SM_89:
* the EXL3 GEMM / GEMV and the cooperative MoE decode kernel take their native `mma.m16n8k16` / `ldmatrix` / `cp.async` paths (the
  upstream exllamav3 ones). Fallbacks if a native path misbehaves on a card: `-DENGINE_EXL3_FORCE_VOLTA_MMA=ON` keeps the Volta
  `m8n8k4` GEMM inner loop everywhere, `-DCMAKE_CUDA_FLAGS=-DMOE_COOP_M8N8K4=1` the Volta MoE tile
* the NVFP4 skinny / MoE kernels and the prefill attention kernels (dense, MLA latent, sparse latent) have `mma.m16n8k16` paths
  over the same tiles and shared-memory staging as their Volta `m8n8k4` forms; the vision tower takes its WMMA attention kernel. No
  `m8n8k4` (which SM_80+ only emulates, at ~80 instructions each) runs there. The fragment maps are checked on the CPU by
  `test_mma_frag`; numerics match the Volta path up to FP32 summation order
* the EXL3 launch choices follow the card: GEMM tile shapes, GEMV configurations and the int8 GEMV's highest bit rate use upstream
  exllamav3's per-architecture tables on SM_86 / SM_89 (the V100 ones are this engine's own measurements); the GEMV kernels for 5 and
  6 bpw are V100-only, as upstream runs those rates through the GEMM
* the SM count is read from the device at startup (cooperative / persistent grids are sized to it). `-DENGINE_SM_COUNT=N` caps it,
  `-DENGINE_MAX_SMEM=bytes` changes the opt-in shared memory the big kernels request (default 98304, allowed by all three)
* startup prints each GPU's architecture and SM count, and warns when the build has no code for a card
* tensor parallelism needs peer access between the cards (NVLink, or PCIe P2P where the driver allows it; GeForce 40xx cards do
  not), so a single consumer card runs `tp` 1

AMD RDNA GPUs (RX 7000 / 9000) build with `-DENGINE_BACKEND=ROCM`: see [ROCM.md](ROCM.md).

### CPU and Vulkan

CPU and Vulkan builds are separate build trees (no CUDA toolchain needed):

```
cmake -S . -B build_cpu -G Ninja -DENGINE_BACKEND=CPU
cmake -S . -B build_vk  -G Ninja -DENGINE_BACKEND=VULKAN
```

They run the portable parts on reference kernels (GGUF projections on llama.cpp's CPU kernels and Vulkan shaders), so they are slow by design:
* the CPU build compiles for the build host (`-march=native`: AVX2, FMA, F16C, about 2x on the reference kernels; `-DENGINE_CPU_NATIVE=OFF` for a
  portable binary) and runs a model on one rank; Flash-Next needs ~66 GiB of host memory
* the Vulkan build runs a model on one device, or tensor-parallel over several (`tp` 2 or 4: one Vulkan device per rank, the all-reduce meets in host
  memory, which the devices read and write themselves where they import it and share timeline semaphores, else through the host threads; execution
  is eager); Flash-Next needs four devices. Models with trellis or FP16 routed experts need eager execution on one device (`--set no_graphs=true`;
  that MoE reads its routes back on the host; resident GGUF experts stay on the device; with `tp` above 1 graphs are off anyway)
* on NVIDIA GPUs whose Vulkan driver launches CUDA kernels (`VK_NV_cuda_kernel_launch`) but has no cooperative matrices (Volta), the Vulkan
  build runs its FP16 matmuls (GGUF prefill chunks among them, the weight dequantized in slices), its NVFP4 linears and its attention
  (prefill chunks, and decode rows on quantized caches) on tensor-core CUDA kernels, built as PTX where nvcc is found at configure time (the driver loads it; no CUDA library is linked);
  `--set vk_cuda_launch=false` keeps them on the shader cores
* on a device with cooperative matrices and no CUDA launches (RDNA3: `VK_KHR_cooperative_matrix`, a 16 x 16 x 16 FP16 shape), the FP16 matmuls (dense FP16,
  NVFP4 and trellis dequantize-then-matmul) run llama.cpp's cooperative-matrix `mul_mm`, and GGUF prefill chunks of the legacy and K-quant types run its
  quantized form (the weight dequantized in shared memory as the tiles load); the lookup-table (IQ) types keep the int8 and shader matmuls.
  `--set vk_coopmat=false` keeps all of it on the shader cores
* the arena of a device is one buffer: on drivers that read a descriptor wrongly into a buffer past 4 GiB (RADV) the bound-buffer shaders get windows of at
  most 3 GiB over its memory
* the routed experts are all resident (no expert tier); the MTP head runs on both (`test_mtp` passes the 27B GGUF on CPU and Vulkan)
* DeepSeek-V4.1, the DFlash / DFlash2 / DSpark drafters and MiniMax-H3 are CUDA only (the ROCm build compiles MiniMax-H3 but does not run it,
  see [ROCM.md](ROCM.md#not-supported-on-rocm))

Which models and formats were checked against CUDA on each backend is in [TESTS.md](TESTS.md); the per-function table is [CAPABILITIES.md](CAPABILITIES.md).

NCCL is optional (CMake looks for it): it enables tensor parallelism for MiniMax-H3, see [VIDEO.md](VIDEO.md).

## Run the server

```
taskset -c 0-9 build/volta_server --model-dir /path/to/Qwen3.8-27B-exl3 \
	--speculative --draft-model-dir /path/to/Qwen3.8-27B-DFlash2-EXL3-5.0bpw --port 8000
taskset -c 0-9 build/volta_server --config configs/example_qwen.json      # the same from a launch file
```

The model type is read from the checkpoint (`qwen3_5`, `glm5_next`, `qwen4_exp`, `gemma4` or `deepseek_v41`; a MiniMax-H3 pipeline directory starts
the video service, see [VIDEO.md](VIDEO.md)); anything else is a fatal error. Configuration (launch file, `--set key=value`, `ENGINE_*` variables, debug
switches) is described in [CONFIG.md](CONFIG.md): flags win over the launch file and the file over the environment, so set options with `--set` or in the
file's `options` object, and check the startup log. `configs/` has examples.

The server speaks the OpenAI chat API and the Anthropic API (Claude Code connects with `ANTHROPIC_BASE_URL=http://host:8000`) and serves a `/monitor`
page; the endpoints, the monitor and the terminal dashboard (`volta_top`, `--tui`) are described in [SERVER.md](SERVER.md), with the settings for
several requests at once. For agent workloads, a large context (`--set max_seq=262144 --set max_slots=2`) and the prefix cache (`prefix_cache_gib`,
pinned host RAM; see [FEATURES.md](FEATURES.md#attention-and-the-cache)) keep a long conversation from being prefilled again on every turn.

`--tp-size 2` runs one model on two GPUs; a second server on the other pair (`CUDA_VISIBLE_DEVICES=2,3`) serves alongside it
([FEATURES.md](FEATURES.md#parallelism-and-scheduling)).

## Models

Per-model setup (checkpoints, weight formats, speculators, images, model-specific options) is in [MODELS.md](MODELS.md):

| Model | Type | Example launch files |
|---|---|---|
| [Qwen3.8-27B](MODELS.md#qwen38-27b-qwen3_5) | `qwen3_5` | `configs/example_qwen.json`, `configs/qwen_agent.json` (256K, 2 slots), `configs/qwen_nvfp4.json` |
| [GLM-5.3-Flash](MODELS.md#glm-53-flash-glm5_next) | `glm5_next` | `configs/glm_256k.json` (256K, 4 slots, DFlash2), `configs/glm_volta.json` (Volta tile order: one shared 512K pool for 4 slots with kv_shard, DFlash2, every expert resident), `configs/glm_mtp.json` (MTP) |
| [Qwen3.8-Flash-Next](MODELS.md#qwen38-flash-next-qwen4_exp) | `qwen4_exp` | `configs/flash_next_nvfp4.json` (the NVFP4 checkpoint, 4 slots of 256K, MTP), `configs/flash_next_nvfp4_only.json`, `configs/flash_next_dflash2.json`, `configs/flash_next_cpu_experts.json` |
| [Gemma 4 31B, E4B](MODELS.md#gemma-4-gemma4) | `gemma4` | none: `--model-dir` and, for the 31B's DFlash drafter, `--speculative --draft-model-dir` |
| [DeepSeek-V4.1-Flash](MODELS.md#deepseek-v41-flash-deepseek_v41) | `deepseek_v41` | `configs/deepseek_v41.json` (DSpark drafter; the model directory needs `engram_hash.bin`) |
| MiniMax-H3 (video) | | [VIDEO.md](VIDEO.md) |

Speed per model and format is in [PERFORMANCE.md](PERFORMANCE.md), bit widths and perplexity in [QUANTIZATION.md](QUANTIZATION.md).

## Tests

`tests/tools/gate.sh` is the regression gate (modes in its header): `static` (builds all three configurations, header and doc
checks), `fast` (tests that need no checkpoint), `models` (tests that load checkpoints from `ENGINE_MODEL_ROOT`, or from
`tests/local.env`), `quick` (build + kernel fingerprint + coherence, for moves and renames), `sass` (order-independent SASS
fingerprint), `bench` and `compare <tag>`. GPU tests run under `taskset -c 0-9`. See [TESTS.md](TESTS.md) for the inventory
and [CAPABILITIES.md](CAPABILITIES.md) for which module functions each backend implements.

Benchmark baselines depend on the GPU power cap: record one per power limit (`gate.sh baseline <tag>` writes `results/baseline_<tag>.txt`).
