# Volta Engine Docs

## Features

* an inference engine for 4xV100 (SM70) systems, with CPU and Vulkan backends for the portable parts
  * one arena per GPU allocated at boot, tensor-parallel decoding over NVLink, CUDA graphs for decode
* models ([MODELS.md](MODELS.md))
  * Qwen3.8-27B (`qwen3_5`): GDN linear attention + full attention
  * GLM-5.3-Flash (`glm5_next`): KDA linear attention + absorbed MLA with a sparse indexer, routed MoE
  * Qwen3.8-Flash-Next (`qwen4_exp`): GDN + QSA block-sparse attention, 512-expert MoE, gated residuals, n-gram embedding layer
  * Gemma 4 31B (`gemma4`): sliding-window (1024) and full attention (head_dim 512, K = V), sandwich norms, an FP32 residual stream; EXL3, NVFP4 and GGUF on CUDA, CPU and Vulkan
  * DeepSeek-V4.1-Flash (`deepseek_v41`): sliding-window + compressed sparse attention, 384-expert MoE, hyper-connection streams, Engram n-gram lookups; CUDA only
  * MiniMax-H3: video + audio generation (see [VIDEO.md](VIDEO.md))
* weight formats
  * EXL3 (trellis) through the ported SM70 GEMM / GEMV / MoE kernels, in the stock or the Volta tile order
  * NVFP4 (compressed-tensors and modelopt) through software-unpacked FP16 `m8n8k4` kernels
  * GGUF (llama.cpp's block formats: Q4_0 to Q8_0, Q2_0, Q2_K to Q6_K, IQ1_S to IQ4_XS) through llama.cpp's dot-product kernels and the engine's tensor-core paths
  * original FP8 checkpoints (E4M3 with 128 x 128 block scales, GLM-5.3-Flash): the routed experts dequantized inside the expert GEMM, streamed from disk when they exceed host RAM
  * mixed checkpoints: EXL3 for everything with NVFP4 experts and / or NVFP4 dense groups swapped in at load
* serving
  * OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs, with reasoning and tool-call parsing
  * a prefix cache in pinned host RAM (with an optional disk tier), interleaved prefill, up to 256K context with two slots
  * speculative decoding with a DFlash2 or DFlash drafter, the checkpoint's own MTP head, DeepSeek's DSpark drafter, or suffix matching on the request's own text
  * image input (Qwen, GLM, Flash-Next, DeepSeek-V4.1, Gemma 4)
  * a `/monitor` page with throughput, per-GPU power / clocks, speculation, prefix cache, and MoE routing heat maps
* two-tier expert residency (hot experts on the GPU, the rest in pinned host RAM) as a capacity valve for MoE models
* runtime steering: per-request control vectors, a direction clamp built by a probe of the prompt, a reflection-marker penalty ([STEERING.md](STEERING.md))

## Docs

* [GETTING_STARTED.md](GETTING_STARTED.md): building (CUDA, other GPUs, CPU, Vulkan) and running the server
* [MODELS.md](MODELS.md): each model type: architecture, weight formats and checkpoints, tensor-parallel sizes, speculators, images, its options, a launch
* [SERVER.md](SERVER.md): the HTTP server, the OpenAI / Anthropic front ends, the output parser, the monitor page, and settings for several requests
* [FEATURES.md](FEATURES.md): the text-side feature inventory (status, option, test per feature, known limits)
* [QUANTIZATION.md](QUANTIZATION.md): weight formats, EXL3 bit widths per checkpoint and perplexity per configuration
* [PERFORMANCE.md](PERFORMANCE.md): serving speed, step latency and prefill for each model, with the conditions they were measured under
* [CONFIG.md](CONFIG.md): every option (generated from the option table, do not edit by hand)
* [STEERING.md](STEERING.md): runtime steering (control vectors, the probe and its direction clamp, the reflection-marker penalty)
* [VIDEO.md](VIDEO.md): MiniMax-H3 video and audio generation
* [ARCHITECTURE.md](ARCHITECTURE.md): the layers, the execution model, and how requests flow
* [CAPABILITIES.md](CAPABILITIES.md): which module functions each backend implements (generated)
* [KERNEL_MAP.md](KERNEL_MAP.md): kernel family -> file -> what verifies it
* [TESTS.md](TESTS.md): the test inventory and what `tests/tools/gate.sh` runs
* [DEADENDS.md](DEADENDS.md): approaches that were measured and did not pay, with the numbers and the reasons
* [ROCM.md](ROCM.md): the AMD RDNA (ROCm) build, how the CUDA sources run there, and what still runs on the emulation
* [HARDWARE_SM70.md](HARDWARE_SM70.md): Volta notes (no `cp.async`, no FP4, fragment layouts, cooperative launch limits, NVLink)

## Supported Systems

* Linux with CUDA 12.x (13+ drops SM_70) and 4xV100 SXM2 for the full experience
  * smaller setups work for the smaller models, and two GPUs can run one model per pair (`tp_size`)
* NVIDIA SM_86 / SM_89 from the same CUDA build; AMD RDNA (RX 7000 / 9000) through the ROCm build ([ROCM.md](ROCM.md))
* the CPU and Vulkan builds run what fits, slowly (reference kernels; GGUF on llama.cpp's CPU kernels and Vulkan shaders); DeepSeek-V4.1 and MiniMax-H3 are CUDA only

License, credits and the vendored code are listed under Notices in the [top-level README](/README.md#notices).
