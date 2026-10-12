# Volta Engine Docs

## Features

* an inference engine for 4xV100 (SM70) systems, with CPU and Vulkan backends for the portable parts
  * one arena per GPU allocated at boot, tensor-parallel decoding over NVLink, CUDA graphs for decode
* models
  * Qwen3.8-27B (`qwen3_5`): GDN linear attention + full attention
  * GLM-5.3-Flash (`glm5_next`): KDA linear attention + absorbed MLA with a sparse indexer, routed MoE
  * Qwen3.8-Flash-Next (`qwen4_exp`): GDN + QSA block-sparse attention, 512-expert MoE, gated residuals, n-gram embedding layer
  * MiniMax-H3: video + audio generation (see [video.md](video.md))
* weight formats
  * EXL3 (trellis) through the ported SM70 GEMM / GEMV / MoE kernels
  * NVFP4 (compressed-tensors and modelopt) through software-unpacked FP16 `m8n8k4` kernels
  * mixed checkpoints: EXL3 for everything with NVFP4 experts and / or NVFP4 dense groups swapped in at load
* serving
  * OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs, with reasoning and tool-call parsing
  * a prefix cache in pinned host RAM, interleaved prefill, up to 256K context with two slots
  * speculative decoding with either a DFlash2 drafter or the checkpoint's own MTP head
  * image input (Qwen, GLM, Flash-Next)
  * a `/monitor` page with throughput, per-GPU power / clocks, speculation, prefix cache, and MoE routing heat maps
* two-tier expert residency (hot experts on the GPU, the rest in pinned host RAM) as a capacity valve for MoE models

## Docs

* [GETTING_STARTED.md](GETTING_STARTED.md): building, running the server, and per-model notes
* [server.md](server.md): the HTTP server, the OpenAI / Anthropic front ends, the output parser, and the monitor page
* [FEATURES.md](FEATURES.md): the text-side feature inventory (status, option, test per feature, known limits)
* [QUANTIZATION.md](QUANTIZATION.md): weight formats, EXL3 bit widths per checkpoint and perplexity per configuration
* [PERFORMANCE.md](PERFORMANCE.md): serving speed, step latency and the per-kernel time split for each model, with the conditions they were measured under
* [CONFIG.md](CONFIG.md): every option (generated from the option table, do not edit by hand)
* [ARCHITECTURE.md](ARCHITECTURE.md): the layers, the execution model, and how requests flow
* [CAPABILITIES.md](CAPABILITIES.md): which module functions each backend implements (generated)
* [KERNEL_MAP.md](KERNEL_MAP.md): kernel family -> file -> what verifies it
* [TESTS.md](TESTS.md): the test inventory and what `tests/tools/gate.sh` runs
* [HARDWARE_SM70.md](HARDWARE_SM70.md): Volta notes (no `cp.async`, no FP4, fragment layouts, cooperative launch limits, NVLink)
* [FLASH_NEXT.md](FLASH_NEXT.md): Qwen3.8-Flash-Next and the NVFP4 paths, with the numbers
* [video.md](video.md): MiniMax-H3

## Supported Systems

* Linux with CUDA 12.x (13+ drops SM_70) and 4xV100 SXM2 for the full experience
  * smaller setups work for the smaller models, and two GPUs can run one model per pair (`tp_size`)
* the CPU and Vulkan builds run what fits (slowly, on reference kernels)

## Notices and Citations

Unless otherwise credited/noted in this repo or within the designated file/folder, this repository is licensed under AGPLv3.

CUDA kernels / code scaffolding are heavily referenced from:
* [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) for NVFP4 kernels
* [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) for NVFP4 kernels
* [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) for EXL3 kernels
* [Niko1221/Strata](https://github.com/Niko1221/Strata/) for online experts streaming
