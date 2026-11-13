# Volta Engine

Yet another (slopped) inference engine, tailored in C++ for Volta, specifically 4xV100 SXM2 systems (~~T180-G20~~ T181-G20).

## Build

CUDA 12.x is required for SM_70, as 13+ drops SM_70 support.
* Additional backends are in place (CPU/Vulkan), but are untuned and merely for scaffolding, not performance.

To compile, run `cmake -B build && cmake --build build`. SM_86 (RTX 30xx) and SM_89 (RTX 40xx) are built with `-DCMAKE_CUDA_ARCHITECTURES="86;89"` (or `"70;86;89"` for one binary covering all three); the tensor-core kernels take native `mma.m16n8k16` paths there (the Volta `m8n8k4` is only emulated on newer cards). Details in [docs/GETTING_STARTED.md](/docs/GETTING_STARTED.md).

AMD RDNA GPUs (e.g. RX 7900 XTX, gfx1100) build with ROCm: `cmake -B build_rocm -DENGINE_BACKEND=ROCM -DCMAKE_HIP_ARCHITECTURES=gfx1100`. The port runs every kernel with the tensor-core instructions emulated, so it is correct before it is fast; see [docs/ROCM.md](/docs/ROCM.md).

## Run

You can either pass a [config.json](/configs/example_qwen.json) via: `build/volta_server --config ./path/to/config.json`, or specify config variables per the auto-generated [docs/CONFIG.md](/docs/CONFIG.md).

Accepts either OpenAI-formatted payloads via `/v1/chat/completions`, or the Anthropic-formatted endpoint via `/v1/messages`.

A monitor page of possibly-useful runtime statistics are provided via `/monitor`.

Additional details can be referenced via the verbose [docs/GETTING_STARTED.md](/docs/GETTING_STARTED.md).

## Why?

This aims to milk every drop of perf from my V100s with the realistic goal of serving GLM-5.3-Flash at acceptable quantization.

Most inference engines treat Volta (SM_70) as a second class citizen and fall back to eager, unfused kernels. Existing solutions do come with their pros and cons:
* 1Cat-vLLM boasts the best performance out of the box, but only really guarantees that with NVFP4 weights and quite a hefty stack. Models that require higher quantization like GLM-5.3-Flash are boxed out (and solutions like `vllm-exl3` are severely capped).
* exllamav3, while boasting higher quantization, better perplexity, and utilization of the CPU for host-side expert residence, but can only achieve ~60% of the cap due to quantization overhead, even with optimizations.
* llama.cpp is a mixed bag, and forking it dooms it to never get rebased without headaches (and submitting it as a valid PR is a fools errand).
* Strata targets one model (Qwen3.8-Flash-Next) really well, but neglects the specialized coverage for SM_70, and is geared towards lower end specs.

### Features

*to-do: fill out later*

### Metrics

Decode speed of one request through the streaming API on 4x V100-SXM2-32GB at 250 W, every expert on the GPUs (DeepSeek: 92%), greedy, thinking off. Agent turns are replayed from a recorded Claude Code session (13-20K-token prompts); math and code are GSM8K and MBPP problems. Each model's best speculator for the kind of text:

| Model | Weights | Agent turns, tok/s | Math / code, tok/s |
|---|---|---|---|
| Qwen3.8-27B | NVFP4 | 154 (DFlash2) | 321 / 292 (DFlash2) |
| Qwen3.8-27B | EXL3 4.0 bpw | 116 (MTP) | 186 / 167 (DFlash2) |
| Qwen3.8-Flash-Next | NVFP4 | 165 (MTP) | 267 / 266 (MTP / DFlash2) |
| Qwen3.8-Flash-Next | EXL3 4.0 bpw, Volta order | 133 (MTP) | 197 / 184 (DFlash2) |
| GLM-5.3-Flash | EXL3 3.05 bpw | 72 (DFlash2) | 127 / 118 (DFlash2) |
| DeepSeek-V4.1-Flash | EXL3 1.68 bpw | | 70 / 71 (DSpark) |

Which speculator wins depends on the text: the checkpoint's MTP head on agent turns, which copy much of their input, and the DFlash2 drafter on math and code. Every configuration measured (GGUF, both EXL3 tile orders, each speculator and MTP depth), step latency, prefill and the method are in [docs/PERFORMANCE.md](/docs/PERFORMANCE.md); perplexity per configuration and the EXL3 bit widths are in [docs/QUANTIZATION.md](/docs/QUANTIZATION.md).

A model's weight format largely decides its speed (NVFP4 decodes faster than EXL3 trellis at a similar size, and a smaller EXL3 quant faster than a bigger one), at the cost of perplexity.

### Why not?

This project was slopped up over a weekend. CUDA is NOT my expertise (Vulkan is).

I don't intend to turn this into an actual project beyond a side-hobby, as existing solutions should be servicable enough for most people's usecases (in other words, I don't indend to extend this as a competitive engine).

A slim amount of [models](/src/models/) and weight formats (EXL3, NVFP4) are supported, but extending them shouldn't be too much of a hassle.

Supported, tested configurations are narrow, as it's limited to what I currently have on hand, so gaps in coverage may exist. On the other side, some values may end up hard-coded, or undocumented, that may only pertain to a specific system used during development.

## Notices

Unless otherwise credited/noted in this repo or within the designated file, this repository is licensed under AGPLv3.

CUDA kernels / code scaffollding are heavily referenced from:
* [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) for NVFP4 kernels
* [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) for NVFP4 kernels
* [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) for EXL3 kernels
* [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for the GGUF block formats, their CUDA and CPU dot-product and dequantization kernels and their Vulkan mat-vec and matmul shaders, vendored unchanged under the MIT license in [src/kernels/cuda/gguf/ggml](/src/kernels/cuda/gguf/ggml), [src/kernels/cpu/gguf/ggml](/src/kernels/cpu/gguf/ggml) and [src/kernels/vulkan/ggml](/src/kernels/vulkan/ggml)
* [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for its iqk CPU matrix multiplications of several GGUF rows, vendored unchanged under the MIT license in [src/kernels/cpu/gguf/ik](/src/kernels/cpu/gguf/ik)
* [Niko1221/Strata](https://github.com/Niko1221/Strata/) for online experts streaming (rather than a static/offline approach)
* a bunch of extra slopped optimizations from abandoned private experiments of the above
