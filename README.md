# Volta Engine

Yet another (slopped) inference engine, tailored in C++ for Volta, specifically 4xV100 SXM2 systems.

## Build

CUDA 12.x is required, as 13+ drops SM_70 support. *Could* run on SM_75+, but no kernels would make use of instructions / specifications introduced later.

To compile, run `cmake -B build && cmake --build build`.

## Run

You can either pass a [config.json](/configs/example_qwen.json) via: `build/volta_server --config ./path/to/config.json`, or specify config variables per the auto-generated [docs/CONFIG.md](/docs/CONFIG.md).

Accepts either OpenAI-formatted payloads via `/v1/chat/completions`, or the Anthropic-formatted endpoint via `/v1/messages`.

A monitor page of possibly-useful runtime statistics are provided via `/monitor`.

Additional details can be referenced via the verbose [docs/GETTING_STARTED.md](/docs/GETTING_STARTED.md).

## Why?

This aims to milk every drop of perf from my V100s with the realistic goal of serving GLM-5.3-Flash at acceptable quantization.

Most inference engines treat Volta (SM_70) as a second class citizen and fall back to eager, unfused kernels. Existing solutions do come with their pros and cons:
* 1Cat-vLLM boasts the best performance out of the box, but only really guarantees that with NVFP4 weights and quite a hefty stack. Models that require higher quantization like GLM-5.3-Flash are boxed out (and solutions like `vllm-exl3` are severely capped).
* exllamav3, while boasting higher quantization and lower perplexity loss, can only achieve ~60% of the cap due to quantization overhead, even with optimizations.
* llama.cpp is a mixed bag, and forking it dooms it to never get rebased without headaches (and submitting it as a valid PR is a fools errand).
* Strata targets one model (Qwen3.8-Flash-Next) really well, but neglects the specialized coverage for SM_70, and is geared towards lower end specs.

### Features

*to-do: fill out later*

### Metrics

Decode speed through the streaming API, one request, 4x V100-SXM2-32GB at a 250 W cap (clocks free up to 1530 MHz), T=1 sampling, 256 new tokens. The numbers are real questions behind 8K tokens of filler, not a best case: the four prompts are listed because speculation speed depends on how predictable the answer is.

| Model | Speculator | tok/s, 4 prompts (mean) |
|---|---|---|
| Qwen3.8-27B (NVFP4) | DFlash2 | 252, 263, 182, 81 (195) |
| Qwen3.8-27B (EXL3 4.0 bpw) | DFlash2 | 163, 169, 138, 93 (141) |
| Qwen3.8-Flash-Next (NVFP4 checkpoint) | MTP | 235, 210, 181, 136 (191) |
| GLM-5.3-Flash (EXL3 3.05 bpw, NVFP4 dense) | DFlash2 | 116, 109, 108, 72 (101) |
| GLM-5.3-Flash (the same) | MTP | 109, 102, 106, 87 (101) |
| GLM-5.3-Flash | none | 64 |

A plain token takes 12 ms (Qwen NVFP4), 14 ms (Flash-Next), 15 ms (GLM) or 16 ms (Qwen EXL3). Repetitive output peaks at over 300 tok/s on Qwen NVFP4. Perplexity per configuration and the bit widths of the EXL3 checkpoints are in [docs/QUANTIZATION.md](/docs/QUANTIZATION.md). Everything else, including where the time goes per kernel and how this compares with 1Cat-vLLM's published figures (different cards, clocks and prompts), is in [docs/PERFORMANCE.md](/docs/PERFORMANCE.md).

### Why not?

This project was slopped up over a weekend. CUDA is NOT my expertise (Vulkan is).

I don't intend to turn this into an actual project beyond a side-hobby, as existing solutions should be servicable enough for most people's usecases (in other words, I don't indend to extend this as a competitive engine).

A slim amount of [models](/src/models/) and weight formats (EXL3, NVFP4) are supported, but extending them shouldn't be too much of a hassle.

## Notices

Unless otherwise credited/noted in this repo or within the designated file, this repository is licensed under AGPLv3.

CUDA kernels / code scaffollding are heavily referenced from:
* [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) for NVFP4 kernels
* [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) for NVFP4 kernels
* [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) for EXL3 kernels
* [Niko1221/Strata](https://github.com/Niko1221/Strata/) for online experts streaming (rather than a static/offline approach)
* a bunch of extra slopped optimizations from abandoned private experiments of the above
