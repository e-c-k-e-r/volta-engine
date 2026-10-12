# Volta Engine

Yet another (slopped) inference engine, tailored in C++ for Volta, specifically 4xV100 SXM2 systems.

## Build

CUDA 12.x is required, as 13+ drops SM_70 support. *Could* run on SM_75+, but no kernels would make use of instructions / specifications introduced later.

To compile, run `cmake -B build && cmake --build build`.
* NCCL is optional, and enables tensor parallelism for MiniMax-H3 (the transformer and its text tower over several GPUs).
* a full CUDA build takes a while (the trellis kernels are template heavy), but incremental builds are quick.

CPU and Vulkan builds are separate build trees (`-DENGINE_BACKEND=CPU` or `-DENGINE_BACKEND=VULKAN`), and run what fits (slowly).

## Run

You can either pass a [config.json](/configs/example_qwen.json) via: `build/volta_server --config ./path/to/config.json`, or specify config variables per the auto-generated [docs/CONFIG.md](/docs/CONFIG.md).

Accepts either OpenAI-formatted payloads via `/v1/chat/completions`, or the Anthropic-formatted endpoint via `/v1/messages`.

A monitor page of possibly-useful runtime statistics are provided via `/monitor`.

For MiniMax-H3 (video + audio generation), point `--model-dir` at the pipeline directory and the server starts a job queue with its own page instead, or use `build/h3_generate` for a single job. Details are in [docs/video.md](/docs/video.md).

Additional details can be referenced via the verbose [docs/GETTING_STARTED.md](/docs/GETTING_STARTED.md).

## Why?

This aims to milk every drop of perf from my V100s with the realistic goal of serving GLM-5.3-Flash at acceptable quantization.

Most inference engines treat Volta (SM_70) as a second class citizen and fall back to eager, unfused kernels. Existing solutions do come with their pros and cons:
* 1Cat-vLLM boasts the best performance out of the box, but only really guarantees that with NVFP4 weights and quite a hefty stack. Models that require higher quantization like GLM-5.3-Flash are boxed out (and solutions like `vllm-exl3` are severely capped).
* exllamav3, while boasting higher quantization and lower perplexity loss, can only achieve ~60% of the cap due to quantization overhead, even with optimizations.
* llama.cpp is a mixed bag, and forking it dooms it to never get rebased without headaches (and submitting it as a valid PR is a fools errand).
* Strata targets one model (Qwen3.8-Flash-Next) really well, but neglects the specialized coverage for SM_70, and is geared towards lower end specs.

### Features

* models (see [src/models/](/src/models/) and [src/video/](/src/video/))
  * Qwen3.8-27B, GLM-5.3-Flash, and Qwen3.8-Flash-Next, with images
  * MiniMax-H3 for video + audio generation (T2VA, first / last keyframes, and image / video / audio references, with the turbo LoRAs)
* weight formats: EXL3 and NVFP4, including mixed checkpoints (EXL3 with NVFP4 experts and / or dense groups swapped in at load)
* up to 256K context with two slots, a prefix cache in pinned host RAM, and prefill interleaved with other slots' decode
* speculative decoding through a DFlash2 drafter or the checkpoint's own MTP head
* a two-tier expert cache (cold experts in pinned host RAM) for when the experts don't fit
* a `/monitor` page, and a job page for the video service

### Metrics

*All numbers are single runs, taken at a power cap, and should be treated as directional. See [docs/FLASH_NEXT.md](/docs/FLASH_NEXT.md) and [docs/video.md](/docs/video.md) for what was measured and how.*

`bench_decode` at a 250 W cap on 4xV100:

| | Qwen3.8-27B | GLM-5.3-Flash | Flash-Next |
|---|---|---|---|
| M=1 decode | 15.9 ms (62.8 tok/s) | 19.0 ms (52.7 tok/s) | 19.8 ms (50.5 tok/s) |
| M=8 verify | 24.9 ms | 41.9 ms | 30.8 ms |
| prefill | 2492 tok/s | 763 tok/s | 1322 tok/s |
| MTP (short prompt) | 185 tok/s | | 173 tok/s |

NVFP4 dense groups on top of NVFP4 experts speed Flash-Next up further at some perplexity cost (M=1: 16.6 -> 14.2 ms for about +0.03 PPL over the NVFP4 experts alone), and on GLM-5.3-Flash NVFP4 dense groups take M=1 from 20.1 ms to 16.0 ms.

MiniMax-H3, 1280x736 at 124 frames with the 4-step turbo LoRA, 4xV100 at 225 W: about 80 to 90 s per job once loaded.

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
