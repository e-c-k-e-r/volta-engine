# Engine

Yet another (slopped) inference engine, tailored in C++ for Volta, specifically 4xV100 systems.

## Build

CUDA 12.x is required, as 13+ drops SM_70 support. *Could* run on SM_75+, but no kernels would make use of instructions / specifications introduced later.

# Engine

Yet another (slopped) inference engine, tailored in C++ for Volta, specifically 4xV100 systems.

## Build

CUDA 12.x is required, as 13+ drops SM_70 support. *Could* run on SM_75+, but no kernels would make use of instructions / specifications introduced later.

To compile, run `cmake -B build && cmake --build build`.

## Run

*to-do: fill out*

## Why?

This aims to milk every drop of perf from my V100s.

Most inference engines treat Volta (SM_70) as a second class citizen and fall back to eager, unfused kernels. Recently, various forks are coming about to address that, but each also come with their pros and cons.
* 1Cat-vLLM boasts the best performance out of the box, but only really guarantees that with NVFP4 releases and is quite a hefty stack. Models that require higher quantization like GLM-5.3-Flash are boxed out (and solutions like `vllm-exl3` are severely capped).
* exllamav3, while boasting higher quantization and lower perplexity loss, can only achieve ~60% of the cap, even when gutting out and tailoring it for SM_70.
* llama.cpp is a mixed bag, and forking it dooms it to never get rebased without headaches (and submitting it as a valid PR is a fools errand).

### Why not?

This project was slopped up over a weekend. CUDA is NOT my expertise (Vulkan is).

I don't intend to turn this into an actual project, as existing solutions should be servicable enough for most people's usecases.

A slim amount of models (Qwen3.8-27B and GLM-5.3-Flash) and model formats (EXL3, NVFP4) are supported.

## Notices

Unless otherwise credited/noted in this repo or within the designated file, this repository is licensed under AGPLv3.

CUDA kernels heavily referenced from:
* [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) for NVFP4 kernels
* [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) for NVFP4 kernels
* [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) for EXL3 kernels
* a bunch of extra slopped optimizations from abandoned private experiments of the above
