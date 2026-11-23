# Architecture

Volta Engine is a from-scratch inference engine for tensor-parallel decoding on 4x Tesla V100 (SM70), with ROCm, CPU and Vulkan backends. Models:
Qwen3.8-27B (`qwen3_5`), GLM-5.3-Flash (`glm5_next`), Qwen3.8-Flash-Next (`qwen4_exp`), DeepSeek-V4.1-Flash (`deepseek_v41`) and Gemma 4 31B
(`gemma4`), with EXL3, NVFP4 or GGUF weights and speculative decoding ([MODELS.md](MODELS.md), [FEATURES.md](FEATURES.md)). The same server also runs
MiniMax-H3 video + audio generation, which is a separate path (`src/video/`, [VIDEO.md](VIDEO.md)) that does not use the language-model runtime. See
[GETTING_STARTED.md](GETTING_STARTED.md) to build and run, [CONFIG.md](CONFIG.md) for options.

## Layers

```
server      src/server          HTTP (epoll), OpenAI and Anthropic protocols, reasoning / tool-call parser, config; src/server/tui the terminal
                                dashboard (volta_top, --tui); src/server/video the video job service
runtime     src/runtime         ModelCluster (ranks, arenas, streams, NVLink mesh), Generator (scheduler thread, slots, stop strings, speculation),
                                PrefixCache, the speculators (DFlash2Runner, MtpRunner, DSparkRunner, SuffixRunner) and SpecController, KV pools,
                                CPU experts, loop detection, Options
models      src/models          Qwen3_5Pipeline, Glm5NextPipeline, Qwen4ExpPipeline, DeepseekV41Pipeline, Gemma4Pipeline: one pipeline instance per
                                rank; their vision towers (device and host CPU forms); loaders; forward bodies
modules     inc/engine/modules  the module API (Linear, Attention, MoE, GDN, MLA, DSA, DSV4, mHC, norms, ...), backend neutral
backend     src/backend/{cuda,cpu,vulkan}  per-backend implementation of the module API and of inc/engine/backend/runtime.h
            src/backend/rocm    compat.h and the CUDA header stand-ins that build the CUDA backend as HIP
            src/backend/stubs   weak stubs of the CUDA-only entry points for the CPU and Vulkan builds
kernels     src/kernels/cuda    hand-tuned SM70 kernels with their launchers: trellis (EXL3 GEMM / GEMV / MoE; trellis/quant the quantizer's
                                encoder), nvfp4, gguf (on llama.cpp's vendored CUDA code), fp8 (the original FP8 checkpoints' routed experts),
                                gemm (the FP16 tensor-core GEMM), h3 (MiniMax-H3 attention), train (training primitives)
            src/kernels/rocm    RDNA forms that the CUDA sources include under USE_ROCM
            src/kernels/cpu     host kernels: AVX2 EXL3 and NVFP4 linears, GGUF on llama.cpp's and ik_llama.cpp's vendored CPU code
            src/kernels/vulkan  llama.cpp's vendored GGUF mat-vec and matmul shaders
            src/kernels/slang   portable Slang kernels (CUDA, SPIR-V and host C++ outputs) and the ref_* reference kernels
core/loader src/core, src/loader  arena, logger, safetensors and GGUF readers, NVFP4 packing, tokenizer, Jinja chat templates, DeepSeek-V4.1 prompt
                                encoding, image preprocessing, n-gram table, unicode tables
comm        src/comm            2- or 4-rank NVLink all-reduce (peer pointers, flag polling)
spec        src/spec            DFlash2 / DFlash drafter, tap ring
train       src/train           the training form of qwen4_exp (with src/kernels/cuda/train: the engine_train library)
tools       src/tools           volta_quantize, volta_e2e, volta_tune, volta_sessions (CUDA builds; see QUANTIZATION.md)
video       src/video           MiniMax-H3 (CUDA only): staged pipeline, transformer, text tower, VAEs
```

`video` sits beside the language-model stack rather than under it: it has its own components (`H3Components`), its own multi-GPU groups (sequence- or tensor-parallel,
the latter over NCCL when the build has it) and its own attention kernel (`src/kernels/cuda/h3/`), and borrows the arena, the safetensors loader, the tokenizer, the
vision tower and the trellis / NVFP4 linears.

Headers under `inc/engine/` are the contracts between layers. Models, runtime and server never include CUDA headers: they use `engine/backend/runtime.h` (streams,
memcpy, pinned memory, graphs) and the module API. The backend is chosen at configure time (`-DENGINE_BACKEND=CUDA|ROCM|CPU|VULKAN`); one build tree is one
backend.

## Execution model

- **Static memory**: every GPU gets one arena allocated at boot (sized from free VRAM minus a reserve unless set); weights, KV
  pools, recurrent state, scratch and graph buffers are bump-allocated from it. Nothing is allocated while decoding.
- **Tensor parallel**: ranks 0..3 run the same pipeline on their shard; one `RankPool` thread per rank pinned to a core
  (CUDA builds). Attention heads, MoE intermediate widths or whole experts and the vocabulary head are sharded; the residual is all-reduced
  over NVLink. `tp` is a runtime value: 1, 2 or 4.
- **CUDA graphs**: decode bodies are captured once per row count (`GraphCache`) and replayed; the prefill path runs eagerly in
  chunks of up to 512 rows.
- **Speculation**: the DFlash2 drafter proposes a block, the target verifies it in one forward (per-row recurrent-state
  snapshots make rollback free); an adaptive controller falls back to plain decode when speculation loses. On targets that
  embed from device ids (GLM, Qwen, Gemma) the verify and its argmax are queued behind the draft from the drafter's device
  tokens, so the host reads the block only once the verify is done.
- **Speculation sources**: the cluster holds one `Speculator`: `DFlash2Runner` (DFlash2 drafters, and Gemma 4's DFlash v1 drafter), `MtpRunner`
  (`speculator=mtp`: the checkpoint's own one-layer MTP head on Qwen, Flash-Next and GLM; it chains on its own output and keeps a KV layer per slot
  next to the model's), `DSparkRunner` (`speculator=dspark`: DeepSeek-V4.1's block drafter, entered through the trunk's stream means) and `SuffixRunner`
  (`speculator=suffix`: the request's own earlier text, no model). All verify through the same batched forward.
- **Images**: each model's vision tower ([FEATURES.md](FEATURES.md#multimodal)) runs on rank 0 and loads last, into what the arena has left, or on the host
  CPU (`vision_cpu`); the server decodes and resizes the picture per model, the generator caches the embeddings by content hash, and the cluster
  overwrites the placeholder rows after the embedding lookup. In the Qwen family image tokens carry 3-D M-RoPE positions (`BatchRows::rope_pos`) and text
  after an image continues from the image's largest grid extent; Gemma 4's image rows attend to their whole image, so a forward holds whole images.
  Prefix-cache keys use the image content, not the placeholder ids.
- **Serving**: the scheduler thread owns the slots; the server thread parses requests, renders the chat template, and each
  response streams through a responder that parses reasoning and tool calls out of the text as it is generated. Connections
  have a non-blocking outbound queue, so a slow client cannot stall generation.
- **Prefix cache**: sequence snapshots (recurrent state, KV rows, drafter taps) in pinned host RAM, stored as a tree of
  segments keyed by exact token prefixes; a request resumes from the deepest cached prefix of its prompt.

## Tiers and tests

Each module entry point has a tier in `inc/engine/backend/tier_table.inc` (generated view: [CAPABILITIES.md](CAPABILITIES.md)):
*Portable* (Slang kernel, usable by every backend), *PortablePending* (plain reference on CPU) and *CudaSm70* (hand-tuned,
CUDA only). Verification is layered: SASS audit (`gate.sh sass`), parity against FP32 references, bandwidth benches,
multi-GPU coherence. [TESTS.md](TESTS.md) lists every test; [KERNEL_MAP.md](KERNEL_MAP.md) maps kernels to files and tests.
Hardware notes: [HARDWARE_SM70.md](HARDWARE_SM70.md).
