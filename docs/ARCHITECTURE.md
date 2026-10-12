# Architecture

Volta Engine is a from-scratch inference engine for tensor-parallel decoding on 4x Tesla V100 (SM70), with CPU and Vulkan
backends for the portable parts. Models: Qwen3.8-27B (`qwen3_5`: GDN linear attention + full attention, EXL3 trellis or NVFP4
weights) and GLM-5.3-Flash (`glm5_next`: KDA linear attention + absorbed MLA with a sparse indexer, routed MoE), each with a
DFlash2 speculative drafter, plus Qwen3.8-Flash-Next (`qwen4_exp`, see [FLASH_NEXT.md](FLASH_NEXT.md)). The same server also runs MiniMax-H3 video + audio generation, which is
a separate path (`src/video/`, [video.md](video.md)) that does not use the language-model runtime. See [GETTING_STARTED.md](GETTING_STARTED.md) to build and run, [CONFIG.md](CONFIG.md) for options.

## Layers

```
server      src/server        HTTP (epoll), OpenAI and Anthropic protocols, reasoning / tool-call parser, config
runtime     src/runtime       ModelCluster (ranks, arenas, streams, NVLink mesh), Generator (scheduler thread, slots,
                              stop strings, speculation), PrefixCache, DFlash2Runner, KV pools, Options
models      src/models        Qwen3_5Pipeline, Glm5NextPipeline: one pipeline instance per rank; loaders; forward bodies
modules     inc/engine/modules  the module API (Linear, Attention, MoE, GDN, MLA, DSA, mHC, norms, ...), backend neutral
backend     src/backend/{cuda,cpu,vulkan}  per-backend implementation of the module API and of inc/engine/backend/runtime.h
kernels     src/kernels/cuda  hand-tuned SM70 kernels (trellis GEMM / GEMV / MoE, NVFP4) with their launchers
            src/kernels/slang portable Slang kernels (CUDA, SPIR-V and host C++ outputs) and the ref_* reference kernels
core/loader src/core, src/loader  arena, logger, safetensors, tokenizer, Jinja chat templates, unicode tables
comm        src/comm          2- or 4-rank NVLink all-reduce (peer pointers, flag polling)
spec        src/spec          DFlash2 drafter, tap ring
video       src/video         MiniMax-H3 (CUDA only): staged pipeline, transformer, text tower, VAEs; src/server/video is its job service
```

`video` sits beside the language-model stack rather than under it: it has its own components (`H3Components`), its own multi-GPU groups (sequence- or tensor-parallel, the latter over NCCL
when the build has it) and its own attention kernel (`src/kernels/cuda/h3/`), and borrows the arena, the safetensors loader, the tokenizer, the vision tower and the trellis / NVFP4 linears.

Headers under `inc/engine/` are the contracts between layers. Models, runtime and server never include CUDA headers: they
use `engine/backend/runtime.h` (streams, memcpy, pinned memory, graphs) and the module API. The backend is chosen at configure
time (`-DENGINE_BACKEND=CUDA|CPU|VULKAN`); one build tree is one backend.

## Execution model

- **Static memory**: every GPU gets one arena allocated at boot (sized from free VRAM minus a reserve unless set); weights, KV
  pools, recurrent state, scratch and graph buffers are bump-allocated from it. Nothing is allocated while decoding.
- **Tensor parallel**: ranks 0..3 run the same pipeline on their shard; one `RankPool` thread per rank pinned to a core
  (CUDA builds). Attention heads, MoE intermediate widths and the vocabulary head are sharded; the residual is all-reduced
  over NVLink. `tp` is a runtime value (1, 2 or 4; GLM 1 or 4).
- **CUDA graphs**: decode bodies are captured once per row count (`GraphCache`) and replayed; the prefill path runs eagerly in
  chunks of up to 512 rows.
- **Speculation**: the DFlash2 drafter proposes a block, the target verifies it in one forward (per-row recurrent-state
  snapshots make rollback free); an adaptive controller falls back to plain decode when speculation loses.
- **Speculation sources**: besides DFlash2, `speculator=mtp` drafts with the checkpoint's own one-layer MTP head (Qwen): it chains on
  its own output, keeps a KV layer per slot next to the model's, and verifies through the same batched forward.
- **Images**: each model's vision tower (Qwen: 27 LayerNorm / GELU blocks; GLM: 24 RMSNorm / SwiGLU blocks with q/k norm; both 2-D rotary) runs on rank 0 and loads last, into what the arena has left; the server decodes and
  resizes the picture (Qwen: Qwen2-VL smart resize; GLM: an aspect-preserving resize onto a zero-padded canvas aligned to 28 px), the generator caches the embeddings by content hash, and the cluster overwrites the
  placeholder rows after the embedding lookup. Image tokens carry 3-D M-RoPE positions (`BatchRows::rope_pos`), text after an image
  continues from the image's largest grid extent. Prefix-cache keys use the image content, not the placeholder ids.
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
