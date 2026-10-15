# Feature inventory

What the language-model side of the engine does today, grouped by feature, with the option that controls it and the test that covers it. [CONFIG.md](CONFIG.md) is the authority on options, [TESTS.md](TESTS.md) on tests, [CAPABILITIES.md](CAPABILITIES.md) on what each backend implements, and the model-specific pages ([FLASH_NEXT.md](FLASH_NEXT.md), [video.md](video.md) for MiniMax-H3) hold the numbers. This page is the map between them.

Status words:
* **default**: on without any option
* **opt-in**: needs an option or a flag
* **debug**: an `ENGINE_DEBUG_*` switch, kept for A/B runs
* **not built**: considered or scaffolded, and not available

Speed figures are indicative: the GPUs run power-capped (150 to 225 W depending on the session), most are single runs, and the proper per-config matrix is still to be done.

## Models

| Model | Type | Layers | GPUs | Notes |
|---|---|---|---|---|
| Qwen3.8-27B | `qwen3_5` | GDN linear attention + full attention | 1, 2 or 4 | EXL3 or NVFP4; the DFlash2 drafter or the MTP head |
| GLM-5.3-Flash | `glm5_next` | KDA linear attention (34 layers) + absorbed MLA with a DSA indexer (11 layers), routed MoE (288 experts, top 8) | 1, 2 or 4 | EXL3 experts; NVFP4 for the dense groups |
| Qwen3.8-Flash-Next | `qwen4_exp` | GDN + QSA block-sparse attention, 512-expert MoE, gated residual streams, one n-gram embedding layer | 4 only | EXL3 3/4 bpw experts with 7/8 bpw projections; NVFP4 experts and dense groups |

The model type is read from the checkpoint, never configured. Anything else is a fatal error at start.

## Weights

| Feature | Status | Option | Covered by |
|---|---|---|---|
| EXL3 (trellis) GEMM / GEMV / MoE kernels, ported to SM70 | default | | `test_trellis`, `test_rows`, `test_moe` |
| int8 dp4a GEMV and MoE tiles for short runs (the 7 / 8-bit projections, GLM experts up to 4 rows) | default | | `test_gemv_int8`, `bench_decode` |
| NVFP4 (compressed-tensors and modelopt), software-unpacked FP16 `m8n8k4` kernels | opt-in | `nvfp4_experts`, `nvfp4_dense`, `nvfp4_dense_src` | `test_nvfp4_linear`, `bench_nvfp4*` |
| NVFP4 dense groups from a calibrated (LDLQ) directory instead of round-to-nearest | opt-in | `nvfp4_dense_dir`, `hessian_dir`, `hessian_layers` | `test_ppl`, `tests/tools/nvfp4_ldlq.py` |
| Mixed checkpoints: EXL3 directory stays the model, NVFP4 pieces swapped in at load | opt-in | as above | `configs/flash_next_nvfp4.json`, `configs/qwen_nvfp4.json`, `configs/glm_nvfp4_dense.json`, `configs/glm_256k.json` |
| Two-tier expert residency (hot experts on the GPU, the rest in pinned host RAM, rebalanced online) | opt-in | `expert_cache`, `expert_adapt_every`, `expert_swaps` | `test_moe`, `bench_expert_heat` |

* NVFP4 experts for GLM are not built: they do not fit in VRAM and the offload penalty rules them out. Dense NVFP4 groups on GLM take the 8-row verify from 48 to about 34 ms.
* The expert tier is a capacity valve, not a speed feature: GLM and Flash-Next routing is trained for balance, so the hit rate at 50% residency is only 0.6 to 0.7 and decode drops to about a third.

## Attention and the cache

| Feature | Status | Option | Covered by |
|---|---|---|---|
| FP16 KV cache | default | | `test_attention`, `test_attention_prefill` |
| Prefill staging of cold experts (device scratch, copied a layer ahead) | on | `expert_stage` | `test_glm_rows`, `bench_prefill` |
| Shared KV token pool for all slots (GLM: latent cache and indexer keys in pages of 512 tokens, admission by free pages, length finish when full) | opt-in | `kv_pool` | `test_kv_pages` |
| FP8 / int8 (`k8v4`) / NVFP4 KV (CPU and Vulkan builds included) | opt-in | `kv_format` | `test_kv_quant`, `test_mla_latent`, `test_ops_attention`, `test_ops_flash` |
| Tensor-core attention for prefill, decode and verify rows (Qwen); tensor-core latent attention (GLM) | default | | `test_attention*`, `test_mla_latent`, `bench_decode` |
| GLM absorbed MLA latent cache (512 halves per token and layer) | default | | `test_mla_latent` |
| GLM DSA indexer with k-pool compression (top 2048 keys + raw tail, tensor-core scorer) | default | `dsa` | `test_dsa` |
| Flash-Next QSA block selection past 2048 tokens of context | default | `qsa_off` (debug, dense) | `test_sparse_attn`, `test_ops_flash` |
| Context window per slot | default 32K (Qwen), 12K (GLM) | `max_seq`, `max_slots` | `test_longctx`, `test_long_context` |
| Prefix cache: snapshots in pinned host RAM as a tree keyed by exact token prefixes, automatic checkpoints inside long prompts | default | `prefix_cache_gib`, `prefix_interval` | `test_prefix_restore`, `tests/4_server/test_api.sh` |
| Chunked prefill interleaved with the other slots' decode steps | default | `prefill_chunk` | `test_prefill_equiv`, `test_server.sh` |
| CUDA graphs for decode, one per row count | default | `no_graphs` (debug) | `test_coherence` |

Validated contexts:
* Qwen: needle retrieval to 258K with 2 slots, TP4 and speculation (25 GiB per GPU); the 258K cold prefill takes about 5 minutes, a cached repeat 1 to 2 s
* Flash-Next: needles at 12K, 32K, 100K and 250K pass (250K: prefill 848 tok/s, decode 42 tok/s); 256K with 2 slots fits
* GLM: 256K with 2 slots runs with `kv_format=nvfp4 expert_cache=0.95` (a 32K prompt: 68 s prefill, 53 tok/s decode; the API checks were run with `kv_format=fp8 expert_cache=0.85`); the 256K needle itself is verified at FP16, 1 slot, no drafter (prefill 208 tok/s, decode 14.8 tok/s)

## Speculation

| Feature | Status | Option | Covered by |
|---|---|---|---|
| DFlash2 drafter (Qwen, GLM), verified in one forward with per-row recurrent-state snapshots | opt-in | `--speculative`, `--draft-model-dir`, `draft_tp` | `test_dflash*`, `test_spec_state`, `test_spec_target_equiv`, `test_coherence` |
| MTP: the checkpoint's own head (Qwen, Flash-Next, GLM) | opt-in | `speculator=mtp`, `mtp_depth` | `test_mtp` |
| Adaptive controller that falls back to plain decode when speculation loses | default | `spec_ctrl` | `bench_accept`, `bench_spec` |
| One draft chain and one verify forward for all speculating slots | debug | `spec_joint` | `test_api.sh` |

* Greedy output with speculation is identical to plain decoding; this is what `test_mtp` and `test_coherence` check.
* Qwen: MTP about 1.7x plain decoding against 2.4x for DFlash2, and it needs no draft model. Flash-Next with MTP: 185 tok/s on a short prompt, 132 tok/s on a 15K prompt (82% and 63% accepted).
* GLM's head is layer 45 of the NVFP4 checkpoint (a plain MLA + MoE layer without hyper-connections; its BF16 weights and 288 experts are quantized to NVFP4 at load, this rank's slice), so it needs `nvfp4_dense_src`. It matches the DFlash2 drafter's mean speed on the natural prompts (101 tok/s, 45 to 56% of drafts accepted) without the drafter's VRAM, and takes about 1 GiB per GPU, which leaves the 12K window about 0.1 GiB of spare arena. Greedy output matches plain decoding for the first tokens; later ones can differ at near-ties, as with the DFlash2 drafter.
* `spec_joint` is about even with the per-slot rounds at two requests and ahead at four, so it stays off.
* After a prefix restore the MTP head does one plain step first, because it does not hold the hidden state of the last restored position.

## Parallelism and scheduling

* tensor parallel over NVLink with a 2- or 4-rank all-reduce (`tp`, `--tp-size`): Qwen 1/2/4, GLM 1, 2 or 4, Flash-Next 4; the Vulkan build does the same over several Vulkan devices with the all-reduce staged through host memory
  * `--tp-size 2` is about 0.7x the speed of four; a second server on the other pair doubles the aggregate (`CUDA_VISIBLE_DEVICES`)
* one arena per GPU, allocated at boot from the free memory minus a reserve (`arena_gib`, `vram_reserve_mib`); nothing allocates while decoding
* up to `max_slots` concurrent sequences, `max_batch_size` of them stepped together; the scheduler thread owns the slots
* the image embeddings and the vision tower run on rank 0 only

Covered by `test_all_reduce`, `bench_all_reduce`, `test_coherence`, `tests/4_server/test_server.sh`.

## Multimodal

| Model | Vision tower | Notes |
|---|---|---|
| Qwen | 27 LayerNorm / GELU blocks, tensor-core attention, M-RoPE (interleaved 11/11/10) | Qwen2-VL smart resize |
| GLM | 24 RMSNorm / SwiGLU blocks with q/k norm, 2x2 conv merge | aspect-preserving resize onto a zero-padded canvas aligned to 28; embeddings only, no M-RoPE |
| Flash-Next | the checkpoint's own tower | |

* Option: `max_image_tokens` (default 1024 language-model tokens per image, 0 turns images off). The tower loads last into what the arena has left; GLM with the drafter leaves too little, so images stay off there unless memory is freed.
* Input is base64 `data:` URLs or Anthropic base64 sources, also inside `tool_result`. Remote URLs are not fetched on purpose (a request-forgery surface, and Claude Code sends base64).
* Embeddings are cached by content hash (1 GiB of host RAM), and prefix-cache keys use the image content instead of the placeholder ids.
* Covered by `test_vision`, `test_ops_vision`, `test_image`, `test_api.sh`.

## Serving

| Feature | Status | Notes |
|---|---|---|
| OpenAI `/v1/chat/completions`, `/v1/models` | default | streaming, `stream_options.include_usage`, `stop`, `seed` |
| Anthropic `/v1/messages`, `/v1/messages/count_tokens` | default | what Claude Code connects to |
| Tools in either shape, `tool_choice` `required` / named | default | best effort, no constrained decoding |
| Reasoning and tool-call parsing from the text stream | default | Qwen and GLM syntaxes; independent of how the stream is chunked |
| Jinja chat templates | default | own subset, tested against reference renders |
| Escaping of control tokens inside message content | default | quoted source cannot end or open a turn |
| Sampling: temperature, top-p, top-k (capped at 16), seed | default | greedy = GPU argmax; sampled = GPU top-16 merged across shards |
| Prefix-cache usage reporting | default | `cached_tokens` / `cache_read_input_tokens` |
| `/health`, `/metrics`, `/metrics/prometheus` | default | |
| `/monitor`: throughput, slots, TTFT and token-gap percentiles, speculation, prefix cache, per-GPU NVML power / clocks / throttle, MoE routing heat | default | `expert_stats` |
| Request dumps for debugging | debug | `dump_requests` |

Covered by `test_output_parser`, `test_jinja`, `test_tokenizer*`, `test_sampling`, `test_sampling_gpu`, `tests/4_server/*`.

Not parsed: repetition / presence / frequency penalties, `logprobs`, `n` above 1.

## Backends

| Backend | State |
|---|---|
| CUDA SM70 | the primary build: everything above |
| CPU | Qwen and Flash-Next on one rank, reference kernels, slow by design (Flash-Next needs ~66 GiB and matches the CUDA logits to cosine 0.9999992) |
| Vulkan | Qwen on one device or tensor-parallel over 2 or 4 devices (all-reduce through the host, eager); Flash-Next on four devices (cosine 0.999999 against CUDA tp=4); quantized KV; models with a routed MoE need `ENGINE_DEBUG_NO_GRAPHS=1` on one device |

The DFlash2 drafter is CUDA only, the MTP head has not run on Vulkan, and the video service is CUDA only. The per-function table is [CAPABILITIES.md](CAPABILITIES.md).

## Known limits and open items

* **GLM chunk sensitivity**: the sigmoid top-8-of-288 router flips experts under any rounding-level noise, so an unaligned prefill split shifts logits by 0.2 to 0.7 (exllamav3 shows the same); aligned cuts are exact. Not an engine bug.
* **NVFP4 on images**: the MiniMax-H3 text tower's NVFP4 checkpoint is worse on image prompts (cosine 0.974) than its EXL3 6 bpw one; see [video.md](video.md).
* **Sampling throughput**: a non-zero temperature is slower than greedy. The sampled step adds a top-k kernel, a sync and a copy, and speculation accepts less under sampling; neither effect has been measured separately.
* **Flash-Next thinking**: a request can end with `end_turn` after an unclosed thinking block and show nothing to the client; candidate fixes are listed in [FLASH_NEXT.md](FLASH_NEXT.md).
* **GLM default window** is 12K because the FP16 drafter needs the last GiB of the arena; 1024-row prefill chunks would be 13% faster but do not fit beside it.
* **Open levers**: an M = 8 variant of the int8 GEMV for Flash-Next's 7 / 8-bit projections, the fused MoE at 8 rows, QSA scoring on tensor cores, fusing the gated-residual kernels.
* **CPU / Vulkan**: the remaining stubs are the entries marked CudaSm70 with a note in [CAPABILITIES.md](CAPABILITIES.md).
