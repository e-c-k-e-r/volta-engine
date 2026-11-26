# Feature inventory

What the language-model side of the engine does, grouped by feature, with the option that controls it and the test that covers it.
[CONFIG.md](CONFIG.md) is the authority on options, [TESTS.md](TESTS.md) on tests, [CAPABILITIES.md](CAPABILITIES.md) on what each backend implements;
[MODELS.md](MODELS.md) describes each model, [VIDEO.md](VIDEO.md) MiniMax-H3, and [PERFORMANCE.md](PERFORMANCE.md) and
[QUANTIZATION.md](QUANTIZATION.md) hold the numbers. This page is the map between them.

Status words:
* **default**: on without any option
* **opt-in**: needs an option, a flag or a request field
* **per request**: set by a field of each request
* **debug**: a debug option (`ENGINE_DEBUG_*` in CONFIG.md), kept for A/B runs

Options are set with `--set key=value` or in the launch file's `options` object; "(A/B)" marks a debug option that turns a default path off.

## Models

| Model | Type | Layers | GPUs | Notes |
|---|---|---|---|---|
| Qwen3.8-27B | `qwen3_5` | GDN linear attention + full attention | 1, 2 or 4 | EXL3, NVFP4 or GGUF; the DFlash2 drafter or the MTP head |
| GLM-5.3-Flash | `glm5_next` | KDA linear attention (34 layers) + absorbed MLA with a DSA indexer (11 layers), routed MoE (288 experts, top 8) | 1, 2 or 4 | EXL3 or GGUF experts; NVFP4 for the dense groups; the DFlash2 drafter or the MTP head |
| Qwen3.8-Flash-Next | `qwen4_exp` | GDN + QSA block-sparse attention, 512-expert MoE, gated residual streams, one n-gram embedding layer | 1, 2 or 4 | EXL3 2/3/4 bpw experts with 7/8 bpw projections; NVFP4 experts and dense groups; GGUF; the MTP head or the DFlash2 drafter |
| DeepSeek-V4.1-Flash | `deepseek_v41` | 40 layers of sliding-window + compressed sparse attention, routed MoE (384 experts, top 6) on four hyper-connection streams, Engram n-gram lookups at two layers | 1, 2 or 4 | EXL3 (also packs that keep the FP8 attention weights) or GGUF; the DSpark drafter or suffix speculation; the CUDA backend only |
| Gemma 4 31B, E4B | `gemma4` | sliding-window attention, every 6th layer full attention (head_dim 512; the 31B's V = K), sandwich norms, an FP32 residual stream; E4B's last 18 layers on earlier layers' K / V and per-layer input embeddings; the layout from the checkpoint's config | 1, 2 or 4 (E4B: 1 or 2) | EXL3, NVFP4 or GGUF; the 31B's DFlash (v1) drafter |

The model type is read from the checkpoint, never configured. Anything else is a fatal error at start.

## Weights

| Feature | Status | Option | Covered by |
|---|---|---|---|
| EXL3 (trellis) GEMM / GEMV / MoE kernels, ported to SM70 | default | | `test_trellis`, `test_rows`, `test_moe` |
| EXL3 checkpoints in the Volta tile order (`"tile_layout": "volta"`): the decoded values are the `m8n8k4` operands, no shuffles (SM_70 CUDA builds; CPU and Vulkan read both orders) | default (per checkpoint) | | `test_trellis_order`, `test_trellis_layout` |
| int8 dp4a GEMV and MoE tiles for short runs (the 7 / 8-bit projections; experts up to 4 rows: every width in the stock tile order, 3 bits in the Volta order) | default | `moe_coop_mode` (debug) | `test_gemv_int8`, `bench_decode`, `bench_moe_routes` |
| NVFP4 (compressed-tensors and modelopt), software-unpacked FP16 `m8n8k4` kernels | opt-in | `nvfp4_experts`, `nvfp4_dense`, `nvfp4_dense_src` | `test_nvfp4_linear`, `bench_nvfp4*` |
| NVFP4 dense groups from a calibrated (LDLQ) directory instead of round-to-nearest | opt-in | `nvfp4_dense_dir`, `hessian_dir`, `hessian_layers` | `test_ppl` |
| Mixed checkpoints: the EXL3 directory stays the model, NVFP4 pieces swapped in at load (examples: `configs/flash_next_nvfp4.json`, `configs/glm_256k.json`) | opt-in | as above | |
| GLM routed experts of chosen layers from the modelopt NVFP4 checkpoint instead of EXL3 (resident, outside the expert tier) | opt-in | `nvfp4_expert_layers` (with `nvfp4_dense_src`) | `test_ppl` |
| GGUF (llama.cpp's block types Q4_0 to Q8_0, Q2_0, Q2_K to Q6_K, IQ1_S to IQ4_XS; Vulkan: all but Q2_0) on CUDA, CPU and Vulkan: integer dot products up to 8 rows, a tensor-core kernel for 9 to 16 rows, a dequantized FP16 GEMM for prefill chunks; routed experts route by route, prefill chunks on tensor-core expert tiles | default (a directory holding `.gguf` files) | `gguf_*` (debug) | `test_gguf_linear`, `test_ops_moe`, `bench_gguf_shapes`, `bench_gguf_moe` |
| DeepSeek-V4.1 packs that keep the original FP8 attention, shared-expert and indexer weights: kept FP8 in device memory (a GEMV that decodes them in registers up to 16 rows, dequantize + dense GEMM above) | default | `fp8_resident` (A/B: FP16 at load) | `test_dsv4_fp8` |
| Original FP8 checkpoints (E4M3 with F32 128 x 128 block scales, `<base>.weight_scale_inv`; GLM-5.3-Flash): routed experts kept FP8 and dequantized inside the expert GEMM (`FP8MoE`, prefill-sized tiles at any row count), every other FP8 tensor converted to FP16 at load | default (a directory whose experts are FP8) | `expert_stream`, `expert_stream_host_gib` (debug) | `test_fp8_moe`, `bench_fp8_moe`, `test_ppl` |
| DeepSeek-V4.1 packs with a width per expert projection (1 to 4 bits): one MoE launch per layer with a per-expert width table | default | `merge_expert_groups` (A/B) | `test_moe` (`MOE_DSV41=1 MOE_SIG=g,u,d`) |
| Two-tier expert residency (hot experts on the GPU, the rest in pinned host RAM, rebalanced online) | opt-in | `expert_cache` (a fraction, or `auto` on GLM), `expert_adapt_every`, `expert_swaps`, `expert_heat_file` | `test_moe`, `bench_expert_heat` |
| Cache-aware routing (a bias on the router score of resident experts, GLM) | opt-in | `route_resident_bias` | `test_router_steer` |
| Routed experts in host memory computed by the CPU during decode, in parallel with the PCIe reads of the others (AVX2 EXL3 kernel on a repacked copy, NVFP4 read in place; a pool per NUMA node, each expert split between them) | opt-in | `cpu_experts`, `cpu_expert_cpus`, `cpu_expert_node`, `cpu_expert_spin_us` | `test_cpu_experts`, `test_cpu_bridge` |
| Prefill: experts routed by many rows of a chunk or window run as FP16 dequantization + a tensor-core GEMM, the rest on the quantized kernel (logits then depend on how a prompt is chunked) | opt-in | `moe_heavy_rows` (256 in the GLM configs) | `bench_moe_prefill` |

## Attention and the cache

| Feature | Status | Option | Covered by |
|---|---|---|---|
| FP16 KV cache | default | | `test_attention`, `test_attention_prefill` |
| FP8 / int8 (`k8v4`) / NVFP4 KV (CPU and Vulkan builds included) | opt-in | `kv_format` | `test_kv_quant`, `test_mla_latent`, `test_ops_attention`, `test_ops_flash` |
| Tensor-core attention for prefill, decode and verify rows (Qwen, Gemma 4); tensor-core latent attention (GLM) | default | | `test_attention*`, `test_mla_latent`, `bench_decode` |
| Sliding-window layers keep a ring of the window plus one prefill chunk per slot (Gemma 4) | default | | `test_attention_prefill`, `test_prefix_restore --gemma` |
| GLM absorbed MLA latent cache (512 halves per token and layer) | default | | `test_mla_latent` |
| GLM DSA indexer with k-pool compression (top 2048 keys + raw tail, tensor-core scorer) | default | `dsa` | `test_dsa` |
| Flash-Next QSA block selection past 2048 tokens of context | default | `qsa_off` (debug, dense) | `test_sparse_attn`, `test_ops_flash` |
| DeepSeek-V4.1 caches in the checkpoint's own codes (window rows E4M3, compressed entries and index keys E2M1, with their block scales; `kv_format` does not apply), indexer on tensor cores for prefill rows | default | `indexer_tc` (A/B) | `test_dsv4_indexer`, `test_dsv41` |
| Shared KV token pool for all slots (GLM: latent cache and indexer keys in pages of 512 tokens, admission by free pages, length finish when full) | opt-in | `kv_pool` | `test_kv_pages` |
| GLM latent cache and indexer pools spread over the GPUs (page p on GPU p % tp, latent read over NVLink, per-GPU top-512 candidates merged) | opt-in | `kv_shard` | `test_kv_shard` |
| Context window per slot (default 32K on Qwen, 12K on GLM) | default | `max_seq`, `max_slots` | `test_longctx` (`gate.sh needle`) |
| Prefix cache: snapshots in pinned host RAM as a tree keyed by exact token prefixes, automatic checkpoints inside long prompts, a snapshot after each reply (an agent's next turn resumes after it); making room drops the least recently used leaf or strips a node with descendants to its KV rows (a long session keeps its chain); optional disk tier (evicted snapshots go to files, all of them when the server stops, and a restart reads them back on demand; kept apart per model and configuration) | default (disk: opt-in) | `prefix_cache_gib`, `prefix_interval`, `prefix_cache_reply`, `prefix_cache_dir`, `prefix_cache_disk_gib` | `test_prefix_restore`, `test_dsv41 snap=`, `tests/4_server/test_api.sh` |
| Runtime steering: per-request control vectors on the residual stream, a direction-clamp probe with a gate, a logit penalty on reflection markers, read-only probes whose scores the usage reports ([STEERING.md](STEERING.md)); Qwen3.5-family, Flash-Next, GLM, DeepSeek | opt-in | `steer`, `steer_*`, `think_penalty`, `probe` | `test_steer`, `test_steer_probe`, `test_steer_model` |
| Chunked prefill interleaved with the other slots' decode steps; `prefill_budget_ms` sizes each piece from its measured time so a decoding request waits about that long | default | `prefill_chunk`, `prefill_budget_ms` | `test_prefill_equiv`, `test_server.sh` |
| GLM, DeepSeek prefill of several 512-row chunks layer by layer, the routed experts running once over all their rows (GLM: 8 chunks; DeepSeek: as many chunks as the arena holds once every routed expert is resident) | default | `prefill_window` | `test_longctx --model glm5_next` (`gate.sh needle`) |
| DeepSeek prefill truncation (YoCo): layers 20-39 run only the tail of a prefill call, exact at -1 | default | `prefill_truncate` | |
| 32-row expert tile with two resident blocks per SM for MoE prefill chunks above 128 rows (CUDA, trellis experts) | default | `moe_prefill_tile` (A/B) | `bench_moe_prefill`, `test_glm_rows` |
| Prefill staging of cold experts (device scratch, copied a layer ahead; not for Flash-Next's trellis experts) | default | `expert_stage` | `test_glm_rows`, `bench_prefill` |
| Reasoning budget (Anthropic `thinking.budget_tokens`): when it runs out the reasoning block is closed for the model, after the next token that ends a sentence or a line (the slot decodes one token a step until then, at most `thinking_soft_close` tokens); requests that turn thinking on without a budget (adaptive thinking, an effort alone) get `thinking_budget`, at most half their `max_tokens` | default | `thinking_soft_close`, `thinking_budget` | |
| CUDA graphs for decode, one per row count | default | `no_graphs` (debug) | `test_coherence` |
| Side streams: independent chains of a decode layer run side by side (GLM: shared expert beside the MoE, KDA gate projections, the MLA kv, indexer-key and indexer-query chains; Flash-Next: shared expert, GDN z and gate projections, attention k / v projections and the QSA indexer); NVFP4 and FP16 projections only, trellis ones share one workspace per device | default | `side_streams` (A/B) | `test_glm_rows` with the option on and off |
| One kernel for the all-reduce and the hyper-connection post-step (GLM and DeepSeek, decode-sized rows) | default | `hc_fuse` (A/B) | `test_all_reduce` (bit-identical to the two launches) |
| 2-bit trellis experts: persistent kernel and fused cooperative decode kernels (Flash-Next 2.0 bpw EXL3) | default | | `bench_decode --model <checkpoint>` |
| Per-pipeline startup memory breakdown (buffers, weights by kind, drafter) | opt-in | `mem_log` | |

## Speculation

| Feature | Status | Option | Covered by |
|---|---|---|---|
| DFlash2 drafter (Qwen, GLM, Flash-Next), verified in one forward (per-row recurrent-state snapshots; GLM journals the KDA rows and replays the accepted ones), a context and tap ring per slot on GLM and Flash-Next (Qwen: one context); tensor-parallel, each rank keeping its share of the tap rows and its vocabulary shard of the selector codebooks | opt-in | `--speculative`, `--draft-model-dir`, `draft_tp` | `test_dflash*`, `test_spec_state`, `test_spec_target_equiv`, `test_coherence` |
| DFlash (v1) drafter (Gemma 4: a `DFlashDraftModel` without convs or selector, the head's argmax per row, a 16-row block of which the first 8 rows are verified; a tap ring per slot) | opt-in | `draft_model_dir` | `test_prefix_restore --gemma <dir> --draft <dir>` |
| MTP: the checkpoint's own head (Qwen, Flash-Next, GLM); GLM's head is layer 45 of the NVFP4 checkpoint, quantized to NVFP4 at load (needs `nvfp4_dense_src`); Flash-Next can take its head from the NVFP4 checkpoint | opt-in | `speculator=mtp`, `mtp_depth`, `mtp_nvfp4`, `mtp_head_dir` | `test_mtp` |
| Sparse drafter head for MTP: draft logits over a vocabulary subset picked from the tokens seen, used while its miss rate stays below the break-even | opt-in | `sparse_head`, `sparse_head_interval`, `sparse_head_file` | `test_sparse_head`, `bench_sparse_head` |
| DFlash2 blocks shortened where the drafter's recent per-position accuracy makes the later rows unlikely to be reached | opt-in | `spec_depth` | `bench_accept --depths` |
| DeepSeek-V4.1 DSpark block drafter (an EXL3 conversion of the checkpoint's `mtp.*` blocks in `draft_model_dir`): 5-token drafts from the trunk's stream means, confidence-cut, one verify forward | opt-in | `speculator=dspark`, `dspark_conf` | |
| Suffix speculation: the block is the continuation of the last earlier occurrence of the current suffix of the request's own text (copied code, tool arguments), no draft pass; inside DFlash2 rounds, or as the only source (`speculator=suffix`, DeepSeek) | opt-in | `spec_suffix`, `spec_suffix_min`, `speculator=suffix` | |
| Adaptive controller that falls back to plain decode when speculation loses | default | `spec_ctrl` | `bench_accept`, `bench_spec` |
| One verify forward for the blocks of all speculating slots (DFlash2: drafted in turn; MTP: off by default, one draft chain too) | default (DFlash2) | `spec_joint` (debug) | `test_api.sh` |

After a prefix restore the MTP head does one plain step first, because it does not hold the hidden state of the last restored position.

## Parallelism and scheduling

* tensor parallel over NVLink with a 2- or 4-rank all-reduce (`tp`, `--tp-size`): every model on 1, 2 or 4 ranks; the Vulkan build does the same over several Vulkan
  devices with the all-reduce meeting in host memory the devices import
  * a second server can run on the other pair of GPUs (`CUDA_VISIBLE_DEVICES=2,3`, `--tp-size 2`)
* one arena per GPU, allocated at boot from the free memory minus a reserve for what the CUDA driver takes later (`arena_gib`, `vram_reserve_mib`, 384 MiB); the
  kernels' workspaces come out of the arena (`scratch_mib`), and nothing allocates while decoding. On Vulkan the free memory is the device-local heap's
  budget less its usage (`VK_EXT_memory_budget`: the driver's and other processes' allocations counted), or without it the heap less this process's allocations
* up to `max_slots` concurrent sequences, `max_batch_size` of them stepped together; the scheduler thread owns the slots
* the image embeddings and the vision tower run on rank 0 only

Covered by `test_all_reduce`, `bench_all_reduce`, `test_coherence`, `tests/4_server/test_server.sh`.

## Multimodal

| Model | Vision tower | Notes |
|---|---|---|
| Qwen | 27 LayerNorm / GELU blocks, tensor-core attention, M-RoPE (interleaved 11/11/10) | Qwen2-VL smart resize |
| GLM | 24 RMSNorm / SwiGLU blocks with q/k norm, 2x2 conv merge | aspect-preserving resize onto a zero-padded canvas aligned to 28; embeddings only, no M-RoPE |
| Flash-Next | the checkpoint's own tower (the Qwen3.5 family's) | |
| DeepSeek | 32 RMSNorm / SwiGLU blocks, fused qkv, 2-D rotary, whole-image attention; a 3x3 aligner (raster order, zero padded) | grey-padded resize under a 1024-token budget; each image is a span of start, rows with a newline each, end; image tokens route with `bias_vl` and skip the Engram n-grams; the tower's memory comes out of rank 0's resident experts |
| Gemma 4 | 27 sandwich-normed ViT blocks (per-head q / k norms, a 2-D rotary, a GELU-tanh gated MLP), 3 x 3 average pooling into at most 280 soft tokens | BF16 `model.vision_tower.*` tensors loaded as FP16, or a GGUF directory's `mmproj` file; image rows attend to their whole image, and a prefill forward holds whole images; CUDA, CPU and Vulkan |

* Option: `max_image_tokens` (default 1024 language-model tokens per image, 0 turns images off). The tower loads last into what the arena has left; GLM with the drafter
  leaves too little, so images stay off there unless memory is freed.
* Input is base64 `data:` URLs or Anthropic base64 sources, also inside `tool_result`. Remote URLs are not fetched on purpose (a request-forgery surface, and Claude
  Code sends base64).
* Option: `vision_cpu` (GLM-5, DeepSeek-V4.1 and Flash-Next (the EXL3 or dense checkpoint tower, or its mmproj GGUF); a token limit like
  `max_image_tokens`, which then stays 0) runs the tower on the host CPU, so the GPUs keep no tower weights or scratch: the tower's projections are
  decoded once into FP16 in host memory and the operators run on `vision_cpus` (default: one thread per core of the last NUMA node, away from the GPU
  rank threads). `vision_dir` names the checkpoint that holds the tower (its `config.json` and `processor_config.json`) when the server runs another
  one, e.g. a GGUF, and `vision_mmproj` a directory with the model's vision projector GGUF (`*mmproj*.gguf`, BF16: the original weights, no EXL3
  decode) to take the weights from instead. A request with images encodes them on its own thread while the slots keep decoding. Covered by
  `test_host_vision` (the host tower against the device tower and, with an mmproj directory, against the BF16 weights) and `test_host_attention`.
* The vision tower of a Volta-order EXL3 checkpoint is read in the stock tile order (`model.visual.*`: the quantizer skips it, see [QUANTIZATION.md](QUANTIZATION.md));
  read in the Volta order its embeddings are noise.
* Embeddings are cached by content hash (1 GiB of host RAM), and prefix-cache keys use the image content instead of the placeholder ids.
* Covered by `test_vision`, `test_ops_vision`, `test_image`, `test_dsv41_vision`, `test_gemma4_vision`, `test_api.sh`.

## Serving

| Feature | Status | Notes |
|---|---|---|
| OpenAI `/v1/chat/completions`, `/v1/models` | default | streaming, `stream_options.include_usage`, `stop`, `seed` |
| Anthropic `/v1/messages`, `/v1/messages/count_tokens` | default | what Claude Code connects to; server tools (`web_search_*`, `web_fetch_*`, `code_execution_*`, ...) are refused with a 400 that names them, since nothing here runs them |
| Waiting requests: a streaming response starts at once and sends keepalives every 10 s while the request waits for a slot or prefills (`ping` events, SSE comments on OpenAI); a client that disconnects cancels its request there too (long prompts are prefilled in 8192-token pieces between the checks) | default | |
| Request priority: lower `priority` values take a free slot first (equal ones in arrival order, running requests are not preempted); requests whose model name contains one of `low_priority_models` (default `haiku`, Claude Code's side calls) default to 1, the rest to 0 | default | `priority` field, `low_priority_models` |
| API key: `/v1/*` requests must carry `Authorization: Bearer KEY` or `x-api-key: KEY`; the server warns at start when it listens beyond localhost without one | opt-in | `api_key` / `--api-key` |
| Loop detection: a request whose output settles into repeating a span is logged or ended (finish reason `stop` / `end_turn`; `/api/stats` counts it as `loop`) | opt-in | `loop_detect` off / report / stop, request field `stop_on_loop` |
| Tools in either shape, `tool_choice` `required` / named | default | best effort, no constrained decoding |
| Reasoning and tool-call parsing from the text stream | default | Qwen, GLM, DeepSeek-V4.1 (DSML) and Gemma 4 syntaxes; independent of how the stream is chunked |
| Jinja chat templates | default | own subset, tested against reference renders; DeepSeek-V4.1 uses the checkpoint's own prompt encoding, built in |
| Escaping of control tokens inside message content | default | quoted source cannot end or open a turn |
| Sampling: temperature, top-p, top-k (capped at 16), seed | default | greedy = GPU argmax; sampled = GPU top-16 merged across shards |
| Penalties: `presence_penalty`, `frequency_penalty` (over the generated tokens) and `repetition_penalty` (multiplicative, over prompt and output) | per request | applied to the top-16 candidates of every row, verify rows included (each row counts the drafts before it), so speculation stays exact; `presence_penalty` and `repetition_penalty` default from `generation_config.json`; the Anthropic front end accepts them as extensions |
| Log-probabilities: `logprobs`, `top_logprobs` (at most 16 alternatives) | per request | raw model log-probabilities (before temperature, truncation and penalties), normalised over the whole vocabulary; streamed with the chunk that carries the token's text; reasoning tokens included |
| Prefix-cache usage reporting | default | `cached_tokens` / `cache_read_input_tokens` |
| `/health`, `/metrics`, `/metrics/prometheus` | default | |
| `/monitor`: throughput, slots, TTFT and token-gap percentiles, speculation, prefix cache, per-GPU NVML power / clocks / throttle, MoE routing heat | default | `expert_stats` |
| Terminal dashboard: `volta_top` attaches to a running server, `--tui` runs it inside the server (slots, throughput, GPUs, log) | default | `tui` |
| Request dumps for debugging and calibration text | debug | `dump_requests` |

Covered by `test_output_parser`, `test_jinja`, `test_dsv41_encoding`, `test_tokenizer*`, `test_loop_detect`, `test_sampling`, `test_sampling_gpu`, `tests/4_server/*`.

Not parsed: `n` above 1.

## Backends

| Backend | State |
|---|---|
| CUDA SM70 | the primary build: everything above |
| ROCm | the CUDA sources built as HIP for RDNA (wave32) GPUs, with RDNA forms of the hot kernels; see [ROCM.md](ROCM.md) |
| CPU | one rank: Qwen, Flash-Next and Gemma 4 checked against CUDA logits (`test_cpu_forward`); reference kernels, GGUF through llama.cpp's and ik_llama.cpp's CPU kernels; slow by design |
| Vulkan | one device or tensor-parallel over 2 or 4 devices (all-reduce in shared host memory; a captured decode graph is split at its cross-device syncs): Qwen, Flash-Next (GGUF and NVFP4 experts), Gemma 4 and GLM's GGUF experts checked against CUDA (`test_vulkan_forward`); quantized KV; GGUF through llama.cpp's shaders; trellis or FP16 routed experts read their routes back, so those decode forwards run eagerly |

The DFlash2 drafter, DeepSeek-V4.1 and the video service need the CUDA backend (CUDA or ROCm builds); the MTP head runs on CPU and Vulkan too (`test_mtp`). The
per-function table is [CAPABILITIES.md](CAPABILITIES.md).
