# Models

One section per model type. The type is read from the checkpoint's `config.json` (`model_type`), never configured; anything else is a fatal
error at start, and a MiniMax-H3 pipeline directory starts the video service instead ([VIDEO.md](VIDEO.md)). Options are described in
[CONFIG.md](CONFIG.md), weight formats, bit widths and perplexity in [QUANTIZATION.md](QUANTIZATION.md), speed in [PERFORMANCE.md](PERFORMANCE.md),
and the features shared by every model (prefix cache, expert tier, speculation, images, serving) in [FEATURES.md](FEATURES.md).

| Model | Type | Weights | `tp` | Speculators | Images | Backends |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | `qwen3_5` | EXL3, NVFP4, GGUF | 1, 2, 4 | DFlash2, MTP | yes | CUDA, CPU, Vulkan |
| GLM-5.3-Flash | `glm5_next` | EXL3 (stock or Volta tile order), NVFP4 dense groups and expert layers, GGUF | 1, 2, 4 | DFlash2, MTP | yes | CUDA, CPU, Vulkan |
| Qwen3.8-Flash-Next | `qwen4_exp` | EXL3, NVFP4, GGUF | 1, 2, 4 | MTP, DFlash2 | yes | CUDA, CPU, Vulkan |
| Gemma 4 31B | `gemma4` | EXL3, NVFP4, GGUF | 1, 2, 4 | DFlash, suffix | yes | CUDA, CPU, Vulkan |
| DeepSeek-V4.1-Flash | `deepseek_v41` | EXL3, GGUF | 1, 2, 4 | DSpark, suffix | yes | CUDA |

* Speculators (`speculator`, with `--speculative` / `use_speculation`): `dflash` (the default: a DFlash2 or DFlash drafter in `draft_model_dir`), `mtp`
  (the checkpoint's own head, no draft model), `dspark` (DeepSeek's block drafter in `draft_model_dir`) and `suffix` (no model: the text that followed the
  last earlier occurrence of the request's current suffix; needs `spec_suffix` > 0 and a model whose caches are indexed by position, so not the models
  with recurrent state). `spec_suffix` also offers such matches inside DFlash2 and DSpark rounds. The drafters are CUDA only.
* GGUF: a model directory holding the `.gguf` file, or every file of a split one (`-00001-of-0000N.gguf` ...), plus the original `config.json`,
  tokenizer files and chat template, loads like a safetensors one. Block types Q4_0 to Q8_0, Q2_0, Q2_K to Q6_K and IQ1_S to IQ4_XS; norms,
  embeddings and other small tensors are dequantized once at load. Routed experts run route by route on llama.cpp's dot products in decode and
  verify blocks, and prefill chunks of more than 64 rows decode each expert's weight tiles to FP16 and multiply its routes on the tensor cores;
  they take part in the expert tier. The CPU backend runs GGUF projections through llama.cpp's CPU kernels (the q8_K types through ik_llama.cpp's
  iqk kernels), the Vulkan backend through llama.cpp's mat-vec and matmul shaders and Slang kernels (every type but Q2_0).
* CPU and Vulkan builds: the routed experts are always resident there (no expert tier). What has been checked against CUDA on each backend
  is listed in [TESTS.md](TESTS.md).
* Images: OpenAI `image_url` parts (base64 `data:` URLs) or Anthropic `image` blocks, also inside `tool_result` blocks; remote URLs are not
  fetched. `max_image_tokens` (default 1024 language-model tokens per image, 0 = off) sizes the tower's scratch; the tower runs on rank 0 and loads
  last, into what is left of the arena. Without a loaded tower an image part reaches the model as the note `[image omitted: this model cannot view
  images]`. `vision_cpu` runs the tower on the host CPU instead (every model but Gemma 4; [FEATURES.md](FEATURES.md#multimodal)).

## Qwen3.8-27B (`qwen3_5`)

* 64 layers: 48 GDN linear-attention layers and every 4th a full-attention layer (24 Q heads, 4 KV heads, head_dim 256, rotary over 64 dims);
  hidden 5120, a dense MLP of 17408, vocabulary 248320. The GDN state and conv history are kept per slot.
* Weights: EXL3 (`Qwen3.8-27B-exl3`), NVFP4 (`Qwen3.8-27B-QUASAR-NVFP4`, runs as it is; `nvfp4_dense=head` quantizes the FP16 head to NVFP4
  at load), GGUF (llama.cpp's `qwen35` architecture; the MTP head is read from the GGUF file).
* Speculation: the DFlash2 drafter (`draft_model_dir`; taps of layers 5, 19, 33, 47 and 61; also a GGUF conversion, llama.cpp's `dflash`
  architecture, with the release's `config.json`), or `speculator=mtp`. `sparse_head` gives the MTP draft steps a head over the most frequent tokens.
* Images: the checkpoint's vision tower (27 LayerNorm / GELU blocks, M-RoPE).
* Defaults: `max_seq` 32768, `max_slots` 8; `embed_host` keeps the embedding table in pinned host memory (one copy for every rank; on Vulkan,
  which has no device-readable pinned host memory, a copy per device).

```
build/volta_server --config configs/example_qwen.json                                     # EXL3 + DFlash2 drafter
build/volta_server --config configs/qwen_agent.json                                       # 256K window, 2 slots
build/volta_server --config configs/qwen_nvfp4.json                                       # the NVFP4 checkpoint
build/volta_server --model-dir <Qwen3.8-27B-exl3> --speculative --set speculator=mtp     # the MTP head
```

## GLM-5.3-Flash (`glm5_next`)

* 45 layers on four hyper-connection streams: 34 KDA linear-attention layers and 11 absorbed-MLA layers (layers 3, 7, ..., 43; the cache holds
  a 512-wide latent per token, no decoupled rope key) whose DSA indexer selects 2048 keys per row (`dsa`; false attends densely). Layers 0-2
  have a dense MLP (12288), layers 3-44 a routed MoE of 288 experts (top 8, intermediate 2048) plus a shared expert. Hidden 4096, vocabulary 154880.
* Weights: EXL3 (`GLM-5.3-Flash-exl3-3.05bpw`, or a re-encode in the Volta tile order, `configs/glm_volta.json`; see
  [QUANTIZATION.md](QUANTIZATION.md#volta-tile-order-exl3)). The dense groups can come from the modelopt NVFP4 checkpoint: `nvfp4_dense_src=<NVFP4 dir>`
  with `nvfp4_dense=kda,mla,idx,head,dense,shared` (`dense` is the MLP of layers 0-2), and `nvfp4_dense_dir` for calibrated tensors; the routed
  experts stay EXL3, except the layers `nvfp4_expert_layers` names, which load their experts from the NVFP4 checkpoint (resident, outside the
  expert tier). GGUF: llama.cpp's `glm5-next` architecture (its MTP block is not read).
* The original checkpoint (`GLM-5.3-Flash`, FP8 E4M3 with F32 128 x 128 block scales, about 350 GB) runs as it is: the routed experts stay FP8 (`FP8MoE`,
  dequantized inside the expert GEMM), the other FP8 tensors and the BF16 ones are converted to FP16 at load, and q / k / v and their convolutions are
  stacked per rank as the conversions' fused `qkv_proj` / `conv1d` hold them. Its experts are larger than host RAM: `expert_cache=auto` with
  `expert_stream=true` reads the cold experts of each layer from the checkpoint before the layer runs, and `expert_stream_host_gib` keeps that many GiB
  of them per rank in pinned RAM instead (read once at load); with the expert tier's staging each layer's cold experts are copied to the device once
  per prefill window, and the next layer's are read from the checkpoint while a layer runs. A teacher pass over a text file is
  `test_ppl --model glm5_next --chunk 512 --topk-out <file>` with `glm_logit_rows` at the prefill window's rows (8192 for `prefill_window=16` chunks of 512), each
  window scoring every row; with `--seq` rows shorter than a window, one window holds several rows, one slot each (`max_slots` of them). It can also be the source of the MTP head (`nvfp4_dense_src=<GLM-5.3-Flash>`: layer 45 quantized to NVFP4 from FP8 at load).
* `tp` 1, 2 or 4; on two GPUs the experts need the expert tier (`expert_cache` below 1) to fit. At `tp` 4 with every expert resident, the experts
  are sharded over the ranks. `shared_slice` (on) cuts the shared expert and the dense MLP of layers 0-2 over the ranks.
* Speculation: the DFlash2 drafter (`draft_model_dir`, taps of layers 5, 14, 24, 33 and 42) keeps a drafter context and tap ring per slot, so
  concurrent requests all speculate; with two slots a round drafts each slot's block in turn and verifies them in one target forward (`spec_joint`, on by
  default for DFlash2). The verify records each KDA row's inputs and replays the accepted rows instead of keeping a state snapshot per row.
  `speculator=mtp` uses layer 45 of the NVFP4 checkpoint or the original FP8 one (a plain MLA + MoE layer, quantized to NVFP4 at load), so it needs
  `nvfp4_dense_src` (`configs/glm_mtp.json`); with `spec_suffix` a round verifies a suffix match over the request's own text instead of the head's chain.
* Images: GLM's own tower (24 RMSNorm / SwiGLU blocks, 2x2 conv merge). With the DFlash2 drafter the arena has too little left for it, so image
  input stays off unless memory is freed, for example `--set expert_cache=0.95`.
* Memory and context (defaults `max_seq` 12288, `max_slots` 1):
  * `expert_cache` (0 to 1, or `auto`): the resident share of the routed experts; the cold ones stay in pinned host memory and are read over PCIe.
    `auto` takes the largest share the arena holds after the other weights, the drafter (its file size) and a 0.25 GiB margin; the log prints the plan.
    Each point of `expert_cache` costs about 0.26 GiB per GPU.
  * `kv_format=nvfp4` stores the single latent tensor in 4 bits (`fp8`, `k8v4` = 8 bits; against FP16 the 4-bit cache moves the perplexity by +0.008 NLL
    on 3000 tokens, FP8 by +0.001); `kv_pool=<tokens>` lets the slots share one pool of KV pages
    instead of owning `max_seq` each: a request starts once the pool holds its prompt, and one that grows past it ends with finish reason `length`.
  * `kv_shard=true` (with `kv_pool`, 2 or 4 GPUs) spreads the latent cache over the GPUs: page p of a slot lives on GPU p % tp, which stores its
    rows, and the attention reads the rows it selects from the GPU that holds them over NVLink, so each GPU keeps a quarter of the latent pages at
    tp 4. The indexer's pooled keys follow the same pages: a GPU builds and scores the pools of its own pages, keeps its best 512 by (score,
    lowest pool), and after a barrier every GPU merges the four lists into the selection the unsharded indexer makes (the global best 512
    are each among their own GPU's best). A GPU reads the owner's copy of a row, which can differ from its own in the last bit (each GPU
    sums the all-reduce starting from its own part), so the results match the unsharded cache up to that rounding.
  * with the expert tier, `expert_stage` copies a layer's cold experts once into device scratch for prefill chunks, `route_resident_bias` adds a bias to the
    router score of resident experts (near-ties go to the device; it changes the routing), `cpu_experts` hands part of the cold experts to the CPU.
  * `prefill_window` (default 8) prefills several chunks layer by layer so the routed experts run once over all their rows; `moe_heavy_rows=256` runs an
    expert that at least 256 rows of a chunk or window route to as FP16 dequantization + a tensor-core GEMM.
  * `mem_log=true` prints the per-GPU breakdown at startup as `[mem]` lines.
* 256K context with the drafter on four 32 GB GPUs: `configs/glm_256k.json` (four slots sharing a 262144-token pool, `kv_format=nvfp4`, `expert_cache=auto`,
  NVFP4 dense groups, images off). Two slots with every expert resident:

```
build/volta_server --model-dir <GLM-5.3-Flash-exl3-3.05bpw> --speculative --draft-model-dir <GLM DFlash2> \
	--set max_seq=262144 --set max_slots=2 --set kv_format=nvfp4 --set kv_pool=262144 --set expert_cache=1.0 --set max_image_tokens=0
```

## Qwen3.8-Flash-Next (`qwen4_exp`)

| Item | Value |
|---|---|
| Layers | 48: every 4th is full attention (QSA), the rest GDN linear attention |
| Hidden | 2560, vocabulary 248320, untied head, no final norm (a final mixer collapses the streams) |
| Full attention | 24 Q heads, 2 KV heads, head_dim 256, partial rotary 0.25, interleaved M-RoPE (11/11/10), q / k RMSNorm, interleaved output gate |
| GDN | 16 K heads x 128, 48 V heads x 128, conv kernel 4, sigmoid-gated output RMSNorm |
| QSA indexer | 4 heads x 128 and one key per token, scored over pools of 4 tokens; past 2048 tokens of context a row attends the best 512 pools plus the incomplete tail |
| MoE | 512 experts, top 10, intermediate 640, plus a shared expert (640) with a sigmoid gate |
| Residual | 4 FP32 streams with a gated residual (low rank 320, per channel) instead of input / post norms |
| PLE | an n-gram embedding layer ahead of decoder layer 1: hashed 3-grams, 16 heads x 160, a table of 128 shards x 2.5M rows, causal dilated conv, grouped RMSNorms, key / value projections into the streams |
| MTP | one full-attention block with its own gated residuals, MoE and mixer, the shared head |
| Vision | the tower of the Qwen3.5 family |

The gated residual, per site (it replaces norm + residual add), with `x` the FP32 streams `[R, 4, 2560]` and FP16 weights (`hc_norm` `[10240]`,
`down` `[320, 10240]`, `up` `[10240, 320]`, `inject` `[4, 10240]`):

```
normed = x * rsqrt(mean(x^2, -1) + eps) * (1 + hc_norm).reshape(4, 2560)
flat   = normed.reshape(R, 10240)
t      = silu(flat @ down^T / 4)                                  # [R, 320]
mixed  = mean_over_streams(sigmoid(t @ up^T).reshape(R, 4, 2560) * normed)   # the sublayer input, FP16
post   = 2 * sigmoid(flat @ inject^T / 4)                         # [R, 4]
after the sublayer: x[:, s, :] += post[:, s] * sublayer_out
```

The final mixer is the same without `post` and gives the head's input.

* Parallelism: `tp` 1, 2 or 4. GDN, attention and the dense projections are split over the ranks, the routed experts are placed whole (512 / `tp` per
  rank), the shared expert runs on rank 0 beside the routed ones, and the vocabulary is sharded for the head. The EXL3 conversion keeps 61 GB of
  weights on the GPUs (15.3 GiB per GPU at `tp` 4); `expert_cache` below 1 keeps the cold experts in pinned host memory (`cpu_experts` hands part of them
  to the CPU, `configs/flash_next_cpu_experts.json`).
* PLE: the n-gram table stays in host memory, memory-mapped (the conversion's `ngram_embedding.safetensors`, 39 GB of trellis rows; the NVFP4
  checkpoint's FP8 shards, 51 GB; a GGUF conversion's block-quantized rows). The rows of a forward are decoded on the host, every rank a share of the
  rows (for prefill chunks while the GPUs run the layers before the PLE layer). The first requests after a cold start fault rows in from disk.
* Weights:
  * EXL3: `Swift-Qwen3.8-Flash-Next-exl3-4.00bpw` (stock tile order) or a Volta-order conversion of the base model; experts 2, 3 or 4 bits, the
    attention and GDN projections 7 / 8 bits ([QUANTIZATION.md](QUANTIZATION.md#bit-widths-of-the-exl3-checkpoints)).
  * NVFP4, with the EXL3 directory as the model (`configs/flash_next_nvfp4.json`): `nvfp4_experts=<NVFP4 dir>` takes the routed experts from the
    modelopt checkpoint (`Qwen3.8-Flash-Next-NVFP4`), `nvfp4_dense=<groups>` quantizes dense groups from its BF16 tensors at load (`gdn`, `attn`, `idx`,
    `head`, `shared_gate`, `shared_up`, `shared_down`, `shared`, `all`; `gdn,attn,idx,head,shared_gate,shared_up` is the recommended set, `shared_down`
    costs perplexity), `nvfp4_dense_dir` loads calibrated tensors instead of round-to-nearest, `mtp_nvfp4=true` reads the MTP head from the NVFP4 checkpoint
    (its FP8 experts and BF16 dense tensors quantized to NVFP4 at load).
  * NVFP4 only: `model_dir` and `nvfp4_experts` both point at the NVFP4 checkpoint, with `nvfp4_dense=all` and `mtp_nvfp4=true`
    (`configs/flash_next_nvfp4_only.json`); the n-gram table is read in place from its FP8 shards.
  * GGUF: llama.cpp's `qwen4exp` architecture. Its MTP block is not read; `mtp_head_dir=<NVFP4 dir>` takes the head from the NVFP4 checkpoint.
  * calibrated NVFP4 dense tensors come from `tests/tools/nvfp4_ldlq.py --model flash|glm` over the Hessians that
    `ENGINE_HESSIAN_DIR=<dir> test_ppl --model qwen4_exp|glm5_next --calibrate --file <text>` writes (GLM: `ENGINE_HESSIAN_LAYERS=lo-hi`, a third of the layers per pass).
* Speculation: `--speculative --set speculator=mtp` drafts with the checkpoint's head (state per slot; `mtp_depth`, default 7, is the most draft
  tokens per round). The DFlash2 drafter (`draft_model_dir` = `Qwen3.8-Flash-Next-DFlash2`, `configs/flash_next_dflash2.json`) drafts a block per round
  from the outputs of layers 5, 15, 25, 35 and 45 (each the mean of the four streams), kept in a per-slot ring of the last 2048 positions that
  prefix-cache snapshots carry; it is tensor-parallel like the target. Greedy output equals plain decoding with either.
* Context: decode attends the QSA selection through a sparse kernel, so it stays nearly flat in context; the 256K window with 2 slots fits.
* Prefix snapshots hold the GDN state, conv history, the n-gram conv ring and token context, KV rows and pooled index keys.
* Tokenizer: the split regex differs from Qwen3.8-27B's (marks are not letters), handled in `tokenizer.cpp`.
* Known limit: when the model writes about its own chat template inside a thinking block it can emit the real `<|im_end|>` token, which ends the
  generation with `end_turn` and no visible output.

```
build/volta_server --config configs/flash_next_nvfp4.json      # EXL3 trunk, NVFP4 experts and dense groups, MTP, 256K x 4 slots
build/volta_server --model-dir <Flash-Next EXL3> --speculative --set speculator=mtp
```

## Gemma 4 31B (`gemma4`)

* The text model of `google/gemma-4-31B-it`: 60 layers, every 6th full attention (10 layers: 4 KV heads, head_dim 512, K = V, proportional RoPE
  over 64 pairs), the other 50 sliding-window attention (1024 positions, 16 KV heads, head_dim 256); 32 Q heads, hidden 5376, MLP 21504,
  sandwich norms, an FP32 residual stream, a tied head with a final logit softcap of 30, vocabulary 262144.
* Weights: EXL3 (exllamav3's conversions), NVFP4 (modelopt: the MLP in NVFP4, attention and the tied head BF16; `nvfp4_dense=attn,head` or `all`
  quantizes those at load), GGUF (llama.cpp's `gemma4` architecture, e.g. the QAT Q4_0 release).
* Caches: the sliding-window layers keep a ring of 1536 positions per slot (the window and one 512-row prefill chunk), the full-attention layers every
  position; prefix-cache snapshots keep the sliding layers' last window.
* Speculation: the DFlash drafter `z-lab/gemma-4-31B-it-DFlash` (a DFlash v1 `DFlashDraftModel`, also as an EXL3 conversion) through
  `draft_model_dir`: a 16-row block of which the first 8 rows are verified, taps of layers 1, 12, 23, 35, 46 and 57, one full-attention layer over every
  position (the drafter takes each prefill chunk's taps as they are written). `speculator=suffix` with `spec_suffix` needs no model.
* Images: the checkpoint's vision tower (ViT blocks with a 2-D rotary, 3 x 3 pooling into at most 280 soft tokens, `max_image_tokens` caps that)
  from the BF16 `model.vision_tower.*` tensors, on CUDA, CPU and Vulkan. An image's tokens attend to the whole image on the sliding-window layers and
  causally on the full-attention ones, and a prefill forward never splits an image. A GGUF directory's tower is its `mmproj` file (llama.cpp's `clip`
  architecture with the `gemma4v` projector).
* Prompts: the chat template's `<|channel>thought ... <channel|>` reasoning and `<|tool_call>` calls are parsed ([SERVER.md](SERVER.md#output_parsercpp)).
* Defaults: `max_seq` 32768, `max_slots` 8.

```
build/volta_server --model-dir <gemma-4-31B-it-exl3> --speculative --draft-model-dir <gemma-4-31B-it-DFlash>
```

## DeepSeek-V4.1-Flash (`deepseek_v41`)

* 40 layers on four hyper-connection streams: every layer attends a 128-token sliding window, and from layer 2 on also compressed KV entries
  chosen by sparse indexers (layers 2, 8, 14 and 20 write the compressed KV, layers 20-39 read layer 20's); 64 heads, hidden 5120, a routed MoE of
  384 experts (top 6, intermediate 2304) plus a shared expert; Engram n-gram lookups at layers 1 and 14; vocabulary 129280. CUDA only; `tp` 1, 2 or 4.
* Weights:
  * EXL3: exllamav3's own conversions (such as `turboderp/DeepSeek-V4.1-Flash-1017-exl3`; its 1.68 bpw experts are 1.5 and 2 bpw) or those of the
    sudoingX / 0xSero layout. exllamav3's releases keep the Engram tables in `ngram_embedding.safetensors`, which the loader opens next to the shards;
    for the other repositories `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors` of `deepseek-ai/DeepSeek-V4.1-Flash-1017`
    go into the model directory.
  * Packs that keep the original FP8 attention, shared-expert and indexer weights and quantize only the routed experts (each projection of an expert at
    its own width, 1 to 4 bits) load as they are: the FP8 tensors stay FP8 in device memory (`fp8_resident`), and a layer's experts run as one launch with a
    per-expert width table.
  * GGUF: llama.cpp's `deepseek41` conversions (such as `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`): a build's parts and the shared Engram file
    (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, linked as the build's next part) in one directory with the original `config.json` without its
    `quantization_config`, the tokenizer files and `engram_hash.bin`. The Engram rows are IQ4_NL blocks, decoded on the host.
  * either way, `python tests/tools/dsv41/engram_hash_tables.py <model dir>` (a venv with `torch`, `transformers` and `sympy`) writes the hashing tables
    the engine reads, `engram_hash.bin`, once per directory.
* Routed experts: they take what the arena holds after the drafter (each rank sizes its share from its own remainder); the rest stay in pinned host
  memory and are read in place. With `expert_heat_file` the route counts are saved at exit and the next start deals the experts to the ranks by heat and
  keeps the hottest on the device; every `expert_adapt_every` decode steps the most used host-memory experts swap with the least used resident ones of the
  same width. `expert_shard` (auto: on for GGUF experts) gives every rank a slice of every expert instead of whole experts.
* Prefill: `prefill_window` (-1: as many chunks, up to 4, as the arena holds once every routed expert is resident, else one chunk at a time; up to 16)
  runs several chunks layer by layer; `prefill_truncate` (-1, exact) runs layers 20-39 only for the last 2541 rows of a prefill call (0 runs every row
  through every layer). The indexer scores prefill rows on tensor cores, and the Engram rows of the next prefill window are read ahead on host threads.
* Caches: the checkpoint's own grids (window rows E4M3, compressed entries and index keys E2M1, each with its block scales), a full copy on every rank:
  about 0.9 KB per token of a slot, plus 0.75 KB per token of `max_seq` for the rotary tables and the indexer's scores. `kv_format` is ignored.
  Prefix-cache snapshots hold the compressed KV and index keys of their positions plus the last 128 window rows of each layer.
* Speculation: `speculator=suffix` with `spec_suffix` (no model), or `speculator=dspark` with the checkpoint's own block drafter (three small blocks of
  window attention and a 128-expert MoE, entered through the trunk's stream means at layers 37-39, with a bigram head and a confidence head):
  `tests/tools/dsv41/quantize_dspark.py --model <checkpoint> --out <dir>` quantizes it to EXL3 (experts at 3 bits by default) plus the official tensors it
  needs, and `draft_model_dir` points at that directory. A round drafts 5 tokens, keeps those whose confidence passes `dspark_conf`, and verifies them in
  one forward. Snapshots carry the trunk taps of the last 128 positions, so a restored slot keeps speculating.
* Images: the checkpoint's ViT and aligner on rank 0 (`max_image_tokens`, 9 patches per token, the span adding a start, an end and a newline token per row);
  image tokens route with the checkpoint's `bias_vl` and stay out of the Engram n-grams. The tower's memory comes out of rank 0's resident experts.
* Prompts: the checkpoint's `encoding/encoding.py`, built in (no chat template): thinking and reasoning effort (low 50, medium 60, high 75, max 100),
  tools as DSML tool calls.
* Defaults: `max_seq` 16384, `max_slots` 1. `configs/deepseek_v41.json` serves the 1.68 bpw EXL3 pack with the DSpark drafter (`dspark_conf` 0.7)
  and 576 image tokens:

```
build/volta_server --config configs/deepseek_v41.json
```
