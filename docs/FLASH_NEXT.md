# Qwen3.8-Flash-Next (`qwen4_exp`) and the NVFP4 paths

Checkpoint: `Swift-Qwen3.8-Flash-Next-exl3-4.00bpw` (architecture `Qwen4ExpForConditionalGeneration`). The parity oracle is our `exllamav3` fork
(`architecture/qwen4_exp.py`, `modules/qsa_indexer.py`, `modules/ple.py`, `modules/hyperconnections.py::GatedResidual`, `architecture/qwen4_exp_mtp.py`), which runs on SM70 in
a single process; it marks the architecture `supports_tp: False`, so the multi-GPU split is the engine's own. Options referred to below are described in [CONFIG.md](CONFIG.md);
usage examples are in [GETTING_STARTED.md](GETTING_STARTED.md) and `configs/`.

## Architecture

| Item | Value |
|---|---|
| Layers | 48: every 4th is full attention (QSA), the rest GDN linear attention |
| Hidden | 2560, vocab 248320, context 262144, untied head, **no final norm** |
| Full attention | 24 Q heads, 2 KV heads, head_dim 256, partial rotary 0.25, mrope interleaved (11/11/10), q/k RMSNorm (+1), interleaved output gate |
| GDN | 16 K heads x 128, 48 V heads x 128, conv kernel 4, **sigmoid-gated** output RMSNorm (Qwen3.8-27B uses silu) |
| QSA indexer | 4 heads, 1 KV head, head_dim 128, compress ratio 4, token budget 2048 (a DSA-like block selector) |
| MoE | 512 experts, top 10, intermediate 640, plus a shared expert (640) with a sigmoid gate; routed weights trellis-quantized (3 or 4 bit per layer: layers 5, 8, 45, 46 are 3-bit) |
| Residual | 4 fp32 streams with a **gated residual** (low rank, per channel) instead of input / post norms |
| PLE | one layer (id 2): hashed 3-gram embedding, 16 heads x 160, a trellis-quantized table of 128 shards x 2.5M rows x 61 int16 (39 GB, host resident), causal dilated conv, grouped RMSNorms, key / value projections into the streams |
| MTP | 1 hybrid layer (full attention), no dedicated embeddings |
| Vision | the tower of the Qwen3.5 family |
| Quantization | attention and GDN projections 7 / 8 bpw trellis (lm_head 6), routed and shared experts 3 / 4 bpw; router, `shared_expert_gate` and GDN `in_proj_a/b` fp16; `conv1d`, `A_log`, `dt_bias`, norms bf16 |
| Size | trunk ~63 GB plus the 39 GB n-gram table; host RAM is 128 GB |

**Gated residual** (per site, replaces norm + residual add), `x` = fp32 streams `[R, 4, 2560]`, weights fp16 (`hc_norm` `[10240]` includes +1 at load,
`input_mix_weight_down` `[320, 10240]`, `input_mix_weight_up` `[10240, 320]`, `block_inject_weight` `[4, 10240]`):

```
normed = x * rsqrt(mean(x^2, -1) + eps) * hc_norm.reshape(4, 2560)
flat   = normed.reshape(R, 10240)
t      = silu(flat @ down^T / 4)                     # [R, 320]
w      = sigmoid(t @ up^T)                           # [R, 10240]
mixed  = mean_over_streams(w.reshape(R,4,2560) * normed)     # sublayer input, fp16
post   = 2 * sigmoid(flat @ inject^T / 4)            # [R, 4]
apply  : x[:, s, :] += post[:, s] * sublayer_out      # fp32 streams
```

The final `hyper_connection_mixer` is the same without `post` and returns `mixed` as the head input.

## How it runs

* **Parallelism**: TP = 4 for GDN, attention and the dense projections; expert parallelism for the 512 experts (128 whole experts per rank, so the I = 640 shape never needs a
  Hadamard-span split); the shared expert on rank 0 on a side stream beside the router and the fused MoE kernels; the vocabulary sharded for the head; the PLE rows decoded on
  the host by all ranks, a quarter each into a shared buffer, overlapped with the GPU. A pair of GPUs per slot (TP = 2 x 2 replicas) does not fit: 61 GB of GPU-resident
  weights is 15.3 GiB per GPU at TP = 4 (arena 19.3 GiB with the default KV and scratch). Batch 2 on TP = 4 shares every launch between the slots. What remains is latency
  (two all-reduces per layer, ~25 us per 8-bit projection launch), not layout.
* **Decode**: CUDA graphs; a fused cooperative MoE kernel (`exl3_moe_coop`, 3 and 4 bpw, up to 10 routes); the dp4a int8 GEMV serves the 7 / 8-bit projections at M = 1; gated-residual
  decode kernels stream the 6.5 MB gate / up weights with all loads in flight (rows <= 8); the gated add after a sublayer is deferred into the next site's norm pass.
* **Prefill**: the gated-residual projections run as cuBLAS GEMM pairs on fp16 hi / lo splits of the fp32 operands at a fixed row capacity (a row's result is bit-identical whatever
  its position or chunk size); QSA sparse attention for prefill rows runs on a tensor-core kernel (keys gathered through the row's top-k list); the PLE row decode overlaps the layers
  queued before it. Interleaved prefill (`prefill_chunk`, default 256): while other requests decode, a new prompt is fed in pieces and the decode steps run between them; snapshot
  buffers for the prompt's planned prefix checkpoints are pinned on a helper thread when its prefill starts, so a checkpoint store does not stall the streams.
* **Context**: QSA block selection past 2048 tokens of context (top-512 + tail, pooled keys, the GLM selector reused); decode is nearly flat in context (a warp-per-key sparse
  attention kernel for decode rows); the 256K window with 2 slots fits (24.3 of 29.8 GiB per GPU).
* **Speculation**: `--speculative --set speculator=mtp` uses the checkpoint's own head (it follows llama.cpp's qwen4exp MTP graph: per-stream norm of the pre-collapse streams, `fc_hidden`
  on every stream plus the broadcast `fc_embedding` branch, one block with its own KV / QSA layer, its own mixer, the shared head). Greedy output equals plain decoding. The head keeps its state per
  slot; with several slots decoding the generator chooses per step between one slot's round and a plain step of all of them by the measured rates (re-measured every 64 steps).
  `spec_joint` (off by default) runs one draft chain and one verify forward over all speculating slots: about even with the controller at two requests, 25% ahead at four on code prompts.
* **Serving**: prefix snapshots (GDN state, conv history, n-gram conv ring and token context, KV rows, pooled index keys), the OpenAI / Anthropic APIs, images through the checkpoint's tower.
  Message content is tokenized as plain text: the template's role markers are real special tokens, but the same strings inside message content are escaped before the template
  renders and encoded without special-token parsing (`chat.cpp`), so quoted source cannot end or open a turn. `ENGINE_LOG_LEVEL=debug` logs the token that stopped each generation.
* **Tokenizer**: the split regex differs from Qwen3.8-27B's (marks are not letters); handled in `tokenizer.cpp`.

## NVFP4

**Flash-Next.** The modelopt checkpoint (`Qwen3.8-Flash-Next-NVFP4`, `MIXED_PRECISION`) has NVFP4 only for the routed experts (`weight` U8 packed, `weight_scale` E4M3 per 16 values,
`weight_scale_2`, `input_scale`; the MTP experts are FP8, the n-gram embedding FP8); everything else is BF16.

* `nvfp4_experts=<dir>` takes the routed experts from it with the EXL3 directory as the model (the MTP experts stay EXL3; activations stay FP16, `input_scale` is unused). Fused kernels
  (`src/kernels/cuda/nvfp4/moe_kernels.cu`): two launches per layer for one row; rows > 1 go through a dispatch pass that groups the routes by expert and multiplies up to 8 rows per pass of the
  weights. Per layer, one row: 19 us for one local route, 26 us for four, 60 us for ten (the EXL3 path: 55 / 58 / 163 us). Experts take 2.76 MB each against 2.46 MB.
* `nvfp4_dense=<groups>` quantizes dense groups at load (`gdn`, `attn`, `idx`, `head`, `shared_gate`, `shared_up`, `shared_down`, `shared`, `all`) from the BF16 tensors of the NVFP4 checkpoint:
  per 16 values the E4M3 scale nearest to amax / 6 and its +-2 code neighbours are tried, one global scale per tensor. `nvfp4_dense_dir=<dir>` loads calibrated tensors instead.
* **Calibration**: `ENGINE_HESSIAN_DIR=<dir> test_ppl --flash --calibrate --file <text> --tokens N` (with `ENGINE_MAX_SEQ` above N) accumulates x^T x of every dense projection input and writes one
  file per site; `tests/tools/nvfp4_ldlq.py --model flash` then runs block LDLQ (damping 2.5% of the mean diagonal, Cholesky, 16-row blocks from the last to the first with the propagated error, the
  NVFP4 tile quantizer in place of the trellis one, no Hadamard rotation; row-parallel inputs use the per-rank Hessians as a block diagonal) and writes a `*-NVFP4-dense` directory (8 layers
  per file: the loader opens at most 128 shard files). Proxy error falls 3 to 10 times against round-to-nearest.
* **MTP head** (`mtp_nvfp4=true`, needs `nvfp4_experts`): the head is read from the NVFP4 checkpoint instead of the model directory. There its 512 experts are FP8 E4M3 with 128 x 128 block scales (`model-fp8-mtp-ple.safetensors`) and everything else is BF16; the experts are dequantized and quantized to NVFP4 at load (round to nearest), the dense tensors go through the same NVFP4 quantizer as the trunk's dense groups. The base model's head accepts fewer drafts than the head of the Swift conversion when the rest of the trunk is the conversion's (natural 8K prompts: acceptance 0.39 to 0.50 against 0.47 to 0.73 at depth 7, 0.39 to 0.59 against 0.51 to 0.84 at depth 4; 92 to 134 tok/s against 123 to 213 at depth 7, 105 to 151 against 127 to 199 at depth 4), so leave it off with a converted trunk. The n-gram (PLE) table is still read from the model directory.
* **Recommended**: `nvfp4_experts` + `nvfp4_dense=gdn,attn,idx,head,shared_gate,shared_up` + the calibrated directory. `shared_down` stays on the 8-bit trellis: it costs +0.016 perplexity in every
  variant tried (round-to-nearest, calibrated, even the BF16 original as dense FP16), the EXL3 tensor simply scores better.

| Flash-Next, qbench perplexity (BF16 1.3825) | PPL | | M=1 | 8-row verify | 2 slots |
|---|---|---|---|---|---|
| EXL3 4.00 bpw | 1.4247 | | 18.3 ms | 29.5 ms | 21.2 ms |
| + NVFP4 experts | 1.4282 | | 16.6 | 25.1 | 19.5 |
| + recommended dense groups (calibrated) | 1.4530 | | 14.2 | 18.6 | 13.2 (152 tok/s aggregate) |
| + all dense groups (calibrated) | 1.4815 | | 11.4 | 17.2 | 12.4 |

(`bench_decode --flash`, 150 W cap, single runs: directional.) Round-to-nearest per group on top of the NVFP4-experts hybrid: gdn 1.4394, shared 1.4479, attn 1.4714, idx 1.4283, head 1.4434, all 1.5322;
calibrated: gdn 1.4382, attn 1.4341, head 1.4348, all 1.4815.

**GLM-5.3-Flash.** `nvfp4_dense_src=<GLM-5.3-Flash-NVFP4>` with `nvfp4_dense=kda,mla,idx,head,dense,shared` (groups: `kda` q / k / v stacked per rank + o_proj, `mla` q_a / q_b / kv_a / o_proj, `idx`,
`shared` or its three parts, `dense` the native NVFP4 MLP layers 0-2, `head`); the routed experts stay EXL3 3.05 bpw. Perplexity (`test_ppl --glm --chunk 8 --file <text> --tokens 12000`, held-out text;
run with `ENGINE_EXPERT_CACHE=0.9 ENGINE_VRAM_RESERVE_MIB=3500 ENGINE_MAX_IMAGE_TOKENS=0` to leave room for the dequantization workspace): EXL3 1.798; round-to-nearest, all groups 1.812; calibrated
(`nvfp4_ldlq.py --model glm`, `ENGINE_HESSIAN_LAYERS=lo-hi` in three passes) 1.804. `bench_decode --glm`: M=1 20.1 -> 16.0 ms, 8-row verify 48.2 -> 34.4 to 36.0 ms. NVFP4 *experts* for GLM are not built:
they are ~172 GB against ~119 GiB of arena, so about 47% could be resident (hit rate ~0.69; routing has little locality: the busiest expert takes 4.9% of a layer's routes against 1.4% uniform),
which means ~27 PCIe misses per token per rank, ~30 ms on top of a 16 ms token.

## Expert tier

`expert_cache` < 1 (GLM, Flash-Next; NVFP4 or EXL3 experts) keeps the codes of the cold experts in pinned host memory, read over PCIe by the expert kernels; scales stay on the device. Results are
identical to the digit (perplexity 1.4282 at 0.5 and 0.25) and the arena per GPU drops (21.5 -> 14.5 GiB at 0.5), but Flash-Next's routing has little locality on varied text (window hit rate
0.56 to 0.66 at 0.5, 20 tok/s served), so the tier is a capacity valve (0.9 to 0.95 for headroom), not a speed trade. The monitor's per-expert counters and the tier both read and clear
`MoELayerWeights::heat`, so the monitor undercounts with the tier on.

## Validation

* **Oracle**: `tests/tools/ref_qwen4_exp_dump.py` writes per-module and block-internal activations and the logits of a 42-token prompt to `results/flash_next_ref/` (untracked; needs the fork's GDN
  b/a fix). Layers 0 to 5 match at bf16 level (rel 4-9e-3, router flips aside); drift grows smoothly to ~0.13 by layer 38 and is reproduced by injecting bf16-sized noise (5e-3) into our own
  residual after every layer (`ENGINE_DEBUG_PERTURB_LAYERS`), so it is the reference's bf16 intermediates, not a localized bug. `ENGINE_DEBUG_TEACHER_STREAMS` (each layer starts from the
  reference's streams) gives 0.2 to 7 permille per layer, bursty where the router flips. Other debug switches: `ENGINE_DEBUG_ROUTER_FP16`, `ENGINE_DEBUG_TRACE_LAYERS`.
* **Perplexity** (`test_ppl --flash --qbench <checkpoint>/qbench_prompts.md`: the 9 traced rows of the author's qbench set, ~20K scored tokens): 1.4247 for EXL3 4.00 bpw (+3.0% over the
  model card's BF16 1.3825). Prefill changes were checked the same way (6000 tokens of docs: 3.835 against 3.829 before, inside the sample noise).
* **Needles**: 12K, 32K, 100K and 250K pass; the 250K needle (250,054 tokens): prefill 848 tok/s, decode 42.2 tok/s. A Claude Code session (`claude -p`, MTP, 2 slots, prefix cache) read server
  sources to 64K tokens, edited a file with the Edit tool and resumed the cached prefix on every turn.
* **Suites**: the Flash API suite with MTP (44 checks), `test_mtp --flash` (greedy output equals plain decoding on long prompts; on short ones it can flip at a late near-tie), prefix restore,
  `test_gr_decode` / `test_gr_tc` (gated-residual parity: cosine 1.0000000), `test_sparse_attn` (FP64 softmax: max abs error 1.2e-4), `bench_nvfp4_moe`, `bench_moe_routes`, `bench_nvfp4_shapes`,
  `test_nvfp4_quantize` (round-to-nearest NVFP4 against the EXL3 tensor of the same matrix).

## Reference numbers

250 W cap, `bench_decode` and `test_mtp` (baselines `results/baseline_cap150.txt`, `results/baseline_cap250.txt`, local files; the 150 W numbers elsewhere are not comparable):

| | Qwen3.8-27B | GLM-5.3-Flash | Flash-Next |
|---|---|---|---|
| M=1 decode | 15.91 ms (62.8 tok/s) | 18.98 ms (52.7 tok/s) | 19.81 ms (50.5 tok/s) |
| M=8 verify | 24.88 ms | 41.94 ms | 30.78 ms |
| 2 slots, per step | | | 22.52 ms (88.8 tok/s aggregate) |
| Prefill, 512-row chunks | 2492 tok/s (2K) | 763 tok/s (2K) | 1322 tok/s (8K) |
| MTP, 34-token prompt | plain 62.8, MTP 185.4 tok/s (82% accepted) | | plain 49.4, MTP 173.3 tok/s (85%) |
| MTP, 15K-token prompt | plain 58.8, MTP 132.2 tok/s (63%) | | plain 44.3, MTP 109.3 tok/s (60%) |

Real serving decode runs at 98% GPU utilization (the `bench_decode` trace shows 44%: a benchmark-loop artifact; nvprof traces also inflate small kernels about 2x against graph replay, so
rank optimization levers with graph-replay benches). Where a Flash-Next M=1 token goes (per GPU): 7 / 8-bit projections ~7 ms (latency-bound, 22 to 36 us per launch), fused MoE ~3 ms, gated
residual ~3 ms, router ~2.6 ms, all-reduce ~1 ms, sparse attention 0.4 ms at short context (a warp-per-key decode kernel keeps M = 1 decode within about 2 ms of that from 65K to 250K tokens). The verify forward is 84% of an MTP round and costs
~1.6 ms per extra row; a round is ~37 ms for ~4 tokens, and the controller's survival threshold (0.3) is about right: deeper drafting does not pay. Speed figures here were taken at a
power cap in single runs, so they are directional; a benchmark matrix with repeats at a stated power limit is still to do.

## Known issues and limits

* **`<|im_end|>` inside thinking** (open): when the model writes about its own chat template inside a thinking block it can emit the real `<|im_end|>` token (id 248046) where it means the
  text, which ends the generation with `end_turn` and the client sees no visible output. Candidate fixes, none applied: mask the EOS logits while a thinking block is open and no tool call has
  started (GPU sampler and MTP verify), or treat the token as literal text there. The model sometimes goes from thinking to a tool call without `</think>`, so "reasoning open" alone does not
  mean a stop is wrong.
* **Prefill gaps**: `prefill_chunk` is global. 256 meets a ~0.5 s stream-gap target on Flash-Next (0.49 s with a 19K-token prompt arriving mid-stream; 128 gives 0.33 s at a slower prompt); on GLM a
  256-row piece costs ~0.72 s, so the stream sees a steady 0.77 s gap (128: 0.53 s). Models with a DFlash2 drafter keep the whole-prompt prefill.
* **Batch composition**: batched rows round differently from single rows, so concurrent greedy outputs can diverge from sequential ones; any numerics change moves the logits by a few tenths
  through router flips.
* **`bench_decode` with `BENCH_CTX`** needs `ENGINE_MAX_SEQ` above it (at exactly `max_seq` it writes past the cache). `test_ppl --glm` cannot score chunks above 8 rows (GLM's logits buffer holds a
  decode batch).
* **GLM layer-0 Hessian**: the captured attention-input Hessian of layer 0 has a negative diagonal, which x^T x cannot have (the hook may read a buffer that is not yet the normed input); the
  solver falls back to round-to-nearest for those matrices.
* **Open levers**: an M = 8 variant of the int8 dp4a GEMV for the 7 / 8-bit projections (est. 1.8 ms per round), the fused MoE at 8 rows (3x off bandwidth), a score kernel that tiles the 8 rows of
  a slot for the QSA pooled keys (about 1 ms at 128K, 2 to 3 ms at 250K), fusing the four gated-residual kernels per sublayer, overlapping the next prefill chunk's PLE rows with the current chunk.
