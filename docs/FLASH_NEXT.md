# Qwen3.8-Flash-Next (`qwen4_exp`) port plan

Checkpoint: `Swift-Qwen3.8-Flash-Next-exl3-4.00bpw` (arch `Qwen4ExpForConditionalGeneration`). Reference: `~/Programs/exllamav3`
(`architecture/qwen4_exp.py`, `modules/qsa_indexer.py`, `modules/ple.py`, `modules/hyperconnections.py::GatedResidual`,
`architecture/qwen4_exp_mtp.py`). The reference runs on SM70 (single process, autosplit over 4 GPUs, loads in ~66 s) and answers
sensibly, so it serves as the parity oracle. It marks the architecture `supports_tp: False`; the multi-GPU split is ours to build.

## Architecture facts

| Item | Value |
|---|---|
| Layers | 48: every 4th is full attention (QSA), the rest GDN linear attention |
| Hidden | 2560, vocab 248320, context 262144, untied head, **no final norm** |
| Full attention | 24 Q heads, 2 KV heads, head_dim 256, partial rotary 0.25, mrope interleaved (11/11/10), q/k RMSNorm (+1), interleaved output gate |
| GDN | 16 K heads x 128, 48 V heads x 128, conv kernel 4, **sigmoid-gated** output RMSNorm (Qwen3.8-27B uses silu) |
| QSA indexer | 4 heads, 1 KV head, head_dim 128, compress ratio 4, token budget 2048 (a DSA-like block selector) |
| MoE | 512 experts, top 10, intermediate 640, plus a shared expert (640) with a sigmoid gate; routed weights trellis-quantized |
| Residual | 4 fp32 streams, **gated residual** (low rank, per-channel) instead of input/post norms; see below |
| PLE | one layer (id 2): hashed 3-gram embedding, 16 heads x 160, trellis-quantized table of 128 shards x 2.5M rows x 61 int16 (39 GB), causal dilated conv, grouped RMSNorms, key/value projections into the streams |
| MTP | 1 hybrid layer (full attention), no dedicated embeddings |
| Vision | tower identical to Qwen3.5's (we already run it) |
| Size | trunk ~63 GB + n-gram table 39 GB; host RAM is 128 GB |

Gated residual (per site, replaces norm + residual add): per-stream grouped RMSNorm (weight zero-init, applied as 1 + w), a low-rank
sigmoid gate over the normed stack picks each stream's share of the elementwise mean that feeds the sublayer, and the sublayer
output is added back to the raw streams with a per-stream scalar `2*sigmoid` gate. No Sinkhorn, no combine matrix (cheaper than
GLM's mHC, same stream layout).

## What exists in the engine

| Need | Reuse |
|---|---|
| GDN recurrence, conv, projections | `modules/gdn` (Qwen3.8-27B); add the sigmoid output gate variant |
| Full attention, partial rotary, M-RoPE, FP16/FP8/K8V4/NVFP4 KV | Qwen path (`attention.cu`, `kv_quant.cuh`) |
| Sparse block selection with pooled keys | GLM DSA (`dsa.cu`): different shapes and scoring, same structure |
| Trellis GEMM, MoE dispatch, expert tier | GLM (`exl3_moe_coop`, `moe.cu`); needs 512 experts, top 10, hidden 2560 |
| Four-stream residual | GLM mHC kernels: different mixing, same layout and tap hooks |
| Vision tower, image cache, MTP speculator | C5 work |
| Prefix snapshots (GDN state + KV rows) | Qwen/GLM snapshot code |

## Work breakdown (milestones, each parity-checked against the reference)

1. **Config and loader**: model type `qwen4_exp`, tensor map, trunk placement over 4 GPUs.
2. **Block parity tools**: reference activation dumps for a fixed prompt (embedding, each site, each block), engine-side trace like `ENGINE_DEBUG_TRACE_LAYERS`.
3. **Gated residual + GDN block** (layers 0-2 minus MoE): kernels and a block parity test.
4. **MoE at 512 experts / top 10**: router (assert is 288 today), dispatch, fused kernels, shared expert + gate; parity.
5. **QSA attention layer**: indexer, block selection, sparse GQA kernel; parity against the reference at several lengths.
6. **PLE layer**: host-resident n-gram table (mmap), trellis row decode, hashing, conv state in snapshots.
7. **Whole model**: TP=4 split (GDN heads, KV heads 2 -> replicate or TP2 groups, experts sharded), logits parity, greedy text, needle.
8. **Serving**: chat template, tool calls (Qwen family format), prefix cache, 256K context and batch 2, images, MTP.
9. **Performance and VRAM**: decode graphs, speculation, `kv_format`, SASS/bench baselines.

## Risks

- **No TP reference**: the sharding of 2 KV heads over 4 ranks and of the GDN state needs a design of our own.
- **PLE table**: 16 random 122-byte row reads per token from a 39 GB mapping; fine from page cache, a cold start reads from disk.
- **Router sensitivity**: like GLM, expect expert-set flips between numerically different paths; compare with route replay, not exact logits.
- **Scale of the port**: comparable to the GLM port (new block structure, new sparse attention, wider MoE).

## Reference details (from exllamav3 and the checkpoint)

Gated residual site (`GatedResidual._mix_ref`), `x` = fp32 streams `[R, 4, 2560]`, weights fp16 (`hc_norm` `[10240]` includes +1 at load,
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

The final `hyper_connection_mixer` is the same without `post` and returns `mixed` as the head input. Quantization: attention and GDN
projections are 8 bpw trellis, routed and shared experts 4 bpw; router `mlp.gate.weight` `[512, 2560]` and `shared_expert_gate` `[1, 2560]`
are fp16; GDN `in_proj_a/b` `[48, 2560]` fp16; `conv1d` `[10240, 1, 4]` and `A_log`, `dt_bias`, `norm` are bf16.

Oracle: `tests/tools/ref_qwen4_exp_dump.py` writes per-module and block-internal activations (and the logits) for a 42-token prompt to
`results/flash_next_ref/` (untracked). Block 0 (GDN + MoE) and block 3 (QSA + MoE) internals are included: hc mix output, sublayer output,
hc apply output, for both the attention and MLP sites.

## Status (2026-10-09)

Working, committed: model type and loader, gated residuals, GDN layers, 512-expert MoE (softmax top-10 router, 3- and 4-bit experts, shared expert,
expert-parallel over 4 ranks), the n-gram embedding layer, full attention with the QSA block selection (indexer projection, pooled keys, scoring,
the GLM selector reused for top-512 + tail, sparse GQA attention), final mixer, vocabulary-sharded head, sampling, prefix snapshots (GDN state, conv
history, n-gram conv ring and token context, KV rows, pooled index keys). The server runs the checkpoint (tp 4 only) and passes the OpenAI /
Anthropic API checks except images. Needles: 8K and 32K pass (dense: prefill 311 tok/s, decode 24; QSA at 32K: 279 / 22.6).

Verification: the oracle is the exllamav3 fork run (`tests/tools/ref_qwen4_exp_dump.py`; needs the fork's GDN b/a fix). Layers 0-5 match at bf16 level
(rel 4-9e-3, rows where the router picks another expert set aside), the 4.3K-token QSA prompt gets the reference's top token and the right answer.
Drift against the oracle grows smoothly to ~0.13 by layer 38. It is reproduced by injecting bf16-sized noise (5e-3) into our own residual after every
layer (`ENGINE_DEBUG_PERTURB_LAYERS`), so the cause is a precision-format mismatch (the reference keeps bf16 intermediates), not a localized bug; with
`ENGINE_DEBUG_TEACHER_STREAMS` (each layer starts from the reference's streams) the isolated per-layer error is 0.2-7 permille, bursty where the
router flips. The reference's last block dump is its prefill early exit (the input of block 47). Rounding the router logits to fp16
(`ENGINE_DEBUG_ROUTER_FP16`) did not reduce the drift.

Findings worth keeping: the exllamav3 fork zeroed the GDN b/a gates (a deferred-load bug, fixed in the fork); the checkpoint mixes 3- and 4-bit expert
layers (5, 8, 45, 46 are 3-bit) and 8 / 7 bpw shared experts and projections; the tokenizer's split regex differs from Qwen3.8-27B's (marks are not
letters), recognised in `tokenizer.cpp`.

MTP (`--speculative --set speculator=mtp`): the head follows llama.cpp's qwen4exp MTP graph (per-stream norm of the pre-collapse streams, fc_hidden on every
stream plus the broadcast fc_embedding branch, one block with its own KV / QSA layer, its own mixer, the shared head). Greedy output equals the plain
decode exactly; 58 tok/s against 31.9 tok/s plain on a short chat request. Decode (plain) was 32.2 ms/token after CUDA graphs, the coalesced gated-residual
kernels and the 3-bit MoE dispatch fix; the 100K-token needle passes (prefill 274 tok/s, decode 22.5 before those optimizations); 256K with 2 slots loads
(arena 24.3 / 29.8 GiB per GPU), its needle is deferred to the end.

Decode profile (per GPU per token, 120-step nvprof): MoE persistent kernel 9.4 ms (213 us per call, ~6x off bandwidth), all-reduce 5.2 (ranks wait for rank 0's
shared expert), 7/8 bpw projections 6.9, gated residual ~5, sparse attention 1.1, router 1.5.

The fused cooperative MoE decode kernel (`exl3_moe_coop`) now also runs 4 bpw experts and up to 10 routes (fp16 tile; the int8 tile stays 3 bpw only): decode 32.2 -> 24.9 ms/token (40.1 tok/s), greedy tokens unchanged.

The dp4a int8 decode GEMV now also serves 7 and 8 bpw (M=1, N%256): 24.9 -> 23.5 ms/token, tokens unchanged. Splitting the shared expert over ranks (5 Hadamard spans: 1,1,1,2) was tried twice and reverted: the narrow N=128 slices miss the int8 GEMV (N%256) and fall to the 85 us GEMM (decode 30.6 ms), and even padded to 256 each rank would pay the same ~25 us per 8-bit launch, since these calls are latency-bound, not bandwidth-bound. The remaining lever is overlapping the shared expert with the MoE on a second stream (needs a backend event API) or fewer launches.

Fused MoE decode breakdown (`bench_moe_routes`, rank 0, 1 row, top 10): 0 local routes 6 us, 1-4 routes ~44 us (a 28 + b 16, flat in the route count: the call is latency-bound by the serial k-loops of the few resident blocks, so rank imbalance is nearly free), 5+ routes jump to ~90-160 us because the kernels are 512 threads x 106 registers (1 block/SM) and more than 80 active blocks run in two waves. Two k-splits for the gate/up stage at 4 bpw (the new default) take a call from ~76 to ~55 us: decode 23.5 -> 22.6 ms/token, tokens unchanged. Levers left: 256-thread tiles (2 blocks/SM) to remove the two-wave cliff, and a split-k for the down stage.

Rank 0's shared expert now runs on a side stream beside the router and the fused MoE decode kernels (`backend::stream_fork/stream_join`, CUDA graph edges; only when `MoE::fused_decode`, since those kernels use none of the trellis GEMM workspace the shared Linear needs): decode 22.6 -> 21.5 ms/token, tokens unchanged, API suite with MTP passes. A rewrite of the gated-residual up-mix / gate kernels with 16-byte weight loads and all loads in flight first was tried and reverted: no gain (22.8 ms) and one greedy token flipped from the changed summation order, so those kernels are not load-latency bound.

The gated add after each sublayer is deferred into the next site's norm pass (`GatedResidual::mix` prev_post / prev_y; flushed before any other stream reader and at the end of every captured body): 21.5 -> 21.3 ms/token, outputs bit-identical (decode tokens, prefix restore max |dlogit| 0, API suite with MTP).

Prefill: the gated-residual gate / up projections were 37% of GPU time at 512-row chunks (fp32 row-tile kernels, FMA bound). For more than 8 rows they now run as cuBLAS tensor-core GEMM pairs on fp16 hi / lo splits of the fp32 operands (fp32 accumulation; `GatedResidual::norm / gemm_f32 / gate_act / up_epilogue`, always at the full 512-row capacity so a row's result is bit-identical whatever its position or chunk size): prefill 504 -> 713 tok/s (8K tokens, chunk 512), 12K needle 675 tok/s. `test_gr_tc`: rel L2 7e-6 against the fp32 kernels, row-shift invariance exact. A plain fp16 version was faster (752) but moved logits by up to 1.0 and was dropped. Decode rows keep the fp32 kernels. Any perturbation moves the logits by a few tenths through router flips; split-vs-cold prefill at an unaligned cut already differed by 0.28 before this change.

The PLE layer's dense fp16 key / value projections (the plain GEMV behind `Linear::forward`, 20 ms per 512-row chunk) now use cuBLAS at the full row capacity for prefill rows: 713 -> 767 tok/s. The 48-token greedy test list moves at one near-tie token (index 38, 9989 -> 22157) with these prefill numerics; the text stays sensible and the API suite passes.

Prefill profile after that (8K tokens, one GPU): persistent 4-bit MoE kernel 28% (3.9 ms per 512-row chunk and layer, ~10 rows per expert so the trellis dequant is not amortized), sparse QSA attention 25% (fp32 CUDA cores, ~0.5 ms per layer and chunk), all-reduce 8% (skew), router logits 4%, GDN recurrence 3%.

The QSA sparse attention for prefill rows (more than 8 rows) now runs on the tensor-core prefill kernel (`attn_prefill_tc_kernel<..., SP = true>`: one block per row and kv head, the keys gathered through the row's top-k list, same fp16 P / fp32 accumulate as the dense prefill path); the fp32 one-key-at-a-time kernel (about 25% of prefill) stays for decode-sized calls: prefill 767 -> 981 tok/s (8K), 12K needle 962 tok/s, 32K needle 927 tok/s (was 279). Needles at 12K and 32K, prefix restore and the API suite with MTP pass.

Persistent prefill MoE kernel, looked at and left alone: 3.4 ms per call at any row count up to ~256 (128 experts x 3 matrices, all touched), 8 co-resident groups are the best split (2 groups 6.1 ms, 4 groups 4.9, 8 groups 4.7 at 512 rows); a dense GEMM of one 0.8 MB expert matrix takes ~12 us on its own (latency-bound), so 26 us per expert for three matrices is already better than serial dense calls. Cutting it further needs a grouped GEMM that schedules (expert, tile) items over all SMs; larger chunks would also amortize it (512 -> 2048 rows: 8.7 -> 5.8 us per row) but the row limit is global.

Quality check of the prefill changes (`test_ppl --flash`, teacher-forced perplexity of 6000 tokens of docs/ with every row's logits read back, `gather_logits` implemented for this model): all new paths 3.835, every old path 3.829 (the tensor-core gated residual alone 3.827, sparse attention alone 3.830). The per-token NLL moves by 0.06 on average (router flips) but the mean moves by 0.0016 nats, inside the noise of a 6000-token sample; no measurable quality change.

MTP on Flash-Next (`test_mtp --flash`, depth 7, full clock at 250 W): 34-token prompt plain 49.4 -> MTP 173.3 tok/s (85% of drafts accepted), 15K-token prompt plain 44.3 -> 109.3 tok/s (60%). Greedy output equals plain decoding on the 15K prompt and for the first 92 of 96 tokens on the short one (a late near-tie; the test holds Flash-Next to the first three quarters). The 158.7 / 112.5 tok/s figures that stood here before were Qwen3.8-27B numbers: `test_mtp` had no Flash-Next mode and loaded the 27B.

GPU utilization that alternated between 100% and 20-40% during prefill (power 64 <-> 150 W, one cycle per 512-row chunk) was the n-gram embedding rows: every rank decoded all 512 rows from the mmapped table on the host before the chunk's graph (113 ms of a 515 ms chunk, GPUs idle). The ranks now decode a quarter of the rows each into a shared host buffer behind counting barriers: prefill 981 -> 1164 tok/s, outputs unchanged. Decode is not affected (0.33 ms of host time per token). The ~2.4 ms holes that show up per decode step in `nvprof --print-gpu-trace` are profiler launch overhead, not present without it (bench_decode --flash: 20.1 ms/token plain). Remaining host-side idle per chunk is about 28 ms; overlapping the next chunk's rows with the current chunk would remove it.

Next: a GEMV for the N=640 shared-expert shapes, a tensor-core QSA scorer and sparse
kernel, the SASS / bench baselines and the 256K needle.

The PLE layer's host n-gram row decode of a prefill chunk now runs at the PLE layer (`forward_body`) instead of before the forward, so it overlaps the GPU running the layers queued before it (decode-sized rows keep the up-front decode for the graph path): prefill 1164 -> 1189 tok/s (8K, chunk 512). Prefix restore and the API suite with MTP pass; `gate.sh compare cap150`: 0 failures (decode within 1.5%).

Parallelism design (decision, 2026-10-09). Today: TP=4 for GDN / attention / dense projections, expert-parallel for the 512 experts (128 per rank, whole experts so the I=640 shape never needs a Hadamard-span split), the shared expert on rank 0 (overlapped on a side stream), the vocabulary sharded for the head, PLE rows decoded on the host by all ranks.
- TP=2 x 2 replicas (a pair per slot, for batch 2) does not fit: the GPU-resident weights are about 61 GB (the 100 GB checkpoint minus the 39 GB host n-gram table), 15.3 GiB per GPU at TP=4 (arena 19.3 GiB with the default KV and scratch), so a pair would hold ~30 GiB of weights per GPU against 29.8 GiB usable, before any KV. Only an expert tier in host RAM would make it fit, at the price of a PCIe read per cold expert. Not worth it.
- Batch 2 on TP=4 already shares every launch between the slots (rows are per-slot in one step), so a second slot costs the extra rows, not a second pass.
- What is left inside TP=4 is latency, not layout: 2 all-reduces per layer (96 per token, ~11 us floor plus the wait for the slowest rank), 8-bit projection launches (~25 us each), the gated-residual kernels. None of these change with a different split; they change with fewer launches (fusing the apply + norm further, a one-shot all-reduce that also applies the residual) and are per-kernel work, not a parallelism change.
- NVFP4 Flash-Next (skips the trellis dequant cost) needs a new checkpoint (download) and a perplexity check; on hold until there is disk space.

Numbers at the 250 W cap (2026-10-09; the 150 W numbers above are not comparable; baselines: `results/baseline_cap150.txt`, `results/baseline_cap250.txt`, local files):

| | Qwen3.8-27B | GLM-5.3-Flash | Flash-Next |
|---|---|---|---|
| M=1 decode (`bench_decode`) | 15.91 ms (62.8 tok/s) | 18.98 ms (52.7 tok/s) | 19.81 ms (50.5 tok/s) |
| M=8 verify | 24.88 ms | 41.94 ms | 30.78 ms |
| 2 slots, per step | | | 22.52 ms (88.8 tok/s aggregate) |
| Prefill, 512-row chunks | 2492 tok/s (2K) | 763 tok/s (2K) | 1322 tok/s (8K) |
| MTP, 34-token prompt (`test_mtp`) | plain 62.8, MTP 185.4 tok/s (82% accepted) | | plain 49.4, MTP 173.3 tok/s (85%) |
| MTP, 15K-token prompt | plain 58.8, MTP 132.2 tok/s (63%) | | plain 44.3, MTP 109.3 tok/s (60%) |

Real serving decode runs at 98% GPU utilization (the `bench_decode` trace shows 44%: a benchmark-loop artifact). A first version of this table carried the 27B's `test_mtp` numbers in the Flash-Next column; they are corrected above.


Several slots decoding with MTP (2026-10-10): the head keeps its state per slot (a history region per slot in the model, `MtpRunner` per-slot state), so a round can run for any slot. With more than one slot decoding the generator picks per step between one slot's round and a plain step of all of them, by the measured rates (a plain step over n slots returns n tokens, a round a few tokens of one slot), and re-measures the unused side every 64 steps. Full clock, creative-text requests (50% acceptance), aggregate tok/s with 2 / 4 concurrent requests: plain batching 78 / 122, interleaved rounds always 92 or 84 / 90, the controller 88 / 112 (a single request with MTP: 75-120). DFlash2 keeps a single shared context and so only speculates for a lone request. The concurrent greedy outputs diverge from the sequential ones (also without speculation, at the same positions: batched rows round differently from single rows), so equality across batch compositions is not a property of this model.

Joint MTP verify (2026-10-10, opt-in `spec_joint`): with several slots speculating, one draft chain (catch-up and each chain depth as one head step over all slots, hidden rows staged by `mtp_gather_hist`) and one verify forward over the blocks of all of them (mixed-slot rows; the recurrent-state snapshots grew to `kSnapRows` = 16 rows per rank so two 8-row blocks fit; `commit_spec_row` takes the global row). The Flash-Next decode graphs are now keyed by the one-slot variant (`attn_tc_`), as the 27B's always were, and graphs are warmed up to 16 rows when the speculator is joint-capable. Aggregate tok/s with 2 / 4 concurrent requests (`conc.py`-style runs of 600 tokens after a warm-up pass):
- creative text (about 50% of drafts accepted), full clock: plain batching 76-78 / 107-121; one slot per round (rate-chosen) 70-83 / 96-102; joint 75-80 / 97-101.
- code prompts (85% accepted), 175 W cap, the joint run throttled to a median 1290 MHz against 1515-1530 for the others: plain batching 60 / 115; one slot per round 143 / 134; joint 127 / 142 (about even at 2 and 25% ahead at 4 once scaled to the same clock).
The verify forward is not flat in rows above about 8 (the routed-expert work grows with the number of rows: M=8 costs 31 ms against 20 ms for one row), so a joint block of 12-16 rows pays for most of what it shares. One slot per round at 2 requests also reproduces the sequential (lone-request) outputs exactly on the code prompts, which batched and joint rows do not (rounding differs with the row structure). Joint stays off by default.

Interleaved prefill (2026-10-10, `prefill_chunk`, default 256 since 2026-10-10, 512 before): a new prompt used to prefill in full inside one scheduler step, so every other slot stopped decoding for the whole prompt. While other requests are decoding, one prompt is now fed `prefill_chunk` tokens per step (`Generator::prefill_slot` is resumable: cache restore, images and the checkpoint list are set up on the first call, checkpoint stores happen when a piece lands on one) and the decode steps of the other slots run between the pieces. A streaming 500-token generation with a 16.8K / 18.1K-token uncached prompt arriving 3 s in: the longest gap between streamed tokens 1.39 / 1.81 s, against 14.97 / 16.04 s with the whole prompt at once (`prefill_chunk=0`), the long request taking 17.5 / 17.8 s instead of 15.0 / 16.0 s. A piece costs more at 17K of context than at the start, so the gap is above the 0.4 s a 512-row chunk takes early on; a smaller chunk shortens it at some cost in prefill speed. With nothing decoding the prompt is fed in full as before. Models with a DFlash2 drafter keep the whole-prompt prefill (its single context would be disturbed by the decode steps in between). API suites (Flash-Next and Qwen 27B with MTP, 44 checks each) and the prefix-restore test pass.

Serving goals check (2026-10-10): the 256K needle passes (250,054 tokens, code at 50%; prefill 333.7 s = 749 tok/s, decode 29.4 tok/s). A Claude Code session (`claude -p`, MTP, 2 slots, prefix cache) read server sources to 64K tokens, edited a file with the Edit tool and resumed the cached prefix on every turn (61,876 of 64,143 tokens on the last one).

Message content is tokenized as plain text: the template's role / turn markers (`<|im_start|>`, `<|im_end|>`, ...) are real special tokens, but the same strings inside message content (a file a tool returned, say) are escaped before the template renders and encoded without special-token parsing (`chat.cpp`: `escape_control_text`, `encode_prompt`; `Tokenizer::control_tokens`), so quoted source cannot end or open a turn. `ENGINE_LOG_LEVEL=debug` logs the token that stopped each generation (`[gen] slot N stops on token ...`).

Known issue (open): when the model writes about its own chat template inside its thinking, it can emit the real `<|im_end|>` token (id 248046) where it means the text, which stops the generation mid-sentence with `end_turn` (the client sees "no visible output" and retries). Seen when the model is asked about code that quotes the markers (6 of 6 replays of one 58K request ended on 248046 in the reasoning; about half mid-thought, the others after a tool call); the context length is not the trigger and the escape above does not change it, so the model prefers the special id over the pieces. Candidate fixes, none applied: mask the EOS logits while a thinking block is open and no tool call has started (GPU sampler and MTP verify), or treat the token as literal text there and keep decoding. Note the model sometimes goes from thinking to a tool call without writing `</think>`, so "reasoning open" alone does not mean a stop is wrong.

Prefill chunk sweep (2026-10-10, Flash-Next MTP, 2 slots, 500-token stream plus a ~16K-token uncached prompt arriving 3 s in, two seeds): `prefill_chunk` 512 gives a longest stream gap of 1.31 / 1.29 s and the long request 19.3 / 18.2 s; 256 gives 1.06 / 1.21 s and 23.1 / 23.2 s; 128 gives 0.22 / 0.23 s and 28.2 / 28.9 s (p99 gap 0.22 s). The 256 maximum being as large as 512's points at one slow piece (or a fixed per-prompt step) rather than the chunk size; 128 is the only setting here that meets the 0.4-0.5 s target.

Perplexity against the model card (2026-10-10): `test_ppl --flash --qbench <checkpoint>/qbench_prompts.md` teacher-forces the 9 traced rows of the author's qbench set (input as context, the 20,487 sampled response tokens scored; token counts match the trace per row). The card (thelastspark/Swift-Qwen3.8-Flash-Next-exl3, qbench against the BF16 model) lists PPL 1.3825 for BF16, 1.3827 for the noise floor and 1.3810 for the 6.05 bpw quant (KL mean 0.00267); the 4.00 bpw quant is "pending" there. This engine on the 4.00 bpw quant: **PPL 1.4247** (mean NLL 0.35397), +3.0% over BF16. Per row: 1.309 1.303 1.363 1.592 1.685 1.707 2.226 1.294 1.265. Our input is one token longer than the trace's count on every row (a template or trailing-newline detail), which is negligible over 20K scored tokens. A KL figure needs the BF16 logits, which are not available locally.

Prefill gap cause and fix (2026-10-10): the 1.3 s stream gap came from the automatic prefix checkpoint (`prefix_interval` 16384) stored mid-prefill: `save_snapshot` pinned its host buffers with `cudaHostAlloc` at store time (756 of the 836 ms for 256 MiB x 4 ranks; the copy itself is 80 ms), stalling every decoding slot. `ModelCluster::prewarm_snapshots` now pins the buffers a prompt's planned checkpoints will need on a helper thread when its prefill starts (one pin per 250 ms so decode steps interleave), and the store finds them in the free list (80 ms). Longest stream gap with a 900-token stream and a 19.4K-token uncached prompt arriving 3 s in: `prefill_chunk` 512 0.63 s, 256 0.49 s, 128 0.33 s (before: 1.3 s at 512). The earlier 128 figure of 0.22 s was an artifact: that stream ended before the prompt reached the checkpoint. `prefill_chunk` 256 now meets the 0.4-0.5 s target; the default is still 512. Debug logging (`ENGINE_LOG_LEVEL=debug`) shows prefill pieces over 300 ms, checkpoint store times and snapshot buffer/copy times.

Default `prefill_chunk` is now 256 (user decision after the sweep above).

GLM-5.3-Flash interleaved prefill (2026-10-10, no drafter, 2 slots, `expert_cache` 0.95, 900-token stream plus a 15.2K-token uncached prompt arriving 3 s in): the path works (one piece per scheduler step, a 1-row decode step between pieces, the 16384-checkpoint store 69 ms with its pre-pinned buffers). A 256-row piece costs about 0.72 s (GLM prefills at ~350 tok/s), so the stream sees a steady 0.77 s gap: `prefill_chunk` 256 gives p99 0.77 s and a 45 s prompt; 128 gives p99 0.53 s and 55 s. Isolated gaps of exactly twice the step (1.55 s at 256, 1.03 s at 128) have no slow step behind them in the server log; they look like a withheld partial UTF-8 character emitted together with the next token. The expert-tier rebalance (every 64 decode steps) took 172 ms once. `prefill_chunk` is global, so GLM at the 256 default misses the 0.4-0.5 s target; 128 gets close.

Decode profile, 2026-10-10 (`nvprof --profile-from-start off`, `bench_decode --flash`, eager launches, TP=4; nvprof records one GPU's kernels here, 50 steps; kernel time 22.8 ms/step under the profiler against 19.9 ms/token of the plain bench without it). Short context, per GPU per token: 8-bit projections `exl3_gemm_kernel<8>` 3.3 ms (92 calls, 36 us each) + int8 GEMVs (8 and 7 bpw) 4.4 ms (202 calls); gated residual (up_mix 2.4, gate 1.6, norm 1.0, sigmoid_gated 0.4) 5.3 ms in 4 kernels per sublayer; fused MoE `a` + `b` 2.9 ms (44 calls); router (logits 1.65 ms = 34 us per layer, softmax-select 1.0) 2.6 ms; all-reduce 1.1 ms (96 calls); sparse attention 0.4 ms. Against the earlier profile (MoE 9.4, all-reduce 5.2, projections 6.9, gated residual ~5, router 1.5 ms): MoE and all-reduce shrank, the gated residual did not, the router grew. At 30K context the sparse attention is 3.0 ms per token (12 calls, 0.25 ms each), the only term that scales with context; M=1 decode 19.9 -> 22.7 ms, Mode A 8-row verify block 33 ms. Levers, by size: the 300 small projection launches are latency-bound (~25-36 us each, not bandwidth); the 4 gated-residual kernels per sublayer could fuse (gate + up_mix share inputs); the router logits GEMV at 34 us is well above a 1-row 512-column GEMV; attention at 256K is unmeasured (`BENCH_CTX` must stay below the arena's max_seq: at exactly max_seq the bench writes past the cache and corrupts the heap).
