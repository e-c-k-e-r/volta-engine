# Dead ends

Approaches that were built or measured and did not pay, with the measurement and the reason. Check here before retrying one. Numbers come
from the hardware of the time (4x V100, power caps as noted); relative results are what matter.

## Speculation and drafting

* **Drafter adaptation (LoRA on the DFlash2 drafter), first attempt, GLM.** Trained on recorded Claude Code sessions (Claude's text, not the
  served model's) with 768-position contexts and served through an unmerged adapter path. +20% tokens per round live on agent turns, but
  decode tok/s was flat: the adapter added 2-5% to every round, and the survival-based depth controller answered the higher acceptance with
  longer, more expensive blocks. Simulated at the true 2048 context the gain was +7%, chat about 0.
* **Drafter adaptation, retry, Qwen3.8-27B.** The served model's own sessions (480 chat prompts, 469 Claude Code agent turns from
  `tests/agentic`), full 2048-position taps, adapter merged into FP16 weights so it costs nothing per round. Paired round simulation on
  held-out turns: hard-label CE adapter agent -0.7% [-1.6, +0.4], chat +3.4%; on-policy distillation (target top-16 recorded by `tap_spool`,
  forward KL on accepted rows, reverse KL from the first rejection, anchors from the drafter's own rollout) agent +0.8% [+0.1, +1.7], chat
  +2.9%. The released drafter is already trained on its target's outputs; without a domain shift there is nothing to adapt to.
* **Survival-cut depth (`spec_depth`) on the 27B and Flash-Next.** It helps GLM (verify cost grows ~2.3 ms per row), but on the 27B a
  7-draft round costs about what a 3-draft round does (35 vs 32 ms), so any cut loses: 0.25 gave -12%. On Flash-Next DFlash2 0.15-0.60
  traded round cost for the same share of tokens. `spec_depth_rate` (expected tokens per measured millisecond) lands within 2% of whole
  blocks on the 27B for the same reason.
* **Candidate re-ranker for the drafter's top-16.** Scores, suffix agreement, n-gram counts and a small MLP over the 16 candidates: walk
  accuracy 72.7% -> 76.6% at best, about +3-5% decode on paper with overlap against `spec_suffix`. The target is in the top-16 91-95% of the
  time, but cheap features cannot close that gap.
* **Better suffix-match policy.** Longest match or voting instead of the last occurrence: under 2% overall; coverage, not the policy, limits
  suffix speculation.
* **Sampled drafts (standard speculative sampling with the drafter's distribution) for sampled requests.** Offline on 147 held-out agent
  turns, against today's argmax drafts: T=1 / top-p 0.95 5.37 -> 5.66 tokens per round (+5.3%, best draft temperature 0.8); T=0.7 / top-p 0.8
  +1.5%.
* **Speeding up the sampled step (T > 0).** Sampled decoding is slower only through speculation's acceptance; the step itself costs the same.
  - Qwen3.8-27B without speculation: 47.3 tok/s greedy against 47.2-47.5 sampled (T=1 / top-p 0.95 / top-k 20, 8K prompts).
  - With DFlash2 the argmax drafts are accepted less often by a sampled target: 5.82 -> 5.69, 6.92 -> 5.90, 5.57 -> 4.55 and 4.06 -> 2.74
    tokens per round, a larger loss on harder prompts.
  - GLM DFlash2 at T=0 against 0.7: 43.1 against 43.2 ms per round, 4.39 against 4.27 tokens per round.

  The top-16 pass, its sync and the penalties (applied on the host over the top-16 candidates) cost nothing measurable. Sampled drafts (the
  entry above) are the only way back.
* **FastMTP-style MTP head retraining on Flash-Next.** It fixes heads whose accuracy decays when reused for later steps. Measured over 5,906
  rounds of Claude Code agent tasks (`/api/stats` `steps`): conditional acceptance 0.825, 0.831, 0.847, 0.857, 0.884, 0.867, 0.875 for
  steps 1-7. No decay to recover; step 1 is the limit.
* **Longer blocks.** About 28% of agent rounds on the 27B accept all 7 drafts, but a longer block needs a drafter trained for it, and it
  only extends rounds that already succeed.
* **`spec_suffix` on Flash-Next DFlash2.** 166.1 / 160.9 tok/s at 8 / 4 against 165.1 off.
* **DeepInfer-style confidence-scheduled verification (DeepSeek DSpark, one stream).** 32.0 ms per token against 41.4 for fixed blocks,
  the same as a plain threshold; its gain comes from a budget shared across a large batch.

* **The Flash-Next shared expert rotated over the ranks** (layer L's on rank L mod tp, instead of all on rank 0): the per-rank kernel time evened
  out as intended (rank 0 -88 ms, the others +27 ms over a 4096-token prefill) but the prefill got 4% slower (1826 vs 1758 ms), idle gaps
  appearing on every rank; estimated from the per-layer critical path, moving it at all (rotated or split four ways) is worth about 1%, since
  the slowest rank of a layer is set by the routed experts' imbalance between the contiguous 128-expert blocks.

## MoE kernels

* **64-row expert tile, first version** (512 threads, 4 row blocks): correct but 30.3 vs 14.7 ms at 1024 rows; it spilled at 128 registers.
  The later 32-row tile at 2 blocks per SM is what shipped.
* **`__launch_bounds__(256,2)` with exact shared memory and a doubled grid**, on its own: 16.5 vs 14.7 ms.
* **64-row expert chunks for the prefill kernel at 2 blocks per SM** (the 64-row tile above 32 rows, chunks of 64 instead of 32, the shared
  memory sized for it): 2.5x slower at 1024 to 8192 rows (4096: 80 vs 32 ms per layer), the tile spilling at the 2-block register budget.
* **Deferred shared-memory stores for the emulated `cp.async`** (loads into registers before the barrier): 19-27% slower; the kernel is
  pinned at 128 registers and spills.
* **L2 prefetch of the B tile ahead of the pipeline:** +8% / -1% / -3% at 512 / 1024 / 2048 rows.
* **Volta-order trellis decode for the MoE prefill kernel by random-access decode:** the shuffle-removed ablation bounds it at 9% of the
  kernel, under 2% end to end. The Volta tile order was built instead at the quantizer (`tile_layout: volta` checkpoints).
* **Transcoding a low-bit EXL3 checkpoint to the Volta order.** The GLM 2.60 bpw re-encode lost the original's Hessian-guided rounding:
  perplexity 4.56 -> 33.3. Volta-order checkpoints are quantized from the original weights (`tests/tools/exl3_quantize.py`).
* **Per-expert slices over all ranks for DeepSeek decode with EXL3 packs (`expert_shard`):** rank waiting fell from 40% to 6%, but the fused
  MoE kernels cost per expert slot, not per byte (348 -> 815 ms), so the slowest rank got slower. GGUF experts, whose kernels cost about the
  same per byte at any slice width, run sharded by default. With the int8 MoE tile at every width it is level: decode -3%, verify +2-3%,
  speculative decode unchanged, 340 MiB more per rank of slice tables.
* **A k-sliced EXL3 GEMV of 2 to 16 rows on the fused MoE's tile** (blocks rotating their own input slice, the last block of a column group
  finishing it, no grid-wide barrier): level with the tensor-core GEMV at 3 to 8 rows (both are bound by decoding the stock tile order, whose
  tiles need operand shuffles for the MMA), slower at 9 to 16 (a decode per 8-row half), and at 2 rows 0.58 to 0.91 of the int8 GEMV on smaller
  projections, but a 2-row call then rounds differently from the 1-row decode, which speculative verification needs to match (Flash-Next's
  MTP greedy check failed). DeepSeek-V4.1's verify: 38.8 -> 38.3 ms a round.

## Prefill and multi-GPU balance

* **Rotating Flash-Next's shared expert over the ranks in prefill:** rank imbalance halved, prefill +0.4% (within noise). A layer's
  all-reduce waits for the slowest rank of that layer.
* **Overlapping rank 0's shared expert with the routed experts in prefill:** +0.2%. Rank 0's shared-expert work does not limit a chunk.
* **Flash-Next routed experts placed by route counts** (each layer's experts dealt to the ranks heaviest first from calibration texts, the
  router's columns permuted to match): on held-out code and agent transcripts the most-loaded rank's expert work per 512-row chunk fell from
  1.15-1.19x to 1.12-1.14x the mean, and an 8192-token prefill of an agent transcript from 3325 to 3305 ms (-0.6%). Most of the imbalance
  changes from chunk to chunk, which no fixed placement follows.
* **Flash-Next routed experts sliced over the ranks** (every rank 160 of each NVFP4 expert's 640 intermediate columns, as GLM's sharded
  layout): the ranks no longer wait for each other at the MoE, but the agent-transcript prefill went from 3326 to 3518 ms (+5.8%), M=1
  decode stayed level (10.41 vs 10.45 ms) and the 8-row verify got 2.3% slower. Each rank then runs every active expert with a quarter of the
  work per weight pass.
* **FP16 residual window for layered prefill:** `hc_collapse_norm` is bound by re-reading its mixing matrix, not by the stream bytes; FP16
  streams would save memory only. Row-blocking the mix was the lever (`hc_rows`).
* **Multi-accumulator `kda_recurrent_rows`:** a latency-bound sequential chain; reordering would break bit-identity with the fused, journal,
  replay and decode kernels for about 4% of prefill.
* **Tensor-core router logits on Flash-Next:** about 3% of prefill at most, and it changes the summation order the decode router shares, so
  expert ties could differ between prefill and decode.
* **Ruled out from measurements:** decode rows piggybacked on prefill chunks (the chunk time sets the stall), 1024-row prefill chunks (MoE
  per row only -22%), FP16 KDA state (~0.1 ms), expert-aware row choice in a verify block (rows form a chain).

## Expert offload and the host link

* **Copy-engine DMA staging of cold experts for DeepSeek prefill** (double buffer, event-ordered): 17.9 vs 13.8 s against the kernels'
  zero-copy reads.
* **One-layer-ahead (L+1) expert prefetch for decode:** a decode layer is ~0.37 ms and a cold expert ~0.8 ms on its own (~3 ms with four
  GPUs on the shared link), so a prefetch hides 10-40% of one read; prefill staging (`expert_stage`) covers the case where it works.
* **CPU cold experts for DeepSeek decode:** no change (45.1 vs 44.5 ms); a group usually has one distinct cold expert, which the CPU split
  never takes. On GLM plain decode the CPU path is slower at every setting (1.7-2.3 ms per expert vs 0.8 ms over PCIe); it pays only with
  speculation (several cold experts per layer) and on Flash-Next NVFP4.
* **CPU experts read in place (EXL3, no repacked copy):** the kernel reached 1.07-1.10x the repacked layout, but decode lost 27%
  (21.2 -> 15.4 tok/s): each thread's small output slice strides across tiles and NUMA nodes. Patch kept in
  `.agents/shelved/cpu_exl3_in_place.patch`.
* **Widening the AVX2 CPU expert kernel to AVX-512 on Skylake-SP (Xeon Gold 6132).** Two of the kernel's tiles per 512-bit vector:
  4.44 -> 4.14 ms per 3-bit GLM expert on one thread, 0.76 -> 0.82 ms on ten (slower), outputs unchanged. Integer accumulation with int16
  activations: 0 to 6%. Skylake-SP issues the 512-bit integer multiply of the state hash on a single port and lowers the clock under heavy
  AVX-512 load; the AVX2 kernel already does about 6 billion weights per second per core. This rules out the widening only: a kernel
  designed for AVX-512BW (exllamav3's int8 paired-row tier: 16-bit state hashing, 32 states per vector, int8 activations with about 1%
  output error, reported about 2x on a Xeon Gold 6148) has not been measured here.

## Quantization

* **Agent-heavy calibration for the GLM Volta requant:** 70% own agent sessions overfit the experts (8% better on calibration rows, 4-16%
  worse on held-out prose and multilingual). Calibrate broadly and score on held-out text of several kinds.
* **Best-of-N quantizer seeds:** 8 sign-vector seeds on three 27B tensors gave the same proxy error within 0.1%.
* **NVFP4 as an intermediate (BF16 -> NVFP4 -> EXL3):** ~0.02 nats worse (perplexity +2%) at 4 bpw than direct; use original weights.
* **EXL3 from a QAT NVFP4 checkpoint:** +0.009 nats against the QAT checkpoint itself; the quantization-aware tuning is lost.
* **Agent-turn NLL as a quantization metric:** too low-entropy to resolve differences; use high-entropy text, weight-space error or KL.

## Vulkan

* **Folding the split-K attention reduce into `fa_decode` (Gemma 4 E4B decode).** Decode keeps the split count chosen at graph capture, sized for
  the longest context, so `fa_reduce` runs on every layer at every step. Skipping it (wrong output, timing only) bounds the gain at 3.8%: 9.33
  -> 8.98 ms per step on a V100.
  - A correct fold needs the last workgroup to read the other splits' partials coherently. Slang rejects `globallycoherent` on the
    buffer-device-address pointers every kernel uses.
  - Atomic loads of 64 splits' partials would cost more than the dispatch.

  llama.cpp's Vulkan flash attention keeps its split-K reduce as a separate dispatch too.

## Steering

* **Static clamp files on GLM-5.3-Flash and DeepSeek-V4.1-Flash:** only the 27B responds; see [STEERING.md](STEERING.md).
* **Router steering against refusals (GLM-5.3-Flash):** a bias on the router scores of the experts that separate declined prompts from
  answered ones (held-out AUC 0.99) left 0.47 to 0.60 of held-out declined targets declining, against 0.60 without it; the experts mark a
  refusal rather than cause it. See [STEERING.md](STEERING.md).

## Reasoning control

* **Per-turn thinking (DART-style: two quick drafts without reasoning, think only when they disagree), Qwen3.8-27B in Claude Code.** On 20
  harder agent tasks thinking off passed 34/40 and thinking high 39/40 (1.79x the wall time), so a gap exists; but replaying 202 recorded
  thinking-high turns, the two no-think drafts agreed on 65% of turns (same kind of step and files) and 25% of those agreed steps differed
  from the reasoned one, mostly hastier (running before fixing, acting before reading, finishing without running). Skipping reasoning on
  agreement would take a hasty step on about one turn in ten. Tool: `tests/agentic/dart_replay.py`.
* **Thinking intervention (a first-person thought in the empty reasoning block of thinking-off turns), Qwen3.8-27B.** "Before I act I make
  sure the step is ready: I read the code or data it depends on, follow the task's rules exactly as stated, and check the result before I
  report it." On the same 20 tasks, 2 runs each: 33/40 with it and 33/40 without (143 s and 147 s per task).
