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
* **FastMTP-style MTP head retraining on Flash-Next.** It fixes heads whose accuracy decays when reused for later steps. Measured over 5,906
  rounds of Claude Code agent tasks (`/api/stats` `steps`): conditional acceptance 0.825, 0.831, 0.847, 0.857, 0.884, 0.867, 0.875 for
  steps 1-7. No decay to recover; step 1 is the limit.
* **Longer blocks.** About 28% of agent rounds on the 27B accept all 7 drafts, but a longer block needs a drafter trained for it, and it
  only extends rounds that already succeed.
* **`spec_suffix` on Flash-Next DFlash2.** 166.1 / 160.9 tok/s at 8 / 4 against 165.1 off.
* **DeepInfer-style confidence-scheduled verification (DeepSeek DSpark, one stream).** 32.0 ms per token against 41.4 for fixed blocks,
  the same as a plain threshold; its gain comes from a budget shared across a large batch.

## MoE kernels

* **64-row expert tile, first version** (512 threads, 4 row blocks): correct but 30.3 vs 14.7 ms at 1024 rows; it spilled at 128 registers.
  The later 32-row tile at 2 blocks per SM is what shipped.
* **`__launch_bounds__(256,2)` with exact shared memory and a doubled grid**, on its own: 16.5 vs 14.7 ms.
* **Deferred shared-memory stores for the emulated `cp.async`** (loads into registers before the barrier): 19-27% slower; the kernel is
  pinned at 128 registers and spills.
* **L2 prefetch of the B tile ahead of the pipeline:** +8% / -1% / -3% at 512 / 1024 / 2048 rows.
* **Volta-order trellis decode for the MoE prefill kernel by random-access decode:** the shuffle-removed ablation bounds it at 9% of the
  kernel, under 2% end to end. The Volta tile order was built instead at the quantizer (`tile_layout: volta` checkpoints).
* **Transcoding a low-bit EXL3 checkpoint to the Volta order.** The GLM 2.60 bpw re-encode lost the original's Hessian-guided rounding:
  perplexity 4.56 -> 33.3. Volta-order checkpoints are quantized from the original weights (`tests/tools/exl3_quantize.py`).
* **Per-expert slices over all ranks for DeepSeek decode (`expert_shard`):** rank waiting fell from 40% to 6%, but the fused MoE kernels
  cost per expert slot, not per byte (348 -> 815 ms), so the slowest rank got slower.

## Prefill and multi-GPU balance

* **Rotating Flash-Next's shared expert over the ranks in prefill:** rank imbalance halved, prefill +0.4% (within noise). A layer's
  all-reduce waits for the slowest rank of that layer.
* **Overlapping rank 0's shared expert with the routed experts in prefill:** +0.2%. Rank 0's shared-expert work does not limit a chunk.
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
