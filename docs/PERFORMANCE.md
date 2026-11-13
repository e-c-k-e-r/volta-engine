# Performance

Serving speed, step latency and prefill for each model and weight format on the target machine, and how they were measured. Every number below
comes from one session on 2026-11-12 (master 54e2f89), so the tables compare with each other; speed depends on the power cap and clocks, so
rerun the whole set rather than one table when either changes.

## Setup

* 4x Tesla V100-SXM2-32GB, NVLink between every pair, water-cooled, power limit 250 W, SM clocks free up to 1530 MHz
* CUDA 12.8, tensor parallel over all four GPUs (`tp=4`), CUDA graphs on, `numactl -N 0 -m 0`
* every expert resident on the GPUs (`expert_cache` 1, or `auto` where noted and it resolved to 100%), except DeepSeek-V4.1-Flash: 92% of
  its experts fit, the rest are read from pinned host memory
* one request at a time (`max_slots=1`), `max_seq=32768` (DeepSeek: 16384), greedy decoding, thinking off
* weights:

| Model | Weights | Checkpoint |
|---|---|---|
| Qwen3.8-27B | EXL3 4.0 bpw (head 5) | `Qwen3.8-27B-exl3` |
| | NVFP4 | `Qwen3.8-27B-QUASAR-NVFP4` |
| | GGUF IQ3_S | `Qwen3.8-27B-GSQ-RCO-GGUF-IQ3_S` |
| Flash-Next | EXL3 4.0 bpw (head 6), Volta tile order, from the base model | `Qwen3.8-Flash-Next-volta-cal` |
| | EXL3 4.0 bpw (head 6), stock order, the Swift fine-tune | `Swift-Qwen3.8-Flash-Next-exl3-4.00bpw` |
| | NVFP4 experts as shipped, dense groups quantized to NVFP4 at load (`nvfp4_dense=all`) | `Qwen3.8-Flash-Next-NVFP4` |
| | GGUF IQ3_S, the Swift fine-tune | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` |
| GLM-5.3-Flash | EXL3 3.05 bpw (head 6), stock order | `GLM-5.3-Flash-exl3-3.05bpw` |
| | EXL3 3.05 bpw (head 6), Volta order | `GLM-5.3-Flash-exl3-3.05bpw-volta-cal` |
| DeepSeek-V4.1-Flash | EXL3 1.68 bpw (head 6) | `DeepSeek-V4.1-Flash-1017-exl3-1.68bpw` |

* speculators:
  * MTP: the checkpoint's own head (GLM's comes from layer 45 of `GLM-5.3-Flash-NVFP4`), `mtp_depth` 7 unless noted. The head stops chaining
    earlier when its recent accuracy is low, so a depth is a ceiling.
  * DFlash2: the whole 7-token block every round (`spec_depth` 0, `spec_depth_rate` off), with these drafters:
    * Qwen3.8-27B: EXL3 5.0 bpw;
    * Flash-Next: EXL3 5.0 bpw, in the target's tile order;
    * GLM: EXL3, 4.39 bpw mean (`calibrated-v2` for stock, `volta-cal` for Volta).
  * DSpark: DeepSeek's drafter, EXL3 3.05 bpw mean (`DSpark-EXL3-e3`), `dspark_conf=0.7`.
  * Suffix: `speculator=suffix`, `spec_suffix=8`.
* GLM's GGUF (GSQ-RCO 3.0-bit) is larger on the GPUs than the 3.05 bpw EXL3 and fits only 83% resident, so it is not in the tables.

## Methodology

Served speed: a server per configuration, a warm-up request, then `tests/tools/agent_replay_run.py` sends each test set one request at a time
through the streaming chat API with `temperature` 0 and `enable_thinking` false. The test sets:

| Set | Prompts | New tokens | What it is |
|---|---|---|---|
| Agent replay | 30 turns, 13-20K tokens each | at most 300 | turns of a recorded Claude Code session (`agent_replay_build.py`): each prompt is the history before a turn, and the server writes the turn; later turns of a segment are served from the prefix cache |
| Chat | 20 | at most 300 | single-turn questions and writing requests |
| GSM8K | 24 (seed 0) | at most 768 | grade-school math word problems |
| MBPP | 24 (seed 0) | at most 768 | Python tasks with their test asserts |

* decode tokens per second: the generated tokens over the time from the first token to the last, summed over the set
* tokens per round: (accepted drafts + rounds) / rounds over the set, from the `/api/stats` counters taken before and after it
* one run per configuration; repeats of a configuration on this machine agree within about 3%
* every configuration starts only when each GPU is at or below 55 C

Step latency: `build/bench_decode` replays the graph-captured decode body on the real weights: one row (a plain token) and blocks of 2 to 16
rows (`BENCH_VL=2,4,8,12,16`, what a speculative round verifies), the median of two or three runs.

Prefill: the cold requests of the agent replay (10-13K new tokens, nothing from the prefix cache), as the servers log them.

## Serving speed

Decode tok/s, tokens per round in parentheses.

| Model | Weights | Speculator | Agent replay | Chat | GSM8K | MBPP |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | EXL3 4.0 bpw | MTP | 116 (4.80) | 102 (3.40) | 143 (5.34) | 117 (4.51) |
| | EXL3 4.0 bpw | DFlash2 | 109 (4.30) | 120 (4.12) | 186 (6.23) | 167 (5.68) |
| | NVFP4 | DFlash2 | 154 (4.02) | 212 (3.93) | 321 (6.33) | 292 (6.04) |
| | GGUF IQ3_S | DFlash2 | 108 (4.04) | 130 (4.02) | 205 (6.33) | 174 (5.70) |
| Flash-Next | EXL3 4.0 bpw, Volta order (base) | MTP | 133 (4.51) | 128 (3.77) | 177 (5.95) | 159 (5.32) |
| | EXL3 4.0 bpw, Volta order (base) | DFlash2 | 115 (3.78) | 123 (3.58) | 197 (5.81) | 184 (5.47) |
| | EXL3 4.0 bpw, stock order (Swift) | MTP | 113 (4.43) | 112 (3.54) | 158 (5.76) | 155 (5.53) |
| | EXL3 4.0 bpw, stock order (Swift) | DFlash2 | 96 (3.63) | 105 (3.40) | 169 (5.59) | 172 (5.63) |
| | NVFP4 | MTP | 165 (4.09) | 187 (3.63) | 267 (5.84) | 247 (5.29) |
| | NVFP4 | DFlash2 | 130 (3.31) | 167 (3.40) | 264 (5.48) | 266 (5.29) |
| | GGUF IQ3_S (Swift) | DFlash2 | 100 (3.79) | 101 (3.44) | 165 (5.54) | 156 (5.28) |
| GLM-5.3-Flash | EXL3 3.05 bpw, stock order | MTP | 71 (3.40) | 81 (2.97) | 94 (3.90) | 85 (3.87) |
| | EXL3 3.05 bpw, stock order | DFlash2 | 70 (3.51) | 81 (3.52) | 127 (5.55) | 115 (5.31) |
| | EXL3 3.05 bpw, Volta order | MTP | 71 (3.33) | 80 (2.92) | 92 (3.72) | 86 (3.86) |
| | EXL3 3.05 bpw, Volta order | DFlash2 | 72 (3.52) | 82 (3.37) | 126 (5.29) | 118 (5.33) |
| DeepSeek-V4.1-Flash | EXL3 1.68 bpw, 92% resident | none | | 32 | 34 | 33 |
| | | DSpark | | 44 (2.10) | 70 (3.35) | 71 (3.50) |
| | | suffix | | 33 (1.01) | 34 (1.01) | 38 (1.16) |

DeepSeek's 16K window does not hold the agent replay's prompts.

* which speculator wins depends on the text:
  * on agent turns, which copy much of their input (file contents, tool arguments), MTP is ahead on Qwen3.8-27B and Flash-Next;
  * on math and code, DFlash2 is ahead on every model;
  * on GLM the two tie on agent and chat text, and DFlash2 is 30-35% faster on math and code.
* NVFP4 is the fastest format on both Qwen models. The EXL3 rows are 4.0 bpw, which costs more to decode per weight than NVFP4's unpacking.
* Flash-Next's two EXL3 checkpoints differ in two ways at once: the tile order and the model (base against the Swift fine-tune). The base
  model is the one the DFlash2 drafter was trained on, which shows in its tokens per round.
* GLM's tile order makes no difference to served speed: the verify blocks are 5-7% faster (step latency below), and the rest of a round
  hides it.

### MTP depth

Flash-Next, EXL3 Volta order, the same sets and settings, `mtp_depth` from 4 to 7 against DFlash2:

| Set | MTP depth 4 | 5 | 6 | 7 | DFlash2 |
|---|---|---|---|---|---|
| Agent replay | 116 (3.71) | 134 (4.20) | 127 (4.11) | 133 (4.51) | 115 (3.78) |
| Chat | 125 (3.43) | 126 (3.60) | 129 (3.72) | 128 (3.77) | 123 (3.58) |
| GSM8K | 162 (4.46) | 167 (5.05) | 172 (5.54) | 177 (5.95) | 197 (5.81) |
| MBPP | 152 (4.20) | 153 (4.63) | 155 (5.00) | 159 (5.32) | 184 (5.47) |

Each MTP draft step is one more pass of the head, about 2 ms per round (GSM8K: 27.5 ms per round at depth 4, 33.5 at depth 7). The DFlash2
drafter writes its whole block in one pass, 29.5 ms per round on the same set.

## Step latency

`bench_decode`, ms per step: one row is a plain token, 2 to 16 rows a verify block.

| Model | Weights | 1 row | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | EXL3 4.0 bpw | 16.0 | | | 24.8 | | |
| | GGUF IQ3_S | 12.6 | 14.4 | 18.3 | 25.1 | 34.7 | 35.9 |
| Flash-Next | EXL3 4.0 bpw, Volta order | 18.4 | 19.2 | 21.9 | 25.3 | 32.4 | 34.9 |
| | EXL3 4.0 bpw, stock order | 18.4 | 20.1 | 23.8 | 28.0 | 35.5 | 38.8 |
| | NVFP4 | 9.8 | 11.0 | 13.3 | 15.9 | 23.4 | 25.3 |
| | GGUF IQ3_S | 14.6 | 15.3 | 20.5 | 29.5 | 41.6 | 47.1 |
| GLM-5.3-Flash | EXL3 3.05 bpw, stock order | 15.9 | 21.2 | 28.5 | 36.2 | 48.2 | 54.6 |
| | EXL3 3.05 bpw, Volta order | 15.9 | 21.5 | 27.0 | 33.6 | 47.6 | 53.0 |

The Qwen3.8-27B EXL3 row is the gate benchmark's (one row and the 8-row block only). Flash-Next's one-row time includes the n-gram embedding
lookups on the host, which vary from run to run by up to 2 ms.

## Prefill

Cold prompts of 10-13K new tokens, tok/s:

| Model | Weights | Prefill |
|---|---|---|
| Qwen3.8-27B | EXL3 4.0 bpw | 2100 |
| | NVFP4 | 2250 |
| | GGUF IQ3_S | 1960 |
| Flash-Next | EXL3 4.0 bpw, Volta order | 1520 |
| | EXL3 4.0 bpw, stock order | 1350 |
| | NVFP4 | 1950-2060 |
| | GGUF IQ3_S | 850 |
| GLM-5.3-Flash | EXL3 3.05 bpw, stock order | 970-990 |
| | EXL3 3.05 bpw, Volta order | 340 |

GLM's Volta order prefills at a third of the stock order's speed: the wide prefill form of the trellis MoE kernel spills more registers in the
Volta-order build of the kernel.

## Settings for several requests

| Requests at once | `max_slots` | Other settings |
|---|---|---|
| 1 | 1 or 2 | `spec_suffix=8` when the output copies its input |
| 2 | 2 | the same, `prefill_budget_ms=300` if token gaps matter more than prompt speed |
| 3 to 4 | 4 | the same; the slots share the KV pool, so their contexts add up to `kv_pool` tokens |

## Reproducing

```
build/volta_server --model-dir <checkpoint> --use-speculation --set speculator=<mtp|dflash|dspark|suffix> [--draft-model-dir <drafter>] \
    --set tp=4 --set max_slots=1 --set max_seq=32768 --set max_image_tokens=0
tests/tools/agent_replay_run.py <turns.json> <out.json> --max-tokens 300      # the agent replay and the chat set; 768 for GSM8K and MBPP
curl -s localhost:8000/api/stats                                             # before and after a set: rounds and accepted drafts
ENGINE_TP=4 BENCH_VL=2,4,8,12,16 build/bench_decode [--glm | --flash]         # step latency; ENGINE_TEST_<ROLE>_DIR picks the checkpoint
```

Record each GPU's power limit and clocks with the numbers, and compare runs only at the same cap and clocks.
