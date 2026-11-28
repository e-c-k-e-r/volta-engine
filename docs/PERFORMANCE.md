# Performance

Serving speed, step latency and prefill for each model and weight format on the target machine, and how they were measured. The tables come
from one session on 2026-11-28 (master ac4de36, step latency of Gemma 4 at beea09c) for the checkpoints on the machine then. The configurations
whose checkpoints are no longer on it are under [Earlier session](#earlier-session-2026-11-12-to-14) with their dates; that session's chat,
GSM8K and MBPP sets were different ones, so compare its numbers on those sets only among themselves. Speed depends on the power cap and clocks,
so rerun the whole set rather than one table when either changes.

## Setup

* 4x Tesla V100-SXM2-32GB, NVLink between every pair, water-cooled, power limit 250 W, SM clocks free up to 1530 MHz
* CUDA 12.8, tensor parallel over all four GPUs (`tp=4`), CUDA graphs on, the server bound to node 0 (`numactl -m 0 --physcpubind=0-9`)
* host: 2x Xeon Gold 6132, turbo on, no frequency cap; prefill and the host side of a speculative round slow down with a capped CPU
* every expert resident on the GPUs (`expert_cache` 1); GLM with `vram_reserve_mib=256`, which the routed experts need beside its drafter
  (27.44 GiB of them a GPU, 0.04 GiB of the arena left)
* one request at a time (`max_slots=1`), `max_seq=32768`, greedy decoding, thinking off
* weights:

| Model | Weights | Checkpoint |
|---|---|---|
| Qwen3.8-27B | NVFP4 | `Qwen3.8-27B-QUASAR-NVFP4` |
| Flash-Next | NVFP4 experts as shipped, every dense group in NVFP4 calibrated (`nvfp4_dense=all`, `nvfp4_dense_dir`) | `Qwen3.8-Flash-Next-NVFP4`, `Qwen3.8-Flash-Next-NVFP4-dense` |
| GLM-5.3-Flash | EXL3 3.05 bpw (head 6), Volta tile order, extended trellis (tile start states) | `GLM-5.3-Flash-exl3-3.05bpw-volta-cal` |
| Gemma 4 31B | EXL3 4.0 bpw (head 6), stock tile order | `gemma-4-31b-it-exl3` |

* speculators:
  * MTP (Flash-Next): the checkpoint's own head, `mtp_depth` 7. The head stops chaining earlier when its recent accuracy is low, so the depth is
    a ceiling.
  * DFlash2: the whole 7-token block every round (`spec_depth` 0, `spec_depth_rate` off), with these drafters:
    * Qwen3.8-27B: EXL3 5.0 bpw (`Qwen3.8-27B-DFlash2-EXL3-5.0bpw`);
    * Flash-Next: EXL3 5.0 bpw in the Volta order (`Qwen3.8-Flash-Next-DFlash2-EXL3-5.0bpw-volta-cal`);
    * GLM: EXL3, 4.39 bpw mean (`GLM-5.3-Flash-DFlash2-exl3-volta-cal`).
  * DFlash (Gemma 4): the first-generation drafter, EXL3 4.0 bpw (`gemma4-31b-it-DFlash-exl3-4.00bpw`).

## Methodology

Served speed: a server per configuration, a warm-up request, then `tests/tools/agent_replay_run.py` sends each test set one request at a time
through the streaming chat API with `temperature` 0 and `enable_thinking` false. The test sets:

| Set | Prompts | New tokens | What it is |
|---|---|---|---|
| Agent replay | 30 turns, 11-20K tokens each | at most 300 | turns of a recorded Claude Code session (`agent_replay_build.py` with its defaults): each prompt is the history before a turn, and the server writes the turn; later turns of a segment are served from the prefix cache |
| Chat | 20 (`tests/data/bench_chat20.json`) | at most 300 | single-turn questions and writing requests |
| GSM8K | 24 (`tests/data/bench_gsm8k24.json`) | at most 768 | grade-school math word problems of the test split |
| MBPP | 24 (`tests/data/bench_mbpp24.json`) | at most 768 | Python tasks of the test split with their test asserts |

`tests/tools/bench_sets.py` draws the GSM8K and MBPP problems (`random.Random(0)`) and writes the three files.

* decode tokens per second: the generated tokens over the time from the first token to the last, summed over the set
* tokens per round: (accepted drafts + rounds) / rounds over the set, from the `/api/stats` counters taken before and after it
* one run per configuration; repeats of a configuration on this machine agree within about 3%
* every configuration starts only when each GPU is at or below 55 C

Step latency: `build/bench_decode` replays the graph-captured decode body on the real weights: one row (a plain token) and blocks of 2 to 16
rows (`BENCH_VL=2,4,8,12,16`, what a speculative round verifies), the median of two runs.

Prefill: the three cold requests of the agent replay (11-13K new tokens, nothing from the prefix cache), as the servers log them.

## Serving speed

Decode tok/s, tokens per round in parentheses.

| Model | Weights | Speculator | Agent replay | Chat | GSM8K | MBPP |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | NVFP4 | DFlash2 | 178 (4.02) | 176 (3.28) | 337 (6.35) | 312 (5.90) |
| Flash-Next | NVFP4 | MTP | 220 (4.73) | 182 (3.10) | 292 (5.98) | 261 (5.32) |
| | NVFP4 | DFlash2 | 172 (3.83) | 156 (3.05) | 300 (5.81) | 275 (5.39) |
| GLM-5.3-Flash | EXL3 3.05 bpw, Volta order | DFlash2 | 92 (3.96) | 73 (2.82) | 137 (5.33) | 135 (5.44) |
| Gemma 4 31B | EXL3 4.0 bpw, stock order | DFlash | 70 (2.87) | 79 (2.87) | 157 (5.64) | 104 (4.70) |

* which speculator wins depends on the text: on Flash-Next, MTP is ahead on agent turns, which copy much of their input (file contents, tool
  arguments), and on chat; DFlash2 on math and code.
* Gemma 4 31B is the slowest per round of the dense models: its 8-row verify block takes 30.6 ms against the Qwen3.8-27B NVFP4's 14.0 (step
  latency below), and its first-generation drafter writes 2.87 tokens a round on agent turns and chat against 3.3-4.0 for the 27B's DFlash2
  (on GSM8K the two are close, 5.64 against 6.35). Two things make its verify blocks slow:
  * its weights are EXL3 trellis in the stock tile order, which costs more to decode than NVFP4's unpacking and more again for two rows or more
    (its 2-row block takes 21.2 ms against 14.8 for one row; GLM's verify blocks of 4 to 16 rows took 24-35% less time in the Volta order than
    in the stock order, earlier session);
  * its full-attention layers cost more as the context grows: at a 16K-token context its one row takes 17.4 ms and its 8-row block 34.0,
    2.6 and 3.4 ms more than at 8 tokens, where the 27B's take 0.9 and 1.0 ms more (its linear-attention layers cost the same at any length).

## Step latency

`bench_decode`, ms per step: one row is a plain token, 2 to 16 rows a verify block.

| Model | Weights | 1 row | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | NVFP4 | 11.1 | 11.7 | 12.3 | 14.0 | 18.3 | 20.8 |
| Flash-Next | NVFP4, every dense group in NVFP4 | 9.0 | 10.3 | 12.5 | 15.6 | 19.5 | 21.3 |
| GLM-5.3-Flash | EXL3 3.05 bpw, Volta order | 15.8 | 18.6 | 24.1 | 31.0 | 38.0 | 42.7 |
| Gemma 4 31B | EXL3 4.0 bpw, stock order | 14.8 | 21.2 | 27.0 | 30.6 | 35.6 | 37.7 |

Flash-Next's one-row time includes the n-gram embedding lookups on the host, which vary from run to run by up to 2 ms; its run needs
`vram_reserve_mib=1536` for the verify graphs of up to 16 rows (and `max_image_tokens=0`).

## Prefill

Cold prompts of 11-13K new tokens, tok/s:

| Model | Weights | Prefill |
|---|---|---|
| Qwen3.8-27B | NVFP4 | 2550-2570 |
| Flash-Next | NVFP4 | 2040-2200 |
| GLM-5.3-Flash | EXL3 3.05 bpw, Volta order | 1330-1340 |
| Gemma 4 31B | EXL3 4.0 bpw, stock order | 1630-1840 |

## Earlier session (2026-11-12 to 14)

The Qwen3.8-27B and Flash-Next numbers come from one session on 2026-11-12 (master 54e2f89), the GLM-5.3-Flash and DeepSeek-V4.1-Flash numbers
from one on 2026-11-13 (master da1f5be) and DeepSeek's GGUF build's on 2026-11-14 (master 25e4e6f), with the setup above except where noted.
The agent replay was the same 30 turns; the chat, GSM8K and MBPP sets were others (20 prompts, 24 and 24 problems drawn with seed 0).

* checkpoints: `Qwen3.8-27B-exl3` (EXL3 4.0 bpw, head 5), `Qwen3.8-27B-GSQ-RCO-GGUF-IQ3_S`, `Qwen3.8-Flash-Next-volta-cal` (EXL3 4.0 bpw, head 6,
  Volta order, base model), `Swift-Qwen3.8-Flash-Next-exl3-4.00bpw` (stock order, the Swift fine-tune), `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`,
  `GLM-5.3-Flash-exl3-3.05bpw` (stock order), the earlier `GLM-5.3-Flash-exl3-3.05bpw-volta-cal` (Volta order, before the 2026-11-25 requant),
  `DeepSeek-V4.1-Flash-1017-exl3-1.68bpw`, `DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` (IQ1_M, 1.75 bpw; the Engram table IQ4_NL); the Flash-Next
  NVFP4 row had its dense groups rounded to nearest at load
* GLM's MTP head came from layer 45 of `GLM-5.3-Flash-NVFP4`; GLM's stock-order drafter was `calibrated-v2`; DeepSeek's drafter DSpark, EXL3
  3.05 bpw mean (`DSpark-EXL3-e3`), `dspark_conf=0.7`; suffix decoding `speculator=suffix`, `spec_suffix=8`
* DeepSeek-V4.1-Flash: `max_seq=16384` (its window does not hold the agent replay's prompts); `vram_reserve_mib=256`, rank 0 holding 99% of its
  expert codes and the other ranks all of theirs; with the DSpark drafter (`vram_reserve_mib=512`) 92% and 95%. Its GGUF IQ1_M build (the
  default 384 MiB reserve): 98% resident, 91% with DSpark. The rest were read from pinned host memory.
* GLM's GGUF (GSQ-RCO 3.0-bit) is larger on the GPUs than the 3.05 bpw EXL3 and fit only 83% resident, so it is not in the tables.

### Serving speed

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
| | NVFP4 (dense rounded to nearest) | MTP | 165 (4.09) | 187 (3.63) | 267 (5.84) | 247 (5.29) |
| | NVFP4 (dense rounded to nearest) | DFlash2 | 130 (3.31) | 167 (3.40) | 264 (5.48) | 266 (5.29) |
| | GGUF IQ3_S (GSQ-RCO) | DFlash2 | 100 (3.79) | 101 (3.44) | 165 (5.54) | 156 (5.28) |
| GLM-5.3-Flash | EXL3 3.05 bpw, stock order | MTP | 73 (3.42) | 82 (2.94) | 99 (3.94) | 86 (3.85) |
| | EXL3 3.05 bpw, stock order | DFlash2 | 67 (3.39) | 81 (3.52) | 127 (5.55) | 114 (5.30) |
| | EXL3 3.05 bpw, Volta order (earlier checkpoint) | MTP | 85 (3.50) | 97 (2.90) | 111 (3.70) | 101 (3.86) |
| | EXL3 3.05 bpw, Volta order (earlier checkpoint) | DFlash2 | 81 (3.49) | 98 (3.43) | 149 (5.30) | 141 (5.38) |
| DeepSeek-V4.1-Flash | EXL3 1.68 bpw | none | | 35 | 35 | 35 |
| | | DSpark | | 49 (2.12) | 74 (3.36) | 73 (3.37) |
| | | suffix | | 34 (1.02) | 34 (1.01) | 38 (1.17) |
| | GGUF IQ1_M | none | | 45 | 47 | 46 |
| | | DSpark | | 56 (2.22) | 89 (3.50) | 89 (3.63) |

* on agent turns MTP was ahead on Qwen3.8-27B, Flash-Next and GLM; on math and code DFlash2 on every model (28-40% on GLM); on GLM's chat set
  the two tied.
* NVFP4 was the fastest format on both Qwen models; the EXL3 rows are 4.0 bpw, which costs more to decode per weight than NVFP4's unpacking.
* Flash-Next's two EXL3 checkpoints differ in two ways at once: the tile order and the model (base against the Swift fine-tune). The base
  model is the one the DFlash2 drafter was trained on, which shows in its tokens per round.
* GLM's Volta tile order served 12-24% faster than the stock order with either speculator: its verify blocks of 4 to 16 rows took 24-35%
  less time.

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

### Step latency

| Model | Weights | 1 row | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | EXL3 4.0 bpw | 16.0 | | | 24.8 | | |
| | GGUF IQ3_S | 12.6 | 14.4 | 18.3 | 25.1 | 34.7 | 35.9 |
| Flash-Next | EXL3 4.0 bpw, Volta order | 18.4 | 19.2 | 21.9 | 25.3 | 32.4 | 34.9 |
| | EXL3 4.0 bpw, stock order | 18.4 | 20.1 | 23.8 | 28.0 | 35.5 | 38.8 |
| | NVFP4 | 9.8 | 11.0 | 13.3 | 15.9 | 23.4 | 25.3 |
| | GGUF IQ3_S | 14.6 | 15.3 | 20.5 | 29.5 | 41.6 | 47.1 |
| GLM-5.3-Flash | EXL3 3.05 bpw, stock order | 15.4 | 20.9 | 28.5 | 36.7 | 47.3 | 53.7 |
| | EXL3 3.05 bpw, Volta order (earlier checkpoint) | 15.3 | 19.0 | 21.7 | 26.5 | 31.7 | 35.1 |

The Qwen3.8-27B EXL3 row is the gate benchmark's (one row and the 8-row block only); GLM's rows are of one run each.

### Prefill

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
| | EXL3 3.05 bpw, Volta order (earlier checkpoint) | 1010-1040 |

## Reproducing

```
tests/tools/agent_replay_build.py <session.jsonl> agent_turns.json            # the agent replay from a recorded Claude Code session
tests/tools/bench_sets.py                                                     # tests/data/bench_{chat20,gsm8k24,mbpp24}.json
build/volta_server --model-dir <checkpoint> --use-speculation --set speculator=<mtp|dflash|dspark|suffix> [--draft-model-dir <drafter>] \
    --set tp=4 --set max_slots=1 --set max_seq=32768 --set max_image_tokens=0
tests/tools/agent_replay_run.py <turns.json> <out.json> --max-tokens 300      # the agent replay and the chat set; 768 for GSM8K and MBPP
curl -s localhost:8000/api/stats                                             # before and after a set: rounds and accepted drafts
BENCH_VL=2,4,8,12,16 build/bench_decode [--model glm5_next | qwen4_exp | gemma4 | DIR]   # step latency
```

Record each GPU's power limit and clocks with the numbers, and compare runs only at the same cap and clocks. Settings for serving several requests
at once are in [SERVER.md](SERVER.md#settings-for-several-requests).
