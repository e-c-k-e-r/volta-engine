# Performance

Reference numbers for the three model families on the target machine, how they were taken, and where a decode token spends its time. Speed depends on the GPU power cap and clocks, so every table says what it was measured at.

## Setup

* 4x Tesla V100-SXM2-32GB, power limit 250 W (the cards still peak at 255 to 270 W in bursts), SM clock free up to 1530 MHz (they run at 1290 to 1530 MHz)
* CUDA 12.8, `taskset -c 0-9`, tensor parallel over all four GPUs, CUDA graphs on, prefix cache off, one request at a time
* speculation on where a model has it: DFlash2 drafter (Qwen, GLM) or the checkpoint's MTP head (Qwen, Flash-Next, GLM)
* weights
  * Qwen3.8-27B: EXL3 4.0 bpw (`Qwen3.8-27B-exl3`), or the NVFP4 checkpoint (`Qwen3.8-27B-QUASAR-NVFP4`) with the BF16 DFlash2 drafter
  * Flash-Next: the normal NVFP4 checkpoint on its own (`configs/flash_next_nvfp4_only.json`: routed experts as shipped, every dense group and the FP8 MTP head quantized to NVFP4 at load, the FP8 n-gram table read in place), or the mix of [FLASH_NEXT.md](FLASH_NEXT.md) (NVFP4 experts and dense groups with the Swift EXL3 4.0 bpw conversion's trunk, head and n-gram table)
  * GLM-5.3-Flash: EXL3 3.05 bpw experts with the calibrated NVFP4 dense groups `kda,mla,idx,head,dense,shared`, all experts resident, 12K window; the MTP head comes from the NVFP4 checkpoint (`configs/glm_mtp.json`)

## Serving speed

`tests/3_bench/bench_serve.py` talks to a running server through the streaming chat API and measures decode speed from the first token to the last (prefill excluded). Sampling is 1Cat-vLLM's release contract (T=1, top-p 0.95, top-k 20, seed 0), 256 new tokens, three runs per case.

The natural prompts are four real questions (two in Chinese) behind about 8K tokens of filler, the same ones as 1Cat-vLLM's `benchmark_sm70_mtp4_round.py`. Speed depends on how well the drafter predicts the answer, so the table lists all four.

| Model, speculator | Natural 8K, tok/s (4 prompts) | Mean | Tokens per round |
|---|---|---|---|
| Qwen3.8-27B NVFP4, DFlash2 | 252, 263, 182, 81 | 195 | 4.1 to 6.0 |
| Qwen3.8-27B EXL3, DFlash2 | 163, 169, 138, 93 | 141 | 3.1 to 5.8 |
| Qwen3.8-27B EXL3, MTP | 169, 124, 116, 84 | 123 | 2.7 to 6.0 |
| Flash-Next (NVFP4 checkpoint only), MTP depth 7 | 235, 210, 181, 136 | 191 | 3.4 to 5.7 |
| Flash-Next (NVFP4 experts and dense, Swift trunk and head), MTP depth 7 | 213, 186, 179, 123 | 175 | 3.0 to 5.7 |
| Flash-Next (the same), MTP depth 4 | 199, 172, 176, 127 | 169 | 2.9 to 4.4 |
| GLM-5.3-Flash, DFlash2 (EXL3 drafter) | 116, 109, 108, 72 | 101 | 3.1 to 5.0 |
| GLM-5.3-Flash, MTP depth 7 (NVFP4 checkpoint's head) | 109, 102, 106, 87 | 101 | 2.6 to 4.0 |
| GLM-5.3-Flash, no speculation | 64, 64, 64, 64 | 64 | 1 |

* repetitive output (a long text the model is told to copy) is the best case for a drafter: Qwen NVFP4 + DFlash2 reaches 316 tok/s at 32K context, 7.8 tokens per round, 97% accepted; Qwen EXL3 + MTP reaches 212 to 225 tok/s; Flash-Next MTP depth 4 233 tok/s. These runs end early now and then, and the output (and so the speed) depends on the model's wording, so they are listed as peaks and not as typical speed
* the DFlash2 drafter format does not matter: the BF16 file and the EXL3 5.0 bpw conversion run the same speed (natural 8K, four prompts: 157, 154, 143, 94 tok/s with BF16 against 163, 169, 138, 93)
* GLM-5.3-Flash at 256K context with two slots (4-bit latent cache, 95% of the experts resident, drafter): a 32K prompt takes 68 s to prefill and decodes at 53 tok/s (T=1, 7.1 tokens per round); the 5% of experts that live in host memory are read over PCIe, which the four GPUs share at 12 GB/s in total (`tests/tools/pcie_bw.cu`). With 85% resident and an FP8 cache the same prompt took 123 s and decoded at 16 tok/s ([GETTING_STARTED.md](GETTING_STARTED.md))
* a sampling temperature above zero lowers the tokens per round (the drafter proposes its most likely chain), not the cost of a step: plain decode runs the same speed greedy and sampled

### Prefill

Time to first token for a cold prompt (no prefix cache), which is the prefill time plus one step.

| Model | 4K | 8K | 32K |
|---|---|---|---|
| Qwen3.8-27B | 1.7 s | 3.4 s | 15.2 to 15.9 s |
| Flash-Next | | 5.0 s | |
| GLM-5.3-Flash | 5.2 s | 10.0 s | 68 s (256K configuration) |

## Step latency

`build/bench_decode` replays the graph-captured decode body on real weights: a plain token, a block of 8 rows (what a speculative round verifies), and two slots decoding together. The context is 8 tokens or 8192 tokens.

| Model | Plain token, 8 / 8192 ctx | 8-row verify, 8 / 8192 ctx | 2 slots, per step |
|---|---|---|---|
| Qwen3.8-27B EXL3 | 15.9 / 16.5 ms (63 / 61 tok/s) | 24.9 / 25.6 ms (1.6x a token) | 20.6 ms |
| Qwen3.8-27B NVFP4 | 11.7 / 12.0 ms (86 / 84 tok/s) | 13.4 / 13.8 ms (1.15x) | 12.0 ms |
| GLM-5.3-Flash, NVFP4 dense | 15.0 / 15.5 ms (67 / 65 tok/s) | 31.0 / 31.6 ms (2.0x) | 18.0 ms |
| Flash-Next, NVFP4 experts and dense | 15.5 / 14.2 ms (65 / 70 tok/s) | 18.4 / 19.1 ms (1.2 to 1.3x) | 16.8 / 15.3 ms |

A verify block costs little over a plain token on the NVFP4 models, which is what lets a speculative round pay off; on GLM it costs twice a token, because 8 rows touch many more experts than one.

## Where the time goes

`nvprof` over the same steps at 8K context, GPU 0. Shares of the kernel time; the kernel time is close to the step time (on Flash-Next the GPU is busy 87% of the span of a step).

Plain token:

| | Qwen EXL3 | Qwen NVFP4 | GLM | Flash-Next |
|---|---|---|---|---|
| Kernel time per step | 17.4 ms | 12.2 ms | 16.0 ms | 12.7 ms |
| Projections (dense weights) | 69% (12.1 ms) | 54% (6.6 ms) | 25% (4.1 ms) | 27% (3.4 ms) |
| MoE experts | | | 36% (5.8 ms) | 21% (2.6 ms) |
| Gated residual (Flash-Next) | | | | 28% (3.6 ms) |
| All-reduce | 10% (1.8 ms) | 14% (1.7 ms) | 7% (1.1 ms) | 8% (1.1 ms) |
| Attention (full, latent, sparse, selection) | 6% (1.0 ms) | 15% (1.8 ms) | 7% (1.2 ms) | 8% (1.0 ms) |
| Linear attention (GDN, KDA) | 7% (1.2 ms) | 9% (1.2 ms) | 8% (1.3 ms) | 6% (0.8 ms) |
| Norms, rope, activation, embedding, sampling | 8% (1.3 ms) | 8% (1.0 ms) | 16% (2.6 ms) | 2% (0.3 ms) |

8-row verify block:

| | Qwen EXL3 | Qwen NVFP4 | GLM | Flash-Next |
|---|---|---|---|---|
| Kernel time per block | 25.4 ms | 13.8 ms | 31.4 ms | 18.7 ms |
| Projections | 78% (19.7 ms) | 52% (7.1 ms) | 14% (4.5 ms) | 19% (3.5 ms) |
| MoE experts | | | 58% (18.1 ms) | 25% (4.7 ms) |
| Gated residual | | | | 33% (6.1 ms) |
| All-reduce | 8% (1.9 ms) | 14% (1.9 ms) | 4% (1.2 ms) | 7% (1.4 ms) |
| Attention | 5% (1.2 ms) | 15% (2.1 ms) | 7% (2.2 ms) | 8% (1.6 ms) |
| Linear attention | 6% (1.5 ms) | 11% (1.5 ms) | 7% (2.3 ms) | 6% (1.1 ms) |

* Qwen EXL3: the trellis decode of the dense weights is the cost, and eight rows share one decode of each weight; NVFP4 weights are cheaper to unpack, so the same step takes 30% less
* GLM: the routed experts dominate and grow with the rows; this is why speculation helps GLM less than the others
* Flash-Next: the four gated-residual kernels per sublayer cost as much as the projections; fusing them is on the open list of [FLASH_NEXT.md](FLASH_NEXT.md)
* the all-reduce is 1.1 to 1.9 ms a step on every model (about 11 us per call over NVLink)

## Against 1Cat-vLLM

1Cat-vLLM publishes SM70 numbers for the same models. The conditions differ, so these are for orientation, not a contest: their Qwen release numbers are from 32 GB SXM2 cards (their roughly 260 tok/s headline demo from 16 GB ones), their Flash-Next measurements state 300 W limits and an application clock of 1290 MHz, and ours run at 250 W with clocks up to 1530 MHz. Their fixed prompts are raw token ids of one repeated sentence, which a drafter predicts almost perfectly; the chat API cannot send those, so the "copy this text" prompt above stands in.

| | 1Cat-vLLM (published) | This engine |
|---|---|---|
| Qwen3.8-27B NVFP4 + DFlash2, one request | 229 tok/s at 4K, 201 at 32K, 171 at 128K (T=1, top-p 0.95, repetitive prompt) | 316 tok/s at 32K (copy prompt); natural prompts 81 to 263, mean 195 |
| Flash-Next MTP4, 8K | 17.4 to 18.5 ms per round, 4.9 tokens per round on the repetitive prompt (about 264 tok/s); natural prompts 3.0 tokens per round, 117 tok/s (NVFP4 target) | MTP depth 4: 233 tok/s on the copy prompt (5.0 tokens per round); natural prompts mean 169 (Swift trunk and head); depth 7: 175 to 191 |
| GLM-5.3-Flash NVFP4, no speculation, 1K prompt | 53 tok/s on 8 V100s (TP4 and PP2, an earlier measurement) | 64 tok/s on 4 V100s (EXL3 3.05 bpw experts) |

The Flash-Next rows of the first kind run the normal model end to end, so acceptance is comparable with theirs; the head from the NVFP4 checkpoint drafts as well as the conversion's own (the second and third rows).

## Reproducing

```
tests/3_bench/compare_1cat.sh <arm> --cases fixed4k,fixed32k,natural8k --repeats 3   # arms: see the header of the script
build/bench_decode [--glm | --flash]                                                  # BENCH_CTX=8192 for a real context
nvprof --profile-from-start off --print-gpu-summary build/bench_decode ...            # kernel shares
```

The serving script records each GPU's power limit and clocks and the highest power draw seen; compare runs only at the same cap and clocks.
