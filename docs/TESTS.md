# Test inventory

`tests/tools/gate.sh fast` runs the first group, `models` the second (`parity` runs both). The tier column orders the checks from structural (1) through parity against references (2) to end-to-end behavior (4).

## Fast: no checkpoint needed

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_trellis`, `test_gemv_int8` | 2 | yes | trellis (1 to 8 bits) and int8 GEMV kernels against FP32 references |
| `test_mhc`, `test_attention`, `test_attention_prefill`, `test_dsa`, `test_dsv4_indexer` (DeepSeek's indexer: the tensor-core scorer against the scalar kernel, selections, candidate masks and scores, with timing), `test_dsv4_fp8` (DeepSeek's FP8 projections against a double-precision product, 1 to 300 rows), `test_kv_pages`, `test_gdn`, `test_norm`, `test_rope`, `test_kv_quant`, `test_prefill_linear` | 2 | yes | module operations against CPU references |
| `test_all_reduce` | 2 | yes (4; `AR_RANKS=2` runs two) | NVLink all-reduce: single, asynchronous multi-layer, fused with the gated add and stream norm (Flash-Next), fused with the hyper-connection post-step (GLM; bit-identical to the all-reduce followed by `HyperConnection::update`) |
| `test_sparse_attn` | 2 | yes | `Attention::forward_sparse` (decode-sized sparse-list attention, head_dim 256) against an FP64 softmax over the same keys |
| `test_vulkan_comm` | 2 | yes (2-4) | the Vulkan build's host-staged all-reduce and all-reduce + RMSNorm on 2 and 4 devices, one thread per rank (skips with fewer devices) |
| `test_sampling` | 2 | no | top-k / top-p sampler |
| `test_slang_cpu`, `test_slang_kernels` | 2 | partly | Slang kernels on CPU and as generated CUDA |
| `test_ref_linear`, `test_ref_attention`, `test_vulkan_kernels`, `test_vulkan_smoke`, `test_vulkan_ref` | 2 | yes | portable reference kernels on CUDA, CPU and Vulkan |
| `test_ops_linear`, `test_ops_moe` (`--softmax` for the Qwen router), `test_ops_attention`, `test_ops_flash`, `test_ops_vision` | 2 | yes | cross-backend op goldens (dumped on CUDA, checked on CUDA, CPU, Vulkan); `test_ops_flash` covers the Flash-Next modules (gated residual, PLE, QSA, selection, sparse attention, softmax router); `test_ops_vision` the vision-tower operators |
| `test_jinja` | 2 | no | Jinja2 subset against reference Jinja2 renders of both chat templates |
| `test_image` | 2 | no | image decode, smart resize and patch layout against PIL / HF-layout references, GLM canvas sizes against exllamav3's (`tests/tools/gen_image_goldens.py`) |
| `test_output_parser` | 2 | no | reasoning / tool-call parser, chunking independence, template round trip |
| `test_trellis_layout` | 2 | yes | the trellis dequantizer against a golden decode in the build's tile order (tests/tools/trellis_layout_golden.py; run by hand, stock or Volta layout) |
| `test_tui_view` | 2 | no | dashboard frames on every tab for full, partial and missing stats at normal and tiny terminal sizes; sparklines, gap histogram, expert heat map, alerts, slot picking, pause; key handling and scroll clamps |
| `test_backend_tiers` | 1 | no | capability registry matches the declared module functions |
| `test_mma_frag` | 1 | no | on a simulated warp, against the PTX m16n8k16 / m8n8k4 layouts: the tensor-core fragment maps of the SM_80+ paths (`mma_frag.h`, `nvfp4_frag.h`) and the ROCm `mma` emulations (`mma_emu.h`) |
| `test_dflash2_blocks` | 2 | yes | DFlash2 backbone building blocks (dynamic grouped convolution, RMSNorm residual split, FP16 GEMV) against explicit FP32 CPU references |
| `test_cpu_bridge` | 2 | yes | GPU to CPU expert handoff with synthetic NVFP4 experts: routes to CPU-owned experts leave the route list, their outputs arrive in the layer output (also from a CUDA graph), layers without CPU routes do not wait |
| `test_cpu_nvfp4_vs_gpu` | 2 | yes | the CPU NVFP4 expert against the GPU NVFP4 MoE kernel on the same random weights and inputs |

## Model-backed: checkpoints from `ENGINE_MODEL_ROOT`

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_rows`, `test_moe` (`MOE_DSV41=1` with `MOE_BITS=n` or `MOE_SIG=g,u,d`: DeepSeek-V4.1 experts of one width, or of the three projection widths a mixed pack gives them), `test_nvfp4_linear`, `test_mla_latent`, `test_spec_state`, `test_dflash` | 2 | yes | layer and speculator parity with real weights (`test_mla_latent` also covers the FP8 / int8 / NVFP4 latent caches) |
| `test_tokenizer`, `test_tokenizer_file --cases` | 2 | no | token-exact against HF tokenizers (fixtures in `tests/data`) |
| `test_dsv41_vision <model dir> <golden dir>` | 2 | yes | DeepSeek-V4.1 vision tower: the engine's preprocessing of a test image matches the checkpoint reference's patches (BF16 rounding), and the tower's image rows match the reference run in FP32 (row cosine; the BF16 reference is itself noisy on a few rows). `tests/tools/dsv41/vision_golden.py <model dir> <golden dir>` writes the goldens with the reference's torch code (gate.sh makes them once under results/) |
| `test_dsv41_encoding <model dir>` | 2 | no | DeepSeek-V4.1 prompt encoding byte-exact against the checkpoint's `encoding/tests` golden cases and cases of its `test_encoding.py` |
| `test_coherence`, `test_sampling_gpu` | 4 | yes | greedy determinism, seeded sampling, speculative equivalence |
| `test_prefix_restore` (`--spec`, `--glm`, `--flash`, `ENGINE_KV_FORMAT=fp8`) | 2 | yes | a restored snapshot continues exactly like the same prefill without one |
| `test_qwen4_exp` (`--gen N`, `ENGINE_DEBUG_TRACE_LAYERS`, `ENGINE_DEBUG_QWEN4_HALT`, `ENGINE_DEBUG_TEACHER_STREAMS`) | 4 | yes | Flash-Next bring-up driver: traces compared with `tests/tools/ref_qwen4_exp_dump.py` oracle dumps by `qwen4_exp_trace_cmp.py`; greedy decode speed |
| `test_chunk_boundary` (`--cuts a,b`, `ENGINE_DSA=0`, `ENGINE_DEBUG_TRACE_LAYERS=<file>`) | 2 | yes | diagnostic, no pass/fail: GLM logit shift when the prompt is prefilled split at a non-chunk-aligned cut (aligned = 0); the trace file holds the last row after every layer and attention half |
| `test_vision` (`--glm` for GLM's tower) | 2 | yes | vision tower: tensor-core attention against the CUDA-core reference, finite deterministic embeddings (the semantic check is `test_api.sh`) |
| `test_mtp` (`--flash`, `--glm`) | 4 | yes | greedy decoding with the MTP head reproduces plain greedy decoding, short and long prompt, and accepts drafts. The logits are FP16, so a first difference is accepted when it is plain decoding's runner-up within one FP16 ulp of its pick (the top-2 logits are printed); Flash-Next is held to the first three quarters of the continuation and GLM to the first 8 tokens |
| `test_dsv41 <model dir> <ids file> [n]` | 4 | yes | DeepSeek-V4.1 prefill + greedy decode of a token-id prompt; `snap=<P>` restores a two-snapshot prefix-cache chain over an overwritten slot and exits 1 if the continuation differs (in `gate.sh models`); `adapt=<N>` rebalances the expert residency every N tokens (with `dump=`, between chunks); `trace_layers=<file>` / `dump=<file>` (with `glm_logit_rows=512`) feed `tests/tools/dsv41/trace_cmp.py` / `logits_cmp.py`, which compare with `tests/tools/dsv41/dsv41_ref.py` (DeepSeek's reference on SM70, layer by layer) |
| `test_glm_rows` | 2 | yes | GLM prefill in chunks of 1 (the M=1 graph) against wider chunks: the M=1 rerun is bit-identical, the pairwise logit spread, the greedy continuation. `tests/tools/glm_option_identity.sh` hashes its output with `side_streams` and `hc_fuse` off and on (NVFP4 dense configuration) |
| `test_prefill_equiv` | 2 | yes | Qwen prefill in 16-row chunks (fused GEMM) against 256-row chunks (dequantize + cuBLAS): final-row logits, next 32 greedy tokens, bit-reproducible rerun |
| `test_ppl` (`--glm`, `--flash`, `--chunk N`, `--file`, `--tokens`, `--calibrate`) | 4 | yes | mean NLL and perplexity of a text file, scored in chunks of N rows (at most 16: the gathered logits hold a decode batch); `--seq N` cuts the text into independent rows of N tokens; `--prompts DIR` replays the last request of each session a server dumped (`dump_requests`) with the rows shared between sessions kept out of the Hessians; `--calibrate` reads no logits, takes any chunk and, with `ENGINE_HESSIAN_DIR` (Qwen3.5-family dense, Flash-Next, GLM), writes the input Hessians of the dense projections. Not run by a gate: used to compare configurations |
| `diag_logits` | 4 | yes | diagnostic, no pass/fail: the top-2 logits that decide a continuation token of `test_mtp`'s Qwen prompt 0 in single-row decoding and in causal verify blocks built from single rows or from blocks |
| `test_longctx` | 4 | yes | needle retrieval at long context (run by hand) |
| `tests/4_server/test_api.sh` | 4 | yes | OpenAI and Anthropic APIs, tool calls, reasoning, prefix cache, image input (`VOLTA_TEST_MODEL=glm5_next` for GLM, `VOLTA_TEST_SPEC=mtp` for the MTP head) |
| `tests/4_server/test_server.sh` | 4 | yes | server integration, concurrency throughput gate |
| `test_cpu_experts <model dir> <layer> <expert>` | 2 | no | host-CPU routed experts: the AVX2 trellis GEMV and the expert pipeline against a scalar model of the EXL3 format (1 to 8 bits; part 2 checks one GLM expert and its speed) |
| `test_cpu_trellis_vs_gpu <model dir> <layer> <expert>` | 2 | yes | the CPU trellis linear against the GPU `Linear::forward` on real GLM expert weights |
| `test_dflash2_quant` | 2 | yes | trellis quantization of the DFlash2 drafter (codes, suh, svh, mul1) against its unquantized checkpoint |
| `test_mixed_rows` (`[chunk] [decode_slots] [order]`) | 2 | yes | diagnostic, no pass/fail: GLM forward of a prefill chunk plus decode rows of other slots against the same rows run as separate forwards |
| `diag_glm_tie PROMPT PREFIX WORD_A WORD_B` | 4 | yes | diagnostic: GLM logits of two candidate continuations in the single-row decode path against verify blocks of L rows (separates a near-tie from a defect) |
| `diag_logits_dump OUT.bin [--flash \| --qwen]` | 4 | yes | diagnostic: the full-vocabulary logits after a fixed 48-token prompt, to a file; `tests/tools/compare_logits.py A B` gives their cosine and top-k overlap (two builds, two checkpoints, or the Volta tile order against the stock one with `ENGINE_DEBUG_GLM_LAYERS=7`) |

## Benchmarks and profiler brackets

Not part of `gate.sh` (except `bench_decode`, `bench_accept` and `bench_prefill` in `gate.sh bench`). Single-GPU ones can be profiled with ncu; the model benches run all ranks.

| Benchmark | Measures |
|---|---|
| `bench_decode` (`--glm`, `--flash`; `BENCH_CTX`) | graph-captured decode body on real weights: a plain token, an 8-row verify block, two slots decoding together. `BENCH_PROF_A=1` brackets the 8-row block and `BENCH_PROF_B=1` the two-slot steps with `cudaProfilerStart/Stop` (default: the plain token) |
| `bench_prefill` (`BENCH_GLM=1`, `BENCH_FLASH=1`; `[tokens] [chunk ...]`) | prefill tok/s for a synthetic prompt at several chunk sizes |
| `bench_accept` | draft acceptance of speculative decoding on a spread of prompts (per-position candidates; `--nothink`) |
| `bench_spec` | DFlash2 rounds against plain greedy decoding on the same prompts (output must be token-identical) |
| `bench_mhc` | `HyperConnection::collapse_norm` per site for 1 to 8 rows (`SINK` sets the Sinkhorn iterations) |
| `bench_router` (`GLM=1` for the sigmoid router) | router logits and top-k per layer over 48 cycled weight copies |
| `bench_kda` | the KDA recurrent step (`GDN::recurrent_step_kda_rows`) per layer for 1 to 8 rows over cycled layer states; `BENCH_ITERS=1` for a profiler |
| `bench_moe_prefill` | routed-MoE prefill time per `MoE::forward` on real GLM layer-3 experts (rank-0 partition), rows routed uniformly, at several chunk sizes |
| `bench_moe_routes` (`BENCH_GLM=1`, `BENCH_ROUTES=<n>` for one route count) | the fused decode MoE kernels at a given number of local routes |
| `bench_nvfp4_shapes` (`K N ...`) | NVFP4 skinny GEMV at decode shapes: time per call and GB/s of the packed weights |
| `bench_nvfp4_moe` | the fused NVFP4 routed-expert path on Flash-Next's expert shapes (synthetic weights): parity with the chained calls, then time per layer |
| `bench_nvfp4`, `bench_trellis` | NVFP4 and trellis linear kernels |
| `bench_allreduce` | NCCL all-reduce of an activation-sized FP16 tensor next to the sequence-parallel alternative (the H3 transformer) |
| `bench_all_reduce` | the engine's own all-reduce kernels |
| `h3_text_repeat <text_encoder dir> <tokenizer dir> "prompt"` | whether the H3 text encoder gives identical hidden states on repeated encodes (fingerprints) |

## Serving benchmarks

Not part of `gate.sh`. `tests/3_bench/bench_serve.py` measures decode speed through a running server's streaming API (`--selftest` checks the plumbing against a mock server, no GPU) `tests/3_bench/bench_agent.py` replays an agent-shaped load (a main agent and parallel subagents whose prompts grow by a tool result every turn, so each request resumes from the prefix cache) and reports the time to the first token as the client sees it next to the server's split of it (queue, prefix restore, prefill kernels, checkpoint stores, other slots' steps), and `tests/3_bench/compare_1cat.sh <arm>` starts a server for one configuration, runs it and stops it.
The workloads are 1Cat-vLLM's published SM70 ones, so numbers can sit beside theirs: `fixedN` tiles one sentence to about N prompt tokens (repetitive, so speculation accepts almost everything), `natural8k` puts four real questions behind 8K tokens of the same filler;
sampling is their contract (T=1, top-p 0.95, top-k 20, seed 0, 256 tokens) or `--greedy`; `--concurrency 4` is their C4 shape. It reports TTFT, decode tok/s from the stream (prefill excluded), speculation tokens per round and acceptance from `/api/stats`,
and the GPUs' power limit, clocks and peak draw: compare runs only at the same power cap.

## MiniMax-H3: checkpoints and goldens

Not part of `gate.sh`: these need the H3 checkpoints and the goldens that `tests/tools/h3_*.py` write to `~/Documents/h3`, and several need four GPUs. Each prints a cosine or PSNR and `ok` / `FAIL`;
the expected values are listed in [video.md](video.md#tests).

| Test | GPU | Checks |
|---|---|---|
| `test_h3_dit`, `test_h3_dit_sp`, `test_h3_dit_tp` | 1 / 1-4 | the transformer forward against the golden: one GPU, sequence-parallel, tensor-parallel (with and without the turbo adapter) |
| `test_h3_attn` | 1 | the attention kernel against a CPU softmax (full, causal, key ranges, head_dim 64 / 128, odd tails) |
| `test_h3_vae`, `test_h3_vae_det`, `test_h3_vae_enc`, `test_h3_audio`, `test_h3_audio_enc` | 1 | VAE decoder, run-to-run determinism, the encoders, the audio decoder |
| `test_h3_text`, `test_h3_text_mm`, `test_h3_text_tp`, `test_h3_vision` | 1 / 1-4 | the text tower (BF16, EXL3, NVFP4) and the vision tower against the BF16 reference |
| `test_h3_caption` | 1-4 | the captioner: greedy decoding continues the same whether half of the answer is decoded or prefilled |
| `test_h3_layout`, `test_h3_ref_layout`, `test_h3_image` | no | packed layouts, resize, keyframe rows |
| `test_h3_text_identity`, `test_h3_text_layers` | 1 | diagnostics: are two text checkpoints the same model, and where do two quantizations of it drift apart |

CPU and Vulkan forwards (`test_cpu_forward`, `test_vulkan_forward`) compare a whole prompt (`--steps N`: and N greedy decode steps) against
CUDA goldens; they run by hand (slow on CPU). `--model <dir> --type <model_type>` selects the checkpoint; a checkpoint that does not fit one
GPU is dumped from a CUDA tp=4 build (`ENGINE_TP=4 build/test_cpu_forward --dump <dir> --steps 6 --model <Flash-Next dir> --type qwen4_exp`) and
checked on the CPU build, whose single rank holds all 512 experts (about 66 GiB of weights; the whole run, load + 4-token prefill + 6 decode
steps, takes 2.5 minutes on the build host's 10 cores: a 4-token prefill ~26 s, a decode step ~10 s). Flash-Next on the CPU matches the CUDA tp=4 logits (cos 0.9999992) and decodes the same 6 greedy tokens. The
Vulkan backend holds Flash-Next on four devices: `ENGINE_TP=4 build_vk/test_vulkan_forward --check <dir> --steps 4 --model <Flash-Next dir> --type qwen4_exp` against the CUDA tp=4 dump matches (cosine 0.999999, same tokens, about 1 s per decoded token); Qwen at `ENGINE_TP=1`, 2 and 4 matches the CUDA tp=1 dump (cosine 0.999999) and its FP8 KV cache too. The K8V4 and NVFP4 caches amplify rounding differences between tensor-parallel layouts (cosine 0.9998 and 0.99995 between CUDA tp=4 and CUDA tp=1 as well); the tokens still match.
