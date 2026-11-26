# Test inventory

Tests live in `tests/2_parity` (parity against references), `tests/3_bench` (benchmarks), `tests/4_coherence` (end-to-end behavior) and `tests/4_server`. The tier
column orders the checks from structural (1) through parity against references (2) to end-to-end behavior (4). Checkpoints are found by role
(`ENGINE_TEST_QWEN_DIR`, `ENGINE_TEST_GLM_DIR`, `ENGINE_TEST_FLASH_DIR`, `ENGINE_TEST_DSV41_DIR`, `ENGINE_TEST_GEMMA_DIR`, ...): `tests/models.env` names them
under `$ENGINE_MODEL_ROOT`, and `tests/local.env` (not committed) or the environment overrides them. GPU tests run under `taskset -c 0-9`.
The end-to-end tools pick their model with `--model <architecture | checkpoint dir>`: an architecture (the `model_type` of a `config.json`:
`qwen3_5`, `qwen4_exp`, `glm5_next`, `deepseek_v41`, `gemma4`) runs that family's test checkpoint, a directory runs that checkpoint (its
architecture read from its `config.json`).

## gate.sh

`tests/tools/gate.sh <mode>` runs from the build directory `GATE_BUILD` (`tests/models.env`) and writes its logs to `results/gate_<mode>_<timestamp>/`. GPU modes
serialize: one job at a time.

| Mode | Runs |
|---|---|
| `static` (the default) | no GPU: header hygiene greps (the backend abstraction), the CUDA build, `docs/CAPABILITIES.md` and `docs/CONFIG.md` against their generators (`test_backend_tiers --markdown`, `volta_server --print-config-docs`), the CPU configuration (configure, build and run `test_cpu_smoke`, build the op tests) and the Vulkan configuration (configure, build the op tests, smoke tests and `test_vulkan_forward`), `spirv-val` on every Slang SPIR-V output |
| `quick` | the CUDA build, the kernel resource fingerprint (`cuobjdump -res-usage` of `bench_decode` and `volta_server`) against `tests/baselines/fingerprint.txt`, `test_coherence --model qwen3_5` and `--model glm5_next`; for moves, renames and config work |
| `fingerprint` | saves the resource fingerprint of the current build as `tests/baselines/fingerprint.txt` |
| `sass` | the build, then an order-independent SASS fingerprint (per-function hashes) and HMMA count of `bench_decode` and `volta_server` against `tests/baselines/sass.txt` (saved if absent) |
| `fast` | the tests below that need no checkpoint |
| `models` | the tests below that load a checkpoint |
| `parity` | `fast` + `models` |
| `bench` | `bench_decode` (`--model qwen3_5` and `glm5_next`), `bench_accept`, `bench_prefill` (Qwen and GLM, 2048 tokens in 512-row chunks) and the HMMA count of `volta_server`, collected in `bench_numbers.txt` |
| `kernels` | GPU 0: the `bench_kernels` projection sweep; `tests/tools/kernel_cliffs.py` lists the cliffs and fails on cells more than 5% slower than `tests/baselines/kernels.csv` (saved if absent; a slow cell is measured twice more and keeps its fastest run) |
| `needle` | `test_longctx` needles: Qwen and GLM at 8K, greedy and speculative; GLM at 14K with `prefill_window` 4; Flash-Next at 8K and at 12K (past its QSA budget) |
| `baseline <tag>` | `parity`, `needle` and `bench`; the bench numbers saved to `results/baseline_<tag>.txt` |
| `compare <tag>` | `bench` again, diffed against `results/baseline_<tag>.txt` (flags regressions above 1%) |

Benchmark baselines depend on the GPU power cap and clocks: record one per setting.

## No checkpoint: `gate.sh fast`

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_trellis`, `test_gemv_int8` | 2 | yes | trellis (1 to 8 bits) and int8 GEMV kernels against FP32 references |
| `test_trellis_order` | 2 | yes | both EXL3 tile orders on the same random codes: the Volta reconstruction equals the permuted stock one bit for bit, and each order's GEMV / GEMM / reconstruct paths match a double-precision product from 1 to 48 rows (SM_70 builds; skips elsewhere) |
| `test_trellis_decode` | 2 | yes | the trellis pair decode (on ROCm its RDNA form) against the 32-bit multiply form, every index of the three codebooks, bit for bit |
| `test_gemm` | 2 | yes | Gemm::tn and Gemm::nn (the engine's FP16 tensor-core GEMM, weights stored [n, k] or [k, n]) against an FP64 host product: odd rows, columns and k tails, FP16 and FP32 output, beta, a strided output, every tile size and split-k |
| `test_mhc`, `test_attention`, `test_attention_prefill` (the tensor-core prefill kernels against the decode kernel: head_dim 256 and 512, sliding windows, ring caches), `test_dsa`, `test_dsv4_indexer` (DeepSeek's indexer: the tensor-core scorer against the scalar kernel, selections, candidate masks and scores, with timing), `test_dsv4_fp8` (DeepSeek's FP8 projections against a double-precision product, 1 to 300 rows), `test_fp8_moe` (FP8MoE: pseudo-random E4M3 experts with F32 block scales at GLM's widths against a host reference, 1 to 1100 rows over two passes, with the route counts), `test_dsv4_engram` (DeepSeek's Engram gate against double precision; rows inside an image span come back untouched), `test_kv_pages`, `test_kv_shard` (GLM's latent cache spread over 2 or 4 GPUs against the one-GPU pool, bit for bit, attend run on the first and the last rank; the indexer's spread pools, scores, local candidates and their merge against the one-GPU selection, rows with more and fewer than 512 pools per GPU), `test_gdn`, `test_norm`, `test_rope`, `test_kv_quant`, `test_prefill_linear` | 2 | yes | module operations against CPU references |
| `test_loop_detect` | 2 | no | the loop detector on synthetic token streams: exact and near loops found, varied text, reused phrases, transcribed prompt items and an indexed list left alone |
| `test_steer`, `test_steer_probe` | 2 | yes / no | runtime steering: the residual edit, probe-score and logit penalty kernels against an fp32 reference; the probe's axis maths on synthetic states |
| `test_router_steer` | 2 | yes | router steering: a per-slot bias on the MoE selection score removes or adds experts for that slot only, weights stay the raw scores renormalised |
| `test_all_reduce` (also with `AR_RANKS=2`) | 2 | yes (4, or 2) | NVLink all-reduce: single, asynchronous multi-layer, fused with the gated add and stream norm (Flash-Next), fused with the hyper-connection post-step (GLM; bit-identical to the all-reduce followed by `HyperConnection::update`), fused with the normed residual add (Gemma 4). `AR_ROWS` sets the rows of the first two; `AR_DUMP=<prefix>` writes every output, to compare two runs bit for bit (`ar_ll` on and off) |
| `test_sampling` | 2 | no | top-k / top-p sampler |
| `test_sparse_attn` | 2 | yes | `Attention::forward_sparse` (decode-sized sparse-list attention, head_dim 256) against an FP64 softmax over the same keys |
| `test_slang_cpu`, `test_slang_kernels` | 2 | partly | Slang kernels on CPU and as generated CUDA |
| `test_jinja` | 2 | no | Jinja2 subset against reference Jinja2 renders of the Qwen, GLM and Gemma 4 chat templates (the 31B's and the newer one of the E4B checkpoints, with a block `set`) |
| `test_image` | 2 | no | image decode, smart resize and patch layout against PIL / HF-layout references, GLM canvas sizes against exllamav3's (`tests/tools/gen_image_goldens.py`) |
| `test_output_parser` | 2 | no | reasoning / tool-call parser (Qwen, GLM, DeepSeek and Gemma 4 formats), chunking independence, template round trip |
| `test_tui_view` | 2 | no | dashboard frames on every tab for full, partial and missing stats at normal and tiny terminal sizes; sparklines, gap histogram, expert heat map, alerts, slot picking, pause; key handling and scroll clamps |
| `test_backend_tiers` | 1 | no | capability registry matches the declared module functions |
| `test_mma_frag` | 1 | no | on a simulated warp, against the PTX m16n8k16 / m8n8k4 layouts: the tensor-core fragment maps of the SM_80+ paths (`mma_frag.h`, `nvfp4_frag.h`) and the ROCm `mma` emulations (`mma_emu.h`) |
| `test_ref_linear`, `test_ref_attention` (each on the host and with `--gpu`), `test_vulkan_kernels`, `test_vulkan_smoke`, `test_vulkan_ref` | 2 | yes | portable reference kernels on CUDA, CPU and Vulkan |
| `test_vulkan_attention` | 2 | yes | (Vulkan build) the tiled attention (`fa.slang`) against `ref_attention` on the same device: head_dim 128 to 512, 1 to 8 query heads per KV head, decode rows in distinct slots or one, prefill chunks, sliding windows on ring caches, image rows, every KV format, gates; relative RMS under 5e-3 per case, and both timed |
| `test_vulkan_comm` | 2 | yes (2-4) | the Vulkan build's all-reduce and all-reduce + RMSNorm on 2 and 4 devices (on the devices through shared host memory and semaphores where the driver allows, else through the host), one thread per rank, against a host sum (skips with fewer devices) |
| `test_ops_linear`, `test_ops_moe` (`--softmax` for Flash-Next's softmax router), `test_ops_attention`, `test_ops_flash`, `test_ops_vision` | 2 | yes | cross-backend op goldens: dumped with the CUDA kernels, checked on CUDA, CPU and Vulkan (the sigmoid-routed `test_ops_moe` not on Vulkan, the softmax-routed one not on CUDA); `test_ops_flash` covers the Flash-Next modules (gated residual, PLE, QSA, selection, sparse attention, softmax router); `test_ops_vision` the vision-tower operators; `test_ops_moe` routes trellis, GGUF (Q4_K gate, IQ3_S up, Q6_K down; GGUF at cosine 0.999, each backend quantizing the activations its own way) and NVFP4 experts |

## No checkpoint, run by hand

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_cpu_smoke` (CPU build; `gate.sh static` runs it) | 2 | no | module entry points through the Slang host kernels against FP64 references, and a CUDA-only entry point aborting with its named message |
| `test_exl3_encode` (`--quick`) | 2 | yes | the EXL3 trellis encoder (`engine_quant`) on Gaussian tiles at every width (1 to 8 bits, 1.5 / 2.5 / 3.5 bpw) and the three codebooks: the error against the rate-distortion bound, and the packed codes read back by `reconstruct_slice` in both tile orders equal to the encoder's own decoded values, bit for bit; the weighted metric with weights of 1 gives the plain path's windows bit for bit and with per-row weights a lower weighted error; the open trellis errs no more than the tail-biting one; the extended trellis (the open trellis' codes with each tile's start state, and with random tile gains) decodes through `reconstruct_slice_ext` in both tile orders to the encoder's values (times the gains) bit for bit |
| `test_exl3_quantize` | 2 | yes | the EXL3 quantizer (`engine_quant`) on synthetic weights and a correlated Hessian with outlier input channels: the reported proxy error equals that of the stored tensors read back by `reconstruct_had_slice` (2 to 5 bits and 2.5 bpw, scaled and shared factors, both tile orders), the error falls with the rate, error feedback beats none, a group gives each weight its error alone; with the LDL-weighted metric the reported error still equals the stored one; re-encoding (`exl3_reencode`, 2 bits and 1.5 bpw, open trellis with tile gains, scales and gains held) toward the tensor's own decoded tiles gives back its codes and start states bit for bit, toward a target nudged by 30% moves codes at a small error, and the metric solve satisfies tr( D^T M D ) = tr( D^T G ) |
| `test_expert_hessian` | 2 | yes | the routed experts' Hessians of the quantizer tools (`src/tools/quant/expert_hessian.h`) on random rows against a double-precision host reference: the down Hessian (plain SiLU and GLM's limit), the routing-weighted (affinity) down Hessian and the affinity gate / up Hessian |
| `test_train_glm5_kda` | 2 | yes | GLM's KDA training operations (`engine/train/glm5_kda.h`): the l2 norm of q and k, the gates (beta and the per-channel log decay) and the gated delta rule with a per-key-channel decay, each forward against a double-precision host reference and each backward against central differences on sampled entries (the += outputs from a nonzero start); the recurrence over two sequences at several checkpoint chunks (partial, several sub-chunks, longer than the sequence) and at chunk 64 against chunk 8 |
| `test_train_glm5_mhc` | 2 | yes | GLM's mHC training operations (`engine/train/glm5_mhc.h`): the pre step (mix logits, pre / post gates, the Sinkhorn mix, the sublayer input norm) against a double-precision host reference and its backward (streams, mixer, base, scales, norm weight) against central differences, the paths without dpost or dcomb, the post step (in place too) and the head's stream mean |
| `test_train_glm5_moe` | 2 | yes | GLM's MoE training operations (`engine/train/glm5_moe.h`): sigmoid routing with the selection bias (8192 rows, top 8 of 288; planted ties go to the lower index), the given-ids weights, the logit gradient against the host and central differences, the chain into the router input, the SwiGLU with the clamp after the SiLU (elementwise and in the grouped-route layout, padding rows zero) |
| `test_train_grad` | 2 | yes | the training primitives (`engine_train`) on a SwiGLU MLP of three scaled linears (a core between per-channel scales): the row-weighted squared-error loss against a double-precision host forward, the gradients of the six scale vectors and of the input against central differences of the host loss (plain SiLU and GLM's limit), Adam steps on multiplicative scale parameters lowering the loss, the row gathers and scatters (FP16 and FP32 sources, routing weights) exactly, and Adam's kernels (log scales and additive deltas, FP32 and 8-bit block-wise moments) against a host reference, updated in parts against one update (bit for bit), 8-bit moments optimizing a noisy quadratic as FP32 ones do, and the 8-bit codes' round trip and stochastic rounding's bias |
| `test_train_qwen4_exp` | 2 | yes | the qwen4_exp training operations (`engine/train/qwen4_exp.h`) against double-precision host references, each forward and its backward against central differences of the host loss: RMS norms, the hyper-connection site, the gated add, the causal 4-tap convolutions, the sigmoid gates, GDN's gated norm and its gated delta rule (chunked checkpoints; a layer's shape on one of four ranks; several sequences in one call against the host recurrence per sequence and against each alone) and gates, RoPE, causal grouped-query attention, QSA past its budget (at a budget of 4 pools: the indexer's queries and pool keys against the host, the selection against the host's scores, the attention over it in query blocks against the host's masked attention and central differences), the routing top-k and its backward pieces, the PLE gate, the KL head over vocabulary chunks, and the tile gains' pieces in both tile orders (random codes with starts and gains: the decoded core against the host's rotations of the decoded tiles, `had_rows`, and the gains' gradient over a view of a tensor against the host) |
| `test_trellis_layout <golden dir> <bits> <tiles_k> <tiles_n> [mcg] [volta]` | 2 | yes | the trellis dequantizer against a golden decode in either tile order (`tests/tools/trellis_layout_golden.py`) |
| `test_dflash2_blocks` | 2 | yes | DFlash2 backbone building blocks (dynamic grouped convolution, RMSNorm residual split, FP16 GEMV) against explicit FP32 CPU references |
| `test_dflash2_backbone [small\|real\|glm]` | 2 | yes | DFlash2 drafter backbone forward against an FP32 CPU reference, on a small geometry or the Qwen or GLM drafter's attention geometry (random weights) |
| `test_cpu_bridge` | 2 | yes | GPU to CPU expert handoff with synthetic NVFP4 experts: routes to CPU-owned experts leave the route list, their outputs arrive in the layer output (also from a CUDA graph), layers without CPU routes do not wait |
| `test_cpu_nvfp4_vs_gpu` | 2 | yes | the CPU NVFP4 expert against the GPU NVFP4 MoE kernel on the same random weights and inputs |
| `test_gr_tc`, `test_gr_decode` | 2 | yes | Flash-Next gated residual: the tensor-core path (hi/lo FP16 GEMM pairs) against the FP32 row-tile kernel, a row's output independent of its position and batch; the streaming decode kernels (8 rows or fewer) against the row-tile kernels, with timing over 96 weight sets |
| `test_host_attention [cpus]` | 2 | no | the host vision towers' attention (`src/models/host_vision_ops.cpp`) against a double-precision reference on random FP16 inputs: head widths 64 and 72, a padded key count, the largest image of a tower; each call timed |
| `test_config_check <model dir> ...` | 2 | no | each checkpoint's `config.json` passes its model's layout check (the layout a pipeline implements, or the one Gemma 4 reads), and copies with one field changed are refused: another layer count, and for the fixed-layout families another hidden size or layer order (Gemma 4: a MoE block, attention biases, another activation) |
| `test_gguf_rows <model dir> <tensor base name> ...` | 2 | yes | a GGUF checkpoint's own projections on random activations at 1, 2, 8 (the q8_1 GEMV), 9, 16 and 64 rows, row 0 against the product of the weight as `LinearGguf::dequant` decodes it: the shapes of a real model without fixtures; the GEMV within 1e-2 (E4B's Q4_0 projections: 4-6e-3 at 1-8 rows, 2e-4 from 9) |
| `test_gguf_linear <fixture dir>` | 2 | yes | (CUDA, CPU and Vulkan builds) LinearGguf for every block type it reads (the K-quants, IQ1_S to IQ4_XS, Q4_0 to Q8_0, Q2_0): 1 to 8 rows (the GEMV, or for Q4_K / Q6_K from 4 / 6 rows the tensor-core kernel, whose FP16 activations stay within the bound) against the q8_1 dot product of the same blocks, 9 to 16 rows (the tile or the tensor-core kernel) and 20 and 64 rows (the dequantize-and-GEMM path) against the exact product; fixtures from `tests/tools/gguf_fixtures.py --libggml <llama.cpp build/bin> --gguf-py <llama.cpp/gguf-py> --out <dir>` (ggml's own quantizer through ctypes, a llama.cpp build that has every type), with `--m 64` for the 64-row case, and with `--k 2304` (rows of an odd number of 256-value blocks: Q6_K rows that are not 8-byte aligned); on CPU every row count meets the exact product at the GEMV bound (activations quantized at every row count), 64 rows of a q8_K type at 1.5e-2 (ik_llama.cpp's iqk GEMM requantizes each weight tile to int8 from 32 rows), and the first row of 2 to 8 rows equal to the 1-row product bit for bit (what an exact speculative verify needs); on CUDA each Q8_0 fixture also repacked (Q8_0R, as the loader repacks dense Q8_0 weights) through its dp4a GEMV at 1 to 16 rows against the q8_1 product, narrow (the fixture's rows) and wide (its rows repeated to 2304), every row of 2 to 16 rows equal to the 1-row product bit for bit |

## Checkpoints: `gate.sh models`

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_rows`, `test_moe` (`MOE_DSV41=1` with `MOE_BITS=n` or `MOE_SIG=g,u,d`: DeepSeek-V4.1 experts of one width, or of the three projection widths a mixed pack gives them; `MOE_SYNTH=1 MOE_EXT=1`: random experts with random tile starts and gains, the extended trellis through the coop kernels at 1 to 16 rows against a host product of `reconstruct_had_slice_ext`'s weights and through the persistent kernel at 24 to 200 rows against those rows one at a time, in both tile orders; a checkpoint whose experts have them runs the same checks on its own experts; `MOE_FIRST=n`: the experts from id n on), `test_nvfp4_linear`, `test_mla_latent`, `test_spec_state`, `test_dflash` | 2 | yes | layer and speculator parity with real weights (`test_mla_latent` also covers the FP8 / int8 / NVFP4 latent caches) |
| `test_glm_rows` | 2 | yes | GLM prefill in chunks of 1 (the M=1 graph) against wider chunks: the M=1 rerun is bit-identical, the pairwise logit spread, the greedy continuation. `tests/tools/glm_option_identity.sh` (not in the gate: it needs the modelopt NVFP4 GLM checkpoint) hashes its output with `side_streams` and `hc_fuse` off and on (NVFP4 dense configuration) |
| `test_prefill_equiv` | 2 | yes | Qwen prefill in 32-row chunks (the fused GEMM tile) against 256-row chunks (dequantize + the tensor-core GEMM): top-1 and final-row logits within the spread among other chunk sizes (both kernel families), bit-reproducible rerun |
| `test_tokenizer`, `test_tokenizer_file <model dir> --cases <cases.json>` (Qwen, GLM, Flash-Next, DeepSeek, Gemma 4) | 2 | no | token-exact against HF tokenizers (fixtures in `tests/data`) |
| `test_dsv41_encoding <model dir>` | 2 | no | DeepSeek-V4.1 prompt encoding byte-exact against the checkpoint's `encoding/tests` golden cases and cases of its `test_encoding.py` |
| `test_dsv41_vision <model dir> <golden dir>` | 2 | yes | DeepSeek-V4.1 vision tower: the engine's preprocessing of a test image matches the checkpoint reference's patches (BF16 rounding), and the tower's image rows match the reference run in FP32 (row cosine; the BF16 reference is itself noisy on a few rows). `tests/tools/dsv41/vision_golden.py <model dir> <golden dir>` writes the goldens with the reference's torch code (the gate makes them once under `results/`) |
| `test_dsv41 <model dir> <ids file> [n]` | 4 | yes | DeepSeek-V4.1 prefill + greedy decode of a token-id prompt; `snap=<P>` restores a two-snapshot prefix-cache chain over an overwritten slot and exits 1 if the continuation differs (the gate runs `snap=1000`); `adapt=<N>` rebalances the expert residency every N tokens (with `dump=`, between chunks); `trace_layers=<file>` / `dump=<file>` (with `glm_logit_rows=512`) feed `tests/tools/dsv41/trace_cmp.py` / `logits_cmp.py`, which compare with `tests/tools/dsv41/dsv41_ref.py` (DeepSeek's reference on SM70, layer by layer) |
| `test_coherence` (`--model qwen3_5` or `glm5_next`: that one only, `--speculative`; `ENGINE_TP=2`) | 4 | yes | greedy determinism against golden token sequences, chunked prefill + M=1 decode, all prompts in concurrent slots, DFlash2 rounds token-identical to greedy decode |
| `test_sampling_gpu` (`--model glm5_next`) | 4 | yes | temperature sampling end to end: at a vanishing temperature the sampled paths (plain decode and speculative rounds, the shard top-k merge, the sampled acceptance) reproduce greedy decode; at a real temperature the same seed reproduces the same stream and different seeds differ |
| `test_prefix_restore` (`--spec`, `--model glm5_next`, `qwen4_exp` or `gemma4` (`--draft <dir>` for Gemma's `--spec`), `ENGINE_KV_FORMAT=fp8`) | 2 | yes | a restored snapshot continues exactly like the same prefill without one, also after a round trip through the disk tier (the gate runs Qwen, `--spec`, FP8, `--glm`, `--glm` FP8 and `--flash`) |
| `test_mtp` (`--model qwen4_exp` or `glm5_next`, `--joint`, `--interleave`, `--short`) | 4 | yes | (CUDA, CPU and Vulkan builds) greedy decoding with the MTP head reproduces plain greedy decoding, short and long prompt (`--short`: the short one only), and accepts drafts. The logits are FP16, so a first difference is accepted when it is plain decoding's runner-up within one FP16 ulp of its pick (the top-2 logits are printed); Flash-Next is held to the first three quarters of the continuation and GLM to the first 8 tokens (the gate runs Qwen and `--flash`) |
| `test_vision` (`--model glm5_next` for GLM's tower, an image path) | 2 | yes | vision tower: tensor-core attention against the CUDA-core reference, finite deterministic embeddings (the semantic check is `test_api.sh`) |

## Checkpoints, run by hand

| Test | Tier | GPU | Checks |
|---|---|---|---|
| `test_longctx` (`--model glm5_next` or `qwen4_exp`, `--tokens N`, `--depth PCT`, `--spec`) | 4 | yes | needle retrieval: a filler prompt of about N tokens with a code buried at a depth, greedy or speculative decode, the answer must carry the code (`gate.sh needle` runs the cases listed above) |
| `test_train_qwen4_exp_layer` (`TRAIN_MODEL_DIR`, layers as arguments) | 2 | yes | one training layer of an EXL3 Flash-Next checkpoint against central differences of its own forward, the routing held at the unperturbed selection: the input gradient on sampled entries, the scale gradients of the projections, the shared expert and routed experts, and the weight gradients of every norm and small dense tensor of the layer (inject mixes, GDN's in_proj_a / b, A_log, dt_bias and convolution, the shared expert's gate, PLE's convolution, the router, the hyper-connection mixes; default layers 1, PLE + GDN, and 3, attention), the routed experts' tile gains by the sum of gain times gradient against the scales' (scaling every gain scales the weight as every input scale does), the router distillation term alone (a random routing teacher, no output gradient) against central differences; the forward with those tensors trainable against it with their FP16 copies (bit for bit); the backward after a kept forward against one that recomputes the sublayers; and the tensor-core products against FP32 ones (the layer's change, the input gradient, the scale gradients); then two sequences of 100 rows through one call against each alone (FP32 products: the layer's change, the input gradient, every tensor's gradients) |
| `test_train_glm5_layer` (`TRAIN_MODEL_DIR`, layers as arguments) | 2 | yes | one training layer of an EXL3 GLM-5.3-Flash checkpoint against central differences of its own forward (small random streams, the routing held at the unperturbed selection): the input gradient on sampled entries and along itself, the scale gradients of every non-expert tensor and the routed experts with the largest gradients, every norm weight's gradient (default layers 0, KDA + dense MLP; 3, MLA + MoE; 4, KDA + MoE); the backward after a kept forward against the recomputing one; the routed experts' codes staged from host memory against codes on the device (bit for bit); the tensor-core products against FP32 ones; two sequences through one call against each alone; with the last layer, the head (the streams' gradient, lm_head's scales, the final norm) against central differences of its KL to a random teacher |
| `test_train_glm5_tp` (`TRAIN_MODEL_DIR`, layers as arguments; NCCL builds, two GPUs or more) | 2 | yes | the tensor-parallel GLM training model over all visible GPUs against the single-GPU one on the same layers and inputs: the output streams, the input gradient, the attention's, dense MLP's, shared expert's and routed experts' scale gradients and the norm weights' (the shared ones summed over the ranks), then the head's KL, streams' gradient, lm_head scale gradients and the final norm's |
| `test_train_qwen4_exp_tp` (`TRAIN_MODEL_DIR`, layers as arguments; NCCL builds, two GPUs or more) | 2 | yes | the tensor-parallel training model over all visible GPUs against the single-GPU one on the same layers and inputs: the output streams, the input gradient, every tensor's scale gradients, the routed experts' tile gains' and the norm weights', small dense tensors', hyper-connection mixes' and router's gradients (the shared ones summed over the ranks), then the head's KL, streams' gradient, lm_head scale gradients and the final mixer's norm |
| `test_gemma4_vision <model dir> <golden dir> <image>` | 2 | yes | (CUDA, CPU and Vulkan builds) Gemma 4 vision tower against exllamav3 (`tests/tools/gemma4_vision_golden.py`): the engine's preprocessing gives the reference's patch grid and patches (within one 8-bit step), the tower on the reference's patches matches its embeddings by row cosine, and on the engine's own patches on average |
| `test_host_vision <model dir> <image> [max_tokens] [cpus] [mmproj dir]` | 2 | yes | the GLM-5 or DeepSeek-V4.1 vision tower on the host CPU (`vision_cpu`) against the engine's tower on the device backend, row cosine and both encode times; with an mmproj directory a second host tower on the projector GGUF's BF16 weights against the EXL3 one |
| `test_qwen4_exp` (`--gen N`, `ENGINE_DEBUG_TRACE_LAYERS`) | 4 | yes | Flash-Next bring-up driver: traces compared with `tests/tools/ref_qwen4_exp_dump.py` oracle dumps by `qwen4_exp_trace_cmp.py`; greedy decode speed |
| `test_chunk_boundary` (`--cuts a,b`, `ENGINE_DSA=0`, `ENGINE_DEBUG_TRACE_LAYERS=<file>`) | 2 | yes | diagnostic, no pass/fail: GLM logit shift when the prompt is prefilled split at a non-chunk-aligned cut (aligned = 0); the trace file holds the last row after every layer and attention half |
| `test_steer_model` (`--tokens`; run with `ENGINE_STEER` unset or `off`/`on`) | 4 | yes | runtime steering on Qwen3.8-27B: a neighbouring slot keeps its tokens while another is steered in the same batches, a vector changes the steered slot's text, the logit penalty removes the tokens it names |
| `test_sparse_head [model dir]` | 2 | yes (one GPU) | the sparse drafter head on one rank's real Flash-Next head shard: columns read through identity blocks give the full head's logits at those columns (FP16 weight rounding), the other columns sit at the floor, the subset's argmax is the full head's restricted to it; times one extraction |
| `test_spec_target_equiv [--prompt-tokens N] [--rows N]` | 4 | yes | speculation losslessness on the 27B: a multi-row verify forward against one-row steps, per row, with the logit margin where they disagree |
| `test_ppl` (`--model glm5_next`, `qwen4_exp` or `gemma4` (its text starts at `<bos>`), `--chunk N`, `--file` or `--ids` (token ids as given; `tests/tools/llama_kld_topk.cpp` turns a `llama-perplexity --kl-divergence-base` file into a `--kl-teacher` file and its ids, to score against llama.cpp on its own tokens), `--tokens`, `--calibrate`) | 4 | yes | mean NLL and perplexity of a text file, scored in chunks of N rows (at most 16 except on Flash-Next: the other models' gathered logits hold a decode batch); `--seq N` cuts the text into independent rows of N tokens; `--kl-teacher FILE` (a `--topk-out` file of the same text) adds the KL against it, the expected acceptance rate (sum of min( teacher, model ) over its top entries; with the tails' overlap as the upper bound) and the top-1 agreement over every position; `--bins B1,B2,..` also averages the NLL (and with `--kl-teacher` the KL) by position within a row or session; `--prompts DIR` replays the last request of each session a server dumped (`dump_requests`) with the rows shared between sessions kept out of the Hessians (`--stride K`: only every K-th chunk past that prefix counts, spreading `--tokens` over long sessions); `--calibrate` reads no logits, takes any chunk and, with `ENGINE_HESSIAN_DIR` (Qwen3.5-family dense, Flash-Next, GLM), writes the input Hessians of the dense projections. Used to compare configurations ([QUANTIZATION.md](QUANTIZATION.md)) |
| `test_ear --generate CORPUS` / `--reference CORPUS REF` / `--compare CORPUS REF` (`--model qwen4_exp` or `glm5_next`, `--prompts`, `--max-new`, `--top`, `--chunk`, `--seed`, `--text-out DIR`: the generated conversations as text for `test_ppl --calibrate --prompts`) | 4 | yes | expected acceptance rate of a quantized model against a reference model: the reference answers `tests/data/ear_prompts.json` (code, math, multilingual, chat, agentic) with its own sampling settings, its distributions over the generated tokens are stored (top 256 plus the remaining mass), and a candidate run over the same tokens reports per kind of text the mean of sum min( p_ref, p_quant ), the KL divergence and top-1 agreement, with the setup. Only generated tokens are scored. Used to compare quantizations |
| `test_mc [--model M] TASKS OUT [--start S] [--n N]` | 4 | yes | multiple-choice scoring: one prefill per task, the log-softmax of the last row at each single-token answer, the most likely answer as the prediction; rows in llama.cpp mc-score's format (task, predicted, correct, log-probability per answer); TASKS is a JSON list of prompt, answers and correct index |
| `diag_logits` | 4 | yes | diagnostic, no pass/fail: the top-2 logits that decide a continuation token of `test_mtp`'s Qwen prompt 0 in single-row decoding and in causal verify blocks built from single rows or from blocks |
| `diag_glm_tie PROMPT PREFIX WORD_A WORD_B` | 4 | yes | diagnostic: GLM logits of two candidate continuations in the single-row decode path against verify blocks of L rows (separates a near-tie from a defect) |
| `diag_logits_dump OUT.bin [--model M]` (default `glm5_next`) | 4 | yes | diagnostic: the full-vocabulary logits after a fixed 48-token prompt, to a file; `tests/tools/compare_logits.py A B` gives their cosine and top-k overlap (two builds, two checkpoints, or the Volta tile order against the stock one with `ENGINE_DEBUG_GLM_LAYERS=7`) |
| `test_mixed_rows` (`[chunk] [decode_slots] [order]`) | 2 | yes | diagnostic, no pass/fail: GLM forward of a prefill chunk plus decode rows of other slots against the same rows run as separate forwards |
| `test_cpu_experts <model dir> <layer> <expert>` | 2 | no | host-CPU routed experts: the AVX2 trellis GEMV and the expert pipeline against a scalar model of the EXL3 format (1 to 8 bits; part 2 checks one GLM expert and its speed) |
| `test_cpu_trellis_vs_gpu <model dir> <layer> <expert>` | 2 | yes | the CPU trellis linear against the GPU `Linear::forward` on real GLM expert weights |
| `test_dflash2_quant` | 2 | yes | trellis quantization of the DFlash2 drafter (codes, suh, svh, mul1) against its unquantized checkpoint |
| `test_glm_qkv <exl3 dir> <source dir> [layer] [rank]` | 2 | yes | GLM linear-attention qkv_proj as loaded (one rank's slice of the fused trellis tensor) against the source's q / k / v rows that rank owns |
| `test_nvfp4_quantize <exl3 dir> <nvfp4 dir> <tensor> [bits]` | 2 | yes | round-to-nearest NVFP4 from a BF16 tensor against the EXL3 version of the same matrix (cosine and relative difference at 1 and 8 rows) |
| `tests/4_server/test_api.sh` | 4 | yes | OpenAI and Anthropic APIs, tool calls, reasoning, prefix cache, image input (`VOLTA_TEST_MODEL=glm5_next` for GLM, `VOLTA_TEST_SPEC=mtp` for the MTP head) |
| `tests/4_server/test_server.sh` | 4 | yes | server integration, concurrency throughput gate |

## CPU and Vulkan forwards

`test_cpu_forward` (CPU build) and `test_vulkan_forward` (Vulkan build, the same source) run a whole prompt (`--steps N`: and N greedy decode steps) and compare the
last row's logits and the tokens with a golden that a CUDA build writes with `--dump <dir>`; `--check <dir>` checks (tier-2 tolerance plus argmax equality). They run
by hand. `--model <dir> --type <model_type>` selects the checkpoint and `ENGINE_TP` the ranks, so a checkpoint that does not fit one GPU is dumped from a CUDA tp=4
build (`ENGINE_TP=4 build/test_cpu_forward --dump <dir> --steps 6 --model <Flash-Next dir> --type qwen4_exp`) and checked on the CPU build's single rank or on four
Vulkan devices (`ENGINE_TP=4 build_vk/test_vulkan_forward --check <dir> --steps 4 --model <Flash-Next dir> --type qwen4_exp`). `--cos C` sets the cosine bound: GGUF
weights round apart by design (llama.cpp's CPU kernels quantize the activations to q8_K, its CUDA ones to q8_1, the Vulkan shaders not at all) and take
`--cos 0.999`; the K8V4 and NVFP4 caches also amplify rounding differences between tensor-parallel layouts. `--ids <int32 file>` sets the prompt (Gemma 4 needs a
chat-formatted one) and `--image <dir>` places precomputed image embeddings, so a reference from outside the engine can serve as the golden (Gemma 4: exllamav3's
full forward for EXL3, llama.cpp's own run for the QAT Q4_0 GGUF).

## MiniMax-H3: checkpoints and goldens

Not part of `gate.sh`: these take the H3 checkpoints and their goldens (safetensors files written by `tests/tools/h3_*.py`) on the command line, and several need four
GPUs. Each prints a cosine or PSNR and `ok` / `FAIL`; the expected values are listed in [VIDEO.md](VIDEO.md#tests).

| Test | GPU | Checks |
|---|---|---|
| `test_h3_dit`, `test_h3_dit_sp`, `test_h3_dit_tp` | 1 / 1-4 | the transformer forward against the golden: one GPU, sequence-parallel, tensor-parallel (with and without the turbo adapter) |
| `test_h3_attn` | 1 | the attention kernel against a CPU softmax (full, causal, key ranges, head_dim 64 / 128, odd tails) |
| `test_h3_vae`, `test_h3_vae_det`, `test_h3_vae_enc`, `test_h3_audio`, `test_h3_audio_enc` | 1 | VAE decoder, run-to-run determinism, the encoders, the audio decoder |
| `test_h3_text`, `test_h3_text_mm`, `test_h3_text_tp`, `test_h3_vision` | 1 / 1-4 | the text tower (BF16, EXL3, NVFP4) and the vision tower against the BF16 reference |
| `test_h3_caption` | 1-4 | the captioner: greedy decoding continues the same whether half of the answer is decoded or prefilled |
| `test_h3_layout`, `test_h3_ref_layout`, `test_h3_image` | no | packed layouts, resize, keyframe rows |
| `test_h3_text_identity`, `test_h3_text_layers` | 1 | diagnostics: are two text checkpoints the same model, and where do two quantizations of it drift apart |

## Benchmarks and profiler brackets

Not part of `gate.sh` except `bench_decode`, `bench_accept` and `bench_prefill` (`gate.sh bench`) and `bench_kernels` (`gate.sh kernels`). Single-GPU ones can be
profiled with ncu; the model benches run all ranks.

| Benchmark | Measures |
|---|---|
| `bench_decode` (`--model M`, repeatable, or `--all`; `BENCH_CTX`, `BENCH_VL`) | graph-captured decode body on real weights: a plain token, an 8-row verify block (`BENCH_VL=2,4,...` also times causal verify blocks of those widths), two slots decoding together. `BENCH_PROF_A=1` brackets the verify block and `BENCH_PROF_B=1` the two-slot steps with `cudaProfilerStart/Stop` (default: the plain token) |
| `bench_kernels` (`--only <shape>`, `--rows 1,2,...`, `--reps N`) | `Linear::forward` on the served models' per-rank projection shapes in every weight format (EXL3 3/4/5 bit in both tile orders, NVFP4, GGUF Q4_K / Q6_K / IQ3_S) at 1 to 16 and 32 to 512 rows, CSV; `tests/tools/kernel_cliffs.py` flags non-monotonic cells, steep steps and Volta-order cells slower than stock, and with `--baseline` the cells slower than an earlier sweep |
| `tests/tools/kernel_sweep.sh OUT.csv ARMS` | whole-model `bench_decode` step cost at 1 to 16 verify rows and contexts 8 / 8K / 32K per model and weight format, CSV for `kernel_cliffs.py` |
| `bench_trellis_order` (`--rows 1,2,4,8,16`, `--reps 5`) | `LinearTrellis::forward` in both EXL3 tile orders on the same shapes and row counts, GPU 0, weight copies rotated past L2, the orders alternating per repeat |
| `bench_prefill` (`BENCH_GLM=1`, `BENCH_FLASH=1`; `[tokens] [chunk ...]`) | prefill tok/s for a synthetic prompt at several chunk sizes |
| `bench_accept` | draft acceptance of speculative decoding on a spread of prompts (per-position candidates; `--nothink`; `--depths`; `--type <model> --model <dir> --draft <dir>` for another target and drafter) |
| `bench_spec` | DFlash2 rounds against plain greedy decoding on the same prompts (output must be token-identical) |
| `bench_mhc` | `HyperConnection::collapse_norm` per site for 1 to 8 rows and a hash of the outputs, bit for bit (`SINK` sets the Sinkhorn iterations; `DS=1` DeepSeek-V4.1's lagged form, `collapse_lagged_norm`) |
| `bench_f16_gemv` (`[N K ...]`) | the plain FP16 GEMV (FP16 linears, the drafter's projections, GLM's indexer) at 1 to 16 rows over cycled weight copies: time per call, GB/s and a hash of the outputs |
| `bench_router` (`GLM=1` for the sigmoid router, `DS=1` for DeepSeek-V4.1's) | router logits and top-k per layer over 48 cycled weight copies, and a hash of every row's experts and weights |
| `bench_dsv4_attend` (`[context rows heads nsel]`) | DeepSeek-V4.1 decode attention (`DSV4::attend` with its split-KV combine) on one GPU, time per layer over 40 cycled layers' caches; `nsel` 0 runs a window-only layer; `BENCH_DSPARK=1` times the drafter's `DSV4::dspark_attend` instead. Layer 0's output is compared with one block a row (no workspace) |
| `bench_kda` | the KDA recurrent step (`GDN::recurrent_step_kda_rows`) per layer for 1 to 8 rows over cycled layer states; `BENCH_ITERS=1` for a profiler |
| `bench_moe_prefill` | routed-MoE prefill time per `MoE::forward` on real GLM layer-3 experts (rank-0 partition), rows routed uniformly, at several chunk sizes |
| `bench_moe_routes` (`BENCH_GLM=1`, `BENCH_ROUTES=<n>` for one route count) | the fused decode MoE kernels at a given number of local routes |
| `bench_gguf_shapes` (`TYPE ... [-- K N ...]`, `BENCH_DEQUANT=1`) | (CUDA and Vulkan builds) GGUF linear at 1 to 16 rows on the 27B's per-rank shapes, in the layout the loader leaves a dense weight in (CUDA Q8_0: Q8_0R; `ENGINE_DEBUG_GGUF_Q8R=0` the file's): time per call and GB/s of the blocks; `BENCH_DEQUANT=1` also times the whole-weight dequantization (file layouts) |
| `bench_gguf_moe` (`[TYPE] [hidden inter global local top_k]`) | GGUF routed experts (MoEGguf) at 1 to 16 rows, routes uniform over the experts, this rank holding the first `local`: time per layer and GB/s of the experts touched, cold L2; default GLM-5.3-Flash's Q2_K experts expert-parallel at TP=4. `BENCH_ROWS=a,b,..` other row counts (the tile path above 64), `BENCH_RANDOM=1` pseudo-random blocks, `BENCH_DUMP=<prefix>` random inputs and each row count's output to a file, to compare two runs (a kernel path A/B), `BENCH_REUSE=p` each row keeps each of the previous row's experts with probability p (a draft's consecutive tokens) |
| `bench_fp8_moe` (`[hidden inter global local top_k]`) | FP8 routed experts (FP8MoE) at prefill row counts (default 256, 1024, 4096; `BENCH_ROWS=a,b,..`), routes uniform over the experts, this rank holding the first `local`: time per layer and TFLOP/s of the routed products; default GLM-5.3-Flash expert-parallel at TP=4 |
| `bench_gemm` (`[--nn] [M ...] [-- N K ...]`) | Gemm::tn / Gemm::nn against cuBLAS on the language models' projection shapes: time per call and TFLOPS of each, cold L2 (graph replay over distinct weights) |
| `bench_trellis_shapes` (`--code C`, `--volta`, `--rows 1,4,...`, `--reps N`, `-- K N ...`) | (CUDA and Vulkan builds) the EXL3 linear at a model's projection shapes (default Gemma 4 31B's, 4 bits): time per call over weight copies past L2, and the GB/s of the packed tiles |
| `bench_tts_ops` (`[L] [reps]`) | (CUDA and Vulkan builds) the TTS operators at the v2.0 export's shapes: the backbone's GEMMs over a pass of L positions, one row's attention, the codec's convolutions; time per call and TFLOP/s |
| `bench_nvfp4_shapes` (`K N ...`) | NVFP4 skinny GEMV at decode shapes: time per call and GB/s of the packed weights |
| `bench_nvfp4_moe` | the fused NVFP4 routed-expert path on Flash-Next's expert shapes (synthetic weights): parity with the chained calls, then time per layer |
| `bench_nvfp4`, `bench_trellis` | NVFP4 and trellis linear kernels |
| `bench_sparse_head [model dir]` | one rank's real Flash-Next head shard (EXL3, 6 bits) through `Linear::forward` for 1 to 4 draft rows against an FP16 GEMV over a 4K to 32K column vocabulary subset (what the sparse drafter head saves per draft step; one GPU) |
| `bench_shapes` | GEMV against GEMM time per Qwen TP4 per-rank shape (GPU 0, random trellis data) |
| `bench_expert_heat` | GLM routed-expert skew: the share of a prompt's routes a resident subset of each layer's experts would serve, from its own or the other prompts' counts |
| `bench_all_reduce` | the engine's own all-reduce kernels, time per call: `BENCH_AR=rms` (the default, with the residual and RMSNorm), `plain`, `hc` (the hyper-connection post-step), `gated` (Flash-Next's gated add and norm), `normed` (Gemma 4's normed add); `BENCH_ROWS` rows |
| `bench_allreduce` (NCCL builds) | NCCL all-reduce of an activation-sized FP16 tensor next to the sequence-parallel alternative (the H3 transformer) |
| `bench_hop [src gpu] [dst gpu] [repeats]` | the hop between two GPUs by message size: a direct peer copy and a bounce through pinned host memory, as the latency of one synchronized copy and as pipelined bandwidth |
| `p2p_check [MB] [repeats]` | peer-to-peer copy integrity and bandwidth between every GPU pair |
| `bench_h3_attn [q_rows k_rows heads causal]` | the H3 attention kernel at the transformer's sequence-parallel shape |
| `bench_h3_dit <checkpoint> [rows layers repeats]` | the H3 transformer forward at a production-sized packed sequence (synthetic inputs, real weights, one GPU) |
| `bench_h3_gemm [rows] [vae]` | a cuBLAS algorithm sweep for the H3 transformer's four GEMM shapes at one rank's rows (`vae`: the video decoder's shapes) |
| `h3_text_repeat <text_encoder dir> <tokenizer dir> "prompt"` | whether the H3 text encoder gives identical hidden states on repeated encodes (fingerprints) |
| `tests/tools/pcie_bw.cu` (standalone, `nvcc -O3`) | host to device and device to host bandwidth per GPU, alone and all at once (a shared uplink shows), and a kernel reading pinned memory in place |
| `tests/tools/ram_bw.c` (standalone, `cc -O3 -march=native -fopenmp`; `[GiB per array]`) | host RAM read / write / triad bandwidth per NUMA node (one thread, every core, every hardware thread) and for all nodes together, threads pinned and pages node-local |

## Serving benchmarks

Not part of `gate.sh`; they run against a running server.

* `tests/3_bench/bench_serve.py` measures decode speed through the streaming API. Workloads: `fixedN` tiles one sentence to about N prompt tokens (repetitive,
  so speculation accepts almost everything), `natural8k` puts four questions behind 8K tokens of the same filler; sampling is T=1, top-p 0.95, top-k 20, seed 0,
  256 tokens, or `--greedy`; `--concurrency N` runs N streams at once. It reports TTFT, decode tok/s from the stream (prefill excluded), speculation tokens per
  round and acceptance from `/api/stats`, and the GPUs' power limit, clocks and peak draw: compare runs only at the same power cap. `--selftest` checks the
  plumbing against a mock server, no GPU.
* `tests/3_bench/bench_agent.py` replays an agent-shaped load (a main agent and parallel subagents whose prompts grow by a tool result every turn, so each request
  resumes from the prefix cache) and reports the time to the first token as the client sees it next to the server's split of it (queue, prefix restore, prefill
  kernels, checkpoint stores, other slots' steps).
* `tests/tools/agent_replay_build.py SESSION.jsonl OUT.json` builds replay turns from a recorded Claude Code session (each turn's prompt the rendered history
  before it, segments sharing one history start so the prefix cache serves their later turns); `tests/tools/agent_replay_run.py TURNS.json OUT.json` runs them
  one at a time, greedy, and records per turn the time to first token, decode tok/s, speculation and suffix rounds from `/api/stats` and the generated text.

## Agentic benchmark: `tests/agentic`

`tests/agentic/bench.py` runs the Claude Code CLI against any server that speaks the Anthropic messages API (`volta_server` serves `/v1/messages`) and scores it on
small software tasks. It needs only a base URL and a model name; nothing in it is specific to this engine. When the server reports its window (`context_length` in
`GET /v1/models`) the client gets it as `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, so it compacts within the window instead of the one it assumes for an unknown model (1M since Claude Code v2.1.323, 200K before).

```
tests/agentic/bench.py selftest [--tier T] [--only ID,ID] [--tasks-dir DIR]
tests/agentic/bench.py run --base-url http://localhost:8000 --model <name> --out results/agentic/<run> [--tasks-dir DIR] [--tier T] [--only ID,ID] \
                           [--repeats N] [--parallel N] [--thinking default|off|low|medium|high[,...]] [--max-turns N] [--timeout S] [--api-key KEY]
tests/agentic/bench.py summary results/agentic/<run>
```

A task is a directory `tests/agentic/tasks/<id>/`:

| file | content |
|---|---|
| `task.json` | `id`, `tier`, `prompt`, `tags` |
| `repo/` | the starting files; each run gets a fresh copy |
| `verify.sh` | runs in the work directory after the client finished (files of `hidden/` are copied in first); exit 0 is a pass |
| `solution.sh` | a reference solution; `selftest` checks that the verifier fails on `repo/` and passes after it |

The tasks in the repository (tier `quick`) cover a bug fix, a feature with tests, a rename across files, a shell fix, a read-only question and a config edit.
`--tier` selects the tasks whose `task.json` carries that tier, `--only` names them.

`tests/agentic/gen_tasks.py --out DIR [--per-family N] [--seed S] [--families a,b] [--tier quick|standard|hard]` generates parametrized instances of 13 families:
`py-bugfix`, `py-feature`, `c-fix`, `js-fix`, `sql-query` (questions over a generated SQLite database), `git-conflict` (merge conflicts), `log-analysis` (questions
over a long access log), `sh-text` (shell text processing), `prose` (writing tasks over a local document, loose verifiers), and the harder `py-multibug` (two bugs in
different modules of a small package), `py-algo` (a function with edge cases in its spec, hidden tests), `data-reconcile` (matching two CSV files by stated rules)
and `log-multi` (joining two logs with retried requests). `--families` picks some (default all, `--per-family` 5 each), `--tier` is the tier written into each
`task.json` (default `standard`). Expected values are computed from a reference implementation at generation time, so an instance is a pure function of its family,
index and seed, and `bench.py selftest --tasks-dir DIR` checks every verifier. `--tasks-dir` also selects them for `run`; `--parallel N` runs N clients at once
(match the server's slots); `--thinking` takes a list that cycles over the tasks.

Every run, and every verifier, executes under bubblewrap: the system is read-only, `/home` and `/mnt` are empty, the environment is cleared, and the task's work
directory is the only writable path. The client run has network access (it must reach the server); verifiers do not. The client's own configuration lives in a
throwaway directory, so the user's Claude Code settings are never read or written.

A run directory holds `meta.json` (client version, server URL, model, thinking level, limits, repository revision), `results.jsonl` (per run: pass, wall time,
turns, tool calls and errors, input and output tokens, thinking and answer characters) and, per task and repetition, the client's event stream (`events.jsonl`) and
the final work directory. `summary` prints pass rate, median wall time, mean turns, tool calls and output tokens, and the share of thinking characters among thinking
plus answer characters. Runs are not deterministic (batching and speculation change the sampled text), so use `--repeats` and compare pass rates, not single
outcomes. `--thinking` sets `MAX_THINKING_TOKENS` for the client (off 0, low 4000, medium 10000, high 31999); `default` leaves it unset.

`tests/agentic/dart_replay.py --dump DIR --url URL --out OUT.jsonl` replays the turns of Claude Code sessions recorded with thinking on (a server's `dump_requests`
directory) with thinking disabled and compares each draft's action (its tool calls, normalised per tool) with the reasoned turn's; `--report OUT.jsonl` summarizes.
