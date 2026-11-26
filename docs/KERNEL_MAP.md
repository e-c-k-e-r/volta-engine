# Kernel map

Kernel family -> file -> what verifies it. The tier of each module entry point (Portable, PortablePending, CudaSm70) is in [CAPABILITIES.md](CAPABILITIES.md);
the tests are described in [TESTS.md](TESTS.md). `gate.sh sass` fingerprints every kernel (any change to a kernel changes its per-function SASS hash).

## Trellis (EXL3) weights: `src/kernels/cuda/trellis/`

| Kernel family | Files | Verified by |
|---|---|---|
| GEMM (prefill and mid-M, HMMA m8n8k4) | `exl3_gemm.cu`, `exl3_gemm_kernel.cuh`, `exl3_gemm_inner.cuh`, `exl3_kernel_map.*`, `exl3_instances.cu` | `test_trellis`, `test_prefill_linear`, `bench_shapes` |
| GEMV / skinny M (M <= 8) | `exl3_gemv.cu`, `exl3_gemv_kernel.cuh`, `exl3_mgemm_gv_kernel.cuh` | `test_trellis`, `bench_decode` |
| int8 dp4a GEMV (M <= 2) | `exl3_gemv_int8.cu`, `exl3_gemv_int8_kernel.cuh` | `test_gemv_int8` |
| Dequantize for large M (recon + the FP16 GEMM) | `reconstruct.cu`, `exl3_dq.cuh`, `codebook.cuh` | `test_trellis`, `test_ops_linear`, `test_trellis_layout` |
| Extended trellis decode (tile start states, tile gains) | `exl3_dq_ext.cuh` | `test_exl3_encode`, `test_moe` (`MOE_SYNTH=1 MOE_EXT=1`) |
| Hadamard rotation | `hadamard.cu`, `hadamard_inner.cuh` | `test_trellis` |
| Fused MoE (persistent and cooperative, int8 tile) | `exl3_moe_kernel.cuh`, `exl3_moe_launch.cu`, `exl3_moe_coop.cu`, `exl3_moe_common.cuh` | `test_moe`, `test_ops_moe`, `bench_moe_prefill`, `bench_moe_routes` |
| Volta tile order copies (every unit above compiled a second time in namespace `exl3_vl` on SM_70 builds) | `exl3_vl.cuh`, `exl3_vl_prelude.cuh`, `exl3_vl_unit.cu.in` | `test_trellis_order`, `bench_trellis_order` |
| Workspace state | `trellis_context.cu/.h` | used by every kernel above |
| Quantizer: trellis encoder (Viterbi over a tile's ring, integer and half-integer rates) and the GPU quantization steps (`engine_quant`) | `quant/exl3_encode*.cu`, `quant/exl3_encode_kernel.cuh`, `quant/exl3_quantize.cu` | `test_exl3_encode`, `test_exl3_quantize` |

## NVFP4: `src/kernels/cuda/nvfp4/`

| File | Kernels | Verified by |
|---|---|---|
| `skinny_kernels.cu`, `nvfp4_common.cuh`, `nvfp4_frag.h` | skinny (M <= 32) and fused-GEMM kernels with FP16-rounded dequant, plus the `LinearNVFP4` module entry (CPU has a mirrored implementation) | `test_nvfp4_linear`, `test_ops_linear`, `bench_nvfp4_shapes`, `test_mma_frag` |
| `moe_kernels.cu` | fused routed experts: dispatch by expert, gate / up / SiLU, down + ordered weighted sum; up to 8 rows per pass of an expert's weights | `bench_nvfp4_moe` (parity with the chained calls), `test_cpu_nvfp4_vs_gpu` |

## GGUF weights

| Files | Kernels | Verified by |
|---|---|---|
| `src/kernels/cuda/gguf/gguf_linear.cu`, `gguf_common.cuh`, `ggml/` (llama.cpp's vendored CUDA code) | `LinearGguf`: q8_1 activations (each block's d times the sum of its quantized values, as llama.cpp's quantizer stores it, which the offset types' `vec_dot` cancels exactly) and a GEMV per block type on llama.cpp's `vec_dot` up to 8 rows, a tile decoded once into shared memory and multiplied on the tensor cores for 9 to 16 rows, dequantize + the FP16 GEMM above; the Q8_0R repack of dense Q8_0 weights | `test_gguf_linear`, `bench_gguf_shapes`, `bench_kernels` |
| `src/kernels/cuda/gguf/gguf_moe.cu` | `MoEGguf`: routes grouped by expert, the expert's weight rows read once for an item's routes; prefill chunks on decoded tensor-core tiles | `test_ops_moe`, `bench_gguf_moe` |
| `src/kernels/cuda/fp8/fp8_moe.cu` | `FP8MoE`: routed experts of an original FP8 checkpoint (E4M3, F32 128 x 128 block scales): tiles of 32 rows decoded to FP16 per 128-wide step, WMMA, the step's sum scaled by its block scale; gate and up with SwiGLU in one pass, down into FP32 partials | `test_fp8_moe`, `bench_fp8_moe` |
| `src/kernels/cuda/moe_items.cuh` | the route grouping (work items per expert) and the per-token combine shared by `MoEGguf` and `FP8MoE` | `test_ops_moe`, `test_fp8_moe` |
| `src/kernels/cpu/gguf/gguf_cpu.cpp`, `ggml/`, `ik/` (llama.cpp's and ik_llama.cpp's vendored CPU code), `src/backend/cpu/linear_gguf.cpp` | CPU: llama.cpp's dot products and block decoders, ik_llama.cpp's iqk GEMM for several rows | `test_gguf_linear` (CPU build), `test_ops_moe` |
| `src/backend/vulkan/mod_gguf.cpp`, `src/kernels/slang/gguf.slang`, `src/kernels/vulkan/ggml/` | Vulkan: Slang block decoders, GEMV and tiled GEMM against FP16 activations, and llama.cpp's vendored mat-vec and matmul shaders | `test_gguf_linear` (Vulkan build), `test_ops_moe`, `bench_gguf_shapes` (Vulkan build) |

## Dense FP16 GEMM: `src/kernels/cuda/gemm/gemm_f16.cu`

`Gemm::tn` and `Gemm::nn` on Volta's tensor cores (`mma.sync.m8n8k4`, FP32 accumulation, swizzled two-stage shared-memory tiles, split-k), the engine's GEMM
for the language models' dense products (on ROCm it forwards to hipBLAS). Verified by `test_gemm`, `bench_gemm`.

## Module kernels: `src/backend/cuda/modules/`

| File | Kernels | Verified by |
|---|---|---|
| `attention.cu` | decode / prefill tensor-core attention, KV store (FP16, FP8, K8V4, NVFP4) | `test_attention`, `test_attention_prefill`, `test_kv_quant`, `test_ops_attention` |
| `gdn.cu` | GDN / KDA projections, conv1d, recurrent step, per-row snapshots | `test_gdn`, `test_slang_kernels`, `test_ops_attention` |
| `mla_latent.cu` | absorbed latent attention (dense, sparse, split-KV; latent cache FP16, FP8, int8 or NVFP4 via kv_format) | `test_mla_latent`, `test_ref_attention` |
| `gated_residual.cu` | Flash-Next gated residual (norm, low-rank gate, mix, post gate, apply), MTP stream norm / broadcast, shared-expert gate | `test_gr_tc`, `test_gr_decode`, `test_ops_flash` |
| `ple.cu` (with the host table reader `src/loader/ngram_table.cpp`: mul1 ring codec, hashing) | Flash-Next n-gram embedding layer: gate, grouped norms, dilated conv ring | `test_ops_flash`, `test_prefix_restore --model qwen4_exp` |
| `qsa.cu` (+ `attention.cu` `forward_sparse`, `dsa.cu` `select`) | Flash-Next QSA indexer: q / k prep, pooled keys, scoring, token-list sparse GQA attention | `test_sparse_attn`, `test_ops_flash`, `test_longctx --model qwen4_exp` (`gate.sh needle`) |
| `dsa.cu` | indexer projection, plane / pool update, score, top-k select | `test_dsa` (FP64 reference) |
| `moe.cu`, `expert_tier.cu` | router, dispatch histogram / scatter, combine, two-tier expert residency | `test_moe`, `test_ops_moe` |
| `cpu_experts.cu` | the device half of the CPU expert path: post the routes handed to the CPU, wait for its answer, add it | `test_cpu_bridge` |
| `mhc.cu` | hyper-connection mix, collapse + norm, update | `test_mhc`, `test_slang_kernels`, `bench_mhc` |
| `dsv4.cu` | DeepSeek-V4.1: window ring store, KV compressor (the caches as packed FP8 / FP4 codes, decoded in the readers), indexer (scalar and tensor-core scorer, top-k select), window + selected-entry attention (one block a row; tensor-core split-KV for decode, verify and the DSpark drafter), DSpark taps, routing, Engram gate, FP8 projections | `test_dsv4_indexer`, `test_dsv4_fp8`, `test_dsv4_engram`, `bench_dsv4_attend`, `test_dsv41` |
| `norm.cu`, `rope.cu`, `swiglu.cu`, `embedding.cu` | RMSNorm (plain, heads, gated), RoPE, SwiGLU, embedding gather | `test_norm`, `test_rope`, `test_slang_kernels` |
| `vision.cu` | vision tower operators: patch-embedding GEMM, LayerNorm, qkv bias + 2-D rotary, register-resident mma.m8n8k4 flash attention (head_dim 64 / 72; WMMA fallback up to 80), bias + GELU, position-embedding interpolation | `test_vision`, `test_ops_vision` |
| `linear_trellis.cu` | trellis `Linear` dispatch (GEMM / GEMV / recon by M) | `test_ops_linear`, `bench_decode` |
| `steer.cu` | runtime steering: residual edit (plain and hyper-connection streams), probe scores, logit penalty | `test_steer`, `test_steer_probe`, `test_steer_model` |
| `sparse_head.cu` | sparse drafter head: column extraction through identity rows, subset logits, scatter | `test_sparse_head`, `bench_sparse_head` |
| `hessian.cu` | calibration: x^T x of projection inputs (`hessian_dir`) | `test_ppl --calibrate` |
| `logit_rows.cu` | per-row maximum, log-sum-exp, target NLL, top k and KL against a stored teacher (scoring tools) | `test_ppl`, `test_ear` |

## Training primitives: `src/kernels/cuda/train/`

| File | Kernels | Verified by |
|---|---|---|
| `grad.cu` | FP32 elementwise kernels and reductions, the scaled linear (a core between per-channel scales) with its gradients, Adam (FP32 and 8-bit moments) | `test_train_grad` |
| `qwen4_exp.cu` (with `src/train/`) | training forms of the qwen4_exp layer operations, each forward with its backward | `test_train_qwen4_exp`, `test_train_qwen4_exp_layer`, `test_train_qwen4_exp_tp` |

## Host CPU kernels: `src/kernels/cpu/`

| File | Kernels | Verified by |
|---|---|---|
| `trellis_linear.cpp` | EXL3 trellis (mul1 codebook, whole bits) repack and the AVX2 tile GEMV of the CPU experts | `test_cpu_experts`, `test_cpu_trellis_vs_gpu` |
| `nvfp4_linear.cpp` | NVFP4 W4A16 linear, the prepacked tiles read in place | `test_cpu_nvfp4_vs_gpu`, `test_cpu_bridge` |
| `pool.cpp`, `cpu_util.cpp` | the CPU kernels' thread pool and helpers | used by the above |

## ROCm forms: `src/kernels/rocm/`

Included by the CUDA sources under `USE_ROCM` ([ROCM.md](ROCM.md)): the `mma.sync` emulation in NVIDIA's fragment layouts (`mma_emu.h`), the WMMA vector types
(`wmma.cuh`), attention (`attn_wmma.cuh`, `attn_decode_rdna.cuh`, `dflash_attn_wmma.cuh`), NVFP4 GEMV and GEMM (`nvfp4_gemv.cuh`, `nvfp4_gemm.cuh`), the MLA
absorb GEMV (`mla_absorb_gemv.cuh`), a vectorized RMSNorm (`rms_norm_vec.cuh`), and in `trellis/` the RDNA trellis decode (`codebook_rdna.cuh`,
`exl3_tile_rdna.cuh`), the EXL3 GEMV, WMMA GEMM and MoE (`exl3_gemv.cu`, `exl3_gemm.cu`, `exl3_moe.cu`) and the GEMM shape selection (`exl3_shape_rdna.cuh`).
Verified by `test_mma_frag` (the emulations on a simulated warp), `test_trellis_decode` (the RDNA decode against the 32-bit multiply form), `test_trellis`,
`test_prefill_linear`, `test_attention_prefill` and `test_moe` on a ROCm build.

## MiniMax-H3: `src/kernels/cuda/h3/`, `src/video/`

| File | Kernels | Verified by |
|---|---|---|
| `src/kernels/cuda/h3/h3_attn.cu` | WMMA flash attention for head_dim 128 (full, causal) and 64: register Q / output, double-buffered K / V, online softmax on the accumulators; key-range variants carrying (max, sum, output) | `test_h3_attn`, `bench_h3_attn` |
| `src/video/h3_kernels.cuh` | AdaLN indexed norm / modulation, gated residuals, SiLU-mul with row scale (and its tensor-parallel split into max and apply passes), QK norm + 3-D RoPE, dtype conversion | `test_h3_dit`, `test_h3_dit_sp`, `test_h3_dit_tp` |
| `src/video/h3_dit.cu` | ConvRot INT8 dequantization + un-rotation to FP16 per matrix, then cuBLAS GEMMs; staged LoRA deltas; sequence- and tensor-parallel blocks | `test_h3_dit*`, `bench_h3_dit`, `bench_h3_gemm` |
| `src/video/h3_vae.cu` | video decoder (ViT) tiles, placement / blending, RGB8 conversion | `test_h3_vae`, `test_h3_vae_det` |
| `src/video/h3_vae_enc.cu`, `h3_audio.cu` | causal 3-D conv encoder as FP16 im2col + GEMM; audio conv1d, snake, alias-free resampling | `test_h3_vae_enc`, `test_h3_audio`, `test_h3_audio_enc` |
| `src/video/h3_text.cu` | Qwen3-VL text tower: interleaved mRoPE, causal attention on `h3_attn.cu`, resident EXL3 / NVFP4 projections through `Linear::forward`, tensor-parallel shards | `test_h3_text`, `test_h3_text_mm`, `test_h3_text_tp` |
| `src/video/h3_vision.cu` | adapter around `Qwen3VisionTower` (dense BF16 tower, fused qkv, DeepStack taps) | `test_h3_vision` |

The H3 tests take the checkpoints and their goldens (safetensors files written by `tests/tools/h3_*.py`) on the command line; see [VIDEO.md](VIDEO.md).

## Portable Slang kernels: `src/kernels/slang/`

`*.slang` compile to CUDA (launchers in `slang/cuda`), SPIR-V and host C++. Reference kernels (`ref_gemm`, `ref_trellis`,
`ref_attention`, `ref_latent`) are the parity oracle for the CPU and Vulkan backends. Verified by `test_slang_kernels` (CUDA),
`test_slang_cpu` (host), `test_ref_linear`, `test_ref_attention`, `test_vulkan_kernels`, `test_vulkan_ref`, `test_cpu_smoke`.
`fa.slang` is the Vulkan backend's attention (decode and prefill): a workgroup takes a KV head's query heads for up to 16 query vectors, stages K and V
tiles in shared memory (16-byte loads, FP16 products accumulated in FP32), and splits a long key range over workgroups whose partials `fa_reduce` combines;
`test_vulkan_attention` holds it against `ref_attention` on the same device. `kv_codec.slang` holds the KV cache element codecs (FP8, int8, Q4, NVFP4 with
their scales) that `kv_store_q` and the attention kernels (`ref_attention`, `fa`, the QSA sparse attention in `flash`, the latent attention in `glm`) share; the stored bytes and scales are checked bit for bit against the CUDA kernels by
`test_ops_attention` and `test_ops_flash` (dumped on CUDA, checked on CPU and Vulkan).

## Speculation and communication

| File | Kernels | Verified by |
|---|---|---|
| `src/spec/dflash_kernels.cu`, `dflash_backbone.cu`, `dflash_speculator.cu` | drafter backbone (DFlash2 and DFlash v1), context projection, selector, bilinear lattice walk | `test_dflash`, `test_dflash2_blocks`, `test_dflash2_backbone`, `test_coherence` |
| `src/spec/tap_ring.cu` | tap ring publication | `test_dflash` |
| `src/comm/all_reduce_ipc.cu` | 2- and 4-rank NVLink all-reduce (flag polling, peer pointers, LL lines for decode rows), fused with the residual norm, the gated add, the hyper-connection post-step and Gemma 4's normed add | `test_all_reduce`, `bench_all_reduce` |
