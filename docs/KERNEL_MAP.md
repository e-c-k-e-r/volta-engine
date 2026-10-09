# Kernel map

Kernel family -> file -> tier -> what verifies it. Tiers are those of [CAPABILITIES.md](CAPABILITIES.md). "Fingerprint" means
`gate.sh sass` (any change to a kernel changes the per-function SASS hash).

## Trellis (EXL3) weights: `src/kernels/cuda/trellis/`

| Kernel family | Files | Tier | Verified by |
|---|---|---|---|
| GEMM (prefill and mid-M, HMMA m8n8k4) | `exl3_gemm.cu`, `exl3_gemm_kernel.cuh`, `exl3_gemm_inner.cuh`, `exl3_kernel_map.*`, `exl3_instances.cu` | CudaSm70 | `test_trellis`, `test_prefill_linear`, `bench_shapes` |
| GEMV / skinny M (M <= 8) | `exl3_gemv.cu`, `exl3_gemv_kernel.cuh`, `exl3_mgemm_gv_kernel.cuh` | CudaSm70 | `test_trellis`, `bench_decode` |
| int8 dp4a GEMV (M <= 2) | `exl3_gemv_int8.cu`, `exl3_gemv_int8_kernel.cuh` | CudaSm70 | `test_gemv_int8` |
| Dequantize for large M (recon + HGEMM) | `reconstruct.cu`, `exl3_dq.cuh`, `codebook.cuh` | CudaSm70 (CPU: native threaded dequant) | `test_trellis`, `test_ops_linear` |
| Hadamard rotation | `hadamard.cu`, `hadamard_inner.cuh` | CudaSm70 | `test_trellis` |
| Fused MoE (persistent and cooperative, int8 tile) | `exl3_moe_kernel.cuh`, `exl3_moe_coop.cu`, `exl3_moe_common.cuh` | CudaSm70 | `test_moe`, `test_ops_moe`, `bench_moe_prefill` |
| Workspace state | `trellis_context.cu/.h` | n/a | used by every kernel above |

## NVFP4: `src/kernels/cuda/nvfp4/skinny_kernels.cu`

Skinny (M <= 32) and fused-GEMM kernels with FP16-rounded dequant, plus the `LinearNVFP4` module entry (CudaSm70; CPU has a
mirrored implementation). Verified by `test_nvfp4_linear`, `test_ops_linear`.

## Module kernels: `src/backend/cuda/modules/`

| File | Kernels | Tier | Verified by |
|---|---|---|---|
| `attention.cu` | decode / prefill tensor-core attention, KV store (FP16, FP8, K8V4, NVFP4) | Portable (FP16 KV) / CudaSm70 | `test_attention`, `test_attention_prefill`, `test_kv_quant`, `test_ops_attention` |
| `gdn.cu` | GDN / KDA projections, conv1d, recurrent step, per-row snapshots | Portable | `test_gdn`, `test_slang_kernels`, `test_ops_attention` |
| `mla.cu`, `mla_latent.cu` | expanded MLA (op-level reference), absorbed latent attention (dense, sparse, split-KV) | Portable / CudaSm70 | `test_mla`, `test_mla_latent`, `test_ref_attention` |
| `dsa.cu` | indexer projection, plane / pool update, score, top-k select | mixed | `test_dsa` (FP64 reference) |
| `moe.cu`, `expert_tier.cu` | router, dispatch histogram / scatter, combine, two-tier expert residency | CudaSm70 | `test_moe`, `test_ops_moe` |
| `mhc.cu` | hyper-connection mix, collapse + norm, update | Portable | `test_mhc`, `test_slang_kernels` |
| `norm.cu`, `rope.cu`, `swiglu.cu`, `embedding.cu` | RMSNorm (plain, heads, gated), RoPE, SwiGLU, embedding gather | Portable | `test_norm`, `test_rope`, `test_slang_kernels` |
| `vision.cu` | vision tower operators: patch-embedding GEMM, LayerNorm, qkv bias + 2-D rotary, register-resident mma.m8n8k4 flash attention (head_dim 64 / 72; WMMA fallback up to 80), bias + GELU, position-embedding interpolation | CudaSm70 | `test_vision` |
| `linear_trellis.cu` | trellis `Linear` dispatch (GEMM / GEMV / recon by M) | CudaSm70 | `test_ops_linear`, `bench_decode` |

## Portable Slang kernels: `src/kernels/slang/`

`*.slang` compile to CUDA (launchers in `slang/cuda`), SPIR-V and host C++. Reference kernels (`ref_gemm`, `ref_trellis`,
`ref_attention`, `ref_latent`) are the parity oracle for the CPU and Vulkan backends. Verified by `test_slang_kernels` (CUDA),
`test_slang_cpu` (host), `test_ref_linear`, `test_ref_attention`, `test_vulkan_kernels`, `test_vulkan_ref`.

## Speculation and communication

| File | Kernels | Verified by |
|---|---|---|
| `src/spec/dflash_kernels.cu`, `dflash_backbone.cu`, `dflash_speculator.cu` | drafter backbone, context projection, selector, bilinear lattice walk | `test_dflash`, `test_dflash2_backbone`, `test_coherence` |
| `src/spec/tap_ring.cu` | tap ring publication | `test_dflash` |
| `src/comm/all_reduce_ipc.cu` | 4-rank NVLink all-reduce (flag polling, peer pointers) | `test_all_reduce`, `bench_all_reduce` |
