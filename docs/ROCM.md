# ROCm (AMD RDNA)

The `ROCM` configuration builds the CUDA backend's sources as HIP for AMD RDNA GPUs (wave32: RX 7000 / RX 9000 series). It is a
port: every kernel runs. The attention kernels, the NVFP4 GEMV and GEMM and the EXL3 MoE run on RDNA3's native WMMA; the remaining kernels
that use NVIDIA's `mma.sync` run on its emulation (see "Still on the vector ALUs or the emulation").

## Build

ROCm 6.2 or newer (developed against 7.2) with hipBLAS, CMake 3.24+ and Ninja:

```
cmake -S . -B build_rocm -G Ninja -DENGINE_BACKEND=ROCM -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
	-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build_rocm
```

`gfx1100` is the RX 7900 XT / XTX (the default); `gfx1101` / `gfx1102` the RX 7800 / 7700 / 7600, `gfx1200` / `gfx1201` the RX 9060 /
9070. A `;`-list builds one binary for several. Wave64 targets (CDNA / GCN, `gfx9xx`) are rejected at configure: every kernel is
written for 32-lane warps. Startup prints each GPU's target, compute units, LDS and wave size, and warns about a GPU the build has no
code for.

## How it works

* `src/backend/rocm/compat.h` is force-included into every translation unit. It maps the CUDA runtime names onto HIP and supplies what
  HIP spells differently or lacks: the warp-sync shuffles (mask dropped, as wave32 makes it exact), `__ldcs` / `__ldcg`, `__dp4a`, half2
  min / max and atomic add, explicit-rounding bf16 conversion, a launch shim that clamps shared memory requests to the device.
* `src/backend/rocm/include` stands in for the CUDA headers: `cuda_runtime.h`, `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h` (the float to
  OCP E4M3 conversion the KV cache uses, written out), `cublas_v2.h` (onto hipBLAS), `cooperative_groups.h`, `cuda/atomic`, and `mma.h`
  (the `nvcuda::wmma` subset on RDNA3's `v_wmma_f32_16x16x16_f16`, with the RDNA4 form and a shuffle fallback).
* Inline PTX has a `USE_ROCM` form at each site: memory-model loads / stores as HIP atomics with the same scope, `cp.async` as a
  synchronous copy (fences and waits become no-ops, as on SM_70), `ldmatrix` from shuffled row pointers, bit operations in C++.
* Tensor-core `mma.sync` (m16n8k16 with FP32 or FP16 accumulation, m16n8k8, m8n8k4) is emulated in NVIDIA's fragment layouts
  (`src/kernels/rocm/mma_emu.h`): each lane gathers the A rows and B columns its outputs need over shuffles and sums the products in
  FP32. The kernels built on them run unchanged; numerics match the tensor cores up to summation order. `test_mma_frag` runs every
  emulation on a simulated warp on the CPU against the PTX layouts (no GPU needed).
* ROCm takes the SM_80+ code paths (the native-`mma` forms of the EXL3, NVFP4 and attention kernels), with the emulated instructions
  under them, except where a kernel has a native WMMA form. Host-side EXL3 choices use the oldest architecture class, as upstream
  exllamav3 does on ROCm. `MMA_SM80` is 1 on ROCm, so a `#elif defined(USE_ROCM)` after `#if MMA_SM80` is never compiled: a ROCm form
  goes in the `#if` itself (`#if MMA_SM80 && !defined(USE_ROCM)` for the CUDA one).

Decode graphs work on ROCm (HIP graphs; the EXL3 GEMV is a plain launch) and are on by default. `ENGINE_DEBUG_NO_GRAPHS=1` runs eager.

## Shared memory

RDNA grants a workgroup 64 KB of LDS (V100: 96 KB). Kernels tuned for more take another route on ROCm, decided at run time from the
device's limit:

| Kernel | Needs | On ROCm |
|---|---|---|
| prefill attention (head_dim 256), sparse attention prefill | ~96 KB | the decode / per-row attention kernels (same results) |
| MLA latent prefill (2 rows x 16 heads per block) | ~80 KB | the per-row latent kernel, dense when there is no selection |
| vision tower attention (WMMA) | ~75 KB | the SIMT attention kernel |
| NVFP4 MoE (8 staged rows) | 16 x K bytes | as many rows per chunk as fit (6 at K = 5120) |
| EXL3 MoE | requested 90 KB | the tile's own footprint (22 KB at most) |
| EXL3 GEMM | 22-90 KB by shape | upstream's RDNA scorer picks among the shapes that fit |
| EXL3 int8 GEMV | ~80 KB | declined (the FP16 GEMV runs) |

## Kernel routing

The emulated `mma` kernels are slow on RDNA, so by row count a linear takes another route. An EXL3 linear takes a GEMV for 1 to 4 rows, a
WMMA GEMM for 5 to 64 rows, and above that dequantizes its weight once into FP16 and runs one hipBLAS GEMM (prefill chunks).

The GEMV (`src/kernels/rocm/trellis/exl3_gemv.cu`, after upstream exllamav3's RDNA GEMV) is a plain launch: the input and output
Hadamard transforms are their own launches, and between them a kernel accumulates `v_dot2_f32_f16` straight from the trellis
decoder's fragments (each lane quad owns two output columns), one block per 16 columns with its warps splitting K. The CUDA GEMV is
one cooperative kernel on `mma.sync`, emulated on ROCm (under 1 tok/s on Qwen3.8-27B-exl3).

The GEMM (`src/kernels/rocm/trellis/exl3_gemm.cu`) keeps the same structure (the Hadamard rotations are separate launches, one block per 16 output
columns, the warps split K) and multiplies on the native wave32 WMMA. A warp decodes each 16 x 16 trellis tile once, stages it column-major in shared memory,
loads the B fragment from there (lane l holds column l & 15), and every 16-row block of the input reuses that fragment; the A fragments are read from the
rotated rows. A warp decodes the next tile into a second buffer while the current one is multiplied and loads the next A fragments ahead; a block has 16 warps up
to 32 rows and 8 at 64 rows (the fixed-order reduction needs their partial tiles in shared memory). The packed weight is read once, so the FP16 copy of the weight that dequantize + hipBLAS writes and re-reads (2.4 GB for the Qwen
LM head) never exists. The warps' partial tiles are summed in a fixed order through shared memory, so a result does not depend on scheduling. It takes
bit widths 1 to 8 and all codebooks, FP16 output, widths that are multiples of 128. `ENGINE_DEBUG_EXL3_GEMV_MAX_M` (default 4) and
`ENGINE_DEBUG_EXL3_WMMA_MAX_M` (default 64; 0 sends everything above the GEMV to dequantize + hipBLAS) move the two boundaries. `test_prefill_linear` checks it
against the dequantized product on the Qwen per-rank and unsharded shapes at 3 to 6 bits for 5 to 64 rows.

NVFP4 linears are routed by rows too: the dequantize + GEMM path writes and re-reads the whole weight as FP16 on every call, about 8x the
packed bytes, so up to 16 rows they take a SIMT GEMV (`rocm_nvfp4_gemv`, skinny_kernels.cu)
that decodes each lane's codes in registers and accumulates with `v_dot2_f32_f16`, reading the packed weight once. Above 16 rows (prefill chunks) a fused dequantize + GEMM on the native WMMA (`rocm_nvfp4_gemm_kernel`, 128 x 128 x 32 tiles) decodes the
weight straight into shared memory. `ENGINE_DEBUG_NVFP4_HIPBLAS=1` takes dequantize + hipBLAS instead, whose FP16 scratch can move as it
grows (which a captured graph cannot follow; decode graphs, at most 8 rows, never take that path). `ENGINE_DEBUG_MOE_GEMV=1` runs the
expert-grouped EXL3 MoE on the v_dot2 GEMV instead of its WMMA kernels.

A config's `tp_size` must not exceed the GPUs present (the files in configs/ are written for four).

## Cooperative launches

The EXL3 GEMV's `grid.sync()` needs every block resident at once. On ROCm its grid is one block per CU rather than HIP's occupancy
figure: a full-width projection at that figure hung a 7900 XTX in the first decode forward (the kernel walks its column tiles, so
the grid size only changes how many each block takes). `test_trellis` runs a full-width GEMV for this.

## Tests

`test_trellis` and `test_mhc` assert zero host heap allocations per forward on CUDA. HIP's launch path allocates a command object on
the host for each kernel launch, so on ROCm they report the count instead of failing on it; the numerical checks are unchanged.

## Not supported on ROCm

* MiniMax-H3 (video): its kernels stage up to 97 KB, and its attention reads WMMA accumulator elements in NVIDIA's layout. It compiles,
  but is not expected to run.
* NCCL tensor parallelism for MiniMax-H3 (RCCL is not wired in). The language models' own collective builds; tensor parallelism needs
  peer access between the cards.

## Still on the vector ALUs or the emulation

* The EXL3 GEMV and GEMM: the GEMV's per-row products use `v_dot2` and its trellis decode is ALU-bound, so an 8-row verify costs 2.3 times a token on EXL3.
* The MoE kernels other than the expert-grouped one (`moe_kernels.cu`, `exl3_moe_coop.cu`) and the int8 GEMV.
* The linear-attention (GDN) prefill recurrence, which is serial per head.

## Performance

One RX 7900 XTX (24 GB), Qwen3.8-27B, `build/bench_decode` (8K context) and `bench_prefill`, decode graphs on, a 10K window:

| | Plain token | 8-row verify | Prefill (512-row chunks) |
|---|---|---|---|
| NVFP4 (head quantized, `nvfp4_dense=head`) | 21.6 ms (46 tok/s) | 27 ms (1.25x) | 1123 tok/s |
| EXL3 4.0 bpw | 29.0 ms (34 tok/s) | 40 ms (1.4x) | 1115 tok/s |

* NVFP4 target with the DFlash2 EXL3 5.0 bpw drafter, natural 8K prompts (`tests/3_bench/bench_serve.py`): 129, 153, 145 and 75 tok/s, 37 ms per round (27 ms verify, 9.5 ms drafter).
* The NVFP4 GEMVs read the packed weights at 80 to 90% of the 920 GB/s a plain read kernel reaches on this card; the prefill GEMM reaches 73 of the ~96 TFLOP/s the WMMA peaks at.
* EXL3 decode is bound by the trellis decode arithmetic (about 29 ms of the plain token with the per-row math removed), not by memory. A verify block of 8 rows costs 1.4 times a token (40 ms), 12 rows 42 ms and 16 rows 45 ms (`bench_decode` with `BENCH_VL=8,12,16`, 8000 tokens of context).
