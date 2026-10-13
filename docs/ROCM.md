# ROCm (AMD RDNA)

The `ROCM` configuration builds the CUDA backend's sources as HIP for AMD RDNA GPUs (wave32: RX 7000 / RX 9000 series). It is a
port for correctness first: every kernel runs, the tensor-core instructions are emulated, and nothing is tuned for RDNA yet.

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
  (`src/kernels/cuda/mma_emu.h`): each lane gathers the A rows and B columns its outputs need over shuffles and sums the products in
  FP32. The kernels built on them run unchanged; numerics match the tensor cores up to summation order. `test_mma_frag` runs every
  emulation on a simulated warp on the CPU against the PTX layouts (no GPU needed).
* ROCm takes the SM_80+ code paths (the native-`mma` forms of the EXL3, NVFP4 and attention kernels), with the emulated instructions
  under them. Host-side EXL3 choices use the oldest architecture class, as upstream exllamav3 does on ROCm.

Execution is eager by default on ROCm: the decode graphs contain cooperative launches (the EXL3 GEMV), which HIP's stream capture may
not accept. `ENGINE_DEBUG_NO_GRAPHS=0` turns graphs back on to try them.

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

## Routing on measurements

Until the tensor-core kernels have RDNA forms, dequantizing a weight once into FP16 and running one hipBLAS GEMM beats the emulated
kernels (7900 XTX: NVFP4 gate_proj 1-5 ms fused against ~0.2 ms; EXL3 prefill 15-125x at M = 16..64). On ROCm every NVFP4 linear takes
that path from 17 rows up, and every EXL3 linear from 9 rows up.

Up to 8 rows, EXL3 linears take a plain-launch GEMV (`exl3_gemv_rocm.cu`, after upstream exllamav3's RDNA GEMV): the input and output
Hadamard transforms as their own launches, and between them a kernel that accumulates `v_dot2_f32_f16` straight from the trellis
decoder's fragments (each lane quad owns two output columns), one block per 16 columns with its warps splitting K. The CUDA GEMV is
one cooperative kernel on `mma.sync`, emulated on ROCm (under 1 tok/s on Qwen3.8-27B-exl3).

Decode is the exception for NVFP4: the dequantize + GEMM path writes and re-reads the whole weight as FP16 on every call, about 8x the
packed bytes (a 27B model measured 5 tok/s). Up to 16 rows, NVFP4 linears take a SIMT GEMV (`rocm_nvfp4_gemv`, skinny_kernels.cu)
that decodes each lane's codes in registers and accumulates with `v_dot2_f32_f16`, reading the packed weight once. The FP16 scratch it uses can move as it grows, which a captured
graph cannot follow: another reason graphs stay off on ROCm by default.

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

## Performance work left

The emulated `mma` costs about 20 shuffles and 32 FMAs per instruction, so the tensor-core kernels run on the vector ALUs. In order of
payoff:

1. The EXL3 GEMV / GEMM and the NVFP4 GEMV on WMMA. Upstream exllamav3 has RDNA kernels for EXL3 (`exllamav3_ext/rocm/quant`:
   the GEMM inner, GEMV, multi-row GEMV and MoE), which use WMMA's own layout (whole rows and columns per lane) rather than NVIDIA's.
2. The attention kernels on WMMA, restaged for 64 KB.
3. The int8 GEMV within 64 KB, and `v_dot4` / `v_dot2` in the decode inner loops.
