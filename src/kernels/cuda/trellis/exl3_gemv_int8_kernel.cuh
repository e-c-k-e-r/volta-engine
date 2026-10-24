#pragma once

// Device side of the fused int8-activation GEMV, ported from exllamav3
// (exllamav3_ext/quant/exl3_gemv_int8_kernel.cuh, commits 701656dd / fe077313). Only the per-slice-
// scale ("sq") variant is ported; the cooperative variant is not, because the sq kernel is a plain
// cudaLaunchKernel with a fixed workspace and no grid.sync, so it captures into our static decode
// graphs. Everything below the sq kernel boundary is upstream code with the includes repointed at
// this tree; the sq kernel carries one extra template parameter (stage_smem) so EXL3_INT8_SMEM can
// select the smem-staged unit per bit width.
//
// The mul1 codebook value is affine in the byte sum of x * 0x83DCD12D, so with activations quantized
// to int8, dp4a(x * M, splat(a_int8), acc) evaluates codebook(x) * a in a single instruction with
// exact int32 accumulation, and
//
//   y[n] = k_inv * (q * acc1[n] + q2 * acc2[n]) + (1024 * k_inv + k_bias) * (q * sum1 + q2 * sum2)
//
// recovers the result. In residual mode (the fixed mode) the int8 rounding error r = a - q*i, which
// has range +/-q/2 by construction, is quantized with the fixed scale q2 = q/254 and accumulated by a
// second dp4a sharing the decoded products, for ~15-16 bit effective activation precision (KL at
// parity with the fp16 kernel or better). A plain int8 mode (no residual pass) was removed.
//
// The sq kernel kills all global pre-GEMV coordination: each k-slice is Hadamard-transformed, maxed
// and quantized block-locally during staging with its own q_s = slicemax/127 (deterministic, so
// duplicating blocks agree bit-exactly, and strictly finer than a global q), units write exact
// per-slice int32 partials with plain stores (exclusive writers - no atomics, no accumulator
// zeroing), and a per-256-column completion counter gates an inline epilogue that combines
// sum_s q_s*acc_s in fixed slice order. Workspace layout, fixed offsets:
//   [counters: SQ_COUNTERS_CAP][{q_s, s1_s, s2_s, pad} x SQ_KSPLIT_CAP][partials: ksplit x M x pstride]
// Counters are zeroed at allocation and reset by the finishing block, so they stay zero between calls
// and across graph replays.
//
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cooperative_groups.h>
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

namespace cg_gemv = cooperative_groups;

// K == 4 uses the wide unit: a warp owns an ADJACENT BLOCK PAIR; lane l's uint2 = words {2m, 2m+1} of
// the contiguous 64-word region are exactly the primary words of runs t = 8*(2m), 8*(2m+1), which
// scatter to the SAME two n values (accumulator count unchanged, splats merge into two uint4 loads,
// single-stage butterfly, boundary word via one shuffle). 256 B contiguous per warp per block row
// with two-row prefetch. Other K use a generic narrow unit (two blocks per warp, pointer-based
// extraction), or the smem-staged unit for the bit widths routed to it.

#define NUM_THREADS 256

// ---------------------------------------------------------------------------------------------------------
// Extract 8 packed 16-bit trellis windows for positions t0..t0+7 as masked uint32s. Index math copied
// from the corresponding paths in exl3_dq.cuh (dq8_aligned_*, dq8, dq4, dq2x2).

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c)
{
#if defined(USE_ROCM)
    return static_cast<int>(rocm_dp4a_us(a, b, static_cast<unsigned int>(c)));
#else
    int d;
    asm ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#endif
}

// i0/i2 land in [0, 2*words); a compare+subtract replaces the modulo (words is not a power of two for
// odd bit widths)
template <int bits>
__device__ __forceinline__ int wrap_idx(int i)
{
    constexpr int words = bits * 256 / 32;
    return i >= words ? i - words : i;
}

template <int bits>
__device__ __forceinline__ void ext4w(const uint32_t* ptr, int t0, uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3)
{
    int b0 = (t0 + 257) * bits - 16;
    int b2 = b0 + 3 * bits + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[wrap_idx<bits>(i0)];
    uint32_t b = ptr[wrap_idx<bits>(i2)];
    w3 = fshift(b, a, s2) & 0xffff;
    w2 = fshift(b, a, s2 + bits) & 0xffff;
    w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext2w(const uint32_t* ptr, int t0, uint32_t& w0, uint32_t& w1)
{
    int b0 = (t0 + 257) * bits - 16;
    int b2 = b0 + bits + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[wrap_idx<bits>(i0)];
    uint32_t b = ptr[wrap_idx<bits>(i2)];
    w1 = fshift(b, a, s2) & 0xffff;
    w0 = fshift(b, a, s2 + bits) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext8w
(
    const uint32_t* ptr, int t0,
    uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3,
    uint32_t& w4, uint32_t& w5, uint32_t& w6, uint32_t& w7
)
{
    if constexpr (bits == 1)
    {
        uint32_t i1 = t0 >> 5;
        uint32_t i0 = (i1 + 7) & 7;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t0) & 24));
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 1);
        BFE16_IMM(w5, b, 2);
        BFE16_IMM(w4, b, 3);
        BFE16_IMM(w3, b, 4);
        BFE16_IMM(w2, b, 5);
        BFE16_IMM(w1, b, 6);
        BFE16_IMM(w0, b, 7);
    }
    else if constexpr (bits == 2)
    {
        uint32_t i1 = t0 >> 4;
        uint32_t i0 = (i1 + 15) & 15;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t0) & 8) << 1);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 2);
        BFE16_IMM(w5, b, 4);
        BFE16_IMM(w4, b, 6);
        BFE16_IMM(w3, b, 8);
        BFE16_IMM(w2, b, 10);
        BFE16_IMM(w1, b, 12);
        BFE16_IMM(w0, b, 14);
    }
    else if constexpr (bits == 3)
    {
        int b1 = (t0 + 257) * bits;
        int b0 = b1 - 16;
        int b2 = b1 + bits * 7;
        int i0 = b0 / 32;
        int i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = ptr[wrap_idx<bits>(i0)];
        uint32_t b = ptr[wrap_idx<bits>(i2)];
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
        w7 &= 0xffff; w6 &= 0xffff; w5 &= 0xffff; w4 &= 0xffff;
        w3 &= 0xffff; w2 &= 0xffff; w1 &= 0xffff; w0 &= 0xffff;
    }
    else if constexpr (bits == 4)
    {
        uint32_t i1 = t0 >> 3;
        uint32_t i0 = (i1 + 31) & 31;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        uint32_t s;
        FSHF_IMM(s, b, a, 20);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 4);
        BFE16_IMM(w5, b, 8);
        BFE16_IMM(w4, b, 12);
        BFE16_IMM(w3, b, 16);
        w2 = s & 0xffff;
        BFE16_IMM(w1, s, 4);
        BFE16_IMM(w0, s, 8);
    }
    else if constexpr (bits == 7)
    {
        ext2w<bits>(ptr, t0,     w0, w1);
        ext2w<bits>(ptr, t0 + 2, w2, w3);
        ext2w<bits>(ptr, t0 + 4, w4, w5);
        ext2w<bits>(ptr, t0 + 6, w6, w7);
    }
    else  // 5, 6, 8
    {
        ext4w<bits>(ptr, t0,     w0, w1, w2, w3);
        ext4w<bits>(ptr, t0 + 4, w4, w5, w6, w7);
    }
}

// 4bpw extraction from two already-loaded words (same window order)
__device__ __forceinline__ void extract8_4bits_words(uint32_t a, uint32_t b,
    uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3,
    uint32_t& w4, uint32_t& w5, uint32_t& w6, uint32_t& w7)
{
    uint32_t s;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
}

// ---------------------------------------------------------------------------------------------------------
// Staging and slice parameters

// The K cap (fall back to the regular kernel in the DRAM-bound regime) is per-arch; see
// exl3_gemv_int8_max_k in exl3_gemv_int8.cu
#define GEMV_STAGE_D 4      // cp.async pipeline depth (rows) for the smem-staged unit

// Per-slice-scale ("sq") kernel parameters
#define SQ_KSPLIT_CAP 64
#define SQ_MINROWS 16       // minimum slice height (staging overhead amortization)
#define SQ_ROWS_MAX 512     // shared memory cap
#define SQ_COUNTERS_CAP 4096                                    // fixed counter region: size_n up to 1M
#define SQ_WS_RESERVED (SQ_COUNTERS_CAP + 4 * SQ_KSPLIT_CAP * 4)    // sq-private region at workspace start (counters + per-slice-per-row scales, m <= 4)

// Wide unit (4 bpw): one (256-column x k-slice) unit, warp per adjacent block pair, uint2 loads.
// M activation rows share the decoded weights: extraction and the B stream are amortized, each row
// only adds its dp4a chains against its own splat region (sh_as + r * slice_stride; residual splats
// at (M + r) * slice_stride - at M = 1 this is exactly the coop kernel's sh_as/sh_as2 layout). Row
// accumulators go to accs + r * acc_stride, atomically or with plain stores (exclusive per-slice
// partials of the sq kernel).
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_wide
(
    const uint16_t* __restrict__ B,
    int* __restrict__ accs,
    size_t acc_stride,
    const uint32_t* __restrict__ sh_as,
    int slice_stride,
    int nb256,
    int kb0,
    int nrows,
    int size_n
)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n * 2;                              // u32 per block row at 4 bpw
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * 64 + 2 * lane;
    int c2 = (lane & 1) ? 4 : 0;
    int shfl_src = (lane & 16) | ((lane + 15) & 15);
#if EXL3_VOLTA_LAYOUT
    const int cs = (lane >> 2) & 1;   // which half of the k16 row: bit 3 of the run index 2 * (lane & 15)
#endif

    int iacc0[M] = {}, iacc1[M] = {}, jacc0[M] = {}, jacc1[M] = {};
    uint2 r0 = *(const uint2*) bp;
    uint2 r1 = {};
    if (nrows > 1) r1 = *(const uint2*) (bp + row_stride);

    for (int kb = 0; kb < nrows; ++kb)
    {
        uint2 r2 = {};
        if (kb + 2 < nrows) r2 = *(const uint2*) (bp + (size_t) (kb + 2) * row_stride);
        uint32_t prev = __shfl_sync(0xffffffff, r0.y, shfl_src);

        uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
        uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
        extract8_4bits_words(prev, r0.x, w0, w1, w2, w3, w4, w5, w6, w7);   // run t = 8*(2m)
        extract8_4bits_words(r0.x, r0.y, v0, v1, v2, v3, v4, v5, v6, v7);   // run t = 8*(2m+1)
        w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
        w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
        v0 *= 0x83DCD12Du; v1 *= 0x83DCD12Du; v2 *= 0x83DCD12Du; v3 *= 0x83DCD12Du;
        v4 *= 0x83DCD12Du; v5 *= 0x83DCD12Du; v6 *= 0x83DCD12Du; v7 *= 0x83DCD12Du;

#if EXL3_VOLTA_LAYOUT
        // Volta tile order: run 2 * lp holds one column over eight consecutive k (8 * cs ..), and so does run 2 * lp + 1 (the next column)
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const uint32_t* as = sh_as + r * slice_stride + (kb << 4) + 8 * cs;
            const uint4 a0 = *(const uint4*) as, a1 = *(const uint4*) (as + 4);
            iacc0[r] = dp4a_us(w0, a0.x, iacc0[r]); iacc0[r] = dp4a_us(w1, a0.y, iacc0[r]); iacc0[r] = dp4a_us(w2, a0.z, iacc0[r]); iacc0[r] = dp4a_us(w3, a0.w, iacc0[r]);
            iacc0[r] = dp4a_us(w4, a1.x, iacc0[r]); iacc0[r] = dp4a_us(w5, a1.y, iacc0[r]); iacc0[r] = dp4a_us(w6, a1.z, iacc0[r]); iacc0[r] = dp4a_us(w7, a1.w, iacc0[r]);
            iacc1[r] = dp4a_us(v0, a0.x, iacc1[r]); iacc1[r] = dp4a_us(v1, a0.y, iacc1[r]); iacc1[r] = dp4a_us(v2, a0.z, iacc1[r]); iacc1[r] = dp4a_us(v3, a0.w, iacc1[r]);
            iacc1[r] = dp4a_us(v4, a1.x, iacc1[r]); iacc1[r] = dp4a_us(v5, a1.y, iacc1[r]); iacc1[r] = dp4a_us(v6, a1.z, iacc1[r]); iacc1[r] = dp4a_us(v7, a1.w, iacc1[r]);
            if constexpr (residual)
            {
                const uint32_t* as2 = sh_as + (M + r) * slice_stride + (kb << 4) + 8 * cs;
                const uint4 b0 = *(const uint4*) as2, b1 = *(const uint4*) (as2 + 4);
                jacc0[r] = dp4a_us(w0, b0.x, jacc0[r]); jacc0[r] = dp4a_us(w1, b0.y, jacc0[r]); jacc0[r] = dp4a_us(w2, b0.z, jacc0[r]); jacc0[r] = dp4a_us(w3, b0.w, jacc0[r]);
                jacc0[r] = dp4a_us(w4, b1.x, jacc0[r]); jacc0[r] = dp4a_us(w5, b1.y, jacc0[r]); jacc0[r] = dp4a_us(w6, b1.z, jacc0[r]); jacc0[r] = dp4a_us(w7, b1.w, jacc0[r]);
                jacc1[r] = dp4a_us(v0, b0.x, jacc1[r]); jacc1[r] = dp4a_us(v1, b0.y, jacc1[r]); jacc1[r] = dp4a_us(v2, b0.z, jacc1[r]); jacc1[r] = dp4a_us(v3, b0.w, jacc1[r]);
                jacc1[r] = dp4a_us(v4, b1.x, jacc1[r]); jacc1[r] = dp4a_us(v5, b1.y, jacc1[r]); jacc1[r] = dp4a_us(v6, b1.z, jacc1[r]); jacc1[r] = dp4a_us(v7, b1.w, jacc1[r]);
            }
        }
#else
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const uint32_t* as = sh_as + r * slice_stride + (kb << 4);
            uint4 as0 = *(const uint4*) (as + c2);
            uint4 as8 = *(const uint4*) (as + c2 + 8);
            iacc0[r] = dp4a_us(w0, as0.x, iacc0[r]);
            iacc0[r] = dp4a_us(w1, as0.y, iacc0[r]);
            iacc0[r] = dp4a_us(w2, as8.x, iacc0[r]);
            iacc0[r] = dp4a_us(w3, as8.y, iacc0[r]);
            iacc1[r] = dp4a_us(w4, as0.x, iacc1[r]);
            iacc1[r] = dp4a_us(w5, as0.y, iacc1[r]);
            iacc1[r] = dp4a_us(w6, as8.x, iacc1[r]);
            iacc1[r] = dp4a_us(w7, as8.y, iacc1[r]);
            iacc0[r] = dp4a_us(v0, as0.z, iacc0[r]);
            iacc0[r] = dp4a_us(v1, as0.w, iacc0[r]);
            iacc0[r] = dp4a_us(v2, as8.z, iacc0[r]);
            iacc0[r] = dp4a_us(v3, as8.w, iacc0[r]);
            iacc1[r] = dp4a_us(v4, as0.z, iacc1[r]);
            iacc1[r] = dp4a_us(v5, as0.w, iacc1[r]);
            iacc1[r] = dp4a_us(v6, as8.z, iacc1[r]);
            iacc1[r] = dp4a_us(v7, as8.w, iacc1[r]);
            if constexpr (residual)
            {
                const uint32_t* as2 = sh_as + (M + r) * slice_stride + (kb << 4);
                uint4 bs0 = *(const uint4*) (as2 + c2);
                uint4 bs8 = *(const uint4*) (as2 + c2 + 8);
                jacc0[r] = dp4a_us(w0, bs0.x, jacc0[r]);
                jacc0[r] = dp4a_us(w1, bs0.y, jacc0[r]);
                jacc0[r] = dp4a_us(w2, bs8.x, jacc0[r]);
                jacc0[r] = dp4a_us(w3, bs8.y, jacc0[r]);
                jacc1[r] = dp4a_us(w4, bs0.x, jacc1[r]);
                jacc1[r] = dp4a_us(w5, bs0.y, jacc1[r]);
                jacc1[r] = dp4a_us(w6, bs8.x, jacc1[r]);
                jacc1[r] = dp4a_us(w7, bs8.y, jacc1[r]);
                jacc0[r] = dp4a_us(v0, bs0.z, jacc0[r]);
                jacc0[r] = dp4a_us(v1, bs0.w, jacc0[r]);
                jacc0[r] = dp4a_us(v2, bs8.z, jacc0[r]);
                jacc0[r] = dp4a_us(v3, bs8.w, jacc0[r]);
                jacc1[r] = dp4a_us(v4, bs0.z, jacc1[r]);
                jacc1[r] = dp4a_us(v5, bs0.w, jacc1[r]);
                jacc1[r] = dp4a_us(v6, bs8.z, jacc1[r]);
                jacc1[r] = dp4a_us(v7, bs8.w, jacc1[r]);
            }
        }

#endif

        r0 = r1;
        r1 = r2;
    }

#if EXL3_VOLTA_LAYOUT
    // The lane pair (l, l ^ 4) holds the two k halves of the same two columns
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        iacc0[r] += __shfl_xor_sync(0xffffffff, iacc0[r], 4);
        iacc1[r] += __shfl_xor_sync(0xffffffff, iacc1[r], 4);
        if constexpr (residual)
        {
            jacc0[r] += __shfl_xor_sync(0xffffffff, jacc0[r], 4);
            jacc1[r] += __shfl_xor_sync(0xffffffff, jacc1[r], 4);
        }
    }
    if (cs == 0)
    {
        const int lp = lane & 15;
        const int nb = nbp * 2 + (lane >> 4);
        const int n0 = nb * 16 + 8 * ((lp >> 1) & 1) + (lp & 1) * 2 + 4 * (lp >> 3);   // column of run 2 * lp; run 2 * lp + 1 is the next one
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            int* acc = accs + r * acc_stride;
            if constexpr (atomic)
            {
                atomicAdd(acc + n0, iacc0[r]);
                atomicAdd(acc + n0 + 1, iacc1[r]);
                if constexpr (residual)
                {
                    atomicAdd(acc + size_n + n0, jacc0[r]);
                    atomicAdd(acc + size_n + n0 + 1, jacc1[r]);
                }
            }
            else
            {
                acc[n0] = iacc0[r];
                acc[n0 + 1] = iacc1[r];
                if constexpr (residual)
                {
                    acc[size_n + n0] = jacc0[r];
                    acc[size_n + n0 + 1] = jacc1[r];
                }
            }
        }
    }
#else
    // Lanes l, l^1 share both n values
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        iacc0[r] += __shfl_xor_sync(0xffffffff, iacc0[r], 1);
        iacc1[r] += __shfl_xor_sync(0xffffffff, iacc1[r], 1);
        if constexpr (residual)
        {
            jacc0[r] += __shfl_xor_sync(0xffffffff, jacc0[r], 1);
            jacc1[r] += __shfl_xor_sync(0xffffffff, jacc1[r], 1);
        }
    }
    if (!(lane & 1))
    {
        int nb = nbp * 2 + (lane >> 4);
        int n0 = nb * 16 + ((lane & 15) >> 1);
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            int* acc = accs + r * acc_stride;
            if constexpr (atomic)
            {
                atomicAdd(acc + n0, iacc0[r]);
                atomicAdd(acc + n0 + 8, iacc1[r]);
                if constexpr (residual)
                {
                    atomicAdd(acc + size_n + n0, jacc0[r]);
                    atomicAdd(acc + size_n + n0 + 8, jacc1[r]);
                }
            }
            else
            {
                acc[n0] = iacc0[r];
                acc[n0 + 8] = iacc1[r];
                if constexpr (residual)
                {
                    acc[size_n + n0] = jacc0[r];
                    acc[size_n + n0 + 8] = jacc1[r];
                }
            }
        }
    }
#endif
}

#if EXL3_VOLTA_LAYOUT
// One k-row of an adjacent block pair, generic K, Volta tile order: lane l decodes run l of each block, one column over the eight
// consecutive k 8 * cs .. 8 * cs + 7 (cs = (l >> 3) & 1); only the A-side accumulators ia0 / ja0 and B-side ib0 / jb0 are used
template <int bits, int M, bool residual>
__device__ __forceinline__ void gemv_int8_pair_row
(
    const uint32_t* blockA, const uint32_t* blockB,
    const uint32_t* as_kb,             // sh_as + (kb << 4)
    int slice_stride, int c2, int t0,
    int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1
)
{
    (void) c2; (void) ia1; (void) ib1; (void) ja1; (void) jb1;
    const int cs = (threadIdx.x >> 3) & 1;
    uint32_t w[8];
    #pragma unroll
    for (int blk = 0; blk < 2; ++blk)
    {
        ext8w<bits>(blk ? blockB : blockA, t0, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) w[j] *= 0x83DCD12Du;
        int* ia = blk ? ib0 : ia0;
        int* ja = blk ? jb0 : ja0;
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const uint32_t* as = as_kb + r * slice_stride + 8 * cs;
            const uint4 a0 = *(const uint4*) as, a1 = *(const uint4*) (as + 4);
            ia[r] = dp4a_us(w[0], a0.x, ia[r]); ia[r] = dp4a_us(w[1], a0.y, ia[r]); ia[r] = dp4a_us(w[2], a0.z, ia[r]); ia[r] = dp4a_us(w[3], a0.w, ia[r]);
            ia[r] = dp4a_us(w[4], a1.x, ia[r]); ia[r] = dp4a_us(w[5], a1.y, ia[r]); ia[r] = dp4a_us(w[6], a1.z, ia[r]); ia[r] = dp4a_us(w[7], a1.w, ia[r]);
            if constexpr (residual)
            {
                const uint32_t* as2 = as_kb + (M + r) * slice_stride + 8 * cs;
                const uint4 b0 = *(const uint4*) as2, b1 = *(const uint4*) (as2 + 4);
                ja[r] = dp4a_us(w[0], b0.x, ja[r]); ja[r] = dp4a_us(w[1], b0.y, ja[r]); ja[r] = dp4a_us(w[2], b0.z, ja[r]); ja[r] = dp4a_us(w[3], b0.w, ja[r]);
                ja[r] = dp4a_us(w[4], b1.x, ja[r]); ja[r] = dp4a_us(w[5], b1.y, ja[r]); ja[r] = dp4a_us(w[6], b1.z, ja[r]); ja[r] = dp4a_us(w[7], b1.w, ja[r]);
            }
        }
    }
}
#else
// One k-row of an adjacent block pair, generic K: extract + dp4a for both blocks from block pointers
// (global or shared memory), M rows sharing the extraction
template <int bits, int M, bool residual>
__device__ __forceinline__ void gemv_int8_pair_row
(
    const uint32_t* blockA, const uint32_t* blockB,
    const uint32_t* as_kb,             // sh_as + (kb << 4)
    int slice_stride, int c2, int t0,
    int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1
)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    ext8w<bits>(blockA, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        const uint32_t* as = as_kb + r * slice_stride;
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        ia0[r] = dp4a_us(w0, as01.x, ia0[r]);
        ia0[r] = dp4a_us(w1, as01.y, ia0[r]);
        ia0[r] = dp4a_us(w2, as89.x, ia0[r]);
        ia0[r] = dp4a_us(w3, as89.y, ia0[r]);
        ia1[r] = dp4a_us(w4, as01.x, ia1[r]);
        ia1[r] = dp4a_us(w5, as01.y, ia1[r]);
        ia1[r] = dp4a_us(w6, as89.x, ia1[r]);
        ia1[r] = dp4a_us(w7, as89.y, ia1[r]);
        if constexpr (residual)
        {
            const uint32_t* as2 = as_kb + (M + r) * slice_stride;
            uint2 bs01 = *(const uint2*) (as2 + c2);
            uint2 bs89 = *(const uint2*) (as2 + c2 + 8);
            ja0[r] = dp4a_us(w0, bs01.x, ja0[r]);
            ja0[r] = dp4a_us(w1, bs01.y, ja0[r]);
            ja0[r] = dp4a_us(w2, bs89.x, ja0[r]);
            ja0[r] = dp4a_us(w3, bs89.y, ja0[r]);
            ja1[r] = dp4a_us(w4, bs01.x, ja1[r]);
            ja1[r] = dp4a_us(w5, bs01.y, ja1[r]);
            ja1[r] = dp4a_us(w6, bs89.x, ja1[r]);
            ja1[r] = dp4a_us(w7, bs89.y, ja1[r]);
        }
    }

    ext8w<bits>(blockB, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        const uint32_t* as = as_kb + r * slice_stride;
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        ib0[r] = dp4a_us(w0, as01.x, ib0[r]);
        ib0[r] = dp4a_us(w1, as01.y, ib0[r]);
        ib0[r] = dp4a_us(w2, as89.x, ib0[r]);
        ib0[r] = dp4a_us(w3, as89.y, ib0[r]);
        ib1[r] = dp4a_us(w4, as01.x, ib1[r]);
        ib1[r] = dp4a_us(w5, as01.y, ib1[r]);
        ib1[r] = dp4a_us(w6, as89.x, ib1[r]);
        ib1[r] = dp4a_us(w7, as89.y, ib1[r]);
        if constexpr (residual)
        {
            const uint32_t* as2 = as_kb + (M + r) * slice_stride;
            uint2 bs01 = *(const uint2*) (as2 + c2);
            uint2 bs89 = *(const uint2*) (as2 + c2 + 8);
            jb0[r] = dp4a_us(w0, bs01.x, jb0[r]);
            jb0[r] = dp4a_us(w1, bs01.y, jb0[r]);
            jb0[r] = dp4a_us(w2, bs89.x, jb0[r]);
            jb0[r] = dp4a_us(w3, bs89.y, jb0[r]);
            jb1[r] = dp4a_us(w4, bs01.x, jb1[r]);
            jb1[r] = dp4a_us(w5, bs01.y, jb1[r]);
            jb1[r] = dp4a_us(w6, bs89.x, jb1[r]);
            jb1[r] = dp4a_us(w7, bs89.y, jb1[r]);
        }
    }
}
#endif

#if EXL3_VOLTA_LAYOUT
// Volta tile order: the lane pair (l, l ^ 8) holds the two k halves of one column; the lanes with cs == 0 write the column of each block
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_pair_tail
(
    int* __restrict__ accs, size_t acc_stride, int nbp, int lane, int size_n,
    int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1
)
{
    (void) ia1; (void) ib1; (void) ja1; (void) jb1;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        ia0[r] += __shfl_xor_sync(0xffffffff, ia0[r], 8);
        ib0[r] += __shfl_xor_sync(0xffffffff, ib0[r], 8);
        if constexpr (residual)
        {
            ja0[r] += __shfl_xor_sync(0xffffffff, ja0[r], 8);
            jb0[r] += __shfl_xor_sync(0xffffffff, jb0[r], 8);
        }
    }
    if (((lane >> 3) & 1) == 0)
    {
        const int nA = (nbp * 2) * 16 + 8 * ((lane >> 2) & 1) + (lane & 3) + 4 * (lane >> 4);
        const int nB = nA + 16;
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            int* acc = accs + r * acc_stride;
            if constexpr (atomic)
            {
                atomicAdd(acc + nA, ia0[r]);
                atomicAdd(acc + nB, ib0[r]);
                if constexpr (residual)
                {
                    atomicAdd(acc + size_n + nA, ja0[r]);
                    atomicAdd(acc + size_n + nB, jb0[r]);
                }
            }
            else
            {
                acc[nA] = ia0[r];
                acc[nB] = ib0[r];
                if constexpr (residual)
                {
                    acc[size_n + nA] = ja0[r];
                    acc[size_n + nB] = jb0[r];
                }
            }
        }
    }
}
#else
// Shared reduction tail for the pair-per-warp generic units (atomic, or exclusive plain stores for
// the per-slice partials of the sq kernel): four lanes share each n
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_pair_tail
(
    int* __restrict__ accs, size_t acc_stride, int nbp, int lane, int size_n,
    int* ia0, int* ia1, int* ib0, int* ib1,
    int* ja0, int* ja1, int* jb0, int* jb1
)
{
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        #pragma unroll
        for (int o = 1; o < 4; o <<= 1)
        {
            ia0[r] += __shfl_xor_sync(0xffffffff, ia0[r], o);
            ia1[r] += __shfl_xor_sync(0xffffffff, ia1[r], o);
            ib0[r] += __shfl_xor_sync(0xffffffff, ib0[r], o);
            ib1[r] += __shfl_xor_sync(0xffffffff, ib1[r], o);
            if constexpr (residual)
            {
                ja0[r] += __shfl_xor_sync(0xffffffff, ja0[r], o);
                ja1[r] += __shfl_xor_sync(0xffffffff, ja1[r], o);
                jb0[r] += __shfl_xor_sync(0xffffffff, jb0[r], o);
                jb1[r] += __shfl_xor_sync(0xffffffff, jb1[r], o);
            }
        }
    }
    if ((lane & 3) == 0)
    {
        int nA = (nbp * 2) * 16 + (lane >> 2);
        int nB = nA + 16;
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            int* acc = accs + r * acc_stride;
            if constexpr (atomic)
            {
                atomicAdd(acc + nA, ia0[r]);
                atomicAdd(acc + nA + 8, ia1[r]);
                atomicAdd(acc + nB, ib0[r]);
                atomicAdd(acc + nB + 8, ib1[r]);
                if constexpr (residual)
                {
                    atomicAdd(acc + size_n + nA, ja0[r]);
                    atomicAdd(acc + size_n + nA + 8, ja1[r]);
                    atomicAdd(acc + size_n + nB, jb0[r]);
                    atomicAdd(acc + size_n + nB + 8, jb1[r]);
                }
            }
            else
            {
                acc[nA] = ia0[r];
                acc[nA + 8] = ia1[r];
                acc[nB] = ib0[r];
                acc[nB + 8] = ib1[r];
                if constexpr (residual)
                {
                    acc[size_n + nA] = ja0[r];
                    acc[size_n + nA + 8] = ja1[r];
                    acc[size_n + nB] = jb0[r];
                    acc[size_n + nB + 8] = jb1[r];
                }
            }
        }
    }
}
#endif

// Narrow generic unit (any K): one (256-column x k-slice) unit, warp per adjacent block pair
// processed sequentially with pointer-based extraction straight from global memory
template <int bits, int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_narrow
(
    const uint16_t* __restrict__ B,
    int* __restrict__ accs,
    size_t acc_stride,
    const uint32_t* __restrict__ sh_as,
    int slice_stride,
    int nb256,
    int kb0,
    int nrows,
    int size_n
)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n * bits / 2;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * (bits * 16);
    int c2 = 2 * (lane & 3);
    int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
    int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

    for (int kb = 0; kb < nrows; ++kb)
    {
        const uint32_t* blockA = bp + (size_t) kb * row_stride;
        gemv_int8_pair_row<bits, M, residual>(blockA, blockA + 8 * bits,
            sh_as + (kb << 4), slice_stride, c2, lane << 3,
            ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
    }
    gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
}

// Smem-staged generic unit: the warp stages its block pair's rows into a warp-private shared memory
// slice with cp.async (coalesced 16 B chunks, one commit group per row, GEMV_STAGE_D rows deep) and
// extracts from shared memory. Used for the K where scattered pointer extraction leaves the most
// load latency exposed (3, 5, 7); warp-private slices need no block-level synchronization.
template <int bits, int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_smem
(
    const uint16_t* __restrict__ B,
    int* __restrict__ accs,
    size_t acc_stride,
    const uint32_t* __restrict__ sh_as,
    int slice_stride,
    uint32_t* __restrict__ sh_b,
    int nb256,
    int kb0,
    int nrows,
    int size_n
)
{
    constexpr int D = GEMV_STAGE_D;
    constexpr int pairwords = 16 * bits;
    constexpr int chunks = pairwords / 4;      // 16 B cp.async chunks per pair row
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = size_n * bits / 2;
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * pairwords;
    uint32_t* sb = sh_b + warp * (D * pairwords);

    auto stage_row = [&] (int kb)
    {
        if (kb < nrows && lane < chunks)
            cp_async(sb + (kb % D) * pairwords + lane * 4, bp + (size_t) kb * row_stride + lane * 4);
        cp_async_fence();
    };
    #pragma unroll
    for (int r = 0; r < D - 1; ++r) stage_row(r);

    int c2 = 2 * (lane & 3);
    int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
    int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

    for (int kb = 0; kb < nrows; ++kb)
    {
        cp_async_wait<D - 2>();
        // Also orders the previous iteration's smem reads (all lanes) before the overwrite below
        __syncwarp();
        stage_row(kb + D - 1);

        const uint32_t* blockA = sb + (kb % D) * pairwords;
        gemv_int8_pair_row<bits, M, residual>(blockA, blockA + 8 * bits,
            sh_as + (kb << 4), slice_stride, c2, lane << 3,
            ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
    }
    gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
}

// K values routed to the smem-staged unit (measured on 3090: K=3 -13%; narrow wins for 2/6/8 which
// are at the ALU floor / DRAM-bound, wide covers 4)
__host__ __device__ constexpr bool gemv_int8_stage_smem(int bits)
{
    return bits == 3 || bits == 5 || bits == 7;
}

// ---------------------------------------------------------------------------------------------------------
// Per-slice-scale ("sq") kernel for m == 1: kills ALL global pre-GEMV coordination. Each k-slice is
// Hadamard-transformed, maxed and quantized block-locally during staging with its own
// q_s = slicemax/127 (deterministic, so duplicating blocks agree bit-exactly - and strictly finer
// quantization than a global q), units write exact per-slice int32 partials with plain stores
// (exclusive writers - no atomics, no accumulator zeroing), and a per-256-column completion counter
// gates an inline epilogue that combines sum_s q_s*acc_s in fixed slice order (deterministic) plus
// the per-slice affine corrections. One REGULAR launch: no cooperative machinery, no grid.sync, no
// A_had traffic. Same 10 kernel arguments (A_had unused). Workspace layout (fixed offsets so mixed
// sq/coop calls of any shape can share the buffer):
//   [counters: SQ_COUNTERS_CAP][{q_s, s1_s, s2_s, pad} x SQ_KSPLIT_CAP][partials: ksplit x pstride]
// The coop kernels use the region beyond SQ_WS_RESERVED. Counters are zeroed when the workspace is
// (re)allocated and reset by the finishing block, so they stay zero between calls.

// Stage one slice for M activation rows: Hadamard from A into sh_ah, slice max -> q_s, splats +
// exact sums per row. Bit-identical in every block that stages the same slice. Rows >= size_m
// (padding at M = 4, m = 3) get zero splats and dummy scales.
template <int M, bool residual>
__device__ __forceinline__ void gemv_int8_stage_slice
(
    const half* __restrict__ A,
    int size_m,
    int size_k,
    const half* __restrict__ suh,
    float* __restrict__ qs,            // qsums + 4 * slice * M, rows consecutive, published (idempotent)
    half* __restrict__ sh_ah,
    uint32_t* __restrict__ sh_as,      // M regions of slice_stride words (+ M more for residual)
    int slice_stride,
    float* __restrict__ sh_red,        // 33 floats
    int kb0,
    int nrows
)
{
    int t = threadIdx.x;
    int nel = nrows * 16;
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        uint32_t* as = sh_as + r * slice_stride;
        uint32_t* as2 = sh_as + (M + r) * slice_stride;
        float* qsr = qs + 4 * r;
        __syncthreads();
        if (r >= size_m)
        {
            for (int i = t; i < nel; i += NUM_THREADS)
            {
                as[i] = 0;
                if constexpr (residual) as2[i] = 0;
            }
            if (t == 0)
            {
                qsr[0] = 1.0f;
                ((int*) qsr)[1] = 0;
                ((int*) qsr)[2] = 0;
            }
            continue;
        }
        const half* Ar = A + (size_t) r * size_k;
        for (int sp = t >> 5; sp < (nel >> 7); sp += NUM_THREADS >> 5)
            had_hf_r_128_inner<true, false>(Ar + (kb0 << 4) + (sp << 7), sh_ah + (sp << 7), suh + (kb0 << 4) + (sp << 7), 0.088388347648f);
        __syncthreads();

        float mx = 0.0f;
        for (int i = t; i < nel; i += NUM_THREADS) mx = fmaxf(mx, fabsf(__half2float(sh_ah[i])));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        if ((t & 31) == 0) sh_red[t >> 5] = mx;
        __syncthreads();
        if (t < 32)
        {
            float v = t < (NUM_THREADS >> 5) ? sh_red[t] : 0.0f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
            if (t == 0) sh_red[32] = fmaxf(v, 1e-8f) / 127.0f;
        }
        __syncthreads();
        float q_s = sh_red[32];

        float rq = 1.0f / q_s;
        float rq2 = rq * 254.0f;
        int l1 = 0, l2 = 0;
        for (int i = t; i < nel; i += NUM_THREADS)
        {
            float a = __half2float(sh_ah[i]);
            int v = __float2int_rn(a * rq);
            v = max(-127, min(127, v));
            as[i] = ((uint32_t)(uint8_t)(int8_t) v) * 0x01010101u;
            l1 += v;
            if constexpr (residual)
            {
                float rr = a - q_s * (float) v;
                int v2 = __float2int_rn(rr * rq2);
                v2 = max(-127, min(127, v2));
                as2[i] = ((uint32_t)(uint8_t)(int8_t) v2) * 0x01010101u;
                l2 += v2;
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            l1 += __shfl_xor_sync(0xffffffff, l1, o);
            l2 += __shfl_xor_sync(0xffffffff, l2, o);
        }
        if ((t & 31) == 0) { ((int*) sh_red)[t >> 5] = l1; sh_red[16 + (t >> 5)] = __int_as_float(l2); }
        __syncthreads();
        if (t < 32)
        {
            int v1 = t < (NUM_THREADS >> 5) ? ((int*) sh_red)[t] : 0;
            int v2 = t < (NUM_THREADS >> 5) ? __float_as_int(sh_red[16 + t]) : 0;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                v1 += __shfl_xor_sync(0xffffffff, v1, o);
                v2 += __shfl_xor_sync(0xffffffff, v2, o);
            }
            if (t == 0)
            {
                qsr[0] = sh_red[32];
                ((int*) qsr)[1] = v1;
                ((int*) qsr)[2] = v2;
            }
        }
    }
    __syncthreads();
}

// Epilogue for one 256-column group: deterministic fixed-order combine over the per-slice partials,
// warp per (row, 128-span). Reads bypass L1 (__ldcg): the contributions arrived from other blocks
// with no grid-wide barrier.
template <int M, bool c_fp32, bool residual>
__device__ __forceinline__ void gemv_int8_epilogue_group_sq
(
    const int* __restrict__ partials,
    const float* __restrict__ qsums,
    int pstride,
    int ksplit,
    int size_m,
    void* __restrict__ C,
    const half* __restrict__ svh,
    float* __restrict__ sh_tmp,
    int nb256,
    int size_n
)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = warp >> 1;
    if (warp >= 2 * M || row >= size_m) return;
    float k_inv  = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    float aff = 1024.0f * k_inv + k_bias;

    int base = nb256 * 256 + (warp & 1) * 128;
    float* tmp = sh_tmp + warp * 128;
    float acc[4] = {};
    float corr = 0.0f;
    for (int sl = 0; sl < ksplit; ++sl)
    {
        int idx = sl * M + row;
        float q_s = qsums[4 * idx];
        float suma = q_s * (float) ((const int*) qsums)[4 * idx + 1];
        const int* p = partials + (size_t) idx * pstride + base;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[i] += q_s * (float) __ldcg(p + lane * 4 + i);
        if constexpr (residual)
        {
            float q2_s = q_s * (1.0f / 254.0f);
            suma += q2_s * (float) ((const int*) qsums)[4 * idx + 2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[i] += q2_s * (float) __ldcg(p + size_n + lane * 4 + i);
        }
        corr += aff * suma;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        tmp[lane * 4 + i] = k_inv * acc[i] + corr;
    __syncwarp();
    if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>(tmp, ((float*) C) + (size_t) row * size_n + base, svh + base, 0.088388347648f);
    else
        had_fh_r_128_inner<false, true>(tmp, ((half*) C) + (size_t) row * size_n + base, svh + base, 0.088388347648f);
}

// Shared-memory cap for the sq decomposition: row halfs (32 B/row) + M splat regions (64 B/row each,
// x2 residual), within ~80 KB so the stage region and epilogue staging still fit under the opt-in max
__host__ __device__ constexpr int gemv_int8_sq_rows_max(int M, bool residual)
{
    int cap = (80 * 1024) / (32 + 64 * M * (residual ? 2 : 1));
    cap &= ~7;
    return cap < SQ_ROWS_MAX ? cap : SQ_ROWS_MAX;
}

template <int bits, int M, bool c_fp32, bool residual, bool stage_smem>
__global__ __launch_bounds__(NUM_THREADS)
void exl3_gemv_int8_sq_kernel
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,       // 1 <= size_m <= M
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ suh,
    half* __restrict__ A_had,
    const half* __restrict__ svh
)
{
    extern __shared__ uint32_t shmem[];

    // Work decomposition: rows_per multiple of 8 (whole 128-spans for the local Hadamard); the host
    // mirrors this for the shared memory size and workspace bound. Swept on 3090: the wall time is
    // ~one unit's duration as long as all (slice x 256-column) units fit in one wave of the grid, so
    // take the smallest slice height with units <= gridDim.x - but at least half-wave occupancy up
    // to 32 rows, since very short slices inflate ksplit and the epilogue's serial per-slice combine
    int rows_total = size_k >> 4;
    int nb256_total = size_n / 256;
    int r = CEIL_DIVIDE(rows_total * nb256_total, (int) gridDim.x);
    int rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
    rows_per = MAX(rows_per, SQ_MINROWS);
    rows_per = MIN(rows_per, gemv_int8_sq_rows_max(M, residual));
    rows_per = MIN(rows_per, (rows_total + 7) & ~7);
    int ksplit = CEIL_DIVIDE(rows_total, rows_per);
    int units = nb256_total * ksplit;
    int slice_stride = rows_per * 16;
    int pstride = size_n * (residual ? 2 : 1);

    int* counters = locks;
    float* qsums = (float*) (locks + SQ_COUNTERS_CAP);
    int* partials = locks + SQ_WS_RESERVED;

    // Shared layout: [slice halfs][M x splats (+ M x residual splats)][B stage][epilogue tmp]
    half* sh_ah = (half*) shmem;
    uint32_t* sh_as = shmem + rows_per * 8;
    uint32_t* sh_b = sh_as + slice_stride * M * (residual ? 2 : 1);
    float* sh_tmp = (float*) (sh_b + (stage_smem ? 8 * GEMV_STAGE_D * 16 * bits : 0));
    __shared__ float sh_red[33];
    __shared__ int sh_last;

    int t = threadIdx.x;
    int prev_slice = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x)
    {
        int slice = unit / nb256_total;
        int nb256 = unit % nb256_total;
        int kb0 = slice * rows_per;
        int nrows = MIN(rows_per, rows_total - kb0);
        if (slice != prev_slice)
        {
            gemv_int8_stage_slice<M, residual>(A, size_m, size_k, suh, qsums + 4 * slice * M,
                                               sh_ah, sh_as, slice_stride, sh_red, kb0, nrows);
            prev_slice = slice;
        }
        int* pacc = partials + (size_t) slice * M * pstride;
        if constexpr (bits == 4)
            gemv_int8_unit_wide<M, residual, false>(B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);
        else if constexpr (stage_smem)
            gemv_int8_unit_smem<bits, M, residual, false>(B, pacc, pstride, sh_as, slice_stride, sh_b, nb256, kb0, nrows, size_n);
        else
            gemv_int8_unit_narrow<bits, M, residual, false>(B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);

        // Completion counter: the ksplit-th contributor runs the epilogue for this 256-column group.
        // (sh_last reuse across iterations is ordered by the next iteration's __syncthreads.)
        __threadfence();
        __syncthreads();
        if (t == 0) sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
        __syncthreads();
        if (sh_last)
        {
            gemv_int8_epilogue_group_sq<M, c_fp32, residual>(partials, qsums, pstride, ksplit, size_m,
                                                             C, svh, sh_tmp, nb256, size_n);
            if (t == 0) counters[nb256] = 0;
        }
    }
}

