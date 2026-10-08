#pragma once

// int8-GEMV-class restructure of the decode (size_m == 1) exl3_mgemm path for sm_70.
//
// The production decode-class call (m=1 per slot, batched experts, K2) ran on the tile pipeline
// (512-thread HMMA shape kernels): 90 KB smem ring staged through sm70's register-staged
// synchronous cp.async gives zero load-use distance, and REG:128/STACK:48 at 1 block/SM leaves
// no occupancy to hide the exposed latency (SM70_DEQUANT_DQV2.md forensics: ~16.8 us of
// synchronous global-load exposure of ~57 us, frag_b 3-stage pipeline fully spilled).
//
// This kernel keeps the mgemm shell (index packing, per-slot input Hadamard, grouped weighted
// reduction) bit-for-bit in structure, and replaces the inner matmul with a GEMV-class loop
// modeled on exl3_gemv_int8_kernel.cuh:
//   - 256-thread blocks, ~30 registers, 2+ blocks/SM (register diet: no frag_b pipeline at all)
//   - B streamed straight from global to registers with a 2-row k16 lookahead (load-use
//     distance >= 2 rows instead of 0)
//   - warp w owns n16-block w of the block's 128 columns for the FULL k column; per k16 row the
//     lane decodes its 8 trellis windows with the production dq8_aligned_* extraction (same
//     window-to-fragment contract as the mma path, so frag0 -> column lane>>2, frag1 -> +8) and
//     FFMA against its A k-set {2*tig, 2*tig+1, 2*tig+8, 2*tig+9} staged once in shared memory
//   - fp32 accumulation (same class as the f32-acc HMMA path), 4-lane butterfly reduce, direct
//     output write; no B fragment shuffles, no mma, no block-wide stage barriers, no locks
//
// The output row per slot is written untransformed; the shell's output Hadamard + svh scale +
// B_weights grouped reduction run unchanged, and the summation order is fixed per launch.


#include "util.cuh"
#include "ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "trellis_context.h"

#include <cooperative_groups.h>
namespace cg_gv = cooperative_groups;

#define GV_NUM_THREADS 256
#define GV_WARPS_N 8
#define GV_COLS_PER_BLOCK (GV_WARPS_N * 16)
#define GV_MAX_SLOTS_PER_GROUP 4
#define GV_MAX_BSZM 512
#define GV_MAX_UNIQUE_GROUPS GV_MAX_BSZM

// Private packing scratch (per-TU device globals; same protocol as exl3_gemm_kernel.cuh's
// v_indices/v_weights/gv_bszm_sync but never aliased with the tile-pipeline kernels' copies)
__device__ int64_t gv_v_indices[GV_MAX_BSZM];
__device__ half gv_v_weights[GV_MAX_BSZM];
__device__ int gv_bszm_sync;

__device__ int gv_num_unique;
__device__ int gv_unique_mat[GV_MAX_UNIQUE_GROUPS];
__device__ int gv_unique_cnt[GV_MAX_UNIQUE_GROUPS];
__device__ int gv_unique_slots[GV_MAX_UNIQUE_GROUPS][GV_MAX_SLOTS_PER_GROUP];

template <int bits, bool c_fp32, int cb, bool v2>
__global__ __launch_bounds__(GV_NUM_THREADS) void exl3_mgemm_gv_kernel(EXL3_MGEMM_ARGS);

void* get_mgemm_gv_kernel(int K, bool c_fp32, int cb);
void* get_mgemm_gv2_kernel(int K, bool c_fp32, int cb);

template <int bits, bool c_fp32, int cb, bool v2>
__global__ __launch_bounds__(GV_NUM_THREADS) void exl3_gemm_gv_kernel(EXL3_GEMM_ARGS);

void* get_gemm_gv_kernel(int K, bool c_fp32, int cb);
void* get_gemm_gv2_kernel(int K, bool c_fp32, int cb);

// ---------------------------------------------------------------------------------------------------------
// dq8 window extraction from register-resident words (bodies mirror dq8_aligned_2bits /
// dq8_aligned_4bits in exl3_dq.cuh; only the shared-memory loads are replaced by the
// lookahead registers)

template <int cb>
__device__ __forceinline__ void gv_dq8_2bits_regs(uint32_t merged, FragB& frag0, FragB& frag1)
{
    uint32_t w7 = merged & 0xffff;
    uint32_t w0, w1, w2, w3, w4, w5, w6;
    BFE16_IMM(w6, merged, 2);
    BFE16_IMM(w5, merged, 4);
    BFE16_IMM(w4, merged, 6);
    BFE16_IMM(w3, merged, 8);
    BFE16_IMM(w2, merged, 10);
    BFE16_IMM(w1, merged, 12);
    BFE16_IMM(w0, merged, 14);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void gv_dq8_4bits_regs(uint32_t prev, uint32_t cur, FragB& frag0, FragB& frag1)
{
    uint32_t s;
    FSHF_IMM(s, cur, prev, 20);
    uint32_t w7 = cur & 0xffff;
    uint32_t w0, w1, w2, w3, w4, w5, w6;
    BFE16_IMM(w6, cur, 4);
    BFE16_IMM(w5, cur, 8);
    BFE16_IMM(w4, cur, 12);
    BFE16_IMM(w3, cur, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

// One k16 row: decode + FFMA into the lane's two fp32 column accumulators.
// Fragment contract (m16n8k16 B, col-major): lane l holds B[k][col] for
// k in {2*tig, 2*tig+1, 2*tig+8, 2*tig+9}, tig = l & 3, col = l >> 2 (frag0) / col + 8 (frag1).
// A k-set is the matching A[row=0] half2 pairs, staged in shared memory.
template <int cb>
__device__ __forceinline__ void gv_row_ffma(
    FragB& frag0, FragB& frag1,
    const half2* sh_a2, int kb, int tig,
    float& c0, float& c1)
{
    half2 a0 = sh_a2[(kb << 3) + tig];        // A[2*tig], A[2*tig+1]
    half2 a1 = sh_a2[(kb << 3) + tig + 4];    // A[2*tig+8], A[2*tig+9]
    float2 af0 = __half22float2(a0);
    float2 af1 = __half22float2(a1);
    float2 f00 = __half22float2(frag0[0]);
    float2 f01 = __half22float2(frag0[1]);
    float2 f10 = __half22float2(frag1[0]);
    float2 f11 = __half22float2(frag1[1]);
    c0 = fmaf(af0.x, f00.x, c0);
    c0 = fmaf(af0.y, f00.y, c0);
    c0 = fmaf(af1.x, f01.x, c0);
    c0 = fmaf(af1.y, f01.y, c0);
    c1 = fmaf(af0.x, f10.x, c1);
    c1 = fmaf(af0.y, f10.y, c1);
    c1 = fmaf(af1.x, f11.x, c1);
    c1 = fmaf(af1.y, f11.y, c1);
}

// GEMV inner for one slot (size_m == 1). blockIdx.y splits k; parts merge through the C row
// with the tile-pipeline kernel's deterministic reverse-stage lock protocol. blockIdx.x covers
// an n128 chunk of the (max) width, one warp per n16 block for the part's k range.
// Arow: the slot's Hadamard-transformed input row (size_k halves). Writes C_ row directly.
// lock: this block's per-column-chunk counter (reset by the finalizing part).
template <int bits, int cb, bool c_fp32, bool v2, int NUM_S>
inline __device__ void exl3_mgemm_gv_inner_impl(
    const half* const* __restrict__ Arows,
    const uint16_t* __restrict__ B,
    void* const* __restrict__ C_ptrs,
    const int size_k,
    const int size_n,
    int* const* __restrict__ locks)
{
    extern __shared__ uint32_t gv_shmem[];
    half2* sh_a2 = (half2*) gv_shmem;
    float2* sh_af = (float2*) gv_shmem;

    const int t = threadIdx.x;
    #pragma unroll
    for (int s = 0; s < NUM_S; ++s)
    {
        const uint4* src = (const uint4*) Arows[s];
        if constexpr (v2 && bits == 2)
        {
            // A staged as float (exact widening of the half values, so downstream math is
            // bit-identical to the half2-staged path) - kills the per-row A-side CVTs
            float* dstf = ((float*) gv_shmem) + (size_t) s * size_k;
            for (int i = t; i < size_k / 8; i += GV_NUM_THREADS)
            {
                uint4 v = src[i];
                const half* h = (const half*) &v;
                float* d = dstf + i * 8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) d[j] = __half2float(h[j]);
            }
        }
        else
        {
            uint4* dst = (uint4*) (sh_a2 + (size_t) s * (size_k / 2));
            for (int i = t; i < size_k / 8; i += GV_NUM_THREADS) dst[i] = src[i];
        }
    }
    __syncthreads();

    const int warp_id = t >> 5;
    const int lane = t & 31;
    const int tig = lane & 3;
    const int rows_total = size_k >> 4;
    const int rows_per = CEIL_DIVIDE(rows_total, gridDim.y);
    const int kb0 = blockIdx.y * rows_per;
    const int kb1 = MIN(kb0 + rows_per, rows_total);
    const int n16_total = size_n >> 4;
    const int nb = blockIdx.x * GV_WARPS_N + warp_id;
    const bool wvalid = nb < n16_total;   // per-matrix width < max grid width (size_n_list path)
    const uint32_t* bp = (const uint32_t*) B + (size_t) nb * (8 * bits);
    const int rstride = n16_total * (8 * bits);

    float c0[NUM_S];
    float c1[NUM_S];
    #pragma unroll
    for (int s = 0; s < NUM_S; ++s)
    {
        c0[s] = 0.0f;
        c1[s] = 0.0f;
    }

    if constexpr (bits == 2)
    {
        if (wvalid)
        {
        const int i1 = lane >> 1;
        const int i0 = (i1 + 15) & 15;
        const int src_lane = i0 << 1;
        const int shift = (lane & 1) ? 0 : 16;
        constexpr int L = 8;

        // One coalesced u32 load per lane per row (lanes 2m, 2m+1 share u32[m]); the preceding
        // word arrives via one shuffle; 8-deep register pipeline gives the memory pipe real
        // load-use distance on sm70 (zero-distance was the dominant mgemm forensics term)
        if constexpr (!v2)
        {
        uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0, w7 = 0;
        auto load_word = [&] (int kb, uint32_t& r)
        {
            if (kb < kb1) r = bp[(size_t) kb * rstride + i1];
        };
        load_word(kb0, w0); load_word(kb0 + 1, w1); load_word(kb0 + 2, w2); load_word(kb0 + 3, w3);
        load_word(kb0 + 4, w4); load_word(kb0 + 5, w5); load_word(kb0 + 6, w6); load_word(kb0 + 7, w7);
        for (int kb = kb0; kb < kb1; ++kb)
        {
            uint32_t wn = 0;
            if (kb + L < kb1) wn = bp[(size_t) (kb + L) * rstride + i1];
            FragB f0, f1;
            uint32_t prev = __shfl_sync(0xffffffffu, w0, src_lane);
            uint32_t merged = fshift(w0, prev, shift);
            gv_dq8_2bits_regs<cb>(merged, f0, f1);
            float2 f00 = __half22float2(f0[0]);
            float2 f01 = __half22float2(f0[1]);
            float2 f10 = __half22float2(f1[0]);
            float2 f11 = __half22float2(f1[1]);
            #pragma unroll
            for (int s = 0; s < NUM_S; ++s)
            {
                const half2* sh_s = sh_a2 + (size_t) s * (size_k / 2);
                half2 a0 = sh_s[(kb << 3) + tig];
                half2 a1 = sh_s[(kb << 3) + tig + 4];
                float2 af0 = __half22float2(a0);
                float2 af1 = __half22float2(a1);
                c0[s] = fmaf(af0.x, f00.x, c0[s]);
                c0[s] = fmaf(af0.y, f00.y, c0[s]);
                c0[s] = fmaf(af1.x, f01.x, c0[s]);
                c0[s] = fmaf(af1.y, f01.y, c0[s]);
                c1[s] = fmaf(af0.x, f10.x, c1[s]);
                c1[s] = fmaf(af0.y, f10.y, c1[s]);
                c1[s] = fmaf(af1.x, f11.x, c1[s]);
                c1[s] = fmaf(af1.y, f11.y, c1[s]);
            }
            w0 = w1; w1 = w2; w2 = w3; w3 = w4; w4 = w5; w5 = w6; w6 = w7; w7 = wn;
        }
        }
        else
        {
        const float2* sh_af2 = sh_af;
        uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0, w7 = 0;
        auto load_word = [&] (int kb, uint32_t& r)
        {
            if (kb < kb1) r = bp[(size_t) kb * rstride + i1];
        };
        auto row = [&] (int kb, uint32_t wcur)
        {
            uint32_t prev = __shfl_sync(0xffffffffu, wcur, src_lane);
            uint32_t merged = fshift(wcur, prev, shift);
            FragB f0, f1;
            gv_dq8_2bits_regs<cb>(merged, f0, f1);
            float2 f00 = __half22float2(f0[0]);
            float2 f01 = __half22float2(f0[1]);
            float2 f10 = __half22float2(f1[0]);
            float2 f11 = __half22float2(f1[1]);
            #pragma unroll
            for (int s = 0; s < NUM_S; ++s)
            {
                const float2* sh_s = sh_af2 + (size_t) s * (size_k / 2);
                float2 af0 = sh_s[(kb << 3) + tig];
                float2 af1 = sh_s[(kb << 3) + tig + 4];
                c0[s] = fmaf(af0.x, f00.x, c0[s]);
                c0[s] = fmaf(af0.y, f00.y, c0[s]);
                c0[s] = fmaf(af1.x, f01.x, c0[s]);
                c0[s] = fmaf(af1.y, f01.y, c0[s]);
                c1[s] = fmaf(af0.x, f10.x, c1[s]);
                c1[s] = fmaf(af0.y, f10.y, c1[s]);
                c1[s] = fmaf(af1.x, f11.x, c1[s]);
                c1[s] = fmaf(af1.y, f11.y, c1[s]);
            }
        };
        load_word(kb0, w0); load_word(kb0 + 1, w1); load_word(kb0 + 2, w2); load_word(kb0 + 3, w3);
        load_word(kb0 + 4, w4); load_word(kb0 + 5, w5); load_word(kb0 + 6, w6); load_word(kb0 + 7, w7);
        int kb = kb0;
        // full batches where the following batch is also full: next loads unconditional
        for (; kb + 2 * L <= kb1; kb += L)
        {
            uint32_t n0, n1, n2, n3, n4, n5, n6, n7;
            const uint32_t* np = bp + (size_t) (kb + L) * rstride + i1;
            n0 = np[0]; n1 = np[rstride]; n2 = np[2 * rstride]; n3 = np[3 * rstride];
            n4 = np[4 * rstride]; n5 = np[5 * rstride]; n6 = np[6 * rstride]; n7 = np[7 * rstride];
            row(kb + 0, w0); row(kb + 1, w1); row(kb + 2, w2); row(kb + 3, w3);
            row(kb + 4, w4); row(kb + 5, w5); row(kb + 6, w6); row(kb + 7, w7);
            w0 = n0; w1 = n1; w2 = n2; w3 = n3; w4 = n4; w5 = n5; w6 = n6; w7 = n7;
        }
        // last full batch: next loads predicated (at most one pass)
        for (; kb + L <= kb1; kb += L)
        {
            uint32_t n0 = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0, n5 = 0, n6 = 0, n7 = 0;
            load_word(kb + L + 0, n0); load_word(kb + L + 1, n1); load_word(kb + L + 2, n2); load_word(kb + L + 3, n3);
            load_word(kb + L + 4, n4); load_word(kb + L + 5, n5); load_word(kb + L + 6, n6); load_word(kb + L + 7, n7);
            row(kb + 0, w0); row(kb + 1, w1); row(kb + 2, w2); row(kb + 3, w3);
            row(kb + 4, w4); row(kb + 5, w5); row(kb + 6, w6); row(kb + 7, w7);
            w0 = n0; w1 = n1; w2 = n2; w3 = n3; w4 = n4; w5 = n5; w6 = n6; w7 = n7;
        }
        // scalar tail
        for (; kb < kb1; ++kb)
        {
            uint32_t wn = 0;
            if (kb + L < kb1) wn = bp[(size_t) (kb + L) * rstride + i1];
            row(kb, w0);
            w0 = w1; w1 = w2; w2 = w3; w3 = w4; w4 = w5; w5 = w6; w6 = w7; w7 = wn;
        }
        }
        }
    }
    else if constexpr (bits == 4)
    {
        if (wvalid)
        {
        const int i1 = lane;
        const int i0 = (i1 + 31) & 31;
        const int src_lane = i0;
        constexpr int L = 8;
        uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0, w7 = 0;
        auto load_word = [&] (int kb, uint32_t& r)
        {
            if (kb < kb1) r = bp[(size_t) kb * rstride + i1];
        };
        load_word(kb0, w0); load_word(kb0 + 1, w1); load_word(kb0 + 2, w2); load_word(kb0 + 3, w3);
        load_word(kb0 + 4, w4); load_word(kb0 + 5, w5); load_word(kb0 + 6, w6); load_word(kb0 + 7, w7);
        for (int kb = kb0; kb < kb1; ++kb)
        {
            uint32_t wn = 0;
            if (kb + L < kb1) wn = bp[(size_t) (kb + L) * rstride + i1];
            FragB f0, f1;
            uint32_t prev = __shfl_sync(0xffffffffu, w0, src_lane);
            gv_dq8_4bits_regs<cb>(prev, w0, f0, f1);
            float2 f00 = __half22float2(f0[0]);
            float2 f01 = __half22float2(f0[1]);
            float2 f10 = __half22float2(f1[0]);
            float2 f11 = __half22float2(f1[1]);
            #pragma unroll
            for (int s = 0; s < NUM_S; ++s)
            {
                const half2* sh_s = sh_a2 + (size_t) s * (size_k / 2);
                half2 a0 = sh_s[(kb << 3) + tig];
                half2 a1 = sh_s[(kb << 3) + tig + 4];
                float2 af0 = __half22float2(a0);
                float2 af1 = __half22float2(a1);
                c0[s] = fmaf(af0.x, f00.x, c0[s]);
                c0[s] = fmaf(af0.y, f00.y, c0[s]);
                c0[s] = fmaf(af1.x, f01.x, c0[s]);
                c0[s] = fmaf(af1.y, f01.y, c0[s]);
                c1[s] = fmaf(af0.x, f10.x, c1[s]);
                c1[s] = fmaf(af0.y, f10.y, c1[s]);
                c1[s] = fmaf(af1.x, f11.x, c1[s]);
                c1[s] = fmaf(af1.y, f11.y, c1[s]);
            }
            w0 = w1; w1 = w2; w2 = w3; w3 = w4; w4 = w5; w5 = w6; w6 = w7; w7 = wn;
        }
        }
    }
    else
    {
        if (wvalid)
        {
        // Generic bitrates (gate path): per-row decode straight from global, L1-served
        for (int kb = kb0; kb < kb1; ++kb)
        {
            const uint32_t* blk = bp + (size_t) kb * rstride;
            FragB f0, f1;
            dq_dispatch<bits, cb>(blk, lane << 3, f0, f1);
            float2 f00 = __half22float2(f0[0]);
            float2 f01 = __half22float2(f0[1]);
            float2 f10 = __half22float2(f1[0]);
            float2 f11 = __half22float2(f1[1]);
            #pragma unroll
            for (int s = 0; s < NUM_S; ++s)
            {
                const half2* sh_s = sh_a2 + (size_t) s * (size_k / 2);
                half2 a0 = sh_s[(kb << 3) + tig];
                half2 a1 = sh_s[(kb << 3) + tig + 4];
                float2 af0 = __half22float2(a0);
                float2 af1 = __half22float2(a1);
                c0[s] = fmaf(af0.x, f00.x, c0[s]);
                c0[s] = fmaf(af0.y, f00.y, c0[s]);
                c0[s] = fmaf(af1.x, f01.x, c0[s]);
                c0[s] = fmaf(af1.y, f01.y, c0[s]);
                c1[s] = fmaf(af0.x, f10.x, c1[s]);
                c1[s] = fmaf(af0.y, f10.y, c1[s]);
                c1[s] = fmaf(af1.x, f11.x, c1[s]);
                c1[s] = fmaf(af1.y, f11.y, c1[s]);
            }
        }
        }
    }

    // 4-lane butterfly
    #pragma unroll
    for (int s = 0; s < NUM_S; ++s)
    {
        c0[s] += __shfl_xor_sync(0xffffffffu, c0[s], 1);
        c0[s] += __shfl_xor_sync(0xffffffffu, c0[s], 2);
        c1[s] += __shfl_xor_sync(0xffffffffu, c1[s], 1);
        c1[s] += __shfl_xor_sync(0xffffffffu, c1[s], 2);
    }

    // Deterministic k-part merge through the C rows
    const int ksplit = gridDim.y;
    const int stage = ksplit - 1 - blockIdx.y;
    const bool first = stage == 0;
    const bool last = stage == ksplit - 1;

    #pragma unroll
    for (int s = 0; s < NUM_S; ++s)
    {
        barrier_acquire(locks[s], stage);
        if (wvalid && tig == 0)
        {
            const int col0 = blockIdx.x * GV_COLS_PER_BLOCK + warp_id * 16 + (lane >> 2);
            if constexpr (c_fp32)
            {
                float* Cf = (float*) C_ptrs[s];
                float v0 = first ? 0.0f : __ldcg(Cf + col0);
                float v1 = first ? 0.0f : __ldcg(Cf + col0 + 8);
                Cf[col0] = v0 + c0[s];
                Cf[col0 + 8] = v1 + c1[s];
            }
            else
            {
                half* Ch = (half*) C_ptrs[s];
                float v0 = first ? 0.0f : __half2float(__ldcg(Ch + col0));
                float v1 = first ? 0.0f : __half2float(__ldcg(Ch + col0 + 8));
                Ch[col0] = __float2half(v0 + c0[s]);
                Ch[col0 + 8] = __float2half(v1 + c1[s]);
            }
        }
        barrier_release(locks[s], 1, last);
    }
}

template <int bits, int cb, bool c_fp32, bool v2>
inline __device__ void exl3_mgemm_gv_inner(
    int num_s,
    const half* const* __restrict__ Arows,
    const uint16_t* __restrict__ B,
    void* const* __restrict__ C_ptrs,
    const int size_k,
    const int size_n,
    int* const* __restrict__ locks)
{
    switch (num_s)
    {
        case 1: exl3_mgemm_gv_inner_impl<bits, cb, c_fp32, v2, 1>(Arows, B, C_ptrs, size_k, size_n, locks); break;
        case 2: exl3_mgemm_gv_inner_impl<bits, cb, c_fp32, v2, 2>(Arows, B, C_ptrs, size_k, size_n, locks); break;
        case 3: exl3_mgemm_gv_inner_impl<bits, cb, c_fp32, v2, 3>(Arows, B, C_ptrs, size_k, size_n, locks); break;
        case 4: exl3_mgemm_gv_inner_impl<bits, cb, c_fp32, v2, 4>(Arows, B, C_ptrs, size_k, size_n, locks); break;
        default: break;
    }
}

// Single-vector convenience overload for dense gemm
template <int bits, int cb, bool c_fp32, bool v2>
inline __device__ void exl3_mgemm_gv_inner(
    const half* __restrict__ Arow,
    const uint16_t* __restrict__ B,
    void* __restrict__ C_,
    const int size_k,
    const int size_n,
    int* __restrict__ lock)
{
    const half* Arows[1] = { Arow };
    void* C_ptrs[1] = { C_ };
    int* locks[1] = { lock };
    exl3_mgemm_gv_inner_impl<bits, cb, c_fp32, v2, 1>(Arows, B, C_ptrs, size_k, size_n, locks);
}

// ---------------------------------------------------------------------------------------------------------
// Full kernel: mgemm shell (index packing, input had, per-slot loop, output had, weighted
// grouped reduction) with the m=1 GEMV inner. Argument list matches exl3_mgemm_kernel exactly
// (graph parameter sites 0/2/10/11 stay valid).

template <int bits, bool c_fp32, int cb, bool v2>
__global__ __launch_bounds__(GV_NUM_THREADS) void exl3_mgemm_gv_kernel(EXL3_MGEMM_ARGS)
{
    int bszm = MAX(bszm_in, bszm_out);
    auto grid = cg_gv::this_grid();

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        if (min_index >= 0)
        {
            if (num_tokens > 1)
            {
                for (int i = 0; i < bszm; ++i)
                {
                    int idx = B_indices[i];
                    bool keep = idx >= min_index && idx < max_index;
                    gv_v_indices[i] = keep ? idx - min_index : -1;
                    if (B_weights) gv_v_weights[i] = keep ? B_weights[i] : __float2half(0.0f);
                }
                gv_bszm_sync = bszm;
            }
            else
            {
                int j = 0;
                for (int i = 0; i < bszm; ++i)
                {
                    int idx = B_indices[i];
                    if (idx >= min_index && idx < max_index)
                    {
                        gv_v_indices[j] = idx - min_index;
                        if (B_weights) gv_v_weights[j] = B_weights[i];
                        j++;
                    }
                }
                gv_bszm_sync = j;
                for (; j < bszm; ++j)
                {
                    gv_v_indices[j] = -1;
                }
            }
        }

        // Build unique active expert groups
        int num_unique = 0;
        const int64_t* eff_indices = (min_index >= 0) ? gv_v_indices : B_indices;
        int eff_bszm = (min_index >= 0) ? gv_bszm_sync : bszm;

        for (int i = 0; i < eff_bszm; ++i)
        {
            int mat = eff_indices ? (int) eff_indices[i] : i;
            if (mat < 0) continue;

            int found_grp = -1;
            for (int g = 0; g < num_unique; ++g)
            {
                if (gv_unique_mat[g] == mat && gv_unique_cnt[g] < GV_MAX_SLOTS_PER_GROUP)
                {
                    found_grp = g;
                    break;
                }
            }
            if (found_grp >= 0)
            {
                int c = gv_unique_cnt[found_grp];
                gv_unique_slots[found_grp][c] = i;
                gv_unique_cnt[found_grp] = c + 1;
            }
            else
            {
                if (num_unique < GV_MAX_UNIQUE_GROUPS)
                {
                    gv_unique_mat[num_unique] = mat;
                    gv_unique_cnt[num_unique] = 1;
                    gv_unique_slots[num_unique][0] = i;
                    num_unique++;
                }
            }
        }
        gv_num_unique = num_unique;
    }
    __threadfence();
    grid.sync();
    if (min_index >= 0)
    {
        B_indices = gv_v_indices;
        if (B_weights) B_weights = gv_v_weights;
        bszm = gv_bszm_sync;
    }

    for (int i = 0; i < gv_num_unique; i += gridDim.z)
    {
        int g = i + blockIdx.z;
        int mat_index = -1;
        int cnt = 0;
        const uint16_t* B = nullptr;
        if (g < gv_num_unique)
        {
            mat_index = gv_unique_mat[g];
            cnt = gv_unique_cnt[g];
            if (mat_index >= 0)
            {
                B = B_list[mat_index];
            }
        }

        // Input Hadamard (blockIdx.y == 0 only: y-copies are k-split parts, one span writer)
        if (B && blockIdx.y == 0)
        {
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
            const half* suh = suh_list[mat_index];

            for (int s = 0; s < cnt; ++s)
            {
                int j = gv_unique_slots[g][s];
                const half* A_ = bszm_in == 1 ? A : A + (size_t) j * size_m * size_k;
                half* A_had_ = A_had + (size_t) j * size_m * size_k;

                for (int w = this_warp; w < total_warps; w += warps_grid)
                {
                    had_hf_r_128_inner<true, false>
                    (
                        A_ + w * 128,
                        A_had_ + w * 128,
                        suh + (w * 128) % size_k,
                        0.088388347648f
                    );
                }
            }
        }

        grid.sync();

        int n_j = (size_n_list && mat_index >= 0) ? size_n_list[mat_index] : size_n;
        if (B && cnt > 0)
        {
            const half* Arows[GV_MAX_SLOTS_PER_GROUP];
            void* C_ptrs[GV_MAX_SLOTS_PER_GROUP];
            int* gv_locks[GV_MAX_SLOTS_PER_GROUP];

            for (int s = 0; s < cnt; ++s)
            {
                int j = gv_unique_slots[g][s];
                Arows[s] = A_had + (size_t) j * size_m * size_k;
                if (C_list && mat_index >= 0)
                    C_ptrs[s] = C_list[mat_index];
                else if constexpr (c_fp32)
                    C_ptrs[s] = (void*) (((float*) C) + (size_t) j * size_m * size_n);
                else
                    C_ptrs[s] = (void*) (((half*) C) + (size_t) j * size_m * size_n);

                gv_locks[s] = locks + (size_t) (blockIdx.z * GV_MAX_SLOTS_PER_GROUP + s) * (size_n / 128) + blockIdx.x;
            }

            exl3_mgemm_gv_inner<bits, cb, c_fp32, v2>
            (
                cnt, Arows, B, C_ptrs, size_k, n_j, gv_locks
            );
        }

        // Orders the inner's C_ write (all blocks) before the output Hadamard below
        grid.sync();

        // Output Hadamard and scales (blockIdx.y == 0 only: in-place, one span writer)
        if (B && blockIdx.y == 0)
        {
            int total_warps = size_m * n_j / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
            const half* svh = svh_list[mat_index];

            for (int s = 0; s < cnt; ++s)
            {
                int j = gv_unique_slots[g][s];
                void* C_j;
                if (C_list && mat_index >= 0)
                    C_j = C_list[mat_index];
                else if constexpr (c_fp32)
                    C_j = (void*) (((float*) C) + (size_t) j * size_m * size_n);
                else
                    C_j = (void*) (((half*) C) + (size_t) j * size_m * size_n);

                float scale = 0.088388347648f;
                if (B_weights) scale *= __half2float(B_weights[j]);

                for (int w = this_warp; w < total_warps; w += warps_grid)
                {
                    if constexpr (c_fp32)
                        had_ff_r_128_inner<false, true>
                        (
                            ((const float*) C_j) + w * 128,
                            ((float*) C_j) + w * 128,
                            svh + (w * 128) % n_j,
                            scale
                        );
                    else
                        had_hf_r_128_inner<false, true>
                        (
                            ((const half*) C_j) + w * 128,
                            ((half*) C_j) + w * 128,
                            svh + (w * 128) % n_j,
                            scale
                        );
                }
            }
        }
    }

    if (B_weights)
        grid.sync();

    // Final grouped reduction (parallelized across all blockIdx.z and blockIdx.x with blockIdx.y == 0)

    if (B_weights && blockIdx.y == 0)
    {
        int total_warps = size_m * size_n / 32;
        int warps_grid = gridDim.x * gridDim.z * blockDim.x / 32;
        int this_warp = (threadIdx.x / 32) + (blockDim.x / 32) * (blockIdx.x + gridDim.x * blockIdx.z);
        int this_lane = threadIdx.x % 32;
        int stride = bszm / num_tokens;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            for (int t = 0; t < num_tokens; ++t)
            {
                int col = this_warp * 32 + this_lane;
                if constexpr (c_fp32)
                {
                    float* C___ = ((float*) C) + t * stride * size_m * size_n + col;
                    float sum = 0.0f;
                    for (int j = 0; j < stride; ++j)
                    {
                        if (!B_indices || B_indices[t * stride + j] >= 0)
                            sum += *C___;
                        C___ += size_m * size_n;
                    }
                    ((float*) C)[t * size_m * size_n + col] = sum;
                }
                else
                {
                    half* C___ = ((half*) C) + t * stride * size_m * size_n + col;
                    half sum = {};
                    for (int j = 0; j < stride; ++j)
                    {
                        if (!B_indices || B_indices[t * stride + j] >= 0)
                            sum = __hadd(sum, *C___);
                        C___ += size_m * size_n;
                    }
                    ((half*) C)[t * size_m * size_n + col] = sum;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Single GEMM version for dense models (size_m == 1, EXL3_SM70_GEMM_GV=1)

template <int bits, bool c_fp32, int cb, bool v2>
__global__ __launch_bounds__(GV_NUM_THREADS) void exl3_gemm_gv_kernel(EXL3_GEMM_ARGS)
{
    // Input Hadamard (blockIdx.y == 0 only: y-copies are k-split parts, one span writer)
    if (blockIdx.y == 0)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f
            );
    }

    auto grid = cg_gv::this_grid();
    grid.sync();

    int* gv_lock = locks + blockIdx.x;
    for (int m_off = 0; m_off < size_m; m_off += 4)
    {
        int num_s = MIN(size_m - m_off, 4);
        const half* Arows[4];
        void* C_ptrs[4];
        int* locks_arr[4];
        #pragma unroll
        for (int s = 0; s < num_s; ++s)
        {
            Arows[s] = A_had + (m_off + s) * size_k;
            if constexpr (c_fp32)
                C_ptrs[s] = (void*) (((float*) C) + (m_off + s) * size_n);
            else
                C_ptrs[s] = (void*) (((half*) C) + (m_off + s) * size_n);
            locks_arr[s] = gv_lock;
        }
        exl3_mgemm_gv_inner<bits, cb, c_fp32, v2>(num_s, Arows, B, C_ptrs, size_k, size_n, locks_arr);
    }

    grid.sync();

    // Output Hadamard and scales (blockIdx.y == 0 only: in-place, one span writer)
    if (blockIdx.y == 0)
    {
        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        float scale = 0.088388347648f;
        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner<false, true>
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    scale
                );
            else
                had_hf_r_128_inner<false, true>
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    scale
                );
        }
    }
}
