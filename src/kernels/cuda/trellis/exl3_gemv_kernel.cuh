#pragma once

// Small-m GEMV path for the EXL3 GEMM, QTIP-style structure (see Cornell-RelaxML/qtip,
// qtip-kernels/src/inference.cu) on the unmodified EXL3 format:
//
// - warps split k and never synchronize during the main loop: no block-wide pipeline barriers;
//   B streams straight to registers with ld.global.cs (evict-first; B is single-use) behind a
//   register prefetch ring
// - the two-word bit windows of the trellis stream are resolved in-warp: with SMEM_STAGE = false
//   via lane shuffles (the extraction helpers in exl3_dq.cuh read exactly two words per lane, at
//   lane-computable indices), with SMEM_STAGE = true by staging the tile words through
//   warp-private shared memory and calling the standard dq_dispatch
// - one m16n8k16 MMA pair per 16x16 weight tile with fp16 accumulation, folded to fp32 on a fixed
//   cadence; per-block cross-warp reduction over the k splits through shared memory
//
// Same launch signature as exl3_gemm_kernel so kernel args and graph parameter patching are
// interchangeable. Cooperative launch: one grid.sync after the input Hadamard stage and one before
// the output stage, no other cross-block coordination. 2, 3 and 4 bpw.
//
// CFG 0 ("narrow", 512 threads, 2 n-tiles/warp, 16 k-splits) wins at attention-projection sizes;
// CFG 1 ("wide", 256 threads, 4 n-tiles/warp, 8 k-splits) wins at large-n FFN sizes. MMODE 0 is
// the m == 1 fast path, MMODE 1 covers 2 <= m <= 8 with row-guarded fragment loads.

#include <cooperative_groups.h>
#include "ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"

#define EXL3_GEMV_MAX_M 8

// Split-K partials: [k_slice][row][col] FP32, summed in fixed slice order after a grid sync so the
// output is bit-identical run to run (FP16 atomicAdd made the target forward nondeterministic).
// Split-K only fires for size_n < 80 * 64 with at most 4 slices and EXL3_GEMV_MAX_M rows.
#define EXL3_GEMV_SPLITK_MAX 4
#define EXL3_GEMV_SPLITK_WS ( EXL3_GEMV_SPLITK_MAX * EXL3_GEMV_MAX_M * 5120 )
__device__ float g_exl3_gemv_splitk_ws[EXL3_GEMV_SPLITK_WS];

namespace exl3_gemv_ns {

// mma.m16n8k16 with the A operand supplied as two FragB halves, fp16 accumulate.
// On Turing this splits into the two k=8 halves it is already expressed as (a01 pairs with
// b[0], a23 with b[1]), accumulating through the same c. Same operands and products; one
// extra rounding at the k=8 boundary. See the longer note in ptx.cuh.
__device__ __forceinline__ void mma_ab_h(const FragB& a01, const FragB& a23, const FragB& b, FragC_h& c)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) < 800
    // Volta fallback: shuffles + half2 FMA, same fragment layouts as the mma. Full cross-lane
    // k-reduction (see ptx_mma_m16n8k16 FragC overload in ../ptx.cuh). Rows 0-7 only (m <= 8).
    // Accumulation uses native half2 HFMA2 chains (same precision class as the f16-accumulator
    // tensor-core path this emulates; fp32 folding to acc0 happens on the caller's cadence).
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, tig = lane & 3;
    const half2* ga0 = reinterpret_cast<const half2*>(&a01);
    const half2* ga1 = reinterpret_cast<const half2*>(&a23);
    const half2* gbb = reinterpret_cast<const half2*>(&b);
    half2* gcc = reinterpret_cast<half2*>(&c);
    half2 a00[4], a02[4];
    #pragma unroll
    for (int tt = 0; tt < 4; ++tt)
    {
        a00[tt] = __shfl_sync(0xffffffffu, ga0[0], g * 4 + tt);
        a02[tt] = __shfl_sync(0xffffffffu, ga1[0], g * 4 + tt);
    }
    half2 bx0[4], bx8[4], by0[4], by8[4];
    #pragma unroll
    for (int tt = 0; tt < 4; ++tt)
    {
        bx0[tt] = __shfl_sync(0xffffffffu, gbb[0], 8 * tig + tt);
        bx8[tt] = __shfl_sync(0xffffffffu, gbb[1], 8 * tig + tt);
        by0[tt] = __shfl_sync(0xffffffffu, gbb[0], 8 * tig + 4 + tt);
        by8[tt] = __shfl_sync(0xffffffffu, gbb[1], 8 * tig + 4 + tt);
    }
    half2 sx2 = __hmul2(a00[0], bx0[0]);
    half2 sy2 = __hmul2(a00[0], by0[0]);
    #pragma unroll
    for (int tt = 1; tt < 4; ++tt)
    {
        sx2 = __hfma2(a00[tt], bx0[tt], sx2);
        sy2 = __hfma2(a00[tt], by0[tt], sy2);
    }
    #pragma unroll
    for (int tt = 0; tt < 4; ++tt)
    {
        sx2 = __hfma2(a02[tt], bx8[tt], sx2);
        sy2 = __hfma2(a02[tt], by8[tt], sy2);
    }
    float sx = __low2float(sx2) + __high2float(sx2);
    float sy = __low2float(sy2) + __high2float(sy2);
    gcc[0].x = __hadd(gcc[0].x, __float2half_rn(sx));
    gcc[0].y = __hadd(gcc[0].y, __float2half_rn(sy));
    return;
#endif
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
#if EXL3_SM75
    asm
    (
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a0[0]), "r"(a0[1]),
           "r"(bb[0])
    );
    asm
    (
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a1[0]), "r"(a1[1]),
           "r"(bb[1])
    );
#else
    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
           "r"(bb[0]), "r"(bb[1])
    );
#endif
}

// Paired variant: two B fragments (adjacent 8-column halves of a 16x16 tile) against the
// same A operand. On Volta the A-fragment gathers are hoisted out of the second half.
__device__ __forceinline__ void mma_ab_h2(const FragB& a01, const FragB& a23, const FragB& b0, const FragB& b1, FragC_h& c0, FragC_h& c1)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) < 800
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, tig = lane & 3;
    const half2* ga0 = reinterpret_cast<const half2*>(&a01);
    const half2* ga1 = reinterpret_cast<const half2*>(&a23);
    half2 a00[4], a02[4];
    #pragma unroll
    for (int tt = 0; tt < 4; ++tt)
    {
        a00[tt] = __shfl_sync(0xffffffffu, ga0[0], g * 4 + tt);
        a02[tt] = __shfl_sync(0xffffffffu, ga1[0], g * 4 + tt);
    }
    #pragma unroll
    for (int half_b = 0; half_b < 2; ++half_b)
    {
        const FragB& b = half_b ? b1 : b0;
        FragC_h& c = half_b ? c1 : c0;
        const half2* gbb = reinterpret_cast<const half2*>(&b);
        half2* gcc = reinterpret_cast<half2*>(&c);
        half2 bx0[4], bx8[4], by0[4], by8[4];
        #pragma unroll
        for (int tt = 0; tt < 4; ++tt)
        {
            bx0[tt] = __shfl_sync(0xffffffffu, gbb[0], 8 * tig + tt);
            bx8[tt] = __shfl_sync(0xffffffffu, gbb[1], 8 * tig + tt);
            by0[tt] = __shfl_sync(0xffffffffu, gbb[0], 8 * tig + 4 + tt);
            by8[tt] = __shfl_sync(0xffffffffu, gbb[1], 8 * tig + 4 + tt);
        }
        half2 sx2 = __hmul2(a00[0], bx0[0]);
        half2 sy2 = __hmul2(a00[0], by0[0]);
        #pragma unroll
        for (int tt = 1; tt < 4; ++tt)
        {
            sx2 = __hfma2(a00[tt], bx0[tt], sx2);
            sy2 = __hfma2(a00[tt], by0[tt], sy2);
        }
        #pragma unroll
        for (int tt = 0; tt < 4; ++tt)
        {
            sx2 = __hfma2(a02[tt], bx8[tt], sx2);
            sy2 = __hfma2(a02[tt], by8[tt], sy2);
        }
        float sx = __low2float(sx2) + __high2float(sx2);
        float sy = __low2float(sy2) + __high2float(sy2);
        gcc[0].x = __hadd(gcc[0].x, __float2half_rn(sx));
        gcc[0].y = __hadd(gcc[0].y, __float2half_rn(sy));
    }
    return;
#endif
    mma_ab_h(a01, a23, b0, c0);
    mma_ab_h(a01, a23, b1, c1);
}

// mul1 codebook pair decode via dp4a byte sum (bit-identical to the vabsdiff4 form)
__device__ __forceinline__ half2 decode_pair_cb2_dp4a_(uint32_t x0, uint32_t x1)
{
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half_uint16 h0((uint16_t) sum0);
    half_uint16 h1((uint16_t) sum1);
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

template <int cb>
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    if constexpr (cb == 2)
    {
        f0[0] = decode_pair_cb2_dp4a_(w0, w1);
        f0[1] = decode_pair_cb2_dp4a_(w2, w3);
        f1[0] = decode_pair_cb2_dp4a_(w4, w5);
        f1[1] = decode_pair_cb2_dp4a_(w6, w7);
    }
    else
    {
        f0[0] = decode_3inst_2<cb>(w0, w1);
        f0[1] = decode_3inst_2<cb>(w2, w3);
        f1[0] = decode_3inst_2<cb>(w4, w5);
        f1[1] = decode_3inst_2<cb>(w6, w7);
    }
}

// Window extraction from two already-loaded words, same order as dq8_aligned_4bits
template <int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1)
{
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8_aligned_2bits: the two words and the funnel shift are lane-dependent
template <int cb>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8<3, cb, 4> with the per-lane funnel alignment precomputed
template <int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> 3;
    w5 = w6 >> 3;
    w4 = w5 >> 3;
    w3 = fshift(b, a, s2 + 12);
    w2 = w3 >> 3;
    w1 = w2 >> 3;
    w0 = w1 >> 3;
    decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

}  // namespace exl3_gemv_ns

inline __device__ uint4 ld_cs_v4(const uint4* ptr)
{
    uint4 ret;
    asm volatile ("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(ptr));
    return ret;
}

template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256)
void exl3_gemv_kernel(EXL3_GEMM_ARGS)
{
    static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6, "exl3_gemv_kernel supports 2, 3, 4, 5 and 6 bpw");
    static_assert(bits != 5 || (CFG == 1 && SMEM_STAGE), "5 bpw requires CFG 1 and SMEM_STAGE");
    constexpr int WK   = CFG == 0 ? 16 : 8;     // k-split (warps per block)
    constexpr int WNT  = CFG == 0 ? 2 : 4;      // adjacent n-tiles per warp
    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth
    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS = WNT * 16;

    constexpr int TWORDS = 8 * bits;                        // uint32 per 16x16 tile
    constexpr int LOADS = bits == 2 ? WNT / 2 :
                          (bits == 5 && CFG == 1) ? 5 :
                          bits == 6 ? WNT * 3 / 2 : WNT; // warp loads per k-slice
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;            // uint32 per load
    static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

    auto grid = cooperative_groups::this_grid();

    // Input scales and Hadamard transform, same as exl3_gemm_kernel
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * gridDim.y * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + (blockDim.x / 32) * (blockIdx.x + blockIdx.y * gridDim.x);

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false, true>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;

    const int k_slice_start = blockIdx.y * CEIL_DIVIDE(kslices, gridDim.y);
    const int k_slice_end   = min(kslices, (blockIdx.y + 1) * CEIL_DIVIDE(kslices, gridDim.y));
    const int k_slice_len   = max(0, k_slice_end - k_slice_start);

    const int chunk = CEIL_DIVIDE(k_slice_len, WK);
    const int ks0 = k_slice_start + warp * chunk;
    const int myn = max(0, min(chunk, k_slice_end - ks0));

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;   // uint32 per k-slice row
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A fragment row indices for this lane
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    // Per-lane extraction constants (see dq8_aligned_2bits / dq8<3, cb, 4> in exl3_dq.cuh)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ float sh_red[WK][ROWS][COLS];
    [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) < 800
    __shared__ uint32_t sh_f[WK][4][32];
#endif

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x)
    {
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;
        const bool is_coalesced = ((uintptr_t) locks & 1) != 0;
        const bool is_zero_shfl = ((uintptr_t) locks & 2) != 0;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        uint4 pf4_0[PF];
        uint4 pf4_1[PF];

        if (is_coalesced && SMEM_STAGE && CFG == 1 && (bits == 5 || bits == 6))
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                if (d < myn)
                {
                    const uint4* p4 = (const uint4*)(B32 + (size_t)(ks0 + d) * slice_stride + group * WNT * TWORDS);
                    pf4_0[d] = ld_cs_v4(p4 + lane);
                    if constexpr (bits == 5)
                    {
                        if (lane < 8) pf4_1[d] = ld_cs_v4(p4 + 32 + lane);
                    }
                    else if constexpr (bits == 6)
                    {
                        if (lane < 16) pf4_1[d] = ld_cs_v4(p4 + 32 + lane);
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
                if (d < myn)
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l)
                        pf[d][l] = ld_b(d, l);
        }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) < 800
        float dacc[WNT][2][8] = {};
#else
        FragC_h ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};
#endif

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];

            if (is_coalesced && SMEM_STAGE && CFG == 1 && (bits == 5 || bits == 6))
            {
                uint4 cur_v0 = pf4_0[d];
                uint4 cur_v1 = pf4_1[d];

                if (i + PF < myn)
                {
                    const uint4* p4 = (const uint4*)(B32 + (size_t)(ks0 + i + PF) * slice_stride + group * WNT * TWORDS);
                    pf4_0[d] = ld_cs_v4(p4 + lane);
                    if constexpr (bits == 5)
                    {
                        if (lane < 8) pf4_1[d] = ld_cs_v4(p4 + 32 + lane);
                    }
                    else if constexpr (bits == 6)
                    {
                        if (lane < 16) pf4_1[d] = ld_cs_v4(p4 + 32 + lane);
                    }
                }

                ((uint4*)&sh_stage[warp][0])[lane] = cur_v0;
                if constexpr (bits == 5)
                {
                    if (lane < 8) ((uint4*)&sh_stage[warp][128])[lane] = cur_v1;
                }
                else if constexpr (bits == 6)
                {
                    if (lane < 16) ((uint4*)&sh_stage[warp][128])[lane] = cur_v1;
                }
                __syncwarp();
            }
            else
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = pf[d][l];

                if (i + PF < myn)
                {
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l)
                        pf[d][l] = ld_b(i + PF, l);
                }

                if constexpr (SMEM_STAGE)
                {
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l)
                        if (bits != 3 || lane < 24)
                            sh_stage[warp][l * LSTRIDE + lane] = bw[l];
                    __syncwarp();
                }
            }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) < 800
            const int r_v = (lane & 3) + 4 * (lane >= 16);
            const size_t a_row_v = (size_t) r_v * (size_k / 2);
            const bool r_v_ok = (MMODE == 0) ? (r_v == 0) : (r_v < size_m);
            const size_t a_col_v = (size_t) (ks0 + i) * 8;
            uint32_t va[4][2];
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                va[c][0] = r_v_ok ? *(const uint32_t*)&A2[a_row_v + a_col_v + 2 * c + 0] : 0;
                va[c][1] = r_v_ok ? *(const uint32_t*)&A2[a_row_v + a_col_v + 2 * c + 1] : 0;
            }
#else
            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;
#endif

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (SMEM_STAGE)
                {
                    const uint32_t* tp = &sh_stage[warp][t * TWORDS];
                    if constexpr (bits == 4)
                        exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                    else if constexpr (bits == 2)
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                    else if constexpr (bits == 3)
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
                    else if constexpr (bits == 5)
                        dq_dispatch<5, cb>(tp, lane << 3, f0, f1);
                    else if constexpr (bits == 6)
                        dq_dispatch<6, cb>(tp, lane << 3, f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    // Two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else if constexpr (bits == 3)
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) < 800
                if (is_zero_shfl)
                {
                    sh_f[warp][0][lane] = *(const uint32_t*)&f0[0];
                    sh_f[warp][1][lane] = *(const uint32_t*)&f0[1];
                    sh_f[warp][2][lane] = *(const uint32_t*)&f1[0];
                    sh_f[warp][3][lane] = *(const uint32_t*)&f1[1];
                    __syncwarp();

                    const int cv = (lane & 3) + 4 * (lane >= 16);
                    // Note: 4 * cv is 16-byte aligned (multiple of 4 words). We can load uint2 (64-bit)!
                    const uint2* sf0 = (const uint2*)&sh_f[warp][0][4 * cv];
                    const uint2* sf1 = (const uint2*)&sh_f[warp][1][4 * cv];
                    const uint2* sf2 = (const uint2*)&sh_f[warp][2][4 * cv];
                    const uint2* sf3 = (const uint2*)&sh_f[warp][3][4 * cv];

                    // Step 0: k0..k3
                    uint2 b01_0 = sf0[0];
                    uint2 b01_1 = sf2[0];
                    volta_mma_m8n8k4(va[0][0], va[0][1], b01_0.x, b01_0.y, dacc[t][0]);
                    volta_mma_m8n8k4(va[0][0], va[0][1], b01_1.x, b01_1.y, dacc[t][1]);

                    // Step 1: k4..k7
                    uint2 b01_0_1 = sf0[1];
                    uint2 b01_1_1 = sf2[1];
                    volta_mma_m8n8k4(va[1][0], va[1][1], b01_0_1.x, b01_0_1.y, dacc[t][0]);
                    volta_mma_m8n8k4(va[1][0], va[1][1], b01_1_1.x, b01_1_1.y, dacc[t][1]);

                    // Step 2: k8..k11
                    uint2 b01_0_2 = sf1[0];
                    uint2 b01_1_2 = sf3[0];
                    volta_mma_m8n8k4(va[2][0], va[2][1], b01_0_2.x, b01_0_2.y, dacc[t][0]);
                    volta_mma_m8n8k4(va[2][0], va[2][1], b01_1_2.x, b01_1_2.y, dacc[t][1]);

                    // Step 3: k12..k15
                    uint2 b01_0_3 = sf1[1];
                    uint2 b01_1_3 = sf3[1];
                    volta_mma_m8n8k4(va[3][0], va[3][1], b01_0_3.x, b01_0_3.y, dacc[t][0]);
                    volta_mma_m8n8k4(va[3][0], va[3][1], b01_1_3.x, b01_1_3.y, dacc[t][1]);
                    __syncwarp();
                }
                else
                {
                    const int cv = (lane & 3) + 4 * (lane >= 16);
                    // Step 0: k0..k3 from f0[0]/f1[0] lanes 4*cv+0, 4*cv+1
                    uint32_t b0_0 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f0[0], 4 * cv + 0);
                    uint32_t b1_0 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f0[0], 4 * cv + 1);
                    volta_mma_m8n8k4(va[0][0], va[0][1], b0_0, b1_0, dacc[t][0]);

                    uint32_t b0_1 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f1[0], 4 * cv + 0);
                    uint32_t b1_1 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f1[0], 4 * cv + 1);
                    volta_mma_m8n8k4(va[0][0], va[0][1], b0_1, b1_1, dacc[t][1]);

                    // Step 1: k4..k7 from f0[0]/f1[0] lanes 4*cv+2, 4*cv+3
                    b0_0 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f0[0], 4 * cv + 2);
                    b1_0 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f0[0], 4 * cv + 3);
                    volta_mma_m8n8k4(va[1][0], va[1][1], b0_0, b1_0, dacc[t][0]);

                    b0_1 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f1[0], 4 * cv + 2);
                    b1_1 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f1[0], 4 * cv + 3);
                    volta_mma_m8n8k4(va[1][0], va[1][1], b0_1, b1_1, dacc[t][1]);

                    // Step 2: k8..k11 from f0[1]/f1[1] lanes 4*cv+0, 4*cv+1
                    b0_0 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f0[1], 4 * cv + 0);
                    b1_0 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f0[1], 4 * cv + 1);
                    volta_mma_m8n8k4(va[2][0], va[2][1], b0_0, b1_0, dacc[t][0]);

                    b0_1 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f1[1], 4 * cv + 0);
                    b1_1 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f1[1], 4 * cv + 1);
                    volta_mma_m8n8k4(va[2][0], va[2][1], b0_1, b1_1, dacc[t][1]);

                    // Step 3: k12..k15 from f0[1]/f1[1] lanes 4*cv+2, 4*cv+3
                    b0_0 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f0[1], 4 * cv + 2);
                    b1_0 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f0[1], 4 * cv + 3);
                    volta_mma_m8n8k4(va[3][0], va[3][1], b0_0, b1_0, dacc[t][0]);

                    b0_1 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f1[1], 4 * cv + 2);
                    b1_1 = __shfl_sync(0xffffffffu, *(const uint32_t*)&f1[1], 4 * cv + 3);
                    volta_mma_m8n8k4(va[3][0], va[3][1], b0_1, b1_1, dacc[t][1]);
                }
#else
                exl3_gemv_ns::mma_ab_h2(a01, a23, f0, f1, ch[t][0], ch[t][1]);
#endif
            }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__) >= 800
            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
#endif
        }
        }

        // Cross-warp reduction over the k splits.
        {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__) < 800
            if constexpr (MMODE == 0)
            {
                if ((lane & 1) == 0 && lane < 16)
                {
                    const int t_lane = lane & 3;
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                    {
                        #pragma unroll
                        for (int r_idx = 0; r_idx < 4; ++r_idx)
                        {
                            const int reg = (r_idx < 2) ? r_idx : (r_idx + 2); // 0, 1, 4, 5
                            const int n = (reg & 1) + 2 * (t_lane >> 1) + 4 * (reg >> 2);
                            sh_red[warp][0][t * 16 + 0 + n] = dacc[t][0][reg];
                            sh_red[warp][0][t * 16 + 8 + n] = dacc[t][1][reg];
                        }
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                {
                    #pragma unroll
                    for (int reg = 0; reg < 8; ++reg)
                    {
                        int m, n;
                        volta_d_slot(lane, reg, &m, &n);
                        if (m < ROWS)
                        {
                            sh_red[warp][m][t * 16 + 0 + n] = dacc[t][0][reg];
                            sh_red[warp][m][t * 16 + 8 + n] = dacc[t][1][reg];
                        }
                    }
                }
            }
#else
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = acc0[t][f].x;
                        sh_red[warp][sr0][col + 1] = acc0[t][f].y;
                    }
            }
#endif
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if (gridDim.y == 1)
            {
                if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
                else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
            }
            else
            {
                g_exl3_gemv_splitk_ws[((size_t) blockIdx.y * size_m + r) * size_n + col] = sum;
            }
        }
        __syncthreads();
    }

    // Fixed-order split-K reduction (deterministic): slice 0 + slice 1 + ... in FP32, one rounding
    if (gridDim.y > 1)
    {
        grid.sync();
        const int total = size_m * size_n;
        const int threads_grid = gridDim.x * gridDim.y * blockDim.x;
        const int this_tid = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
        for (int idx = this_tid; idx < total; idx += threads_grid)
        {
            float sum = 0.0f;
            for (int s = 0; s < (int) gridDim.y; ++s)
                sum += g_exl3_gemv_splitk_ws[(size_t) s * total + idx];
            if constexpr (c_fp32) ((float*) C)[idx] = sum;
            else                  ((half*)  C)[idx] = __float2half_rn(sum);
        }
    }

    // Output scales and Hadamard transform, same semantics as the inner GEMM epilogue
    {
        grid.sync();

        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * gridDim.y * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + (blockDim.x / 32) * (blockIdx.x + blockIdx.y * gridDim.x);

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner<false, true, true>
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
            else
                had_hf_r_128_inner<false, true, true>
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
}
