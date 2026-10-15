#pragma once

#include "ptx.cuh"
#include "util.cuh"

// Constants
#define EXL3_GEMM_BASE_THREADS 256
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6

#include "exl3_dq.cuh"
// For exl3_gemm_smem_bytes(), the shared definition of this kernel's shared memory footprint
#include "exl3_kernel_map.cuh"

// On GA10x, HMMA with fp32 accumulation runs at half rate and dominates the m=1 (decode-bound) case.
// Accumulate MMA results in fp16 instead and fold into the fp32 accumulators once per k-slice: ~14%
// faster at bsz 1 on RTX 3090 together with the codebook.cuh IMUL change (see benchmarks/exl3_m1_bench).
// Max observed error vs fp32 accumulation is ~1% of output RMS at k=4096, well below quantization noise.
// Only enabled for sm_86 for now; unvalidated on other archs where fp32-acc HMMA is also half rate.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
    #define EXL3_GEMM_H_ACC 1
#else
    #define EXL3_GEMM_H_ACC 0
#endif

// ---------------------------------------------------------------------------
// Large-M inner for the SM70 prefill path (Avenue 2, EXL3_SM70_PREFILL_M64=1).
//
// Same pipeline skeleton, C/reduce/lock protocol and accumulation order as
// exl3_gemm_kernel_inner, but the tile spans 64 rows (4 row-blocks of 16)
// instead of 16: each warp dequantizes its 2 B (n-)fragments once per k16
// step and feeds them to all 4 row-blocks' HMMAs, amortizing the B dequant
// 4x over the tile rows, while the caller advances in/out pointers by 64
// rows per pass instead of 16.
//
// Constraints:
//  - Volta HMMA only; guarded by EXL3_VOLTA && EXL3_SM70_HMMA, so other archs
//    never instantiate it (host tables carry nullptr and fall back to M=16).
//  - shmem_out_had == false (the fused output-Hadamard staging does not fit
//    the smem budget at 64 rows); the MoE call site passes false.
//  - c_fp32 == false.
//  - TILESIZE_N == 128: each warp holds all 4 row-blocks' accumulators for
//    its 2 n-fragments (4 * 8 = 32 fp32 D registers per lane); at TN=256 the
//    accumulator set alone would blow the 128-reg/thread budget at 512
//    threads.
//  - TILESIZE_K == 32: two sub_k groups of 8 warps (512 threads), the same
//    block layout the MoE kernel already launches.
//
// Rows [size_m, 64) of the A tile are never loaded (cp_async predicate) and
// their HMMA results are garbage but strictly row-local: output writes
// predicate on r < size_m, the same class of behavior as the M=16 kernel's
// size_m <= 8 row handling.
// ---------------------------------------------------------------------------
#if EXL3_VOLTA && EXL3_SM70_HMMA
template<EXL3_GEMM_T_ARGS, bool shmem_out_had>
inline __device__
void exl3_gemm_kernel_inner_m64
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale
)
{
    constexpr int TM = TILESIZE_M;   // 32 or 64 rows: 2 or 4 row-blocks sharing one decode of the B fragments
    constexpr int ROW_BLOCKS = TM / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    const int sh_a_stage_size = TM * TILESIZE_K;                                 // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;   // in uint16s
    const int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP;         // in floats

    const int A_COLS = TILESIZE_K / 8;
    const int A_SWIZZLE_MASK = A_COLS - 1;
    const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M == 32 || TILESIZE_M == 64, "m64 inner requires TILESIZE_M == 32 or 64");
    static_assert(TILESIZE_K == 16 || TILESIZE_K == 32, "m64 inner requires TILESIZE_K == 16 or 32");
    static_assert(TILESIZE_N == 128 || TILESIZE_N == 256, "m64 inner requires TILESIZE_N == 128 or 256");
    static_assert(!shmem_out_had, "m64 inner: shmem_out_had not supported");
    static_assert(!c_fp32, "m64 inner: c_fp32 not supported");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for m64 shape)"
    );

    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    int tiles_k = size_k / TILESIZE_K;
    int tiles_n = size_n / TILESIZE_N;
    int blocks_n = tiles_n * TILEBLOCKS_N;

    int num_slices = gridDim.x;
    int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TM * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TM * TILESIZE_K;
    const half* gl_a_ptr = A + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < size_m;
    }

    int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles_k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles_k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TM * size_n;

    half* gl_c_ptr_16 = ((half*) C) + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice2_n * gl_c_stride_n;

    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    // Volta D accumulators, one 8-float D row per (row-block, nfrag-pair). Each
    // warp's B fragments are shared by all 4 row-blocks' HMMAs. The C fragments
    // they convert into live only inside reduce(), so the two arrays can share
    // register storage.
    register float dacc[ROW_BLOCKS][FRAGS_N_PER_WARP / 2][8];
    // A smem stage captured per pipeline buffer at load_frags time: matmul must
    // read the A stage that matches the B fragments staged for that buffer, and
    // advance1() has already moved sh1_a_ptr on by the time matmul runs.
    half* sh_a_buf[FRAG_STAGES];

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            gl_c_ptr_16 += gl_c_stride_n;
            gl_c_ptr_32 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (sub_k)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
                }
            }

            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                {
                    if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
                }
            }
            advance0();
        }

        cp_async_fence();
    };

    // Load B fragments only; A fragments are gathered inline in matmul() per
    // row-block (staging all 4 row-blocks' A fragments in registers would blow
    // the 128-reg budget at 512 threads)
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        sh_a_buf[buf] = sh1_a_ptr;

        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);

            dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }

        __syncthreads();
        advance1();
    };

    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int rb = 0; rb < ROW_BLOCKS; ++rb)
            #pragma unroll
            for (int r = 0; r < FRAGS_N_PER_WARP / 2; ++r)
            #pragma unroll
            for (int i = 0; i < 8; ++i) dacc[rb][r][i] = 0.0f;
    };

    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    // Output reduction. Everything below is scoped inside this lambda so the
    // m16n8k16 C fragments can share register storage with the D accumulators
    // (disjoint lifetimes), which keeps the 4-row-block state in registers
    // instead of spilling to local memory.
    auto reduce = [&] ()
    {
        register FragC frag_c[ROW_BLOCKS][FRAGS_N_PER_WARP];

        // Convert each row-block's Volta D accumulators to the m16n8k16 C
        // layout via the (m,n)-indexed smem tile: one row-block per pass, same
        // warp-aliased per-sub_k schedule as the M=16 kernel.
        for (int rb = 0; rb < ROW_BLOCKS; ++rb)
        {
            for (int sb = 0; sb < TILEBLOCKS_K; ++sb)
            {
                if (sub_k == sb)
                {
                    float* tile = sh_c + warp_id * (FRAGS_N_PER_WARP * 128);
                    const int q_ = (lane_id >> 2) & 3;
                    #pragma unroll
                    for (int r = 0; r < FRAGS_N_PER_WARP / 2; ++r)
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                    {
                        int m, n; volta_d_slot(lane_id, i, &m, &n);
                        tile[(2 * r + (q_ & 1)) * 128 + (m + 8 * (q_ >> 1)) * 8 + n] = dacc[rb][r][i];
                    }
                    __syncwarp();
                    const int g_ = lane_id >> 2, t_ = lane_id & 3;
                    #pragma unroll
                    for (int nf = 0; nf < FRAGS_N_PER_WARP; ++nf)
                    {
                        frag_c[rb][nf][0] = tile[nf * 128 + g_ * 8 + 2 * t_];
                        frag_c[rb][nf][1] = tile[nf * 128 + g_ * 8 + 2 * t_ + 1];
                        frag_c[rb][nf][2] = tile[nf * 128 + (g_ + 8) * 8 + 2 * t_];
                        frag_c[rb][nf][3] = tile[nf * 128 + (g_ + 8) * 8 + 2 * t_ + 1];
                    }
                }
                __syncthreads();
            }
        }

        // Threadblock reduction across the two sub_k groups, per row-block.
        // Same scratch region and store/add pattern as the M=16 kernel.
        if constexpr (TILEBLOCKS_K == 2) {
        for (int rb = 0; rb < ROW_BLOCKS; ++rb)
        {
            if (sub_k == 1)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) sh_red[n * 4 + j] = frag_c[rb][n][j];
            }
            __syncthreads();
            if (sub_k == 0)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[rb][n][j] += sh_red[n * 4 + j];
            }
            // sh_red is reused by the next row block: sub_k 1 must not overwrite it before sub_k 0 has read it
            __syncthreads();
        }
        }

        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        if (!sub_k)
        {
            if (!first)
            {
                // Second and subsequent threadblocks in column read back the
                // intermediate sum from global memory
                int n0 = warp_id * FRAGS_N_PER_WARP;
                #pragma unroll
                for (int rb = 0; rb < ROW_BLOCKS; ++rb)
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    int r0 = rb * 16 + lane_id / 4;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < size_m)
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                        float2 interm = __half22float2(*c_ptr);
                        frag_c[rb][n][0] += interm.x;
                        frag_c[rb][n][1] += interm.y;
                    }
                    if (r1 < size_m)
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                        float2 interm = __half22float2(*c_ptr);
                        frag_c[rb][n][2] += interm.x;
                        frag_c[rb][n][3] += interm.y;
                    }
                }
            }

            // All but last threadblock write the intermediate result; the last
            // one writes the final row-major output (no fused output Hadamard
            // on the m64 path; the MoE caller applies it separately)
            write_sum:
            {
                int n0 = warp_id * FRAGS_N_PER_WARP;
                #pragma unroll
                for (int rb = 0; rb < ROW_BLOCKS; ++rb)
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    int r0 = rb * 16 + lane_id / 4;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < size_m)
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                        half2 sum = __floats2half2_rn(frag_c[rb][n][0], frag_c[rb][n][1]);
                        *c_ptr = sum;
                    }
                    if (r1 < size_m)
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                        half2 sum = __floats2half2_rn(frag_c[rb][n][2], frag_c[rb][n][3]);
                        *c_ptr = sum;
                    }
                }
            }
        }

        barrier_release(lock, lock_d, last);

        // A block whose slice spans more than one output column starts the next one from zero
        clear_frag_c();
    };

    // Perform tensor core matmul on current tile. The A fragments are gathered
    // inline from smem per (row-block, k4-chunk): staging all four row-blocks'
    // A fragments would exceed the register budget. The B fragments are
    // dequantized once per (k16, nfrag) and shared by all four row-blocks'
    // HMMAs, which is the entire point of the large-M tile.
    auto matmul = [&] (int buf)
    {
        const int q_ = (lane_id >> 2) & 3;
        const int cv = (lane_id & 3) + 4 * (lane_id >= 16);
        const int rh = q_ >> 1;          // row-half within the 16-row block
        const int nf_sel = q_ & 1;       // which nfrag of the pair this quarter feeds
        #pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            uint32_t b0s[FRAGS_N_PER_WARP], b1s[FRAGS_N_PER_WARP];
            #pragma unroll
            for (int nf = 0; nf < FRAGS_N_PER_WARP; ++nf)
            {
                b0s[nf] = __shfl_sync(0xffffffffu, *(const uint32_t*) &frag_b[buf][nf][c >> 1], 4 * cv + 2 * (c & 1));
                b1s[nf] = __shfl_sync(0xffffffffu, *(const uint32_t*) &frag_b[buf][nf][c >> 1], 4 * cv + 2 * (c & 1) + 1);
            }
            #pragma unroll
            for (int rb = 0; rb < ROW_BLOCKS; ++rb)
            {
                uint32_t a0, a1;
                int k0 = sub_k * 16 + 4 * c;
                int row_v = rb * 16 + 8 * rh + (lane_id & 3) + 4 * (lane_id >= 16);
                int c_swizzled = (k0 >> 3) ^ ((row_v >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                    sh_a_buf[buf] + row_v * TILESIZE_K + c_swizzled * 8 + 4 * (c & 1)));
                asm ("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(a0), "=r"(a1) : "r"(addr));
                #pragma unroll
                for (int r = 0; r < FRAGS_N_PER_WARP / 2; ++r)
                {
                    uint32_t bb0 = nf_sel ? b0s[2 * r + 1] : b0s[2 * r];
                    uint32_t bb1 = nf_sel ? b1s[2 * r + 1] : b1s[2 * r];
                    volta_mma_m8n8k4(a0, a1, bb0, bb1, dacc[rb][r]);
                }
            }
        }
    };

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    #define FSTAGE_M64(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load); \

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            FSTAGE_M64(1, 0);
            FSTAGE_M64(0, 1);
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            FSTAGE_M64(1, 0);
            FSTAGE_M64(2, 1);
            FSTAGE_M64(0, 2);
        }
    }
}
#endif  // EXL3_VOLTA && EXL3_SM70_HMMA

template<EXL3_GEMM_T_ARGS, bool shmem_out_had>
inline __device__
void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale
)
{
    const int TILEBLOCKS_M = MAX(1, TILESIZE_M / 16);
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    // const int FRAGS_M = TILEBLOCKS_M;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                         // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;   // in uint16s
    const int sh_c_size = MAX  // in floats
    (
        4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP,
        shmem_out_had ? TILESIZE_N * TILESIZE_M : 0
    );

    // XOR-swizzle constants for bank-conflict-free A fragment loads
    // col_swizzled = col ^ ((row >> SHIFT) & MASK)
    const int A_COLS = TILESIZE_K / 8;                                            // int4 columns per row
    const int A_SWIZZLE_MASK = A_COLS - 1;
    const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    // Sanity checks
    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M == 16 || TILESIZE_M == 8 || TILESIZE_M == 4, "Invalid kernel params");                     // strictly assume size_m <= 16
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );
    // The host filters shapes by asking exl3_gemm_smem_bytes() what this layout costs; if that
    // function and the layout above ever diverge, the filter would vet a footprint the kernel
    // does not actually have. Assert they agree, per instantiation, at compile time.
    static_assert
    (
        exl3_gemm_smem_bytes(TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES,
                             bits, shmem_out_had)
            == SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "exl3_gemm_smem_bytes() disagrees with the kernel's shared memory layout"
    );

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    // Thread index
    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    // Dimensions
    //int tiles_m = CEIL_DIVIDE(size_m, TILESIZE_M);
    int tiles_k = size_k / TILESIZE_K;
    int tiles_n = size_n / TILESIZE_N;
    //int blocks_m = 1;
    //int blocks_k = tiles_k * TILEBLOCKS_K;
    int blocks_n = tiles_n * TILEBLOCKS_N;

    // Start and end index of current slice, must span at least one tile
    int num_slices = gridDim.x;
    int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_m = [&] (int slice_i) { return 0; }; //blockIdx.y; };
    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    // Batch dimension
    // int slice_m = index_m(slice_beg);
    // int max_m = MIN(size_m - slice_m * TILESIZE_M, TILESIZE_M);
    const int slice_m = 0;

    // Pipe 0, global A, B tile and shared A, B tile
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < size_m;
    }

    int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles_k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // Pipe 1, shared A, B tile and registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles_k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    // Pipe 2
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n;

    half* gl_c_ptr_16 = ((half*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    register FragA frag_a[FRAG_STAGES];
    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    register FragC frag_c[FRAGS_N_PER_WARP];
    #if EXL3_GEMM_H_ACC
        register FragC_h frag_c_h[FRAGS_N_PER_WARP];
    #endif
    #if EXL3_VOLTA
        // Volta HMMA path (EXL3_SM70_HMMA=1). A: the four k4-chunks of each k16 step, loaded
        // directly from swizzled smem in native m8n8k4 A layout. D: per lane one 8-float
        // accumulator per quarter-round, summed over the whole k column, converted to the
        // m16n8k16 C layout once per column in reduce().
        register uint32_t va[FRAG_STAGES][4][2];
        register float dacc[FRAGS_N_PER_WARP / 2][8];
    #endif
    #if EXL3_VOLTA && EXL3_SM70_HMMA
        constexpr bool volta = true;
    #else
        constexpr bool volta = false;
    #endif

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32)
                gl_c_ptr_32 += gl_c_stride_n;
            else
                gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (sub_k)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            // Copy tile from row-major A matrix (XOR-swizzled for bank-conflict-free ldmatrix)
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
                }
            }

            // Copy tile of 256-element blocks from quantized B matrix
            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                {
                    // cp_async_pred(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i], pred_b_gl[i]);
                    if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
                }
            }
            advance0();
        }

        // Sync and advance
        cp_async_fence();
    };

    // Load fragments
    // Ref. for fragment layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A fragments (XOR-swizzled shared memory layout)
        if (volta)
        {
            #if EXL3_VOLTA
            // Native m8n8k4 A layout: lane 4q+r feeds A[r][k0..k0+3] of quarter q (k0 =
            // sub_k*16 + 4*chunk); lane 16+4q+r feeds rows r+4. For size_m <= 8 the quarters
            // pack k-halves and every quarter spans rows 0-7, so the row only depends on the
            // lane; for size_m > 8 the quarters pack row halves and rows shift by 8*rh.
            const int q_ = (lane_id >> 2) & 3;
            // Uniform (nfrag, row-half) quarter packing for every size_m: rows 8..15 carry
            // stale data when size_m <= 8 and are discarded downstream.
            const int row_v_unclamped = 8 * (q_ >> 1) + (lane_id & 3) + 4 * (lane_id >= 16);
            const int row_in_tile = (TILESIZE_M == 4) ? (lane_id & 3) : ((lane_id & 3) + 4 * (lane_id >= 16));
            const int row_v = (TILESIZE_M < 16) ? (row_in_tile < size_m ? row_in_tile : 0) : row_v_unclamped;
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                int k0 = sub_k * 16 + 4 * c;
                int c_swizzled = (k0 >> 3) ^ ((row_v >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                    sh1_a_ptr + row_v * TILESIZE_K + c_swizzled * 8 + 4 * (c & 1)));
                asm ("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(va[buf][c][0]), "=r"(va[buf][c][1]) : "r"(addr));
            }
            #endif
        }
        else
        {
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int base_c = lane_id / 16 + sub_k * 2;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                int R = r + m * 16;
                int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                ldsm4(frag_a[buf], (int4*) sh1_a_ptr + R * A_COLS + c_swizzled);
            }
        }

        // B fragments
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);

            dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }

        __syncthreads();
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            frag_c[n] = {};
        #if EXL3_GEMM_H_ACC
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c_h[n] = {};
        #endif
        #if EXL3_VOLTA
            if (volta)
            {
                #pragma unroll
                for (int r = 0; r < FRAGS_N_PER_WARP / 2; ++r)
                #pragma unroll
                for (int i = 0; i < 8; ++i) dacc[r][i] = 0.0f;
            }
        #endif
    };

    // Threadblock reduction
    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[n][j];
                }
            }
            __syncthreads();
        };

        auto add = [&] (int i)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[n][j] += *sh_red++;
                }
            }
        };

        auto store_small = [&] (int i)
        {
            if (sub_k == i && lane_id / 4 < size_m)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    *sh_red++ = frag_c[n][0];
                    *sh_red++ = frag_c[n][1];
                }
            }
            __syncthreads();
        };

        auto add_small = [&] (int i)
        {
            if (sub_k == i && lane_id / 4 < size_m)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    frag_c[n][0] += *sh_red++;
                    frag_c[n][1] += *sh_red++;
                }
            }
        };

        if (size_m <= 8)
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store_small(1);
                add_small(0);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store_small(1);
                add_small(0);
                store_small(2);
                add_small(0);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store_small(3);
                add_small(2);
                store_small(1);
                add_small(0);
                store_small(2);
                add_small(0);
            }
        }
        else
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store(1);
                add(0);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store(1);
                add(0);
                store(2);
                add(0);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store(3);
                add(2);
                store(1);
                add(0);
                store(2);
                add(0);
            }
        }
    };

    // Pre-hadamard: Write final output tile to shmem
    auto write_sum_tile_sh = [&]()
    {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        const int r0 = lane_id / 4;
        const int r1 = r0 + 8;
        if (r0 < size_m)
        {
            const int c = (lane_id % 4) * 2;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                float* c_ptr = ((float*) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[n][0];
                *c_ptr++ = frag_c[n][1];
            }
        }
        if (r1 < size_m)
        {
            const int c = (lane_id % 4) * 2;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                float* c_ptr = ((float*) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[n][2];
                *c_ptr++ = frag_c[n][3];
            }
        }
    };

    // Copy output tile to global with hadamard transform and out scale
    auto output_had_sh_gl = [&]()
    {
        int sh_warp = warp_id;
        constexpr int active_warps = EXL3_GEMM_BASE_THREADS / 32;
        for (;; sh_warp += active_warps)
        {
            int col = sh_warp % (TILESIZE_N / 128);
            int row = sh_warp / (TILESIZE_N / 128);
            if (row >= size_m) break;

            const float* had_in = sh_c + row * TILESIZE_N + col * 128;
            const half* post_scale_c = post_scale + slice2_n * gl_c_stride_n + col * 128;

            if constexpr (c_fp32)
            {
                float* had_out = gl_c_ptr_32 + row * size_n + col * 128;
                had_ff_r_128_inner<false, true, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
            else
            {
                half* had_out = gl_c_ptr_16 + row * size_n + col * 128;
                had_fh_r_128_inner<false, true, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
        }
    };

    auto read_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    frag_c[n][0] += *c_ptr++;
                    frag_c[n][1] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[n][0] += interm.x;
                    frag_c[n][1] += interm.y;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    frag_c[n][2] += *c_ptr++;
                    frag_c[n][3] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[n][2] += interm.x;
                    frag_c[n][3] += interm.y;
                }
            }
        }
    };

    auto write_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[n][0];
                    *c_ptr++ = frag_c[n][1];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[n][0], frag_c[n][1]);
                    *c_ptr = sum;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[n][2];
                    *c_ptr++ = frag_c[n][3];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[n][2], frag_c[n][3]);
                    *c_ptr = sum;
                }
            }
        }
    };

    // Output reduction
    auto reduce = [&] ()
    {
        #if EXL3_VOLTA
        if (volta)
        {
            // Convert the Volta D accumulators to the m16n8k16 C fragment layout via a
            // (m,n)-indexed smem tile: a shuffle-based gather would need dynamically indexed
            // source registers, which shfl does not provide (the source lane would read its
            // own register index). Footprint is FRAGS_N_PER_WARP * 128 floats per warp,
            // exactly the idle sh_c budget. The tile is shared by the sub_k groups
            // (warp_id wraps per 256 threads), so the groups convert sequentially.
            for (int sb = 0; sb < TILEBLOCKS_K; ++sb)
            {
            if (sub_k == sb)
            {
            float* tile = sh_c + warp_id * (FRAGS_N_PER_WARP * 128);
            const int q_ = (lane_id >> 2) & 3;
#if defined(EXL3_HMMA_DBG) && EXL3_VOLTA && EXL3_SM70_HMMA
            if (blockIdx.x == 0 && sub_k == 0)
            {
                float* dbg2 = (float*) C + (size_t) size_m * size_n / 2 + 512;
                for (int i = 0; i < 8; ++i) dbg2[lane_id * 8 + i] = dacc[0][i];
            }
#endif
            if (TILESIZE_M > 8 || (q_ >> 1 == 0))
            {
                #pragma unroll
                for (int r = 0; r < FRAGS_N_PER_WARP / 2; ++r)
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    int m, n; volta_d_slot(lane_id, i, &m, &n);
                    tile[(2 * r + (q_ & 1)) * 128 + (m + 8 * (q_ >> 1)) * 8 + n] = dacc[r][i];
                }
            }
            __syncwarp();
            const int g_ = lane_id >> 2, t_ = lane_id & 3;
            #pragma unroll
            for (int nf = 0; nf < FRAGS_N_PER_WARP; ++nf)
            {
                frag_c[nf][0] = tile[nf * 128 + g_ * 8 + 2 * t_];
                frag_c[nf][1] = tile[nf * 128 + g_ * 8 + 2 * t_ + 1];
                if (TILESIZE_M > 8 && size_m > 8)
                {
                    frag_c[nf][2] = tile[nf * 128 + (g_ + 8) * 8 + 2 * t_];
                    frag_c[nf][3] = tile[nf * 128 + (g_ + 8) * 8 + 2 * t_ + 1];
                }
            }
            }   // sub_k == sb
            __syncthreads();
            }   // sb
        }
        #endif

        #if EXL3_GEMM_H_ACC
            // Fold the fp16 MMA accumulators into the fp32 accumulators once per k-slice
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                float2 f0 = __half22float2(frag_c_h[n][0]);
                float2 f1 = __half22float2(frag_c_h[n][1]);
                frag_c[n][0] += f0.x; frag_c[n][1] += f0.y;
                frag_c[n][2] += f1.x; frag_c[n][3] += f1.y;
            }
        #endif

        // First reduce all partial sums along k for the current slice
        threadblock_reduce();

        // Process (partial) slices within column in reverse order so the threadblock doing the bottom slice is
        // free to proceed to the next column right away
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks_n + slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        // Second and subsequent threadblocks in column read back the intermediate sum from global memory
        if (!sub_k && !first)
        {
            read_sum_gl();
        }

        // All but last threadblock in column write the intermediate result to global memory
        if (!sub_k && !last)
        {
            write_sum_gl();
        }

        // Last block writes in row-major format
        if (!sub_k && last)
        {
            if constexpr (shmem_out_had)
                write_sum_tile_sh();
            else
                write_sum_gl();
        }

        if constexpr (shmem_out_had)
        {
            if (last) __syncthreads();
            if (!sub_k && last)
                output_had_sh_gl();
        }

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    // Wait until there are at most SH_STAGES - 2 async copies pending, i.e. at least one stage has finished loading
    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    // Perform tensor core matmul on current tile
    auto matmul = [&] (int buf)
    {
        #if EXL3_VOLTA
        if (volta)
        {
            // Quarters: q = (lane >> 2) & 3. size_m <= 8 packs nfrags into the quarters and
            // only needs output rows 0-7 (1 HMMA per nfrag per k16; FRAGS_N == 2 additionally
            // splits k across the quarter halves, 2 HMMAs per k16 total, merged at reduce).
            // size_m > 8 packs (nfrag, row-half) pairs (2 HMMAs per nfrag per k16).
            //
            // mma.sync is warp-collective, so every HMMA is issued unconditionally by all 32
            // lanes; the per-lane quarter role is resolved into the operands with register
            // selects (compile-time index arms), never a runtime index into a register array
            // (silently reads the wrong register) and never a divergent mma.
            const int q_ = (lane_id >> 2) & 3;
            const int cv = (lane_id & 3) + 4 * (lane_id >= 16);
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                uint32_t b0s[FRAGS_N_PER_WARP], b1s[FRAGS_N_PER_WARP];
                #pragma unroll
                for (int nf = 0; nf < FRAGS_N_PER_WARP; ++nf)
                {
                    b0s[nf] = __shfl_sync(0xffffffffu, *(const uint32_t*) &frag_b[buf][nf][c >> 1], 4 * cv + 2 * (c & 1));
                    b1s[nf] = __shfl_sync(0xffffffffu, *(const uint32_t*) &frag_b[buf][nf][c >> 1], 4 * cv + 2 * (c & 1) + 1);
                }
                #pragma unroll
                for (int r = 0; r < FRAGS_N_PER_WARP / 2; ++r)
                {
                    uint32_t bb0 = (q_ & 1) ? b0s[2 * r + 1] : b0s[2 * r];
                    uint32_t bb1 = (q_ & 1) ? b1s[2 * r + 1] : b1s[2 * r];
                    volta_mma_m8n8k4(va[buf][c][0], va[buf][c][1], bb0, bb1, dacc[r]);
                }
            }
            return;
        }
        #endif
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            #if EXL3_GEMM_H_ACC
                ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c_h[n]);
            #else
                ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[n]);
            #endif
        }
    };

    // Start global to shared pipeline
    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    // Start shared to register pipeline.
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

#if defined(EXL3_HMMA_DBG) && EXL3_VOLTA && EXL3_SM70_HMMA
    if (blockIdx.x == 0 && sub_k == 0 && volta)
    {
        uint32_t* dbg = (uint32_t*) C + (size_t) size_m * size_n / 2;
        for (int c = 0; c < 4; ++c)
        {
            dbg[lane_id * 8 + c * 2 + 0] = va[0][c][0];
            dbg[lane_id * 8 + c * 2 + 1] = va[0][c][1];
        }
        dbg[256 + lane_id * 4 + 0] = *(const uint32_t*) &frag_b[0][0][0];
        dbg[256 + lane_id * 4 + 1] = *(const uint32_t*) &frag_b[0][0][1];
        dbg[256 + lane_id * 4 + 2] = *(const uint32_t*) &frag_b[0][1][0];
        dbg[256 + lane_id * 4 + 3] = *(const uint32_t*) &frag_b[0][1][1];
    }
#endif

    // Main loop. Fragments are double buffered to allow more interleaving. This is especially important to hide the
    // dequantization overhead, but we need two different iterations of the main loop to avoid confusing the compiler
    // and making it (sometimes) place the fragment arrays in local memory

    #define FSTAGE_OLD(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \

    #define FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load); \

    if constexpr (FRAG_STAGES == 1)
    {
        while (true)
        {
            FSTAGE_OLD(0, 0);
        }
    }

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(0, 1);
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(0, 2);
        }
    }

    if constexpr (FRAG_STAGES == 4)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(0, 3);
        }
    }

    if constexpr (FRAG_STAGES == 5)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(4, 3);
            FSTAGE(0, 4);
        }
    }
}
