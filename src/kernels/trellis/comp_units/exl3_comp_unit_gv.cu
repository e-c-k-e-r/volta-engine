// Instantiations for the sm70 m=1 GEMV-class mgemm restructure (exl3_mgemm_gv_kernel)
#include "../exl3_mgemm_gv_kernel.cuh"

template __global__ void exl3_mgemm_gv_kernel<2, false, 0, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<2, false, 2, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<3, false, 0, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<3, false, 2, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<4, false, 0, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<4, false, 2, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<5, false, 2, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<2, true, 0, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<2, true, 2, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<3, true, 0, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<3, true, 2, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<4, true, 2, false>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<5, true, 2, false>(EXL3_MGEMM_ARGS);

void* get_mgemm_gv_kernel(int K, bool c_fp32, int cb)
{
    if (cb == 0)
    {
        if (K == 2) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<2, true, 0, false>
                                  : (void*) exl3_mgemm_gv_kernel<2, false, 0, false>;
        if (K == 3) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<3, true, 0, false>
                                  : (void*) exl3_mgemm_gv_kernel<3, false, 0, false>;
        if (K == 4) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<4, true, 0, false>
                                  : (void*) exl3_mgemm_gv_kernel<4, false, 0, false>;
    }
    if (cb == 2)
    {
        if (K == 2) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<2, true, 2, false>
                                  : (void*) exl3_mgemm_gv_kernel<2, false, 2, false>;
        if (K == 3) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<3, true, 2, false>
                                  : (void*) exl3_mgemm_gv_kernel<3, false, 2, false>;
        if (K == 4) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<4, true, 2, false>
                                  : (void*) exl3_mgemm_gv_kernel<4, false, 2, false>;
        if (K == 5) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<5, true, 2, false>
                                  : (void*) exl3_mgemm_gv_kernel<5, false, 2, false>;
    }
    return nullptr;
}

// v2 (EXL3_SM70_CB0V2): K2/cb0 batch-pipelined inner, A staged as float
template __global__ void exl3_mgemm_gv_kernel<2, false, 0, true>(EXL3_MGEMM_ARGS);
template __global__ void exl3_mgemm_gv_kernel<2, true, 0, true>(EXL3_MGEMM_ARGS);

void* get_mgemm_gv2_kernel(int K, bool c_fp32, int cb)
{
    if (K == 2 && cb == 0)
        return c_fp32 ? (void*) exl3_mgemm_gv_kernel<2, true, 0, true>
                      : (void*) exl3_mgemm_gv_kernel<2, false, 0, true>;
    return nullptr;
}

// Instantiations for the sm70 m=1 GEMV-class single gemm restructure (exl3_gemm_gv_kernel)
template __global__ void exl3_gemm_gv_kernel<2, false, 0, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<2, false, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<3, false, 0, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<3, false, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<4, false, 0, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<4, false, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<5, false, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<6, false, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<7, false, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<2, true, 0, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<2, true, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<3, true, 0, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<3, true, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<4, true, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<5, true, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<6, true, 2, false>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<7, true, 2, false>(EXL3_GEMM_ARGS);

void* get_gemm_gv_kernel(int K, bool c_fp32, int cb)
{
    if (cb == 0)
    {
        if (K == 2) return c_fp32 ? (void*) exl3_gemm_gv_kernel<2, true, 0, false>
                                  : (void*) exl3_gemm_gv_kernel<2, false, 0, false>;
        if (K == 3) return c_fp32 ? (void*) exl3_gemm_gv_kernel<3, true, 0, false>
                                  : (void*) exl3_gemm_gv_kernel<3, false, 0, false>;
        if (K == 4) return c_fp32 ? (void*) exl3_gemm_gv_kernel<4, true, 0, false>
                                  : (void*) exl3_gemm_gv_kernel<4, false, 0, false>;
    }
    if (cb == 2)
    {
        if (K == 2) return c_fp32 ? (void*) exl3_gemm_gv_kernel<2, true, 2, false>
                                  : (void*) exl3_gemm_gv_kernel<2, false, 2, false>;
        if (K == 3) return c_fp32 ? (void*) exl3_gemm_gv_kernel<3, true, 2, false>
                                  : (void*) exl3_gemm_gv_kernel<3, false, 2, false>;
        if (K == 4) return c_fp32 ? (void*) exl3_gemm_gv_kernel<4, true, 2, false>
                                  : (void*) exl3_gemm_gv_kernel<4, false, 2, false>;
        if (K == 5) return c_fp32 ? (void*) exl3_gemm_gv_kernel<5, true, 2, false>
                                  : (void*) exl3_gemm_gv_kernel<5, false, 2, false>;
        if (K == 6) return c_fp32 ? (void*) exl3_gemm_gv_kernel<6, true, 2, false>
                                  : (void*) exl3_gemm_gv_kernel<6, false, 2, false>;
        if (K == 7) return c_fp32 ? (void*) exl3_gemm_gv_kernel<7, true, 2, false>
                                  : (void*) exl3_gemm_gv_kernel<7, false, 2, false>;
    }
    return nullptr;
}

// v2 (EXL3_SM70_CB0V2): K2/cb0 batch-pipelined inner, A staged as float
template __global__ void exl3_gemm_gv_kernel<2, false, 0, true>(EXL3_GEMM_ARGS);
template __global__ void exl3_gemm_gv_kernel<2, true, 0, true>(EXL3_GEMM_ARGS);

void* get_gemm_gv2_kernel(int K, bool c_fp32, int cb)
{
    if (K == 2 && cb == 0)
        return c_fp32 ? (void*) exl3_gemm_gv_kernel<2, true, 0, true>
                      : (void*) exl3_gemm_gv_kernel<2, false, 0, true>;
    return nullptr;
}
