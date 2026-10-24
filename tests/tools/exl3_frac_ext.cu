// Standalone build of upstream exllamav3's fractional-rate quantizer and tile packer (the installed extension predates them)
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/codebook.cuh"
#include "quant/frac.cuh"
#include "quant/comp_units/quantize_tiles_instances.cuh"

void quantize_tiles_frac(at::Tensor input_tiles, at::Tensor output_tiles, at::Tensor output_indices, at::Tensor temp_costs, at::Tensor temp_edges, int KA, int64_t MASK)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int L = input_tiles.size(1);
    TORCH_CHECK(L == 256, "tile length must be 256");
    fp_quantize_tiles_kernel kernel = nullptr;
    struct { int ka; uint32_t mask; fp_quantize_tiles_kernel (*fn)(); } const table[] = {
        {1, 0xaaaau, &quantize_tiles_frac_kernel_a1_maaaa},
        {2, 0xaaaau, &quantize_tiles_frac_kernel_a2_maaaa},
        {3, 0xaaaau, &quantize_tiles_frac_kernel_a3_maaaa},
    };
    for (const auto& e : table) if (e.ka == KA && e.mask == (uint32_t) MASK) kernel = e.fn();
    TORCH_CHECK(kernel, "no instance for (KA, MASK)");
    const int num_tiles = input_tiles.size(0);
    if (!num_tiles) return;
    const int shmem = L * sizeof(half) + 32 * sizeof(int) + 128;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    const int max_batch_size = (int) MIN(temp_costs.size(0), temp_edges.size(0));
    for (int batch_i = 0; batch_i < num_tiles; batch_i += max_batch_size)
    {
        const int bsz = MIN(max_batch_size, num_tiles - batch_i);
        kernel<<<bsz, 512, shmem, stream>>>(
            ((const float*) input_tiles.data_ptr()) + (int64_t) L * batch_i,
            ((float*) output_tiles.data_ptr()) + (int64_t) L * batch_i,
            ((uint16_t*) output_indices.data_ptr()) + (int64_t) L * batch_i,
            (half*) temp_costs.data_ptr(), (uint16_t*) temp_edges.data_ptr(), nullptr);
        cuda_check(cudaPeekAtLastError());
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_tiles_frac", &quantize_tiles_frac, "quantize_tiles_frac");
    m.def("pack_trellis_frac", &pack_trellis_frac, "pack_trellis_frac");
    m.def("unpack_trellis_frac", &unpack_trellis_frac, "unpack_trellis_frac");
}
