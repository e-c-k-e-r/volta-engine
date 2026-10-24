// Bindings for the part of upstream exllamav3's extension that its quantizer (exllamav3/modules/quant/exl3_lib/quantize.py) calls:
// the trellis encoder, the tile packers and the codebook decode. Built on first use as the module "exllamav3_ext" (see exl3_quantize.py).
#include <torch/extension.h>
#include "quant/quantize.cuh"
#include "quant/pack.cuh"
#include "quant/frac.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_tiles", &quantize_tiles, "quantize_tiles");
    m.def("quantize_tiles_scratch", &quantize_tiles_scratch, "quantize_tiles_scratch");
    m.def("quantize_tiles_frac", &quantize_tiles_frac, "quantize_tiles_frac");
    m.def("test_distribution", &test_distribution, "test_distribution");
    m.def("decode", &decode, "decode");
    m.def("pack_trellis", &pack_trellis, "pack_trellis");
    m.def("unpack_trellis", &unpack_trellis, "unpack_trellis");
    m.def("pack_trellis_frac", &pack_trellis_frac, "pack_trellis_frac");
    m.def("unpack_trellis_frac", &unpack_trellis_frac, "unpack_trellis_frac");
    m.def("pack_signs", &pack_signs, "pack_signs");
}
