#pragma once

// The GGUF block kernels of the CPU backend (gguf_cpu.cpp: llama.cpp's own CPU kernels), for LinearGguf and MoEGguf on CPU
#include "engine/modules/gguf.h"
#include <cstdint>

namespace engine::modules::gguf_cpu {

// Whether the type has a CPU dot product and decoder
bool supported( GgmlType type );
// The block type a weight type's dot product reads its activations in (q8_0, q8_1 or q8_K)
GgmlType act_type( GgmlType type );
// One activation row of k values (FP32) in its block type
void quantize_act( GgmlType act, const float* x, void* out, int k );
// One weight row of k values times one activation row in act_type( type )
float dot( GgmlType type, const void* w_row, const void* act_row, int k );
// `elems` values of a row of blocks, in their stored order
void to_float( GgmlType type, const void* blocks, float* out, int64_t elems );
// ik_llama.cpp's iqk GEMM, for several activation rows of the types whose dot product reads q8_K: the rows quantized to ik's q8_K
// (iqk_quantize_act, iqk_act_bytes per row: not mainline's layout), out[m][n] (FP32) = the n weight rows times them, piece ith of nth.
// False when iqk has no kernel for the type (the caller takes the dot products)
bool iqk_rows( GgmlType type );
size_t iqk_act_bytes( int k );
void iqk_quantize_act( const float* x, void* out, int k );
bool mul_rows( GgmlType type, const void* w, size_t w_stride, int n, int k, const void* acts, int m, float* out, int ith, int nth );

} // namespace engine::modules::gguf_cpu
