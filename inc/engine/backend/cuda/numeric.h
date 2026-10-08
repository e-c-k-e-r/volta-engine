#pragma once

#include "engine/backend/types.h"

// CUDA backend: matching cuda_fp16/cuda_bf16 intrinsics (round to nearest even)
namespace engine::backend {

inline float to_float( half_t h ) { return __half2float( h ); }
inline float to_float( bf16_t b ) { return __bfloat162float( b ); }

// float -> fp16, round to nearest even
inline half_t from_float_half( float f ) { return __float2half_rn( f ); }
// float -> bf16, round to nearest even
inline bf16_t from_float_bf16( float f ) { return __float2bfloat16_rn( f ); }

} // namespace engine::backend
