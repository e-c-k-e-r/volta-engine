#pragma once

// The width code of an EXL3 trellis tensor. A whole number of bits K (1 to 8) has 16 K uint16 per 16x16 tile; the half-integer rates
// 1.5, 2.5 and 3.5 (mul1 codebook only) carry 16 K + 8 and are coded 8 + K, i.e. 9, 10 and 11, so one int still names every width the
// kernels, loaders and tables handle. In a half-integer tile the windows advance by K + 1 and K bits alternately (see dq8_half in
// exl3_dq.cuh).
#ifdef __CUDACC__
	#define ENGINE_TB_FN __host__ __device__ constexpr
#else
	#define ENGINE_TB_FN constexpr
#endif

namespace engine::modules {

constexpr int TRELLIS_HALF_CODE = 8;   // code = 8 + K for the rate K + 0.5

ENGINE_TB_FN bool trellis_is_half( int code ) { return code > TRELLIS_HALF_CODE; }
ENGINE_TB_FN int trellis_base_bits( int code ) { return code > TRELLIS_HALF_CODE ? code - TRELLIS_HALF_CODE : code; }
ENGINE_TB_FN int trellis_tile_u16( int code ) { return code > TRELLIS_HALF_CODE ? 16 * ( code - TRELLIS_HALF_CODE ) + 8 : 16 * code; }
ENGINE_TB_FN int trellis_tile_u32( int code ) { return trellis_tile_u16( code ) / 2; }
// The code of a tensor whose last dimension (uint16 per tile) is `w`
ENGINE_TB_FN int trellis_code_from_u16( int w ) { return w % 16 == 8 ? TRELLIS_HALF_CODE + w / 16 : w / 16; }
// Bits per weight in half-bit units (2 K for a whole rate, 2 K + 1 for K + 0.5)
ENGINE_TB_FN int trellis_k2( int code ) { return code > TRELLIS_HALF_CODE ? 2 * ( code - TRELLIS_HALF_CODE ) + 1 : 2 * code; }

} // namespace engine::modules

#undef ENGINE_TB_FN
