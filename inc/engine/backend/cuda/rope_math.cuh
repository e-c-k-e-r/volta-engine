#pragma once

// Rotary-embedding math shared by every CUDA / HIP kernel that rotates a vector by a position: RoPE::forward (attention q and k), the
// selector prep kernels (qsa_prep) and the drafter's fused norm + rope. One place for the frequency formula and for the choice of the
// sin / cos implementation, so an optimization or a precision change is made once.
#include <cmath>

namespace engine::math {

// Inverse frequency of rotary pair i out of rot_dim channels: base^(-2i / rot_dim)
__device__ __forceinline__ float rope_inv_freq( int i, int rot_dim, float base ) {
	return powf( base, -( 2.0f * i / static_cast<float>( rot_dim ) ) );
}

// cos and sin of pos * inv_freq(i). Fast uses the __cosf / __sinf intrinsics (what the attention RoPE has always used); otherwise sincosf
// (the selector prep kernels and the drafter).
template <bool Fast>
__device__ __forceinline__ void rope_cos_sin( int pos, int i, int rot_dim, float base, float& c, float& s ) {
	const float theta = static_cast<float>( pos ) * rope_inv_freq( i, rot_dim, base );
	if constexpr ( Fast ) {
		c = __cosf( theta );
		s = __sinf( theta );
	}
	else {
		sincosf( theta, &s, &c );
	}
}

} // namespace engine::math
