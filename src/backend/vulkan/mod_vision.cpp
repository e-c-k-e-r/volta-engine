// Vulkan backend: the vision-tower operators on the Slang kernels of src/kernels/slang/vision.slang (norms, patch embedding, interpolated position
// table, rotary with biases, bidirectional attention, bias / activation / residual helpers, the merger's gather); the dense GEMM is the plain FP16 GEMM.
#include "mod_common.h"
#include "engine/modules/linear.h"
#include "engine/modules/vision.h"
#include <algorithm>

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;

inline uint32_t tiles( int64_t n, int tile ) { return static_cast<uint32_t>( ( n + tile - 1 ) / tile ); }

}

void Vision::f32_to_f16( const float* in, backend::half_t* out, size_t n, backend::stream_t stream ) {
	vki::F32ToF16Params p{ vki::dp( in ), vki::dp( out ), static_cast<int64_t>( n ) };
	const int64_t groups = ( static_cast<int64_t>( n ) + 255 ) / 256;
	vki::launch( "vision_f32_to_f16", p, stream, static_cast<uint32_t>( std::min<int64_t>( groups, 65535 ) ), static_cast<uint32_t>( ( groups + 65534 ) / 65535 ) );
}

void Vision::pos_interp( const backend::half_t* table, backend::half_t* out, int gh, int gw, int merge, int side, int dim, backend::stream_t stream ) {
	vki::PosInterpParams p{ vki::dp( table ), vki::dp( out ), gh, gw, merge, side, dim };
	vki::launch( "vision_pos_interp", p, stream, gh * gw );
}

void Vision::patch_embed( const backend::half_t* patches, const backend::half_t* w, const backend::half_t* bias, const backend::half_t* add, float* out, int m, int n, int k, backend::stream_t stream ) {
	vki::PatchEmbedParams p{ vki::dp( patches ), vki::dp( w ), vki::dp( bias ), vki::dp( add ), vki::dp( out ), m, n, k, add != nullptr ? 1 : 0 };
	vki::launch( "vision_patch_embed", p, stream, tiles( n, 64 ), m );
}

void Vision::rms_norm( const float* x, const backend::half_t* w, backend::half_t* out, int m, int n, float eps, backend::stream_t stream ) {
	vki::VNormParams p{ vki::dp( x ), 0, vki::dp( w ), 0, vki::dp( out ), n, 1, 2, eps };
	vki::launch( "vision_norm", p, stream, m );
}

void Vision::layer_norm_half( const backend::half_t* x, const backend::half_t* w, const backend::half_t* b, backend::half_t* out, int m, int n, float eps, backend::stream_t stream ) {
	vki::VNormParams p{ 0, vki::dp( x ), vki::dp( w ), vki::dp( b ), vki::dp( out ), n, 1, 1, eps };
	vki::launch( "vision_norm", p, stream, m );
}

void Vision::layer_norm( const float* x, const backend::half_t* w, const backend::half_t* b, backend::half_t* out, int m, int n, float eps, backend::stream_t stream ) {
	vki::VNormParams p{ vki::dp( x ), 0, vki::dp( w ), vki::dp( b ), vki::dp( out ), n, 1, 0, eps };
	vki::launch( "vision_norm", p, stream, m );
}

void Vision::layer_norm_merge( const float* x, const backend::half_t* w, const backend::half_t* b, backend::half_t* out, int m, int n, int group, float eps, backend::stream_t stream ) {
	ENGINE_ASSERT( m % group == 0, "Vision::layer_norm_merge: rows must be a multiple of the group" );
	vki::VNormParams p{ vki::dp( x ), 0, vki::dp( w ), vki::dp( b ), vki::dp( out ), n, group, 0, eps };
	vki::launch( "vision_norm", p, stream, m );
}

void Vision::qkv_bias_norm_rope( backend::half_t* q, backend::half_t* k, backend::half_t* v, const backend::half_t* bq, const backend::half_t* bk, const backend::half_t* bv,
	const backend::half_t* qn, const backend::half_t* kn, const float* cos_table, const float* sin_table, int m, int heads, int head_dim, float eps, backend::stream_t stream ) {
	ENGINE_ASSERT( head_dim == 64 && heads <= 32, "Vision::qkv_bias_norm_rope: 64-wide heads, at most 32" );
	vki::QkvRopeParams p{ vki::dp( q ), vki::dp( k ), vki::dp( v ), vki::dp( bq ), vki::dp( bk ), vki::dp( bv ), vki::dp( qn ), vki::dp( kn ), vki::dp( cos_table ), vki::dp( sin_table ), heads, head_dim, eps, 1 };
	vki::launch( "vision_qkv_rope", p, stream, m );
}

void Vision::qkv_bias_rope( backend::half_t* q, backend::half_t* k, backend::half_t* v, const backend::half_t* bq, const backend::half_t* bk, const backend::half_t* bv, const float* cos_table,
	const float* sin_table, int m, int heads, int head_dim, backend::stream_t stream ) {
	ENGINE_ASSERT( head_dim % 2 == 0 && heads * head_dim / 2 <= 1024, "Vision::qkv_bias_rope: even head_dim, at most 1024 rotary pairs per row" );
	vki::QkvRopeParams p{ vki::dp( q ), vki::dp( k ), vki::dp( v ), vki::dp( bq ), vki::dp( bk ), vki::dp( bv ), 0, 0, vki::dp( cos_table ), vki::dp( sin_table ), heads, head_dim, 0.0f, 0 };
	vki::launch( "vision_qkv_rope", p, stream, m );
}

void Vision::merge_gather( const backend::half_t* h, backend::half_t* out, int tokens, int n, int group, backend::stream_t stream ) {
	vki::MergeGatherParams p{ vki::dp( h ), vki::dp( out ), n, group };
	vki::launch( "vision_merge_gather", p, stream, tokens );
}

void Vision::dense_gemm( const backend::half_t* a, const backend::half_t* w, backend::half_t* out, int m, int n, int k, backend::stream_t stream ) {
	LinearPlainGEMV::forward( w, a, out, m, n, k, stream );
}

void Vision::attention( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, backend::half_t* out, int m, int heads, int head_dim, float scale, backend::stream_t stream ) {
	attention_simt( q, k, v, out, m, heads, head_dim, scale, stream );
}

void Vision::attention_simt( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, backend::half_t* out, int m, int heads, int head_dim, float scale, backend::stream_t stream ) {
	ENGINE_ASSERT( head_dim <= 256, "Vision::attention (Vulkan): head_dim <= 256" );
	vki::VAttnParams p{ vki::dp( q ), vki::dp( k ), vki::dp( v ), vki::dp( out ), m, heads, head_dim, scale };
	vki::launch( "vision_attention", p, stream, heads, m );
}

void Vision::bias_act( backend::half_t* x, const backend::half_t* bias, int m, int n, int act, backend::stream_t stream ) {
	const int64_t total = static_cast<int64_t>( m ) * n;
	vki::BiasActParams p{ vki::dp( x ), vki::dp( bias ), total, n, act, bias != nullptr ? 1 : 0 };
	const int64_t groups = ( total + 255 ) / 256;
	vki::launch( "vision_bias_act", p, stream, static_cast<uint32_t>( std::min<int64_t>( groups, 65535 ) ), static_cast<uint32_t>( ( groups + 65534 ) / 65535 ) );
}

void Vision::bias_residual( float* hidden, const backend::half_t* y, const backend::half_t* bias, int m, int n, backend::stream_t stream ) {
	const int64_t total = static_cast<int64_t>( m ) * n;
	vki::BiasResidualParams p{ vki::dp( hidden ), vki::dp( y ), vki::dp( bias ), total, n };
	const int64_t groups = ( total + 255 ) / 256;
	vki::launch( "vision_bias_residual", p, stream, static_cast<uint32_t>( std::min<int64_t>( groups, 65535 ) ), static_cast<uint32_t>( ( groups + 65534 ) / 65535 ) );
}

} // namespace engine::modules
