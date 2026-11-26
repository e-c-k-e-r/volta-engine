// Vulkan backend: the vall-e TTS and NeMo codec operators (engine/modules/tts.h) on the Slang kernels of src/kernels/slang/tts.slang; the GEMMs on
// llama.cpp's matmul (FP16 with FP16 / FP32 output, and FP32 for the codec), the backbone's attention on fa_attn without slots ([L, H, hd]
// rows), the rotary table and the timestep embedding made on the host.
#include "mod_attention.h"
#include "mod_common.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/modules/tts.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
using vki::dp;

struct DigitsParams { uint64_t digits, vals; int64_t n; };
struct EmbedMixParams { uint64_t x, a, text_emb, ids, kind, masked, mask_emb, len_q, segment, mode_vec; int d; };
struct AddNormParams { uint64_t x, branch, w, out16, out32; int d; float eps; };
struct RowBiasParams { uint64_t y, bias; int64_t n; int cols; };
struct QkNormRopeParams { uint64_t qkv, qn, kn, q, k, v, pos, cs, sn; int L, H, hd, rot; float eps; int head_major; };
struct HeadsGateParams { uint64_t o, g, y; int L, H, hd, head_major; };
struct TtsSwigluParams { uint64_t up, act; int F; };
struct GatherParams { uint64_t src, pos, dst; int offset, d; };
struct BiasActHParams { uint64_t x, bias, y; int64_t n; int cols, act; };
struct CondSiluParams { uint64_t cond, temb, sc; int64_t n; int W; };
struct LnModParams { uint64_t x, mod, out_; int mstride, W; };
struct GateAddParams { uint64_t x, branch, bias, mod; int64_t n; int mstride, W; };
struct ReplicateParams { uint64_t x, y; int64_t total; int n; };
struct GuideParams { uint64_t v, bias, x, vt; int n, V, R; float cfg, cfg_prompt, dt; };
struct SnapParams { uint64_t x, out_; int n; };
struct DequantParams { uint64_t digits, x; int T, h0, h1, h2, h3; };
struct QuantParams { uint64_t x, digits; int T; float sc[4], of[4], sh[4]; int hl[4]; };
struct CodecActParams { uint64_t x, y, alpha; int64_t n; int T, sc; };
struct ColsParams { uint64_t x, cols; int ci, k, dil, stride, pad, T, t0, n; };
struct UpColsParams { uint64_t x, cols; int ci, T, q0, n; };
struct InterleaveParams { uint64_t ph, bias, y; int co, s, T_out, q0, n; };
struct ChannelBiasParams { uint64_t y, b; int64_t n; int T; };
struct AxpyParams { uint64_t y, x; int64_t n; float a, b; };
struct ClampParams { uint64_t y; int64_t n; };

// n elements, 256 a workgroup, the groups over x (at most 65535) and y
template<typename P>
void launch_n( const char* name, const P& p, int64_t n, backend::stream_t s ) {
	const int64_t groups = ( n + 255 ) / 256;
	vki::launch( name, p, s, static_cast<uint32_t>( std::min<int64_t>( groups, 65535 ) ), static_cast<uint32_t>( ( groups + 65534 ) / 65535 ) );
}

// The matmul's tile rows: a chunk of columns is a whole number of them, so the tile loads past its last row stay inside the columns
constexpr int kTileRows = 128;

} // namespace

void TtsOps::gemm( const backend::half_t* A, const backend::half_t* W, void* C, int M, int N, int K, bool c32, backend::stream_t stream ) {
	ENGINE_ASSERT( vki::ggml_f16_matmul( W, A, C, M, N, K, c32, stream ), "TtsOps::gemm (Vulkan): needs K a multiple of 32 and 16-byte aligned operands" );
}

void TtsOps::digits_values( const uint8_t* digits, backend::half_t* vals, int n, backend::stream_t stream ) {
	launch_n( "tts_digits_values", DigitsParams{ dp( digits ), dp( vals ), n }, n, stream );
}

void TtsOps::embed_mix( float* x, const float* a, const backend::half_t* text_emb, const int* ids, const uint8_t* kind, const uint8_t* masked, const float* mask_emb,
	const float* len_q, const float* segment, const float* mode_vec, int L, int d, backend::stream_t stream ) {
	EmbedMixParams p{ dp( x ), dp( a ), dp( text_emb ), dp( ids ), dp( kind ), dp( masked ), dp( mask_emb ), dp( len_q ), dp( segment ), dp( mode_vec ), d };
	vki::launch( "tts_embed_mix", p, stream, static_cast<uint32_t>( L ) );
}

// On the host as the CUDA kernel's (powf frequencies, sincosf): the GPU's sin / cos lose precision at the angles of long rows
void TtsOps::rope_table( float* cs, float* sn, int n, int rot, float theta, backend::stream_t stream ) {
	const int half_rot = rot / 2;
	std::vector<float> c( static_cast<size_t>( n ) * half_rot ), s( c.size() );
	for ( int p = 0; p < n; ++p ) {
		for ( int i = 0; i < half_rot; ++i ) {
			const float a = static_cast<float>( p ) * std::pow( theta, -( 2.0f * i / static_cast<float>( rot ) ) );
			c[static_cast<size_t>( p ) * half_rot + i] = std::cos( a );
			s[static_cast<size_t>( p ) * half_rot + i] = std::sin( a );
		}
	}
	backend::memcpy_async( cs, c.data(), c.size() * 4, backend::Copy::H2D, stream );
	backend::memcpy_async( sn, s.data(), s.size() * 4, backend::Copy::H2D, stream );
	backend::stream_sync( stream );
}

void TtsOps::add_rms_norm( float* x, const float* branch, const float* w, int rows, int d, float eps, backend::half_t* out16, float* out32, backend::stream_t stream ) {
	ENGINE_ASSERT( d <= 256 * 24, "TtsOps::add_rms_norm (Vulkan): d at most 6144" );
	vki::launch( "tts_add_norm", AddNormParams{ dp( x ), dp( branch ), dp( w ), dp( out16 ), dp( out32 ), d, eps }, stream, static_cast<uint32_t>( rows ) );
}

void TtsOps::add_bias( float* y, const float* bias, int rows, int cols, backend::stream_t stream ) {
	const int64_t n = static_cast<int64_t>( rows ) * cols;
	launch_n( "tts_add_bias", RowBiasParams{ dp( y ), dp( bias ), n, cols }, n, stream );
}

bool TtsOps::head_major( int ) { return false; }

size_t TtsOps::attention_scratch( int, int, int ) { return 0; }

void TtsOps::qk_norm_rope( const backend::half_t* qkv, const float* qn, const float* kn, backend::half_t* q, backend::half_t* k, backend::half_t* v, const int* pos, int L, int H,
	int hd, int rot, const float* cs, const float* sn, float eps, bool head_major, backend::stream_t stream ) {
	ENGINE_ASSERT( H * hd <= 4096 && H <= 64, "TtsOps::qk_norm_rope (Vulkan): at most 64 heads and 4096 channels" );
	QkNormRopeParams p{ dp( qkv ), dp( qn ), dp( kn ), dp( q ), dp( k ), dp( v ), dp( pos ), dp( cs ), dp( sn ), L, H, hd, rot, eps, head_major ? 1 : 0 };
	vki::launch( "tts_qk_norm_rope", p, stream, static_cast<uint32_t>( L ) );
}

void TtsOps::attention( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, backend::half_t* o, int s, int Lr, int L, int H, int hd, float scale,
	void*, size_t, backend::stream_t stream ) {
	( void ) L;
	const size_t at = static_cast<size_t>( s ) * H * hd;   // [L, H, hd]
	vk_attn::dense( q + at, k + at, v + at, o + at, Lr, Lr, H, H, hd, scale, stream );
}

void TtsOps::heads_gate( const backend::half_t* o, const backend::half_t* g, backend::half_t* y, int L, int H, int hd, bool head_major, backend::stream_t stream ) {
	vki::launch( "tts_heads_gate", HeadsGateParams{ dp( o ), dp( g ), dp( y ), L, H, hd, head_major ? 1 : 0 }, stream, static_cast<uint32_t>( L ) );
}

void TtsOps::swiglu( const backend::half_t* up, backend::half_t* act, int rows, int F, backend::stream_t stream ) {
	vki::launch( "tts_swiglu", TtsSwigluParams{ dp( up ), dp( act ), F }, stream, static_cast<uint32_t>( rows ) );
}

void TtsOps::gather_rows( const float* src, const int* pos, int offset, backend::half_t* dst, int n, int d, backend::stream_t stream ) {
	vki::launch( "tts_gather_rows", GatherParams{ dp( src ), dp( pos ), dp( dst ), offset, d }, stream, static_cast<uint32_t>( n ) );
}

// On the host as the CUDA kernel's (expf / cosf / sinf in FP32): the angles reach 1000 radians, where the GPU's sin / cos lose precision
void TtsOps::time_embedding( float t, backend::half_t* out, backend::stream_t stream ) {
	backend::half_t e[256];
	for ( int i = 0; i < 128; ++i ) {
		const float f = std::exp( -std::log( 10000.0f ) * i / 128.0f );
		const float a = t * 1000.0f * f;
		e[i] = backend::from_float_half( std::cos( a ) );
		e[128 + i] = backend::from_float_half( std::sin( a ) );
	}
	backend::memcpy_async( out, e, sizeof( e ), backend::Copy::H2D, stream );
}

void TtsOps::bias_act( const float* x, const float* bias, backend::half_t* y, int rows, int cols, bool silu, backend::stream_t stream ) {
	const int64_t n = static_cast<int64_t>( rows ) * cols;
	launch_n( "tts_bias_act", BiasActHParams{ dp( x ), dp( bias ), dp( y ), n, cols, silu ? 1 : 0 }, n, stream );
}

void TtsOps::cond_silu( const float* cond, const float* temb, backend::half_t* sc, int rows, int W, backend::stream_t stream ) {
	const int64_t n = static_cast<int64_t>( rows ) * W;
	launch_n( "tts_cond_silu", CondSiluParams{ dp( cond ), dp( temb ), dp( sc ), n, W }, n, stream );
}

void TtsOps::ln_mod( const float* x, const float* mod, int mstride, backend::half_t* out, int rows, int W, backend::stream_t stream ) {
	vki::launch( "tts_ln_mod", LnModParams{ dp( x ), dp( mod ), dp( out ), mstride, W }, stream, static_cast<uint32_t>( rows ) );
}

void TtsOps::gate_add( float* x, const float* branch, const float* bias, const float* mod, int mstride, int rows, int W, backend::stream_t stream ) {
	const int64_t n = static_cast<int64_t>( rows ) * W;
	launch_n( "tts_gate_add", GateAddParams{ dp( x ), dp( branch ), dp( bias ), dp( mod ), n, mstride, W }, n, stream );
}

void TtsOps::replicate( const float* x, backend::half_t* y, int n, int R, backend::stream_t stream ) {
	const int64_t total = static_cast<int64_t>( n ) * R;
	launch_n( "tts_replicate", ReplicateParams{ dp( x ), dp( y ), total, n }, total, stream );
}

void TtsOps::guide_step( const float* v, const float* bias, float* x, float* vt, int n, int V, int R, float cfg, float cfg_prompt, float dt, backend::stream_t stream ) {
	launch_n( "tts_guide_step", GuideParams{ dp( v ), dp( bias ), dp( x ), dp( vt ), n, V, R, cfg, cfg_prompt, dt }, n, stream );
}

void TtsOps::snap( const float* x, uint8_t* out, int n, backend::stream_t stream ) {
	launch_n( "tts_snap", SnapParams{ dp( x ), dp( out ), n }, n, stream );
}

// ---- the codec: the columns K-major ([time, ci * k]), so a chunk is the matmul's A against the weight as its B and the output lands
// channel-major in y (rows T_out apart), at a chunk start that is a multiple of the tile rows (16-byte aligned)

void CodecOps::conv( const float* x, const float* w, const float* b, float* y, int ci, int co, int k, int dil, int stride, int pad, int T, int T_out, float* cols,
	size_t chunk, backend::stream_t stream ) {
	const int K = ci * k;
	ENGINE_ASSERT( chunk / K >= static_cast<size_t>( kTileRows ), "CodecOps::conv (Vulkan): a chunk holds fewer than 128 columns" );
	const int n = static_cast<int>( chunk / K / kTileRows * kTileRows );
	for ( int t0 = 0; t0 < T_out; t0 += n ) {
		const int m = std::min( n, T_out - t0 );
		launch_n( "codec_cols", ColsParams{ dp( x ), dp( cols ), ci, k, dil, stride, pad, T, t0, m }, static_cast<int64_t>( m ) * K, stream );
		ENGINE_ASSERT( vki::ggml_f32_matmul( cols, w, y + t0, m, co, K, T_out, stream ), "CodecOps::conv (Vulkan): the FP32 matmul" );
	}
	const int64_t total = static_cast<int64_t>( co ) * T_out;
	launch_n( "codec_channel_bias", ChannelBiasParams{ dp( y ), dp( b ), total, T_out }, total, stream );
}

void CodecOps::up_conv( const float* x, const float* w, const float* b, float* y, int ci, int co, int s, int T_in, float* cols, float* ph, size_t chunk,
	backend::stream_t stream ) {
	const int K = 3 * ci, rows = s * co;
	const size_t fit = std::min( chunk / K, chunk / rows );
	ENGINE_ASSERT( fit >= static_cast<size_t>( kTileRows ), "CodecOps::up_conv (Vulkan): a chunk holds fewer than 128 columns" );
	const int n = static_cast<int>( fit / kTileRows * kTileRows );
	for ( int q0 = 0; q0 < T_in; q0 += n ) {
		const int m = std::min( n, T_in - q0 );
		launch_n( "codec_up_cols", UpColsParams{ dp( x ), dp( cols ), ci, T_in, q0, m }, static_cast<int64_t>( m ) * K, stream );
		ENGINE_ASSERT( vki::ggml_f32_matmul( cols, w, ph, m, rows, K, m, stream ), "CodecOps::up_conv (Vulkan): the FP32 matmul" );
		launch_n( "codec_interleave", InterleaveParams{ dp( ph ), dp( b ), dp( y ), co, s, T_in * s, q0, m }, static_cast<int64_t>( rows ) * m, stream );
	}
}

void CodecOps::act( const float* x, float* y, const float* alpha, int C, int T, backend::stream_t stream ) {
	const int64_t n = static_cast<int64_t>( C ) * T;
	launch_n( "codec_act", CodecActParams{ dp( x ), dp( y ), dp( alpha ), n, T, alpha ? C / 2 : 0 }, n, stream );
}

void CodecOps::axpy( float* y, const float* x, float a, float b, size_t n, backend::stream_t stream ) {
	launch_n( "codec_axpy", AxpyParams{ dp( y ), dp( x ), static_cast<int64_t>( n ), a, b }, static_cast<int64_t>( n ), stream );
}

void CodecOps::clamp( float* y, size_t n, backend::stream_t stream ) {
	launch_n( "codec_clamp", ClampParams{ dp( y ), static_cast<int64_t>( n ) }, static_cast<int64_t>( n ), stream );
}

void CodecOps::dequant( const uint8_t* digits, float* x, int T, const int* levels, backend::stream_t stream ) {
	DequantParams p{ dp( digits ), dp( x ), T, levels[0] / 2, levels[1] / 2, levels[2] / 2, levels[3] / 2 };
	launch_n( "codec_dequant", p, static_cast<int64_t>( 32 ) * T, stream );
}

void CodecOps::quant( const float* x, uint8_t* digits, int T, const int* levels, float eps, backend::stream_t stream ) {
	QuantParams p{ dp( x ), dp( digits ), T, {}, {}, {}, {} };
	for ( int j = 0; j < 4; ++j ) {
		const int L = levels[j];
		p.sc[j] = ( L - 1 ) * 0.5f * ( 1.0f - eps );
		p.of[j] = L % 2 == 0 ? 0.5f : 0.0f;
		p.sh[j] = std::tan( p.of[j] / p.sc[j] );
		p.hl[j] = L / 2;
	}
	launch_n( "codec_quant", p, static_cast<int64_t>( 32 ) * T, stream );
}

} // namespace engine::modules
