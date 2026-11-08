// Vulkan backend: GGUF block weights on the Slang kernels of src/kernels/slang/gguf.slang (the weight decoded in the shader, fp32
// accumulation against the FP16 activations: no activation quantization). The IQ grids, sign table and IQ4 values come from llama.cpp's
// ggml-common.h (vendored in src/kernels/cpu/gguf/ggml) into one word buffer per device, in the order the shader's offsets name.
#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "../../kernels/cpu/gguf/ggml/ggml-common.h"

#include "mod_common.h"
#include "engine/backend/runtime.h"
#include "engine/core/batch.h"
#include "engine/modules/gguf.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;

struct GgufParams { uint64_t w, tables, x, out_; int64_t row_bytes; int type, m, k, n, heads, dim; };
struct GgufQuantParams { uint64_t xv, xq; int m, k, heads, dim; };
struct GgufDotParams { uint64_t w, tables, xq, out_; int64_t row_bytes; int type, m, k, n; };
struct GgufPermuteParams { uint64_t x, out_; int m, k, heads, dim; };
// llama.cpp's mat-vec push constants (mul_mat_vec_base.glsl, without MUL_MAT_ID)
struct MmvParams { uint32_t ncols, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, fusion_flags, base_work_group_y, ne02, ne12,
	broadcast2, broadcast3; };
struct GgufDequantParams { uint64_t w, tables, out_; int64_t elems; int type; };

const void* tables() {
	static std::mutex mu;
	static void* buf[64] = {};
	const int dev = backend::get_device();
	std::lock_guard<std::mutex> lk( mu );
	if ( buf[dev] == nullptr ) {
		std::vector<uint8_t> t;
		auto put = [&]( const void* p, size_t bytes ) { t.insert( t.end(), static_cast<const uint8_t*>( p ), static_cast<const uint8_t*>( p ) + bytes ); };
		put( iq2xxs_grid, sizeof( iq2xxs_grid ) );   // word 0
		put( iq2xs_grid, sizeof( iq2xs_grid ) );     // 512
		put( iq2s_grid, sizeof( iq2s_grid ) );       // 1536
		put( iq3xxs_grid, sizeof( iq3xxs_grid ) );   // 3584
		put( iq3s_grid, sizeof( iq3s_grid ) );       // 3840
		put( iq1s_grid, sizeof( iq1s_grid ) );       // 4352
		put( ksigns_iq2xs, sizeof( ksigns_iq2xs ) ); // 8448
		put( kvalues_iq4nl, sizeof( kvalues_iq4nl ) );   // 8480
		ENGINE_ASSERT( t.size() == 8484 * 4, "GGUF (Vulkan): table layout does not match gguf.slang" );
		buf[dev] = backend::device_malloc( t.size() );
		backend::memcpy_sync( buf[dev], t.data(), t.size(), backend::Copy::H2D );
	}
	return buf[dev];
}

// Bytes per row of k values (the module layer links without the loader's ggml_block)
int64_t row_bytes( GgmlType t, int k ) {
	switch ( t ) {
		case GgmlType::Q4_0: return k / QK4_0 * sizeof( block_q4_0 );
		case GgmlType::Q4_1: return k / QK4_1 * sizeof( block_q4_1 );
		case GgmlType::Q5_0: return k / QK5_0 * sizeof( block_q5_0 );
		case GgmlType::Q5_1: return k / QK5_1 * sizeof( block_q5_1 );
		case GgmlType::Q8_0: return k / QK8_0 * sizeof( block_q8_0 );
		case GgmlType::IQ4_NL: return k / QK4_NL * sizeof( block_iq4_nl );
		case GgmlType::Q2_K: return k / QK_K * sizeof( block_q2_K );
		case GgmlType::Q3_K: return k / QK_K * sizeof( block_q3_K );
		case GgmlType::Q4_K: return k / QK_K * sizeof( block_q4_K );
		case GgmlType::Q5_K: return k / QK_K * sizeof( block_q5_K );
		case GgmlType::Q6_K: return k / QK_K * sizeof( block_q6_K );
		case GgmlType::IQ2_XXS: return k / QK_K * sizeof( block_iq2_xxs );
		case GgmlType::IQ2_XS: return k / QK_K * sizeof( block_iq2_xs );
		case GgmlType::IQ2_S: return k / QK_K * sizeof( block_iq2_s );
		case GgmlType::IQ3_XXS: return k / QK_K * sizeof( block_iq3_xxs );
		case GgmlType::IQ3_S: return k / QK_K * sizeof( block_iq3_s );
		case GgmlType::IQ1_S: return k / QK_K * sizeof( block_iq1_s );
		case GgmlType::IQ1_M: return k / QK_K * sizeof( block_iq1_m );
		case GgmlType::IQ4_XS: return k / QK_K * sizeof( block_iq4_xs );
		default: return 0;
	}
}

// The q8 activation rows of the integer path (gguf.slang: 10 words per 32 values), sized at prepare for the most rows and the widest k so a
// captured decode graph never sees it move
constexpr int kDotMaxRows = pod::MAX_BATCH, kDotMaxK = 65536;
constexpr size_t kDotBytes = static_cast<size_t>( kDotMaxRows ) * ( kDotMaxK / 32 ) * 40;
void* dot_scratch( backend::stream_t stream ) { return vki::scratch( 16, kDotBytes, stream ); }

// llama.cpp's mat-vec shaders (src/kernels/vulkan/ggml, one SPIR-V per type): output rows per 32-thread workgroup as llama.cpp picks them on
// NVIDIA before Turing (the IQ types 4, the K-quants 2, Q8_0 1, the other 32-value types 2), one pipeline per activation-row count. Up to 8
// rows, the decode and a speculative verify on the same arithmetic (a row's sum does not depend on the row count), so a verify accepts
// exactly what plain decoding emits; the int8 dot path below is faster at 8 rows but rounds apart from the 1-row product
constexpr int kMmvMaxRows = 8;
int mmv_rows( GgmlType t ) {
	switch ( t ) {
		case GgmlType::Q8_0: return 1;
		case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0: case GgmlType::Q5_1:
		case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K: case GgmlType::Q5_K: case GgmlType::Q6_K: return 2;
		default: return 4;
	}
}
backend::vk::Pipeline* mmv_pipeline( GgmlType type, int m ) {
	static std::mutex mu;
	static std::map<std::pair<int, int>, backend::vk::Pipeline*> cache[64];
	std::lock_guard<std::mutex> lk( mu );
	auto& c = cache[backend::get_device()];
	const auto key = std::make_pair( static_cast<int>( type ), m );
	auto it = c.find( key );
	if ( it != c.end() ) return it->second;
	backend::vk::Pipeline* p = nullptr;
	if ( backend::vk::bound_pipelines() ) {
		const char* dir = std::getenv( "ENGINE_SPV_DIR" );
		const std::string path = std::string( dir ? dir : SLANG_SPV_DIR ) + "/ggml_mmv_" + std::to_string( static_cast<int>( type ) ) + ".spv";
		const uint32_t spec[3] = { 32u, static_cast<uint32_t>( mmv_rows( type ) ), static_cast<uint32_t>( m ) };
		p = backend::vk::pipeline_create_bound( path.c_str(), sizeof( MmvParams ), 5, spec, 3 );   // A, B, D and the two (unused) fusion inputs
	}
	c[key] = p;
	return p;
}
// The permuted activation rows for the tiled out_proj, sized at prepare like the q8 rows
constexpr size_t kPermBytes = static_cast<size_t>( kMmvMaxRows ) * kDotMaxK * sizeof( backend::half_t );
void* perm_scratch( backend::stream_t stream ) { return vki::scratch( 17, kPermBytes, stream ); }

} // namespace

bool gguf_linear_supported( GgmlType type ) {
	switch ( type ) {
		case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0: case GgmlType::Q5_1: case GgmlType::Q8_0:
		case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K: case GgmlType::Q5_K: case GgmlType::Q6_K:
		case GgmlType::IQ2_XXS: case GgmlType::IQ2_XS: case GgmlType::IQ2_S: case GgmlType::IQ3_XXS: case GgmlType::IQ3_S:
		case GgmlType::IQ1_S: case GgmlType::IQ1_M: case GgmlType::IQ4_NL: case GgmlType::IQ4_XS: return true;
		default: return false;
	}
}

void LinearGguf::prepare() {
	(void) tables();
	if ( backend::vk::int_dot() ) (void) dot_scratch( nullptr );
	if ( backend::vk::bound_pipelines() ) (void) perm_scratch( nullptr );
}

void LinearGguf::dequant( const void* w, GgmlType type, int64_t elems, backend::half_t* out, backend::stream_t stream ) {
	ENGINE_ASSERT( gguf_linear_supported( type ), "LinearGguf::dequant (Vulkan): unsupported block type" );
	GgufDequantParams p{ vki::dp( w ), vki::dp( tables() ), vki::dp( out ), elems, static_cast<int>( type ) };
	const int64_t groups = ( elems + 255 ) / 256;
	vki::launch( "gguf_dequant", p, stream, static_cast<uint32_t>( std::min<int64_t>( groups, 65535 ) ), static_cast<uint32_t>( ( groups + 65534 ) / 65535 ) );
}

void LinearGguf::forward( const pod::TensorView& x, const void* w, GgmlType type, int n, int k, pod::TensorView& out, backend::stream_t stream, GgufColTiling tiling ) {
	ENGINE_ASSERT( gguf_linear_supported( type ), "LinearGguf::forward (Vulkan): unsupported block type" );
	int m = 1;
	for ( int d = 0; d < x.ndim - 1; ++d ) m *= x.shape[d];
	GgufParams p{ vki::dp( w ), vki::dp( tables() ), vki::dp( x.data ), vki::dp( out.data ), row_bytes( type, k ), static_cast<int>( type ), m, k, n,
		tiling.heads, tiling.dim };
	ENGINE_ASSERT( k % 8 == 0 && ( tiling.heads == 0 || tiling.dim % 8 == 0 ) && reinterpret_cast<uintptr_t>( x.data ) % 16 == 0,
		"LinearGguf::forward (Vulkan): the activations are read in runs of 8 halves" );
	// gguf.slang: up to 16 rows on a device with int8 dot products the q8 rows against the integer weights, a subgroup per column (4 per
	// workgroup); otherwise up to 8 rows the same with FP16 activations, more a workgroup per 64 x 64 tile
	backend::vk::Pipeline* mmv = m <= kMmvMaxRows ? mmv_pipeline( type, m ) : nullptr;
	if ( mmv != nullptr ) {
		const void* xs = x.data;
		if ( tiling.heads != 0 ) {
			ENGINE_ASSERT( k <= kDotMaxK, "LinearGguf::forward (Vulkan): k beyond the permuted-row scratch" );
			void* xp = perm_scratch( stream );
			GgufPermuteParams q{ vki::dp( x.data ), vki::dp( xp ), m, k, tiling.heads, tiling.dim };
			vki::launch( "gguf_permute_rows", q, stream, static_cast<uint32_t>( ( k + 255 ) / 256 ), static_cast<uint32_t>( m ) );
			xs = xp;
		}
		const uint32_t uk = static_cast<uint32_t>( k ), un = static_cast<uint32_t>( n );
		MmvParams mp{ uk, uk, uk, un, uk * un, uk, un, 0u, 0u, 1u, 1u, 1u, 1u };
		const void* bufs[5] = { w, xs, out.data, out.data, out.data };
		// workgroups of mmv_rows output rows; beyond the x limit the shader reads the rest from z
		const uint32_t groups = static_cast<uint32_t>( ( n + mmv_rows( type ) - 1 ) / mmv_rows( type ) );
		const uint32_t gx = std::min<uint32_t>( groups, 65535u ), gz = ( groups + gx - 1 ) / gx;
		backend::vk::dispatch_bound( stream, mmv, &mp, bufs, gx, 1, gz );
	}
	else if ( m <= kDotMaxRows && backend::vk::int_dot() ) {
		ENGINE_ASSERT( k <= kDotMaxK, "LinearGguf::forward (Vulkan): k beyond the q8 activation scratch" );
		void* xq = dot_scratch( stream );
		GgufQuantParams q{ vki::dp( x.data ), vki::dp( xq ), m, k, tiling.heads, tiling.dim };
		vki::launch( "gguf_quantize_q8", q, stream, static_cast<uint32_t>( ( k / 32 + 127 ) / 128 ), static_cast<uint32_t>( m ) );
		GgufDotParams d{ vki::dp( w ), vki::dp( tables() ), vki::dp( xq ), vki::dp( out.data ), row_bytes( type, k ), static_cast<int>( type ), m, k, n };
		static const char* const names[32] = { nullptr, nullptr, "gguf_gemv_dot_2", "gguf_gemv_dot_3", nullptr, nullptr, "gguf_gemv_dot_6", "gguf_gemv_dot_7",
			"gguf_gemv_dot_8", nullptr, "gguf_gemv_dot_10", "gguf_gemv_dot_11", "gguf_gemv_dot_12", "gguf_gemv_dot_13", "gguf_gemv_dot_14", nullptr,
			"gguf_gemv_dot_16", "gguf_gemv_dot_17", "gguf_gemv_dot_18", "gguf_gemv_dot_19", "gguf_gemv_dot_20", "gguf_gemv_dot_21", "gguf_gemv_dot_22",
			"gguf_gemv_dot_23", nullptr, nullptr, nullptr, nullptr, nullptr, "gguf_gemv_dot_29", nullptr, nullptr };
		const int ti = static_cast<int>( type );
		ENGINE_ASSERT( ti >= 0 && ti < 32 && names[ti] != nullptr, "LinearGguf::forward (Vulkan): no integer kernel for this block type" );
		vki::launch( names[ti], d, stream, static_cast<uint32_t>( ( n + 3 ) / 4 ), static_cast<uint32_t>( ( m + 7 ) / 8 ) );
	}
	else if ( m > 8 ) vki::launch( "gguf_gemm_tile", p, stream, static_cast<uint32_t>( ( n + 63 ) / 64 ), static_cast<uint32_t>( ( m + 63 ) / 64 ) );
	else vki::launch( "gguf_gemv_sg", p, stream, static_cast<uint32_t>( ( n + 3 ) / 4 ), static_cast<uint32_t>( ( m + 7 ) / 8 ) );
}

} // namespace engine::modules
