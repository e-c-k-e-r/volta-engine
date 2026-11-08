// Vulkan backend: GGUF block weights on the Slang kernels of src/kernels/slang/gguf.slang (the weight decoded in the shader, fp32
// accumulation against the FP16 activations: no activation quantization). The IQ grids, sign table and IQ4 values come from llama.cpp's
// ggml-common.h (vendored in src/kernels/cpu/gguf/ggml) into one word buffer per device, in the order the shader's offsets name.
#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "../../kernels/cpu/gguf/ggml/ggml-common.h"

#include "mod_common.h"
#include "engine/backend/runtime.h"
#include "engine/modules/gguf.h"
#include <cstring>
#include <mutex>
#include <vector>

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;

struct GgufParams { uint64_t w, tables, x, out_; int64_t row_bytes; int type, m, k, n, heads, dim; };
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

void LinearGguf::prepare() { (void) tables(); }

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
	vki::launch( "gguf_gemv", p, stream, static_cast<uint32_t>( n ), static_cast<uint32_t>( ( m + 7 ) / 8 ) );   // gguf.slang: a workgroup per column and 8 rows
}

} // namespace engine::modules
