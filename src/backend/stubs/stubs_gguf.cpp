// Weak stubs: the GGUF block kernels exist on the CUDA backend only.
#include "engine/modules/gguf.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) bool gguf_linear_supported( GgmlType ) { return false; }

__attribute__(( weak )) void LinearGguf::prepare() {}

__attribute__(( weak )) void LinearGguf::dequant( const void*, GgmlType, int64_t, backend::half_t*, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "LinearGguf::dequant" );
	__builtin_unreachable();
}

__attribute__(( weak )) void LinearGguf::scale_blocks( void*, GgmlType, size_t, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "LinearGguf::scale_blocks" );
	__builtin_unreachable();
}

// Backends without a repacked layout hold every weight in its file layout
__attribute__(( weak )) GgmlType LinearGguf::repack( void*, GgmlType type, int, int, backend::stream_t ) { return type; }

__attribute__(( weak )) void LinearGguf::scale_weight( void* w, GgmlType type, int n, int k, float factor, backend::stream_t stream ) {
	const GgmlBlock b = ggml_block( type );
	LinearGguf::scale_blocks( w, type, static_cast<size_t>( n ) * ( k / b.elems ) * b.bytes, factor, stream );
}

__attribute__(( weak )) void LinearGguf::forward( const pod::TensorView&, const void*, GgmlType, int, int, pod::TensorView&, backend::stream_t, GgufColTiling ) {
	ENGINE_UNSUPPORTED( "LinearGguf::forward" );
	__builtin_unreachable();
}

__attribute__(( weak )) bool LinearGguf::forward2( const pod::TensorView&, const void*, int, pod::TensorView&, const void*, int, pod::TensorView&, GgmlType, int,
	backend::stream_t ) {
	return false;
}

__attribute__(( weak )) size_t MoEGguf::scratch_bytes( int, int, int ) { return 0; }

__attribute__(( weak )) void MoEGguf::forward( const Table&, const backend::half_t*, int, const int*, const backend::half_t*, int, backend::half_t*, void*, uint32_t*, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "MoEGguf::forward" );
	__builtin_unreachable();
}

} // namespace engine::modules
