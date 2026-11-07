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

__attribute__(( weak )) void LinearGguf::forward( const pod::TensorView&, const void*, GgmlType, int, int, pod::TensorView&, backend::stream_t, GgufColTiling ) {
	ENGINE_UNSUPPORTED( "LinearGguf::forward" );
	__builtin_unreachable();
}

} // namespace engine::modules
