// Vulkan backend: SwiGLU through the SPIR-V kernels
#include "mod_common.h"
#include "engine/modules/swiglu.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

void SwiGLU::forward( const pod::TensorView& gate, const pod::TensorView& up, pod::TensorView& out, backend::stream_t stream, float limit ) {
	const int n = static_cast<int>( gate.numel() );
	vki::SwigluParams p{ vki::dp( gate.data ), vki::dp( up.data ), vki::dp( out.data ), n, limit };
	vki::launch( "swiglu", p, stream, ( n + 255 ) / 256 );
}

void SwiGLU::forward_fused( const pod::TensorView& gate_up, pod::TensorView& out, backend::stream_t stream, float limit ) {
	const int total = static_cast<int>( out.numel() );
	const int inter_dim = ( out.ndim >= 2 ) ? out.shape[1] : total;
	vki::SwigluFusedParams p{ vki::dp( gate_up.data ), vki::dp( out.data ), inter_dim, total, limit };
	vki::launch( "swiglu_fused", p, stream, ( total + 255 ) / 256 );
}

} // namespace engine::modules
