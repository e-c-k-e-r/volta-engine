// CPU backend: SwiGLU through the Slang host kernels
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/swiglu.h"

namespace engine::modules {

void SwiGLU::forward( const pod::TensorView& gate, const pod::TensorView& up, pod::TensorView& out, backend::stream_t, float limit ) {
	const int n = static_cast<int>( gate.numel() );
	SwigluParams p{ const_cast<half*>( gate.ptr<const half>() ), const_cast<half*>( up.ptr<const half>() ), out.ptr<half>(), n, limit };
	impl::cpu_dispatch( ::swiglu, &p, ( n + 255 ) / 256 );
}

void SwiGLU::forward_fused( const pod::TensorView& gate_up, pod::TensorView& out, backend::stream_t, float limit ) {
	const int total = static_cast<int>( out.numel() );
	const int inter_dim = ( out.ndim >= 2 ) ? out.shape[1] : total;
	SwigluFusedParams p{ const_cast<half*>( gate_up.ptr<const half>() ), out.ptr<half>(), inter_dim, total, limit };
	impl::cpu_dispatch( ::swiglu_fused, &p, ( total + 255 ) / 256 );
}

} // namespace engine::modules
