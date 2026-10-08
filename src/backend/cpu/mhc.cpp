// CPU backend: mHC entry points with a Slang kernel (stream init, hyper-head mean). collapse / update stay CUDA-only for now.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/mhc.h"

namespace engine::modules {

void HyperConnection::init_streams( const pod::TensorView& in, pod::TensorView& streams, int hidden_size, backend::stream_t stream, int rows ) {
	init_streams( in.ptr<const half>(), streams.ptr<float>(), hidden_size, stream, rows );
}

void HyperConnection::init_streams( const backend::half_t* in, float* streams, int hidden_size, backend::stream_t, int rows ) {
	InitStreamsParams p{ const_cast<half*>( in ), streams, hidden_size };
	impl::cpu_dispatch( ::init_streams, &p, ( hidden_size + 255 ) / 256, rows );
}

void HyperHead::forward( const pod::TensorView& streams, pod::TensorView& final_hidden, int hidden_size, bool is_mean, backend::stream_t stream, int rows ) {
	if ( is_mean ) {
		mean( streams, final_hidden, hidden_size, stream, rows );
	}
}

void HyperHead::mean( const pod::TensorView& streams, pod::TensorView& final_hidden, int hidden_size, backend::stream_t stream, int rows ) {
	mean( streams.ptr<const float>(), final_hidden.ptr<half>(), hidden_size, stream, rows );
}

void HyperHead::mean( const float* streams, backend::half_t* final_hidden, int hidden_size, backend::stream_t, int rows ) {
	HeadMeanParams p{ const_cast<float*>( streams ), final_hidden, hidden_size };
	impl::cpu_dispatch( ::hyperhead_mean, &p, ( hidden_size + 255 ) / 256, rows );
}

} // namespace engine::modules
