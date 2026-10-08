// Vulkan backend: mHC entry points with a Slang kernel (stream init, hyper-head mean). collapse / update stay weak stubs
#include "mod_common.h"
#include "engine/modules/mhc.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

void HyperConnection::init_streams( const pod::TensorView& in, pod::TensorView& streams, int hidden_size, backend::stream_t stream, int rows ) {
	init_streams( in.ptr<const backend::half_t>(), streams.ptr<float>(), hidden_size, stream, rows );
}

void HyperConnection::init_streams( const backend::half_t* in, float* streams, int hidden_size, backend::stream_t stream, int rows ) {
	vki::InitStreamsParams p{ vki::dp( in ), vki::dp( streams ), hidden_size };
	vki::launch( "init_streams", p, stream, ( hidden_size + 255 ) / 256, rows );
}

void HyperHead::forward( const pod::TensorView& streams, pod::TensorView& final_hidden, int hidden_size, bool is_mean, backend::stream_t stream, int rows ) {
	if ( is_mean ) {
		mean( streams, final_hidden, hidden_size, stream, rows );
	}
}

void HyperHead::mean( const pod::TensorView& streams, pod::TensorView& final_hidden, int hidden_size, backend::stream_t stream, int rows ) {
	mean( streams.ptr<const float>(), final_hidden.ptr<backend::half_t>(), hidden_size, stream, rows );
}

void HyperHead::mean( const float* streams, backend::half_t* final_hidden, int hidden_size, backend::stream_t stream, int rows ) {
	vki::HeadMeanParams p{ vki::dp( streams ), vki::dp( final_hidden ), hidden_size };
	vki::launch( "hyperhead_mean", p, stream, ( hidden_size + 255 ) / 256, rows );
}

} // namespace engine::modules
