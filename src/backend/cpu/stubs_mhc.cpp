// CPU backend: weak stubs for the mhc entry points that have no CPU implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/mhc.h.
#include "engine/modules/mhc.h"
#include "engine/modules/gated_residual.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void HyperConnection::collapse( const pod::TensorView& streams, const pod::TensorView& fn, const pod::TensorView& base, const pod::TensorView& scale, pod::TensorView& collapsed_out, pod::TensorView& post_out, pod::TensorView& comb_out, pod::TensorView& inv_rms_buf, pod::TensorView& mix_logits_buf, pod::TensorView& pre_buf, int hidden_size, int sinkhorn_iters, float eps, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "HyperConnection::collapse" );
	__builtin_unreachable();
}

__attribute__(( weak )) void HyperConnection::collapse( const float* streams, const float* fn, const float* base, const float* scale, backend::half_t* collapsed_out, float* post_out, float* comb_out, float* inv_rms_buf, float* mix_logits_buf, float* pre_buf, int hidden_size, int sinkhorn_iters, float eps, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "HyperConnection::collapse" );
	__builtin_unreachable();
}

__attribute__(( weak )) bool HyperConnection::collapse_norm( const pod::TensorView& streams, const pod::TensorView& fn, const pod::TensorView& base, const pod::TensorView& scale, const pod::TensorView& norm_weight, pod::TensorView& norm_out, pod::TensorView& post_out, pod::TensorView& comb_out, pod::TensorView& inv_rms_buf, pod::TensorView& mix_logits_buf, pod::TensorView& pre_buf, int hidden_size, int sinkhorn_iters, float eps, float norm_eps, float norm_bias, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "HyperConnection::collapse_norm" );
	__builtin_unreachable();
}

__attribute__(( weak )) void HyperConnection::update( pod::TensorView& streams, const pod::TensorView& sublayer_out, const pod::TensorView& post, const pod::TensorView& comb, int hidden_size, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "HyperConnection::update" );
	__builtin_unreachable();
}

__attribute__(( weak )) void HyperConnection::update( pod::TensorView& streams_out, const pod::TensorView& streams_in, const pod::TensorView& sublayer_out, const pod::TensorView& post, const pod::TensorView& comb, int hidden_size, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "HyperConnection::update" );
	__builtin_unreachable();
}

__attribute__(( weak )) void HyperConnection::update( float* streams, const backend::half_t* sublayer_out, const float* post, const float* comb, int hidden_size, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "HyperConnection::update" );
	__builtin_unreachable();
}

__attribute__(( weak )) void HyperConnection::update( float* streams_out, const float* streams_in, const backend::half_t* sublayer_out, const float* post, const float* comb, int hidden_size, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "HyperConnection::update" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GatedResidual::init_streams( const backend::half_t*, float*, int, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "GatedResidual::init_streams" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GatedResidual::mix( const float*, const backend::half_t*, const backend::half_t*, const backend::half_t*, const backend::half_t*,
	backend::half_t*, float*, float*, int, int, int, int, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "GatedResidual::mix" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GatedResidual::add_sigmoid_gated( backend::half_t*, const backend::half_t*, const backend::half_t*, const backend::half_t*, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "GatedResidual::add_sigmoid_gated" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GatedResidual::apply( float*, const float*, const backend::half_t*, int, int, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "GatedResidual::apply" );
	__builtin_unreachable();
}

} // namespace engine::modules
