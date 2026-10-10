// Vulkan backend: mHC entry points on Slang kernels: stream init, hyper-head mean, collapse (hc_mix_partials + hc_mix_finalize) and the stream update
// (hc_apply), composed as the CUDA driver does. collapse_norm stays a weak stub
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

void HyperConnection::collapse(
	const pod::TensorView& streams, const pod::TensorView& fn, const pod::TensorView& base, const pod::TensorView& scale,
	pod::TensorView& collapsed_out, pod::TensorView& post_out, pod::TensorView& comb_out, pod::TensorView& inv_rms_buf,
	pod::TensorView& mix_logits_buf, pod::TensorView& pre_buf, int hidden_size, int sinkhorn_iters, float eps, backend::stream_t stream, int rows
) {
	collapse( streams.ptr<const float>(), fn.ptr<const float>(), base.ptr<const float>(), scale.ptr<const float>(), collapsed_out.ptr<backend::half_t>(),
		post_out.ptr<float>(), comb_out.ptr<float>(), inv_rms_buf.ptr<float>(), mix_logits_buf.ptr<float>(), pre_buf.ptr<float>(),
		hidden_size, sinkhorn_iters, eps, stream, rows );
}

void HyperConnection::collapse(
	const float* streams, const float* fn, const float* base, const float* scale, backend::half_t* collapsed_out, float* post_out,
	float* comb_out, float* inv_rms_buf, float* mix_logits_buf, float* pre_buf, int hidden_size, int sinkhorn_iters, float eps,
	backend::stream_t stream, int rows
) {
	ENGINE_ASSERT( rows >= 1 && rows <= pod::MAX_ROWS, "HyperConnection::collapse: rows out of range" );
	const int row_len = 4 * hidden_size;
	const int chunk_cols = ( row_len + 63 ) / 64;
	const int num_chunks = ( row_len + chunk_cols - 1 ) / chunk_cols;
	float* partials = static_cast<float*>( vki::scratch( 14, static_cast<size_t>( rows ) * num_chunks * 25 * sizeof( float ), stream ) );
	vki::HcPartialsParams pa{ vki::dp( streams ), vki::dp( fn ), vki::dp( partials ), row_len, chunk_cols, num_chunks };
	vki::launch( "hc_mix_partials", pa, stream, num_chunks, rows );
	// One group per row covers every output column
	vki::HcFinalizeParams pf{ vki::dp( streams ), vki::dp( partials ), vki::dp( base ), vki::dp( scale ), vki::dp( post_out ), vki::dp( comb_out ), vki::dp( collapsed_out ),
		vki::dp( inv_rms_buf ), vki::dp( mix_logits_buf ), vki::dp( pre_buf ), hidden_size, num_chunks, hidden_size, sinkhorn_iters, 1e-5f, eps };
	vki::launch( "hc_mix_finalize", pf, stream, 1, rows );
}

// The fused collapse + RMSNorm is a CUDA launch; false (nothing launched) makes the caller run collapse + RMSNorm::forward, which is the same arithmetic
bool HyperConnection::collapse_norm( const pod::TensorView&, const pod::TensorView&, const pod::TensorView&, const pod::TensorView&, const pod::TensorView&, pod::TensorView&,
	pod::TensorView&, pod::TensorView&, pod::TensorView&, pod::TensorView&, pod::TensorView&, int, int, float, float, float, backend::stream_t, int ) {
	return false;
}

void HyperConnection::update( pod::TensorView& streams, const pod::TensorView& sublayer_out, const pod::TensorView& post, const pod::TensorView& comb, int hidden_size, backend::stream_t stream, int rows ) {
	update( streams.ptr<float>(), streams.ptr<const float>(), sublayer_out.ptr<const backend::half_t>(), post.ptr<const float>(), comb.ptr<const float>(), hidden_size, stream, rows );
}

void HyperConnection::update( pod::TensorView& streams_out, const pod::TensorView& streams_in, const pod::TensorView& sublayer_out, const pod::TensorView& post, const pod::TensorView& comb, int hidden_size, backend::stream_t stream, int rows ) {
	update( streams_out.ptr<float>(), streams_in.ptr<const float>(), sublayer_out.ptr<const backend::half_t>(), post.ptr<const float>(), comb.ptr<const float>(), hidden_size, stream, rows );
}

void HyperConnection::update( float* streams, const backend::half_t* sublayer_out, const float* post, const float* comb, int hidden_size, backend::stream_t stream, int rows ) {
	update( streams, streams, sublayer_out, post, comb, hidden_size, stream, rows );
}

void HyperConnection::update( float* streams_out, const float* streams_in, const backend::half_t* sublayer_out, const float* post, const float* comb, int hidden_size, backend::stream_t stream, int rows ) {
	// hc_apply reads a column's four stream values before writing them, so in-place is safe
	vki::HcApplyParams p{ vki::dp( streams_in ), vki::dp( streams_out ), vki::dp( sublayer_out ), vki::dp( post ), vki::dp( comb ), hidden_size, 1 };
	vki::launch( "hc_apply", p, stream, ( hidden_size + 255 ) / 256, rows );
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
