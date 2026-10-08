// CPU backend: mHC collapse and update composed from the portable Slang kernels the way the CUDA driver does
// (hc_mix_partials + hc_mix_finalize for the collapse, hc_apply for the stream update). collapse_norm stays CUDA-only.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/mhc.h"
#include "engine/core/batch.h"
#include <vector>

namespace engine::modules {

void HyperConnection::collapse(
	const pod::TensorView& streams, const pod::TensorView& fn, const pod::TensorView& base, const pod::TensorView& scale,
	pod::TensorView& collapsed_out, pod::TensorView& post_out, pod::TensorView& comb_out, pod::TensorView& inv_rms_buf,
	pod::TensorView& mix_logits_buf, pod::TensorView& pre_buf, int hidden_size, int sinkhorn_iters, float eps, backend::stream_t stream, int rows
) {
	collapse( streams.ptr<const float>(), fn.ptr<const float>(), base.ptr<const float>(), scale.ptr<const float>(), collapsed_out.ptr<half>(),
		post_out.ptr<float>(), comb_out.ptr<float>(), inv_rms_buf.ptr<float>(), mix_logits_buf.ptr<float>(), pre_buf.ptr<float>(),
		hidden_size, sinkhorn_iters, eps, stream, rows );
}

void HyperConnection::collapse(
	const float* streams, const float* fn, const float* base, const float* scale, backend::half_t* collapsed_out, float* post_out,
	float* comb_out, float* inv_rms_buf, float* mix_logits_buf, float* pre_buf, int hidden_size, int sinkhorn_iters, float eps,
	backend::stream_t, int rows
) {
	ENGINE_ASSERT( rows >= 1 && rows <= pod::MAX_ROWS, "HyperConnection::collapse: rows out of range" );
	const int row_len = 4 * hidden_size;
	const int chunk_cols = ( row_len + 63 ) / 64;
	const int num_chunks = ( row_len + chunk_cols - 1 ) / chunk_cols;
	std::vector<float> partials( static_cast<size_t>( rows ) * num_chunks * 25 );
	PartialsParams pa{ const_cast<float*>( streams ), const_cast<float*>( fn ), partials.data(), row_len, chunk_cols, num_chunks };
	impl::cpu_dispatch( ::hc_mix_partials, &pa, num_chunks, rows );
	// One group per row covers every output column
	FinalizeParams pf{ const_cast<float*>( streams ), partials.data(), const_cast<float*>( base ), const_cast<float*>( scale ), post_out, comb_out,
		collapsed_out, inv_rms_buf, mix_logits_buf, pre_buf, hidden_size, num_chunks, hidden_size, sinkhorn_iters, 1e-5f, eps };
	impl::cpu_dispatch( ::hc_mix_finalize, &pf, 1, rows );
}

void HyperConnection::update( pod::TensorView& streams, const pod::TensorView& sublayer_out, const pod::TensorView& post, const pod::TensorView& comb, int hidden_size, backend::stream_t stream, int rows ) {
	update( streams.ptr<float>(), streams.ptr<const float>(), sublayer_out.ptr<const half>(), post.ptr<const float>(), comb.ptr<const float>(), hidden_size, stream, rows );
}

void HyperConnection::update( pod::TensorView& streams_out, const pod::TensorView& streams_in, const pod::TensorView& sublayer_out, const pod::TensorView& post, const pod::TensorView& comb, int hidden_size, backend::stream_t stream, int rows ) {
	update( streams_out.ptr<float>(), streams_in.ptr<const float>(), sublayer_out.ptr<const half>(), post.ptr<const float>(), comb.ptr<const float>(), hidden_size, stream, rows );
}

void HyperConnection::update( float* streams, const backend::half_t* sublayer_out, const float* post, const float* comb, int hidden_size, backend::stream_t stream, int rows ) {
	update( streams, streams, sublayer_out, post, comb, hidden_size, stream, rows );
}

void HyperConnection::update( float* streams_out, const float* streams_in, const backend::half_t* sublayer_out, const float* post, const float* comb, int hidden_size, backend::stream_t, int rows ) {
	// hc_apply reads a column's four stream values before writing them, so in-place is safe
	ApplyParams p{ const_cast<float*>( streams_in ), streams_out, const_cast<half*>( sublayer_out ), const_cast<float*>( post ), const_cast<float*>( comb ), hidden_size, 1 };
	impl::cpu_dispatch( ::hc_apply, &p, ( hidden_size + 255 ) / 256, rows );
}

} // namespace engine::modules
