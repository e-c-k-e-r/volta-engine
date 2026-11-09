// CPU backend: RMSNorm and Elementwise through the Slang host kernels
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/norm.h"

namespace engine::modules {

void RMSNorm::forward(
	const pod::TensorView& x,
	const pod::TensorView& weight,
	pod::TensorView& out,
	float eps,
	float constant_bias,
	backend::stream_t
) {
	const int hidden_size = x.shape[x.ndim - 1];
	int rows = 1;
	for ( int i = 0; i < x.ndim - 1; ++i ) rows *= x.shape[i];
	if ( x.dtype == pod::DType::FP32 ) {
		// an FP32 residual stream (Gemma 4)
		NormF32Params p{ const_cast<float*>( x.ptr<const float>() ), out.ptr<half>(), const_cast<half*>( weight.ptr<const half>() ), hidden_size, weight.data != nullptr, eps, constant_bias };
		impl::cpu_dispatch( ::rms_norm_f32, &p, rows );
		return;
	}
	NormParams p{ const_cast<half*>( x.ptr<const half>() ), out.ptr<half>(), const_cast<half*>( weight.ptr<const half>() ), hidden_size, weight.data != nullptr, eps, constant_bias };
	impl::cpu_dispatch( ::rms_norm, &p, rows );
}

void RMSNorm::forward_heads(
	pod::TensorView& qk,
	const pod::TensorView& weight,
	int num_heads,
	int head_dim,
	float eps,
	float constant_bias,
	backend::stream_t
) {
	const int rows = ( qk.ndim >= 2 && qk.shape[0] > 0 ) ? qk.shape[0] : 1;
	HeadNormParams p{ qk.ptr<half>(), const_cast<half*>( weight.ptr<const half>() ), num_heads, head_dim, weight.data != nullptr, eps, constant_bias };
	impl::cpu_dispatch( ::rms_norm_heads, &p, num_heads, rows );
}

void Elementwise::add_residual( pod::TensorView& hidden, const pod::TensorView& residual, backend::stream_t ) {
	const int n = static_cast<int>( hidden.numel() );
	AddParams p{ hidden.ptr<half>(), const_cast<half*>( residual.ptr<const half>() ), n };
	impl::cpu_dispatch( ::add_residual, &p, ( n + 255 ) / 256 );
}

void Elementwise::add_normed( pod::TensorView& resid, const pod::TensorView& y, const pod::TensorView& weight, float eps, float constant_bias,
	float scale, backend::stream_t ) {
	ENGINE_ASSERT( resid.dtype == pod::DType::FP32 && y.dtype == pod::DType::FP16, "Elementwise::add_normed: FP32 residual, FP16 input" );
	const int hidden_size = y.shape[y.ndim - 1];
	AddNormedParams p{ resid.ptr<float>(), const_cast<half*>( y.ptr<const half>() ), const_cast<half*>( weight.ptr<const half>() ), hidden_size, weight.data != nullptr,
		eps, constant_bias, scale };
	impl::cpu_dispatch( ::add_normed, &p, static_cast<uint32_t>( y.numel() / hidden_size ) );
}

void Elementwise::scale_to_f32( const pod::TensorView& x, pod::TensorView& out, float scale, backend::stream_t ) {
	const int n = static_cast<int>( x.numel() );
	ScaleF32Params p{ const_cast<half*>( x.ptr<const half>() ), out.ptr<float>(), n, scale };
	impl::cpu_dispatch( ::scale_to_f32, &p, ( n + 255 ) / 256 );
}

void Elementwise::softcap( pod::TensorView& x, float cap, backend::stream_t ) {
	const int n = static_cast<int>( x.numel() );
	SoftcapParams p{ x.ptr<half>(), n, cap };
	impl::cpu_dispatch( ::softcap, &p, ( n + 255 ) / 256 );
}

void Elementwise::argmax( const pod::TensorView& logits, int* d_out_idx, int numel, backend::stream_t ) {
	ArgmaxParams p{ const_cast<half*>( logits.ptr<const half>() ), d_out_idx, nullptr, numel, 1, 0, numel, 0 };
	impl::cpu_dispatch( ::batched_argmax, &p, 1 );
}

void Elementwise::batched_argmax( const pod::TensorView& logits, int* d_out_idx, int numel, int batch_size, backend::stream_t ) {
	ArgmaxParams p{ const_cast<half*>( logits.ptr<const half>() ), d_out_idx, nullptr, numel, batch_size, 0, logits.shape[1], 0 };
	impl::cpu_dispatch( ::batched_argmax, &p, batch_size );
}

// The Slang kernel carries the range form (columns, offset, winning value) natively
void Elementwise::batched_argmax_range(
	const pod::TensorView& logits,
	int* d_out_idx,
	float* d_out_val,
	int columns,
	int batch_size,
	int index_offset,
	backend::stream_t
) {
	ENGINE_ASSERT( columns > 0 && columns <= logits.shape[1], "batched_argmax_range: columns outside the row" );
	ArgmaxParams p{ const_cast<half*>( logits.ptr<const half>() ), d_out_idx, d_out_val, columns, batch_size, index_offset, logits.shape[1], d_out_val != nullptr };
	impl::cpu_dispatch( ::batched_argmax, &p, batch_size );
}

} // namespace engine::modules
