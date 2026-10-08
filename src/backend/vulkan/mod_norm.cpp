// Vulkan backend: RMSNorm and Elementwise through the SPIR-V kernels
#include "mod_common.h"
#include "engine/modules/norm.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

void RMSNorm::forward(
	const pod::TensorView& x,
	const pod::TensorView& weight,
	pod::TensorView& out,
	float eps,
	float constant_bias,
	backend::stream_t stream
) {
	const int hidden_size = x.shape[x.ndim - 1];
	int rows = 1;
	for ( int i = 0; i < x.ndim - 1; ++i ) rows *= x.shape[i];
	vki::NormParams p{ vki::dp( x.data ), vki::dp( out.data ), vki::dp( weight.data ), hidden_size, weight.data != nullptr, eps, constant_bias };
	vki::launch( "rms_norm", p, stream, rows );
}

void RMSNorm::forward_heads(
	pod::TensorView& qk,
	const pod::TensorView& weight,
	int num_heads,
	int head_dim,
	float eps,
	float constant_bias,
	backend::stream_t stream
) {
	const int rows = ( qk.ndim >= 2 && qk.shape[0] > 0 ) ? qk.shape[0] : 1;
	vki::HeadNormParams p{ vki::dp( qk.data ), vki::dp( weight.data ), num_heads, head_dim, weight.data != nullptr, eps, constant_bias };
	vki::launch( "rms_norm_heads", p, stream, num_heads, rows );
}

void Elementwise::add_residual( pod::TensorView& hidden, const pod::TensorView& residual, backend::stream_t stream ) {
	const int n = static_cast<int>( hidden.numel() );
	vki::AddParams p{ vki::dp( hidden.data ), vki::dp( residual.data ), n };
	vki::launch( "add_residual", p, stream, ( n + 255 ) / 256 );
}

void Elementwise::argmax( const pod::TensorView& logits, int* d_out_idx, int numel, backend::stream_t stream ) {
	vki::ArgmaxParams p{ vki::dp( logits.data ), vki::dp( d_out_idx ), 0, numel, 1, 0, numel, 0 };
	vki::launch( "batched_argmax", p, stream, 1 );
}

void Elementwise::batched_argmax( const pod::TensorView& logits, int* d_out_idx, int numel, int batch_size, backend::stream_t stream ) {
	vki::ArgmaxParams p{ vki::dp( logits.data ), vki::dp( d_out_idx ), 0, numel, batch_size, 0, logits.shape[1], 0 };
	vki::launch( "batched_argmax", p, stream, batch_size );
}

// The Slang kernel carries the range form (columns, offset, winning value) natively
void Elementwise::batched_argmax_range(
	const pod::TensorView& logits,
	int* d_out_idx,
	float* d_out_val,
	int columns,
	int batch_size,
	int index_offset,
	backend::stream_t stream
) {
	ENGINE_ASSERT( columns > 0 && columns <= logits.shape[1], "batched_argmax_range: columns outside the row" );
	vki::ArgmaxParams p{ vki::dp( logits.data ), vki::dp( d_out_idx ), vki::dp( d_out_val ), columns, batch_size, index_offset, logits.shape[1], d_out_val != nullptr };
	vki::launch( "batched_argmax", p, stream, batch_size );
}

} // namespace engine::modules
