// CPU backend: Gated DeltaNet entry points with a Slang kernel (gates, L2 norm, multi-row conv and recurrence, gated norm).
// The KDA forms and the single-row conv / recurrence stay CUDA-only (weak stubs in stubs_gdn.cpp).
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/gdn.h"
#include <cmath>

namespace engine::modules {

namespace {

inline half* mut( const pod::TensorView& t ) { return const_cast<half*>( t.ptr<const half>() ); }

}

void GDN::project_gates(
	const pod::TensorView& norm_hidden,
	const pod::TensorView& in_proj_a,
	const pod::TensorView& in_proj_b,
	const pod::TensorView& A_log,
	const pod::TensorView& dt_bias,
	pod::TensorView& gate_out,
	pod::TensorView& beta_out,
	int num_v_heads,
	int hidden_size,
	backend::stream_t,
	int rows
) {
	GatesParams p{ mut( norm_hidden ), mut( in_proj_a ), mut( in_proj_b ), mut( A_log ), mut( dt_bias ),
		gate_out.ptr<float>(), beta_out.ptr<float>(), num_v_heads, hidden_size, 1, 1 };
	impl::cpu_dispatch( ::project_gates, &p, num_v_heads, rows );
}

void GDN::l2_norm_heads(
	pod::TensorView& qkv,
	int num_k_heads,
	int head_dim,
	backend::stream_t,
	bool scale_q,
	int rows,
	size_t row_stride
) {
	L2Params p{ qkv.ptr<half>(), num_k_heads, head_dim, scale_q ? 1 : 0, static_cast<int64_t>( row_stride ) };
	impl::cpu_dispatch( ::l2_norm_heads, &p, num_k_heads * 2, rows );
}

void GDN::causal_conv1d_silu_rows(
	pod::TensorView& qkv,
	float* history,
	const pod::TensorView& conv_weights,
	int channels,
	int rows,
	const pod::BatchRows* d_rows,
	size_t slot_stride,
	float* snap,
	size_t snap_stride,
	int snap_rows,
	const int* snap_on,
	backend::stream_t
) {
	ConvRowsParams p{ qkv.ptr<half>(), history, mut( conv_weights ), const_cast<int*>( d_rows->slot ), snap, const_cast<int*>( snap_on ),
		channels, rows, snap_rows, conv_weights.data != nullptr, static_cast<int64_t>( slot_stride ), static_cast<int64_t>( snap_stride ) };
	impl::cpu_dispatch( ::causal_conv1d_silu_rows, &p, ( channels + 255 ) / 256 );
}

void GDN::causal_conv1d_silu_tiled(
	const pod::TensorView& src,
	pod::TensorView& dst,
	float* history,
	const pod::TensorView& conv_weights,
	int channels,
	int rows,
	const pod::BatchRows* d_rows,
	size_t slot_stride,
	backend::stream_t
) {
	// Tile height only partitions the (bit-identical) row loop
	const int tile = 16;
	ConvTiledParams p{ mut( src ), dst.ptr<half>(), history, mut( conv_weights ), const_cast<int*>( d_rows->slot ),
		channels, rows, tile, conv_weights.data != nullptr, static_cast<int64_t>( slot_stride ) };
	impl::cpu_dispatch( ::causal_conv1d_silu_tiled, &p, ( channels + 127 ) / 128, ( rows + tile - 1 ) / tile );
	impl::cpu_dispatch( ::causal_conv1d_history, &p, ( channels + 127 ) / 128 );
}

void GDN::recurrent_step_rows(
	float* gdn_state,
	const pod::TensorView& qkv,
	const pod::TensorView& gate,
	const pod::TensorView& beta,
	pod::TensorView& attn_out,
	int num_k_heads,
	int num_v_heads,
	int head_dim,
	int rows,
	const pod::BatchRows* d_rows,
	size_t slot_stride,
	float* snap,
	size_t snap_stride,
	int snap_rows,
	const int* snap_on,
	backend::stream_t
) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_rows (CPU): the kernel is specialized for head_dim 128" );
	RecurrentParams p{ gdn_state, mut( qkv ), const_cast<float*>( gate.ptr<const float>() ), const_cast<float*>( beta.ptr<const float>() ),
		attn_out.ptr<half>(), const_cast<int*>( d_rows->slot ), snap, const_cast<int*>( snap_on ), num_k_heads, num_v_heads, rows, snap_rows,
		1.0f / std::sqrt( static_cast<float>( head_dim ) ), static_cast<int64_t>( slot_stride ), static_cast<int64_t>( snap_stride ) };
	// Host entry: one group per (head, column)
	impl::cpu_dispatch( ::recurrent_step_rows, &p, num_v_heads, 1, head_dim );
}

void GDN::gated_rmsnorm(
	pod::TensorView& attn_out,
	const pod::TensorView& z,
	const pod::TensorView& norm_weight,
	int num_v_heads,
	int head_dim,
	backend::stream_t,
	bool use_sigmoid,
	int rows
) {
	GatedNormParams p{ attn_out.ptr<half>(), mut( z ), mut( norm_weight ), num_v_heads, head_dim, use_sigmoid ? 1 : 0, z.data != nullptr, norm_weight.data != nullptr };
	impl::cpu_dispatch( ::gated_rmsnorm, &p, num_v_heads, rows );
}

} // namespace engine::modules
