// Vulkan backend: Gated DeltaNet entry points with a Slang kernel (gates, L2 norm, multi-row conv and recurrence, gated norm). The KDA forms and the single-row conv / recurrence stay weak stubs (stubs_gdn.cpp).
#include "mod_common.h"
#include "engine/modules/gdn.h"
#include <cmath>

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
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
	backend::stream_t stream,
	int rows
) {
	vki::GatesParams p{ vki::dp( norm_hidden.data ), vki::dp( in_proj_a.data ), vki::dp( in_proj_b.data ), vki::dp( A_log.data ), vki::dp( dt_bias.data ),
		vki::dp( gate_out.data ), vki::dp( beta_out.data ), num_v_heads, hidden_size, 1, 1 };
	vki::launch( "project_gates", p, stream, num_v_heads, rows );
}

void GDN::l2_norm_heads(
	pod::TensorView& qkv,
	int num_k_heads,
	int head_dim,
	backend::stream_t stream,
	bool scale_q,
	int rows,
	size_t row_stride
) {
	vki::L2Params p{ vki::dp( qkv.data ), num_k_heads, head_dim, scale_q ? 1 : 0, static_cast<int64_t>( row_stride ) };
	vki::launch( "l2_norm_heads", p, stream, num_k_heads * 2, rows );
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
	backend::stream_t stream
) {
	vki::ConvRowsParams p{ vki::dp( qkv.data ), vki::dp( history ), vki::dp( conv_weights.data ), vki::rows_slot( d_rows ), vki::dp( snap ), vki::dp( snap_on ),
		channels, rows, snap_rows, conv_weights.data != nullptr, static_cast<int64_t>( slot_stride ), static_cast<int64_t>( snap_stride ) };
	vki::launch( "causal_conv1d_silu_rows", p, stream, ( channels + 255 ) / 256 );
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
	backend::stream_t stream
) {
	const int tile = 16;
	vki::ConvTiledParams p{ vki::dp( src.data ), vki::dp( dst.data ), vki::dp( history ), vki::dp( conv_weights.data ), vki::rows_slot( d_rows ),
		channels, rows, tile, conv_weights.data != nullptr, static_cast<int64_t>( slot_stride ) };
	vki::launch( "causal_conv1d_silu_tiled", p, stream, ( channels + 127 ) / 128, ( rows + tile - 1 ) / tile );
	vki::launch( "causal_conv1d_history", p, stream, ( channels + 127 ) / 128 );
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
	backend::stream_t stream
) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_rows (Vulkan): the kernel is specialized for head_dim 128" );
	vki::RecurrentParams p{ vki::dp( gdn_state ), vki::dp( qkv.data ), vki::dp( gate.data ), vki::dp( beta.data ), vki::dp( attn_out.data ),
		vki::rows_slot( d_rows ), vki::dp( snap ), vki::dp( snap_on ), num_k_heads, num_v_heads, rows, snap_rows,
		1.0f / std::sqrt( static_cast<float>( head_dim ) ), static_cast<int64_t>( slot_stride ), static_cast<int64_t>( snap_stride ) };
	// Group = (head, 1, column block), as the CUDA launcher: 32 x 4 threads, 32 blocks along z
	vki::launch( "recurrent_step_rows", p, stream, num_v_heads, 1, 32 );
}

void GDN::gated_rmsnorm(
	pod::TensorView& attn_out,
	const pod::TensorView& z,
	const pod::TensorView& norm_weight,
	int num_v_heads,
	int head_dim,
	backend::stream_t stream,
	bool use_sigmoid,
	int rows
) {
	vki::GatedNormParams p{ vki::dp( attn_out.data ), vki::dp( z.data ), vki::dp( norm_weight.data ), num_v_heads, head_dim, use_sigmoid ? 1 : 0, z.data != nullptr, norm_weight.data != nullptr };
	vki::launch( "gated_rmsnorm", p, stream, num_v_heads, rows );
}

} // namespace engine::modules
