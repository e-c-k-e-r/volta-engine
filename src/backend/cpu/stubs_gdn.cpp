// CPU backend: weak stubs for the gdn entry points that have no CPU implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/gdn.h.
#include "engine/modules/gdn.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void GDN::causal_conv1d_silu( pod::TensorView& qkv, pod::TensorView& conv_history, const pod::TensorView& conv_weights, int channels, backend::stream_t stream, const pod::SlotRef& slot, float* snap, const int* snap_on ) {
	ENGINE_UNSUPPORTED( "GDN::causal_conv1d_silu" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GDN::project_kda_gates( const pod::TensorView& norm_hidden, const pod::TensorView& b_proj, const pod::TensorView& f_a_proj, const pod::TensorView& f_b_proj, const pod::TensorView& g_a_proj, const pod::TensorView& g_b_proj, const pod::TensorView& A_log, const pod::TensorView& dt_bias, pod::TensorView& beta_out, pod::TensorView& g_decay_out, pod::TensorView& z_out, pod::TensorView& f_a_buf, pod::TensorView& g_a_buf, int num_heads, int head_dim, int hidden_size, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "GDN::project_kda_gates" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GDN::recurrent_step( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& gate, const pod::TensorView& beta, pod::TensorView& attn_out, int num_k_heads, int num_v_heads, int head_dim, backend::stream_t stream, const pod::SlotRef& slot, float* snap, const int* snap_on ) {
	ENGINE_UNSUPPORTED( "GDN::recurrent_step" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GDN::recurrent_step_kda( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, pod::TensorView& attn_out, int num_heads, int head_dim, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "GDN::recurrent_step_kda" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GDN::recurrent_step_kda_fused( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, const pod::TensorView& z, const pod::TensorView& o_norm, pod::TensorView& attn_out, int num_heads, int head_dim, float eps, backend::stream_t stream, const pod::SlotRef& slot ) {
	ENGINE_UNSUPPORTED( "GDN::recurrent_step_kda_fused" );
	__builtin_unreachable();
}

__attribute__(( weak )) void GDN::recurrent_step_kda_rows( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, const pod::TensorView& z, const pod::TensorView& o_norm, pod::TensorView& attn_out, int num_heads, int head_dim, float eps, int rows, const pod::BatchRows* d_rows, size_t slot_stride, float* snap, size_t snap_stride, int snap_rows, const int* snap_on, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "GDN::recurrent_step_kda_rows" );
	__builtin_unreachable();
}

} // namespace engine::modules
