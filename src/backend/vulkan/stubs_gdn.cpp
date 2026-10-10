// Vulkan backend: weak stubs for the gdn entry points that have no Vulkan implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/gdn.h.
#include "engine/modules/gdn.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void GDN::recurrent_step_kda( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, pod::TensorView& attn_out, int num_heads, int head_dim, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "GDN::recurrent_step_kda" );
	__builtin_unreachable();
}

} // namespace engine::modules
