// Vulkan backend: weak stubs for the mhc entry points that have no Vulkan implementation (collapse_norm).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/mhc.h.
#include "engine/modules/mhc.h"
#include "engine/modules/gated_residual.h"
#include "engine/modules/ple.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) bool HyperConnection::collapse_norm( const pod::TensorView& streams, const pod::TensorView& fn, const pod::TensorView& base, const pod::TensorView& scale, const pod::TensorView& norm_weight, pod::TensorView& norm_out, pod::TensorView& post_out, pod::TensorView& comb_out, pod::TensorView& inv_rms_buf, pod::TensorView& mix_logits_buf, pod::TensorView& pre_buf, int hidden_size, int sinkhorn_iters, float eps, float norm_eps, float norm_bias, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "HyperConnection::collapse_norm" );
	__builtin_unreachable();
}

} // namespace engine::modules
