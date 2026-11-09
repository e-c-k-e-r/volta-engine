// Vulkan backend: RoPE through the SPIR-V kernel
#include "mod_common.h"
#include "engine/modules/rope.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

void RoPE::forward(
	pod::TensorView& q,
	pod::TensorView& k,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int rot_dim,
	const pod::BatchRows* d_rows,
	int count,
	float freq_base,
	backend::stream_t stream,
	bool interleaved,
	int rot_pairs
) {
	ENGINE_ASSERT( rot_pairs >= 0 && rot_pairs <= rot_dim / 2, "RoPE::forward: rot_pairs beyond the rotary pairs" );
	vki::RopeParams p{ vki::dp( q.data ), vki::dp( k.data ), vki::rows_pos( d_rows ), num_q_heads, num_kv_heads, head_dim, rot_dim, interleaved ? 1 : 0, freq_base, rot_pairs };
	vki::launch( "rope", p, stream, count );
}

} // namespace engine::modules
