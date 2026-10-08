// Vulkan backend: DSA indexer entry points with a Slang kernel (prep, pool update). Projection, scoring and selection stay weak stubs
#include "mod_common.h"
#include "engine/modules/dsa.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

void DSA::prep(
	const backend::half_t* proj, const backend::half_t* ln_w, const backend::half_t* ln_b, backend::half_t* plane, backend::half_t* w_out,
	int layer, int max_seq, int rows, const pod::BatchRows* d_rows, size_t plane_slot_stride, backend::stream_t stream
) {
	vki::PrepParams p{ vki::dp( proj ), vki::dp( ln_w ), vki::dp( ln_b ), vki::dp( plane ), vki::dp( w_out ),
		vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), layer, static_cast<int64_t>( plane_slot_stride ) };
	vki::launch( "dsa_prep", p, stream, rows );
}

void DSA::pool_update(
	const backend::half_t* plane, backend::half_t* pool_plane, const backend::half_t* ape, int layer, int max_seq, int rows,
	const pod::BatchRows* d_rows, size_t plane_slot_stride, size_t pool_slot_stride, backend::stream_t stream
) {
	vki::PoolParams p{ vki::dp( plane ), vki::dp( pool_plane ), vki::dp( ape ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
		layer, max_seq, static_cast<int64_t>( plane_slot_stride ), static_cast<int64_t>( pool_slot_stride ) };
	vki::launch( "dsa_pool_update", p, stream, rows );
}

} // namespace engine::modules
