// CPU backend: DSA indexer entry points with a Slang kernel (prep, pool update). Projection, scoring and selection stay CUDA-only.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/dsa.h"

namespace engine::modules {

void DSA::prep(
	const backend::half_t* proj, const backend::half_t* ln_w, const backend::half_t* ln_b, backend::half_t* plane, backend::half_t* w_out,
	int layer, int max_seq, int rows, const pod::BatchRows* d_rows, size_t plane_slot_stride, backend::stream_t
) {
	PrepParams p{ const_cast<half*>( proj ), const_cast<half*>( ln_w ), const_cast<half*>( ln_b ), plane, w_out,
		const_cast<int*>( d_rows->slot ), const_cast<int*>( d_rows->pos ), layer, static_cast<int64_t>( plane_slot_stride ) };
	impl::cpu_dispatch( ::dsa_prep, &p, rows );
}

void DSA::pool_update(
	const backend::half_t* plane, backend::half_t* pool_plane, const backend::half_t* ape, int layer, int max_seq, int rows,
	const pod::BatchRows* d_rows, size_t plane_slot_stride, size_t pool_slot_stride, backend::stream_t
) {
	PoolParams p{ const_cast<half*>( plane ), pool_plane, const_cast<half*>( ape ), const_cast<int*>( d_rows->slot ), const_cast<int*>( d_rows->pos ),
		layer, max_seq, static_cast<int64_t>( plane_slot_stride ), static_cast<int64_t>( pool_slot_stride ) };
	impl::cpu_dispatch( ::dsa_pool_update, &p, rows );
}

} // namespace engine::modules
