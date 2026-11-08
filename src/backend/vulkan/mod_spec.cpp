// Vulkan backend: the head's top-k, log-sum-exp and the ranks' top-k merge on src/kernels/slang/spec_topk.slang (strong definitions over the
// stubs in src/backend/stubs/stubs_dflash.cpp)
#include "mod_common.h"
#include "engine/spec/dflash.h"

namespace engine::spec {

namespace {
namespace vki = ::engine::backend::impl;
struct TopkShardParams { uint64_t logits, values, indices; int rows, row_stride, columns, index_offset; };
struct LseShardParams { uint64_t logits, out_; int rows, row_stride, columns; };
struct TopkMergeParams { uint64_t part_vals, part_ix, out_vals, out_ix; int tp_size, rows, k; };
} // namespace

void dflash2_topk_shard( const backend::half_t* logits, int rows, int row_stride, int columns, int, float* values, int* indices, int index_offset, float*, int*,
	int*, backend::stream_t stream ) {
	if ( rows <= 0 || columns <= 0 ) return;
	TopkShardParams p{ vki::dp( logits ), vki::dp( values ), vki::dp( indices ), rows, row_stride, columns, index_offset };
	vki::launch( "topk_shard", p, stream, static_cast<uint32_t>( rows ) );
}

void logits_lse_shard( const backend::half_t* logits, int rows, int row_stride, int columns, float* out, backend::stream_t stream ) {
	if ( rows <= 0 || columns <= 0 ) return;
	LseShardParams p{ vki::dp( logits ), vki::dp( out ), rows, row_stride, columns };
	vki::launch( "lse_shard", p, stream, static_cast<uint32_t>( rows ) );
}

void dflash2_topk_merge( const float* part_vals, const int* part_ix, int tp_size, int rows, int k, float* out_vals, int* out_ix, int, backend::stream_t stream ) {
	if ( rows <= 0 || k <= 0 ) return;
	ENGINE_ASSERT( tp_size * k <= 128, "dflash2_topk_merge (Vulkan): tp_size * k beyond 128 candidates" );
	TopkMergeParams p{ vki::dp( part_vals ), vki::dp( part_ix ), vki::dp( out_vals ), vki::dp( out_ix ), tp_size, rows, k };
	vki::launch( "topk_merge", p, stream, static_cast<uint32_t>( ( rows + 63 ) / 64 ) );
}

} // namespace engine::spec
