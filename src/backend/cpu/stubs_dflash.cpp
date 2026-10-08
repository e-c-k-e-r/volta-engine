// CPU backend: weak stubs for the DFlash speculator and its top-k kernels (src/spec/*.cu, CUDA/SM70 only).
// A strong definition in another translation unit overrides these.
#include "engine/spec/dflash.h"
#include "engine/backend/tiers.h"

namespace engine::spec {

__attribute__(( weak )) void dflash2_topk_shard(
	const backend::half_t* logits, int rows, int row_stride, int columns, int k, float* values, int* indices, int index_offset,
	float* part_v, int* part_ix, int* counters, backend::stream_t stream
) {
	ENGINE_UNSUPPORTED( "dflash2_topk_shard" );
	__builtin_unreachable();
}

__attribute__(( weak )) void dflash2_topk_merge(
	const float* part_vals, const int* part_ix, int tp_size, int rows, int k, float* out_vals, int* out_ix, int part_rows, backend::stream_t stream
) {
	ENGINE_UNSUPPORTED( "dflash2_topk_merge" );
	__builtin_unreachable();
}

__attribute__(( weak )) DFlashSpeculator::~DFlashSpeculator() {
	// Nothing was ever initialized: init() is unsupported on this backend
}

__attribute__(( weak )) bool DFlashSpeculator::init( const pod::DFlashConfig& config, int device, pod::MemoryArena& arena, backend::stream_t stream, int rank, int tp_size ) {
	ENGINE_UNSUPPORTED( "DFlashSpeculator::init" );
	__builtin_unreachable();
}

__attribute__(( weak )) bool DFlashSpeculator::load_weights( const char* model_dir, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DFlashSpeculator::load_weights" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DFlashSpeculator::draft_backbone( const pod::TensorView& embed_table, int anchor_token, const backend::half_t* tapped_states, int block_pos0, int taps_cap, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DFlashSpeculator::draft_backbone" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DFlashSpeculator::project_gate( int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DFlashSpeculator::project_gate" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DFlashSpeculator::set_walk_anchor( int anchor, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DFlashSpeculator::set_walk_anchor" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DFlashSpeculator::draft_select( const float* vals, const int* ix, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DFlashSpeculator::draft_select" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DFlashSpeculator::destroy() {
	// Nothing to release: init() is unsupported on this backend
}

} // namespace engine::spec
