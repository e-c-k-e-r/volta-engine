#include "engine/models/model.h"
#include "engine/backend/runtime.h"
#include "engine/models/qwen3_5.h"
#include "engine/models/glm5_next.h"
#include "engine/models/qwen4_exp.h"
#include "engine/models/deepseek_v41.h"

namespace engine::models {

void alloc_draft_taps( pod::MemoryArena& arena, DraftHooks& hooks, DraftScratch& s, int max_seq, int hidden, int slots, bool per_slot ) {
	hooks.taps_cap = MAX( MIN( max_seq, 2048 ), 256 );
	const int tap_cols = 5 * hidden;
	int shape[2] = { ( per_slot ? slots : 1 ) * hooks.taps_cap, tap_cols };
	if ( per_slot ) hooks.taps_slot_stride = static_cast<size_t>( hooks.taps_cap ) * tap_cols;
	s.taps = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	backend::memset_sync( s.taps.data, 0, static_cast<size_t>( shape[0] ) * shape[1] * sizeof( backend::half_t ) );
	s.taps_owner = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	backend::memset_sync( s.taps_owner, 0xFF, sizeof( int ) );
	hooks.taps = s.taps.ptr<const backend::half_t>();
	hooks.taps_owner = s.taps_owner;
}

void alloc_draft_head( pod::MemoryArena& arena, DraftHooks& hooks, DraftScratch& s, int hidden, int lm_shard_cols, int tile, int shard_offset, int shard_real_cols ) {
	const int kM = kDraftM;
	int ds[2] = { kM, hidden };
	hooks.lm_in = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ds, 2 ).ptr<backend::half_t>();
	int dl[2] = { kM, lm_shard_cols };
	s.lm_logits = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, dl, 2 );
	const int splits = ( lm_shard_cols + tile - 1 ) / tile;
	const size_t per = static_cast<size_t>( kM ) * splits * kDraftTopK;
	s.cand_part_v = static_cast<float*>( engine::core::arena_alloc( arena, per * sizeof( float ) ) );
	s.cand_part_i = static_cast<int*>( engine::core::arena_alloc( arena, per * sizeof( int ) ) );
	s.cand_counters = static_cast<int*>( engine::core::arena_alloc( arena, kM * sizeof( int ) ) );
	hooks.cand_vals = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( kM ) * kDraftTopK * sizeof( float ) ) );
	hooks.cand_idx = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( kM ) * kDraftTopK * sizeof( int ) ) );
	backend::memset_sync( s.cand_counters, 0, kM * sizeof( int ) );
	hooks.shard_offset = shard_offset;
	hooks.shard_real_cols = shard_real_cols;
}

std::unique_ptr<ModelPipeline> create_model_pipeline( const std::string& model_type ) {
	if ( model_type == "qwen3_5" ) {
		return std::make_unique<Qwen3_5Pipeline>();
	}
	if ( model_type == "glm5_next" ) {
		return std::make_unique<Glm5NextPipeline>();
	}
	if ( model_type == "qwen4_exp" ) {
		return std::make_unique<Qwen4ExpPipeline>();
	}
	if ( model_type == "deepseek_v41" ) {
		return std::make_unique<DeepseekV41Pipeline>();
	}
	return nullptr;
}

bool is_known_model_type( const std::string& model_type ) {
	return model_type == "qwen3_5" || model_type == "glm5_next" || model_type == "qwen4_exp" || model_type == "deepseek_v41";
}

} // namespace engine::models
