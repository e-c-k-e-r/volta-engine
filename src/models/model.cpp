#include "engine/models/model.h"
#include "engine/backend/runtime.h"
#include "engine/models/qwen3_5.h"
#include "engine/models/glm5_next.h"
#include "engine/models/qwen4_exp.h"
#include "engine/models/deepseek_v41.h"
#include "engine/loader/safetensors.h"
#include <cstdio>

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

bool read_draft_taps( const DraftHooks& hooks, int slot_id, int slots, int first_pos, int count, int hidden, backend::half_t* host_dst, backend::stream_t stream ) {
	const int cap = hooks.taps_cap;
	if ( hooks.taps == nullptr || slot_id < 0 || slot_id >= slots || first_pos < 0 || count <= 0 || count > cap ) return false;
	if ( hooks.taps_slot_stride == 0 && slot_id != 0 ) return false;   // one shared ring: only a single-slot server reads it unambiguously
	const size_t row = static_cast<size_t>( 5 ) * hidden;
	const backend::half_t* ring = hooks.taps + static_cast<size_t>( slot_id ) * hooks.taps_slot_stride;
	// Position p lives in ring row p % cap: at most two contiguous runs
	for ( int done = 0; done < count; ) {
		const int r = ( first_pos + done ) % cap;
		const int n = MIN( count - done, cap - r );
		backend::memcpy_async( host_dst + static_cast<size_t>( done ) * row, ring + static_cast<size_t>( r ) * row, static_cast<size_t>( n ) * row * sizeof( backend::half_t ), backend::Copy::D2H, stream );
		done += n;
	}
	backend::stream_sync( stream );
	return true;
}

bool alloc_expert_stacks( const engine::loader::SafeTensorsRepo& repo, pod::MemoryArena& arena, const char* layer_prefix, int first, int count, ExpertStacks& out,
	int slices ) {
	static const char* const kinds[3] = { "gate_proj", "up_proj", "down_proj" };
	char nm[512];
	for ( int p = 0; p < 3; ++p ) {
		std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.%s.weight", layer_prefix, first, kinds[p] );
		const pod::SafeTensorMeta* meta = repo.find( nm );
		if ( meta == nullptr || meta->dtype != pod::DType::GGUF ) return false;
		out.stride[p] = ( static_cast<size_t>( meta->size_bytes ) / slices + 255 ) / 256 * 256;
	}
	for ( int p = 0; p < 3; ++p ) {
		// 64 bytes of slack past the last expert: kernels that read blocks in aligned stretches may run a few bytes over
		out.base[p] = static_cast<uint8_t*>( engine::core::arena_alloc( arena, out.stride[p] * count + 64, 256 ) );
		ENGINE_ASSERT( out.base[p] != nullptr, "alloc_expert_stacks: arena full" );
	}
	return true;
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
