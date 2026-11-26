#include "engine/models/model.h"
#include "engine/backend/runtime.h"
#include "engine/models/qwen3_5.h"
#include "engine/models/gemma4.h"
#include "engine/models/glm5_next.h"
#include "engine/models/qwen4_exp.h"
#include "engine/models/deepseek_v41.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/moe.h"
#include "engine/runtime/options.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace engine::models {

void alloc_draft_taps( pod::MemoryArena& arena, DraftHooks& hooks, DraftScratch& s, int max_seq, int hidden, int slots, bool per_slot, int rank, int tp, int taps_n ) {
	hooks.taps_cap = MAX( MIN( max_seq, 2048 ), 256 );
	hooks.taps_n = taps_n;
#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1
	constexpr bool cuda = true;   // the drafters are CUDA only
#else
	constexpr bool cuda = false;
#endif
	const auto& opt = runtime::options();
	const bool shard = cuda && tp > 1 && opt.draft_tp && opt.dflash_hessian_dir.empty() && ( taps_n * hidden ) % ( tp * 128 ) == 0;
	const int tap_cols = shard ? taps_n * hidden / tp : taps_n * hidden;
	hooks.tap_slice = { hidden, taps_n, shard ? rank * tap_cols : 0, tap_cols };
	int shape[2] = { ( per_slot ? slots : 1 ) * hooks.taps_cap, tap_cols };
	if ( per_slot ) hooks.taps_slot_stride = static_cast<size_t>( hooks.taps_cap ) * tap_cols;
	s.taps = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	backend::memset_sync( s.taps.data, 0, static_cast<size_t>( shape[0] ) * shape[1] * sizeof( backend::half_t ) );
	s.taps_owner = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	backend::memset_sync( s.taps_owner, 0xFF, sizeof( int ) );
	hooks.taps = s.taps.ptr<const backend::half_t>();
	hooks.taps_owner = s.taps_owner;
}

bool read_draft_taps( const DraftHooks& hooks, int slot_id, int slots, int first_pos, int count, backend::half_t* host_dst, backend::stream_t stream ) {
	const int cap = hooks.taps_cap;
	if ( hooks.taps == nullptr || slot_id < 0 || slot_id >= slots || first_pos < 0 || count <= 0 || count > cap ) return false;
	if ( hooks.taps_slot_stride == 0 && slot_id != 0 ) return false;   // one shared ring: only a single-slot server reads it unambiguously
	const engine::spec::TapSlice& sl = hooks.tap_slice;
	const size_t row = static_cast<size_t>( sl.taps_n ) * sl.hidden, kept = static_cast<size_t>( sl.cols );
	const backend::half_t* ring = hooks.taps + static_cast<size_t>( slot_id ) * hooks.taps_slot_stride;
	std::vector<backend::half_t> part( sl.whole() ? 0 : static_cast<size_t>( count ) * kept );
	backend::half_t* dst = sl.whole() ? host_dst : part.data();
	// Position p lives in ring row p % cap: at most two contiguous runs
	for ( int done = 0; done < count; ) {
		const int r = ( first_pos + done ) % cap;
		const int n = MIN( count - done, cap - r );
		backend::memcpy_async( dst + static_cast<size_t>( done ) * kept, ring + static_cast<size_t>( r ) * kept, static_cast<size_t>( n ) * kept * sizeof( backend::half_t ), backend::Copy::D2H, stream );
		done += n;
	}
	backend::stream_sync( stream );
	if ( !sl.whole() ) {
		for ( int i = 0; i < count; ++i ) std::copy_n( part.data() + i * kept, kept, host_dst + i * row + sl.col0 );
	}
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
	if ( model_type == "gemma4" ) {
		return std::make_unique<Gemma4Pipeline>();
	}
	return nullptr;
}

size_t device_stack_bytes( const std::string& model_type, const std::string& model_dir ) {
#if ENGINE_BACKEND_CUDA
	// DeepSeek's mixed-width EXL3 packs take the run-time-width expert kernels; a GGUF conversion's experts never do
	if ( model_type == "deepseek_v41" ) {
		std::error_code ec;
		for ( const auto& f : std::filesystem::directory_iterator( model_dir, ec ) ) if ( f.path().extension() == ".gguf" ) return 0;
		return ext::exl3_moe_mixed_stack_bytes();
	}
#else
	(void) model_type;
	(void) model_dir;
#endif
	return 0;
}

bool is_known_model_type( const std::string& model_type ) {
	return model_type == "qwen3_5" || model_type == "glm5_next" || model_type == "qwen4_exp" || model_type == "deepseek_v41" || model_type == "gemma4";
}

bool check_model_config( const std::string& model_type, const char* model_dir, bool log ) {
	std::unique_ptr<ModelPipeline> p = create_model_pipeline( model_type );
	if ( !p ) return false;
	p->set_model_dir( model_dir );
	return p->check_config( log );
}

} // namespace engine::models
