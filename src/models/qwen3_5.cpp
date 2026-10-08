#include "engine/models/qwen3_5.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include "engine/loader/safetensors.h"
#include "engine/spec/dflash.h"
#include "engine/spec/tap_ring.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>

namespace engine::models {

Qwen3_5Pipeline::Qwen3_5Pipeline() = default;

Qwen3_5Pipeline::~Qwen3_5Pipeline() {
	destroy();
}

bool Qwen3_5Pipeline::init(
	const Qwen3_5Config& config,
	int rank,
	pod::MemoryArena& arena,
	engine::comm::CollectiveRing* ring
) {
	config_ = config;
	rank_ = rank;
	arena_ = &arena;
	ring_ = ring;
	weights_loaded_ = false;

	backend::set_device( rank_ );

	const int tp_size = ( ring_ != nullptr ) ? requested_tp_ : 1;
	ENGINE_ASSERT( tp_size == 1 || tp_size == 4, "Qwen3_5Pipeline: the grouped weight shards are laid out for 4 ranks" );
	tp_ranks_ = tp_size;
	tp_size_ = tp_size;
	q_heads_ = config_.full_num_q_heads / tp_size;
	kv_heads_ = config_.full_num_kv_heads / tp_size;
	lin_k_heads_ = config_.linear_num_k_heads / tp_size;
	lin_v_heads_ = config_.linear_num_v_heads / tp_size;
	lin_channels_ = config_.linear_channels / tp_size;
	inter_size_ = config_.intermediate_size / tp_size;
	const int full_q_heads = q_heads_;
	const int full_kv_heads = kv_heads_;
	const int lin_v_heads = lin_v_heads_;
	const int lin_channels = lin_channels_;
	const int inter_size = inter_size_;
	const int R = pod::MAX_ROWS;   // prefill chunk capacity; decode graphs use up to MAX_BATCH
	const int S = config_.max_slots;

	// 1. Recurrent GDN state per slot: [S, 48, v_heads, 128, 128] FP32 + one-slot checkpoint
	state_.slot_gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_v_heads * 128 * 128;
	size_t gdn_floats = static_cast<size_t>( S ) * state_.slot_gdn_floats;
	state_.gdn_states = static_cast<float*>( engine::core::arena_alloc( arena, gdn_floats * sizeof( float ) ) );
	backend::memset_sync( state_.gdn_states, 0, gdn_floats * sizeof( float ) );
	state_.gdn_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, state_.slot_gdn_floats * sizeof( float ) ) );

	// 2. Causal conv1d history per slot: [S, 48, channels, 3] FP32 + one-slot checkpoint
	state_.slot_conv_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_channels * 3;
	size_t conv_hist_floats = static_cast<size_t>( S ) * state_.slot_conv_floats;
	state_.conv_history = static_cast<float*>( engine::core::arena_alloc( arena, conv_hist_floats * sizeof( float ) ) );
	backend::memset_sync( state_.conv_history, 0, conv_hist_floats * sizeof( float ) );
	state_.conv_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, state_.slot_conv_floats * sizeof( float ) ) );

	// 2b. Per-row snapshots for the speculative verify forward (kDraftM rows, one speculating slot).
	state_.gdn_snap = static_cast<float*>( engine::core::arena_alloc( arena, kDraftM * state_.slot_gdn_floats * sizeof( float ) ) );
	state_.conv_snap = static_cast<float*>( engine::core::arena_alloc( arena, kDraftM * state_.slot_conv_floats * sizeof( float ) ) );
	state_.d_snap_on = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	backend::memset_sync( state_.d_snap_on, 0, sizeof( int ) );

	// 3. Full-attention KV cache pool: [S, 16, max_seq, kv_heads, 256] FP16
	kv_pool_.init( rank_, arena, S, config_.max_seq, config_.num_attn_layers, full_kv_heads, config_.full_head_dim );

	// 4. Residual stream & normed activations (MAX_BATCH rows)
	int h_shape[2] = { R, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	backend::memset_sync( hidden_states_.data, 0, hidden_states_.size_bytes() );

	// Tapped hidden states for DFlash2: [R, 5, hidden]
	const size_t tap_bytes = static_cast<size_t>( R ) * 5 * config_.hidden_size * sizeof( backend::half_t );
	tapped_states_ = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, tap_bytes ) );
	backend::memset_sync( tapped_states_, 0, tap_bytes );

	// Drafter context taps. tapped_states_ only survives the last forward (MAX_BATCH rows), but the
	// drafter's context stage needs every position in the window, so keep a position-indexed ring of
	// the same per-position [5, hidden] tap set.
	//
	// The ring is per position, not per slot (a per-slot ring would cost ~840 MiB per rank), so only
	// one slot may publish into it: rows from every other slot are skipped on device. The owner is a
	// device scalar so the captured forward graph stays valid when it changes. -1 publishes nothing.
	draft_hooks_.taps_cap = MAX( MIN( config_.max_seq, 2048 ), 256 );
	int wtap_shape[2] = { draft_hooks_.taps_cap, 5 * config_.hidden_size };
	draft_taps_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, wtap_shape, 2 );
	backend::memset_sync( draft_taps_.data, 0, wtap_shape[0] * wtap_shape[1] * sizeof( backend::half_t ) );
	d_taps_owner_ = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	backend::memset_sync( d_taps_owner_, 0xFF, sizeof( int ) );
	draft_hooks_.taps = draft_taps_.ptr<const backend::half_t>();
	draft_hooks_.taps_owner = d_taps_owner_;
	draft_hooks_.embedding = &embed_tokens_;
	d_token_ids_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
	d_rows_ = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );

	// 5. Full attention buffers
	int q_gate_shape[2] = { R, full_q_heads * config_.full_head_dim * 2 };
	q_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_gate_shape, 2 );

	int q_shape[2] = { R, full_q_heads * config_.full_head_dim };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	gate_attn_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );

	int kv_shape[2] = { R, full_kv_heads * config_.full_head_dim };
	k_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
	v_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );

	// 6. Linear attention (GDN) buffers
	int qkv_shape[2] = { R, lin_channels };
	qkv_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );
	{
		int conv_shape[2] = { R, lin_channels };
		conv_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, conv_shape, 2 );   // prefill conv output
	}

	int z_shape[2] = { R, lin_v_heads * config_.linear_head_dim };
	z_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	gdn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );

	int gdn_param_shape[1] = { R * lin_v_heads };
	gdn_gate_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdn_param_shape, 1 );
	gdn_beta_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdn_param_shape, 1 );

	// 7. MLP buffers
	int mlp_shape[2] = { R, inter_size };
	gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	mlp_inter_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	mlp_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	// 8. Logits & argmax. The vocabulary is sharded the same way the rest of the head is: each rank
	// projects only its own columns, which takes the 5120 x 248320 LM head off rank 0's critical path.
	// The shard is padded to a multiple of 256 columns (one extra 128-column Hadamard block for
	// 62080 -> 62208) so the dp4a int8 GEMV engages; pad columns are excluded from the argmax.
	// EXL3_LM_SHARD=0 restores the single-rank full-vocabulary head for A/B measurement.
	shard_lm_ = tp_size > 1;
	if ( runtime::options().lm_shard >= 0 ) shard_lm_ = ( runtime::options().lm_shard != 0 );
	tp_size_ = shard_lm_ ? tp_size : 1;
	lm_shard_real_cols_ = config_.vocab_size / tp_size_;
	int pad_cols = 0;
	if ( tp_size_ > 1 ) {
		pad_cols = ( 256 - ( lm_shard_real_cols_ % 256 ) ) % 256;
	}
	lm_shard_cols_ = lm_shard_real_cols_ + pad_cols;
	{
		int logit_shape[2] = { R, lm_shard_cols_ };
		logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, logit_shape, 2 );
		d_argmax_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
		d_argmax_val_ = static_cast<float*>( engine::core::arena_alloc( arena, R * sizeof( float ) ) );
		h_argmax_idx_ = static_cast<int*>( engine::core::host_alloc_pinned( R * sizeof( int ) ) );
		std::memset( h_argmax_idx_, 0, R * sizeof( int ) );
		if ( tp_size_ > 1 ) {
			// Top-k sampling scratch over the target logits (up to R rows).
			const int splits = ( lm_shard_cols_ + kTopkTile - 1 ) / kTopkTile;
			const size_t per = static_cast<size_t>( R ) * splits * 16;
			samp_part_v_ = static_cast<float*>( engine::core::arena_alloc( arena, per * sizeof( float ) ) );
			samp_part_i_ = static_cast<int*>( engine::core::arena_alloc( arena, per * sizeof( int ) ) );
			samp_counters_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
			d_samp_vals_ = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * 16 * sizeof( float ) ) );
			d_samp_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * 16 * sizeof( int ) ) );
			backend::memset_sync( samp_counters_, 0, R * sizeof( int ) );
		}
		if ( rank_ == 0 && tp_size_ > 1 ) {
			// Only the DFlash2 drafter needs the whole distribution; sampling reduces per shard.
			int full_shape[2] = { pod::MAX_BATCH, config_.vocab_size };   // the drafter / tests read at most a decode block
			full_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, full_shape, 2 );
		}

		// Drafter head scratch, on this rank's device: the drafter's [kDraftM, hidden] rows land in
		// lm_in, the head shard writes [kDraftM, shard] logits, and the local top-k comes out in
		// cand_vals / cand_idx. Each row's candidate set is the top-k over the union of every rank's
		// shard, so every rank needs all of it.
		{
			const int kM = kDraftM;
			int ds[2] = { kM, config_.hidden_size };
			draft_hooks_.lm_in = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ds, 2 ).ptr<backend::half_t>();
			int dl[2] = { kM, lm_shard_cols_ };
			draft_lm_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, dl, 2 );
			const int splits = ( lm_shard_cols_ + kTopkTile - 1 ) / kTopkTile;
			const size_t per = static_cast<size_t>( kM ) * splits * kDraftTopK;
			draft_cand_part_v_ = static_cast<float*>( engine::core::arena_alloc( arena, per * sizeof( float ) ) );
			draft_cand_part_i_ = static_cast<int*>( engine::core::arena_alloc( arena, per * sizeof( int ) ) );
			draft_cand_counters_ = static_cast<int*>( engine::core::arena_alloc( arena, kM * sizeof( int ) ) );
			draft_hooks_.cand_vals = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( kM ) * kDraftTopK * sizeof( float ) ) );
			draft_hooks_.cand_idx = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( kM ) * kDraftTopK * sizeof( int ) ) );
			backend::memset_sync( draft_cand_counters_, 0, kM * sizeof( int ) );
			draft_hooks_.shard_offset = rank_ * lm_shard_real_cols_;
			draft_hooks_.shard_real_cols = lm_shard_real_cols_;
		}
	}

	for ( int L = 0; L < config_.num_layers; ++L ) {
		layers_[L].is_full_attention = ( ( L + 1 ) % 4 == 0 );
	}

#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );
#endif

	metadata_.model_name = "qwen3_5";
	metadata_.hidden_size = config_.hidden_size;
	metadata_.vocab_size = config_.vocab_size;
	metadata_.num_layers = config_.num_layers;
	metadata_.num_heads = config_.full_num_q_heads;
	metadata_.num_kv_heads = config_.full_num_kv_heads;
	metadata_.head_dim = config_.full_head_dim;
	metadata_.max_seq_len = config_.max_seq;
	metadata_.tp_size = tp_size;
	metadata_.max_slots = S;
	metadata_.max_rows = R;
	metadata_.eos_ids[0] = 248044;
	metadata_.eos_ids[1] = 248046;

	return true;
}

bool Qwen3_5Pipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	Qwen3_5Config config;
	if ( runtime::options().max_seq > 0 ) config.max_seq = runtime::options().max_seq;  // context window (KV pool size)
	if ( runtime::options().max_slots > 0 ) config.max_slots = runtime::options().max_slots;
	return init( config, device_id, arena, ring );
}


void Qwen3_5Pipeline::reset_state( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen3_5Pipeline::reset_state: slot out of range" );
	backend::set_device( rank_ );
	kv_pool_.reset_slot( slot_id );
	backend::memset_sync( state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, 0, state_.slot_gdn_floats * sizeof( float ) );
	backend::memset_sync( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, 0, state_.slot_conv_floats * sizeof( float ) );
}

bool Qwen3_5Pipeline::load_weights( const char* model_dir, backend::stream_t stream ) {
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) {
		std::printf( "    [loader] FAILED to open model directory %s\n", model_dir );
		return false;
	}

	// 1. Embed tokens
	embed_tokens_ = repo.load( "model.language_model.embed_tokens.weight", *arena_, stream, pod::DType::BF16 );

	// 2. Final norm
	final_norm_ = repo.load( "model.language_model.norm.weight", *arena_, stream, pod::DType::FP16 );

	// NVFP4 checkpoint (compressed-tensors): projections are W4A16 through the skinny NVFP4 kernels, the LM head is a dense
	// BF16 matrix served as FP16.
	const bool fp4 = repo.find( "model.language_model.layers.0.mlp.gate_proj.weight_packed" ) != nullptr;
	if ( fp4 ) std::printf( "    [RANK %d] NVFP4 checkpoint\n", rank_ );

	// 3. LM Head, sharded across the vocabulary dimension
	if ( fp4 ) {
		ENGINE_ASSERT( tp_size_ > 1 || tp_ranks_ == 1, "Qwen3_5Pipeline: the NVFP4 checkpoint needs the vocabulary-parallel head" );
		lm_head_ = repo.load_dense_rows_padded( "lm_head.weight", *arena_, rank_ * lm_shard_real_cols_, lm_shard_real_cols_,
			lm_shard_cols_ - lm_shard_real_cols_, stream );
	}
	else if ( tp_size_ > 1 ) {
		lm_head_ = repo.load_linear_slice_padded( "lm_head", *arena_, 5, rank_, tp_size_, lm_shard_cols_ - lm_shard_real_cols_, stream );
	}
	else if ( rank_ == 0 ) {
		lm_head_ = repo.load_linear( "lm_head", *arena_, 5, stream );
	}

	const int tp_size = tp_ranks_;

	// 4. Per-layer weights across 64 layers
	char name_buf[256];
	// Column-parallel (output split, dim 1) and row-parallel (input split, dim 0) projection loads for either format
	auto load_col = [&]( const char* name, int bits ) {
		if ( fp4 ) return repo.load_nvfp4_slice( name, *arena_, true, rank_, tp_size, stream );
		return ( tp_size > 1 ) ? repo.load_linear_slice( name, *arena_, bits, 1, rank_, tp_size, stream ) : repo.load_linear( name, *arena_, bits, stream );
	};
	auto load_row = [&]( const char* name, int bits ) {
		if ( fp4 ) return repo.load_nvfp4_slice( name, *arena_, false, rank_, tp_size, stream );
		return ( tp_size > 1 ) ? repo.load_linear_slice( name, *arena_, bits, 0, rank_, tp_size, stream ) : repo.load_linear( name, *arena_, bits, stream );
	};
	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.is_full_attention = ( ( L + 1 ) % 4 == 0 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.input_layernorm.weight", L );
		lw.input_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.post_attention_layernorm.weight", L );
		lw.post_attn_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		if ( lw.is_full_attention ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_proj", L );
			lw.q_proj = load_col( name_buf, 4 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_norm.weight", L );
			lw.q_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.k_proj", L );
			lw.k_proj = load_col( name_buf, 5 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.k_norm.weight", L );
			lw.k_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.v_proj", L );
			lw.v_proj = load_col( name_buf, 5 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj", L );
			lw.o_proj = load_row( name_buf, 4 );
		}
		else {
			static constexpr int QWEN_QKV_BLOCKS[3] = { 128, 128, 384 };
			static constexpr int QWEN_QKV_RANK_BLOCKS[3] = { 32, 32, 96 };
			static constexpr int QWEN_CONV_CHANNELS[3] = { 2048, 2048, 6144 };
			static constexpr int QWEN_CONV_RANK_CHANNELS[3] = { 512, 512, 1536 };

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_qkv", L );
			if ( fp4 ) {
				// Rows are [q | k | v] groups; each rank takes its share of every group
				int n_begin[3], n_count[3], start = 0;
				for ( int g = 0; g < 3; ++g ) {
					n_count[g] = QWEN_QKV_RANK_BLOCKS[g] * 16;
					n_begin[g] = start + rank_ * n_count[g];
					start += QWEN_QKV_BLOCKS[g] * 16;
				}
				lw.in_proj_qkv = repo.load_nvfp4( name_buf, *arena_, n_begin, n_count, 3, 0, 5120, stream );
			}
			else {
				lw.in_proj_qkv = ( tp_size > 1 )
					? repo.load_grouped_trellis_slice( name_buf, *arena_, QWEN_QKV_BLOCKS, QWEN_QKV_RANK_BLOCKS, 3, rank_, stream )
					: repo.load_linear( name_buf, *arena_, 5, stream );
			}

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.conv1d.weight", L );
			lw.conv1d_w = ( tp_size > 1 )
				? repo.load_grouped_slice( name_buf, *arena_, QWEN_CONV_CHANNELS, QWEN_CONV_RANK_CHANNELS, 3, rank_, 0, stream )
				: repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.A_log", L );
			lw.A_log = ( tp_size > 1 ) ? repo.load_slice( name_buf, *arena_, 0, rank_, tp_size, stream, pod::DType::FP16 )
			                            : repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.dt_bias", L );
			lw.dt_bias = ( tp_size > 1 ) ? repo.load_slice( name_buf, *arena_, 0, rank_, tp_size, stream, pod::DType::FP16 )
			                              : repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", L );
			if ( fp4 ) {
				name_buf[std::strlen( name_buf ) - 7] = 0;  // drop ".weight"
				lw.in_proj_a = repo.load_nvfp4_dense( name_buf, *arena_, true, rank_, tp_size, stream );
			}
			else lw.in_proj_a = ( tp_size > 1 ) ? repo.load_slice( name_buf, *arena_, 0, rank_, tp_size, stream, pod::DType::FP16 )
			                                : repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", L );
			if ( fp4 ) {
				name_buf[std::strlen( name_buf ) - 7] = 0;
				lw.in_proj_b = repo.load_nvfp4_dense( name_buf, *arena_, true, rank_, tp_size, stream );
			}
			else lw.in_proj_b = ( tp_size > 1 ) ? repo.load_slice( name_buf, *arena_, 0, rank_, tp_size, stream, pod::DType::FP16 )
			                                : repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_z", L );
			lw.in_proj_z = load_col( name_buf, 5 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.norm.weight", L );
			lw.linear_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.out_proj", L );
			lw.out_proj = load_row( name_buf, 5 );
		}

		// MLP
		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj", L );
		lw.gate_proj = load_col( name_buf, 3 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj", L );
		lw.up_proj = load_col( name_buf, 3 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj", L );
		lw.down_proj = load_row( name_buf, 3 );
	}

	if ( stream ) {
		backend::stream_sync( stream );
	}
	repo.close();
	weights_loaded_ = true;
	return true;
}

void Qwen3_5Pipeline::set_rows( int m ) {
	hidden_states_.shape[0] = m;
	norm_hidden_.shape[0] = m;
	q_gate_buf_.shape[0] = m;
	q_buf_.shape[0] = m;
	gate_attn_buf_.shape[0] = m;
	attn_out_.shape[0] = m;
	k_buf_.shape[0] = m;
	v_buf_.shape[0] = m;
	qkv_buf_.shape[0] = m;
	z_buf_.shape[0] = m;
	gdn_out_.shape[0] = m;
	gate_buf_.shape[0] = m;
	up_buf_.shape[0] = m;
	mlp_inter_.shape[0] = m;
	mlp_out_.shape[0] = m;
	if ( logits_.data ) logits_.shape[0] = m;
}

void Qwen3_5Pipeline::embed_tokens( const int* token_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_ROWS, "Qwen3_5Pipeline::embed_tokens: count out of range" );
	backend::set_device( rank_ );
	// Pageable H2D: the source is staged before return, so the caller's array may be reused immediately
	backend::memcpy_async( d_token_ids_, token_ids, count * sizeof( int ), backend::Copy::H2D, stream );
	set_rows( count );
	engine::modules::Embedding::forward( embed_tokens_, d_token_ids_, count, hidden_states_, stream );
}

void Qwen3_5Pipeline::sample_tokens( int* out_preds, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( tp_size_ == 1, "Qwen3_5Pipeline::sample_tokens: vocabulary is sharded, use sample_local" );
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Qwen3_5Pipeline::sample_tokens: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax( logits_, d_argmax_idx_, lm_shard_cols_, count, stream );
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_preds, h_argmax_idx_, count * sizeof( int ) );
}

bool Qwen3_5Pipeline::sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream ) {
	if ( tp_size_ <= 1 ) return false;   // single-rank head: the caller samples on rank 0
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Qwen3_5Pipeline::sample_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	// Scan only this rank's real columns: the pad block decodes to 0.0 and would beat a real negative
	// logit. index_offset turns the shard-local winner into a global token id.
	engine::modules::Elementwise::batched_argmax_range(
		logits_,
		d_argmax_idx_,
		d_argmax_val_,
		lm_shard_real_cols_,
		count,
		rank_ * lm_shard_real_cols_,
		stream
	);
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_tokens, h_argmax_idx_, count * sizeof( int ) );
	if ( out_vals != nullptr ) {
		std::vector<float> h_vals( count );
		backend::memcpy_async( h_vals.data(), d_argmax_val_, count * sizeof( float ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
		std::memcpy( out_vals, h_vals.data(), count * sizeof( float ) );
	}
	return true;
}

bool Qwen3_5Pipeline::topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream ) {
	if ( tp_size_ <= 1 ) return false;
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Qwen3_5Pipeline::topk_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	// Real columns only (pad columns decode to 0.0), global ids via the shard offset.
	engine::spec::dflash2_topk_shard(
		logits_.ptr<const backend::half_t>(), count, lm_shard_cols_, lm_shard_real_cols_, 16,
		d_samp_vals_, d_samp_idx_, rank_ * lm_shard_real_cols_,
		samp_part_v_, samp_part_i_, samp_counters_, stream );
	backend::memcpy_async( out_vals, d_samp_vals_, static_cast<size_t>( count ) * 16 * sizeof( float ), backend::Copy::D2H, stream );
	backend::memcpy_async( out_ids, d_samp_idx_, static_cast<size_t>( count ) * 16 * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

const pod::TensorView& Qwen3_5Pipeline::gather_logits(
	int rows,
	const backend::half_t* const* peer_shard_ptrs,
	int peer_shard_cols,
	backend::stream_t stream
) {
	ENGINE_ASSERT( tp_size_ == 1 || rank_ == 0, "Qwen3_5Pipeline::gather_logits: only rank 0 materializes the full vocabulary" );
	if ( tp_size_ == 1 ) {
		return logits_;
	}
	ENGINE_ASSERT( rows >= 1 && rows <= last_rows_, "Qwen3_5Pipeline::gather_logits: rows exceeds last forward rows" );
	ENGINE_ASSERT( peer_shard_cols == lm_shard_cols_, "Qwen3_5Pipeline::gather_logits: shard width mismatch" );

	full_logits_.shape[0] = rows;
	// Row r of the gathered buffer is [shard 0 | shard 1 | ...], so each row needs one copy per rank:
	// the source rows are lm_shard_cols_ apart (pad block included), the destination rows
	// lm_shard_real_cols_ apart.
	const size_t shard_elems = static_cast<size_t>( lm_shard_real_cols_ );
	for ( int d = 0; d < tp_ranks_; ++d ) {
		const backend::half_t* src_base = ( d == rank_ ) ? logits_.ptr<const backend::half_t>() : peer_shard_ptrs[d];
		ENGINE_ASSERT( src_base != nullptr, "Qwen3_5Pipeline::gather_logits: missing peer shard pointer" );
		for ( int r = 0; r < rows; ++r ) {
			const backend::half_t* src = src_base + static_cast<size_t>( r ) * lm_shard_cols_;
			backend::half_t* dst = full_logits_.ptr<backend::half_t>() + ( static_cast<size_t>( r ) * tp_ranks_ + d ) * shard_elems;
			backend::peer_copy_async( dst, rank_, src, d, shard_elems * sizeof( backend::half_t ), stream );
		}
	}
	return full_logits_;
}

// Runs this rank's head shard over the drafter's hidden rows and leaves the local top-k in the hooks'
// cand buffers. Not a target forward: the rows are the drafter's, with no slot or position.
void Qwen3_5Pipeline::draft_head_topk( int rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "Qwen3_5Pipeline::draft_head_topk: weights not loaded" );
	ENGINE_ASSERT( rows >= 1 && rows <= kDraftM, "Qwen3_5Pipeline::draft_head_topk: rows out of range" );
	backend::set_device( rank_ );
	const int H = config_.hidden_size;
	// All kDraftM rows, whatever `rows` is: the head's fast path only covers m in {1, 2, 4, 8}, and any
	// other m falls through to the cooperative GEMM (far slower). The caller consumes the first rows.
	pod::TensorView in = pod::make_tensor_2d( draft_hooks_.lm_in, pod::DType::FP16, kDraftM, H );
	engine::modules::Linear::forward( in, lm_head_, draft_lm_logits_, stream );
	// Columns are the slice's real width: pad columns hold the next rank's tokens, so scanning them
	// would put the same token in two ranks' candidate sets and let the merge emit it twice.
	engine::spec::dflash2_topk_shard(
		draft_lm_logits_.ptr<const backend::half_t>(), kDraftM, lm_shard_cols_, lm_shard_real_cols_, kDraftTopK,
		draft_hooks_.cand_vals, draft_hooks_.cand_idx, draft_hooks_.shard_offset,
		draft_cand_part_v_, draft_cand_part_i_, draft_cand_counters_, stream );
}

void Qwen3_5Pipeline::forward( const pod::BatchRows& rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "Qwen3_5Pipeline::forward: weights not loaded" );
	ENGINE_ASSERT( rows.count >= 1 && rows.count <= pod::MAX_ROWS, "Qwen3_5Pipeline::forward: row count out of range" );
	for ( int r = 0; r < rows.count; ++r ) {
		ENGINE_ASSERT( rows.slot[r] >= 0 && rows.slot[r] < config_.max_slots, "Qwen3_5Pipeline::forward: slot out of range" );
		ENGINE_ASSERT( rows.pos[r] >= 0 && rows.pos[r] < config_.max_seq, "Qwen3_5Pipeline::forward: position out of range" );
	}
	backend::set_device( rank_ );
	const int m = rows.count;
	last_slot_ = rows.slot[0];
	key_hint_ = rows.pos[m - 1] + 1;
	// Logits selection applies to eager prefill chunks only (decode graphs are keyed on m alone).
	lm_mode_ = ( m > pod::MAX_BATCH ) ? rows.logits_mode : pod::BatchRows::LOGITS_ALL;
	last_rows_ = ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) ? 0 : ( lm_mode_ == pod::BatchRows::LOGITS_LAST ? 1 : m );
	set_rows( m );

	// Row descriptor goes to the device; the layer stack only reads slots/positions from there,
	// so its launch sequence depends on m alone and is replayed from a per-m CUDA graph.
	// Pageable source: the copy is staged before return, so `rows` may be a temporary.
	backend::memcpy_async( d_rows_, &rows, sizeof( pod::BatchRows ), backend::Copy::H2D, stream );
	// Tensor-core attention shares K/V tiles across the rows of one slot (verify blocks, single-row decode); rows of
	// distinct slots get a block each. The graph body differs, so the two shapes get separate graphs.
	attn_tc_ = true;
	for ( int r = 1; r < m; ++r ) attn_tc_ = attn_tc_ && rows.slot[r] == rows.slot[0];
	graphs_.run( m, stream, [&]( backend::stream_t s ) { forward_body( m, s ); }, attn_tc_ ? 1 : 0 );
}

void Qwen3_5Pipeline::forward_body( int m, backend::stream_t stream ) {
	const pod::KVRef kv = kv_pool_.ref();
	const size_t kv_slot_stride = kv_pool_.slot_stride_elems;

	int full_attn_idx = 0;
	int linear_idx = 0;

	for ( int layer = 0; layer < config_.num_layers; ++layer ) {
		const auto& lw = layers_[layer];

		// 1. Input RMSNorm: norm_hidden = rms_norm(hidden_states)
		engine::modules::RMSNorm::forward(
			hidden_states_, lw.input_norm, norm_hidden_, config_.rms_norm_eps, config_.rms_norm_bias, stream
		);

		// 2. Attention
		if ( lw.is_full_attention ) {
			const int q_heads = q_heads_;
			const int kv_heads = kv_heads_;

			// Q proj: projects norm_hidden (5120) to Q+Gate (12288 or 3072)
			engine::modules::Linear::forward(
				norm_hidden_, lw.q_proj, q_gate_buf_, stream
			);

			// Unpack Q and Gate
			engine::modules::Attention::unpack_q_gate(
				q_gate_buf_, q_buf_, gate_attn_buf_,
				q_heads, config_.full_head_dim, stream
			);

			// RMSNorm on Q heads
			engine::modules::RMSNorm::forward_heads(
				q_buf_, lw.q_norm, q_heads, config_.full_head_dim, 1e-6f, 1.0f, stream
			);

			// K proj: projects norm_hidden (5120) to K (1024 or 256)
			engine::modules::Linear::forward(
				norm_hidden_, lw.k_proj, k_buf_, stream
			);

			// RMSNorm on K heads
			engine::modules::RMSNorm::forward_heads(
				k_buf_, lw.k_norm, kv_heads, config_.full_head_dim, 1e-6f, 1.0f, stream
			);

			// V proj: projects norm_hidden (5120) to V (1024 or 256)
			engine::modules::Linear::forward(
				norm_hidden_, lw.v_proj, v_buf_, stream
			);

			// RoPE applied to Q and K
			engine::modules::RoPE::forward(
				q_buf_, k_buf_,
				q_heads, kv_heads,
				config_.full_head_dim, config_.full_rot_dim,
				d_rows_, m, config_.rope_theta, stream
			);

			// Store K and V to KV Cache
			engine::modules::Attention::store_kv_cache(
				kv,
				k_buf_, v_buf_,
				kv_slot_stride, full_attn_idx, d_rows_, m,
				kv_heads, config_.full_head_dim,
				config_.max_seq, stream
			);

			// Attend over past tokens + sigmoid gate
			if ( config_.full_head_dim == 256 ) {
				// Prefill chunk: tiled kernel that shares K/V tiles across queries.
				engine::modules::Attention::forward_prefill(
					kv, q_buf_, gate_attn_buf_,
					attn_out_,
					kv_slot_stride, full_attn_idx, d_rows_, m,
					q_heads, kv_heads,
					config_.full_head_dim, config_.max_seq, stream, m > pod::MAX_BATCH ? key_hint_ : 0, m <= pod::MAX_BATCH && !attn_tc_
				);
			}
			else {
				engine::modules::Attention::forward_decode(
					kv, q_buf_, gate_attn_buf_,
					attn_out_,
					kv_slot_stride, full_attn_idx, d_rows_, m,
					q_heads, kv_heads,
					config_.full_head_dim, config_.max_seq, stream
				);
			}
			full_attn_idx++;

			// O proj: projects attn_out (6144 or 1536) to mlp_out (5120)
			engine::modules::Linear::forward(
				attn_out_, lw.o_proj, mlp_out_, stream
			);
		}
		else {
			const int k_heads = lin_k_heads_;
			const int v_heads = lin_v_heads_;
			const int lin_channels = lin_channels_;

			// In proj QKV: projects norm_hidden (5120) to QKV (10240 or 2560) across all M rows
			engine::modules::Linear::forward(
				norm_hidden_, lw.in_proj_qkv, qkv_buf_, stream
			);

			// In proj Z: projects norm_hidden (5120) to Z (6144 or 1536) across all M rows
			engine::modules::Linear::forward(
				norm_hidden_, lw.in_proj_z, z_buf_, stream
			);

			// Rows advance their own slot's recurrent state in order: a causal block (Mode A) re-uses
			// one slot row after row; independent requests (Mode B) each touch a distinct slot. Each stage
			// runs once over all m rows (the conv and the recurrence walk rows in order inside the kernel and
			// resolve each row's slot from d_rows_ on device).
			float* hist_ptr = state_.conv_history + static_cast<size_t>( linear_idx ) * lin_channels * 3;
			float* layer_gdn_state = state_.gdn_states + static_cast<size_t>( linear_idx ) * v_heads * 128 * 128;
			const int snap_rows = MIN( m, kDraftM );
			float* conv_snap = state_.conv_snap + static_cast<size_t>( linear_idx ) * lin_channels * 3;
			float* gdn_snap = state_.gdn_snap + static_cast<size_t>( linear_idx ) * v_heads * 128 * 128;

			pod::TensorView qkv_rows = pod::make_tensor_2d( qkv_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, lin_channels );
			pod::TensorView out_rows = pod::make_tensor_2d( gdn_out_.ptr<backend::half_t>(), pod::DType::FP16, m, v_heads * 128 );
			pod::TensorView z_rows = pod::make_tensor_2d( z_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, v_heads * 128 );
			pod::TensorView gate_rows = pod::make_tensor_2d( gdn_gate_.ptr<float>(), pod::DType::FP32, m, v_heads );
			pod::TensorView beta_rows = pod::make_tensor_2d( gdn_beta_.ptr<float>(), pod::DType::FP32, m, v_heads );

			// Causal conv1d + SiLU. Prefill chunks run it in parallel row tiles into a separate buffer; the rest of
			// the layer then reads that buffer instead of the in-place result.
			if ( m > pod::MAX_BATCH ) {
				pod::TensorView conv_rows = pod::make_tensor_2d( conv_out_.ptr<backend::half_t>(), pod::DType::FP16, m, lin_channels );
				engine::modules::GDN::causal_conv1d_silu_tiled(
					qkv_rows, conv_rows, hist_ptr, lw.conv1d_w, lin_channels, m, d_rows_, state_.slot_conv_floats, stream );
				qkv_rows = conv_rows;
			}
			else {
				engine::modules::GDN::causal_conv1d_silu_rows(
					qkv_rows, hist_ptr, lw.conv1d_w, lin_channels, m, d_rows_, state_.slot_conv_floats,
					conv_snap, state_.slot_conv_floats, snap_rows, state_.d_snap_on, stream
				);
			}

			// L2 norm on Q and K heads
			engine::modules::GDN::l2_norm_heads(
				qkv_rows, k_heads, config_.linear_head_dim, stream, false, m, lin_channels
			);

			// Project gates: in_proj_a, in_proj_b -> gdn_gate, gdn_beta
			engine::modules::GDN::project_gates(
				norm_hidden_, lw.in_proj_a, lw.in_proj_b, lw.A_log, lw.dt_bias,
				gate_rows, beta_rows, v_heads, config_.hidden_size, stream, m
			);

			// GDN recurrent step: advances state causally
			engine::modules::GDN::recurrent_step_rows(
				layer_gdn_state, qkv_rows, gate_rows, beta_rows, out_rows,
				k_heads, v_heads, config_.linear_head_dim, m, d_rows_, state_.slot_gdn_floats,
				gdn_snap, state_.slot_gdn_floats, snap_rows, state_.d_snap_on, stream
			);

			// Gated RMSNorm: gdn_out * rsqrt * norm_weight * silu(z)
			engine::modules::GDN::gated_rmsnorm(
				out_rows, z_rows, lw.linear_norm, v_heads, config_.linear_head_dim, stream, false, m
			);

			// Out proj: projects gdn_out (6144 or 1536) to mlp_out (5120) across all M rows
			engine::modules::Linear::forward(
				gdn_out_, lw.out_proj, mlp_out_, stream
			);
			linear_idx++;
		}

		if ( ring_ ) {
			ring_->all_reduce_rms_norm(
				mlp_out_,
				hidden_states_,
				lw.post_attn_norm,
				config_.rms_norm_eps,
				stream,
				config_.rms_norm_bias,
				&norm_hidden_
			);
		}
		else {
			engine::modules::Elementwise::add_residual( hidden_states_, mlp_out_, stream );
			engine::modules::RMSNorm::forward(
				hidden_states_, lw.post_attn_norm, norm_hidden_, config_.rms_norm_eps, config_.rms_norm_bias, stream
			);
		}

		// 4. MLP
		engine::modules::Linear::forward(
			norm_hidden_, lw.gate_proj, gate_buf_, stream
		);

		engine::modules::Linear::forward(
			norm_hidden_, lw.up_proj, up_buf_, stream
		);

		engine::modules::SwiGLU::forward( gate_buf_, up_buf_, mlp_inter_, stream );

		engine::modules::Linear::forward(
			mlp_inter_, lw.down_proj, mlp_out_, stream
		);

		if ( ring_ ) {
			ring_->all_reduce( mlp_out_, stream );
		}
		engine::modules::Elementwise::add_residual( hidden_states_, mlp_out_, stream );

		// Tapped layer states for DFlash2: tapped_states_[r][tap] = hidden after layer tap_layers[tap]
		for ( int tap = 0; tap < 5; ++tap ) {
			if ( layer != config_.tap_layers[tap] ) continue;
			backend::memcpy2d_async( tapped_states_ + static_cast<size_t>( tap ) * config_.hidden_size, 5 * config_.hidden_size * sizeof( backend::half_t ), hidden_states_.data, config_.hidden_size * sizeof( backend::half_t ), config_.hidden_size * sizeof( backend::half_t ), m, backend::Copy::D2D, stream );
			// Also keep it position-indexed for the drafter's context stage. Each row lands in its own
			// ring slot, so this cannot be the same strided copy as above. The slot, the position and
			// the owner all come from device memory: forward_body is replayed from a CUDA graph keyed
			// only on the row count, and a host-side value would be frozen at capture time.
			engine::spec::dflash_publish_taps(
				draft_taps_.ptr<backend::half_t>(), hidden_states_.ptr<const backend::half_t>(), d_rows_, d_taps_owner_,
				m, tap, config_.hidden_size, draft_hooks_.taps_cap, stream );
		}
	}

	// Final norm + LM head. Every rank projects its own vocabulary shard, so the 5120 x 248320 head is
	// four 5120 x 62208 GEMVs running in parallel instead of one 5120 x 248320 GEMV gating rank 0.
	if ( tp_size_ > 1 || rank_ == 0 ) {
		if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
		if ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) {
			// Prefill: only the final row's logits are read. A one-row head also takes the int8 GEMV path.
			const int H = config_.hidden_size;
			pod::TensorView h1 = pod::make_tensor_2d( hidden_states_.ptr<backend::half_t>() + static_cast<size_t>( m - 1 ) * H, pod::DType::FP16, 1, H );
			pod::TensorView n1 = pod::make_tensor_2d( norm_hidden_.ptr<backend::half_t>(), pod::DType::FP16, 1, H );
			pod::TensorView l1 = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, 1, lm_shard_cols_ );
			engine::modules::RMSNorm::forward( h1, final_norm_, n1, config_.rms_norm_eps, config_.rms_norm_bias, stream );
			engine::modules::Linear::forward( n1, lm_head_, l1, stream );
			return;
		}
		engine::modules::RMSNorm::forward(
			hidden_states_, final_norm_, norm_hidden_, config_.rms_norm_eps, config_.rms_norm_bias, stream
		);
		engine::modules::Linear::forward( norm_hidden_, lm_head_, logits_, stream );
	}
}

void Qwen3_5Pipeline::destroy() {
	if ( arena_ != nullptr ) {
		backend::set_device( rank_ );
		graphs_.reset();
	}
	if ( h_argmax_idx_ != nullptr ) {
		engine::core::host_free_pinned( h_argmax_idx_ );
		h_argmax_idx_ = nullptr;
	}
}

void Qwen3_5Pipeline::checkpoint_state( int slot_id, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen3_5Pipeline::checkpoint_state: slot out of range" );
	backend::set_device( rank_ );
	const size_t offset_gdn = static_cast<size_t>( slot_id ) * state_.slot_gdn_floats;
	const size_t offset_conv = static_cast<size_t>( slot_id ) * state_.slot_conv_floats;
	backend::memcpy_async( state_.gdn_checkpoint, state_.gdn_states + offset_gdn, state_.slot_gdn_floats * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_checkpoint, state_.conv_history + offset_conv, state_.slot_conv_floats * sizeof( float ), backend::Copy::D2D, stream );
}

void Qwen3_5Pipeline::rollback_state( int slot_id, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen3_5Pipeline::rollback_state: slot out of range" );
	backend::set_device( rank_ );
	const size_t offset_gdn = static_cast<size_t>( slot_id ) * state_.slot_gdn_floats;
	const size_t offset_conv = static_cast<size_t>( slot_id ) * state_.slot_conv_floats;
	backend::memcpy_async( state_.gdn_states + offset_gdn, state_.gdn_checkpoint, state_.slot_gdn_floats * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_history + offset_conv, state_.conv_checkpoint, state_.slot_conv_floats * sizeof( float ), backend::Copy::D2D, stream );
}

void Qwen3_5Pipeline::set_spec_snapshot( bool on, backend::stream_t stream ) {
	backend::set_device( rank_ );
	const int v = on ? 1 : 0;
	// Pageable source: staged before return, so a stack value is fine.
	backend::memcpy_async( state_.d_snap_on, &v, sizeof( int ), backend::Copy::H2D, stream );
}

void Qwen3_5Pipeline::commit_spec_row( int slot_id, int row, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen3_5Pipeline::commit_spec_row: slot out of range" );
	ENGINE_ASSERT( row >= 0 && row < kDraftM, "Qwen3_5Pipeline::commit_spec_row: row out of range" );
	backend::set_device( rank_ );
	backend::memcpy_async( state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, state_.gdn_snap + static_cast<size_t>( row ) * state_.slot_gdn_floats, state_.slot_gdn_floats * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, state_.conv_snap + static_cast<size_t>( row ) * state_.slot_conv_floats, state_.slot_conv_floats * sizeof( float ), backend::Copy::D2D, stream );
}

// -- Sequence snapshots (prefix cache). Host layout: [GDN states][conv history][tap rows][per attention layer: K rows, V rows
// for positions [from, tokens)].
namespace {
constexpr size_t kSeqAlign = 256;
}

size_t Qwen3_5Pipeline::tap_snapshot_bytes( int tokens ) const {
	if ( !snap_taps_ ) return 0;
	return static_cast<size_t>( std::min( tokens, draft_hooks_.taps_cap ) ) * 5 * config_.hidden_size * sizeof( backend::half_t );
}

size_t Qwen3_5Pipeline::seq_state_bytes( int from, int tokens ) const {
	if ( kv_pool_.format != pod::KVFormat::FP16 || from < 0 || tokens < from || tokens > config_.max_seq ) return 0;
	const size_t recurrent = ( state_.slot_gdn_floats + state_.slot_conv_floats ) * sizeof( float );
	const size_t kv_row = static_cast<size_t>( kv_pool_.num_heads ) * kv_pool_.head_dim * sizeof( backend::half_t );
	return recurrent + tap_snapshot_bytes( tokens ) + 2 * static_cast<size_t>( kv_pool_.num_layers ) * ( tokens - from ) * kv_row + kSeqAlign;
}

bool Qwen3_5Pipeline::save_seq_state( int slot_id, int from, int tokens, void* host_dst, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* dst = static_cast<char*>( host_dst );
	const size_t gdn = state_.slot_gdn_floats * sizeof( float ), conv = state_.slot_conv_floats * sizeof( float );
	backend::memcpy_async( dst, state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, gdn, backend::Copy::D2H, stream );
	backend::memcpy_async( dst + gdn, state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, conv, backend::Copy::D2H, stream );
	dst += gdn + conv;
	if ( const size_t tap_bytes = tap_snapshot_bytes( tokens ) ) {
		backend::memcpy_async( dst, draft_taps_.data, tap_bytes, backend::Copy::D2H, stream );
		dst += tap_bytes;
	}
	const size_t bytes = static_cast<size_t>( tokens - from ) * kv_pool_.num_heads * kv_pool_.head_dim * sizeof( backend::half_t );
	for ( int layer = 0; layer < kv_pool_.num_layers && bytes > 0; ++layer ) {
		backend::memcpy_async( dst, kv_pool_.get_k_ptr( slot_id, layer, from ), bytes, backend::Copy::D2H, stream );
		dst += bytes;
		backend::memcpy_async( dst, kv_pool_.get_v_ptr( slot_id, layer, from ), bytes, backend::Copy::D2H, stream );
		dst += bytes;
	}
	return true;
}

bool Qwen3_5Pipeline::load_seq_state( int slot_id, int from, int tokens, const void* host_src, bool kv_only, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	const char* src = static_cast<const char*>( host_src );
	const size_t gdn = state_.slot_gdn_floats * sizeof( float ), conv = state_.slot_conv_floats * sizeof( float );
	const size_t tap_bytes = tap_snapshot_bytes( tokens );
	if ( !kv_only ) {
		backend::memcpy_async( state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, src, gdn, backend::Copy::H2D, stream );
		backend::memcpy_async( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, src + gdn, conv, backend::Copy::H2D, stream );
		if ( tap_bytes ) backend::memcpy_async( draft_taps_.data, src + gdn + conv, tap_bytes, backend::Copy::H2D, stream );
	}
	src += gdn + conv + tap_bytes;
	const size_t bytes = static_cast<size_t>( tokens - from ) * kv_pool_.num_heads * kv_pool_.head_dim * sizeof( backend::half_t );
	for ( int layer = 0; layer < kv_pool_.num_layers && bytes > 0; ++layer ) {
		backend::memcpy_async( kv_pool_.get_k_ptr( slot_id, layer, from ), src, bytes, backend::Copy::H2D, stream );
		src += bytes;
		backend::memcpy_async( kv_pool_.get_v_ptr( slot_id, layer, from ), src, bytes, backend::Copy::H2D, stream );
		src += bytes;
	}
	return true;
}

} // namespace engine::models
