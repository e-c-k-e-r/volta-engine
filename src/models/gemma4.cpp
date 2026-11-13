#include "engine/models/gemma4.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/rope.h"
#include "engine/modules/attention.h"
#include "engine/modules/linear.h"
#include "engine/modules/embedding.h"
#include "engine/spec/dflash.h"
#include "engine/spec/tap_ring.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace engine::models {

Gemma4Pipeline::Gemma4Pipeline() = default;

Gemma4Pipeline::~Gemma4Pipeline() {
	destroy();
}

bool Gemma4Pipeline::init( const Gemma4Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	config_ = config;
	rank_ = rank;
	arena_ = &arena;
	ring_ = ring;
	weights_loaded_ = false;

	backend::set_device( rank_ );

	const int tp_size = ( ring_ != nullptr ) ? requested_tp_ : 1;
	ENGINE_ASSERT( tp_size == 1 || tp_size == 2 || tp_size == 4, "Gemma4Pipeline: tensor parallelism of 1, 2 or 4 ranks" );
	tp_size_ = tp_size;
	q_heads_ = config_.num_q_heads / tp_size;
	swa_kv_heads_ = config_.swa_kv_heads / tp_size;
	full_kv_heads_ = config_.full_kv_heads / tp_size;
	inter_size_ = config_.intermediate_size / tp_size;
	const int R = pod::MAX_ROWS;
	const int S = config_.max_slots;

	for ( int L = 0; L < config_.num_layers; ++L ) layers_[L].is_full = ( L + 1 ) % config_.full_every == 0;
	const int n_full = config_.num_layers / config_.full_every;
	// The sliding-window layers keep a ring of window + MAX_ROWS rows: a forward stores all its rows before attending, so a chunk's
	// first row still finds its window after the chunk's last row was written. A context no longer than that keeps every position.
	const int swa_ring = config_.max_seq > config_.swa_window + R ? config_.swa_window + R : 0;
	swa_pool_.init( rank_, arena, S, swa_ring > 0 ? swa_ring : config_.max_seq, config_.num_layers - n_full, swa_kv_heads_, config_.swa_head_dim );
	swa_pool_.ring = swa_ring;
	full_pool_.init( rank_, arena, S, config_.max_seq, n_full, full_kv_heads_, config_.full_head_dim );

	const int H = config_.hidden_size;
	int h_shape[2] = { R, H };
	resid_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, h_shape, 2 );
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	mlp_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	backend::memset_sync( resid_.data, 0, resid_.size_bytes() );

	d_token_ids_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
	d_rows_ = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );

	// Attention buffers sized for the wider of the two layer kinds
	const int q_cols = q_heads_ * std::max( config_.swa_head_dim, config_.full_head_dim );
	const int kv_cols = std::max( swa_kv_heads_ * config_.swa_head_dim, full_kv_heads_ * config_.full_head_dim );
	int q_shape[2] = { R, q_cols };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	int kv_shape[2] = { R, kv_cols };
	k_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
	v_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );

	int mlp_shape[2] = { R, inter_size_ };
	gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	mlp_inter_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );

	// Vocabulary-parallel head: every rank projects vocab / tp columns, padded to a multiple of 256 (the int8 GEMV's width)
	lm_shard_real_cols_ = config_.vocab_size / tp_size_;
	lm_shard_cols_ = lm_shard_real_cols_ + ( tp_size_ > 1 ? ( 256 - lm_shard_real_cols_ % 256 ) % 256 : 0 );
	{
		int logit_shape[2] = { pod::MAX_BATCH, lm_shard_cols_ };
		logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, logit_shape, 2 );
		d_argmax_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
		d_argmax_val_ = static_cast<float*>( engine::core::arena_alloc( arena, R * sizeof( float ) ) );
		h_argmax_idx_ = static_cast<int*>( engine::core::host_alloc_pinned( R * sizeof( int ) ) );
		std::memset( h_argmax_idx_, 0, R * sizeof( int ) );
		const int splits = ( lm_shard_cols_ + kTopkTile - 1 ) / kTopkTile;
		const size_t per = static_cast<size_t>( R ) * splits * 16;
		samp_part_v_ = static_cast<float*>( engine::core::arena_alloc( arena, per * sizeof( float ) ) );
		samp_part_i_ = static_cast<int*>( engine::core::arena_alloc( arena, per * sizeof( int ) ) );
		samp_counters_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
		d_samp_vals_ = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * 16 * sizeof( float ) ) );
		d_samp_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * 16 * sizeof( int ) ) );
		backend::memset_sync( samp_counters_, 0, R * sizeof( int ) );
		if ( rank_ == 0 && tp_size_ > 1 ) {
			int full_shape[2] = { pod::MAX_BATCH, config_.vocab_size };
			full_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, full_shape, 2 );
		}
	}

	// x sqrt( hidden ) on the embedding rows, the multiplier rounded through BF16 as the reference does
	{
		const float s = std::sqrt( static_cast<float>( H ) );
		uint32_t u;
		std::memcpy( &u, &s, 4 );
		u = ( u + 0x7FFF + ( ( u >> 16 ) & 1 ) ) & 0xFFFF0000u;
		std::memcpy( &embed_scale_, &u, 4 );
	}

	// DFlash drafter: a tap ring per slot (the hidden after each tapped layer) and the head scratch
	if ( speculation_ == Speculation::DFlash ) {
		engine::models::alloc_draft_taps( arena, draft_hooks_, draft_, config_.max_seq, H, S, true, Gemma4Config::kTaps );
		engine::models::alloc_draft_head( arena, draft_hooks_, draft_, H, lm_shard_cols_, kTopkTile, rank_ * lm_shard_real_cols_, lm_shard_real_cols_ );
		draft_hooks_.embedding = &embed_tokens_;
		draft_hooks_.embed_scale = embed_scale_;
		tap_shift_ = runtime::options().dflash_tap_shift;
	}

#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );
#endif

	metadata_.model_name = "gemma4";
	metadata_.hidden_size = H;
	metadata_.vocab_size = config_.vocab_size;
	metadata_.num_layers = config_.num_layers;
	metadata_.num_heads = config_.num_q_heads;
	metadata_.num_kv_heads = config_.swa_kv_heads;
	metadata_.head_dim = config_.swa_head_dim;
	metadata_.max_seq_len = config_.max_seq;
	metadata_.tp_size = tp_size;
	metadata_.max_slots = S;
	metadata_.max_rows = R;
	metadata_.eos_ids[0] = 1;     // <eos>
	metadata_.eos_ids[1] = 106;   // <turn|>
	metadata_.eos_ids[2] = 50;    // <|tool_response>: after its calls the model hands the turn to the tool
	return true;
}

bool Gemma4Pipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	Gemma4Config config;
	if ( runtime::options().max_seq > 0 ) config.max_seq = runtime::options().max_seq;
	if ( runtime::options().max_slots > 0 ) config.max_slots = runtime::options().max_slots;
	return init( config, device_id, arena, ring );
}

void Gemma4Pipeline::reset_state( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Gemma4Pipeline::reset_state: slot out of range" );
	backend::set_device( rank_ );
	swa_pool_.reset_slot( slot_id );
	full_pool_.reset_slot( slot_id );
}

// A one-element tensor (BF16, FP16 or FP32) read on the host
static float host_scalar( const engine::loader::SafeTensorsRepo& repo, const char* name ) {
	const pod::SafeTensorMeta* meta = repo.find( name );
	const void* p = repo.cpu_data( name );
	ENGINE_ASSERT( meta != nullptr && p != nullptr, "Gemma4Pipeline: missing scalar tensor" );
	if ( meta->dtype == pod::DType::FP32 ) return *static_cast<const float*>( p );
	if ( meta->dtype == pod::DType::BF16 ) return backend::to_float( *static_cast<const backend::bf16_t*>( p ) );
	ENGINE_ASSERT( meta->dtype == pod::DType::FP16, "Gemma4Pipeline: scalar tensor of an unexpected type" );
	return backend::to_float( *static_cast<const backend::half_t*>( p ) );
}

bool Gemma4Pipeline::load_weights( const char* model_dir, backend::stream_t stream ) {
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) {
		LOG_ERROR( "[loader] FAILED to open model directory %s\n", model_dir );
		return false;
	}
	const char* P = "model.language_model";
	char name[256];

	std::snprintf( name, sizeof( name ), "%s.embed_tokens.weight", P );
	if ( runtime::options().embed_host ) embed_tokens_ = repo.shared_host_bf16( name, stream );
	if ( !embed_tokens_.data ) embed_tokens_ = repo.load( name, *arena_, stream, pod::DType::BF16 );

	std::snprintf( name, sizeof( name ), "%s.norm.weight", P );
	final_norm_ = repo.load( name, *arena_, stream, pod::DType::FP16 );

	// NVFP4 checkpoint (modelopt): the MLP in NVFP4, attention and the tied head dense BF16. nvfp4_dense groups (attn, head, or all)
	// quantize those to NVFP4 at load
	std::snprintf( name, sizeof( name ), "%s.layers.0.mlp.gate_proj.weight_scale", P );
	const bool fp4 = repo.find( name ) != nullptr;
	const std::string nv_groups = "," + runtime::options().nvfp4_dense + ",";
	auto nv_group = [&]( const char* g ) { return nv_groups == ",all," || nv_groups.find( std::string( "," ) + g + "," ) != std::string::npos; };
	if ( fp4 ) LOG_ERROR( "[RANK %d] NVFP4 checkpoint\n", rank_ );

	// Tied head: the EXL3 checkpoint quantizes it as lm_head, the NVFP4 one reads the embedding
	const int pad = lm_shard_cols_ - lm_shard_real_cols_;
	if ( fp4 && ( tp_size_ > 1 || rank_ == 0 ) ) {
		std::snprintf( name, sizeof( name ), "%s.embed_tokens", P );
		const int beg = rank_ * lm_shard_real_cols_, cnt = lm_shard_real_cols_;
		if ( nv_group( "head" ) ) lm_head_ = repo.quantize_nvfp4( name, *arena_, &beg, &cnt, 1, 0, config_.hidden_size, lm_shard_cols_, stream );
		else {
			std::snprintf( name, sizeof( name ), "%s.embed_tokens.weight", P );
			lm_head_ = repo.load_dense_rows_padded( name, *arena_, beg, cnt, pad, stream );
		}
	}
	else if ( tp_size_ > 1 ) lm_head_ = repo.load_linear_slice_padded( "lm_head", *arena_, 6, rank_, tp_size_, pad, stream );
	else if ( rank_ == 0 ) lm_head_ = repo.load_linear( "lm_head", *arena_, 6, stream );

	const int tp = tp_size_;
	// A dense projection: FP16 sliced along its output rows (column parallel) or input columns (row parallel), or NVFP4 at load
	auto dense_proj = [&]( const char* base, bool split_n ) {
		if ( nv_group( "attn" ) ) return repo.quantize_nvfp4_slice( base, *arena_, split_n, rank_, tp, 0, stream );
		char b[320];
		std::snprintf( b, sizeof( b ), "%s.weight", base );
		pod::TensorView w = tp > 1 ? repo.load_slice( b, *arena_, split_n ? 0 : 1, rank_, tp, stream, pod::DType::FP16 ) : repo.load( b, *arena_, stream, pod::DType::FP16 );
		ENGINE_ASSERT( w.data != nullptr && w.ndim == 2, "Gemma4Pipeline: missing dense projection" );
		std::swap( w.shape[0], w.shape[1] );   // [n, k] in memory, tagged [k, n] for make_fp16
		return pod::LinearWeight::make_fp16( w );
	};
	auto packed = [&]( const char* base ) {
		char b[320];
		std::snprintf( b, sizeof( b ), "%s.weight_scale", base );
		return repo.find( b ) != nullptr;
	};
	auto load_col = [&]( const char* n ) {
		if ( fp4 ) return packed( n ) ? repo.load_nvfp4_slice( n, *arena_, true, rank_, tp, stream ) : dense_proj( n, true );
		return tp > 1 ? repo.load_linear_slice( n, *arena_, 4, 1, rank_, tp, stream ) : repo.load_linear( n, *arena_, 4, stream );
	};
	auto load_row = [&]( const char* n ) {
		if ( fp4 ) return packed( n ) ? repo.load_nvfp4_slice( n, *arena_, false, rank_, tp, stream ) : dense_proj( n, false );
		return tp > 1 ? repo.load_linear_slice( n, *arena_, 4, 0, rank_, tp, stream ) : repo.load_linear( n, *arena_, 4, stream );
	};
	auto load_norm = [&]( int L, const char* n ) {
		std::snprintf( name, sizeof( name ), "%s.layers.%d.%s.weight", P, L, n );
		return repo.load( name, *arena_, stream, pod::DType::FP16 );
	};

	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.input_norm = load_norm( L, "input_layernorm" );
		lw.post_attn_norm = load_norm( L, "post_attention_layernorm" );
		lw.pre_ffn_norm = load_norm( L, "pre_feedforward_layernorm" );
		lw.post_ffn_norm = load_norm( L, "post_feedforward_layernorm" );
		lw.q_norm = load_norm( L, "self_attn.q_norm" );
		lw.k_norm = load_norm( L, "self_attn.k_norm" );
		std::snprintf( name, sizeof( name ), "%s.layers.%d.layer_scalar", P, L );
		lw.layer_scalar = host_scalar( repo, name );

		std::snprintf( name, sizeof( name ), "%s.layers.%d.self_attn.q_proj", P, L );
		lw.q_proj = load_col( name );
		std::snprintf( name, sizeof( name ), "%s.layers.%d.self_attn.k_proj", P, L );
		lw.k_proj = load_col( name );
		if ( !lw.is_full ) {
			std::snprintf( name, sizeof( name ), "%s.layers.%d.self_attn.v_proj", P, L );
			lw.v_proj = load_col( name );
		}
		std::snprintf( name, sizeof( name ), "%s.layers.%d.self_attn.o_proj", P, L );
		lw.o_proj = load_row( name );

		std::snprintf( name, sizeof( name ), "%s.layers.%d.mlp.gate_proj", P, L );
		lw.gate_proj = load_col( name );
		std::snprintf( name, sizeof( name ), "%s.layers.%d.mlp.up_proj", P, L );
		lw.up_proj = load_col( name );
		std::snprintf( name, sizeof( name ), "%s.layers.%d.mlp.down_proj", P, L );
		lw.down_proj = load_row( name );
	}
	if ( stream ) backend::stream_sync( stream );
	repo.close();
	weights_loaded_ = true;
	return true;
}

void Gemma4Pipeline::set_rows( int m ) {
	resid_.shape[0] = m;
	hidden_states_.shape[0] = m;
	norm_hidden_.shape[0] = m;
	q_buf_.shape[0] = m;
	k_buf_.shape[0] = m;
	v_buf_.shape[0] = m;
	attn_out_.shape[0] = m;
	gate_buf_.shape[0] = m;
	up_buf_.shape[0] = m;
	mlp_inter_.shape[0] = m;
	mlp_out_.shape[0] = m;
	logits_.shape[0] = std::min( m, pod::MAX_BATCH );
}

void Gemma4Pipeline::embed_tokens( const int* token_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_ROWS, "Gemma4Pipeline::embed_tokens: count out of range" );
	backend::set_device( rank_ );
	backend::memcpy_async( d_token_ids_, token_ids, count * sizeof( int ), backend::Copy::H2D, stream );
	set_rows( count );
	engine::modules::Embedding::forward( embed_tokens_, d_token_ids_, count, hidden_states_, stream );
	engine::modules::Elementwise::scale_to_f32( hidden_states_, resid_, embed_scale_, stream );
}

void Gemma4Pipeline::sample_tokens( int* out_preds, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( tp_size_ == 1, "Gemma4Pipeline::sample_tokens: vocabulary is sharded, use sample_local" );
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Gemma4Pipeline::sample_tokens: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax( logits_, d_argmax_idx_, lm_shard_cols_, count, stream );
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_preds, h_argmax_idx_, count * sizeof( int ) );
}

bool Gemma4Pipeline::sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream ) {
	if ( tp_size_ <= 1 ) return false;
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Gemma4Pipeline::sample_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax_range( logits_, d_argmax_idx_, d_argmax_val_, lm_shard_real_cols_, count, rank_ * lm_shard_real_cols_, stream );
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	if ( out_vals != nullptr ) backend::memcpy_async( out_vals, d_argmax_val_, count * sizeof( float ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_tokens, h_argmax_idx_, count * sizeof( int ) );
	return true;
}

bool Gemma4Pipeline::lse_local( float* out, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Gemma4Pipeline::lse_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	engine::spec::logits_lse_shard( logits_.ptr<const backend::half_t>(), count, lm_shard_cols_, lm_shard_real_cols_, d_samp_vals_, stream );
	backend::memcpy_async( out, d_samp_vals_, static_cast<size_t>( count ) * 2 * sizeof( float ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

bool Gemma4Pipeline::topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Gemma4Pipeline::topk_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	engine::spec::dflash2_topk_shard( logits_.ptr<const backend::half_t>(), count, lm_shard_cols_, lm_shard_real_cols_, 16,
		d_samp_vals_, d_samp_idx_, rank_ * lm_shard_real_cols_, samp_part_v_, samp_part_i_, samp_counters_, stream );
	backend::memcpy_async( out_vals, d_samp_vals_, static_cast<size_t>( count ) * 16 * sizeof( float ), backend::Copy::D2H, stream );
	backend::memcpy_async( out_ids, d_samp_idx_, static_cast<size_t>( count ) * 16 * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

const pod::TensorView& Gemma4Pipeline::gather_logits( int rows, const backend::half_t* const* peer_shard_ptrs, int peer_shard_cols, backend::stream_t stream ) {
	ENGINE_ASSERT( tp_size_ == 1 || rank_ == 0, "Gemma4Pipeline::gather_logits: only rank 0 materializes the full vocabulary" );
	if ( tp_size_ == 1 ) return logits_;
	ENGINE_ASSERT( rows >= 1 && rows <= last_rows_ && rows <= pod::MAX_BATCH, "Gemma4Pipeline::gather_logits: rows out of range" );
	ENGINE_ASSERT( peer_shard_cols == lm_shard_cols_, "Gemma4Pipeline::gather_logits: shard width mismatch" );
	full_logits_.shape[0] = rows;
	const size_t shard_elems = static_cast<size_t>( lm_shard_real_cols_ );
	for ( int d = 0; d < tp_size_; ++d ) {
		const backend::half_t* src_base = ( d == rank_ ) ? logits_.ptr<const backend::half_t>() : peer_shard_ptrs[d];
		ENGINE_ASSERT( src_base != nullptr, "Gemma4Pipeline::gather_logits: missing peer shard pointer" );
		for ( int r = 0; r < rows; ++r ) {
			backend::half_t* dst = full_logits_.ptr<backend::half_t>() + ( static_cast<size_t>( r ) * tp_size_ + d ) * shard_elems;
			backend::peer_copy_async( dst, rank_, src_base + static_cast<size_t>( r ) * lm_shard_cols_, d, shard_elems * sizeof( backend::half_t ), stream );
		}
	}
	return full_logits_;
}

// The head shard over the drafter's hidden rows, then this rank's top-k (the softcap keeps the order, so it is skipped)
void Gemma4Pipeline::draft_head_topk( int rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_ && draft_.lm_logits.data != nullptr, "Gemma4Pipeline::draft_head_topk: no drafter head" );
	ENGINE_ASSERT( rows >= 1 && rows <= kDraftM, "Gemma4Pipeline::draft_head_topk: rows out of range" );
	backend::set_device( rank_ );
	pod::TensorView in = pod::make_tensor_2d( draft_hooks_.lm_in, pod::DType::FP16, kDraftM, config_.hidden_size );
	engine::modules::Linear::forward( in, lm_head_, draft_.lm_logits, stream );
	engine::spec::dflash2_topk_shard( draft_.lm_logits.ptr<const backend::half_t>(), kDraftM, lm_shard_cols_, lm_shard_real_cols_, kDraftTopK,
		draft_hooks_.cand_vals, draft_hooks_.cand_idx, draft_hooks_.shard_offset, draft_.cand_part_v, draft_.cand_part_i, draft_.cand_counters, stream );
}

bool Gemma4Pipeline::load_vision( const char* model_dir, backend::stream_t stream ) {
	if ( rank_ != 0 || runtime::options().max_image_tokens <= 0 ) return false;
	backend::set_device( rank_ );
	return vision_.load( model_dir, *arena_, runtime::options().max_image_tokens, stream );
}

bool Gemma4Pipeline::encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( rank_ != 0 || !vision_.loaded() ) return false;
	backend::set_device( rank_ );
	return vision_.encode( image, out, stream );
}

bool Gemma4Pipeline::inject_embeddings( const backend::half_t* host_rows, int dst_row, int rows, backend::stream_t stream ) {
	ENGINE_ASSERT( dst_row >= 0 && rows >= 1 && dst_row + rows <= pod::MAX_ROWS, "Gemma4Pipeline::inject_embeddings: rows out of range" );
	backend::set_device( rank_ );
	const int H = config_.hidden_size;
	backend::half_t* stage = hidden_states_.ptr<backend::half_t>() + static_cast<size_t>( dst_row ) * H;
	backend::memcpy_async( stage, host_rows, static_cast<size_t>( rows ) * H * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	pod::TensorView in = pod::make_tensor_2d( stage, pod::DType::FP16, rows, H );
	pod::TensorView dst = pod::make_tensor_2d( resid_.ptr<float>() + static_cast<size_t>( dst_row ) * H, pod::DType::FP32, rows, H );
	engine::modules::Elementwise::scale_to_f32( in, dst, 1.0f, stream );
	return true;
}

bool Gemma4Pipeline::read_taps( int slot_id, int first_pos, int count, backend::half_t* host_dst, backend::stream_t stream ) {
	backend::set_device( rank_ );
	return read_draft_taps( draft_hooks_, slot_id, config_.max_slots, first_pos, count, config_.hidden_size, host_dst, stream );
}

void Gemma4Pipeline::forward( const pod::BatchRows& rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "Gemma4Pipeline::forward: weights not loaded" );
	ENGINE_ASSERT( rows.count >= 1 && rows.count <= pod::MAX_ROWS, "Gemma4Pipeline::forward: row count out of range" );
	for ( int r = 0; r < rows.count; ++r ) {
		ENGINE_ASSERT( rows.slot[r] >= 0 && rows.slot[r] < config_.max_slots, "Gemma4Pipeline::forward: slot out of range" );
		ENGINE_ASSERT( rows.pos[r] >= 0 && rows.pos[r] < config_.max_seq, "Gemma4Pipeline::forward: position out of range" );
	}
	backend::set_device( rank_ );
	const int m = rows.count;
	lm_mode_ = m > pod::MAX_BATCH ? rows.logits_mode : pod::BatchRows::LOGITS_ALL;
	last_rows_ = ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) ? 0 : ( lm_mode_ == pod::BatchRows::LOGITS_LAST ? 1 : m );
	set_rows( m );
	key_hint_ = rows.pos[m - 1] + 1;
	backend::memcpy_async( d_rows_, &rows, rows.upload_bytes(), backend::Copy::H2D, stream );
	// Tensor-core attention shares K/V tiles across the rows of one slot; rows of distinct slots get a block each (a separate graph)
	attn_tc_ = true;
	for ( int r = 1; r < m; ++r ) attn_tc_ = attn_tc_ && rows.slot[r] == rows.slot[0];
	graphs_.run( m, stream, [&]( backend::stream_t s ) { forward_body( m, s ); }, attn_tc_ ? 1 : 0 );
}

// One decoder layer over the m rows of d_rows_ (resid_ in and out):
//   x += post_attn_norm( attn( input_norm( x ) ) ); x += post_ffn_norm( mlp( pre_ffn_norm( x ) ) ); x *= layer_scalar
// With TP the reductions of the two sublayer outputs carry the sandwich-norm add and the next norm (all_reduce_normed_add): the attention one writes
// the pre-FFN norm, the FFN one the next layer's input norm (next_norm), so that layer starts from norm_hidden_ (input_normed). Returns whether the
// next input norm was written.
bool Gemma4Pipeline::layer_forward( const Gemma4LayerWeights& lw, int kv_layer, int m, backend::stream_t stream, bool input_normed, const pod::TensorView* next_norm ) {
	using namespace engine::modules;
	const float eps = config_.rms_norm_eps;
	const engine::runtime::KVCachePool& pool = lw.is_full ? full_pool_ : swa_pool_;
	const pod::KVRef kv = pool.ref();
	const int hd = lw.is_full ? config_.full_head_dim : config_.swa_head_dim;
	const int kvh = lw.is_full ? full_kv_heads_ : swa_kv_heads_;

	if ( !input_normed ) RMSNorm::forward( resid_, lw.input_norm, norm_hidden_, eps, 0.0f, stream );

	pod::TensorView q = pod::make_tensor_2d( q_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, q_heads_ * hd );
	pod::TensorView k = pod::make_tensor_2d( k_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, kvh * hd );
	pod::TensorView v = pod::make_tensor_2d( v_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, kvh * hd );
	pod::TensorView o = pod::make_tensor_2d( attn_out_.ptr<backend::half_t>(), pod::DType::FP16, m, q_heads_ * hd );
	Linear::forward( norm_hidden_, lw.q_proj, q, stream );
	Linear::forward( norm_hidden_, lw.k_proj, k, stream );
	// Full layers take V from the raw K projection (attention_k_eq_v)
	if ( lw.is_full ) backend::memcpy_async( v.data, k.data, k.size_bytes(), backend::Copy::D2D, stream );
	else Linear::forward( norm_hidden_, lw.v_proj, v, stream );

	RMSNorm::forward_heads( q, lw.q_norm, q_heads_, hd, eps, 0.0f, stream );
	RMSNorm::forward_heads( k, lw.k_norm, kvh, hd, eps, 0.0f, stream );
	RMSNorm::forward_heads( v, pod::TensorView{}, kvh, hd, eps, 0.0f, stream );   // unweighted v_norm
	if ( lw.is_full ) RoPE::forward( q, k, q_heads_, kvh, hd, hd, d_rows_, m, config_.full_rope_theta, stream, false, config_.full_rot_pairs );
	else RoPE::forward( q, k, q_heads_, kvh, hd, hd, d_rows_, m, config_.swa_rope_theta, stream );

	Attention::store_kv_cache( kv, k, v, pool.slot_stride_elems, kv_layer, d_rows_, m, kvh, hd, pool.max_seq_len, stream );
	// Tensor cores: K/V tiles shared across the rows of one slot, a block per row otherwise
	Attention::forward_prefill( kv, q, pod::TensorView{}, o, pool.slot_stride_elems, kv_layer, d_rows_, m, q_heads_, kvh, hd, pool.max_seq_len, stream,
		m > pod::MAX_BATCH ? key_hint_ : 0, m <= pod::MAX_BATCH && !attn_tc_, 1.0f, lw.is_full ? 0 : config_.swa_window );

	Linear::forward( o, lw.o_proj, mlp_out_, stream );
	const bool fuse = ring_ != nullptr && runtime::options().normed_add_fuse;
	if ( fuse ) ring_->all_reduce_normed_add( mlp_out_, resid_, lw.post_attn_norm, 1.0f, &lw.pre_ffn_norm, &norm_hidden_, eps, 0.0f, stream );
	else {
		if ( ring_ ) ring_->all_reduce( mlp_out_, stream );
		Elementwise::add_normed( resid_, mlp_out_, lw.post_attn_norm, eps, 0.0f, 1.0f, stream );
		RMSNorm::forward( resid_, lw.pre_ffn_norm, norm_hidden_, eps, 0.0f, stream );
	}
	Linear::forward( norm_hidden_, lw.gate_proj, gate_buf_, stream );
	Linear::forward( norm_hidden_, lw.up_proj, up_buf_, stream );
	SwiGLU::forward( gate_buf_, up_buf_, mlp_inter_, stream, 0.0f, true );   // gelu_tanh( gate ) * up
	Linear::forward( mlp_inter_, lw.down_proj, mlp_out_, stream );
	if ( fuse ) {
		ring_->all_reduce_normed_add( mlp_out_, resid_, lw.post_ffn_norm, lw.layer_scalar, next_norm, next_norm != nullptr ? &norm_hidden_ : nullptr, eps, 0.0f, stream );
		return next_norm != nullptr;
	}
	if ( ring_ ) ring_->all_reduce( mlp_out_, stream );
	Elementwise::add_normed( resid_, mlp_out_, lw.post_ffn_norm, eps, 0.0f, lw.layer_scalar, stream );
	return false;
}

void Gemma4Pipeline::forward_body( int m, backend::stream_t stream ) {
	int swa_idx = 0, full_idx = 0;
	bool normed = false;
	for ( int L = 0; L < config_.num_layers; ++L ) {
		const Gemma4LayerWeights& lw = layers_[L];
		normed = layer_forward( lw, lw.is_full ? full_idx++ : swa_idx++, m, stream, normed, L + 1 < config_.num_layers ? &layers_[L + 1].input_norm : nullptr );
		// Drafter taps: the hidden after the tapped layers, into each row's slot ring (slot and position from d_rows_: graph-replayable)
		for ( int tap = 0; tap < Gemma4Config::kTaps && draft_.taps.data != nullptr; ++tap ) {
			if ( L != config_.tap_layers[tap] + tap_shift_ ) continue;
			engine::spec::dflash_publish_taps_slots_f32( draft_.taps.ptr<backend::half_t>(), resid_.ptr<const float>(), d_rows_, m, tap, config_.hidden_size,
				draft_hooks_.taps_cap, draft_hooks_.taps_slot_stride, stream, Gemma4Config::kTaps );
		}
	}

	// Final norm, the head shard, the final logit softcap
	if ( tp_size_ == 1 && rank_ != 0 ) return;
	if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
	const int H = config_.hidden_size;
	if ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) {
		pod::TensorView r1 = pod::make_tensor_2d( resid_.ptr<float>() + static_cast<size_t>( m - 1 ) * H, pod::DType::FP32, 1, H );
		pod::TensorView n1 = pod::make_tensor_2d( norm_hidden_.ptr<backend::half_t>(), pod::DType::FP16, 1, H );
		pod::TensorView l1 = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, 1, lm_shard_cols_ );
		engine::modules::RMSNorm::forward( r1, final_norm_, n1, config_.rms_norm_eps, 0.0f, stream );
		engine::modules::Linear::forward( n1, lm_head_, l1, stream );
		engine::modules::Elementwise::softcap( l1, config_.final_softcap, stream );
		return;
	}
	ENGINE_ASSERT( m <= pod::MAX_BATCH, "Gemma4Pipeline: every row's logits are kept for up to MAX_BATCH rows" );
	engine::modules::RMSNorm::forward( resid_, final_norm_, norm_hidden_, config_.rms_norm_eps, 0.0f, stream );
	engine::modules::Linear::forward( norm_hidden_, lm_head_, logits_, stream );
	engine::modules::Elementwise::softcap( logits_, config_.final_softcap, stream );
}

void Gemma4Pipeline::destroy() {
	if ( arena_ != nullptr ) {
		backend::set_device( rank_ );
		graphs_.reset();
	}
	if ( h_argmax_idx_ != nullptr ) {
		engine::core::host_free_pinned( h_argmax_idx_ );
		h_argmax_idx_ = nullptr;
	}
}

// -- Sequence snapshots (prefix cache). Host layout: the drafter tap rows (with taps), the sliding pool's layers, then the full pool's; per
// layer K data, K scales, V data, V scales of positions [from, tokens). The sliding pool keeps only the last window of the range: a
// restore needs the window before its end, which this segment and its parents' last windows cover.
namespace {

constexpr size_t kSeqAlign = 256;

// First position of [from, tokens) a pool's segment keeps
int kept_from( const engine::runtime::KVCachePool& pool, int window, int from, int tokens ) {
	return pool.ring > 0 ? std::max( from, tokens - window ) : from;
}

size_t pool_rows_bytes( const engine::runtime::KVCachePool& pool, int rows ) {
	const size_t row_elems = static_cast<size_t>( pool.num_heads ) * pool.head_dim;
	const pod::KVQ kq = pod::kv_k_quant( pool.format ), vq = pod::kv_v_quant( pool.format );
	const size_t e = row_elems * rows;
	return static_cast<size_t>( pool.num_layers ) * ( pod::kv_data_bytes( kq, e ) + pod::kv_scale_bytes( kq, e ) + pod::kv_data_bytes( vq, e ) + pod::kv_scale_bytes( vq, e ) );
}

}

size_t Gemma4Pipeline::tap_snapshot_bytes( int tokens ) const {
	if ( !snap_taps_ || draft_.taps.data == nullptr ) return 0;
	return static_cast<size_t>( std::min( tokens, draft_hooks_.taps_cap ) ) * draft_hooks_.taps_n * config_.hidden_size * sizeof( backend::half_t );
}

size_t Gemma4Pipeline::seq_state_bytes( int from, int tokens, bool with_taps ) const {
	if ( !swa_pool_.k_data || from < 0 || tokens < from || tokens > config_.max_seq ) return 0;
	const int swa_from = kept_from( swa_pool_, config_.swa_window, from, tokens );
	return ( with_taps ? tap_snapshot_bytes( tokens ) : 0 ) + pool_rows_bytes( swa_pool_, tokens - swa_from ) + pool_rows_bytes( full_pool_, tokens - from ) + kSeqAlign;
}

void Gemma4Pipeline::copy_kv_rows( const engine::runtime::KVCachePool& pool, int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream ) {
	from = kept_from( pool, config_.swa_window, from, tokens );
	const size_t row_elems = static_cast<size_t>( pool.num_heads ) * pool.head_dim;
	if ( tokens <= from ) return;
	const pod::KVQ kqs[2] = { pod::kv_k_quant( pool.format ), pod::kv_v_quant( pool.format ) };
	char* data_base[2] = { static_cast<char*>( pool.k_data ), static_cast<char*>( pool.v_data ) };
	char* scale_base[2] = { static_cast<char*>( pool.k_scale ), static_cast<char*>( pool.v_scale ) };
	const size_t slot_data[2] = { pool.k_slot_bytes, pool.v_slot_bytes };
	const size_t slot_scale[2] = { pool.k_scale_slot_bytes, pool.v_scale_slot_bytes };
	const backend::Copy dir = to_host ? backend::Copy::D2H : backend::Copy::H2D;
	for ( int layer = 0; layer < pool.num_layers; ++layer ) {
		for ( int kv = 0; kv < 2; ++kv ) {
			// positions [from, tokens) in runs of contiguous cache rows (a ring wraps at most once over a kept range)
			for ( int p = from; p < tokens; ) {
				const int row = pool.ring > 0 ? p % pool.ring : p;
				const int n = pool.ring > 0 ? std::min( tokens - p, pool.ring - row ) : tokens - p;
				const size_t elem0 = ( static_cast<size_t>( layer ) * pool.max_seq_len + row ) * row_elems;
				const size_t db = pod::kv_data_bytes( kqs[kv], row_elems * n );
				char* dev = data_base[kv] + static_cast<size_t>( slot_id ) * slot_data[kv] + pod::kv_data_bytes( kqs[kv], elem0 );
				if ( to_host ) backend::memcpy_async( host, dev, db, dir, stream );
				else backend::memcpy_async( dev, host, db, dir, stream );
				host += db;
				if ( const size_t sb = pod::kv_scale_bytes( kqs[kv], row_elems * n ) ) {
					char* sdev = scale_base[kv] + static_cast<size_t>( slot_id ) * slot_scale[kv] + pod::kv_scale_bytes( kqs[kv], elem0 );
					if ( to_host ) backend::memcpy_async( host, sdev, sb, dir, stream );
					else backend::memcpy_async( sdev, host, sb, dir, stream );
					host += sb;
				}
				p += n;
			}
		}
	}
}

bool Gemma4Pipeline::save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* dst = static_cast<char*>( host_dst );
	if ( const size_t tap_bytes = with_taps ? tap_snapshot_bytes( tokens ) : 0 ) {
		backend::memcpy_async( dst, draft_.taps.ptr<backend::half_t>() + static_cast<size_t>( slot_id ) * draft_hooks_.taps_slot_stride, tap_bytes, backend::Copy::D2H, stream );
		dst += tap_bytes;
	}
	copy_kv_rows( swa_pool_, slot_id, from, tokens, dst, true, stream );
	copy_kv_rows( full_pool_, slot_id, from, tokens, dst + pool_rows_bytes( swa_pool_, tokens - kept_from( swa_pool_, config_.swa_window, from, tokens ) ), true, stream );
	return true;
}

bool Gemma4Pipeline::load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* src = const_cast<char*>( static_cast<const char*>( host_src ) );
	if ( const size_t tap_bytes = with_taps ? tap_snapshot_bytes( tokens ) : 0 ) {
		if ( !kv_only ) backend::memcpy_async( draft_.taps.ptr<backend::half_t>() + static_cast<size_t>( slot_id ) * draft_hooks_.taps_slot_stride, src, tap_bytes, backend::Copy::H2D, stream );
		src += tap_bytes;
	}
	copy_kv_rows( swa_pool_, slot_id, from, tokens, src, false, stream );
	copy_kv_rows( full_pool_, slot_id, from, tokens, src + pool_rows_bytes( swa_pool_, tokens - kept_from( swa_pool_, config_.swa_window, from, tokens ) ), false, stream );
	return true;
}

} // namespace engine::models
