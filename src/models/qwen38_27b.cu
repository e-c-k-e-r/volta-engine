#include "engine/models/qwen38_27b.h"
#include "engine/loader/safetensors.h"
#include "trellis_context.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>

namespace engine::models {

Qwen38Pipeline::Qwen38Pipeline() = default;

Qwen38Pipeline::~Qwen38Pipeline() {
	destroy();
}

bool Qwen38Pipeline::init(
	const Qwen38Config& config,
	int rank,
	pod::MemoryArena& arena,
	engine::comm::CollectiveRing* ring
) {
	config_ = config;
	rank_ = rank;
	arena_ = &arena;
	ring_ = ring;
	weights_loaded_ = false;

	CUDA_CHECK( cudaSetDevice( rank_ ) );

	const int tp_size = ( ring_ != nullptr ) ? ENGINE_NUM_GPUS : 1;
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
	const int R = pod::MAX_BATCH;
	const int S = config_.max_slots;

	// 1. Recurrent GDN state per slot: [S, 48, v_heads, 128, 128] FP32 + one-slot checkpoint
	state_.slot_gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_v_heads * 128 * 128;
	size_t gdn_floats = static_cast<size_t>( S ) * state_.slot_gdn_floats;
	state_.gdn_states = static_cast<float*>( engine::core::arena_alloc( arena, gdn_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.gdn_states, 0, gdn_floats * sizeof( float ) ) );
	state_.gdn_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, state_.slot_gdn_floats * sizeof( float ) ) );

	// 2. Causal conv1d history per slot: [S, 48, channels, 3] FP32 + one-slot checkpoint
	state_.slot_conv_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_channels * 3;
	size_t conv_hist_floats = static_cast<size_t>( S ) * state_.slot_conv_floats;
	state_.conv_history = static_cast<float*>( engine::core::arena_alloc( arena, conv_hist_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.conv_history, 0, conv_hist_floats * sizeof( float ) ) );
	state_.conv_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, state_.slot_conv_floats * sizeof( float ) ) );

	// 3. Full-attention KV cache pool: [S, 16, max_seq, kv_heads, 256] FP16
	kv_pool_.init( rank_, arena, S, config_.max_seq, config_.num_attn_layers, full_kv_heads, config_.full_head_dim );

	// 4. Residual stream & normed activations (MAX_BATCH rows)
	int h_shape[2] = { R, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	CUDA_CHECK( cudaMemset( hidden_states_.data, 0, hidden_states_.size_bytes() ) );

	// Tapped hidden states for DFlash2: [R, 5, hidden]
	const size_t tap_bytes = static_cast<size_t>( R ) * 5 * config_.hidden_size * sizeof( half );
	tapped_states_ = static_cast<half*>( engine::core::arena_alloc( arena, tap_bytes ) );
	CUDA_CHECK( cudaMemset( tapped_states_, 0, tap_bytes ) );
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
	lm_shard_real_cols_ = config_.vocab_size / tp_size;
	int pad_cols = 0;
	if ( tp_size > 1 ) {
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
		if ( rank_ == 0 && tp_size_ > 1 ) {
			// Only the DFlash2 drafter needs the whole distribution; sampling reduces per shard.
			int full_shape[2] = { R, config_.vocab_size };
			full_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, full_shape, 2 );
		}
	}

	for ( int L = 0; L < config_.num_layers; ++L ) {
		layers_[L].is_full_attention = ( ( L + 1 ) % 4 == 0 );
	}

	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );

	metadata_.model_name = "qwen38";
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

bool Qwen38Pipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	return init( Qwen38Config{}, device_id, arena, ring );
}


void Qwen38Pipeline::reset_state( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen38Pipeline::reset_state: slot out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	kv_pool_.reset_slot( slot_id );
	CUDA_CHECK( cudaMemset( state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, 0, state_.slot_gdn_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, 0, state_.slot_conv_floats * sizeof( float ) ) );
}

bool Qwen38Pipeline::load_weights( const char* model_dir, cudaStream_t stream ) {
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) {
		std::printf( "    [loader] FAILED to open model directory %s\n", model_dir );
		return false;
	}

	// 1. Embed tokens
	embed_tokens_ = repo.load( "model.language_model.embed_tokens.weight", *arena_, stream, pod::DType::BF16 );

	// 2. Final norm
	final_norm_ = repo.load( "model.language_model.norm.weight", *arena_, stream, pod::DType::FP16 );

	// 3. LM Head, sharded across the vocabulary dimension
	if ( tp_size_ > 1 ) {
		lm_head_ = repo.load_linear_slice_padded( "lm_head", *arena_, 5, rank_, tp_size_, lm_shard_cols_ - lm_shard_real_cols_, stream );
	}
	else {
		lm_head_ = repo.load_linear( "lm_head", *arena_, 5, stream );
	}

	int tp_size = ( ring_ != nullptr ) ? 4 : 1;

	// 4. Per-layer weights across 64 layers
	char name_buf[256];
	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.is_full_attention = ( ( L + 1 ) % 4 == 0 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.input_layernorm.weight", L );
		lw.input_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.post_attention_layernorm.weight", L );
		lw.post_attn_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		if ( lw.is_full_attention ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_proj", L );
			lw.q_proj = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 4, 1, rank_, 4, stream )
			                             : repo.load_linear( name_buf, *arena_, 4, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_norm.weight", L );
			lw.q_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.k_proj", L );
			lw.k_proj = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 5, 1, rank_, 4, stream )
			                             : repo.load_linear( name_buf, *arena_, 5, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.k_norm.weight", L );
			lw.k_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.v_proj", L );
			lw.v_proj = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 5, 1, rank_, 4, stream )
			                             : repo.load_linear( name_buf, *arena_, 5, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj", L );
			lw.o_proj = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 4, 0, rank_, 4, stream )
			                             : repo.load_linear( name_buf, *arena_, 4, stream );
		}
		else {
			static constexpr int QWEN_QKV_BLOCKS[3] = { 128, 128, 384 };
			static constexpr int QWEN_QKV_RANK_BLOCKS[3] = { 32, 32, 96 };
			static constexpr int QWEN_CONV_CHANNELS[3] = { 2048, 2048, 6144 };
			static constexpr int QWEN_CONV_RANK_CHANNELS[3] = { 512, 512, 1536 };

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_qkv", L );
			lw.in_proj_qkv = ( tp_size == 4 )
				? repo.load_grouped_trellis_slice( name_buf, *arena_, QWEN_QKV_BLOCKS, QWEN_QKV_RANK_BLOCKS, 3, rank_, stream )
				: repo.load_linear( name_buf, *arena_, 5, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.conv1d.weight", L );
			lw.conv1d_w = ( tp_size == 4 )
				? repo.load_grouped_slice( name_buf, *arena_, QWEN_CONV_CHANNELS, QWEN_CONV_RANK_CHANNELS, 3, rank_, 0, stream )
				: repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.A_log", L );
			lw.A_log = ( tp_size == 4 ) ? repo.load_slice( name_buf, *arena_, 0, rank_, 4, stream, pod::DType::FP16 )
			                            : repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.dt_bias", L );
			lw.dt_bias = ( tp_size == 4 ) ? repo.load_slice( name_buf, *arena_, 0, rank_, 4, stream, pod::DType::FP16 )
			                              : repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", L );
			lw.in_proj_a = ( tp_size == 4 ) ? repo.load_slice( name_buf, *arena_, 0, rank_, 4, stream, pod::DType::FP16 )
			                                : repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", L );
			lw.in_proj_b = ( tp_size == 4 ) ? repo.load_slice( name_buf, *arena_, 0, rank_, 4, stream, pod::DType::FP16 )
			                                : repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_z", L );
			lw.in_proj_z = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 5, 1, rank_, 4, stream )
			                                : repo.load_linear( name_buf, *arena_, 5, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.norm.weight", L );
			lw.linear_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.out_proj", L );
			lw.out_proj = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 5, 0, rank_, 4, stream )
			                               : repo.load_linear( name_buf, *arena_, 5, stream );
		}

		// MLP
		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj", L );
		lw.gate_proj = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 3, 1, rank_, 4, stream )
		                                : repo.load_linear( name_buf, *arena_, 3, stream );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj", L );
		lw.up_proj = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 3, 1, rank_, 4, stream )
		                              : repo.load_linear( name_buf, *arena_, 3, stream );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj", L );
		lw.down_proj = ( tp_size == 4 ) ? repo.load_linear_slice( name_buf, *arena_, 3, 0, rank_, 4, stream )
		                                : repo.load_linear( name_buf, *arena_, 3, stream );
	}

	if ( stream ) {
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
	}
	repo.close();
	weights_loaded_ = true;
	return true;
}

void Qwen38Pipeline::set_rows( int m ) {
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

void Qwen38Pipeline::embed_tokens( const int* token_ids, int count, cudaStream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_BATCH, "Qwen38Pipeline::embed_tokens: count out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	// Pageable H2D: the source is staged before return, so the caller's array may be reused immediately
	CUDA_CHECK( cudaMemcpyAsync( d_token_ids_, token_ids, count * sizeof( int ), cudaMemcpyHostToDevice, stream ) );
	set_rows( count );
	engine::modules::Embedding::forward( embed_tokens_, d_token_ids_, count, hidden_states_, stream );
}

void Qwen38Pipeline::sample_tokens( int* out_preds, int count, cudaStream_t stream ) {
	ENGINE_ASSERT( tp_size_ == 1, "Qwen38Pipeline::sample_tokens: vocabulary is sharded, use sample_local" );
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Qwen38Pipeline::sample_tokens: count exceeds last forward rows" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax( logits_, d_argmax_idx_, lm_shard_cols_, count, stream );
	CUDA_CHECK( cudaMemcpyAsync( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	std::memcpy( out_preds, h_argmax_idx_, count * sizeof( int ) );
}

bool Qwen38Pipeline::sample_local( int* out_tokens, float* out_vals, int count, cudaStream_t stream ) {
	ENGINE_ASSERT( tp_size_ > 1, "Qwen38Pipeline::sample_local: vocabulary is not sharded" );
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Qwen38Pipeline::sample_local: count exceeds last forward rows" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
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
	CUDA_CHECK( cudaMemcpyAsync( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	std::memcpy( out_tokens, h_argmax_idx_, count * sizeof( int ) );
	if ( out_vals != nullptr ) {
		std::vector<float> h_vals( count );
		CUDA_CHECK( cudaMemcpyAsync( h_vals.data(), d_argmax_val_, count * sizeof( float ), cudaMemcpyDeviceToHost, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		std::memcpy( out_vals, h_vals.data(), count * sizeof( float ) );
	}
	return true;
}

const pod::TensorView& Qwen38Pipeline::gather_logits(
	int rows,
	const half* const* peer_shard_ptrs,
	int peer_shard_cols,
	cudaStream_t stream
) {
	ENGINE_ASSERT( tp_size_ == 1 || rank_ == 0, "Qwen38Pipeline::gather_logits: only rank 0 materializes the full vocabulary" );
	if ( tp_size_ == 1 ) {
		return logits_;
	}
	ENGINE_ASSERT( rows >= 1 && rows <= last_rows_, "Qwen38Pipeline::gather_logits: rows exceeds last forward rows" );
	ENGINE_ASSERT( peer_shard_cols == lm_shard_cols_, "Qwen38Pipeline::gather_logits: shard width mismatch" );

	full_logits_.shape[0] = rows;
	// Row r of the gathered buffer is [shard 0 | shard 1 | ...], so each row needs one copy per rank:
	// the source rows are lm_shard_cols_ apart (pad block included), the destination rows
	// lm_shard_real_cols_ apart.
	const size_t shard_elems = static_cast<size_t>( lm_shard_real_cols_ );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		const half* src_base = ( d == rank_ ) ? logits_.ptr<const half>() : peer_shard_ptrs[d];
		ENGINE_ASSERT( src_base != nullptr, "Qwen38Pipeline::gather_logits: missing peer shard pointer" );
		for ( int r = 0; r < rows; ++r ) {
			const half* src = src_base + static_cast<size_t>( r ) * lm_shard_cols_;
			half* dst = full_logits_.ptr<half>() + ( static_cast<size_t>( r ) * ENGINE_NUM_GPUS + d ) * shard_elems;
			CUDA_CHECK( cudaMemcpyPeerAsync( dst, rank_, src, d, shard_elems * sizeof( half ), stream ) );
		}
	}
	return full_logits_;
}

void Qwen38Pipeline::forward( const pod::BatchRows& rows, cudaStream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "Qwen38Pipeline::forward: weights not loaded" );
	ENGINE_ASSERT( rows.count >= 1 && rows.count <= pod::MAX_BATCH, "Qwen38Pipeline::forward: row count out of range" );
	for ( int r = 0; r < rows.count; ++r ) {
		ENGINE_ASSERT( rows.slot[r] >= 0 && rows.slot[r] < config_.max_slots, "Qwen38Pipeline::forward: slot out of range" );
		ENGINE_ASSERT( rows.pos[r] >= 0 && rows.pos[r] < config_.max_seq, "Qwen38Pipeline::forward: position out of range" );
	}
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	const int m = rows.count;
	last_slot_ = rows.slot[0];
	last_rows_ = m;
	set_rows( m );

	// Row descriptor goes to the device; the layer stack only reads slots/positions from there,
	// so its launch sequence depends on m alone and is replayed from a per-m CUDA graph.
	// Pageable source: the copy is staged before return, so `rows` may be a temporary.
	CUDA_CHECK( cudaMemcpyAsync( d_rows_, &rows, sizeof( pod::BatchRows ), cudaMemcpyHostToDevice, stream ) );
	graphs_.run( m, stream, [&]( cudaStream_t s ) { forward_body( m, s ); } );
}

void Qwen38Pipeline::forward_body( int m, cudaStream_t stream ) {
	half* slot_k = static_cast<half*>( kv_pool_.get_k_ptr( 0, 0, 0 ) );
	half* slot_v = static_cast<half*>( kv_pool_.get_v_ptr( 0, 0, 0 ) );
	const size_t kv_slot_stride = kv_pool_.slot_stride_bytes / sizeof( half );

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
				slot_k, slot_v,
				k_buf_, v_buf_,
				kv_slot_stride, full_attn_idx, d_rows_, m,
				kv_heads, config_.full_head_dim,
				config_.max_seq, stream
			);

			// Attend over past tokens + sigmoid gate
			engine::modules::Attention::forward_decode(
				q_buf_, slot_k, slot_v, gate_attn_buf_,
				attn_out_,
				kv_slot_stride, full_attn_idx, d_rows_, m,
				q_heads, kv_heads,
				config_.full_head_dim, config_.max_seq, stream
			);
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
			// one slot row after row; independent requests (Mode B) each touch a distinct slot.
			// Slot-0 layer pointers; the kernels add d_rows_->slot[m_idx] * stride on device.
			for ( int m_idx = 0; m_idx < m; ++m_idx ) {
				const pod::SlotRef conv_slot{ d_rows_, m_idx, state_.slot_conv_floats };
				const pod::SlotRef gdn_slot{ d_rows_, m_idx, state_.slot_gdn_floats };
				float* hist_ptr = state_.conv_history + static_cast<size_t>( linear_idx ) * lin_channels * 3;
				pod::TensorView conv_hist = pod::make_tensor_2d( hist_ptr, pod::DType::FP32, lin_channels, 3 );
				float* layer_gdn_state = state_.gdn_states + static_cast<size_t>( linear_idx ) * v_heads * 128 * 128;

				pod::TensorView qkv_row = pod::make_tensor_2d( qkv_buf_.ptr<half>() + m_idx * lin_channels, pod::DType::FP16, 1, lin_channels );
				pod::TensorView norm_row = pod::make_tensor_2d( norm_hidden_.ptr<half>() + m_idx * config_.hidden_size, pod::DType::FP16, 1, config_.hidden_size );
				pod::TensorView z_row = pod::make_tensor_2d( z_buf_.ptr<half>() + m_idx * v_heads * 128, pod::DType::FP16, 1, v_heads * 128 );
				pod::TensorView out_row = pod::make_tensor_2d( gdn_out_.ptr<half>() + m_idx * v_heads * 128, pod::DType::FP16, 1, v_heads * 128 );
				pod::TensorView gate_row = pod::make_tensor_2d( gdn_gate_.ptr<float>() + m_idx * v_heads, pod::DType::FP32, 1, v_heads );
				pod::TensorView beta_row = pod::make_tensor_2d( gdn_beta_.ptr<float>() + m_idx * v_heads, pod::DType::FP32, 1, v_heads );

				// Causal conv1d + SiLU
				engine::modules::GDN::causal_conv1d_silu(
					qkv_row, conv_hist, lw.conv1d_w, lin_channels, stream, conv_slot
				);

				// L2 norm on Q and K heads
				engine::modules::GDN::l2_norm_heads(
					qkv_row, k_heads, config_.linear_head_dim, stream
				);

				// Project gates: in_proj_a, in_proj_b -> gdn_gate, gdn_beta
				engine::modules::GDN::project_gates(
					norm_row, lw.in_proj_a, lw.in_proj_b, lw.A_log, lw.dt_bias,
					gate_row, beta_row, v_heads, config_.hidden_size, stream
				);

				// GDN recurrent step: advances state causally
				engine::modules::GDN::recurrent_step(
					layer_gdn_state, qkv_row, gate_row, beta_row,
					out_row, k_heads, v_heads,
					config_.linear_head_dim, stream, gdn_slot
				);

				// Gated RMSNorm: gdn_out * rsqrt * norm_weight * silu(z)
				engine::modules::GDN::gated_rmsnorm(
					out_row, z_row, lw.linear_norm, v_heads, config_.linear_head_dim, stream, false
				);
			}

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
			CUDA_CHECK( cudaMemcpy2DAsync(
				tapped_states_ + static_cast<size_t>( tap ) * config_.hidden_size,
				5 * config_.hidden_size * sizeof( half ),
				hidden_states_.data,
				config_.hidden_size * sizeof( half ),
				config_.hidden_size * sizeof( half ),
				m,
				cudaMemcpyDeviceToDevice, stream
			) );
		}
	}

	// Final norm + LM head. Every rank projects its own vocabulary shard, so the 5120 x 248320 head is
	// four 5120 x 62208 GEMVs running in parallel instead of one 5120 x 248320 GEMV gating rank 0.
	engine::modules::RMSNorm::forward(
		hidden_states_, final_norm_, norm_hidden_, config_.rms_norm_eps, config_.rms_norm_bias, stream
	);
	engine::modules::Linear::forward( norm_hidden_, lm_head_, logits_, stream );
}


void Qwen38Pipeline::destroy() {
	if ( arena_ != nullptr ) {
		CUDA_CHECK( cudaSetDevice( rank_ ) );
		graphs_.reset();
	}
	if ( h_argmax_idx_ != nullptr ) {
		engine::core::host_free_pinned( h_argmax_idx_ );
		h_argmax_idx_ = nullptr;
	}
}

void Qwen38Pipeline::checkpoint_state( int slot_id, cudaStream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen38Pipeline::checkpoint_state: slot out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	const size_t offset_gdn = static_cast<size_t>( slot_id ) * state_.slot_gdn_floats;
	const size_t offset_conv = static_cast<size_t>( slot_id ) * state_.slot_conv_floats;
	CUDA_CHECK( cudaMemcpyAsync( state_.gdn_checkpoint, state_.gdn_states + offset_gdn, state_.slot_gdn_floats * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( state_.conv_checkpoint, state_.conv_history + offset_conv, state_.slot_conv_floats * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
}

void Qwen38Pipeline::rollback_state( int slot_id, cudaStream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen38Pipeline::rollback_state: slot out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	const size_t offset_gdn = static_cast<size_t>( slot_id ) * state_.slot_gdn_floats;
	const size_t offset_conv = static_cast<size_t>( slot_id ) * state_.slot_conv_floats;
	CUDA_CHECK( cudaMemcpyAsync( state_.gdn_states + offset_gdn, state_.gdn_checkpoint, state_.slot_gdn_floats * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( state_.conv_history + offset_conv, state_.conv_checkpoint, state_.slot_conv_floats * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
}

} // namespace engine::models
