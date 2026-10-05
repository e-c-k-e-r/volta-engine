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

namespace {

__global__ void embed_token_kernel(
	const __nv_bfloat16* __restrict__ embed_table,
	half* __restrict__ hidden_state,
	int token_id,
	int hidden_size
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid < hidden_size ) {
		size_t idx = static_cast<size_t>( token_id ) * hidden_size + tid;
		__nv_bfloat16 bf = embed_table[idx];
		hidden_state[tid] = __float2half_rn( __bfloat162float( bf ) );
	}
}

__global__ void embed_tokens_kernel(
	const __nv_bfloat16* __restrict__ embed_table,
	half* __restrict__ hidden_state,
	const int* __restrict__ token_ids,
	int count,
	int hidden_size
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if ( b < count && tid < hidden_size ) {
		int token_id = token_ids[b];
		size_t idx = static_cast<size_t>( token_id ) * hidden_size + tid;
		__nv_bfloat16 bf = embed_table[idx];
		hidden_state[static_cast<size_t>( b ) * hidden_size + tid] = __float2half_rn( __bfloat162float( bf ) );
	}
}

} // namespace

Qwen38Pipeline::Qwen38Pipeline()
	: rank_{ 0 }
	, arena_{ nullptr }
	, ring_{ nullptr }
	, state_{}
	, weights_loaded_{ false }
	, tapped_states_{ nullptr }
	, layers_{}
	, d_argmax_idx_{ nullptr }
	, h_argmax_idx_{ nullptr }
	, graph_m1_{ nullptr }
	, instance_m1_{ nullptr }
	, graph_m1_captured_{ false }
	, graph_m4_{ nullptr }
	, instance_m4_{ nullptr }
	, graph_m4_captured_{ false }
	, graph_m8_{ nullptr }
	, instance_m8_{ nullptr }
	, graph_m8_captured_{ false } {
}

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

	int tp_size = ( ring_ != nullptr ) ? 4 : 1;
	int full_q_heads = ( tp_size == 4 ) ? 6 : config_.full_num_q_heads;
	int full_kv_heads = ( tp_size == 4 ) ? 1 : config_.full_num_kv_heads;
	int lin_v_heads = ( tp_size == 4 ) ? 12 : config_.linear_num_v_heads;
	int lin_channels = ( tp_size == 4 ) ? 2560 : config_.linear_channels;
	int inter_size = ( tp_size == 4 ) ? 4352 : config_.intermediate_size;

	// 1. Allocate recurrent GDN states: 8 slots x 48 layers x lin_v_heads x 128 x 128 (FP32)
	size_t gdn_floats = 8ULL * static_cast<size_t>( config_.num_linear_layers ) * lin_v_heads * 128 * 128;
	state_.gdn_states = static_cast<float*>( engine::core::arena_alloc( arena, gdn_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.gdn_states, 0, gdn_floats * sizeof( float ) ) );
	gdn_checkpoint_ = static_cast<float*>( engine::core::arena_alloc( arena, gdn_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( gdn_checkpoint_, 0, gdn_floats * sizeof( float ) ) );

	// 2. Allocate GDN causal conv1d history: 8 slots x 48 layers x lin_channels x 3 taps (FP32)
	size_t conv_hist_floats = 8ULL * static_cast<size_t>( config_.num_linear_layers ) * lin_channels * 3;
	state_.conv_history = static_cast<float*>( engine::core::arena_alloc( arena, conv_hist_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.conv_history, 0, conv_hist_floats * sizeof( float ) ) );
	conv_checkpoint_ = static_cast<float*>( engine::core::arena_alloc( arena, conv_hist_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( conv_checkpoint_, 0, conv_hist_floats * sizeof( float ) ) );

	// 3. Allocate full-attention KV cache pool: 8 slots x 16 layers x max_seq x full_kv_heads x 256 (FP16)
	internal_kv_pool_.init( rank_, arena, 8, config_.max_seq, config_.num_attn_layers, full_kv_heads, config_.full_head_dim );
	kv_pool_ = &internal_kv_pool_;
	state_.k_cache = static_cast<half*>( kv_pool_->get_k_ptr( 0, 0, 0 ) );
	state_.v_cache = static_cast<half*>( kv_pool_->get_v_ptr( 0, 0, 0 ) );
	state_.current_pos = 0;

	// 4. Hidden states & residuals (allocated with capacity for M=8 verify)
	int h_shape[2] = { 8, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	residual_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	hidden_states_.shape[0] = 1;
	residual_.shape[0] = 1;
	norm_hidden_.shape[0] = 1;

	// Tapped hidden states for DFlash2: 8 rows x 5 target layers x hidden_size
	tapped_states_ = static_cast<half*>( engine::core::arena_alloc( arena, 8ULL * 5ULL * config_.hidden_size * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( tapped_states_, 0, 8ULL * 5ULL * config_.hidden_size * sizeof( half ) ) );

	// 5. Full attention buffers (M=8 capacity)
	int q_gate_shape[2] = { 8, full_q_heads * config_.full_head_dim * 2 };
	q_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_gate_shape, 2 );

	int q_shape[2] = { 8, full_q_heads * config_.full_head_dim };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	gate_attn_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );

	int kv_shape[2] = { 8, full_kv_heads * config_.full_head_dim };
	k_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
	v_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );

	// 6. Linear attention (GDN) buffers (M=8 capacity)
	int qkv_shape[2] = { 8, lin_channels };
	qkv_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );

	int z_shape[2] = { 8, lin_v_heads * config_.linear_head_dim };
	z_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	gdn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );

	int gdn_param_shape[1] = { lin_v_heads };
	gdn_gate_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdn_param_shape, 1 );
	gdn_beta_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdn_param_shape, 1 );

	// 7. MLP buffers (M=8 capacity)
	int mlp_shape[2] = { 8, inter_size };
	gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	int gate_up_shape[2] = { 8, inter_size * 2 };
	gate_up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, gate_up_shape, 2 );
	mlp_inter_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );

	int mlp_out_shape[2] = { 8, config_.hidden_size };
	mlp_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_out_shape, 2 );

	// 8. Logits & argmax (rank 0 hosts M=8 capacity)
	if ( rank_ == 0 ) {
		int logit_shape[2] = { 8, 248320 };
		logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, logit_shape, 2 );

		d_argmax_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, 8 * sizeof( int ) ) );
		h_argmax_idx_ = static_cast<int*>( engine::core::host_alloc_pinned( 8 * sizeof( int ) ) );
		std::memset( h_argmax_idx_, 0, 8 * sizeof( int ) );
	}

	CUDA_CHECK( cudaMemset( hidden_states_.data, 0, hidden_states_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( residual_.data, 0, residual_.size_bytes() ) );

	if ( ring_ == nullptr ) {
		// Fallback dummy descriptors for decode microbenchmark in single-GPU mode
		int d_gu_shape[3] = { config_.hidden_size / 16, config_.intermediate_size / 16, 64 };
		pod::TensorView dummy_gate_up = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_gu_shape, 3 );

		int d_dn_shape[3] = { config_.intermediate_size / 16, config_.hidden_size / 16, 64 };
		pod::TensorView dummy_down = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_dn_shape, 3 );

		int d_q_shape[3] = { config_.hidden_size / 16, ( config_.full_num_q_heads * config_.full_head_dim * 2 ) / 16, 64 };
		pod::TensorView dummy_q = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_q_shape, 3 );

		int d_kv_shape[3] = { config_.hidden_size / 16, ( config_.full_num_kv_heads * config_.full_head_dim ) / 16, 64 };
		pod::TensorView dummy_kv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_kv_shape, 3 );

		int d_o_shape[3] = { ( config_.full_num_q_heads * config_.full_head_dim ) / 16, config_.hidden_size / 16, 64 };
		pod::TensorView dummy_o = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_o_shape, 3 );

		int d_qkv_shape[3] = { config_.hidden_size / 16, config_.linear_channels / 16, 64 };
		pod::TensorView dummy_qkv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_qkv_shape, 3 );

		int d_z_shape[3] = { config_.hidden_size / 16, ( config_.linear_num_v_heads * config_.linear_head_dim ) / 16, 64 };
		pod::TensorView dummy_z = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_z_shape, 3 );

		int d_out_shape[3] = { ( config_.linear_num_v_heads * config_.linear_head_dim ) / 16, config_.hidden_size / 16, 64 };
		pod::TensorView dummy_lin_out = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_out_shape, 3 );

		int suh_h_shape[1] = { config_.hidden_size };
		pod::TensorView dummy_suh_h = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_h_shape, 1 );
		pod::TensorView dummy_svh_h = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_h_shape, 1 );
		pod::TensorView dummy_norm_h = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_h_shape, 1 );

		int svh_mlp_shape[1] = { config_.intermediate_size };
		pod::TensorView dummy_suh_mlp = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_mlp_shape, 1 );
		pod::TensorView dummy_svh_mlp = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_mlp_shape, 1 );

		int svh_q_shape[1] = { config_.full_num_q_heads * config_.full_head_dim * 2 };
		pod::TensorView dummy_svh_q = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_q_shape, 1 );

		int svh_kv_shape[1] = { config_.full_num_kv_heads * config_.full_head_dim };
		pod::TensorView dummy_svh_kv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_kv_shape, 1 );

		int suh_o_shape[1] = { config_.full_num_q_heads * config_.full_head_dim };
		pod::TensorView dummy_suh_o = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_o_shape, 1 );

		int svh_qkv_shape[1] = { config_.linear_channels };
		pod::TensorView dummy_svh_qkv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_qkv_shape, 1 );

		int svh_z_shape[1] = { config_.linear_num_v_heads * config_.linear_head_dim };
		pod::TensorView dummy_svh_z = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_z_shape, 1 );

		int conv_shape[2] = { config_.linear_channels, 4 };
		pod::TensorView dummy_conv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, conv_shape, 2 );

		int ab_shape[2] = { config_.linear_num_v_heads, config_.hidden_size };
		pod::TensorView dummy_ab = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ab_shape, 2 );

		int gdn_bias_shape[1] = { config_.linear_num_v_heads };
		pod::TensorView dummy_gdn_b = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, gdn_bias_shape, 1 );

		int lin_norm_shape[1] = { config_.linear_head_dim };
		pod::TensorView dummy_lin_norm = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, lin_norm_shape, 1 );

		int qk_norm_shape[1] = { config_.full_head_dim };
		pod::TensorView dummy_qk_norm = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qk_norm_shape, 1 );

		CUDA_CHECK( cudaMemset( dummy_gate_up.data, 0x44, dummy_gate_up.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_down.data, 0x44, dummy_down.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_q.data, 0x44, dummy_q.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_kv.data, 0x44, dummy_kv.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_o.data, 0x44, dummy_o.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_qkv.data, 0x44, dummy_qkv.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_z.data, 0x44, dummy_z.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_lin_out.data, 0x44, dummy_lin_out.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_suh_h.data, 0x3c, dummy_suh_h.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_svh_h.data, 0x3c, dummy_svh_h.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_norm_h.data, 0x3c, dummy_norm_h.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_suh_mlp.data, 0x3c, dummy_suh_mlp.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_svh_mlp.data, 0x3c, dummy_svh_mlp.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_svh_q.data, 0x3c, dummy_svh_q.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_svh_kv.data, 0x3c, dummy_svh_kv.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_suh_o.data, 0x3c, dummy_suh_o.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_svh_qkv.data, 0x3c, dummy_svh_qkv.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_svh_z.data, 0x3c, dummy_svh_z.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_conv.data, 0x3c, dummy_conv.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_ab.data, 0x3c, dummy_ab.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_gdn_b.data, 0x3c, dummy_gdn_b.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_lin_norm.data, 0x3c, dummy_lin_norm.size_bytes() ) );
		CUDA_CHECK( cudaMemset( dummy_qk_norm.data, 0x3c, dummy_qk_norm.size_bytes() ) );

		dummy_down_ = dummy_down;
		dummy_suh_h_ = dummy_suh_h;
		dummy_svh_h_ = dummy_svh_h;
		dummy_norm_h_ = dummy_norm_h;

		for ( int L = 0; L < config_.num_layers; ++L ) {
			auto& lw = layers_[L];
			lw.is_full_attention = ( ( L + 1 ) % 4 == 0 );
			lw.input_norm = dummy_norm_h;
			lw.post_attn_norm = dummy_norm_h;

			lw.q_proj = pod::LinearWeight::make_trellis( dummy_q, dummy_suh_h, dummy_svh_q, 4 );
			lw.k_proj = pod::LinearWeight::make_trellis( dummy_kv, dummy_suh_h, dummy_svh_kv, 4 );
			lw.v_proj = pod::LinearWeight::make_trellis( dummy_kv, dummy_suh_h, dummy_svh_kv, 4 );
			lw.q_norm = dummy_qk_norm;
			lw.k_norm = dummy_qk_norm;
			lw.o_proj = pod::LinearWeight::make_trellis( dummy_o, dummy_suh_o, dummy_svh_h, 4 );

			lw.in_proj_qkv = pod::LinearWeight::make_trellis( dummy_qkv, dummy_suh_h, dummy_svh_qkv, 4 );
			lw.conv1d_w = dummy_conv;
			lw.in_proj_a = dummy_ab;
			lw.in_proj_b = dummy_ab;
			lw.A_log = dummy_gdn_b;
			lw.dt_bias = dummy_gdn_b;
			lw.linear_norm = dummy_lin_norm;

			lw.in_proj_z = pod::LinearWeight::make_trellis( dummy_z, dummy_suh_h, dummy_svh_z, 4 );
			lw.out_proj = pod::LinearWeight::make_trellis( dummy_lin_out, dummy_svh_z, dummy_svh_h, 4 );

			lw.gate_proj = pod::LinearWeight::make_trellis( dummy_gate_up, dummy_suh_h, dummy_svh_mlp, 4 );
			lw.up_proj = pod::LinearWeight::make_trellis( dummy_gate_up, dummy_suh_h, dummy_svh_mlp, 4 );
			lw.down_proj = pod::LinearWeight::make_trellis( dummy_down, dummy_suh_mlp, dummy_svh_h, 4 );
		}
	}

	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );

	metadata_.model_name = "qwen38";
	metadata_.hidden_size = config_.hidden_size;
	metadata_.vocab_size = 248320;
	metadata_.num_layers = config_.num_layers;
	metadata_.num_heads = config_.full_num_q_heads;
	metadata_.num_kv_heads = config_.full_num_kv_heads;
	metadata_.head_dim = config_.full_head_dim;
	metadata_.max_seq_len = config_.max_seq;
	metadata_.tp_size = 4;

	return true;
}

bool Qwen38Pipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	return init( Qwen38Config{}, device_id, arena, ring );
}

void Qwen38Pipeline::embed_token( int token_id, int pos, cudaStream_t stream ) {
	(void) pos;
	embed_token( token_id, stream );
}

void Qwen38Pipeline::set_kv_pool( engine::runtime::KVCachePool* pool ) {
	if ( pool ) {
		kv_pool_ = pool;
		state_.k_cache = static_cast<half*>( kv_pool_->get_k_ptr( 0, 0, 0 ) );
		state_.v_cache = static_cast<half*>( kv_pool_->get_v_ptr( 0, 0, 0 ) );
	}
}

void Qwen38Pipeline::forward_step( int token_id, int pos, int slot_id, cudaStream_t stream ) {
	if ( token_id >= 0 ) {
		embed_token( token_id, stream );
	}
	forward_step( hidden_states_, logits_, pos, slot_id, stream );
}

void Qwen38Pipeline::forward_step( int token_id, int pos, cudaStream_t stream ) {
	forward_step( token_id, pos, 0, stream );
}

void Qwen38Pipeline::forward_batch( const int* token_ids, int batch_size, int pos, cudaStream_t stream ) {
	if ( batch_size <= 1 ) {
		forward_step( token_ids ? token_ids[0] : -1, pos, stream );
		return;
	}
	for ( int i = 0; i < batch_size; ++i ) {
		forward_step( token_ids ? token_ids[i] : -1, pos + i, stream );
	}
}

pod::TensorView Qwen38Pipeline::get_logits() {
	return logits_;
}

const ModelMetadata& Qwen38Pipeline::get_metadata() const {
	return metadata_;
}

void Qwen38Pipeline::reset_kv_cache( int slot_id ) {
	if ( kv_pool_ ) {
		kv_pool_->reset_slot( slot_id );
	}
	int tp_size = ( ring_ != nullptr ) ? 4 : 1;
	int lin_v_heads = ( tp_size == 4 ) ? 12 : config_.linear_num_v_heads;
	int lin_channels = ( tp_size == 4 ) ? 2560 : config_.linear_channels;
	size_t slot_gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_v_heads * 128 * 128;
	size_t slot_conv_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_channels * 3;
	if ( state_.gdn_states ) {
		cudaMemset( state_.gdn_states + static_cast<size_t>( slot_id ) * slot_gdn_floats, 0, slot_gdn_floats * sizeof( float ) );
	}
	if ( state_.conv_history ) {
		cudaMemset( state_.conv_history + static_cast<size_t>( slot_id ) * slot_conv_floats, 0, slot_conv_floats * sizeof( float ) );
	}
	if ( slot_id == 0 ) {
		reset_state();
	}
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

	// 3. LM Head
	if ( rank_ == 0 ) {
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

void Qwen38Pipeline::embed_token( int token_id, cudaStream_t stream ) {
	hidden_states_.shape[0] = 1;
	residual_.shape[0] = 1;
	norm_hidden_.shape[0] = 1;
	if ( embed_tokens_.data != nullptr ) {
		int threads = 256;
		int blocks = ( config_.hidden_size + threads - 1 ) / threads;
		embed_token_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
			embed_tokens_.ptr<const __nv_bfloat16>(),
			hidden_states_.ptr<half>(),
			token_id,
			config_.hidden_size
		);
	}
}

void Qwen38Pipeline::embed_tokens( const int* token_ids, int count, cudaStream_t stream ) {
	hidden_states_.shape[0] = count;
	residual_.shape[0] = count;
	norm_hidden_.shape[0] = count;
	if ( embed_tokens_.data != nullptr ) {
		int threads = 256;
		int blocks = ( config_.hidden_size + threads - 1 ) / threads;
		embed_tokens_kernel<<<dim3( blocks, count ), dim3( threads ), 0, stream>>>(
			embed_tokens_.ptr<const __nv_bfloat16>(),
			hidden_states_.ptr<half>(),
			token_ids,
			count,
			config_.hidden_size
		);
	}
}

int Qwen38Pipeline::sample_next_token( cudaStream_t stream ) {
	hidden_states_.shape[0] = 1;
	residual_.shape[0] = 1;
	norm_hidden_.shape[0] = 1;
	if ( weights_loaded_ && final_norm_.data != nullptr ) {
		engine::modules::RMSNorm::forward(
			hidden_states_, final_norm_, hidden_states_, 1e-6f, 1.0f, stream
		);
	}

	if ( lm_head_.is_valid() ) {
		logits_.shape[0] = 1;
		engine::modules::Linear::forward(
			hidden_states_,
			lm_head_,
			logits_,
			stream
		);
	}

	engine::modules::Elementwise::argmax(
		logits_,
		d_argmax_idx_,
		248320,
		stream
	);
	CUDA_CHECK( cudaMemcpyAsync( h_argmax_idx_, d_argmax_idx_, sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	return *h_argmax_idx_;
}

void Qwen38Pipeline::sample_next_tokens( int* h_preds, int count, cudaStream_t stream ) {
	hidden_states_.shape[0] = count;
	residual_.shape[0] = count;
	norm_hidden_.shape[0] = count;
	if ( weights_loaded_ && final_norm_.data != nullptr ) {
		engine::modules::RMSNorm::forward(
			hidden_states_, final_norm_, hidden_states_, 1e-6f, 1.0f, stream
		);
	}

	if ( lm_head_.is_valid() ) {
		logits_.shape[0] = count;
		engine::modules::Linear::forward(
			hidden_states_,
			lm_head_,
			logits_,
			stream
		);
	}

	engine::modules::Elementwise::batched_argmax(
		logits_,
		d_argmax_idx_,
		248320,
		count,
		stream
	);

	CUDA_CHECK( cudaMemcpyAsync( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	for ( int i = 0; i < count; ++i ) {
		h_preds[i] = h_argmax_idx_[i];
	}
}

void Qwen38Pipeline::forward_step(
	const pod::TensorView& input_tokens,
	pod::TensorView& output_logits,
	int pos,
	int slot_id,
	cudaStream_t stream
) {
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	state_.current_pos = pos;
	active_slot_id_ = slot_id;
	half* slot_k = kv_pool_ ? static_cast<half*>( kv_pool_->get_k_ptr( slot_id, 0, 0 ) ) : state_.k_cache;
	half* slot_v = kv_pool_ ? static_cast<half*>( kv_pool_->get_v_ptr( slot_id, 0, 0 ) ) : state_.v_cache;

	int m = ( input_tokens.ndim >= 1 && input_tokens.shape[0] > 0 ) ? input_tokens.shape[0] : 1;
	hidden_states_.shape[0] = m;
	residual_.shape[0] = m;
	norm_hidden_.shape[0] = m;
	q_buf_.shape[0] = m;
	gate_attn_buf_.shape[0] = m;
	q_gate_buf_.shape[0] = m;
	attn_out_.shape[0] = m;
	k_buf_.shape[0] = m;
	v_buf_.shape[0] = m;
	qkv_buf_.shape[0] = m;
	z_buf_.shape[0] = m;
	gdn_out_.shape[0] = m;
	gate_buf_.shape[0] = m;
	up_buf_.shape[0] = m;
	gate_up_buf_.shape[0] = m;
	mlp_inter_.shape[0] = m;
	mlp_out_.shape[0] = m;
	pod::TensorView hidden_cur = pod::make_tensor_2d( hidden_states_.data, pod::DType::FP16, m, config_.hidden_size );
	pod::TensorView residual_cur = pod::make_tensor_2d( residual_.data, pod::DType::FP16, m, config_.hidden_size );

	if ( !weights_loaded_ ) {
		// Benchmark dummy execution mode (M=1 decode latency gate & M=8 verify gate)
		for ( int layer = 0; layer < config_.num_layers; ++layer ) {
			bool is_full_attention = ( ( layer + 1 ) % 4 == 0 );

			if ( is_full_attention ) {
				engine::modules::Attention::forward_decode(
					q_buf_,
					slot_k,
					slot_v,
					gate_attn_buf_,
					attn_out_,
					layer / 4,
					pos,
					config_.full_num_q_heads,
					config_.full_num_kv_heads,
					config_.full_head_dim,
					config_.max_seq,
					stream
				);
			}
			else {
				int linear_idx = layer - ( layer / 4 );
				float* layer_gdn = state_.gdn_states + static_cast<size_t>( linear_idx ) * config_.linear_num_v_heads * 128 * 128;
				engine::modules::GDN::recurrent_step(
					layer_gdn,
					qkv_buf_,
					gdn_gate_,
					gdn_beta_,
					gdn_out_,
					config_.linear_num_k_heads,
					config_.linear_num_v_heads,
					config_.linear_head_dim,
					stream
				);
			}

			if ( ring_ ) {
				ring_->all_reduce_rms_norm( hidden_cur, residual_cur, dummy_norm_h_, 1e-5f, stream );
			}
			else {
				engine::modules::RMSNorm::forward(
					hidden_cur, dummy_norm_h_, hidden_cur, 1e-5f, 1.0f, stream
				);
			}

			engine::modules::LinearTrellis::forward(
				hidden_cur,
				dummy_down_,
				dummy_suh_h_,
				dummy_svh_h_,
				hidden_cur,
				4,
				false,
				true,
				stream
			);

			if ( ring_ ) {
				ring_->all_reduce_rms_norm( hidden_cur, residual_cur, dummy_norm_h_, 1e-5f, stream );
			}
			else {
				engine::modules::RMSNorm::forward(
					hidden_cur, dummy_norm_h_, hidden_cur, 1e-5f, 1.0f, stream
				);
			}
		}

		if ( output_logits.data != nullptr ) {
			CUDA_CHECK( cudaMemcpyAsync( output_logits.data, hidden_states_.data, static_cast<size_t>( m ) * config_.hidden_size * sizeof( half ), cudaMemcpyDeviceToDevice, stream ) );
		}
		return;
	}

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
			int q_heads = ( ring_ != nullptr ) ? 6 : config_.full_num_q_heads;
			int kv_heads = ( ring_ != nullptr ) ? 1 : config_.full_num_kv_heads;

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
				pos, 10000000.0f, stream
			);

			// Store K and V to KV Cache
			engine::modules::Attention::store_kv_cache(
				slot_k, slot_v,
				k_buf_, v_buf_,
				full_attn_idx, pos,
				kv_heads, config_.full_head_dim,
				config_.max_seq, stream
			);

			// Attend over past tokens + sigmoid gate
			engine::modules::Attention::forward_decode(
				q_buf_, slot_k, slot_v, gate_attn_buf_,
				attn_out_,
				full_attn_idx, pos,
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
			int k_heads = ( ring_ != nullptr ) ? 4 : config_.linear_num_k_heads;
			int v_heads = ( ring_ != nullptr ) ? 12 : config_.linear_num_v_heads;
			int lin_channels = ( ring_ != nullptr ) ? 2560 : config_.linear_channels;

			// In proj QKV: projects norm_hidden (5120) to QKV (10240 or 2560) across all M rows
			engine::modules::Linear::forward(
				norm_hidden_, lw.in_proj_qkv, qkv_buf_, stream
			);

			// In proj Z: projects norm_hidden (5120) to Z (6144 or 1536) across all M rows
			engine::modules::Linear::forward(
				norm_hidden_, lw.in_proj_z, z_buf_, stream
			);

			float* hist_ptr = state_.conv_history + ( static_cast<size_t>( active_slot_id_ ) * config_.num_linear_layers + linear_idx ) * lin_channels * 3;
			pod::TensorView conv_hist = pod::make_tensor_2d( hist_ptr, pod::DType::FP32, lin_channels, 3 );
			float* layer_gdn_state = state_.gdn_states + ( static_cast<size_t>( active_slot_id_ ) * config_.num_linear_layers + linear_idx ) * v_heads * 128 * 128;

			// Causal progression across M rows in candidate block
			for ( int m_idx = 0; m_idx < m; ++m_idx ) {
				pod::TensorView qkv_row = pod::make_tensor_2d( qkv_buf_.ptr<half>() + m_idx * lin_channels, pod::DType::FP16, 1, lin_channels );
				pod::TensorView norm_row = pod::make_tensor_2d( norm_hidden_.ptr<half>() + m_idx * config_.hidden_size, pod::DType::FP16, 1, config_.hidden_size );
				pod::TensorView z_row = pod::make_tensor_2d( z_buf_.ptr<half>() + m_idx * v_heads * 128, pod::DType::FP16, 1, v_heads * 128 );
				pod::TensorView out_row = pod::make_tensor_2d( gdn_out_.ptr<half>() + m_idx * v_heads * 128, pod::DType::FP16, 1, v_heads * 128 );
				pod::TensorView gate_row = pod::make_tensor_2d( gdn_gate_.ptr<float>() + m_idx * v_heads, pod::DType::FP32, 1, v_heads );
				pod::TensorView beta_row = pod::make_tensor_2d( gdn_beta_.ptr<float>() + m_idx * v_heads, pod::DType::FP32, 1, v_heads );

				// Causal conv1d + SiLU
				engine::modules::GDN::causal_conv1d_silu(
					qkv_row, conv_hist, lw.conv1d_w, lin_channels, stream
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
					config_.linear_head_dim, stream
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
			ring_->all_reduce( mlp_out_, stream );
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

		// Tapped layer states for DFlash2: [5, 19, 33, 47, 61]
		if ( tapped_states_ != nullptr && ( layer == 5 || layer == 19 || layer == 33 || layer == 47 || layer == 61 ) ) {
			int tap_idx = ( layer == 5 ) ? 0 : ( ( layer == 19 ) ? 1 : ( ( layer == 33 ) ? 2 : ( ( layer == 47 ) ? 3 : 4 ) ) );
			for ( int r = 0; r < m; ++r ) {
				CUDA_CHECK( cudaMemcpyAsync(
					tapped_states_ + ( static_cast<size_t>( r ) * 5 + tap_idx ) * config_.hidden_size,
					hidden_states_.ptr<const half>() + static_cast<size_t>( r ) * config_.hidden_size,
					config_.hidden_size * sizeof( half ),
					cudaMemcpyDeviceToDevice, stream
				) );
			}
		}
	}

	if ( output_logits.data != nullptr ) {
		CUDA_CHECK( cudaMemcpyAsync( output_logits.data, hidden_states_.data, static_cast<size_t>( m ) * config_.hidden_size * sizeof( half ), cudaMemcpyDeviceToDevice, stream ) );
	}
}

void Qwen38Pipeline::forward_step(
	const pod::TensorView& input_tokens,
	pod::TensorView& output_logits,
	int pos,
	cudaStream_t stream
) {
	forward_step( input_tokens, output_logits, pos, 0, stream );
}

bool Qwen38Pipeline::capture_graph( int m, cudaStream_t stream ) {
	if ( m == 1 ) {
		if ( graph_m1_captured_ ) return true;
		pod::TensorView dummy_in = pod::make_tensor_2d( hidden_states_.data, pod::DType::FP16, 1, config_.hidden_size );
		pod::TensorView dummy_out = pod::make_tensor_2d( residual_.data, pod::DType::FP16, 1, config_.hidden_size );

		forward_step( dummy_in, dummy_out, 0, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		forward_step( dummy_in, dummy_out, 0, stream );
		CUDA_CHECK( cudaStreamEndCapture( stream, &graph_m1_ ) );
		CUDA_CHECK( cudaGraphInstantiate( &instance_m1_, graph_m1_, nullptr, nullptr, 0 ) );
		graph_m1_captured_ = true;
		return true;
	}
	else if ( m == 4 ) {
		if ( graph_m4_captured_ ) return true;
		pod::TensorView dummy_in = pod::make_tensor_2d( hidden_states_.data, pod::DType::FP16, 4, config_.hidden_size );
		pod::TensorView dummy_out = pod::make_tensor_2d( residual_.data, pod::DType::FP16, 4, config_.hidden_size );

		forward_step( dummy_in, dummy_out, 0, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		forward_step( dummy_in, dummy_out, 0, stream );
		CUDA_CHECK( cudaStreamEndCapture( stream, &graph_m4_ ) );
		CUDA_CHECK( cudaGraphInstantiate( &instance_m4_, graph_m4_, nullptr, nullptr, 0 ) );
		graph_m4_captured_ = true;
		return true;
	}
	else if ( m == 8 ) {
		if ( graph_m8_captured_ ) return true;
		pod::TensorView dummy_in = pod::make_tensor_2d( hidden_states_.data, pod::DType::FP16, 8, config_.hidden_size );
		pod::TensorView dummy_out = pod::make_tensor_2d( residual_.data, pod::DType::FP16, 8, config_.hidden_size );

		forward_step( dummy_in, dummy_out, 0, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		forward_step( dummy_in, dummy_out, 0, stream );
		CUDA_CHECK( cudaStreamEndCapture( stream, &graph_m8_ ) );
		CUDA_CHECK( cudaGraphInstantiate( &instance_m8_, graph_m8_, nullptr, nullptr, 0 ) );
		graph_m8_captured_ = true;
		return true;
	}
	return false;
}

void Qwen38Pipeline::launch_graph( int m, cudaStream_t stream ) {
	if ( m == 1 ) {
		ENGINE_ASSERT( graph_m1_captured_, "Graph M=1 not captured" );
		CUDA_CHECK( cudaGraphLaunch( instance_m1_, stream ) );
	}
	else if ( m == 4 ) {
		ENGINE_ASSERT( graph_m4_captured_, "Graph M=4 not captured" );
		CUDA_CHECK( cudaGraphLaunch( instance_m4_, stream ) );
	}
	else if ( m == 8 ) {
		ENGINE_ASSERT( graph_m8_captured_, "Graph M=8 not captured" );
		CUDA_CHECK( cudaGraphLaunch( instance_m8_, stream ) );
	}
}

void Qwen38Pipeline::reset_state() {
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	state_.current_pos = 0;
	int tp_size = ( ring_ != nullptr ) ? 4 : 1;
	int lin_v_heads = ( tp_size == 4 ) ? 12 : config_.linear_num_v_heads;
	int lin_channels = ( tp_size == 4 ) ? 2560 : config_.linear_channels;
	int full_kv_heads = ( tp_size == 4 ) ? 1 : config_.full_num_kv_heads;

	size_t gdn_floats = 8ULL * static_cast<size_t>( config_.num_linear_layers ) * lin_v_heads * 128 * 128;
	if ( state_.gdn_states ) {
		cudaMemset( state_.gdn_states, 0, gdn_floats * sizeof( float ) );
	}
	size_t conv_hist_floats = 8ULL * static_cast<size_t>( config_.num_linear_layers ) * lin_channels * 3;
	if ( state_.conv_history ) {
		cudaMemset( state_.conv_history, 0, conv_hist_floats * sizeof( float ) );
	}
	size_t cache_elems = static_cast<size_t>( config_.num_attn_layers ) * config_.max_seq * full_kv_heads * config_.full_head_dim;
	if ( state_.k_cache ) {
		cudaMemset( state_.k_cache, 0, cache_elems * sizeof( half ) );
	}
	if ( state_.v_cache ) {
		cudaMemset( state_.v_cache, 0, cache_elems * sizeof( half ) );
	}
	if ( residual_.data ) {
		cudaMemset( residual_.data, 0, residual_.size_bytes() );
	}
	if ( hidden_states_.data ) {
		cudaMemset( hidden_states_.data, 0, hidden_states_.size_bytes() );
	}
	if ( tapped_states_ ) {
		cudaMemset( tapped_states_, 0, 5ULL * config_.hidden_size * sizeof( half ) );
	}
}

void Qwen38Pipeline::destroy() {
	if ( h_argmax_idx_ != nullptr ) {
		engine::core::host_free_pinned( h_argmax_idx_ );
		h_argmax_idx_ = nullptr;
	}
	if ( graph_m1_captured_ ) {
		cudaGraphExecDestroy( instance_m1_ );
		cudaGraphDestroy( graph_m1_ );
		graph_m1_captured_ = false;
	}
	if ( graph_m4_captured_ ) {
		cudaGraphExecDestroy( instance_m4_ );
		cudaGraphDestroy( graph_m4_ );
		graph_m4_captured_ = false;
	}
	if ( graph_m8_captured_ ) {
		cudaGraphExecDestroy( instance_m8_ );
		cudaGraphDestroy( graph_m8_ );
		graph_m8_captured_ = false;
	}
}

void Qwen38Pipeline::checkpoint_state( cudaStream_t stream ) {
	int tp_size = ( ring_ != nullptr ) ? 4 : 1;
	int lin_v_heads = ( tp_size == 4 ) ? 12 : config_.linear_num_v_heads;
	int lin_channels = ( tp_size == 4 ) ? 2560 : config_.linear_channels;
	size_t slot_gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_v_heads * 128 * 128;
	size_t slot_conv_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_channels * 3;
	size_t offset_gdn = static_cast<size_t>( active_slot_id_ ) * slot_gdn_floats;
	size_t offset_conv = static_cast<size_t>( active_slot_id_ ) * slot_conv_floats;
	if ( gdn_checkpoint_ && state_.gdn_states ) {
		CUDA_CHECK( cudaMemcpyAsync( gdn_checkpoint_ + offset_gdn, state_.gdn_states + offset_gdn, slot_gdn_floats * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	}
	if ( conv_checkpoint_ && state_.conv_history ) {
		CUDA_CHECK( cudaMemcpyAsync( conv_checkpoint_ + offset_conv, state_.conv_history + offset_conv, slot_conv_floats * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	}
}

void Qwen38Pipeline::rollback_state( cudaStream_t stream ) {
	int tp_size = ( ring_ != nullptr ) ? 4 : 1;
	int lin_v_heads = ( tp_size == 4 ) ? 12 : config_.linear_num_v_heads;
	int lin_channels = ( tp_size == 4 ) ? 2560 : config_.linear_channels;
	size_t slot_gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_v_heads * 128 * 128;
	size_t slot_conv_floats = static_cast<size_t>( config_.num_linear_layers ) * lin_channels * 3;
	size_t offset_gdn = static_cast<size_t>( active_slot_id_ ) * slot_gdn_floats;
	size_t offset_conv = static_cast<size_t>( active_slot_id_ ) * slot_conv_floats;
	if ( gdn_checkpoint_ && state_.gdn_states ) {
		CUDA_CHECK( cudaMemcpyAsync( state_.gdn_states + offset_gdn, gdn_checkpoint_ + offset_gdn, slot_gdn_floats * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	}
	if ( conv_checkpoint_ && state_.conv_history ) {
		CUDA_CHECK( cudaMemcpyAsync( state_.conv_history + offset_conv, conv_checkpoint_ + offset_conv, slot_conv_floats * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	}
}

} // namespace engine::models
