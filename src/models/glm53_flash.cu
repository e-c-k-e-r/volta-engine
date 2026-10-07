#include "engine/models/glm53_flash.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/gdn.h"
#include "engine/modules/embedding.h"
#include "engine/spec/dflash.h"
#include "trellis_context.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

namespace engine::models {

namespace impl {

pod::TensorView load_router_weights_glm(
	engine::loader::SafeTensorsRepo& repo,
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	int shard_idx = -1;
	const auto* meta = repo.find( name, &shard_idx );
	if ( !meta || shard_idx < 0 ) return pod::TensorView{};

	int num_experts = meta->shape[0]; // 288
	int hidden_size = meta->shape[1]; // 4096
	const half* src = static_cast<const half*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard_idx ), meta ) );
	if ( !src ) return pod::TensorView{};

	int shape[2] = { hidden_size, num_experts };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	if ( tv.data != nullptr ) {
		std::vector<half> h_transposed( static_cast<size_t>( hidden_size ) * num_experts );
		for ( int e = 0; e < num_experts; ++e ) {
			for ( int d = 0; d < hidden_size; ++d ) {
				h_transposed[d * num_experts + e] = src[e * hidden_size + d];
			}
		}
		CUDA_CHECK( cudaMemcpy( tv.data, h_transposed.data(), h_transposed.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
	}
	return tv;
}

} // namespace impl

GLM53Pipeline::GLM53Pipeline() = default;

GLM53Pipeline::~GLM53Pipeline() {
	destroy();
}

bool GLM53Pipeline::init(
	const GLM53Config& config,
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

	const int R = pod::MAX_ROWS;
	const int S = config_.max_slots;
	const int E_local = config_.num_experts / config_.tp_size;

	// 1. Router outputs + routed-MoE dispatch scratch
	state_.d_topk_indices = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.top_k * sizeof( int ) ) );
	state_.d_topk_weights = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.top_k * sizeof( half ) ) );
	state_.moe = engine::modules::MoE::alloc_scratch( arena, E_local, config_.top_k, config_.hidden_size, config_.intermediate_size, R );

	// KDA recurrent state + conv history per slot: [S, 45, ...] FP32, plus one-slot checkpoints
	state_.layer_kda_floats = static_cast<size_t>( config_.kda_heads ) * config_.kda_head_dim * config_.kda_head_dim;
	state_.layer_conv_floats = static_cast<size_t>( config_.kda_channels ) * 3;
	const size_t slot_kda = state_.layer_kda_floats * config_.num_layers;
	const size_t slot_conv = state_.layer_conv_floats * config_.num_layers;
	state_.kda_states = static_cast<float*>( engine::core::arena_alloc( arena, S * slot_kda * sizeof( float ) ) );
	state_.conv_history = static_cast<float*>( engine::core::arena_alloc( arena, S * slot_conv * sizeof( float ) ) );
	state_.kda_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, slot_kda * sizeof( float ) ) );
	state_.conv_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, slot_conv * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.kda_states, 0, S * slot_kda * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.conv_history, 0, S * slot_conv * sizeof( float ) ) );

	// 2. Activations: [R, ...] row buffers, viewed as [m, ...] per forward
	int rows_shape[2] = { R, config_.hidden_size };
	embed_rows_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rows_shape, 2 );
	d_token_ids_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
	d_rows_ = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );
	int h_shape[2] = { R, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	int streams_shape[2] = { 4 * R, config_.hidden_size };
	hidden_streams_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, streams_shape, 2 );
	mhc_collapsed_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	int rms_shape[1] = { R };
	mhc_inv_rms_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, rms_shape, 1 );

	int mix_shape[1] = { 24 * R };
	mhc_mix_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, mix_shape, 1 );

	int hc_shape[1] = { 4 * R };
	mhc_pre_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, hc_shape, 1 );
	mhc_post_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, hc_shape, 1 );

	int comb_shape[2] = { 4 * R, 4 };
	mhc_comb_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, comb_shape, 2 );

	// 3. MLA KV cache pool [S, 11, max_seq, 16 * 256] FP16 & execution buffers
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	kv_pool_.init( rank_, arena, S, config_.max_seq, config_.num_mla_layers, 16, 256 );

	int q_lora_shape[2] = { R, 1536 };
	q_lora_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_lora_shape, 2 );
	q_lora_norm_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_lora_shape, 2 );

	int q_shape[2] = { R, 4096 };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );

	int kv_lora_shape[2] = { R, 512 };
	kv_lora_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_lora_shape, 2 );
	kv_lora_norm_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_lora_shape, 2 );

	int kv_head_shape[2] = { R, 4096 };
	k_head_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );
	v_head_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );
	mla_heads_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );

	// KDA linear attention buffers
	int qkv_shape[2] = { R, 6144 };
	qkv_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );

	int fa_shape[2] = { R, 128 };
	f_a_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, fa_shape, 2 );
	g_a_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, fa_shape, 2 );

	int beta_shape[2] = { R, 16 };
	beta_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, beta_shape, 2 );

	int gdec_shape[2] = { R, 2048 };
	g_decay_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdec_shape, 2 );

	int z_shape[2] = { R, 2048 };
	z_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	attn_head_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );

	// 4. Dense / shared-expert FFN buffers
	int moe_inter_shape[2] = { R, config_.dense_intermediate_size };
	moe_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_inter_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_expert_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	local_moe_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	ffn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	// 5. Logits & argmax (rank 0 owns the unsharded LM head)
	if ( rank_ == 0 ) {
		int logit_shape[2] = { pod::MAX_BATCH, config_.vocab_size };
		logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, logit_shape, 2 );
		d_argmax_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, pod::MAX_BATCH * sizeof( int ) ) );
		h_argmax_idx_ = static_cast<int*>( engine::core::host_alloc_pinned( pod::MAX_BATCH * sizeof( int ) ) );
		const int B = pod::MAX_BATCH;
		const int splits = ( config_.vocab_size + kTopkTile - 1 ) / kTopkTile;
		const size_t per = static_cast<size_t>( B ) * splits * 16;
		samp_part_v_ = static_cast<float*>( engine::core::arena_alloc( arena, per * sizeof( float ) ) );
		samp_part_i_ = static_cast<int*>( engine::core::arena_alloc( arena, per * sizeof( int ) ) );
		samp_counters_ = static_cast<int*>( engine::core::arena_alloc( arena, B * sizeof( int ) ) );
		d_samp_vals_ = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( B ) * 16 * sizeof( float ) ) );
		d_samp_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( B ) * 16 * sizeof( int ) ) );
		CUDA_CHECK( cudaMemset( samp_counters_, 0, B * sizeof( int ) ) );
	}

	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );

	metadata_.model_name = "glm53";
	metadata_.hidden_size = config_.hidden_size;
	metadata_.vocab_size = config_.vocab_size;
	metadata_.num_layers = config_.num_layers;
	metadata_.num_heads = 16;
	metadata_.num_kv_heads = 16;
	metadata_.head_dim = 256;
	metadata_.max_seq_len = config_.max_seq;
	metadata_.tp_size = config_.tp_size;
	metadata_.max_slots = S;
	metadata_.max_rows = R;
	metadata_.eos_ids[0] = 154820;
	metadata_.eos_ids[1] = 154827;
	metadata_.eos_ids[2] = 154829;

	return true;
}

bool GLM53Pipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	return init( GLM53Config{}, device_id, arena, ring );
}

void GLM53Pipeline::reset_state( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "GLM53Pipeline::reset_state: slot out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	kv_pool_.reset_slot( slot_id );
	const size_t slot_kda = state_.layer_kda_floats * config_.num_layers;
	const size_t slot_conv = state_.layer_conv_floats * config_.num_layers;
	CUDA_CHECK( cudaMemset( state_.kda_states + slot_id * slot_kda, 0, slot_kda * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.conv_history + slot_id * slot_conv, 0, slot_conv * sizeof( float ) ) );
}

void GLM53Pipeline::checkpoint_state( int slot_id, cudaStream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "GLM53Pipeline::checkpoint_state: slot out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	const size_t slot_kda = state_.layer_kda_floats * config_.num_layers;
	const size_t slot_conv = state_.layer_conv_floats * config_.num_layers;
	CUDA_CHECK( cudaMemcpyAsync( state_.kda_checkpoint, state_.kda_states + slot_id * slot_kda, slot_kda * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( state_.conv_checkpoint, state_.conv_history + slot_id * slot_conv, slot_conv * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
}

void GLM53Pipeline::rollback_state( int slot_id, cudaStream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "GLM53Pipeline::rollback_state: slot out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	const size_t slot_kda = state_.layer_kda_floats * config_.num_layers;
	const size_t slot_conv = state_.layer_conv_floats * config_.num_layers;
	CUDA_CHECK( cudaMemcpyAsync( state_.kda_states + slot_id * slot_kda, state_.kda_checkpoint, slot_kda * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( state_.conv_history + slot_id * slot_conv, state_.conv_checkpoint, slot_conv * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
}

bool GLM53Pipeline::load_weights( const char* model_dir, cudaStream_t stream ) {
	size_t bytes_before = arena_->offset_bytes;
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) {
		std::printf( "    [loader] FAILED to open model directory %s\n", model_dir );
		return false;
	}

	int missing_count = 0;

	// 1. Embed tokens (rank 0 hosts full vocabulary embedding and LM head)
	if ( rank_ == 0 ) {
		embed_tokens_ = repo.load( "model.language_model.embed_tokens.weight", *arena_, stream, pod::DType::BF16 );
		if ( !embed_tokens_.data ) { std::printf( "[RANK %d] MISSING: embed_tokens\n", rank_ ); ++missing_count; }
		final_norm_ = repo.load( "model.language_model.norm.weight", *arena_, stream, pod::DType::FP16 );
		if ( !final_norm_.data ) { std::printf( "[RANK %d] MISSING: norm\n", rank_ ); ++missing_count; }
		lm_head_ = repo.load_linear( "lm_head", *arena_, 6, stream );
		if ( !lm_head_.is_valid() ) { std::printf( "[RANK %d] MISSING: lm_head\n", rank_ ); ++missing_count; }
	}

	// 2. Per-layer weights across 45 layers
	char name_buf[256];
	int experts_per_rank = config_.num_experts / config_.tp_size; // 288 / 4 = 72
	int start_expert = rank_ * experts_per_rank;

	static constexpr int GLM_QKV_BLOCKS[3] = { 512, 512, 512 };
	static constexpr int GLM_QKV_RANK_BLOCKS[3] = { 128, 128, 128 };
	static constexpr int GLM_CONV_CHANNELS[3] = { 8192, 8192, 8192 };
	static constexpr int GLM_CONV_RANK_CHANNELS[3] = { 2048, 2048, 2048 };

	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.is_moe = ( L >= 3 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.input_layernorm.weight", L );
		lw.input_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.post_attention_layernorm.weight", L );
		lw.post_attn_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.qkv_proj", L );
		lw.qkv = repo.load_grouped_trellis_slice( name_buf, *arena_, GLM_QKV_BLOCKS, GLM_QKV_RANK_BLOCKS, 3, rank_, stream );
		lw.is_linear_attn = lw.qkv.is_valid();

		if ( lw.is_linear_attn ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.conv1d.weight", L );
			lw.conv1d_weights = repo.load_grouped_slice( name_buf, *arena_, GLM_CONV_CHANNELS, GLM_CONV_RANK_CHANNELS, 3, rank_, 0, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.A_log", L );
			lw.A_log = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.dt_bias", L );
			lw.dt_bias = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.b_proj.weight", L );
			lw.b_proj = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.f_a_proj.weight", L );
			lw.f_a_proj = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.f_b_proj.weight", L );
			lw.f_b_proj = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.g_a_proj.weight", L );
			lw.g_a_proj = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.g_b_proj.weight", L );
			lw.g_b_proj = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_norm.weight", L );
			lw.o_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj", L );
			lw.o_proj = repo.load_linear_slice( name_buf, *arena_, 5, 0, rank_, config_.tp_size, stream );
		}

		bool is_mla = ( ( L % 4 == 3 ) && L < 44 );
		lw.is_mla = is_mla;
		if ( lw.is_mla ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_a_proj", L );
			lw.q_a = repo.load_linear( name_buf, *arena_, 5, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_a_layernorm.weight", L );
			lw.q_a_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_b_proj", L );
			lw.q_b = repo.load_linear_slice( name_buf, *arena_, 5, 1, rank_, config_.tp_size, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.kv_a_proj_with_mqa", L );
			lw.kv_a = repo.load_linear( name_buf, *arena_, 5, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.kv_a_layernorm.weight", L );
			lw.kv_a_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.kv_b_proj.weight", L );
			lw.kv_b_proj = repo.load_slice( name_buf, *arena_, 0, rank_, config_.tp_size, stream, pod::DType::FP16 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj", L );
			lw.o_proj = repo.load_linear_slice( name_buf, *arena_, 5, 0, rank_, config_.tp_size, stream );
		}

		if ( lw.is_moe ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate.weight", L );
			lw.router_weights = impl::load_router_weights_glm( repo, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate.e_score_correction_bias", L );
			lw.e_score_bias = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			// 72 local experts for this rank
			auto& mw = lw.moe_weights;
			mw.type = pod::QuantType::TRELLIS;
			mw.num_local_experts = experts_per_rank;
			for ( int e = 0; e < experts_per_rank; ++e ) {
				int gid = start_expert + e;

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, gid );
				mw.gate[e] = repo.load_linear( name_buf, *arena_, 4, stream );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, gid );
				mw.up[e] = repo.load_linear( name_buf, *arena_, 4, stream );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, gid );
				mw.down[e] = repo.load_linear( name_buf, *arena_, 4, stream );
			}
			engine::modules::MoE::init_device_ptrs( mw, *arena_, stream );

			// Shared experts
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.gate_proj", L );
			lw.shared_expert.gate = repo.load_linear( name_buf, *arena_, 4, stream );
			lw.has_shared_expert = lw.shared_expert.gate.is_valid();

			if ( lw.has_shared_expert ) {
				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.up_proj", L );
				lw.shared_expert.up = repo.load_linear( name_buf, *arena_, 4, stream );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.down_proj", L );
				lw.shared_expert.down = repo.load_linear( name_buf, *arena_, 4, stream );
			}
		}
		else {
			// Dense layer
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj", L );
			lw.dense_mlp.gate = repo.load_linear( name_buf, *arena_, 4, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj", L );
			lw.dense_mlp.up = repo.load_linear( name_buf, *arena_, 4, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj", L );
			lw.dense_mlp.down = repo.load_linear( name_buf, *arena_, 4, stream );
		}

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_attn_base", L );
		lw.hc_attn_base = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_attn_scale", L );
		lw.hc_attn_scale = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_attn_fn", L );
		lw.hc_attn_fn = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_ffn_base", L );
		lw.hc_ffn_base = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_ffn_scale", L );
		lw.hc_ffn_scale = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.hc_ffn_fn", L );
		lw.hc_ffn_fn = repo.load( name_buf, *arena_, stream, pod::DType::FP32 );

		if ( !lw.input_norm.data ) { std::printf( "[RANK %d L%d] MISSING: input_norm\n", rank_, L ); ++missing_count; }
		if ( !lw.post_attn_norm.data ) { std::printf( "[RANK %d L%d] MISSING: post_attn_norm\n", rank_, L ); ++missing_count; }
		if ( lw.is_linear_attn ) {
			if ( !lw.qkv.is_valid() ) { std::printf( "[RANK %d L%d] MISSING: qkv\n", rank_, L ); ++missing_count; }
			if ( !lw.conv1d_weights.data ) { std::printf( "[RANK %d L%d] MISSING: conv1d_weights\n", rank_, L ); ++missing_count; }
			if ( !lw.A_log.data ) { std::printf( "[RANK %d L%d] MISSING: A_log\n", rank_, L ); ++missing_count; }
			if ( !lw.dt_bias.data ) { std::printf( "[RANK %d L%d] MISSING: dt_bias\n", rank_, L ); ++missing_count; }
			if ( !lw.b_proj.data ) { std::printf( "[RANK %d L%d] MISSING: b_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.f_a_proj.data ) { std::printf( "[RANK %d L%d] MISSING: f_a_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.f_b_proj.data ) { std::printf( "[RANK %d L%d] MISSING: f_b_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.g_a_proj.data ) { std::printf( "[RANK %d L%d] MISSING: g_a_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.g_b_proj.data ) { std::printf( "[RANK %d L%d] MISSING: g_b_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.o_norm.data ) { std::printf( "[RANK %d L%d] MISSING: o_norm\n", rank_, L ); ++missing_count; }
			if ( !lw.o_proj.is_valid() ) { std::printf( "[RANK %d L%d] MISSING: o_proj (linear)\n", rank_, L ); ++missing_count; }
		}
		if ( lw.is_mla ) {
			if ( !lw.q_a.is_valid() ) { std::printf( "[RANK %d L%d] MISSING: q_a\n", rank_, L ); ++missing_count; }
			if ( !lw.q_a_norm.data ) { std::printf( "[RANK %d L%d] MISSING: q_a_norm\n", rank_, L ); ++missing_count; }
			if ( !lw.q_b.is_valid() ) { std::printf( "[RANK %d L%d] MISSING: q_b\n", rank_, L ); ++missing_count; }
			if ( !lw.kv_a.is_valid() ) { std::printf( "[RANK %d L%d] MISSING: kv_a\n", rank_, L ); ++missing_count; }
			if ( !lw.kv_a_norm.data ) { std::printf( "[RANK %d L%d] MISSING: kv_a_norm\n", rank_, L ); ++missing_count; }
			if ( !lw.kv_b_proj.data ) { std::printf( "[RANK %d L%d] MISSING: kv_b_proj\n", rank_, L ); ++missing_count; }
			if ( !lw.o_proj.is_valid() ) { std::printf( "[RANK %d L%d] MISSING: o_proj (mla)\n", rank_, L ); ++missing_count; }
		}
		if ( lw.is_moe ) {
			if ( !lw.router_weights.data ) { std::printf( "[RANK %d L%d] MISSING: router_weights\n", rank_, L ); ++missing_count; }
			if ( !lw.e_score_bias.data ) { std::printf( "[RANK %d L%d] MISSING: e_score_bias\n", rank_, L ); ++missing_count; }
			int expert_miss = 0;
			for ( int e = 0; e < experts_per_rank; ++e ) {
				const auto& mw = lw.moe_weights;
				if ( !mw.gate[e].is_valid() || !mw.up[e].is_valid() || !mw.down[e].is_valid() ) {
					++expert_miss;
				}
			}
			if ( expert_miss > 0 ) {
				std::printf( "[RANK %d L%d] MISSING: %d/%d local experts invalid!\n", rank_, L, expert_miss, experts_per_rank );
				missing_count += expert_miss;
			}
			if ( !lw.shared_expert.gate.is_valid() || !lw.shared_expert.up.is_valid() || !lw.shared_expert.down.is_valid() ) {
				std::printf( "[RANK %d L%d] MISSING: shared_expert\n", rank_, L );
				++missing_count;
			}
		}
		else {
			if ( !lw.dense_mlp.gate.is_valid() || !lw.dense_mlp.up.is_valid() || !lw.dense_mlp.down.is_valid() ) {
				std::printf( "[RANK %d L%d] MISSING: dense_mlp\n", rank_, L );
				++missing_count;
			}
		}
		if ( !lw.hc_attn_base.data ) { std::printf( "[RANK %d L%d] MISSING: hc_attn_base\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_attn_scale.data ) { std::printf( "[RANK %d L%d] MISSING: hc_attn_scale\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_attn_fn.data ) { std::printf( "[RANK %d L%d] MISSING: hc_attn_fn\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_ffn_base.data ) { std::printf( "[RANK %d L%d] MISSING: hc_ffn_base\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_ffn_scale.data ) { std::printf( "[RANK %d L%d] MISSING: hc_ffn_scale\n", rank_, L ); ++missing_count; }
		if ( !lw.hc_ffn_fn.data ) { std::printf( "[RANK %d L%d] MISSING: hc_ffn_fn\n", rank_, L ); ++missing_count; }
	}

	size_t bytes_after = arena_->offset_bytes;
	if ( missing_count > 0 || rank_ == 0 ) {
		std::printf( "    [RANK %d] load_weights complete: missing_count=%d | arena: before=%.3f GiB, after=%.3f GiB, loaded=%.3f GiB\n",
			rank_, missing_count,
			bytes_before / ( 1024.0 * 1024.0 * 1024.0 ),
			bytes_after / ( 1024.0 * 1024.0 * 1024.0 ),
			( bytes_after - bytes_before ) / ( 1024.0 * 1024.0 * 1024.0 ) );
	}

	if ( stream ) {
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
	}
	repo.close();
	weights_loaded_ = true;
	return true;
}

void GLM53Pipeline::embed_tokens( const int* token_ids, int count, cudaStream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_ROWS, "GLM53Pipeline::embed_tokens: count out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	set_rows( count );
	// Rank 0 hosts the vocabulary table; other ranks contribute zeros to the all-reduce broadcast
	if ( rank_ == 0 ) {
		CUDA_CHECK( cudaMemcpyAsync( d_token_ids_, token_ids, count * sizeof( int ), cudaMemcpyHostToDevice, stream ) );
		engine::modules::Embedding::forward( embed_tokens_, d_token_ids_, count, embed_rows_, stream );
	}
	else {
		CUDA_CHECK( cudaMemsetAsync( embed_rows_.data, 0, embed_rows_.size_bytes(), stream ) );
	}
	if ( ring_ ) {
		ring_->all_reduce( embed_rows_, stream );
	}
}

void GLM53Pipeline::sample_tokens( int* out_preds, int count, cudaStream_t stream ) {
	ENGINE_ASSERT( rank_ == 0, "GLM53Pipeline::sample_tokens: logits live on rank 0" );
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "GLM53Pipeline::sample_tokens: count exceeds last forward rows" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax( logits_, d_argmax_idx_, config_.vocab_size, count, stream );
	CUDA_CHECK( cudaMemcpyAsync( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	std::memcpy( out_preds, h_argmax_idx_, count * sizeof( int ) );
}

void GLM53Pipeline::set_rows( int m ) {
	hidden_states_.shape[0] = m;
	norm_hidden_.shape[0] = m;
	mhc_collapsed_.shape[0] = m;
	attn_out_.shape[0] = m;
	q_lora_buf_.shape[0] = m;
	q_lora_norm_buf_.shape[0] = m;
	q_buf_.shape[0] = m;
	kv_lora_buf_.shape[0] = m;
	kv_lora_norm_buf_.shape[0] = m;
	k_head_buf_.shape[0] = m;
	v_head_buf_.shape[0] = m;
	mla_heads_out_.shape[0] = m;
	qkv_buf_.shape[0] = m;
	f_a_buf_.shape[0] = m;
	g_a_buf_.shape[0] = m;
	beta_buf_.shape[0] = m;
	g_decay_buf_.shape[0] = m;
	z_buf_.shape[0] = m;
	attn_head_out_.shape[0] = m;
	moe_gate_buf_.shape[0] = m;
	moe_up_buf_.shape[0] = m;
	moe_inter_buf_.shape[0] = m;
	moe_expert_out_.shape[0] = m;
	local_moe_out_.shape[0] = m;
	ffn_out_.shape[0] = m;
	embed_rows_.shape[0] = m;
}

bool GLM53Pipeline::topk_local( float* out_vals, int* out_ids, int count, cudaStream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_BATCH, "GLM53Pipeline::topk_local: count out of range" );
	if ( rank_ != 0 ) {
		// The vocabulary is not sharded: other ranks contribute nothing to the cluster-wide merge
		for ( int i = 0; i < count * 16; ++i ) {
			out_vals[i] = -INFINITY;
			out_ids[i] = 0;
		}
		return true;
	}
	ENGINE_ASSERT( count <= last_rows_, "GLM53Pipeline::topk_local: count exceeds last forward rows" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	engine::spec::dflash2_topk_shard(
		logits_.ptr<const half>(), count, config_.vocab_size, config_.vocab_size, 16,
		d_samp_vals_, d_samp_idx_, 0, samp_part_v_, samp_part_i_, samp_counters_, stream );
	CUDA_CHECK( cudaMemcpyAsync( out_vals, d_samp_vals_, static_cast<size_t>( count ) * 16 * sizeof( float ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( out_ids, d_samp_idx_, static_cast<size_t>( count ) * 16 * sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	return true;
}

void GLM53Pipeline::forward( const pod::BatchRows& rows, cudaStream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "GLM53Pipeline::forward: weights not loaded" );
	ENGINE_ASSERT( rows.count >= 1 && rows.count <= pod::MAX_ROWS, "GLM53Pipeline::forward: row count out of range" );
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	for ( int r = 0; r < rows.count; ++r ) {
		ENGINE_ASSERT( rows.slot[r] >= 0 && rows.slot[r] < config_.max_slots, "GLM53Pipeline::forward: slot out of range" );
		ENGINE_ASSERT( rows.pos[r] >= 0 && rows.pos[r] < config_.max_seq, "GLM53Pipeline::forward: position out of range" );
	}
	const int m = rows.count;
	// Logits selection applies to eager prefill chunks only (decode graphs are keyed on m alone)
	lm_mode_ = ( m > pod::MAX_BATCH ) ? rows.logits_mode : pod::BatchRows::LOGITS_ALL;
	last_rows_ = ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) ? 0 : ( lm_mode_ == pod::BatchRows::LOGITS_LAST ? 1 : m );
	set_rows( m );
	// Pageable source: the copy is staged before return, so `rows` may be a temporary
	CUDA_CHECK( cudaMemcpyAsync( d_rows_, &rows, sizeof( pod::BatchRows ), cudaMemcpyHostToDevice, stream ) );
	graphs_.run( m, stream, [&]( cudaStream_t s ) { forward_body( m, s ); } );
}

void GLM53Pipeline::forward_body( int m, cudaStream_t stream ) {
	// Slot-0 base pointers; slot and position of each row are resolved on device from d_rows_
	half* slot_k = static_cast<half*>( kv_pool_.get_k_ptr( 0, 0, 0 ) );
	half* slot_v = static_cast<half*>( kv_pool_.get_v_ptr( 0, 0, 0 ) );
	const size_t kv_slot_stride = kv_pool_.slot_stride_bytes / sizeof( half );
	const pod::SlotRef kv_ref{ d_rows_, 0, kv_slot_stride };
	const size_t slot_kda_floats = state_.layer_kda_floats * config_.num_layers;
	const size_t slot_conv_floats = state_.layer_conv_floats * config_.num_layers;
	const int H = config_.hidden_size;

	// mHC: expand every row's embedding into its 4 residual streams
	engine::modules::HyperConnection::init_streams( embed_rows_, hidden_streams_, H, stream, m );

	// Routed experts are partitioned across ranks (TP=4)
	int experts_per_rank = config_.num_experts / config_.tp_size; // 72
	int start_expert = rank_ * experts_per_rank;
	int end_expert = start_expert + experts_per_rank;

	for ( int layer = 0; layer < config_.num_layers; ++layer ) {
		auto& lw = layers_[layer];

		if ( lw.hc_attn_fn.data != nullptr ) {
			// mHC pre-step: attention
			engine::modules::HyperConnection::collapse(
				hidden_streams_, lw.hc_attn_fn, lw.hc_attn_base, lw.hc_attn_scale,
				mhc_collapsed_, mhc_post_, mhc_comb_, mhc_inv_rms_, mhc_mix_logits_, mhc_pre_,
				H, 20, 1e-6f, stream, m
			);

			// 1. Input RMSNorm: norm_hidden = rms_norm(mhc_collapsed_)
			engine::modules::RMSNorm::forward(
				mhc_collapsed_, lw.input_norm, norm_hidden_, 1e-5f, 0.0f, stream
			);
		}
		else {
			// 1. Input RMSNorm: norm_hidden = rms_norm(hidden_states)
			engine::modules::RMSNorm::forward(
				hidden_states_, lw.input_norm, norm_hidden_, 1e-5f, 0.0f, stream
			);
		}

		// 2. Attention
		if ( lw.is_linear_attn && lw.qkv.is_valid() ) {
			float* layer_kda = state_.kda_states + static_cast<size_t>( layer ) * state_.layer_kda_floats;
			float* layer_conv = state_.conv_history + static_cast<size_t>( layer ) * state_.layer_conv_floats;

			engine::modules::Linear::forward(
				norm_hidden_, lw.qkv, qkv_buf_, stream
			);

			engine::modules::GDN::causal_conv1d_silu_rows(
				qkv_buf_, layer_conv, lw.conv1d_weights, config_.kda_channels, m, d_rows_, slot_conv_floats,
				nullptr, 0, 0, nullptr, stream
			);

			engine::modules::GDN::l2_norm_heads(
				qkv_buf_, 16, 128, stream, true, m, config_.kda_channels
			);

			engine::modules::GDN::project_kda_gates(
				norm_hidden_, lw.b_proj, lw.f_a_proj, lw.f_b_proj, lw.g_a_proj, lw.g_b_proj,
				lw.A_log, lw.dt_bias, beta_buf_, g_decay_buf_, z_buf_, f_a_buf_, g_a_buf_,
				16, 128, H, stream, m
			);

			engine::modules::GDN::recurrent_step_kda_rows(
				layer_kda, qkv_buf_, g_decay_buf_, beta_buf_, z_buf_, lw.o_norm, attn_head_out_,
				16, 128, 1e-5f, m, d_rows_, slot_kda_floats, nullptr, 0, 0, nullptr, stream
			);

			if ( lw.o_proj.is_valid() ) {
				engine::modules::Linear::forward(
					attn_head_out_, lw.o_proj, attn_out_, stream
				);
			}

			if ( ring_ ) {
				ring_->all_reduce( attn_out_, stream );
			}

			if ( lw.hc_attn_fn.data != nullptr ) {
				// mHC post-step: attention update
				engine::modules::HyperConnection::update(
					hidden_streams_, attn_out_, mhc_post_, mhc_comb_, H, stream, m
				);
			}
			else {
				// Residual connection: hidden_states += attn_out
				engine::modules::Elementwise::add_residual( hidden_states_, attn_out_, stream );
			}
		}
		else if ( lw.is_mla && lw.q_a.is_valid() ) {
			int mla_idx = layer / 4;

			// Project Q: q_a_proj -> q_a_norm -> q_b_proj
			engine::modules::Linear::forward(
				norm_hidden_, lw.q_a, q_lora_buf_, stream
			);

			engine::modules::RMSNorm::forward(
				q_lora_buf_, lw.q_a_norm, q_lora_norm_buf_, 1e-5f, 0.0f, stream
			);

			engine::modules::Linear::forward(
				q_lora_norm_buf_, lw.q_b, q_buf_, stream
			);

			// Project KV: kv_a_proj_with_mqa -> kv_a_norm -> kv_b_proj
			engine::modules::Linear::forward(
				norm_hidden_, lw.kv_a, kv_lora_buf_, stream
			);

			engine::modules::RMSNorm::forward(
				kv_lora_buf_, lw.kv_a_norm, kv_lora_norm_buf_, 1e-5f, 0.0f, stream
			);

			engine::modules::MLA::project_kv_b(
				kv_lora_norm_buf_, lw.kv_b_proj, k_head_buf_, v_head_buf_,
				16, 256, stream, m
			);

			// Store every row's K/V first, then attend: row r reads positions 0..pos[r] of its slot
			engine::modules::MLA::store_kv_cache(
				slot_k, slot_v, k_head_buf_, v_head_buf_,
				mla_idx, 0, 4096, config_.max_seq, stream, kv_ref, m
			);

			engine::modules::MLA::forward_decode(
				q_buf_, slot_k, slot_v, mla_heads_out_,
				mla_idx, 0, 16, 256, config_.max_seq, stream, kv_ref, m
			);

			// Project output: o_proj -> ring_->all_reduce across the 4 ranks
			if ( lw.o_proj.is_valid() ) {
				engine::modules::Linear::forward(
					mla_heads_out_, lw.o_proj, attn_out_, stream
				);
			}

			if ( ring_ ) {
				ring_->all_reduce( attn_out_, stream );
			}

			if ( lw.hc_attn_fn.data != nullptr ) {
				// mHC post-step: attention update
				engine::modules::HyperConnection::update(
					hidden_streams_, attn_out_, mhc_post_, mhc_comb_, H, stream, m
				);
			}
			else {
				// Residual connection: hidden_states += attn_out
				engine::modules::Elementwise::add_residual( hidden_states_, attn_out_, stream );
			}
		}

		if ( lw.hc_ffn_fn.data != nullptr ) {
			// mHC pre-step: FFN
			engine::modules::HyperConnection::collapse(
				hidden_streams_, lw.hc_ffn_fn, lw.hc_ffn_base, lw.hc_ffn_scale,
				mhc_collapsed_, mhc_post_, mhc_comb_, mhc_inv_rms_, mhc_mix_logits_, mhc_pre_,
				H, 20, 1e-6f, stream, m
			);

			// 3. Post-attention RMSNorm: norm_hidden = rms_norm(mhc_collapsed_)
			engine::modules::RMSNorm::forward(
				mhc_collapsed_, lw.post_attn_norm, norm_hidden_, 1e-5f, 0.0f, stream
			);
		}
		else {
			// 3. Post-attention RMSNorm: norm_hidden = rms_norm(hidden_states)
			engine::modules::RMSNorm::forward(
				hidden_states_, lw.post_attn_norm, norm_hidden_, 1e-5f, 0.0f, stream
			);
		}

		// 4. MoE FFN
		if ( lw.is_moe ) {
			// Replicated Top-K Router: 288 experts, top-8 selection evaluated on all ranks
			engine::modules::MoERouter::route_topk(
				norm_hidden_, lw.router_weights, lw.e_score_bias,
				state_.d_topk_indices, state_.d_topk_weights,
				H, config_.num_experts, config_.top_k, 2.5f, stream, m
			);

				// Routed experts in this rank's partition: device-resident dispatch, no host sync
			engine::modules::MoE::forward(
				norm_hidden_, lw.moe_weights,
				state_.d_topk_indices, state_.d_topk_weights,
				local_moe_out_, start_expert, end_expert,
				state_.moe, 10.0f, stream
			);

				// Shared expert (executed on rank 0, or partitioned)
			if ( lw.has_shared_expert && rank_ == 0 ) {
				engine::modules::Linear::forward(
					norm_hidden_, lw.shared_expert.gate, moe_gate_buf_, stream
				);
				engine::modules::Linear::forward(
					norm_hidden_, lw.shared_expert.up, moe_up_buf_, stream
				);
				engine::modules::SwiGLU::forward( moe_gate_buf_, moe_up_buf_, moe_inter_buf_, stream, 10.0f );
				engine::modules::Linear::forward(
					moe_inter_buf_, lw.shared_expert.down, moe_expert_out_, stream
				);

				engine::modules::MoERouter::accumulate(
					local_moe_out_.ptr<half>(),
					moe_expert_out_.ptr<const half>(),
					__float2half_rn( 1.0f ),
					m * H,
					stream
				);
			}

				// Sum local expert contributions across NVLink mesh
			if ( ring_ ) {
				ring_->all_reduce( local_moe_out_, stream );
			}
		}
		else {
			// Dense layer
			if ( lw.dense_mlp.gate.is_valid() ) {
				engine::modules::Linear::forward(
					norm_hidden_, lw.dense_mlp.gate, moe_gate_buf_, stream
				);
				engine::modules::Linear::forward(
					norm_hidden_, lw.dense_mlp.up, moe_up_buf_, stream
				);
				engine::modules::SwiGLU::forward( moe_gate_buf_, moe_up_buf_, moe_inter_buf_, stream, 10.0f );
				engine::modules::Linear::forward(
					moe_inter_buf_, lw.dense_mlp.down, ffn_out_, stream
				);
			}
		}

		const pod::TensorView& ffn_res = lw.is_moe ? local_moe_out_ : ffn_out_;

		if ( lw.hc_ffn_fn.data != nullptr ) {
			// mHC post-step: FFN update
			engine::modules::HyperConnection::update(
				hidden_streams_, ffn_res, mhc_post_, mhc_comb_, H, stream, m
			);
		}
		else {
			if ( lw.is_moe ) {
				engine::modules::Elementwise::add_residual( hidden_states_, local_moe_out_, stream );
			}
			else {
				engine::modules::Elementwise::add_residual( hidden_states_, ffn_out_, stream );
			}
		}
	}

	if ( layers_[0].hc_attn_fn.data != nullptr ) {
		// HyperHead unweighted mean across 4 streams
		engine::modules::HyperHead::mean( hidden_streams_, hidden_states_, H, stream, m );
	}

	// Final norm + LM head (rank 0 owns the unsharded vocabulary)
	if ( rank_ == 0 ) {
		if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
		if ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) {
			// Prefill: only the final row's logits are read
			pod::TensorView h1 = pod::make_tensor_2d( hidden_states_.ptr<half>() + static_cast<size_t>( m - 1 ) * H, pod::DType::FP16, 1, H );
			pod::TensorView n1 = pod::make_tensor_2d( norm_hidden_.ptr<half>(), pod::DType::FP16, 1, H );
			pod::TensorView l1 = pod::make_tensor_2d( logits_.ptr<half>(), pod::DType::FP16, 1, config_.vocab_size );
			engine::modules::RMSNorm::forward( h1, final_norm_, n1, 1e-5f, 0.0f, stream );
			engine::modules::Linear::forward( n1, lm_head_, l1, stream );
			return;
		}
		logits_.shape[0] = m;
		engine::modules::RMSNorm::forward( hidden_states_, final_norm_, norm_hidden_, 1e-5f, 0.0f, stream );
		engine::modules::Linear::forward( norm_hidden_, lm_head_, logits_, stream );
	}
}

void GLM53Pipeline::destroy() {
	if ( arena_ != nullptr ) {
		CUDA_CHECK( cudaSetDevice( rank_ ) );
		graphs_.reset();
	}
	if ( h_argmax_idx_ != nullptr ) {
		engine::core::host_free_pinned( h_argmax_idx_ );
		h_argmax_idx_ = nullptr;
	}
}

} // namespace engine::models
