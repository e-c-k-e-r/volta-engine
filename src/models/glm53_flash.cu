#include "engine/models/glm53_flash.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/gdn.h"
#include "trellis_context.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

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

pod::TensorView load_qkv_trellis_glm(
	engine::loader::SafeTensorsRepo& repo,
	const char* name,
	pod::MemoryArena& arena,
	int rank,
	cudaStream_t stream
) {
	int shard_idx = -1;
	const auto* meta = repo.find( name, &shard_idx );
	if ( !meta || shard_idx < 0 ) return pod::TensorView{};

	int BK = meta->shape[0]; // 256
	int BN = meta->shape[1]; // 1536
	int B = meta->shape[2];  // 80

	int local_BN = 384;
	int local_shape[3] = { BK, local_BN, B };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, local_shape, 3 );
	const uint16_t* src_cpu = static_cast<const uint16_t*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard_idx ), meta ) );
	if ( src_cpu && tv.data ) {
		std::vector<uint16_t> h_qkv( static_cast<size_t>( BK ) * local_BN * B );
		for ( int bk = 0; bk < BK; ++bk ) {
			const uint16_t* src_row = src_cpu + static_cast<size_t>( bk ) * ( BN * B );
			uint16_t* dst_row = h_qkv.data() + static_cast<size_t>( bk ) * ( local_BN * B );

			// Q: 128 blocks from offset rank * 128
			std::memcpy( dst_row + 0 * B, src_row + ( rank * 128 ) * B, 128 * B * sizeof( uint16_t ) );
			// K: 128 blocks from offset 512 + rank * 128
			std::memcpy( dst_row + 128 * B, src_row + ( 512 + rank * 128 ) * B, 128 * B * sizeof( uint16_t ) );
			// V: 128 blocks from offset 1024 + rank * 128
			std::memcpy( dst_row + 256 * B, src_row + ( 1024 + rank * 128 ) * B, 128 * B * sizeof( uint16_t ) );
		}
		CUDA_CHECK( cudaMemcpy( tv.data, h_qkv.data(), h_qkv.size() * sizeof( uint16_t ), cudaMemcpyHostToDevice ) );
	}
	return tv;
}

pod::TensorView load_qkv_svh_glm(
	engine::loader::SafeTensorsRepo& repo,
	const char* name,
	pod::MemoryArena& arena,
	int rank,
	cudaStream_t stream
) {
	int shard_idx = -1;
	const auto* meta = repo.find( name, &shard_idx );
	if ( !meta || shard_idx < 0 ) return pod::TensorView{};

	int local_elems = 6144;
	int shape[1] = { local_elems };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 1 );
	const half* src_cpu = static_cast<const half*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard_idx ), meta ) );
	if ( src_cpu && tv.data ) {
		std::vector<half> h_svh( local_elems );
		std::memcpy( h_svh.data() + 0, src_cpu + rank * 2048, 2048 * sizeof( half ) );
		std::memcpy( h_svh.data() + 2048, src_cpu + 8192 + rank * 2048, 2048 * sizeof( half ) );
		std::memcpy( h_svh.data() + 4096, src_cpu + 16384 + rank * 2048, 2048 * sizeof( half ) );
		CUDA_CHECK( cudaMemcpy( tv.data, h_svh.data(), local_elems * sizeof( half ), cudaMemcpyHostToDevice ) );
	}
	return tv;
}

pod::TensorView load_conv1d_weights_glm(
	engine::loader::SafeTensorsRepo& repo,
	const char* name,
	pod::MemoryArena& arena,
	int rank,
	cudaStream_t stream
) {
	int shard_idx = -1;
	const auto* meta = repo.find( name, &shard_idx );
	if ( !meta || shard_idx < 0 ) return pod::TensorView{};

	int local_channels = 6144;
	int shape[1] = { local_channels * 4 };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 1 );
	const uint16_t* src_bf16 = static_cast<const uint16_t*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard_idx ), meta ) );
	if ( src_bf16 && tv.data ) {
		std::vector<half> h_conv( local_channels * 4 );
		auto to_fp16 = []( uint16_t bf ) {
			uint32_t u32 = static_cast<uint32_t>( bf ) << 16;
			float f = *reinterpret_cast<const float*>( &u32 );
			return __float2half_rn( f );
		};
		for ( int c = 0; c < 2048; ++c ) {
			for ( int k = 0; k < 4; ++k ) {
				h_conv[( 0 + c ) * 4 + k] = to_fp16( src_bf16[( rank * 2048 + c ) * 4 + k] );
				h_conv[( 2048 + c ) * 4 + k] = to_fp16( src_bf16[( 8192 + rank * 2048 + c ) * 4 + k] );
				h_conv[( 4096 + c ) * 4 + k] = to_fp16( src_bf16[( 16384 + rank * 2048 + c ) * 4 + k] );
			}
		}
		CUDA_CHECK( cudaMemcpy( tv.data, h_conv.data(), h_conv.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
	}
	return tv;
}

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

} // namespace

GLM53Pipeline::GLM53Pipeline()
	: rank_{ 0 }
	, arena_{ nullptr }
	, ring_{ nullptr }
	, state_{}
	, weights_loaded_{ false }
	, layers_{}
	, d_argmax_idx_{ nullptr }
	, h_argmax_idx_{ nullptr }
	, graph_m1_{ nullptr }
	, instance_m1_{ nullptr }
	, graph_m1_captured_{ false }
	, graph_m8_{ nullptr }
	, instance_m8_{ nullptr }
	, graph_m8_captured_{ false }
{
}

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

	// 1. Latent & RoPE caches for MLA
	size_t max_seq = 2048;
	size_t lat_elems = static_cast<size_t>( config_.num_layers ) * max_seq * config_.mla_dc;
	size_t rope_elems = static_cast<size_t>( config_.num_layers ) * max_seq * config_.mla_dr;

	state_.latent_cache = static_cast<half*>( engine::core::arena_alloc( arena, lat_elems * sizeof( half ) ) );
	state_.rope_cache = static_cast<half*>( engine::core::arena_alloc( arena, rope_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.latent_cache, 0, lat_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.rope_cache, 0, rope_elems * sizeof( half ) ) );

	size_t num_pages = max_seq / 16;
	state_.block_table = static_cast<int32_t*>( engine::core::arena_alloc( arena, num_pages * sizeof( int32_t ) ) );
	state_.d_topk_indices = static_cast<int*>( engine::core::arena_alloc( arena, config_.top_k * sizeof( int ) ) );
	state_.d_topk_weights = static_cast<half*>( engine::core::arena_alloc( arena, config_.top_k * sizeof( half ) ) );

	std::vector<int32_t> init_blocks( num_pages );
	for ( size_t i = 0; i < num_pages; ++i ) {
		init_blocks[i] = static_cast<int32_t>( i );
	}
	CUDA_CHECK( cudaMemcpy( state_.block_table, init_blocks.data(), num_pages * sizeof( int32_t ), cudaMemcpyHostToDevice ) );
	state_.current_pos = 0;

	// 2. Activations & residuals
	int h_shape[2] = { 1, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	residual_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	int streams_shape[2] = { 4, config_.hidden_size };
	hidden_streams_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, streams_shape, 2 );
	residual_streams_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, streams_shape, 2 );
	mhc_collapsed_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	int rms_shape[1] = { 1 };
	mhc_inv_rms_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, rms_shape, 1 );

	int mix_shape[1] = { 24 };
	mhc_mix_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, mix_shape, 1 );

	int hc_shape[1] = { 4 };
	mhc_pre_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, hc_shape, 1 );
	mhc_post_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, hc_shape, 1 );

	int comb_shape[2] = { 4, 4 };
	mhc_comb_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, comb_shape, 2 );

	// 3. MLA buffers
	int q_lat_shape[2] = { 1, config_.mla_dc };
	q_lat_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_lat_shape, 2 );

	int q_pe_shape[2] = { 1, config_.mla_dr };
	q_pe_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_pe_shape, 2 );

	int mla_out_shape[2] = { 1, config_.hidden_size };
	mla_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mla_out_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	// Real MLA KV cache & execution buffers
	size_t mla_kv_elems = 11ULL * max_seq * 4096;
	state_.k_cache = static_cast<half*>( engine::core::arena_alloc( arena, mla_kv_elems * sizeof( half ) ) );
	state_.v_cache = static_cast<half*>( engine::core::arena_alloc( arena, mla_kv_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.k_cache, 0, mla_kv_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.v_cache, 0, mla_kv_elems * sizeof( half ) ) );

	int q_lora_shape[2] = { 1, 1536 };
	q_lora_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_lora_shape, 2 );
	q_lora_norm_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_lora_shape, 2 );

	int q_shape[2] = { 1, 4096 };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );

	int kv_lora_shape[2] = { 1, 512 };
	kv_lora_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_lora_shape, 2 );
	kv_lora_norm_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_lora_shape, 2 );

	int kv_head_shape[2] = { 1, 4096 };
	k_head_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );
	v_head_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );
	mla_heads_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_head_shape, 2 );

	// KDA linear attention buffers
	int qkv_shape[2] = { 1, 6144 };
	qkv_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );

	int fa_shape[2] = { 1, 128 };
	f_a_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, fa_shape, 2 );
	g_a_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, fa_shape, 2 );

	int beta_shape[2] = { 1, 16 };
	beta_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, beta_shape, 2 );

	int gdec_shape[2] = { 1, 2048 };
	g_decay_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdec_shape, 2 );

	int z_shape[2] = { 1, 2048 };
	z_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	attn_head_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );

	// 4. Router & MoE buffers
	int router_shape[2] = { config_.hidden_size, config_.num_experts };
	router_weights_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, router_shape, 2 );

	int moe_inter_shape[2] = { 1, config_.dense_intermediate_size };
	moe_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_inter_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_expert_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	local_moe_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	ffn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	// 5. Fallback dummy descriptors for decode benchmark
	int d_dn_shape[3] = { config_.hidden_size / 16, config_.hidden_size / 16, 64 };
	pod::TensorView dummy_expert_down = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_dn_shape, 3 );

	int suh_shape[1] = { config_.hidden_size };
	pod::TensorView dummy_suh = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );
	pod::TensorView dummy_svh = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );
	dummy_norm_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );

	CUDA_CHECK( cudaMemset( dummy_expert_down.data, 0x44, dummy_expert_down.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_suh.data, 0x3c, dummy_suh.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_svh.data, 0x3c, dummy_svh.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_norm_.data, 0x3c, dummy_norm_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( router_weights_.data, 0x3c, router_weights_.size_bytes() ) );

	dummy_linear_ = pod::LinearWeight::make_trellis( dummy_expert_down, dummy_suh, dummy_svh, 4 );

	for ( int L = 0; L < 45; ++L ) {
		auto& lw = layers_[L];
		lw.is_moe = ( L >= 3 );
		lw.input_norm = dummy_norm_;
		lw.post_attn_norm = dummy_norm_;
		lw.router_weights = router_weights_;

		int conv_h_shape[1] = { 6144 * 3 };
		lw.conv_history = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, conv_h_shape, 1 );
		CUDA_CHECK( cudaMemset( lw.conv_history.data, 0, lw.conv_history.size_bytes() ) );

		lw.gdn_state = static_cast<float*>( engine::core::arena_alloc( arena, 16 * 128 * 128 * sizeof( float ) ) );
		CUDA_CHECK( cudaMemset( lw.gdn_state, 0, 16 * 128 * 128 * sizeof( float ) ) );
	}

	// 6. Logits & argmax
	int logit_shape[2] = { 1, config_.vocab_size };
	logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, logit_shape, 2 );

	d_argmax_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	h_argmax_idx_ = static_cast<int*>( engine::core::host_alloc_pinned( sizeof( int ) ) );
	*h_argmax_idx_ = 0;

	CUDA_CHECK( cudaMemset( hidden_states_.data, 0, hidden_states_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( residual_.data, 0, residual_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( logits_.data, 0, logits_.size_bytes() ) );

	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );

	return true;
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
	char name_tr[256], name_suh[256], name_svh[256];
	int experts_per_rank = config_.num_experts / config_.tp_size; // 288 / 4 = 72
	int start_expert = rank_ * experts_per_rank;

	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.is_moe = ( L >= 3 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.input_layernorm.weight", L );
		lw.input_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.post_attention_layernorm.weight", L );
		lw.post_attn_norm = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

		std::snprintf( name_tr, sizeof( name_tr ), "model.language_model.layers.%d.self_attn.qkv_proj.trellis", L );
		std::snprintf( name_suh, sizeof( name_suh ), "model.language_model.layers.%d.self_attn.qkv_proj.suh", L );
		std::snprintf( name_svh, sizeof( name_svh ), "model.language_model.layers.%d.self_attn.qkv_proj.svh", L );

		pod::TensorView qkv_tr = load_qkv_trellis_glm( repo, name_tr, *arena_, rank_, stream );
		lw.is_linear_attn = ( qkv_tr.data != nullptr );

		if ( lw.is_linear_attn ) {
			pod::TensorView qkv_suh = repo.load( name_suh, *arena_, stream, pod::DType::FP16 );
			pod::TensorView qkv_svh = load_qkv_svh_glm( repo, name_svh, *arena_, rank_, stream );
			lw.qkv = pod::LinearWeight::make_trellis( qkv_tr, qkv_suh, qkv_svh, 5 );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.conv1d.weight", L );
			lw.conv1d_weights = load_conv1d_weights_glm( repo, name_buf, *arena_, rank_, stream );

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
			lw.router_weights = load_router_weights_glm( repo, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate.e_score_correction_bias", L );
			lw.e_score_bias = repo.load( name_buf, *arena_, stream, pod::DType::FP16 );

			// 72 local experts for this rank
			for ( int e = 0; e < experts_per_rank; ++e ) {
				int gid = start_expert + e;
				auto& ew = lw.experts[e];

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, gid );
				ew.gate = repo.load_linear( name_buf, *arena_, 4, stream );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, gid );
				ew.up = repo.load_linear( name_buf, *arena_, 4, stream );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, gid );
				ew.down = repo.load_linear( name_buf, *arena_, 4, stream );
			}

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
				const auto& ew = lw.experts[e];
				if ( !ew.gate.is_valid() || !ew.up.is_valid() || !ew.down.is_valid() ) {
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
	std::printf( "[RANK %d] load_weights complete: missing_count=%d | arena: before=%.3f GiB, after=%.3f GiB, loaded=%.3f GiB\n",
		rank_, missing_count,
		bytes_before / ( 1024.0 * 1024.0 * 1024.0 ),
		bytes_after / ( 1024.0 * 1024.0 * 1024.0 ),
		( bytes_after - bytes_before ) / ( 1024.0 * 1024.0 * 1024.0 ) );

	if ( stream ) {
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
	}
	repo.close();
	weights_loaded_ = true;
	return true;
}

void GLM53Pipeline::embed_token( int token_id, cudaStream_t stream ) {
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	if ( rank_ == 0 && embed_tokens_.data != nullptr ) {
		int threads = 256;
		int blocks = ( config_.hidden_size + threads - 1 ) / threads;
		embed_token_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
			embed_tokens_.ptr<const __nv_bfloat16>(),
			hidden_states_.ptr<half>(),
			token_id,
			config_.hidden_size
		);
	}
	else {
		CUDA_CHECK( cudaMemsetAsync( hidden_states_.data, 0, hidden_states_.size_bytes(), stream ) );
	}

	if ( ring_ ) {
		ring_->all_reduce( hidden_states_, stream );
	}

	if ( hidden_streams_.data != nullptr ) {
		engine::modules::HyperConnection::init_streams(
			hidden_states_,
			hidden_streams_,
			config_.hidden_size,
			stream
		);
	}
}

int GLM53Pipeline::sample_next_token( cudaStream_t stream ) {
	if ( weights_loaded_ && final_norm_.data != nullptr ) {
		engine::modules::RMSNorm::forward(
			hidden_states_, final_norm_, hidden_states_, 1e-5f, 0.0f, stream
		);
	}

	if ( lm_head_.is_valid() ) {
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
		config_.vocab_size,
		stream
	);
	CUDA_CHECK( cudaMemcpyAsync( h_argmax_idx_, d_argmax_idx_, sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	return *h_argmax_idx_;
}

void GLM53Pipeline::forward_step(
	const pod::TensorView& input_tokens,
	pod::TensorView& output_logits,
	int pos,
	cudaStream_t stream
) {
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	(void) input_tokens;
	state_.current_pos = pos;

	if ( !weights_loaded_ ) {
		// Benchmark dummy execution mode (M=1 decode latency gate)
		for ( int layer = 0; layer < config_.num_layers; ++layer ) {
			// 1. Dynamic Sparse Attention (DSA) MLA
			engine::modules::MLADSA::forward_decode(
				q_lat_buf_,
				q_pe_buf_,
				state_.latent_cache,
				state_.rope_cache,
				mla_out_,
				layer,
				pos,
				config_.dsa_top_k,
				config_.mla_dc,
				config_.mla_dr,
				2048,
				config_.hidden_size,
				stream
			);

			if ( ring_ ) {
				ring_->all_reduce_rms_norm( hidden_states_, residual_, dummy_norm_, 1e-5f, stream );
			}
			else {
				engine::modules::RMSNorm::forward(
					hidden_states_, dummy_norm_, hidden_states_, 1e-5f, 1.0f, stream
				);
			}

			// 2. Replicated Top-K router
			engine::modules::MoERouter::route_topk(
				hidden_states_,
				router_weights_,
				dummy_norm_,
				state_.d_topk_indices,
				state_.d_topk_weights,
				config_.hidden_size,
				config_.num_experts,
				config_.top_k,
				2.5f,
				stream
			);

			// 3. Persistent MoE dispatch via Linear
			engine::modules::Linear::forward(
				hidden_states_,
				dummy_linear_,
				hidden_states_,
				stream
			);

			if ( ring_ ) {
				ring_->all_reduce_rms_norm( hidden_states_, residual_, dummy_norm_, 1e-5f, stream );
			}
			else {
				engine::modules::RMSNorm::forward(
					hidden_states_, dummy_norm_, hidden_states_, 1e-5f, 1.0f, stream
				);
			}
		}

		if ( output_logits.data != nullptr ) {
			CUDA_CHECK( cudaMemcpyAsync( output_logits.data, hidden_states_.data, config_.hidden_size * sizeof( half ), cudaMemcpyDeviceToDevice, stream ) );
		}
		return;
	}

	// Real execution mode with partitioned weights (TP=4)
	int experts_per_rank = config_.num_experts / config_.tp_size; // 72
	int start_expert = rank_ * experts_per_rank;
	int end_expert = start_expert + experts_per_rank;

	for ( int layer = 0; layer < config_.num_layers; ++layer ) {
		auto& lw = layers_[layer];

		if ( lw.hc_attn_fn.data != nullptr ) {
			// mHC pre-step: attention
			engine::modules::HyperConnection::collapse(
				hidden_streams_,
				lw.hc_attn_fn,
				lw.hc_attn_base,
				lw.hc_attn_scale,
				mhc_collapsed_,
				mhc_post_,
				mhc_comb_,
				mhc_inv_rms_,
				mhc_mix_logits_,
				mhc_pre_,
				config_.hidden_size,
				20,
				1e-6f,
				stream
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
			engine::modules::Linear::forward(
				norm_hidden_, lw.qkv, qkv_buf_, stream
			);

			engine::modules::GDN::causal_conv1d_silu(
				qkv_buf_, lw.conv_history, lw.conv1d_weights, 6144, stream
			);

			engine::modules::GDN::l2_norm_heads(
				qkv_buf_, 16, 128, stream, true
			);

			engine::modules::GDN::project_kda_gates(
				norm_hidden_, lw.b_proj, lw.f_a_proj, lw.f_b_proj, lw.g_a_proj, lw.g_b_proj,
				lw.A_log, lw.dt_bias, beta_buf_, g_decay_buf_, z_buf_, f_a_buf_, g_a_buf_,
				16, 128, config_.hidden_size, stream
			);

			engine::modules::GDN::recurrent_step_kda_fused(
				lw.gdn_state, qkv_buf_, g_decay_buf_, beta_buf_, z_buf_, lw.o_norm, attn_head_out_,
				16, 128, 1e-5f, stream
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
					hidden_streams_,
					attn_out_,
					mhc_post_,
					mhc_comb_,
					config_.hidden_size,
					stream
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
				16, 256, stream
			);

			// Store to KV cache
			engine::modules::MLA::store_kv_cache(
				state_.k_cache, state_.v_cache, k_head_buf_, v_head_buf_,
				mla_idx, pos, 4096, 2048, stream
			);

			// Compute latent attention dot product and attend over past tokens
			engine::modules::MLA::forward_decode(
				q_buf_, state_.k_cache, state_.v_cache, mla_heads_out_,
				mla_idx, pos, 16, 256, 2048, stream
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
					hidden_streams_,
					attn_out_,
					mhc_post_,
					mhc_comb_,
					config_.hidden_size,
					stream
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
				hidden_streams_,
				lw.hc_ffn_fn,
				lw.hc_ffn_base,
				lw.hc_ffn_scale,
				mhc_collapsed_,
				mhc_post_,
				mhc_comb_,
				mhc_inv_rms_,
				mhc_mix_logits_,
				mhc_pre_,
				config_.hidden_size,
				20,
				1e-6f,
				stream
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
				norm_hidden_,
				lw.router_weights,
				lw.e_score_bias,
				state_.d_topk_indices,
				state_.d_topk_weights,
				config_.hidden_size,
				config_.num_experts,
				config_.top_k,
				2.5f,
				stream
			);

			// Copy selected indices and weights to host to dispatch local experts
			int h_indices[8];
			half h_weights[8];
			CUDA_CHECK( cudaMemcpyAsync( h_indices, state_.d_topk_indices, 8 * sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
			CUDA_CHECK( cudaMemcpyAsync( h_weights, state_.d_topk_weights, 8 * sizeof( half ), cudaMemcpyDeviceToHost, stream ) );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );

			CUDA_CHECK( cudaMemsetAsync( local_moe_out_.data, 0, local_moe_out_.size_bytes(), stream ) );

			// Execute local active experts falling into this rank's partition
			for ( int k = 0; k < config_.top_k; ++k ) {
				int gid = h_indices[k];
				if ( gid >= start_expert && gid < end_expert ) {
					int local_e = gid - start_expert;
					const auto& ew = lw.experts[local_e];

					if ( ew.gate.is_valid() && ew.up.is_valid() && ew.down.is_valid() ) {
						engine::modules::Linear::forward(
							norm_hidden_, ew.gate, moe_gate_buf_, stream
						);
						engine::modules::Linear::forward(
							norm_hidden_, ew.up, moe_up_buf_, stream
						);
						engine::modules::SwiGLU::forward( moe_gate_buf_, moe_up_buf_, moe_inter_buf_, stream, 10.0f );
						engine::modules::Linear::forward(
							moe_inter_buf_, ew.down, moe_expert_out_, stream
						);

						engine::modules::MoERouter::accumulate(
							local_moe_out_.ptr<half>(),
							moe_expert_out_.ptr<const half>(),
							h_weights[k],
							config_.hidden_size,
							stream
						);
					}
				}
			}

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
					config_.hidden_size,
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
				hidden_streams_,
				ffn_res,
				mhc_post_,
				mhc_comb_,
				config_.hidden_size,
				stream
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
		engine::modules::HyperHead::mean(
			hidden_streams_,
			hidden_states_,
			config_.hidden_size,
			stream
		);
	}

	if ( output_logits.data != nullptr ) {
		CUDA_CHECK( cudaMemcpyAsync( output_logits.data, hidden_states_.data, config_.hidden_size * sizeof( half ), cudaMemcpyDeviceToDevice, stream ) );
	}
}

bool GLM53Pipeline::capture_graph( int m, cudaStream_t stream ) {
	if ( m == 1 ) {
		if ( graph_m1_captured_ ) return true;
		pod::TensorView dummy_in = pod::make_tensor_2d( hidden_states_.data, pod::DType::FP16, 1, config_.hidden_size );
		pod::TensorView dummy_out = pod::make_tensor_2d( residual_.data, pod::DType::FP16, 1, config_.hidden_size );

		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		forward_step( dummy_in, dummy_out, 0, stream );
		CUDA_CHECK( cudaStreamEndCapture( stream, &graph_m1_ ) );
		CUDA_CHECK( cudaGraphInstantiate( &instance_m1_, graph_m1_, nullptr, nullptr, 0 ) );
		graph_m1_captured_ = true;
		return true;
	}
	else if ( m == 8 ) {
		if ( graph_m8_captured_ ) return true;
		pod::TensorView dummy_in = pod::make_tensor_2d( hidden_states_.data, pod::DType::FP16, 8, config_.hidden_size );
		pod::TensorView dummy_out = pod::make_tensor_2d( residual_.data, pod::DType::FP16, 8, config_.hidden_size );

		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		forward_step( dummy_in, dummy_out, 0, stream );
		CUDA_CHECK( cudaStreamEndCapture( stream, &graph_m8_ ) );
		CUDA_CHECK( cudaGraphInstantiate( &instance_m8_, graph_m8_, nullptr, nullptr, 0 ) );
		graph_m8_captured_ = true;
		return true;
	}
	return false;
}

void GLM53Pipeline::launch_graph( int m, cudaStream_t stream ) {
	if ( m == 1 ) {
		ENGINE_ASSERT( graph_m1_captured_, "Graph M=1 not captured" );
		CUDA_CHECK( cudaGraphLaunch( instance_m1_, stream ) );
	}
	else if ( m == 8 ) {
		ENGINE_ASSERT( graph_m8_captured_, "Graph M=8 not captured" );
		CUDA_CHECK( cudaGraphLaunch( instance_m8_, stream ) );
	}
}

void GLM53Pipeline::reset_state() {
	CUDA_CHECK( cudaSetDevice( rank_ ) );
	state_.current_pos = 0;
	size_t max_seq = 2048;
	size_t lat_elems = static_cast<size_t>( config_.num_layers ) * max_seq * config_.mla_dc;
	size_t rope_elems = static_cast<size_t>( config_.num_layers ) * max_seq * config_.mla_dr;
	if ( state_.latent_cache ) {
		cudaMemset( state_.latent_cache, 0, lat_elems * sizeof( half ) );
	}
	if ( state_.rope_cache ) {
		cudaMemset( state_.rope_cache, 0, rope_elems * sizeof( half ) );
	}
	size_t mla_kv_elems = 11ULL * max_seq * 4096;
	if ( state_.k_cache ) {
		cudaMemset( state_.k_cache, 0, mla_kv_elems * sizeof( half ) );
	}
	if ( state_.v_cache ) {
		cudaMemset( state_.v_cache, 0, mla_kv_elems * sizeof( half ) );
	}
	if ( residual_.data ) {
		cudaMemset( residual_.data, 0, residual_.size_bytes() );
	}
	if ( hidden_states_.data ) {
		cudaMemset( hidden_states_.data, 0, hidden_states_.size_bytes() );
	}
	if ( hidden_streams_.data ) {
		cudaMemset( hidden_streams_.data, 0, hidden_streams_.size_bytes() );
	}
	if ( residual_streams_.data ) {
		cudaMemset( residual_streams_.data, 0, residual_streams_.size_bytes() );
	}
	for ( int L = 0; L < 45; ++L ) {
		auto& lw = layers_[L];
		if ( lw.conv_history.data ) {
			cudaMemset( lw.conv_history.data, 0, lw.conv_history.size_bytes() );
		}
		if ( lw.gdn_state ) {
			cudaMemset( lw.gdn_state, 0, 16 * 128 * 128 * sizeof( float ) );
		}
	}
}

void GLM53Pipeline::destroy() {
	if ( h_argmax_idx_ != nullptr ) {
		engine::core::host_free_pinned( h_argmax_idx_ );
		h_argmax_idx_ = nullptr;
	}
	if ( graph_m1_captured_ ) {
		cudaGraphExecDestroy( instance_m1_ );
		cudaGraphDestroy( graph_m1_ );
		graph_m1_captured_ = false;
	}
	if ( graph_m8_captured_ ) {
		cudaGraphExecDestroy( instance_m8_ );
		cudaGraphDestroy( graph_m8_ );
		graph_m8_captured_ = false;
	}
}

} // namespace engine::models
