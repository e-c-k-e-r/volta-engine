#include "engine/models/qwen38_27b.h"
#include "engine/loader/safetensors.h"
#include "trellis_context.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
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

const pod::SafeTensorMeta* find_meta(
	pod::SafeTensorsFile& st1,
	pod::SafeTensorsFile& st2,
	const char* name,
	pod::SafeTensorsFile*& out_st
) {
	const auto* m = engine::loader::safetensors_find( st1, name );
	if ( m != nullptr ) {
		out_st = &st1;
		return m;
	}
	m = engine::loader::safetensors_find( st2, name );
	if ( m != nullptr ) {
		out_st = &st2;
		return m;
	}
	out_st = nullptr;
	return nullptr;
}

pod::TensorView load_trellis(
	pod::SafeTensorsFile& st1,
	pod::SafeTensorsFile& st2,
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	pod::SafeTensorsFile* st = nullptr;
	const auto* meta = find_meta( st1, st2, name, st );
	if ( !meta || !st ) return pod::TensorView{};

	int shape[3] = { meta->shape[0], meta->shape[1], meta->shape[2] };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, shape, 3 );
	if ( tv.data != nullptr ) {
		engine::loader::safetensors_load_to_device( *st, meta, tv.data, stream );
	}
	return tv;
}

pod::TensorView load_fp16_vector(
	pod::SafeTensorsFile& st1,
	pod::SafeTensorsFile& st2,
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	pod::SafeTensorsFile* st = nullptr;
	const auto* meta = find_meta( st1, st2, name, st );
	if ( !meta || !st ) return pod::TensorView{};

	int shape[1] = { meta->shape[0] };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 1 );
	if ( tv.data != nullptr ) {
		const void* src_cpu = engine::loader::safetensors_get_data_cpu( *st, meta );
		if ( src_cpu != nullptr ) {
			CUDA_CHECK( cudaMemcpyAsync( tv.data, src_cpu, meta->size_bytes, cudaMemcpyHostToDevice, stream ) );
		}
	}
	return tv;
}

pod::TensorView load_bf16_as_fp16_vector(
	pod::SafeTensorsFile& st1,
	pod::SafeTensorsFile& st2,
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	pod::SafeTensorsFile* st = nullptr;
	const auto* meta = find_meta( st1, st2, name, st );
	if ( !meta || !st ) return pod::TensorView{};

	int total_elems = meta->shape[0];
	const uint16_t* src_bf16 = static_cast<const uint16_t*>( engine::loader::safetensors_get_data_cpu( *st, meta ) );
	if ( !src_bf16 ) return pod::TensorView{};

	int shape[1] = { total_elems };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 1 );
	if ( tv.data != nullptr ) {
		std::vector<half> h_fp16( total_elems );
		for ( int i = 0; i < total_elems; ++i ) {
			uint32_t u32 = static_cast<uint32_t>( src_bf16[i] ) << 16;
			float f = *reinterpret_cast<const float*>( &u32 );
			h_fp16[i] = __float2half_rn( f );
		}
		CUDA_CHECK( cudaMemcpyAsync( tv.data, h_fp16.data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
	}
	return tv;
}

pod::TensorView load_fp16_2d(
	pod::SafeTensorsFile& st1,
	pod::SafeTensorsFile& st2,
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	pod::SafeTensorsFile* st = nullptr;
	const auto* meta = find_meta( st1, st2, name, st );
	if ( !meta || !st ) return pod::TensorView{};

	int shape[2] = { meta->shape[0], meta->shape[1] };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	if ( tv.data != nullptr ) {
		const void* src_cpu = engine::loader::safetensors_get_data_cpu( *st, meta );
		if ( src_cpu != nullptr ) {
			CUDA_CHECK( cudaMemcpyAsync( tv.data, src_cpu, meta->size_bytes, cudaMemcpyHostToDevice, stream ) );
		}
	}
	return tv;
}

pod::TensorView load_conv1d_weight(
	pod::SafeTensorsFile& st1,
	pod::SafeTensorsFile& st2,
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	pod::SafeTensorsFile* st = nullptr;
	const auto* meta = find_meta( st1, st2, name, st );
	if ( !meta || !st ) return pod::TensorView{};

	int channels = meta->shape[0];
	int shape[2] = { channels, 4 };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );

	const uint16_t* src_bf16 = static_cast<const uint16_t*>( engine::loader::safetensors_get_data_cpu( *st, meta ) );
	if ( src_bf16 != nullptr && tv.data != nullptr ) {
		size_t total_elems = static_cast<size_t>( channels ) * 4;
		std::vector<half> h_conv( total_elems );
		for ( size_t i = 0; i < total_elems; ++i ) {
			uint32_t u = static_cast<uint32_t>( src_bf16[i] ) << 16;
			float f = *reinterpret_cast<const float*>( &u );
			h_conv[i] = __float2half_rn( f );
		}
		CUDA_CHECK( cudaMemcpyAsync( tv.data, h_conv.data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
	}
	return tv;
}

} // namespace

Qwen38Pipeline::Qwen38Pipeline()
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

	// 1. Allocate recurrent GDN states: 48 layers x 48 value heads x 128 x 128 (FP32)
	size_t gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * config_.linear_num_v_heads * 128 * 128;
	state_.gdn_states = static_cast<float*>( engine::core::arena_alloc( arena, gdn_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.gdn_states, 0, gdn_floats * sizeof( float ) ) );

	// 2. Allocate GDN causal conv1d history: 48 layers x 10240 channels x 3 taps (FP32)
	size_t conv_hist_floats = static_cast<size_t>( config_.num_linear_layers ) * config_.linear_channels * 3;
	state_.conv_history = static_cast<float*>( engine::core::arena_alloc( arena, conv_hist_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.conv_history, 0, conv_hist_floats * sizeof( float ) ) );

	// 3. Allocate full-attention KV cache: 16 layers x max_seq x 4 kv_heads x 256 (FP16)
	size_t cache_elems = static_cast<size_t>( config_.num_attn_layers ) * config_.max_seq * config_.full_num_kv_heads * config_.full_head_dim;
	state_.k_cache = static_cast<half*>( engine::core::arena_alloc( arena, cache_elems * sizeof( half ) ) );
	state_.v_cache = static_cast<half*>( engine::core::arena_alloc( arena, cache_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.k_cache, 0, cache_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.v_cache, 0, cache_elems * sizeof( half ) ) );
	state_.current_pos = 0;

	// 4. Hidden states & residuals
	int h_shape[2] = { 1, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	residual_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	// 5. Full attention buffers
	int q_gate_shape[2] = { 1, config_.full_num_q_heads * config_.full_head_dim * 2 }; // 12288
	q_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_gate_shape, 2 );

	int q_shape[2] = { 1, config_.full_num_q_heads * config_.full_head_dim }; // 6144
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	gate_attn_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );

	int kv_shape[2] = { 1, config_.full_num_kv_heads * config_.full_head_dim }; // 1024
	k_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
	v_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );

	// 6. Linear attention (GDN) buffers
	int qkv_shape[2] = { 1, config_.linear_channels }; // 10240
	qkv_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );

	int z_shape[2] = { 1, config_.linear_num_v_heads * config_.linear_head_dim }; // 6144
	z_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	gdn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );

	int gdn_param_shape[1] = { config_.linear_num_v_heads }; // 48
	gdn_gate_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdn_param_shape, 1 );
	gdn_beta_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gdn_param_shape, 1 );

	// 7. MLP buffers
	int mlp_shape[2] = { 1, config_.intermediate_size }; // 17408
	gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	mlp_inter_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );

	int mlp_out_shape[2] = { 1, config_.hidden_size }; // 5120
	mlp_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_out_shape, 2 );

	// 8. Logits & argmax
	int logit_shape[2] = { 1, 248320 };
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

bool Qwen38Pipeline::load_weights( const char* model_dir, cudaStream_t stream ) {
	char path1[512];
	char path2[512];
	std::snprintf( path1, sizeof( path1 ), "%s/model-00001-of-00002.safetensors", model_dir );
	std::snprintf( path2, sizeof( path2 ), "%s/model-00002-of-00002.safetensors", model_dir );

	pod::SafeTensorsFile st1{};
	pod::SafeTensorsFile st2{};

	if ( !engine::loader::safetensors_open( st1, path1 ) ) {
		std::printf( "    [loader] FAILED to open %s\n", path1 );
		return false;
	}
	if ( !engine::loader::safetensors_open( st2, path2 ) ) {
		std::printf( "    [loader] FAILED to open %s\n", path2 );
		engine::loader::safetensors_close( st1 );
		return false;
	}

	// 1. Embed tokens
	pod::SafeTensorsFile* st_embed = nullptr;
	const auto* meta_embed = find_meta( st1, st2, "model.language_model.embed_tokens.weight", st_embed );
	if ( meta_embed != nullptr && st_embed != nullptr ) {
		int emb_shape[2] = { 248320, config_.hidden_size };
		embed_tokens_ = engine::core::arena_alloc_tensor( *arena_, pod::DType::BF16, emb_shape, 2 );
		engine::loader::safetensors_load_to_device( *st_embed, meta_embed, embed_tokens_.data, stream );
	}

	// 2. Final norm
	final_norm_ = load_bf16_as_fp16_vector( st1, st2, "model.language_model.norm.weight", *arena_, stream );

	// 3. LM Head
	lm_head_suh_ = load_fp16_vector( st1, st2, "lm_head.suh", *arena_, stream );
	lm_head_svh_ = load_fp16_vector( st1, st2, "lm_head.svh", *arena_, stream );
	lm_head_trellis_ = load_trellis( st1, st2, "lm_head.trellis", *arena_, stream );

	// 4. Per-layer weights across 64 layers
	char name_buf[256];
	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.is_full_attention = ( ( L + 1 ) % 4 == 0 );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.input_layernorm.weight", L );
		lw.input_norm = load_bf16_as_fp16_vector( st1, st2, name_buf, *arena_, stream );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.post_attention_layernorm.weight", L );
		lw.post_attn_norm = load_bf16_as_fp16_vector( st1, st2, name_buf, *arena_, stream );

		if ( lw.is_full_attention ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_proj.trellis", L );
			lw.q_proj_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
			lw.q_proj_bits = lw.q_proj_tr.data ? ( lw.q_proj_tr.shape[2] / 16 ) : 4;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_proj.suh", L );
			lw.q_proj_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_proj.svh", L );
			lw.q_proj_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.q_norm.weight", L );
			lw.q_norm = load_bf16_as_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.k_proj.trellis", L );
			lw.k_proj_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
			lw.k_proj_bits = lw.k_proj_tr.data ? ( lw.k_proj_tr.shape[2] / 16 ) : 5;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.k_proj.suh", L );
			lw.k_proj_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.k_proj.svh", L );
			lw.k_proj_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.k_norm.weight", L );
			lw.k_norm = load_bf16_as_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.v_proj.trellis", L );
			lw.v_proj_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
			lw.v_proj_bits = lw.v_proj_tr.data ? ( lw.v_proj_tr.shape[2] / 16 ) : 5;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.v_proj.suh", L );
			lw.v_proj_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.v_proj.svh", L );
			lw.v_proj_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj.trellis", L );
			lw.o_proj_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
			lw.o_proj_bits = lw.o_proj_tr.data ? ( lw.o_proj_tr.shape[2] / 16 ) : 4;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj.suh", L );
			lw.o_proj_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj.svh", L );
			lw.o_proj_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );
		}
		else {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_qkv.trellis", L );
			lw.in_proj_qkv_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
			lw.in_proj_qkv_bits = lw.in_proj_qkv_tr.data ? ( lw.in_proj_qkv_tr.shape[2] / 16 ) : 5;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_qkv.suh", L );
			lw.in_proj_qkv_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_qkv.svh", L );
			lw.in_proj_qkv_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.conv1d.weight", L );
			lw.conv1d_w = load_conv1d_weight( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.A_log", L );
			lw.A_log = load_bf16_as_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.dt_bias", L );
			lw.dt_bias = load_bf16_as_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", L );
			lw.in_proj_a = load_fp16_2d( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", L );
			lw.in_proj_b = load_fp16_2d( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_z.trellis", L );
			lw.in_proj_z_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
			lw.in_proj_z_bits = lw.in_proj_z_tr.data ? ( lw.in_proj_z_tr.shape[2] / 16 ) : 5;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_z.suh", L );
			lw.in_proj_z_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.in_proj_z.svh", L );
			lw.in_proj_z_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.norm.weight", L );
			lw.linear_norm = load_bf16_as_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.out_proj.trellis", L );
			lw.out_proj_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
			lw.out_proj_bits = lw.out_proj_tr.data ? ( lw.out_proj_tr.shape[2] / 16 ) : 5;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.out_proj.suh", L );
			lw.out_proj_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.linear_attn.out_proj.svh", L );
			lw.out_proj_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );
		}

		// MLP
		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj.trellis", L );
		lw.gate_proj_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
		lw.gate_proj_bits = lw.gate_proj_tr.data ? ( lw.gate_proj_tr.shape[2] / 16 ) : 3;

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj.suh", L );
		lw.gate_proj_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj.svh", L );
		lw.gate_proj_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj.trellis", L );
		lw.up_proj_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
		lw.up_proj_bits = lw.up_proj_tr.data ? ( lw.up_proj_tr.shape[2] / 16 ) : 3;

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj.suh", L );
		lw.up_proj_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj.svh", L );
		lw.up_proj_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj.trellis", L );
		lw.down_proj_tr = load_trellis( st1, st2, name_buf, *arena_, stream );
		lw.down_proj_bits = lw.down_proj_tr.data ? ( lw.down_proj_tr.shape[2] / 16 ) : 3;

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj.suh", L );
		lw.down_proj_suh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj.svh", L );
		lw.down_proj_svh = load_fp16_vector( st1, st2, name_buf, *arena_, stream );
	}

	engine::loader::safetensors_close( st1 );
	engine::loader::safetensors_close( st2 );

	weights_loaded_ = true;
	if ( stream ) {
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
	}
	return true;
}

void Qwen38Pipeline::embed_token( int token_id, cudaStream_t stream ) {
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

int Qwen38Pipeline::sample_next_token( cudaStream_t stream ) {
	if ( weights_loaded_ && final_norm_.data != nullptr ) {
		engine::modules::RMSNorm::forward(
			hidden_states_, final_norm_, hidden_states_, 1e-6f, 1.0f, stream
		);
	}

	if ( lm_head_trellis_.data != nullptr ) {
		engine::modules::LinearTrellis::forward(
			hidden_states_,
			lm_head_trellis_,
			lm_head_suh_,
			lm_head_svh_,
			logits_,
			5,
			false,
			true,
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

void Qwen38Pipeline::forward_step(
	const pod::TensorView& input_tokens,
	pod::TensorView& output_logits,
	int pos,
	cudaStream_t stream
) {
	(void) input_tokens;
	state_.current_pos = pos;

	int full_attn_idx = 0;
	int linear_idx = 0;

	for ( int layer = 0; layer < config_.num_layers; ++layer ) {
		const auto& lw = layers_[layer];

		// 1. Input RMSNorm: norm_hidden = rms_norm(hidden_states)
		engine::modules::RMSNorm::forward(
			hidden_states_, lw.input_norm, norm_hidden_, 1e-6f, 1.0f, stream
		);

		// 2. Attention
		if ( lw.is_full_attention ) {
			// Q proj: projects norm_hidden (5120) to Q+Gate (12288 = 24 * 512)
			engine::modules::LinearTrellis::forward(
				norm_hidden_, lw.q_proj_tr, lw.q_proj_suh, lw.q_proj_svh,
				q_gate_buf_, lw.q_proj_bits, false, true, stream
			);

			// Unpack Q and Gate
			engine::modules::Attention::unpack_q_gate(
				q_gate_buf_, q_buf_, gate_attn_buf_,
				config_.full_num_q_heads, config_.full_head_dim, stream
			);

			// RMSNorm on Q heads (24 heads of 256)
			engine::modules::RMSNorm::forward_heads(
				q_buf_, lw.q_norm, config_.full_num_q_heads, config_.full_head_dim, 1e-6f, 1.0f, stream
			);

			// K proj: projects norm_hidden (5120) to K (1024 = 4 * 256)
			engine::modules::LinearTrellis::forward(
				norm_hidden_, lw.k_proj_tr, lw.k_proj_suh, lw.k_proj_svh,
				k_buf_, lw.k_proj_bits, false, true, stream
			);

			// RMSNorm on K heads (4 heads of 256)
			engine::modules::RMSNorm::forward_heads(
				k_buf_, lw.k_norm, config_.full_num_kv_heads, config_.full_head_dim, 1e-6f, 1.0f, stream
			);

			// V proj: projects norm_hidden (5120) to V (1024 = 4 * 256)
			engine::modules::LinearTrellis::forward(
				norm_hidden_, lw.v_proj_tr, lw.v_proj_suh, lw.v_proj_svh,
				v_buf_, lw.v_proj_bits, false, true, stream
			);

			// RoPE applied to Q and K
			engine::modules::RoPE::forward(
				q_buf_, k_buf_,
				config_.full_num_q_heads, config_.full_num_kv_heads,
				config_.full_head_dim, config_.full_rot_dim,
				pos, 10000000.0f, stream
			);

			// Store K and V to KV Cache
			engine::modules::Attention::store_kv_cache(
				state_.k_cache, state_.v_cache,
				k_buf_, v_buf_,
				full_attn_idx, pos,
				config_.full_num_kv_heads, config_.full_head_dim,
				config_.max_seq, stream
			);

			// Attend over past tokens + sigmoid gate
			engine::modules::Attention::forward_decode(
				q_buf_, state_.k_cache, state_.v_cache, gate_attn_buf_,
				attn_out_,
				full_attn_idx, pos,
				config_.full_num_q_heads, config_.full_num_kv_heads,
				config_.full_head_dim, config_.max_seq, stream
			);
			full_attn_idx++;

			// O proj: projects attn_out (6144) to mlp_out (5120)
			engine::modules::LinearTrellis::forward(
				attn_out_, lw.o_proj_tr, lw.o_proj_suh, lw.o_proj_svh,
				mlp_out_, lw.o_proj_bits, false, true, stream
			);
		}
		else {
			// In proj QKV: projects norm_hidden (5120) to QKV (10240)
			engine::modules::LinearTrellis::forward(
				norm_hidden_, lw.in_proj_qkv_tr, lw.in_proj_qkv_suh, lw.in_proj_qkv_svh,
				qkv_buf_, lw.in_proj_qkv_bits, false, true, stream
			);

			// Causal conv1d + SiLU
			float* hist_ptr = state_.conv_history + static_cast<size_t>( linear_idx ) * config_.linear_channels * 3;
			pod::TensorView conv_hist = pod::make_tensor_2d( hist_ptr, pod::DType::FP32, config_.linear_channels, 3 );
			engine::modules::GDN::causal_conv1d_silu(
				qkv_buf_, conv_hist, lw.conv1d_w, config_.linear_channels, stream
			);

			// L2 norm on Q and K heads (16 heads each of dim 128)
			engine::modules::GDN::l2_norm_heads(
				qkv_buf_, config_.linear_num_k_heads, config_.linear_head_dim, stream
			);

			// Project gates: in_proj_a, in_proj_b -> gdn_gate, gdn_beta
			engine::modules::GDN::project_gates(
				norm_hidden_, lw.in_proj_a, lw.in_proj_b, lw.A_log, lw.dt_bias,
				gdn_gate_, gdn_beta_, config_.linear_num_v_heads, config_.hidden_size, stream
			);

			// In proj Z: projects norm_hidden (5120) to Z (6144 = 48 * 128)
			engine::modules::LinearTrellis::forward(
				norm_hidden_, lw.in_proj_z_tr, lw.in_proj_z_suh, lw.in_proj_z_svh,
				z_buf_, lw.in_proj_z_bits, false, true, stream
			);

			// GDN recurrent step
			float* layer_gdn_state = state_.gdn_states + static_cast<size_t>( linear_idx ) * config_.linear_num_v_heads * 128 * 128;
			engine::modules::GDN::recurrent_step(
				layer_gdn_state, qkv_buf_, gdn_gate_, gdn_beta_,
				gdn_out_, config_.linear_num_k_heads, config_.linear_num_v_heads,
				config_.linear_head_dim, stream
			);

			// Gated RMSNorm: gdn_out * rsqrt * norm_weight * silu(z)
			engine::modules::GDN::gated_rmsnorm(
				gdn_out_, z_buf_, lw.linear_norm, config_.linear_num_v_heads, config_.linear_head_dim, stream
			);

			// Out proj: projects gdn_out (6144) to mlp_out (5120)
			engine::modules::LinearTrellis::forward(
				gdn_out_, lw.out_proj_tr, lw.out_proj_suh, lw.out_proj_svh,
				mlp_out_, lw.out_proj_bits, false, true, stream
			);
			linear_idx++;
		}

		if ( ring_ ) {
			ring_->all_reduce( mlp_out_, stream );
		}

		// Residual connection: hidden_states += mlp_out
		engine::modules::Elementwise::add_residual( hidden_states_, mlp_out_, stream );

		// 3. Post-attention RMSNorm: norm_hidden = rms_norm(hidden_states)
		engine::modules::RMSNorm::forward(
			hidden_states_, lw.post_attn_norm, norm_hidden_, 1e-6f, 1.0f, stream
		);

		// 4. MLP
		engine::modules::LinearTrellis::forward(
			norm_hidden_, lw.gate_proj_tr, lw.gate_proj_suh, lw.gate_proj_svh,
			gate_buf_, lw.gate_proj_bits, false, true, stream
		);

		engine::modules::LinearTrellis::forward(
			norm_hidden_, lw.up_proj_tr, lw.up_proj_suh, lw.up_proj_svh,
			up_buf_, lw.up_proj_bits, false, true, stream
		);

		engine::modules::SwiGLU::forward( gate_buf_, up_buf_, mlp_inter_, stream );

		engine::modules::LinearTrellis::forward(
			mlp_inter_, lw.down_proj_tr, lw.down_proj_suh, lw.down_proj_svh,
			mlp_out_, lw.down_proj_bits, false, true, stream
		);

		if ( ring_ ) {
			ring_->all_reduce( mlp_out_, stream );
		}

		// Residual connection: hidden_states += mlp_out
		engine::modules::Elementwise::add_residual( hidden_states_, mlp_out_, stream );
	}

	if ( output_logits.data != nullptr ) {
		CUDA_CHECK( cudaMemcpyAsync( output_logits.data, hidden_states_.data, config_.hidden_size * sizeof( half ), cudaMemcpyDeviceToDevice, stream ) );
	}
}

bool Qwen38Pipeline::capture_graph( int m, cudaStream_t stream ) {
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

void Qwen38Pipeline::launch_graph( int m, cudaStream_t stream ) {
	if ( m == 1 ) {
		ENGINE_ASSERT( graph_m1_captured_, "Graph M=1 not captured" );
		CUDA_CHECK( cudaGraphLaunch( instance_m1_, stream ) );
	}
	else if ( m == 8 ) {
		ENGINE_ASSERT( graph_m8_captured_, "Graph M=8 not captured" );
		CUDA_CHECK( cudaGraphLaunch( instance_m8_, stream ) );
	}
}

void Qwen38Pipeline::reset_state() {
	state_.current_pos = 0;
	size_t gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * config_.linear_num_v_heads * 128 * 128;
	if ( state_.gdn_states ) {
		cudaMemset( state_.gdn_states, 0, gdn_floats * sizeof( float ) );
	}
	size_t conv_hist_floats = static_cast<size_t>( config_.num_linear_layers ) * config_.linear_channels * 3;
	if ( state_.conv_history ) {
		cudaMemset( state_.conv_history, 0, conv_hist_floats * sizeof( float ) );
	}
	size_t cache_elems = static_cast<size_t>( config_.num_attn_layers ) * config_.max_seq * config_.full_num_kv_heads * config_.full_head_dim;
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
	if ( graph_m8_captured_ ) {
		cudaGraphExecDestroy( instance_m8_ );
		cudaGraphDestroy( graph_m8_ );
		graph_m8_captured_ = false;
	}
}

} // namespace engine::models
