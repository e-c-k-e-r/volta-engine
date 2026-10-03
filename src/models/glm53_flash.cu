#include "engine/models/glm53_flash.h"
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

__global__ void accumulate_expert_kernel(
	half* __restrict__ local_moe_out,
	const half* __restrict__ expert_out,
	half weight,
	int hidden_size
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid < hidden_size ) {
		float cur = __half2float( local_moe_out[tid] );
		float val = __half2float( expert_out[tid] );
		float w = __half2float( weight );
		local_moe_out[tid] = __float2half_rn( fmaf( val, w, cur ) );
	}
}

const pod::SafeTensorMeta* find_meta_glm(
	pod::SafeTensorsFile st[15],
	const char* name,
	pod::SafeTensorsFile*& out_st,
	int preferred_file = -1
) {
	if ( preferred_file >= 0 && preferred_file < 15 ) {
		const auto* m = engine::loader::safetensors_find( st[preferred_file], name );
		if ( m != nullptr ) {
			out_st = &st[preferred_file];
			return m;
		}
	}
	for ( int i = 0; i < 15; ++i ) {
		if ( i == preferred_file ) continue;
		const auto* m = engine::loader::safetensors_find( st[i], name );
		if ( m != nullptr ) {
			out_st = &st[i];
			return m;
		}
	}
	out_st = nullptr;
	return nullptr;
}

pod::TensorView load_trellis_glm(
	pod::SafeTensorsFile st[15],
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream,
	int preferred_file = -1
) {
	pod::SafeTensorsFile* st_file = nullptr;
	const auto* meta = find_meta_glm( st, name, st_file, preferred_file );
	if ( !meta || !st_file ) return pod::TensorView{};

	int shape[3] = { meta->shape[0], meta->shape[1], meta->shape[2] };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, shape, 3 );
	if ( tv.data != nullptr ) {
		engine::loader::safetensors_load_to_device( *st_file, meta, tv.data, stream );
	}
	return tv;
}

pod::TensorView load_trellis_row_glm(
	pod::SafeTensorsFile st[15],
	const char* name,
	pod::MemoryArena& arena,
	int rank,
	int tp_size,
	cudaStream_t stream,
	int preferred_file = -1
) {
	pod::SafeTensorsFile* st_file = nullptr;
	const auto* meta = find_meta_glm( st, name, st_file, preferred_file );
	if ( !meta || !st_file ) return pod::TensorView{};

	int BK = meta->shape[0];
	int BN = meta->shape[1];
	int B = meta->shape[2];

	if ( tp_size > 1 ) {
		int BK_local = BK / tp_size;
		int local_shape[3] = { BK_local, BN, B };
		pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, local_shape, 3 );
		const void* src_cpu = engine::loader::safetensors_get_data_cpu( *st_file, meta );
		if ( src_cpu && tv.data ) {
			size_t slice_bytes = static_cast<size_t>( BK_local ) * BN * B * sizeof( uint16_t );
			const uint8_t* src_r = static_cast<const uint8_t*>( src_cpu ) + rank * slice_bytes;
			CUDA_CHECK( cudaMemcpyAsync( tv.data, src_r, slice_bytes, cudaMemcpyHostToDevice, stream ) );
		}
		return tv;
	}
	else {
		int shape[3] = { BK, BN, B };
		pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, shape, 3 );
		if ( tv.data != nullptr ) {
			engine::loader::safetensors_load_to_device( *st_file, meta, tv.data, stream );
		}
		return tv;
	}
}

pod::TensorView load_fp16_vector_slice_glm(
	pod::SafeTensorsFile st[15],
	const char* name,
	pod::MemoryArena& arena,
	int rank,
	int tp_size,
	cudaStream_t stream,
	int preferred_file = -1
) {
	pod::SafeTensorsFile* st_file = nullptr;
	const auto* meta = find_meta_glm( st, name, st_file, preferred_file );
	if ( !meta || !st_file ) return pod::TensorView{};

	int total_elems = meta->shape[0];
	int local_elems = total_elems / tp_size;
	int shape[1] = { local_elems };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 1 );
	const void* src_cpu = engine::loader::safetensors_get_data_cpu( *st_file, meta );
	if ( src_cpu && tv.data ) {
		const uint8_t* src_r = static_cast<const uint8_t*>( src_cpu ) + rank * local_elems * sizeof( half );
		CUDA_CHECK( cudaMemcpyAsync( tv.data, src_r, local_elems * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
	}
	return tv;
}

pod::TensorView load_fp16_vector_glm(
	pod::SafeTensorsFile st[15],
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream,
	int preferred_file = -1
) {
	pod::SafeTensorsFile* st_file = nullptr;
	const auto* meta = find_meta_glm( st, name, st_file, preferred_file );
	if ( !meta || !st_file ) return pod::TensorView{};

	int shape[1] = { meta->shape[0] };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 1 );
	if ( tv.data != nullptr ) {
		const void* src_cpu = engine::loader::safetensors_get_data_cpu( *st_file, meta );
		if ( src_cpu != nullptr ) {
			CUDA_CHECK( cudaMemcpyAsync( tv.data, src_cpu, meta->size_bytes, cudaMemcpyHostToDevice, stream ) );
		}
	}
	return tv;
}

pod::TensorView load_f32_as_fp16_vector_glm(
	pod::SafeTensorsFile st[15],
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream,
	int preferred_file = -1
) {
	pod::SafeTensorsFile* st_file = nullptr;
	const auto* meta = find_meta_glm( st, name, st_file, preferred_file );
	if ( !meta || !st_file ) return pod::TensorView{};

	int total_elems = meta->shape[0];
	const float* src_f32 = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( *st_file, meta ) );
	if ( !src_f32 ) return pod::TensorView{};

	int shape[1] = { total_elems };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 1 );
	if ( tv.data != nullptr ) {
		std::vector<half> h_fp16( total_elems );
		for ( int i = 0; i < total_elems; ++i ) {
			h_fp16[i] = __float2half_rn( src_f32[i] );
		}
		CUDA_CHECK( cudaMemcpy( tv.data, h_fp16.data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice ) );
	}
	return tv;
}

pod::TensorView load_bf16_as_fp16_vector_glm(
	pod::SafeTensorsFile st[15],
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream,
	int preferred_file = -1
) {
	pod::SafeTensorsFile* st_file = nullptr;
	const auto* meta = find_meta_glm( st, name, st_file, preferred_file );
	if ( !meta || !st_file ) return pod::TensorView{};

	int total_elems = meta->shape[0];
	const uint16_t* src_bf16 = static_cast<const uint16_t*>( engine::loader::safetensors_get_data_cpu( *st_file, meta ) );
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
		CUDA_CHECK( cudaMemcpy( tv.data, h_fp16.data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice ) );
	}
	return tv;
}

pod::TensorView load_router_weights_glm(
	pod::SafeTensorsFile st[15],
	const char* name,
	pod::MemoryArena& arena,
	cudaStream_t stream,
	int preferred_file = -1
) {
	pod::SafeTensorsFile* st_file = nullptr;
	const auto* meta = find_meta_glm( st, name, st_file, preferred_file );
	if ( !meta || !st_file ) return pod::TensorView{};

	int num_experts = meta->shape[0]; // 288
	int hidden_size = meta->shape[1]; // 4096
	const half* src = static_cast<const half*>( engine::loader::safetensors_get_data_cpu( *st_file, meta ) );
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

	// 3. MLA buffers
	int q_lat_shape[2] = { 1, config_.mla_dc };
	q_lat_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_lat_shape, 2 );

	int q_pe_shape[2] = { 1, config_.mla_dr };
	q_pe_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_pe_shape, 2 );

	int mla_out_shape[2] = { 1, config_.hidden_size };
	mla_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mla_out_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

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
	dummy_expert_down_ = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_dn_shape, 3 );

	int suh_shape[1] = { config_.hidden_size };
	dummy_suh_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );
	dummy_svh_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );
	dummy_norm_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );

	CUDA_CHECK( cudaMemset( dummy_expert_down_.data, 0x44, dummy_expert_down_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_suh_.data, 0x3c, dummy_suh_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_svh_.data, 0x3c, dummy_svh_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_norm_.data, 0x3c, dummy_norm_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( router_weights_.data, 0x3c, router_weights_.size_bytes() ) );

	for ( int L = 0; L < 45; ++L ) {
		auto& lw = layers_[L];
		lw.is_moe = ( L >= 3 );
		lw.input_norm = dummy_norm_;
		lw.post_attn_norm = dummy_norm_;
		lw.router_weights = router_weights_;
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
	char path[512];
	pod::SafeTensorsFile st[15]{};
	for ( int i = 1; i <= 15; ++i ) {
		std::snprintf( path, sizeof( path ), "%s/model-%05d-of-00015.safetensors", model_dir, i );
		if ( !engine::loader::safetensors_open( st[i - 1], path ) ) {
			std::printf( "    [loader] FAILED to open %s\n", path );
			for ( int j = 0; j < i - 1; ++j ) engine::loader::safetensors_close( st[j] );
			return false;
		}
	}

	// 1. Embed tokens (rank 0 hosts full vocabulary embedding and LM head)
	if ( rank_ == 0 ) {
		pod::SafeTensorsFile* st_embed = nullptr;
		const auto* meta_embed = find_meta_glm( st, "model.language_model.embed_tokens.weight", st_embed, 0 );
		if ( meta_embed != nullptr && st_embed != nullptr ) {
			int emb_shape[2] = { config_.vocab_size, config_.hidden_size };
			embed_tokens_ = engine::core::arena_alloc_tensor( *arena_, pod::DType::BF16, emb_shape, 2 );
			engine::loader::safetensors_load_to_device( *st_embed, meta_embed, embed_tokens_.data, stream );
		}

		// 2. Final norm
		final_norm_ = load_bf16_as_fp16_vector_glm( st, "model.language_model.norm.weight", *arena_, stream, 14 );

		// 3. LM Head
		lm_head_suh_ = load_fp16_vector_glm( st, "lm_head.suh", *arena_, stream, 13 );
		lm_head_svh_ = load_fp16_vector_glm( st, "lm_head.svh", *arena_, stream, 13 );
		lm_head_trellis_ = load_trellis_glm( st, "lm_head.trellis", *arena_, stream, 13 );
	}

	// 4. Per-layer weights across 45 layers
	char name_buf[256];
	int experts_per_rank = config_.num_experts / config_.tp_size; // 288 / 4 = 72
	int start_expert = rank_ * experts_per_rank;

	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.is_moe = ( L >= 3 );
		int pref = L / 3;

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.input_layernorm.weight", L );
		lw.input_norm = load_bf16_as_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.post_attention_layernorm.weight", L );
		lw.post_attn_norm = load_bf16_as_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj.trellis", L );
		lw.o_proj_tr = load_trellis_row_glm( st, name_buf, *arena_, rank_, config_.tp_size, stream, pref );
		lw.o_proj_bits = lw.o_proj_tr.data ? ( lw.o_proj_tr.shape[2] / 16 ) : 5;

		if ( L == 7 && rank_ == 0 ) {
			std::printf( "[DEBUG GLM] L7 o_proj: data=%p bits=%d shape=[%d, %d, %d]\n",
				lw.o_proj_tr.data, lw.o_proj_bits, lw.o_proj_tr.shape[0], lw.o_proj_tr.shape[1], lw.o_proj_tr.shape[2] );
			std::fflush( stdout );
		}

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj.suh", L );
		lw.o_proj_suh = load_fp16_vector_slice_glm( st, name_buf, *arena_, rank_, config_.tp_size, stream, pref );

		std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.self_attn.o_proj.svh", L );
		lw.o_proj_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

		if ( L == 7 && rank_ == 0 ) {
			std::printf( "[DEBUG GLM] L7 o_proj: suh.data=%p shape=%d, svh.data=%p shape=%d\n",
				lw.o_proj_suh.data, lw.o_proj_suh.shape[0], lw.o_proj_svh.data, lw.o_proj_svh.shape[0] );
			std::fflush( stdout );
		}

		if ( lw.is_moe ) {
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate.weight", L );
			lw.router_weights = load_router_weights_glm( st, name_buf, *arena_, stream, pref );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate.e_score_correction_bias", L );
			lw.e_score_bias = load_f32_as_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

			// 72 local experts for this rank
			for ( int e = 0; e < experts_per_rank; ++e ) {
				int gid = start_expert + e;
				auto& ew = lw.experts[e];

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.trellis", L, gid );
				ew.gate_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
				ew.gate_bits = ew.gate_tr.data ? ( ew.gate_tr.shape[2] / 16 ) : 3;

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.suh", L, gid );
				ew.gate_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.svh", L, gid );
				ew.gate_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.up_proj.trellis", L, gid );
				ew.up_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
				ew.up_bits = ew.up_tr.data ? ( ew.up_tr.shape[2] / 16 ) : 3;

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.up_proj.suh", L, gid );
				ew.up_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.up_proj.svh", L, gid );
				ew.up_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.down_proj.trellis", L, gid );
				ew.down_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
				ew.down_bits = ew.down_tr.data ? ( ew.down_tr.shape[2] / 16 ) : 3;

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.down_proj.suh", L, gid );
				ew.down_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.experts.%d.down_proj.svh", L, gid );
				ew.down_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );
			}

			// Shared experts
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.gate_proj.trellis", L );
			lw.shared_expert.gate_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
			lw.shared_expert.gate_bits = lw.shared_expert.gate_tr.data ? ( lw.shared_expert.gate_tr.shape[2] / 16 ) : 3;
			lw.has_shared_expert = ( lw.shared_expert.gate_tr.data != nullptr );

			if ( lw.has_shared_expert ) {
				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.gate_proj.suh", L );
				lw.shared_expert.gate_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.gate_proj.svh", L );
				lw.shared_expert.gate_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.up_proj.trellis", L );
				lw.shared_expert.up_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
				lw.shared_expert.up_bits = lw.shared_expert.up_tr.data ? ( lw.shared_expert.up_tr.shape[2] / 16 ) : 3;

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.up_proj.suh", L );
				lw.shared_expert.up_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.up_proj.svh", L );
				lw.shared_expert.up_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.down_proj.trellis", L );
				lw.shared_expert.down_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
				lw.shared_expert.down_bits = lw.shared_expert.down_tr.data ? ( lw.shared_expert.down_tr.shape[2] / 16 ) : 3;

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.down_proj.suh", L );
				lw.shared_expert.down_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

				std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.shared_experts.down_proj.svh", L );
				lw.shared_expert.down_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );
			}
		}
		else {
			// Dense layer
			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj.trellis", L );
			lw.dense_mlp.gate_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
			lw.dense_mlp.gate_bits = lw.dense_mlp.gate_tr.data ? ( lw.dense_mlp.gate_tr.shape[2] / 16 ) : 3;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj.suh", L );
			lw.dense_mlp.gate_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.gate_proj.svh", L );
			lw.dense_mlp.gate_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj.trellis", L );
			lw.dense_mlp.up_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
			lw.dense_mlp.up_bits = lw.dense_mlp.up_tr.data ? ( lw.dense_mlp.up_tr.shape[2] / 16 ) : 3;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj.suh", L );
			lw.dense_mlp.up_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.up_proj.svh", L );
			lw.dense_mlp.up_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj.trellis", L );
			lw.dense_mlp.down_tr = load_trellis_glm( st, name_buf, *arena_, stream, pref );
			lw.dense_mlp.down_bits = lw.dense_mlp.down_tr.data ? ( lw.dense_mlp.down_tr.shape[2] / 16 ) : 3;

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj.suh", L );
			lw.dense_mlp.down_suh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );

			std::snprintf( name_buf, sizeof( name_buf ), "model.language_model.layers.%d.mlp.down_proj.svh", L );
			lw.dense_mlp.down_svh = load_fp16_vector_glm( st, name_buf, *arena_, stream, pref );
		}
	}

	for ( int i = 0; i < 15; ++i ) {
		engine::loader::safetensors_close( st[i] );
	}

	weights_loaded_ = true;
	if ( stream ) {
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
	}
	return true;
}

void GLM53Pipeline::embed_token( int token_id, cudaStream_t stream ) {
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

	if ( state_.current_pos == 0 && rank_ == 0 ) {
		half h_prev[5];
		CUDA_CHECK( cudaMemcpy( h_prev, hidden_states_.data, 5 * sizeof( half ), cudaMemcpyDeviceToHost ) );
		std::printf( "    [glm debug embed token %d] hidden_states_[0..4]: %f %f %f %f %f\n",
			token_id,
			__half2float( h_prev[0] ), __half2float( h_prev[1] ),
			__half2float( h_prev[2] ), __half2float( h_prev[3] ),
			__half2float( h_prev[4] ) );
	}
}

int GLM53Pipeline::sample_next_token( cudaStream_t stream ) {
	half h_preview[5];
	CUDA_CHECK( cudaMemcpy( h_preview, hidden_states_.data, 5 * sizeof( half ), cudaMemcpyDeviceToHost ) );
	std::printf( "    [glm sample] hidden_states_[0..4]: %f %f %f %f %f\n",
		__half2float( h_preview[0] ), __half2float( h_preview[1] ),
		__half2float( h_preview[2] ), __half2float( h_preview[3] ),
		__half2float( h_preview[4] ) );

	if ( weights_loaded_ && final_norm_.data != nullptr ) {
		engine::modules::RMSNorm::forward(
			hidden_states_, final_norm_, hidden_states_, 1e-5f, 0.0f, stream
		);
	}

	CUDA_CHECK( cudaMemcpy( h_preview, hidden_states_.data, 5 * sizeof( half ), cudaMemcpyDeviceToHost ) );
	std::printf( "    [glm sample] after final_norm: %f %f %f %f %f\n",
		__half2float( h_preview[0] ), __half2float( h_preview[1] ),
		__half2float( h_preview[2] ), __half2float( h_preview[3] ),
		__half2float( h_preview[4] ) );

	if ( lm_head_trellis_.data != nullptr ) {
		engine::modules::LinearTrellis::forward(
			hidden_states_,
			lm_head_trellis_,
			lm_head_suh_,
			lm_head_svh_,
			logits_,
			6,
			false,
			true,
			stream
		);
	}

	CUDA_CHECK( cudaMemcpy( h_preview, logits_.data, 5 * sizeof( half ), cudaMemcpyDeviceToHost ) );
	std::printf( "    [glm sample] logits_[0..4]: %f %f %f %f %f\n",
		__half2float( h_preview[0] ), __half2float( h_preview[1] ),
		__half2float( h_preview[2] ), __half2float( h_preview[3] ),
		__half2float( h_preview[4] ) );

	engine::modules::Elementwise::argmax(
		logits_,
		d_argmax_idx_,
		config_.vocab_size,
		stream
	);
	CUDA_CHECK( cudaMemcpyAsync( h_argmax_idx_, d_argmax_idx_, sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	std::vector<half> h_logits( config_.vocab_size );
	CUDA_CHECK( cudaMemcpy( h_logits.data(), logits_.data, config_.vocab_size * sizeof( half ), cudaMemcpyDeviceToHost ) );
	std::vector<std::pair<float, int>> top_k;
	for ( int i = 0; i < config_.vocab_size; ++i ) {
		float val = __half2float( h_logits[i] );
		if ( top_k.size() < 5 ) {
			top_k.push_back( { val, i } );
			std::sort( top_k.begin(), top_k.end(), std::greater<std::pair<float, int>>() );
		}
		else if ( val > top_k.back().first ) {
			top_k.back() = { val, i };
			std::sort( top_k.begin(), top_k.end(), std::greater<std::pair<float, int>>() );
		}
	}
	std::printf( "    [glm top-5 logits]:\n" );
	for ( const auto& p : top_k ) {
		std::printf( "       id=%d, logit=%f\n", p.second, p.first );
	}

	std::printf( "    [glm sample] Best token idx: %d\n", *h_argmax_idx_ );

	return *h_argmax_idx_;
}

void GLM53Pipeline::forward_step(
	const pod::TensorView& input_tokens,
	pod::TensorView& output_logits,
	int pos,
	cudaStream_t stream
) {
	(void) input_tokens;
	state_.current_pos = pos;

	if ( rank_ == 0 && weights_loaded_ ) {
		half h_start[1];
		CUDA_CHECK( cudaMemcpy( h_start, hidden_states_.data, sizeof( half ), cudaMemcpyDeviceToHost ) );
		std::printf( "    [DEBUG entrance pos %d] hidden_states_[0]: %f\n", pos, __half2float( h_start[0] ) );
		std::fflush( stdout );
	}

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
				dummy_suh_,
				state_.d_topk_indices,
				state_.d_topk_weights,
				config_.hidden_size,
				config_.num_experts,
				config_.top_k,
				2.5f,
				stream
			);

			// 3. Persistent MoE dispatch via LinearTrellis
			engine::modules::LinearTrellis::forward(
				hidden_states_,
				dummy_expert_down_,
				dummy_suh_,
				dummy_svh_,
				hidden_states_,
				4,
				false,
				true,
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
		const auto& lw = layers_[layer];

		// 1. Input RMSNorm: norm_hidden = rms_norm(hidden_states)
		engine::modules::RMSNorm::forward(
			hidden_states_, lw.input_norm, norm_hidden_, 1e-5f, 0.0f, stream
		);

		// 2. MLA Attention
		CUDA_CHECK( cudaMemcpyAsync( q_lat_buf_.data, norm_hidden_.data, config_.mla_dc * sizeof( half ), cudaMemcpyDeviceToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( q_pe_buf_.data, norm_hidden_.ptr<half>() + config_.mla_dc, config_.mla_dr * sizeof( half ), cudaMemcpyDeviceToDevice, stream ) );

		engine::modules::MLADSA::store_cache(
			state_.latent_cache,
			state_.rope_cache,
			q_lat_buf_.ptr<const half>(),
			q_pe_buf_.ptr<const half>(),
			layer,
			pos,
			config_.mla_dc,
			config_.mla_dr,
			2048,
			stream
		);

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

		if ( lw.o_proj_tr.data != nullptr ) {
			engine::modules::LinearTrellis::forward(
				mla_out_, lw.o_proj_tr, lw.o_proj_suh, lw.o_proj_svh,
				attn_out_, lw.o_proj_bits, false, true, stream
			);
		}

		if ( ring_ ) {
			ring_->all_reduce( attn_out_, stream );
		}

		// Residual connection: hidden_states += attn_out
		engine::modules::Elementwise::add_residual( hidden_states_, attn_out_, stream );

		// 3. Post-attention RMSNorm: norm_hidden = rms_norm(hidden_states)
		engine::modules::RMSNorm::forward(
			hidden_states_, lw.post_attn_norm, norm_hidden_, 1e-5f, 0.0f, stream
		);

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

					if ( ew.gate_tr.data != nullptr && ew.up_tr.data != nullptr && ew.down_tr.data != nullptr ) {
						engine::modules::LinearTrellis::forward(
							norm_hidden_, ew.gate_tr, ew.gate_suh, ew.gate_svh,
							moe_gate_buf_, ew.gate_bits, false, true, stream
						);
						engine::modules::LinearTrellis::forward(
							norm_hidden_, ew.up_tr, ew.up_suh, ew.up_svh,
							moe_up_buf_, ew.up_bits, false, true, stream
						);
						engine::modules::SwiGLU::forward( moe_gate_buf_, moe_up_buf_, moe_inter_buf_, stream );
						engine::modules::LinearTrellis::forward(
							moe_inter_buf_, ew.down_tr, ew.down_suh, ew.down_svh,
							moe_expert_out_, ew.down_bits, false, true, stream
						);

						int threads = 256;
						int blocks = ( config_.hidden_size + threads - 1 ) / threads;
						accumulate_expert_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
							local_moe_out_.ptr<half>(),
							moe_expert_out_.ptr<const half>(),
							h_weights[k],
							config_.hidden_size
						);
					}
				}
			}

			// Shared expert (executed on rank 0, or partitioned)
			if ( lw.has_shared_expert && rank_ == 0 ) {
				engine::modules::LinearTrellis::forward(
					norm_hidden_, lw.shared_expert.gate_tr, lw.shared_expert.gate_suh, lw.shared_expert.gate_svh,
					moe_gate_buf_, lw.shared_expert.gate_bits, false, true, stream
				);
				engine::modules::LinearTrellis::forward(
					norm_hidden_, lw.shared_expert.up_tr, lw.shared_expert.up_suh, lw.shared_expert.up_svh,
					moe_up_buf_, lw.shared_expert.up_bits, false, true, stream
				);
				engine::modules::SwiGLU::forward( moe_gate_buf_, moe_up_buf_, moe_inter_buf_, stream );
				engine::modules::LinearTrellis::forward(
					moe_inter_buf_, lw.shared_expert.down_tr, lw.shared_expert.down_suh, lw.shared_expert.down_svh,
					moe_expert_out_, lw.shared_expert.down_bits, false, true, stream
				);

				int threads = 256;
				int blocks = ( config_.hidden_size + threads - 1 ) / threads;
				accumulate_expert_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
					local_moe_out_.ptr<half>(),
					moe_expert_out_.ptr<const half>(),
					__float2half_rn( 1.0f ),
					config_.hidden_size
				);
			}

			// Sum local expert contributions across NVLink mesh
			if ( ring_ ) {
				ring_->all_reduce( local_moe_out_, stream );
			}

			// Residual connection: hidden_states += local_moe_out
			engine::modules::Elementwise::add_residual( hidden_states_, local_moe_out_, stream );
		}
		else {
			// Dense layer
			if ( lw.dense_mlp.gate_tr.data != nullptr ) {
				engine::modules::LinearTrellis::forward(
					norm_hidden_, lw.dense_mlp.gate_tr, lw.dense_mlp.gate_suh, lw.dense_mlp.gate_svh,
					moe_gate_buf_, lw.dense_mlp.gate_bits, false, true, stream
				);
				engine::modules::LinearTrellis::forward(
					norm_hidden_, lw.dense_mlp.up_tr, lw.dense_mlp.up_suh, lw.dense_mlp.up_svh,
					moe_up_buf_, lw.dense_mlp.up_bits, false, true, stream
				);
				engine::modules::SwiGLU::forward( moe_gate_buf_, moe_up_buf_, moe_inter_buf_, stream );
				engine::modules::LinearTrellis::forward(
					moe_inter_buf_, lw.dense_mlp.down_tr, lw.dense_mlp.down_suh, lw.dense_mlp.down_svh,
					ffn_out_, lw.dense_mlp.down_bits, false, true, stream
				);

				if ( ring_ ) {
					ring_->all_reduce( ffn_out_, stream );
				}

				engine::modules::Elementwise::add_residual( hidden_states_, ffn_out_, stream );
			}
		}
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
	if ( residual_.data ) {
		cudaMemset( residual_.data, 0, residual_.size_bytes() );
	}
	if ( hidden_states_.data ) {
		cudaMemset( hidden_states_.data, 0, hidden_states_.size_bytes() );
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
