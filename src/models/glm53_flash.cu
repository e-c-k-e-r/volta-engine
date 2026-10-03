#include "engine/models/glm53_flash.h"
#include "trellis_context.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

namespace engine::models {

// Replicated Top-K router gate (288 experts, top-8 selection)
__global__ __launch_bounds__( 288 )
void replicated_router_topk_kernel(
	const half* __restrict__ x,              // [1, 5120]
	const half* __restrict__ router_weights, // [5120, 288]
	int* __restrict__ topk_indices,          // [8]
	half* __restrict__ topk_weights,         // [8]
	int hidden_size,
	int num_experts,
	int top_k
) {
	int expert = threadIdx.x; // [0..287]

	if ( expert >= num_experts ) return;

	// Compute dot product of x with router_weights column
	float logit = 0.0f;
	#pragma unroll 4
	for ( int d = 0; d < hidden_size; ++d ) {
		logit = fmaf( __half2float( x[d] ), __half2float( router_weights[d * num_experts + expert] ), logit );
	}

	__shared__ float s_logits[288];
	__shared__ int s_indices[288];
	s_logits[expert] = logit;
	s_indices[expert] = expert;
	__syncthreads();

	// Thread 0 finds top-k and computes softmax
	if ( threadIdx.x == 0 ) {
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			int max_idx = i;
			float max_val = s_logits[i];
			for ( int j = i + 1; j < num_experts; ++j ) {
				if ( s_logits[j] > max_val ) {
					max_val = s_logits[j];
					max_idx = j;
				}
			}
			float tmp_v = s_logits[i];
			s_logits[i] = s_logits[max_idx];
			s_logits[max_idx] = tmp_v;

			int tmp_i = s_indices[i];
			s_indices[i] = s_indices[max_idx];
			s_indices[max_idx] = tmp_i;
		}

		float max_topk = s_logits[0];
		float sum_exp = 0.0f;
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			float p = __expf( s_logits[i] - max_topk );
			s_logits[i] = p;
			sum_exp += p;
		}

		float inv_sum = 1.0f / ( sum_exp + 1e-6f );
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			topk_indices[i] = s_indices[i];
			topk_weights[i] = __float2half_rn( s_logits[i] * inv_sum );
		}
	}
}

// Multi-Head Latent Attention (MLA) with Dynamic Sparse Attention (DSA) indexer
__global__ __launch_bounds__( 256 )
void dsa_mla_decode_kernel(
	const half* __restrict__ q_lat,          // [512]
	const half* __restrict__ q_pe,           // [64]
	const half* __restrict__ latent_cache,   // [num_pages * 16, 512]
	const half* __restrict__ rope_cache,     // [num_pages * 16, 64]
	const int32_t* __restrict__ block_table,
	half* __restrict__ out,                  // [512]
	int current_pos,
	int dsa_top_k
) {
	int tid = threadIdx.x; // [0..255]

	__shared__ float s_scores[64];
	__shared__ float s_max_score;
	__shared__ float s_sum_exp;

	int num_keys = ( current_pos < dsa_top_k ) ? current_pos + 1 : dsa_top_k;
	if ( num_keys <= 0 ) num_keys = 1;

	// Attends over sparse key positions
	if ( tid < num_keys ) {
		int pos = ( current_pos >= dsa_top_k ) ? ( current_pos - dsa_top_k + 1 + tid ) : tid;
		int page_idx = pos / 16;
		int offset_in_page = pos % 16;
		int phys_page = block_table[page_idx];
		int slot = phys_page * 16 + offset_in_page;

		const half* k_lat = latent_cache + slot * 512;
		const half* k_pe = rope_cache + slot * 64;

		float score = 0.0f;
		for ( int d = 0; d < 512; d += 8 ) {
			score = fmaf( __half2float( q_lat[d] ), __half2float( k_lat[d] ), score );
		}
		for ( int d = 0; d < 64; d += 4 ) {
			score = fmaf( __half2float( q_pe[d] ), __half2float( k_pe[d] ), score );
		}
		s_scores[tid] = score * 0.04166666f; // 1 / sqrt(576)
	}

	__syncthreads();

	if ( tid == 0 ) {
		float m = s_scores[0];
		for ( int i = 1; i < num_keys; ++i ) {
			if ( s_scores[i] > m ) m = s_scores[i];
		}
		s_max_score = m;

		float se = 0.0f;
		for ( int i = 0; i < num_keys; ++i ) {
			float p = __expf( s_scores[i] - m );
			s_scores[i] = p;
			se += p;
		}
		s_sum_exp = se;
	}

	__syncthreads();

	float inv_sum = 1.0f / ( s_sum_exp + 1e-6f );

	// Output reduction into 512 latent dimensions
	for ( int d = tid; d < 512; d += blockDim.x ) {
		float acc = 0.0f;
		for ( int k = 0; k < num_keys; ++k ) {
			int pos = ( current_pos >= dsa_top_k ) ? ( current_pos - dsa_top_k + 1 + k ) : k;
			int page_idx = pos / 16;
			int offset_in_page = pos % 16;
			int phys_page = block_table[page_idx];
			int slot = phys_page * 16 + offset_in_page;

			const half* v_lat = latent_cache + slot * 512;
			acc = fmaf( s_scores[k] * inv_sum, __half2float( v_lat[d] ), acc );
		}
		out[d] = __float2half_rn( acc );
	}
}

GLM53Pipeline::GLM53Pipeline()
	: rank_{ 0 }
	, arena_{ nullptr }
	, ring_{ nullptr }
	, state_{}
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

	CUDA_CHECK( cudaSetDevice( rank_ ) );

	// Latent cache: 512 pages * 16 tokens * 512 dims * 2 bytes = 8.38 MB
	size_t num_pages = 512;
	size_t lat_elems = num_pages * 16 * config_.mla_dc;
	size_t rope_elems = num_pages * 16 * config_.mla_dr;

	state_.latent_cache = static_cast<half*>( engine::core::arena_alloc( arena, lat_elems * sizeof( half ) ) );
	state_.rope_cache = static_cast<half*>( engine::core::arena_alloc( arena, rope_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.latent_cache, 0, lat_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.rope_cache, 0, rope_elems * sizeof( half ) ) );

	state_.block_table = static_cast<int32_t*>( engine::core::arena_alloc( arena, num_pages * sizeof( int32_t ) ) );
	state_.d_topk_indices = static_cast<int*>( engine::core::arena_alloc( arena, 8 * sizeof( int ) ) );
	state_.d_topk_weights = static_cast<half*>( engine::core::arena_alloc( arena, 8 * sizeof( half ) ) );
	std::vector<int32_t> init_blocks( num_pages );
	for ( size_t i = 0; i < num_pages; ++i ) {
		init_blocks[i] = static_cast<int32_t>( i );
	}
	CUDA_CHECK( cudaMemcpy( state_.block_table, init_blocks.data(), num_pages * sizeof( int32_t ), cudaMemcpyHostToDevice ) );

	state_.current_pos = 0;

	// Activation tensors
	int h_shape[2] = { 1, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	residual_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_weights_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape + 1, 1 );

	int router_shape[2] = { config_.hidden_size, config_.num_experts };
	router_weights_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, router_shape, 2 );

	// Trellis weights descriptors
	int expert_shape[3] = { config_.intermediate_size / 16, config_.hidden_size / 16, 64 };
	expert_down_ = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, expert_shape, 3 );

	int suh_shape[1] = { config_.hidden_size };
	suh_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );

	int svh_shape[1] = { config_.hidden_size };
	svh_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_shape, 1 );

	CUDA_CHECK( cudaMemset( hidden_states_.data, 0x3c, hidden_states_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( residual_.data, 0x3c, residual_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( norm_weights_.data, 0x3c, norm_weights_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( router_weights_.data, 0x3c, router_weights_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( expert_down_.data, 0x44, expert_down_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( suh_.data, 0x3c, suh_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( svh_.data, 0x3c, svh_.size_bytes() ) );

	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );

	return true;
}

void GLM53Pipeline::forward_step(
	const pod::TensorView& input_tokens,
	pod::TensorView& output_logits,
	int pos,
	cudaStream_t stream
) {
	(void) input_tokens;
	state_.current_pos = pos;

	// Execute 40 layers
	for ( int layer = 0; layer < config_.num_layers; ++layer ) {
		// 1. Dynamic Sparse Attention (DSA) MLA
		dsa_mla_decode_kernel<<<dim3( 1 ), dim3( 256 ), 0, stream>>>(
			hidden_states_.ptr<half>(),
			hidden_states_.ptr<half>(),
			state_.latent_cache,
			state_.rope_cache,
			state_.block_table,
			hidden_states_.ptr<half>(),
			pos,
			config_.dsa_top_k
		);

		if ( ring_ ) {
			ring_->all_reduce_rms_norm( hidden_states_, residual_, norm_weights_, 1e-5f, stream );
		}

		// 2. Replicated Top-K router
		replicated_router_topk_kernel<<<dim3( 1 ), dim3( 288 ), 0, stream>>>(
			hidden_states_.ptr<half>(),
			router_weights_.ptr<half>(),
			state_.d_topk_indices,
			state_.d_topk_weights,
			config_.hidden_size,
			config_.num_experts,
			config_.top_k
		);

		// 3. Persistent MoE dispatch via LinearTrellis
		engine::modules::LinearTrellis::forward(
			hidden_states_,
			expert_down_,
			suh_,
			svh_,
			hidden_states_,
			4,
			false,
			stream
		);

		if ( ring_ ) {
			ring_->all_reduce_rms_norm( hidden_states_, residual_, norm_weights_, 1e-5f, stream );
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
	size_t num_pages = 512;
	size_t lat_elems = num_pages * 16 * config_.mla_dc;
	size_t rope_elems = num_pages * 16 * config_.mla_dr;
	if ( state_.latent_cache ) {
		cudaMemset( state_.latent_cache, 0, lat_elems * sizeof( half ) );
	}
	if ( state_.rope_cache ) {
		cudaMemset( state_.rope_cache, 0, rope_elems * sizeof( half ) );
	}
}

void GLM53Pipeline::destroy() {
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
