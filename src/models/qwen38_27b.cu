#include "engine/models/qwen38_27b.h"
#include "trellis_context.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

namespace engine::models {

// GDN linear attention recurrent state update (128x128 state per head)
__global__ __launch_bounds__( 128 )
void gdn_recurrent_update_kernel(
	float* __restrict__ gdn_state, // [40, 128, 128]
	const half* __restrict__ q,    // [40, 128]
	const half* __restrict__ k,    // [40, 128]
	const half* __restrict__ v,    // [40, 128]
	half* __restrict__ out,        // [40, 128]
	int num_heads,
	float decay
) {
	int head = blockIdx.x;
	int tid = threadIdx.x; // row index [0..127]

	if ( head >= num_heads ) return;

	float* state_row = gdn_state + ( head * 128 + tid ) * 128;
	const half* q_head = q + head * 128;
	const half* k_head = k + head * 128;
	const half* v_head = v + head * 128;
	half* out_head = out + head * 128;

	__shared__ float s_k[128];
	__shared__ float s_q[128];
	s_k[tid] = __half2float( k_head[tid] );
	s_q[tid] = __half2float( q_head[tid] );
	__syncthreads();

	// Step 1: retrieve dot product of state_row with k
	float pred = 0.0f;
	#pragma unroll 4
	for ( int j = 0; j < 128; ++j ) {
		pred = fmaf( state_row[j], s_k[j], pred );
	}

	float target = __half2float( v_head[tid] );
	float err = target - pred;

	// Step 2: update state row S[tid, j] = decay * S[tid, j] + err * k[j]
	// Step 3: compute output o[tid] = dot(state_row_new, q)
	float o_val = 0.0f;
	#pragma unroll 4
	for ( int j = 0; j < 128; ++j ) {
		float new_val = fmaf( state_row[j], decay, err * s_k[j] );
		state_row[j] = new_val;
		o_val = fmaf( new_val, s_q[j], o_val );
	}

	out_head[tid] = __float2half_rn( o_val );
}

// FlashAttention-V100 across 16 SWA self-attention layers (sliding window truncation SWA=16)
__global__ __launch_bounds__( 128 )
void paged_swa_attention_kernel(
	const half* __restrict__ q,       // [40, 128]
	const half* __restrict__ k_cache, // [num_pages, 16, 40, 128]
	const half* __restrict__ v_cache, // [num_pages, 16, 40, 128]
	const int32_t* __restrict__ block_table,
	half* __restrict__ out,           // [40, 128]
	int num_heads,
	int current_pos,
	int swa_window
) {
	int head = blockIdx.x;
	int tid = threadIdx.x;

	if ( head >= num_heads ) return;

	const half* q_head = q + head * 128;
	half* out_head = out + head * 128;

	__shared__ float s_q[128];
	s_q[tid] = __half2float( q_head[tid] );
	__syncthreads();

	// Attend over last 16 tokens in sliding window
	int start_pos = ( current_pos >= swa_window ) ? ( current_pos - swa_window + 1 ) : 0;
	int num_tokens = current_pos - start_pos + 1;
	if ( num_tokens > 16 ) num_tokens = 16;

	float scores[16];
	float max_score = -1e9f;

	#pragma unroll
	for ( int t = 0; t < 16; ++t ) {
		if ( t < num_tokens ) {
			int pos = start_pos + t;
			int page_idx = pos / 16;
			int offset_in_page = pos % 16;
			int phys_page = block_table[page_idx];

			const half* k_ptr = k_cache + ( ( ( phys_page * 16 + offset_in_page ) * num_heads + head ) * 128 );

			float dot = 0.0f;
			#pragma unroll 4
			for ( int d = 0; d < 128; ++d ) {
				dot = fmaf( s_q[d], __half2float( k_ptr[d] ), dot );
			}
			float scaled_dot = dot * 0.088388347648f; // 1 / sqrt(128)
			scores[t] = scaled_dot;
			if ( scaled_dot > max_score ) {
				max_score = scaled_dot;
			}
		}
		else {
			scores[t] = -1e9f;
		}
	}

	float sum_exp = 0.0f;
	#pragma unroll
	for ( int t = 0; t < 16; ++t ) {
		if ( t < num_tokens ) {
			float p = __expf( scores[t] - max_score );
			scores[t] = p;
			sum_exp += p;
		}
		else {
			scores[t] = 0.0f;
		}
	}

	float inv_sum = 1.0f / ( sum_exp + 1e-6f );
	#pragma unroll
	for ( int t = 0; t < 16; ++t ) {
		scores[t] *= inv_sum;
	}

	// Output: weighted sum of V vectors
	float o_dim = 0.0f;
	#pragma unroll
	for ( int t = 0; t < 16; ++t ) {
		if ( t < num_tokens ) {
			int pos = start_pos + t;
			int page_idx = pos / 16;
			int offset_in_page = pos % 16;
			int phys_page = block_table[page_idx];

			const half* v_ptr = v_cache + ( ( ( phys_page * 16 + offset_in_page ) * num_heads + head ) * 128 );
			o_dim = fmaf( scores[t], __half2float( v_ptr[tid] ), o_dim );
		}
	}

	out_head[tid] = __float2half_rn( o_dim );
}

__global__ void swiglu_kernel(
	const half* __restrict__ gate_up, // [M, 2, Intermediate]
	half* __restrict__ intermediate,  // [M, Intermediate]
	int numel
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < numel ) {
		float g = __half2float( gate_up[idx] );
		float u = __half2float( gate_up[idx + numel] );
		float silu_g = g / ( 1.0f + __expf( -g ) );
		intermediate[idx] = __float2half_rn( silu_g * u );
	}
}

Qwen38Pipeline::Qwen38Pipeline()
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

	CUDA_CHECK( cudaSetDevice( rank_ ) );

	// Allocate GDN states: 48 layers * 40 heads * 128 * 128 floats = 31.45 MB
	size_t gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * config_.num_heads * 128 * 128;
	state_.gdn_states = static_cast<float*>( engine::core::arena_alloc( arena, gdn_floats * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( state_.gdn_states, 0, gdn_floats * sizeof( float ) ) );

	// Allocate Paged KV cache: 512 pages * 16 tokens * 40 heads * 128 dims * 2 bytes = 83.88 MB
	size_t num_pages = 512;
	size_t cache_elems = num_pages * config_.block_size * config_.num_heads * config_.head_dim;
	state_.k_cache = static_cast<half*>( engine::core::arena_alloc( arena, cache_elems * sizeof( half ) ) );
	state_.v_cache = static_cast<half*>( engine::core::arena_alloc( arena, cache_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.k_cache, 0, cache_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.v_cache, 0, cache_elems * sizeof( half ) ) );

	state_.block_table = static_cast<int32_t*>( engine::core::arena_alloc( arena, num_pages * sizeof( int32_t ) ) );
	std::vector<int32_t> init_blocks( num_pages );
	for ( size_t i = 0; i < num_pages; ++i ) {
		init_blocks[i] = static_cast<int32_t>( i );
	}
	CUDA_CHECK( cudaMemcpy( state_.block_table, init_blocks.data(), num_pages * sizeof( int32_t ), cudaMemcpyHostToDevice ) );

	state_.current_pos = 0;

	// Allocate activation tensors
	int h_shape[2] = { 1, config_.hidden_size };
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	residual_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_weights_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape + 1, 1 );

	// Trellis weights descriptors
	int qkv_shape[3] = { config_.hidden_size / 16, config_.hidden_size / 16, 64 };
	qkv_proj_ = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, qkv_shape, 3 );
	out_proj_ = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, qkv_shape, 3 );

	int gu_shape[3] = { config_.hidden_size / 16, config_.intermediate_size / 16, 64 };
	gate_up_proj_ = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, gu_shape, 3 );

	int dn_shape[3] = { config_.intermediate_size / 16, config_.hidden_size / 16, 64 };
	down_proj_ = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, dn_shape, 3 );

	int suh_shape[1] = { config_.hidden_size };
	suh_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );

	int svh_shape[1] = { config_.hidden_size };
	svh_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_shape, 1 );

	CUDA_CHECK( cudaMemset( qkv_proj_.data, 0x44, qkv_proj_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( out_proj_.data, 0x44, out_proj_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( gate_up_proj_.data, 0x44, gate_up_proj_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( down_proj_.data, 0x44, down_proj_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( suh_.data, 0x3c, suh_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( svh_.data, 0x3c, svh_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( norm_weights_.data, 0x3c, norm_weights_.size_bytes() ) );

	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );

	return true;
}

void Qwen38Pipeline::forward_step(
	const pod::TensorView& input_tokens,
	pod::TensorView& output_logits,
	int pos,
	cudaStream_t stream
) {
	(void) input_tokens;
	state_.current_pos = pos;

	// Execute 64 layers
	for ( int layer = 0; layer < config_.num_layers; ++layer ) {
		bool is_full_attention = ( ( layer + 1 ) % 4 == 0 );

		if ( is_full_attention ) {
			paged_swa_attention_kernel<<<dim3( config_.num_heads ), dim3( 128 ), 0, stream>>>(
				hidden_states_.ptr<half>(),
				state_.k_cache,
				state_.v_cache,
				state_.block_table,
				hidden_states_.ptr<half>(),
				config_.num_heads,
				pos,
				config_.swa_window
			);
		}
		else {
			int linear_idx = layer - ( layer / 4 );
			float* layer_gdn = state_.gdn_states + linear_idx * config_.num_heads * 128 * 128;
			gdn_recurrent_update_kernel<<<dim3( config_.num_heads ), dim3( 128 ), 0, stream>>>(
				layer_gdn,
				hidden_states_.ptr<half>(),
				hidden_states_.ptr<half>(),
				hidden_states_.ptr<half>(),
				hidden_states_.ptr<half>(),
				config_.num_heads,
				0.95f
			);
		}

		if ( ring_ ) {
			ring_->all_reduce_rms_norm( hidden_states_, residual_, norm_weights_, 1e-5f, stream );
		}

		// MLP forward via LinearTrellis
		engine::modules::LinearTrellis::forward(
			hidden_states_,
			down_proj_,
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
	size_t gdn_floats = static_cast<size_t>( config_.num_linear_layers ) * config_.num_heads * 128 * 128;
	if ( state_.gdn_states ) {
		cudaMemset( state_.gdn_states, 0, gdn_floats * sizeof( float ) );
	}
}

void Qwen38Pipeline::destroy() {
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
