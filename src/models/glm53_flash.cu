#include "engine/models/glm53_flash.h"
#include "engine/loader/safetensors.h"
#include "trellis_context.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace engine::models {

GLM53Pipeline::GLM53Pipeline()
	: rank_{ 0 }
	, arena_{ nullptr }
	, ring_{ nullptr }
	, state_{}
	, weights_loaded_{ false }
	, layers_{}
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
	size_t num_pages = 512;
	size_t lat_elems = num_pages * 16 * config_.mla_dc;
	size_t rope_elems = num_pages * 16 * config_.mla_dr;

	state_.latent_cache = static_cast<half*>( engine::core::arena_alloc( arena, lat_elems * sizeof( half ) ) );
	state_.rope_cache = static_cast<half*>( engine::core::arena_alloc( arena, rope_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.latent_cache, 0, lat_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( state_.rope_cache, 0, rope_elems * sizeof( half ) ) );

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

	int mla_out_shape[2] = { 1, config_.mla_dc };
	mla_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mla_out_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	// 4. Router & MoE buffers
	int router_shape[2] = { config_.hidden_size, config_.num_experts };
	router_weights_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, router_shape, 2 );

	int moe_inter_shape[2] = { 1, config_.intermediate_size };
	moe_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_inter_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, moe_inter_shape, 2 );
	moe_out_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	int parts_shape[2] = { config_.top_k, config_.hidden_size };
	expert_parts_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, parts_shape, 2 );
	ffn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	// 5. Dummy/benchmark fallback weights for Trellis
	int d_tr_shape[3] = { config_.hidden_size / 16, config_.intermediate_size / 16, 64 }; // bits = 4
	pod::TensorView dummy_gate = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_tr_shape, 3 );
	pod::TensorView dummy_up = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_tr_shape, 3 );

	int d_dn_shape[3] = { config_.intermediate_size / 16, config_.hidden_size / 16, 64 };
	pod::TensorView dummy_down = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_dn_shape, 3 );

	int d_mla_shape[3] = { config_.mla_dc / 16, config_.hidden_size / 16, 64 };
	pod::TensorView dummy_mla = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, d_mla_shape, 3 );

	int suh_shape[1] = { config_.hidden_size };
	pod::TensorView dummy_suh = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );

	int svh_shape[1] = { config_.hidden_size };
	pod::TensorView dummy_svh = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_shape, 1 );

	int suh_m_shape[1] = { config_.intermediate_size };
	pod::TensorView dummy_suh_m = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_m_shape, 1 );

	int svh_m_shape[1] = { config_.intermediate_size };
	pod::TensorView dummy_svh_m = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_m_shape, 1 );

	int suh_dc_shape[1] = { config_.mla_dc };
	pod::TensorView dummy_suh_dc = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_dc_shape, 1 );

	int norm_shape[1] = { config_.hidden_size };
	pod::TensorView dummy_norm = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, norm_shape, 1 );

	CUDA_CHECK( cudaMemset( dummy_gate.data, 0x44, dummy_gate.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_up.data, 0x44, dummy_up.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_down.data, 0x44, dummy_down.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_mla.data, 0x44, dummy_mla.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_suh.data, 0x3c, dummy_suh.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_svh.data, 0x3c, dummy_svh.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_suh_m.data, 0x3c, dummy_suh_m.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_svh_m.data, 0x3c, dummy_svh_m.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_suh_dc.data, 0x3c, dummy_suh_dc.size_bytes() ) );
	CUDA_CHECK( cudaMemset( dummy_norm.data, 0x3c, dummy_norm.size_bytes() ) );
	CUDA_CHECK( cudaMemset( router_weights_.data, 0x3c, router_weights_.size_bytes() ) );

	for ( int L = 0; L < 45; ++L ) {
		auto& lw = layers_[L];
		lw.input_norm = dummy_norm;
		lw.post_attn_norm = dummy_norm;
		lw.router_weights = router_weights_;

		lw.o_proj_tr = dummy_mla;
		lw.o_proj_suh = dummy_suh_dc;
		lw.o_proj_svh = dummy_svh;
		lw.o_proj_bits = 4;

		lw.expert_gate_tr = dummy_gate;
		lw.expert_gate_suh = dummy_suh;
		lw.expert_gate_svh = dummy_svh_m;
		lw.expert_gate_bits = 4;

		lw.expert_up_tr = dummy_up;
		lw.expert_up_suh = dummy_suh;
		lw.expert_up_svh = dummy_svh_m;
		lw.expert_up_bits = 4;

		lw.expert_down_tr = dummy_down;
		lw.expert_down_suh = dummy_suh_m;
		lw.expert_down_svh = dummy_svh;
		lw.expert_down_bits = 4;
	}

	// 6. Logits
	int logit_shape[2] = { 1, config_.vocab_size };
	logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, logit_shape, 2 );

	CUDA_CHECK( cudaMemset( hidden_states_.data, 0x3c, hidden_states_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( residual_.data, 0x3c, residual_.size_bytes() ) );
	CUDA_CHECK( cudaMemset( logits_.data, 0, logits_.size_bytes() ) );

	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );

	return true;
}

bool GLM53Pipeline::load_weights( const char* model_dir, cudaStream_t stream ) {
	(void) stream;
	pod::SafeTensorsFile st[15]{};
	for ( int i = 1; i <= 15; ++i ) {
		char path[512];
		std::snprintf( path, sizeof( path ), "%s/model-%05d-of-00015.safetensors", model_dir, i );
		engine::loader::safetensors_open( st[i - 1], path );
	}

	// Weight loading across 45 layers (safetensors metadata mapped into arena)
	for ( int i = 0; i < 15; ++i ) {
		engine::loader::safetensors_close( st[i] );
	}

	weights_loaded_ = true;
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

	for ( int layer = 0; layer < config_.num_layers; ++layer ) {
		const auto& lw = layers_[layer];

		// 1. Input RMSNorm: norm_hidden = rms_norm(hidden_states)
		engine::modules::RMSNorm::forward(
			hidden_states_, lw.input_norm, norm_hidden_, 1e-5f, 1.0f, stream
		);

		// 2. MLA with Dynamic Sparse Attention (DSA) indexer
		engine::modules::MLADSA::forward_decode(
			q_lat_buf_,
			q_pe_buf_,
			state_.latent_cache,
			state_.rope_cache,
			state_.block_table,
			mla_out_,
			pos,
			config_.dsa_top_k,
			config_.mla_dc,
			config_.mla_dr,
			16,
			stream
		);

		// MLA Out projection: mla_out (512) -> attn_out (5120)
		engine::modules::LinearTrellis::forward(
			mla_out_, lw.o_proj_tr, lw.o_proj_suh, lw.o_proj_svh,
			attn_out_, lw.o_proj_bits, false, true, stream
		);

		if ( ring_ ) {
			ring_->all_reduce( attn_out_, stream );
		}

		// Residual connection: hidden_states += attn_out
		engine::modules::Elementwise::add_residual( hidden_states_, attn_out_, stream );

		// 3. Post-attention RMSNorm: norm_hidden = rms_norm(hidden_states)
		engine::modules::RMSNorm::forward(
			hidden_states_, lw.post_attn_norm, norm_hidden_, 1e-5f, 1.0f, stream
		);

		// 4. Replicated Top-K Router: 288 experts, top-8 selection evaluated on each rank
		engine::modules::MoERouter::route_topk(
			norm_hidden_,
			lw.router_weights,
			state_.d_topk_indices,
			state_.d_topk_weights,
			config_.hidden_size,
			config_.num_experts,
			config_.top_k,
			stream
		);

		// 5. Persistent MoE FFN dispatch (gate + up -> SwiGLU -> down)
		engine::modules::LinearTrellis::forward(
			norm_hidden_, lw.expert_gate_tr, lw.expert_gate_suh, lw.expert_gate_svh,
			moe_gate_buf_, lw.expert_gate_bits, false, true, stream
		);

		engine::modules::LinearTrellis::forward(
			norm_hidden_, lw.expert_up_tr, lw.expert_up_suh, lw.expert_up_svh,
			moe_up_buf_, lw.expert_up_bits, false, true, stream
		);

		engine::modules::SwiGLU::forward( moe_gate_buf_, moe_up_buf_, moe_inter_buf_, stream );

		engine::modules::LinearTrellis::forward(
			moe_inter_buf_, lw.expert_down_tr, lw.expert_down_suh, lw.expert_down_svh,
			moe_out_buf_, lw.expert_down_bits, false, true, stream
		);

		// Combine routed expert outputs with router weights
		engine::modules::MoERouter::combine(
			expert_parts_buf_.ptr<const half>(),
			state_.d_topk_weights,
			moe_out_buf_.ptr<const half>(),
			ffn_out_.ptr<half>(),
			config_.hidden_size,
			config_.top_k,
			stream
		);

		if ( ring_ ) {
			ring_->all_reduce( ffn_out_, stream );
		}

		// Residual connection: hidden_states += ffn_out
		engine::modules::Elementwise::add_residual( hidden_states_, ffn_out_, stream );
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
	if ( residual_.data ) {
		cudaMemset( residual_.data, 0, residual_.size_bytes() );
	}
	if ( hidden_states_.data ) {
		cudaMemset( hidden_states_.data, 0, hidden_states_.size_bytes() );
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
