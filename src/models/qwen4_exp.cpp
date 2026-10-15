#include <condition_variable>
#include "engine/models/qwen4_exp.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/modules/hessian.h"
#include <memory>
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/gdn.h"
#include "engine/modules/gated_residual.h"
#include "engine/modules/vision.h"
#include "engine/modules/embedding.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/ple.h"
#include "engine/modules/qsa.h"
#include "engine/modules/dsa.h"
#include "engine/modules/attention.h"
#include "engine/spec/dflash.h"
#include "engine/modules/norm.h"
#include "engine/modules/rope.h"
#include <mutex>
#include "engine/modules/linear.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <vector>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <sstream>

namespace engine::models {

namespace {
// Row count of the dense fp16 GEMMs of a prefill chunk. CUDA runs them at the full row capacity (one cuBLAS algorithm, so a row's result does not depend on its
// chunk); the CPU and Vulkan GEMMs have no algorithm choice and run the rows there are (the full capacity costs hundreds of times the work)
inline int gemm_rows( int m ) {
#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1
	( void )m;
	return pod::MAX_ROWS;
#else
	return m;
#endif
}
} // namespace

namespace {

// Checks the hard-coded defaults of Qwen4ExpConfig against the checkpoint's config.json
bool validate_config( const char* model_dir, const Qwen4ExpConfig& c ) {
	std::ifstream in( std::string( model_dir ) + "/config.json" );
	if ( !in ) {
		LOG_ERROR( "Qwen4ExpPipeline: %s/config.json is unreadable\n", model_dir );
		return false;
	}
	std::stringstream ss;
	ss << in.rdbuf();
	const auto j = nlohmann::json::parse( ss.str(), nullptr, false );
	if ( j.is_discarded() ) return false;
	const auto& t = j.contains( "text_config" ) ? j["text_config"] : j;
	bool ok = true;
	auto check = [&]( const char* key, int want ) {
		const int got = t.value( key, -1 );
		if ( got != want ) {
			LOG_ERROR( "Qwen4ExpPipeline: config %s is %d, the engine is built for %d\n", key, got, want );
			ok = false;
		}
	};
	check( "num_hidden_layers", c.num_layers );
	check( "hidden_size", c.hidden_size );
	check( "vocab_size", c.vocab_size );
	check( "hc_count", c.hc_mult );
	check( "num_attention_heads", c.num_q_heads );
	check( "num_key_value_heads", c.num_kv_heads );
	check( "head_dim", c.head_dim );
	check( "linear_num_key_heads", c.linear_num_k_heads );
	check( "linear_num_value_heads", c.linear_num_v_heads );
	check( "linear_key_head_dim", c.linear_head_dim );
	check( "num_experts", c.num_experts );
	check( "num_experts_per_tok", c.top_k );
	check( "moe_intermediate_size", c.moe_intermediate );
	check( "shared_expert_intermediate_size", c.shared_intermediate );
	check( "indexer_n_heads", c.indexer_heads );
	check( "indexer_head_dim", c.indexer_head_dim );
	check( "indexer_budget", c.indexer_budget );
	check( "indexer_compress_ratio", c.indexer_compress );
	return ok;
}

// The router kernel reads the gate as [hidden, experts]; the checkpoint stores [experts, hidden]
pod::TensorView load_router_transposed( engine::loader::SafeTensorsRepo& repo, const char* name, pod::MemoryArena& arena ) {
	int shard_idx = -1;
	const auto* meta = repo.find( name, &shard_idx );
	if ( !meta || shard_idx < 0 ) return pod::TensorView{};
	const int E = meta->shape[0], H = meta->shape[1];
	const void* raw = engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard_idx ), meta );
	if ( !raw || ( meta->dtype != pod::DType::FP16 && meta->dtype != pod::DType::BF16 ) ) return pod::TensorView{};
	// FP16 as stored, or BF16 (the NVFP4 checkpoint) converted: a BF16 gate read as FP16 routes every token to the wrong experts
	std::vector<backend::half_t> conv;
	const backend::half_t* src = static_cast<const backend::half_t*>( raw );
	if ( meta->dtype == pod::DType::BF16 ) {
		conv.resize( static_cast<size_t>( E ) * H );
		const uint16_t* bf = static_cast<const uint16_t*>( raw );
		for ( size_t i = 0; i < conv.size(); ++i ) {
			const uint32_t u = static_cast<uint32_t>( bf[i] ) << 16;
			float f;
			std::memcpy( &f, &u, 4 );
			conv[i] = backend::from_float_half( f );
		}
		src = conv.data();
	}
	int shape[2] = { H, E };
	pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape, 2 );
	if ( tv.data != nullptr ) {
		std::vector<backend::half_t> t( static_cast<size_t>( H ) * E );
		for ( int e = 0; e < E; ++e ) for ( int d = 0; d < H; ++d ) t[static_cast<size_t>( d ) * E + e] = src[static_cast<size_t>( e ) * H + d];
		backend::memcpy_sync( tv.data, t.data(), t.size() * sizeof( backend::half_t ), backend::Copy::H2D );
	}
	return tv;
}

engine::loader::NgramTable& ngram_table() {
	static engine::loader::NgramTable table;
	return table;
}

} // namespace

Qwen4ExpPipeline::Qwen4ExpPipeline() = default;
Qwen4ExpPipeline::~Qwen4ExpPipeline() {
	if ( arena_ != nullptr ) {
		backend::set_device( rank_ );
		hessian_dump();
		graphs_.reset();
		if ( side_stream_ != nullptr ) backend::stream_destroy( side_stream_ );
		if ( full_logits_mem_ != nullptr ) backend::device_free( full_logits_mem_ );
		engine::core::arena_destroy( host_arena_ );
	}
}

bool Qwen4ExpPipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	rank_ = device_id;
	arena_ = &arena;
	ring_ = ring;
#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1
	side_stream_ = backend::stream_create();
#endif
	tp_ = ( ring != nullptr ) ? requested_tp_ : 1;
	if ( runtime::options().max_seq > 0 ) config_.max_seq = runtime::options().max_seq;
	if ( runtime::options().max_slots > 0 ) config_.max_slots = runtime::options().max_slots;
	if ( tp_ != 1 && tp_ != 2 && tp_ != 4 ) {
		LOG_ERROR( "Qwen4ExpPipeline: tp=%d unsupported (experts are placed whole, num_experts / tp per rank: tp 1, 2 or 4)\n", tp_ );
		return false;
	}
	backend::set_device( rank_ );
	layers_.assign( config_.num_layers, Qwen4ExpLayerWeights{} );
	for ( int L = 0; L < config_.num_layers; ++L ) layers_[L].is_full_attention = config_.is_full_attention( L );

	// Vocabulary-parallel head, padded to a multiple of 256 columns like the other models
	lm_shard_real_cols_ = config_.vocab_size / tp_;
	lm_shard_cols_ = lm_shard_real_cols_ + ( 256 - lm_shard_real_cols_ % 256 ) % 256;

	const int R = pod::MAX_ROWS;
	const int H = config_.hidden_size;
	const int S = config_.max_slots;
	q_heads_ = config_.num_q_heads / tp_;
	kv_heads_ = tp_ >= 2 ? 1 : 2;   // 2 KV heads: with tp 4 ranks 2j and 2j + 1 each hold head j, with tp 2 rank j holds head j, with tp 1 the one rank holds both
	lin_k_heads_ = config_.linear_num_k_heads / tp_;
	lin_v_heads_ = config_.linear_num_v_heads / tp_;
	lin_channels_ = config_.linear_channels / tp_;
	int linear_layers = 0;
	for ( int L = 0; L < config_.num_layers; ++L ) linear_layers += config_.is_full_attention( L ) ? 0 : 1;

	// Recurrent GDN state and conv history per slot
	state_.slot_gdn_floats = static_cast<size_t>( linear_layers ) * lin_v_heads_ * 128 * 128;
	state_.gdn_states = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( S ) * state_.slot_gdn_floats * sizeof( float ) ) );
	backend::memset_sync( state_.gdn_states, 0, static_cast<size_t>( S ) * state_.slot_gdn_floats * sizeof( float ) );
	state_.slot_conv_floats = static_cast<size_t>( linear_layers ) * lin_channels_ * 3;
	state_.conv_history = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( S ) * state_.slot_conv_floats * sizeof( float ) ) );
	backend::memset_sync( state_.conv_history, 0, static_cast<size_t>( S ) * state_.slot_conv_floats * sizeof( float ) );
	state_.gdn_snap = static_cast<float*>( engine::core::arena_alloc( arena, kSnapRows * state_.slot_gdn_floats * sizeof( float ) ) );
	state_.conv_snap = static_cast<float*>( engine::core::arena_alloc( arena, kSnapRows * state_.slot_conv_floats * sizeof( float ) ) );
	state_.d_snap_on = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	backend::memset_sync( state_.d_snap_on, 0, sizeof( int ) );

	int full_layers = 0;
	for ( int L = 0; L < config_.num_layers; ++L ) full_layers += config_.is_full_attention( L ) ? 1 : 0;
	mtp_on_ = runtime::options().speculator == "mtp";
	attn_layers_ = full_layers + ( mtp_on_ ? 1 : 0 );   // the MTP block's attention is the last layer of the KV pool and the QSA state
	kv_pool_.init( rank_, arena, S, config_.max_seq, attn_layers_, kv_heads_, config_.head_dim );
	d_token_ids_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
	d_rows_ = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );
	int h_shape[2] = { R, H };
	embed_rows_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	mixed_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	y_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	int st_shape[3] = { R, config_.hc_mult, H };
	streams_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, st_shape, 3 );
	post_ = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.hc_mult * sizeof( float ) ) );
	gr_scratch_ = static_cast<float*>( engine::core::arena_alloc( arena,
		engine::modules::GatedResidual::scratch_floats( R, H, config_.hc_mult, config_.hc_rank ) * sizeof( float ) ) );
	{
		const int flat = config_.hc_mult * H;
		int s_flat[2] = { R, flat }, s_z[2] = { R, config_.hc_rank + 8 }, s_t[2] = { R, config_.hc_rank };
		gr16_hi_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, s_flat, 2 );
		gr16_lo_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, s_flat, 2 );
		gr16_u_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, s_flat, 2 );
		gr16_z_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, s_z, 2 );
		gr16_thi_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, s_t, 2 );
		gr16_tlo_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, s_t, 2 );
	}
	int qg_shape[2] = { R, q_heads_ * config_.head_dim * 2 };
	q_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qg_shape, 2 );
	int q_shape[2] = { R, q_heads_ * config_.head_dim };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	gate_attn_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	int kv_shape[2] = { R, kv_heads_ * config_.head_dim };
	k_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
	v_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
	state_.gdn_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, state_.slot_gdn_floats * sizeof( float ) ) );
	state_.conv_checkpoint = static_cast<float*>( engine::core::arena_alloc( arena, state_.slot_conv_floats * sizeof( float ) ) );
	if ( mtp_on_ ) {
		int e_shape[2] = { R, H };
		mtp_emb_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, e_shape, 2 );
		mtp_xe_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, e_shape, 2 );
		mtp_e2_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, e_shape, 2 );
		int s_shape[2] = { R * config_.hc_mult, H };
		mtp_hn_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, s_shape, 2 );
		mtp_h2_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, s_shape, 2 );
		const size_t stack = static_cast<size_t>( config_.hc_mult ) * H * sizeof( float );
		mtp_src_ = static_cast<float*>( engine::core::arena_alloc( arena, R * stack ) );
		mtp_hist_ = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( config_.max_slots ) * 2 * ( kDraftM + 1 ) * stack ) );   // a region per slot (the runner offsets by slot)
		mtp_chain_ = static_cast<float*>( engine::core::arena_alloc( arena, kSnapRows * stack ) );
	}
	// QSA indexer state
	{
		qsa_max_pools_ = config_.max_seq / engine::modules::QSA::POOL;
		qsa_ring_slot_ = static_cast<size_t>( attn_layers_ ) * engine::modules::QSA::RING * engine::modules::QSA::DIM;
		qsa_pool_slot_ = static_cast<size_t>( attn_layers_ ) * qsa_max_pools_ * engine::modules::QSA::DIM;
		qsa_ring_ = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, static_cast<size_t>( S ) * qsa_ring_slot_ * sizeof( backend::half_t ) ) );
		qsa_pool_ = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, static_cast<size_t>( S ) * qsa_pool_slot_ * sizeof( backend::half_t ) ) );
		backend::memset_sync( qsa_ring_, 0, static_cast<size_t>( S ) * qsa_ring_slot_ * sizeof( backend::half_t ) );
		backend::memset_sync( qsa_pool_, 0, static_cast<size_t>( S ) * qsa_pool_slot_ * sizeof( backend::half_t ) );
		int qk_shape[2] = { R, engine::modules::QSA::PROJ };
		qsa_qk_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qk_shape, 2 );
		int qq_shape[2] = { R, engine::modules::QSA::HEADS * engine::modules::QSA::DIM };
		qsa_q_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qq_shape, 2 );
		qsa_scores_ = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * qsa_max_pools_ * sizeof( backend::half_t ) ) );
		qsa_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * engine::modules::DSA::KPAD * sizeof( int ) ) );
		qsa_nkeys_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * sizeof( int ) ) );
	}
	// Logits (vocabulary-parallel head) and the sampling scratch
	{
		int logit_shape[2] = { R, lm_shard_cols_ };
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
	}
	// N-gram embedding layer buffers and per-slot histories
	{
		int e_shape[2] = { R, config_.ple_embed_dim };
		ple_emb_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, e_shape, 2 );
		int k_shape[2] = { R, config_.hc_mult * H };
		ple_key_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, k_shape, 2 );
		int v_shape[2] = { R, H };
		ple_value_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, v_shape, 2 );
		ple_gated_ = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.hc_mult * H * sizeof( float ) ) );
		ple_normed_ = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.hc_mult * H * sizeof( backend::half_t ) ) );
		const size_t ring_bytes = static_cast<size_t>( S ) * engine::modules::PLE::RING * config_.hc_mult * H * sizeof( backend::half_t );
		ple_ring_ = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, ring_bytes ) );
		backend::memset_sync( ple_ring_, 0, ring_bytes );
		hist_tok_.assign( S, std::vector<int>( config_.max_seq, 0 ) );
		hist_seg_.assign( S, std::vector<int>( config_.max_seq, 0 ) );
	}
	// MoE: router outputs, the dispatch / expert scratch for this rank's experts, and the shared expert's buffers
	topk_idx_ = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.top_k * sizeof( int ) ) );
	topk_w_ = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * config_.top_k * sizeof( backend::half_t ) ) );
	moe_scratch_ = engine::modules::MoE::alloc_scratch( arena, config_.num_experts / tp_, config_.top_k, H, config_.moe_intermediate, R, !runtime::options().nvfp4_experts.empty() );
	int sh_shape[2] = { R, config_.shared_intermediate };
	sh_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sh_shape, 2 );
	sh_up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sh_shape, 2 );
	sh_inter_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sh_shape, 2 );
	sh_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	sh_gate_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, h_shape, 1 );
	int qkv_shape[2] = { R, lin_channels_ };
	qkv_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );
	conv_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );
	int z_shape[2] = { R, lin_v_heads_ * config_.linear_head_dim };
	z_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	gdn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	int gp_shape[1] = { R * lin_v_heads_ };
	gdn_gate_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gp_shape, 1 );
	gdn_beta_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gp_shape, 1 );
#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );
#endif

	metadata_.model_name = "qwen4_exp";
	metadata_.hidden_size = config_.hidden_size;
	metadata_.vocab_size = config_.vocab_size;
	metadata_.num_layers = config_.num_layers;
	metadata_.num_heads = config_.num_q_heads;
	metadata_.num_kv_heads = config_.num_kv_heads;
	metadata_.head_dim = config_.head_dim;
	metadata_.max_seq_len = config_.max_seq;
	metadata_.tp_size = tp_;
	metadata_.max_slots = config_.max_slots;
	metadata_.max_rows = pod::MAX_ROWS;
	metadata_.eos_ids[0] = 248044;
	metadata_.eos_ids[1] = 248046;
	return true;
}

bool Qwen4ExpPipeline::load_weights( const char* model_dir, backend::stream_t stream ) {
	if ( !validate_config( model_dir, config_ ) ) return false;
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) {
		LOG_ERROR( "[loader] FAILED to open model directory %s\n", model_dir );
		return false;
	}
	std::unique_ptr<engine::loader::SafeTensorsRepo> nv_repo;
	if ( const std::string& nv_dir = runtime::options().nvfp4_experts; !nv_dir.empty() ) {
		nv_repo = std::make_unique<engine::loader::SafeTensorsRepo>();
		if ( !nv_repo->open( nv_dir.c_str() ) ) {
			LOG_ERROR( "[loader] FAILED to open the NVFP4 expert directory %s\n", nv_dir.c_str() );
			return false;
		}
		LOG_INFO( "[RANK %d] routed experts from the NVFP4 checkpoint %s\n", rank_, nv_dir.c_str() );
	}
	std::unique_ptr<engine::loader::SafeTensorsRepo> ov_repo;
	if ( const std::string& ov_dir = runtime::options().nvfp4_dense_dir; !ov_dir.empty() ) {
		ov_repo = std::make_unique<engine::loader::SafeTensorsRepo>();
		if ( !ov_repo->open( ov_dir.c_str() ) ) {
			LOG_ERROR( "[loader] FAILED to open the calibrated NVFP4 directory %s\n", ov_dir.c_str() );
			return false;
		}
	}
	backend::set_device( rank_ );
	const int H = config_.hidden_size;
	const int tp = tp_;
	int missing = 0;
	char nm[256];
	if ( !runtime::options().hessian_dir.empty() ) {
		// Calibration accumulators (sites per layer: see hessian()); the replicated sites live on rank 0, the o_proj / out_proj inputs on every rank
		hess_.assign( static_cast<size_t>( config_.num_layers ) * 5, HessianSite{} );
		auto site = [&]( int layer, int idx, int k ) {
			HessianSite& hs = hess_[static_cast<size_t>( layer ) * 5 + idx];
			hs.k = k;
			hs.h = static_cast<float*>( engine::core::arena_alloc( *arena_, static_cast<size_t>( k ) * k * sizeof( float ), 128 ) );
			backend::memset_async( hs.h, 0, static_cast<size_t>( k ) * k * sizeof( float ), stream );
		};
		for ( int L = 0; L < config_.num_layers; ++L ) {
			if ( rank_ == 0 ) { site( L, 0, config_.hidden_size ); site( L, 1, config_.hidden_size ); site( L, 3, config_.shared_intermediate ); }
			site( L, 2, config_.is_full_attention( L ) ? q_heads_ * config_.head_dim : lin_v_heads_ * 128 );
		}
		if ( rank_ == 0 ) site( 0, 4, config_.hidden_size );
		backend::stream_sync( stream );
	}

	// The MTP head comes from the NVFP4 checkpoint when mtp_nvfp4 is set: its dense tensors are BF16 there and its experts FP8 with block scales
	const bool mtp_nv = mtp_on_ && runtime::options().mtp_nvfp4;
	if ( mtp_nv && !nv_repo ) {
		LOG_ERROR( "[RANK %d] mtp_nvfp4 needs nvfp4_experts (the checkpoint that holds the head)\n", rank_ );
		return false;
	}
	engine::loader::SafeTensorsRepo* src = &repo;   // where dense tensors are read from: the model directory, or the NVFP4 checkpoint for the MTP head
	auto dense = [&]( const char* name ) {
		const pod::TensorView t = src->load( name, *arena_, stream, pod::DType::FP16 );
		if ( !t.data ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, name ); ++missing; }
		return t;
	};
	auto dense_slice = [&]( const char* name ) {
		const pod::TensorView t = src->load_slice( name, *arena_, 0, rank_, tp, stream, pod::DType::FP16 );
		if ( !t.data ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, name ); ++missing; }
		return t;
	};
	auto check = [&]( const pod::LinearWeight& w, const char* name ) {
		if ( !w.is_valid() ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, name ); ++missing; }
		return w;
	};
	auto load_gr = [&]( const char* prefix_in, bool inject ) {
		Qwen4ExpGatedResidual g;
		char prefix[200], n2[256];
		std::snprintf( prefix, sizeof( prefix ), "%s", prefix_in );   // callers pass nm itself
		std::snprintf( n2, sizeof( n2 ), "%s.hc_norm.weight", prefix ); g.norm = dense( n2 );
		std::snprintf( n2, sizeof( n2 ), "%s.input_mix_weight_down.weight", prefix ); g.down = dense( n2 );
		std::snprintf( n2, sizeof( n2 ), "%s.input_mix_weight_up.weight", prefix ); g.up = dense( n2 );
		if ( inject ) { std::snprintf( n2, sizeof( n2 ), "%s.block_inject_weight.weight", prefix ); g.inject = dense( n2 ); }
		if ( g.down.data && g.up.data ) {
			// Dense fp16 views for the tensor-core path: [rank (+ hc, padded to 8), flat] row-major = an [n, k] matrix tagged [k, n]
			const int flat = config_.hc_mult * config_.hidden_size, rank = config_.hc_rank;
			const int n_cat = inject ? rank + 8 : rank;
			const size_t row_bytes = static_cast<size_t>( flat ) * sizeof( backend::half_t );
			auto* cat = static_cast<char*>( engine::core::arena_alloc( *arena_, static_cast<size_t>( n_cat ) * row_bytes, 128 ) );
			backend::memcpy_async( cat, g.down.data, static_cast<size_t>( rank ) * row_bytes, backend::Copy::D2D, stream );
			if ( inject ) {
				backend::memcpy_async( cat + static_cast<size_t>( rank ) * row_bytes, g.inject.data, static_cast<size_t>( config_.hc_mult ) * row_bytes, backend::Copy::D2D, stream );
				backend::memset_async( cat + static_cast<size_t>( rank + config_.hc_mult ) * row_bytes, 0, static_cast<size_t>( 8 - config_.hc_mult ) * row_bytes, stream );
			}
			g.down_lw = pod::LinearWeight::make_fp16( pod::make_tensor_2d( cat, pod::DType::FP16, flat, n_cat ) );
			g.up_lw = pod::LinearWeight::make_fp16( pod::make_tensor_2d( g.up.data, pod::DType::FP16, rank, flat ) );
		}
		return g;
	};
	// Column-parallel (output split) and row-parallel (input split) trellis projections
	auto load_col = [&]( const char* name, int bits ) { return check( repo.load_linear_slice( name, *arena_, bits, 1, rank_, tp, stream ), name ); };
	auto load_row = [&]( const char* name, int bits ) { return check( repo.load_linear_slice( name, *arena_, bits, 0, rank_, tp, stream ), name ); };
	// Dense groups quantized to NVFP4 at load from the BF16 tensors of the NVFP4 checkpoint (ENGINE_NVFP4_DENSE); otherwise the trellis tensors
	const std::string& nv_groups = runtime::options().nvfp4_dense;
	bool in_mtp = false;   // the MTP head keeps its own tensors: EXL3, or (mtp_nvfp4) quantized from the NVFP4 checkpoint's FP8 / BF16 ones
	// A group is on when named in the list, or "all"; `shared` also covers its parts shared_gate / shared_up / shared_down
	auto nv_listed = [&]( const std::string& g ) { return ( "," + nv_groups + "," ).find( "," + g + "," ) != std::string::npos; };
	auto nv_on = [&]( const char* group ) {
		if ( !nv_repo ) return false;
		if ( in_mtp ) return mtp_nv && std::strcmp( group, "head" ) != 0;   // the whole head, every projection
		if ( nv_groups.empty() ) return false;
		const std::string g = group;
		return nv_groups == "all" || nv_listed( g ) || ( g.rfind( "shared_", 0 ) == 0 && nv_listed( "shared" ) );
	};
	int nv_calibrated = 0, nv_rtn = 0;
	auto nv_slice = [&]( const char* name, bool split_n, int rank, int parts, int pad_n ) -> pod::LinearWeight {
		const std::string key = std::string( name ) + ".weight_scale_2";
		if ( ov_repo && ov_repo->find( key.c_str() ) ) {
			++nv_calibrated;
			return check( ov_repo->load_nvfp4_slice( name, *arena_, split_n, rank, parts, stream, pad_n ), name );
		}
		++nv_rtn;
		return check( nv_repo->quantize_nvfp4_slice( name, *arena_, split_n, rank, parts, pad_n, stream ), name );
	};
	auto load_col_g = [&]( const char* group, const char* name, int bits ) {
		return nv_on( group ) ? nv_slice( name, true, rank_, tp, 0 ) : load_col( name, bits );
	};
	auto load_row_g = [&]( const char* group, const char* name, int bits ) {
		return nv_on( group ) ? nv_slice( name, false, rank_, tp, 0 ) : load_row( name, bits );
	};
	auto load_full_g = [&]( const char* group, const char* name, int bits ) {
		return nv_on( group ) ? nv_slice( name, true, 0, 1, 0 ) : check( repo.load_linear( name, *arena_, bits, stream ), name );
	};

	const char* P = "model.language_model";
	std::snprintf( nm, sizeof( nm ), "%s.embed_tokens.weight", P );
	embed_table_ = repo.load( nm, *arena_, stream, pod::DType::BF16 );
	if ( !embed_table_.data ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, nm ); ++missing; }
	std::snprintf( nm, sizeof( nm ), "%s.hyper_connection_mixer", P );
	final_mixer_ = load_gr( nm, false );
	if ( nv_on( "head" ) ) {
		const int cnt = lm_shard_real_cols_, beg = rank_ * cnt;
		const std::string key = "lm_head.weight_scale_2";
		const bool cal = ov_repo && ov_repo->find( key.c_str() );
		++( cal ? nv_calibrated : nv_rtn );
		lm_head_ = cal ? check( ov_repo->load_nvfp4( "lm_head", *arena_, &beg, &cnt, 1, 0, config_.hidden_size, stream, lm_shard_cols_ ), "lm_head" )
			: check( nv_repo->quantize_nvfp4( "lm_head", *arena_, &beg, &cnt, 1, 0, config_.hidden_size, lm_shard_cols_, stream ), "lm_head" );
	}
	else lm_head_ = check( repo.load_linear_slice_padded( "lm_head", *arena_, 6, rank_, tp, lm_shard_cols_ - lm_shard_real_cols_, stream ), "lm_head" );

	// GDN projections: q / k / v row groups of 128 / 128 / 384 blocks of 16, every rank takes its share of each
	static const int QKV_BLOCKS[3] = { 128, 128, 384 };
	const int qkv_rank_blocks[3] = { 128 / tp, 128 / tp, 384 / tp };
	static const int CONV_CHANNELS[3] = { 2048, 2048, 6144 };
	const int conv_rank_channels[3] = { 2048 / tp, 2048 / tp, 6144 / tp };

	{
		char pre[160];
		std::snprintf( pre, sizeof( pre ), "%s.layers.%d.ple", P, config_.ple_layer );
		std::snprintf( nm, sizeof( nm ), "%s.key_proj", pre ); ple_.key_proj = check( repo.load_linear( nm, *arena_, 16, stream ), nm );
		std::snprintf( nm, sizeof( nm ), "%s.value_proj", pre ); ple_.value_proj = check( repo.load_linear( nm, *arena_, 16, stream ), nm );
		std::snprintf( nm, sizeof( nm ), "%s.norm_key.weight", pre ); ple_.norm_key = dense( nm );
		std::snprintf( nm, sizeof( nm ), "%s.norm_query.weight", pre ); ple_.norm_query = dense( nm );
		std::snprintf( nm, sizeof( nm ), "%s.norm_conv.weight", pre ); ple_.norm_conv = dense( nm );
		std::snprintf( nm, sizeof( nm ), "%s.conv1d.weight", pre ); ple_.conv_w = dense( nm );
		static std::once_flag once;
		static bool table_ok = true;
		std::call_once( once, [&]() {
			// The conversion's own file (trellis rows), or, for the normal NVFP4 checkpoint used as the model directory, its FP8 rows
			const std::string file = std::string( model_dir ) + "/ngram_embedding.safetensors";
			if ( ::access( file.c_str(), R_OK ) == 0 ) table_ok = ngram_table().open( file );
			else {
				// The rows stay mapped for the life of the process, like the trellis file: a repo of their own that is never closed
				static engine::loader::SafeTensorsRepo table_repo;
				table_ok = table_repo.open( model_dir ) && ngram_table().open_fp8( table_repo, std::string( P ) + ".layers." + std::to_string( config_.ple_layer ) + ".ple.ple_embedding" );
			}
		} );
		if ( !table_ok ) { LOG_ERROR( "[RANK %d] the n-gram table failed to open\n", rank_ ); ++missing; }
	}
	// One decoder block (a trunk layer or the MTP head's): gated residual sites, attention or GDN, MoE
	auto load_block = [&]( Qwen4ExpLayerWeights& lw, const char* pre ) {
		std::snprintf( nm, sizeof( nm ), "%s.attn_hyper_connection", pre ); lw.attn_hc = load_gr( nm, true );
		std::snprintf( nm, sizeof( nm ), "%s.mlp_hyper_connection", pre ); lw.mlp_hc = load_gr( nm, true );

		if ( lw.is_full_attention ) {
			const int kv_slices = tp >= 2 ? 2 : 1, kv_slice = tp >= 2 ? rank_ / ( tp / 2 ) : 0;
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.q_proj", pre ); lw.q_proj = load_col_g( "attn", nm, 8 );
			// 2 KV heads: ranks 2j and 2j + 1 (tp 4) or rank j (tp 2) take head j (256 of the 512 columns), a single rank takes both
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.k_proj", pre );
			lw.k_proj = nv_on( "attn" ) ? nv_slice( nm, true, kv_slice, kv_slices, 0 ) : check( repo.load_linear_slice( nm, *arena_, 8, 1, kv_slice, kv_slices, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.v_proj", pre );
			lw.v_proj = nv_on( "attn" ) ? nv_slice( nm, true, kv_slice, kv_slices, 0 ) : check( repo.load_linear_slice( nm, *arena_, 8, 1, kv_slice, kv_slices, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.o_proj", pre ); lw.o_proj = load_row_g( "attn", nm, 8 );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.q_norm.weight", pre ); lw.q_norm = dense( nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.k_norm.weight", pre ); lw.k_norm = dense( nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.indexer.index_qk_proj", pre );
			lw.idx_qk_proj = load_full_g( "idx", nm, 8 );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.indexer.q_layernorm.weight", pre ); lw.idx_q_norm = dense( nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.indexer.k_layernorm.weight", pre ); lw.idx_k_norm = dense( nm );
		}
		else {
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.in_proj_qkv", pre );
			if ( nv_on( "gdn" ) ) {
				// q / k / v row groups of 2048 / 2048 / 6144, every rank takes its share of each
				const int rows[3] = { 2048 / tp, 2048 / tp, 6144 / tp }, begin[3] = { rank_ * rows[0], 2048 + rank_ * rows[1], 4096 + rank_ * rows[2] };
				const std::string key = std::string( nm ) + ".weight_scale_2";
				const bool cal = ov_repo && ov_repo->find( key.c_str() );
				++( cal ? nv_calibrated : nv_rtn );
				lw.in_proj_qkv = cal ? check( ov_repo->load_nvfp4( nm, *arena_, begin, rows, 3, 0, config_.hidden_size, stream ), nm )
					: check( nv_repo->quantize_nvfp4( nm, *arena_, begin, rows, 3, 0, config_.hidden_size, 0, stream ), nm );
			}
			else lw.in_proj_qkv = check( repo.load_grouped_trellis_slice( nm, *arena_, QKV_BLOCKS, qkv_rank_blocks, 3, rank_, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.in_proj_z", pre ); lw.in_proj_z = load_col_g( "gdn", nm, 8 );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.out_proj", pre ); lw.out_proj = load_row_g( "gdn", nm, 8 );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.conv1d.weight", pre );
			lw.conv1d_w = repo.load_grouped_slice( nm, *arena_, CONV_CHANNELS, conv_rank_channels, 3, rank_, 0, stream );
			if ( !lw.conv1d_w.data ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, nm ); ++missing; }
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.A_log", pre ); lw.A_log = dense_slice( nm );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.dt_bias", pre ); lw.dt_bias = dense_slice( nm );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.in_proj_a.weight", pre ); lw.in_proj_a = dense_slice( nm );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.in_proj_b.weight", pre ); lw.in_proj_b = dense_slice( nm );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.norm.weight", pre ); lw.linear_norm = dense( nm );
		}

		// MoE: replicated router, 128 whole experts per rank, shared expert (replicated, run on one rank) and its gate
		std::snprintf( nm, sizeof( nm ), "%s.mlp.gate.weight", pre );
		lw.router = load_router_transposed( *src, nm, *arena_ );
		if ( !lw.router.data ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, nm ); ++missing; }
		auto& mw = lw.experts;
		const bool nv_experts = nv_repo && ( !in_mtp || mtp_nv );
		mw.type = nv_experts ? pod::QuantType::NVFP4 : pod::QuantType::TRELLIS;
		const int per_rank = config_.num_experts / tp;
		ENGINE_ASSERT( per_rank <= MOE_MAX_EXPERTS, "Qwen4ExpPipeline: experts per rank exceed MOE_MAX_EXPERTS" );
		mw.num_local_experts = per_rank;
		mw.top_k = config_.top_k;
		// Tiered experts (ENGINE_EXPERT_CACHE < 1): the codes of resident experts go to device slots, the rest to pinned host memory; the scales
		// (NVFP4) and sign vectors (trellis) stay on the device
		const double tier_fraction = engine::modules::ExpertTier::resident_fraction();
		ENGINE_ASSERT( tier_fraction > 0.0, "Qwen4Exp: expert_cache=auto is for GLM; give the resident fraction" );
		int tier_layer = -1;
		if ( tier_fraction < 1.0 && !in_mtp ) {
			if ( tier_layers_ == 0 ) {
				size_t bytes[3];
				const char* kinds[3] = { "gate_proj", "up_proj", "down_proj" };
				for ( int p = 0; p < 3; ++p ) {
					if ( nv_experts ) bytes[p] = static_cast<size_t>( config_.hidden_size ) * config_.moe_intermediate / 2;
					else {
						std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.%s", pre, rank_ * per_rank, kinds[p] );
						bytes[p] = repo.trellis_bytes( nm );
						if ( bytes[p] == 0 && runtime::options().expert_synth >= 2 ) bytes[p] = repo.synth_trellis_bytes( nm );
						ENGINE_ASSERT( bytes[p] > 0, "Qwen4ExpPipeline: expert codes not found for the tier" );
					}
				}
				tier_.start( per_rank, tier_fraction, bytes, rank_, tp_, config_.num_layers, false, host_arena_ );
			}
			tier_.add_layer( &mw, *arena_, host_arena_, tier_.initial_set( tier_layers_ ) );
			tier_layer = tier_layers_++;
		}
		for ( int e = 0; e < per_rank; ++e ) {
			const int gid = rank_ * per_rank + e;
			pod::MemoryArena ca[3];
			if ( tier_layer >= 0 ) for ( int p = 0; p < 3; ++p ) tier_.code_arena( tier_layer, e, p, ca[p] );
			pod::MemoryArena* cp[3] = { tier_layer >= 0 ? &ca[0] : nullptr, tier_layer >= 0 ? &ca[1] : nullptr, tier_layer >= 0 ? &ca[2] : nullptr };
			if ( runtime::options().expert_synth >= 2 && !nv_experts && tier_layer >= 0 ) {
				// Timing only: experts that are not in the checkpoint. The resident ones are generated, a cold one is a copy of a resident one
				if ( e >= tier_.resident ) {
					mw.gate[e] = mw.gate[e % tier_.resident];
					mw.up[e] = mw.up[e % tier_.resident];
					mw.down[e] = mw.down[e % tier_.resident];
					continue;
				}
				static const char* const kinds[3] = { "gate_proj", "up_proj", "down_proj" };
				pod::LinearWeight* dst[3] = { &mw.gate[e], &mw.up[e], &mw.down[e] };
				for ( int p = 0; p < 3; ++p ) {
					std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.%s", pre, gid, kinds[p] );
					*dst[p] = repo.synth_linear( nm, *arena_, stream, cp[p] );
					ENGINE_ASSERT( dst[p]->w0.data != nullptr, "Qwen4ExpPipeline: no entry for a generated expert in quantization_config.json" );
				}
				continue;
			}
			if ( nv_experts && in_mtp ) {
				// The head's experts are FP8 with 128 x 128 block scales: dequantized and quantized to NVFP4 (round to nearest) here
				std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.gate_proj", pre, gid ); mw.gate[e] = check( nv_repo->quantize_nvfp4_slice( nm, *arena_, true, 0, 1, 0, stream ), nm );
				std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.up_proj", pre, gid ); mw.up[e] = check( nv_repo->quantize_nvfp4_slice( nm, *arena_, true, 0, 1, 0, stream ), nm );
				std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.down_proj", pre, gid ); mw.down[e] = check( nv_repo->quantize_nvfp4_slice( nm, *arena_, true, 0, 1, 0, stream ), nm );
				continue;
			}
			if ( nv_experts ) {
				// modelopt NVFP4 experts: [N, K/2] codes, E4M3 block scales, one FP32 global scale per matrix
				std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.gate_proj", pre, gid ); mw.gate[e] = check( nv_repo->load_nvfp4_slice( nm, *arena_, true, 0, 1, stream, 0, cp[0] ), nm );
				std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.up_proj", pre, gid ); mw.up[e] = check( nv_repo->load_nvfp4_slice( nm, *arena_, true, 0, 1, stream, 0, cp[1] ), nm );
				std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.down_proj", pre, gid ); mw.down[e] = check( nv_repo->load_nvfp4_slice( nm, *arena_, true, 0, 1, stream, 0, cp[2] ), nm );
				continue;
			}
			std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.gate_proj", pre, gid ); mw.gate[e] = check( repo.load_linear( nm, *arena_, 4, stream, cp[0] ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.up_proj", pre, gid ); mw.up[e] = check( repo.load_linear( nm, *arena_, 4, stream, cp[1] ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.down_proj", pre, gid ); mw.down[e] = check( repo.load_linear( nm, *arena_, 4, stream, cp[2] ), nm );
		}
		engine::modules::MoE::init_device_ptrs( mw, *arena_, stream );
		mw.heat = static_cast<uint32_t*>( engine::core::arena_alloc( *arena_, static_cast<size_t>( per_rank ) * sizeof( uint32_t ), 128 ) );
		backend::memset_async( mw.heat, 0, static_cast<size_t>( per_rank ) * sizeof( uint32_t ), stream );
		std::snprintf( nm, sizeof( nm ), "%s.mlp.shared_expert.gate_proj", pre ); lw.shared_gate = load_full_g( "shared_gate", nm, 4 );
		std::snprintf( nm, sizeof( nm ), "%s.mlp.shared_expert.up_proj", pre ); lw.shared_up = load_full_g( "shared_up", nm, 4 );
		std::snprintf( nm, sizeof( nm ), "%s.mlp.shared_expert.down_proj", pre ); lw.shared_down = load_full_g( "shared_down", nm, 4 );
		std::snprintf( nm, sizeof( nm ), "%s.mlp.shared_expert_gate.weight", pre ); lw.shared_expert_gate = dense( nm );
	};
	for ( int L = 0; L < config_.num_layers; ++L ) {
		char pre[160];
		std::snprintf( pre, sizeof( pre ), "%s.layers.%d", P, L );
		load_block( layers_[L], pre );
	}
	if ( mtp_on_ ) {
		// MTP head: one full-attention block with its own gated-residual sites and MoE, the input combine and its own mixer
		mtp_layer_.is_full_attention = true;
		in_mtp = true;
		if ( mtp_nv ) {
			src = nv_repo.get();
			LOG_INFO( "[RANK %d] MTP head from the NVFP4 checkpoint (FP8 experts and BF16 dense tensors quantized to NVFP4)\n", rank_ );
		}
		load_block( mtp_layer_, "mtp.layers.0" );
		mtp_pre_norm_hidden_ = dense( "mtp.pre_fc_norm_hidden.weight" );
		mtp_pre_norm_embedding_ = dense( "mtp.pre_fc_norm_embedding.weight" );
		mtp_fc_hidden_ = mtp_nv ? check( nv_repo->quantize_nvfp4_slice( "mtp.fc_hidden", *arena_, true, 0, 1, 0, stream ), "mtp.fc_hidden" ) : check( repo.load_linear( "mtp.fc_hidden", *arena_, 4, stream ), "mtp.fc_hidden" );
		mtp_fc_embedding_ = mtp_nv ? check( nv_repo->quantize_nvfp4_slice( "mtp.fc_embedding", *arena_, true, 0, 1, 0, stream ), "mtp.fc_embedding" ) : check( repo.load_linear( "mtp.fc_embedding", *arena_, 4, stream ), "mtp.fc_embedding" );
		mtp_mixer_ = load_gr( "mtp.hyper_connection_mixer", false );
	}
	(void) H;
	if ( stream ) backend::stream_sync( stream );
	repo.close();
	if ( nv_calibrated + nv_rtn > 0 ) LOG_INFO( "[RANK %d] NVFP4 dense tensors: %d calibrated, %d round-to-nearest\n", rank_, nv_calibrated, nv_rtn );
	if ( missing > 0 ) {
		LOG_ERROR( "[RANK %d] %d tensors missing\n", rank_, missing );
		return false;
	}
	weights_loaded_ = true;
	return true;
}

bool Qwen4ExpPipeline::load_vision( const char* model_dir, backend::stream_t stream ) {
	if ( rank_ != 0 || runtime::options().max_image_tokens <= 0 ) return false;
	backend::set_device( rank_ );
	return vision_.load( model_dir, *arena_, runtime::options().max_image_tokens * 4, stream );
}

bool Qwen4ExpPipeline::encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( rank_ != 0 || !vision_.loaded() ) return false;
	backend::set_device( rank_ );
	return vision_.encode( image, out, stream );
}

bool Qwen4ExpPipeline::inject_embeddings( const backend::half_t* host_rows, int dst_row, int rows, backend::stream_t stream ) {
	ENGINE_ASSERT( dst_row >= 0 && rows >= 1 && dst_row + rows <= pod::MAX_ROWS, "Qwen4ExpPipeline::inject_embeddings: rows out of range" );
	backend::set_device( rank_ );
	const size_t row_bytes = static_cast<size_t>( config_.hidden_size ) * sizeof( backend::half_t );
	backend::memcpy_async( embed_rows_.ptr<char>() + static_cast<size_t>( dst_row ) * row_bytes, host_rows, static_cast<size_t>( rows ) * row_bytes, backend::Copy::H2D, stream );
	return true;
}

void Qwen4ExpPipeline::set_rows( int m ) {
	embed_rows_.shape[0] = m;
	mixed_.shape[0] = m;
	y_.shape[0] = m;
	streams_.shape[0] = m;
	qkv_buf_.shape[0] = m;
	z_buf_.shape[0] = m;
	gdn_out_.shape[0] = m;
	sh_gate_buf_.shape[0] = m;
	sh_up_buf_.shape[0] = m;
	sh_inter_buf_.shape[0] = m;
	sh_out_.shape[0] = m;
	ple_emb_.shape[0] = m;
	ple_key_.shape[0] = m;
	ple_value_.shape[0] = m;
	if ( logits_.data ) logits_.shape[0] = m;
	qsa_qk_.shape[0] = m;
	qsa_q_.shape[0] = m;
	q_gate_buf_.shape[0] = m;
	q_buf_.shape[0] = m;
	gate_attn_buf_.shape[0] = m;
	attn_out_.shape[0] = m;
	k_buf_.shape[0] = m;
	v_buf_.shape[0] = m;
}

void Qwen4ExpPipeline::embed_tokens( const int* token_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_ROWS, "Qwen4ExpPipeline::embed_tokens: count out of range" );
	backend::set_device( rank_ );
	backend::memcpy_async( d_token_ids_, token_ids, count * sizeof( int ), backend::Copy::H2D, stream );
	host_tokens_.assign( token_ids, token_ids + count );
	set_rows( count );
	engine::modules::Embedding::forward( embed_table_, d_token_ids_, count, embed_rows_, stream );
}

void Qwen4ExpPipeline::forward( const pod::BatchRows& rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "Qwen4ExpPipeline::forward: weights not loaded" );
	ENGINE_ASSERT( rows.count >= 1 && rows.count <= pod::MAX_ROWS, "Qwen4ExpPipeline::forward: row count out of range" );
	backend::set_device( rank_ );
	const int m = rows.count;
	key_hint_ = rows.pos[m - 1] + 1;
	lm_mode_ = ( m > pod::MAX_BATCH ) ? rows.logits_mode : pod::BatchRows::LOGITS_ALL;
	last_rows_ = ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) ? 0 : ( lm_mode_ == pod::BatchRows::LOGITS_LAST ? 1 : m );
	set_rows( m );
	backend::memcpy_async( d_rows_, &rows, rows.upload_bytes(), backend::Copy::H2D, stream );
	attn_tc_ = true;
	for ( int r = 1; r < m; ++r ) attn_tc_ = attn_tc_ && rows.slot[r] == rows.slot[0];
	if ( static const double eps = runtime::options().perturb_embed; eps != 0.0 && m > pod::MAX_BATCH ) {
		// Debug: rounding-scale noise on the embeddings, a function of (position, dim) only (see the GLM pipeline)
		const int H = config_.hidden_size;
		std::vector<backend::half_t> e( static_cast<size_t>( m ) * H );
		backend::stream_sync( stream );
		backend::memcpy_async( e.data(), embed_rows_.ptr<const backend::half_t>(), e.size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
		for ( int r = 0; r < m; ++r ) {
			for ( int d = 0; d < H; ++d ) {
				uint32_t h = static_cast<uint32_t>( rows.pos[r] ) * 2654435761u ^ static_cast<uint32_t>( d ) * 40503u;
				h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
				const float u = ( static_cast<float>( h & 0xffff ) / 32767.5f ) - 1.0f;
				backend::half_t& v = e[static_cast<size_t>( r ) * H + d];
				v = backend::from_float_half( backend::to_float( v ) * ( 1.0f + static_cast<float>( eps ) * u ) );
			}
		}
		backend::memcpy_async( embed_rows_.ptr<backend::half_t>(), e.data(), e.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
		backend::stream_sync( stream );
	}
	// Prefill rows decode the n-gram embeddings on the host while the GPU runs the layers before the PLE layer (forward_body); decode rows
	// (replayed graph) have them ready up front
	ple_rows_ = ( m > pod::MAX_BATCH ) ? &rows : nullptr;
	if ( !ple_rows_ ) ple_prepare( rows, m, stream );
	// Decode-sized forwards replay a per-row-count graph (the body reads slots, positions and the n-gram rows from device memory)
	graphs_.run( m, stream, [&]( backend::stream_t s ) { forward_body( m, s ); }, attn_tc_ ? 1 : 0 );   // the attention path differs for rows of one slot
}

void Qwen4ExpPipeline::trace_rows( int tag, const void* dev, bool streams, int m, backend::stream_t stream, int cols ) {
	static const std::string& path = runtime::options().trace_layers;
	if ( path.empty() || rank_ != 0 || m <= pod::MAX_BATCH ) return;
	if ( streams ) flush_apply( m, stream );
	const int H = config_.hidden_size;
	if ( cols <= 0 ) cols = H;
	const size_t elems = static_cast<size_t>( m ) * ( streams ? config_.hc_mult * H : cols );
	std::vector<float> out( elems );
	backend::stream_sync( stream );
	if ( streams ) {
		backend::memcpy_async( out.data(), dev, elems * sizeof( float ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
	}
	else {
		std::vector<backend::half_t> h( elems );
		backend::memcpy_async( h.data(), dev, elems * sizeof( backend::half_t ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
		for ( size_t i = 0; i < elems; ++i ) out[i] = backend::to_float( h[i] );
	}
	if ( FILE* f = std::fopen( path.c_str(), "ab" ) ) {
		const int hdr[3] = { tag, m, key_hint_ - m };
		std::fwrite( hdr, sizeof( int ), 3, f );
		std::fwrite( out.data(), sizeof( float ), elems, f );
		std::fclose( f );
	}
}

// Router output of the m rows: per row top_k expert ids then top_k weights, as floats
// Debug (route_dump): record {layer, rows, first position, K}, the rows' expert ids (int32) and the router's input rows (FP16)
void Qwen4ExpPipeline::dump_router( int m, backend::stream_t stream ) {
	static const std::string& path = runtime::options().route_dump;
	if ( path.empty() || rank_ != 0 || m <= pod::MAX_BATCH ) return;
	const int K = config_.top_k, H = config_.hidden_size;
	std::vector<int> ids( static_cast<size_t>( m ) * K );
	std::vector<backend::half_t> x( static_cast<size_t>( m ) * H );
	backend::stream_sync( stream );
	backend::memcpy_async( ids.data(), topk_idx_, ids.size() * sizeof( int ), backend::Copy::D2H, stream );
	backend::memcpy_async( x.data(), mixed_.data, x.size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	if ( FILE* f = std::fopen( path.c_str(), "ab" ) ) {
		const int hdr[4] = { hess_layer_, m, key_hint_ - m, K };
		std::fwrite( hdr, sizeof( int ), 4, f );
		std::fwrite( ids.data(), sizeof( int ), ids.size(), f );
		std::fwrite( x.data(), sizeof( backend::half_t ), x.size(), f );
		std::fclose( f );
	}
}

void Qwen4ExpPipeline::trace_route( int tag, int m, backend::stream_t stream ) {
	static const std::string& path = runtime::options().trace_layers;
	if ( path.empty() || rank_ != 0 || m <= pod::MAX_BATCH ) return;
	const int K = config_.top_k;
	std::vector<int> ids( static_cast<size_t>( m ) * K );
	std::vector<backend::half_t> w( ids.size() );
	backend::stream_sync( stream );
	backend::memcpy_async( ids.data(), topk_idx_, ids.size() * sizeof( int ), backend::Copy::D2H, stream );
	backend::memcpy_async( w.data(), topk_w_, w.size() * sizeof( backend::half_t ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::vector<float> out( static_cast<size_t>( m ) * 2 * K );
	for ( int r = 0; r < m; ++r ) {
		for ( int k = 0; k < K; ++k ) {
			out[static_cast<size_t>( r ) * 2 * K + k] = static_cast<float>( ids[static_cast<size_t>( r ) * K + k] );
			out[static_cast<size_t>( r ) * 2 * K + K + k] = backend::to_float( w[static_cast<size_t>( r ) * K + k] );
		}
	}
	if ( FILE* f = std::fopen( path.c_str(), "ab" ) ) {
		const int hdr[3] = { tag, m, key_hint_ - m };
		std::fwrite( hdr, sizeof( int ), 3, f );
		std::fwrite( out.data(), sizeof( float ), out.size(), f );
		std::fclose( f );
	}
}

void Qwen4ExpPipeline::gr_mix( const Qwen4ExpGatedResidual& g, bool with_post, int m, backend::stream_t stream ) {
	// Prefill rows: the gate and up projections are dense fp16 GEMMs on the tensor cores (the row-tile kernels below are fp32 FMA bound at hundreds
	// of rows); decode rows (a few) keep the fp32 kernels, whose reads of the 13 MB of weights are the floor there
	// The GEMMs always run at the full row capacity: cuBLAS then picks one algorithm and a row's result is bit-identical whatever its position in the chunk or the
	// chunk size (test_gr_tc); the rows past m are stale but never read.
	// Prefill rows: the gate and up projections run on the tensor cores as fp16 hi / lo GEMM pairs with fp32 accumulation (the row-tile kernels below are fp32
	// FMA bound at hundreds of rows); decode rows (a few) keep the fp32 kernels, whose reads of the 13 MB of weights are the floor there
	if ( with_post && m > 8 && g.down_lw.is_valid() ) {
		using GR = engine::modules::GatedResidual;
		using half_t = backend::half_t;
		const int H = config_.hidden_size, hc = config_.hc_mult, flat = hc * H, rank = config_.hc_rank;
		GR::norm( streams_.ptr<float>(), g.norm.ptr<const half_t>(), gr_scratch_, gr16_hi_.ptr<half_t>(), gr16_lo_.ptr<half_t>(), m, H, hc, config_.rms_norm_eps,
			apply_pending_ ? post_ : nullptr, apply_pending_ ? y_.ptr<const half_t>() : nullptr, stream );
		apply_pending_ = false;
		const half_t* wd = g.down_lw.w0.ptr<const half_t>();
		const half_t* wu = g.up_lw.w0.ptr<const half_t>();
		GR::gemm_f32( gr16_hi_.ptr<const half_t>(), wd, gr16_z_.ptr<float>(), gemm_rows( m ), rank + 8, flat, false, stream );
		GR::gemm_f32( gr16_lo_.ptr<const half_t>(), wd, gr16_z_.ptr<float>(), gemm_rows( m ), rank + 8, flat, true, stream );
		GR::gate_act( gr16_z_.ptr<const float>(), rank + 8, gr16_thi_.ptr<half_t>(), gr16_tlo_.ptr<half_t>(), post_, m, rank, hc, true, stream );
		GR::gemm_f32( gr16_thi_.ptr<const half_t>(), wu, gr16_u_.ptr<float>(), gemm_rows( m ), flat, rank, false, stream );
		GR::gemm_f32( gr16_tlo_.ptr<const half_t>(), wu, gr16_u_.ptr<float>(), gemm_rows( m ), flat, rank, true, stream );
		GR::up_epilogue( gr16_u_.ptr<const float>(), gr_scratch_, mixed_.ptr<half_t>(), m, H, hc, stream );
		return;
	}
#if ENGINE_BACKEND_CUDA
	if ( normed_ready_ ) {
		engine::modules::GatedResidual::mix_prenormed( g.down.ptr<const backend::half_t>(), g.up.ptr<const backend::half_t>(), with_post ? g.inject.ptr<const backend::half_t>() : nullptr,
			mixed_.ptr<backend::half_t>(), post_, gr_scratch_, m, config_.hidden_size, config_.hc_mult, config_.hc_rank, stream );
		normed_ready_ = false;
		return;
	}
#endif
	engine::modules::GatedResidual::mix(
		streams_.ptr<const float>(), g.norm.ptr<const backend::half_t>(), g.down.ptr<const backend::half_t>(), g.up.ptr<const backend::half_t>(),
		with_post ? g.inject.ptr<const backend::half_t>() : nullptr, mixed_.ptr<backend::half_t>(), post_, gr_scratch_,
		m, config_.hidden_size, config_.hc_mult, config_.hc_rank, config_.rms_norm_eps, stream,
		apply_pending_ ? post_ : nullptr, apply_pending_ ? y_.ptr<const backend::half_t>() : nullptr );
	apply_pending_ = false;
}

// The gated add after a sublayer is deferred into the next mix's norm pass (one fewer launch and one fewer sweep of the streams per site); anything
// else that reads or writes the streams flushes it first, and every captured body ends flushed
void Qwen4ExpPipeline::defer_apply() {
	apply_pending_ = true;
}

void Qwen4ExpPipeline::flush_apply( int m, backend::stream_t stream ) {
	if ( !apply_pending_ ) return;
	engine::modules::GatedResidual::apply( streams_.ptr<float>(), post_, y_.ptr<const backend::half_t>(), m, config_.hidden_size, config_.hc_mult, stream );
	apply_pending_ = false;
}

// Full-attention sublayer: mixed_ in, the rank's partial o_proj result in y_. Dense causal attention; the QSA block selection only
// restricts it once a row sees more than indexer_budget tokens.
void Qwen4ExpPipeline::attention_forward( const Qwen4ExpLayerWeights& lw, int attn_idx, int m, backend::stream_t stream ) {
	const pod::KVRef kv = kv_pool_.ref();
	const size_t kv_slot_stride = kv_pool_.slot_stride_elems;
	const int hd = config_.head_dim;
	engine::modules::Linear::forward( mixed_, lw.q_proj, q_gate_buf_, stream );
	engine::modules::Attention::unpack_q_gate( q_gate_buf_, q_buf_, gate_attn_buf_, q_heads_, hd, stream );
	engine::modules::RMSNorm::forward_heads( q_buf_, lw.q_norm, q_heads_, hd, config_.rms_norm_eps, 1.0f, stream );
	engine::modules::Linear::forward( mixed_, lw.k_proj, k_buf_, stream );
	engine::modules::RMSNorm::forward_heads( k_buf_, lw.k_norm, kv_heads_, hd, config_.rms_norm_eps, 1.0f, stream );
	engine::modules::Linear::forward( mixed_, lw.v_proj, v_buf_, stream );
	engine::modules::RoPE::forward( q_buf_, k_buf_, q_heads_, kv_heads_, hd, config_.rot_dim, d_rows_, m, config_.rope_theta, stream );
	engine::modules::Attention::store_kv_cache( kv, k_buf_, v_buf_, kv_slot_stride, attn_idx, d_rows_, m, kv_heads_, hd, config_.max_seq, stream );
	// QSA indexer: the raw-key ring and the pooled keys are kept for every row; the selection only changes the attention once a row sees more
	// than the budget (indexer_budget tokens = 512 complete pools): rows below it stay dense
	// Note: the indexer ropes queries and pooled keys at the plain KV positions (rows->pos), as the exllamav3 reference does; only the attention above
	// uses the M-RoPE positions (rope_pos). If long contexts with images (past the budget) select the wrong blocks, try the M-RoPE positions here
	// and in qsa.cu (prep / pool_update); there is no image oracle for it yet
	using engine::modules::QSA;
	engine::modules::Linear::forward( mixed_, lw.idx_qk_proj, qsa_qk_, stream );
	QSA::prep( qsa_qk_.ptr<const backend::half_t>(), lw.idx_q_norm.ptr<const backend::half_t>(), qsa_q_.ptr<backend::half_t>(), qsa_ring_, attn_idx, m, d_rows_,
		qsa_ring_slot_, config_.rot_dim, config_.rope_theta, config_.rms_norm_eps, stream );
	QSA::pool_update( qsa_ring_, qsa_pool_, lw.idx_k_norm.ptr<const backend::half_t>(), attn_idx, m, d_rows_, qsa_ring_slot_, qsa_pool_slot_, qsa_max_pools_,
		config_.rot_dim, config_.rope_theta, config_.rms_norm_eps, stream );
	static const bool qsa_off = runtime::options().qsa_off;
	// Decode-sized rows always take the sparse path (it runs rows below the budget densely): the captured graph cannot depend on the position
	if ( !qsa_off && ( m <= pod::MAX_BATCH || key_hint_ >= ( config_.indexer_budget / QSA::POOL + 1 ) * QSA::POOL ) ) {
		QSA::score( qsa_q_.ptr<const backend::half_t>(), qsa_pool_, qsa_scores_, qsa_max_pools_, attn_idx, qsa_max_pools_, m, d_rows_, qsa_pool_slot_, stream );
		engine::modules::DSA::select( qsa_scores_, qsa_max_pools_, qsa_idx_, qsa_nkeys_, m, d_rows_, stream );
		engine::modules::Attention::forward_sparse( kv, q_buf_, gate_attn_buf_, attn_out_, kv_slot_stride, attn_idx, d_rows_, m, q_heads_, kv_heads_, hd,
			config_.max_seq, qsa_idx_, qsa_nkeys_, engine::modules::DSA::KPAD, stream );
	}
	else {
		engine::modules::Attention::forward_prefill( kv, q_buf_, gate_attn_buf_, attn_out_, kv_slot_stride, attn_idx, d_rows_, m, q_heads_, kv_heads_, hd,
			config_.max_seq, stream, m > pod::MAX_BATCH ? key_hint_ : 0, m <= pod::MAX_BATCH && !attn_tc_ );
	}
	hessian( hess_layer_, 2, attn_out_.data, q_heads_ * hd, m, stream );
	engine::modules::Linear::forward( attn_out_, lw.o_proj, y_, stream );
}

// Linear-attention sublayer: mixed_ in, the rank's partial out_proj result in y_
void Qwen4ExpPipeline::gdn_forward( const Qwen4ExpLayerWeights& lw, int linear_idx, int m, backend::stream_t stream ) {
	const int k_heads = lin_k_heads_, v_heads = lin_v_heads_, channels = lin_channels_;
	engine::modules::Linear::forward( mixed_, lw.in_proj_qkv, qkv_buf_, stream );
	engine::modules::Linear::forward( mixed_, lw.in_proj_z, z_buf_, stream );
	if ( linear_idx == 0 ) {
		trace_rows( 2000, qkv_buf_.data, false, m, stream, channels );
		trace_rows( 2001, z_buf_.data, false, m, stream, v_heads * 128 );
	}

	float* hist = state_.conv_history + static_cast<size_t>( linear_idx ) * channels * 3;
	float* layer_state = state_.gdn_states + static_cast<size_t>( linear_idx ) * v_heads * 128 * 128;
	const int snap_rows = MIN( m, kSnapRows );
	float* conv_snap = state_.conv_snap + static_cast<size_t>( linear_idx ) * channels * 3;
	float* gdn_snap = state_.gdn_snap + static_cast<size_t>( linear_idx ) * v_heads * 128 * 128;

	pod::TensorView qkv_rows = pod::make_tensor_2d( qkv_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, channels );
	pod::TensorView out_rows = pod::make_tensor_2d( gdn_out_.ptr<backend::half_t>(), pod::DType::FP16, m, v_heads * 128 );
	pod::TensorView z_rows = pod::make_tensor_2d( z_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, v_heads * 128 );
	pod::TensorView gate_rows = pod::make_tensor_2d( gdn_gate_.ptr<float>(), pod::DType::FP32, m, v_heads );
	pod::TensorView beta_rows = pod::make_tensor_2d( gdn_beta_.ptr<float>(), pod::DType::FP32, m, v_heads );

	if ( m > pod::MAX_BATCH ) {
		pod::TensorView conv_rows = pod::make_tensor_2d( conv_out_.ptr<backend::half_t>(), pod::DType::FP16, m, channels );
		engine::modules::GDN::causal_conv1d_silu_tiled( qkv_rows, conv_rows, hist, lw.conv1d_w, channels, m, d_rows_, state_.slot_conv_floats, stream );
		qkv_rows = conv_rows;
	}
	else {
		engine::modules::GDN::causal_conv1d_silu_rows( qkv_rows, hist, lw.conv1d_w, channels, m, d_rows_, state_.slot_conv_floats,
			conv_snap, state_.slot_conv_floats, snap_rows, state_.d_snap_on, stream );
	}
	if ( linear_idx == 0 ) trace_rows( 2004, qkv_rows.data, false, m, stream, channels );   // after conv + silu
	engine::modules::GDN::l2_norm_heads( qkv_rows, k_heads, config_.linear_head_dim, stream, false, m, channels );
	engine::modules::GDN::project_gates( mixed_, lw.in_proj_a, lw.in_proj_b, lw.A_log, lw.dt_bias, gate_rows, beta_rows, v_heads, config_.hidden_size, stream, m );
	engine::modules::GDN::recurrent_step_rows( layer_state, qkv_rows, gate_rows, beta_rows, out_rows, k_heads, v_heads, config_.linear_head_dim, m, d_rows_,
		state_.slot_gdn_floats, gdn_snap, state_.slot_gdn_floats, snap_rows, state_.d_snap_on, stream );
	// Sigmoid-gated RMSNorm (the KDA flavor): rmsnorm(out) * w * sigmoid(z)
	if ( linear_idx == 0 ) trace_rows( 2002, out_rows.data, false, m, stream, v_heads * 128 );   // recurrence output before the norm
	engine::modules::GDN::gated_rmsnorm( out_rows, z_rows, lw.linear_norm, v_heads, config_.linear_head_dim, stream, true, m );
	if ( linear_idx == 0 ) trace_rows( 2003, out_rows.data, false, m, stream, v_heads * 128 );
	hessian( hess_layer_, 2, gdn_out_.data, v_heads * 128, m, stream );
	engine::modules::Linear::forward( gdn_out_, lw.out_proj, y_, stream );
}

// Hashes each row's n-gram (token, previous tokens within the eos segment) into the table rows and uploads the decoded embedding
namespace {

// The n-gram rows are the same for every rank (the layer is replicated), so the ranks split the decode of a forward's rows between them (row r by rank r % tp)
// and share the host buffer through a few counting barriers instead of each decoding all rows
struct PleHostShare {
	std::mutex mu;
	std::condition_variable cv;
	int arrived = 0;
	unsigned generation = 0;
	std::vector<backend::half_t> emb;

	void barrier( int parties ) {
		std::unique_lock<std::mutex> lk( mu );
		const unsigned gen = generation;
		if ( ++arrived == parties ) {
			arrived = 0;
			++generation;
			cv.notify_all();
		}
		else cv.wait( lk, [&] { return generation != gen; } );
	}
};

PleHostShare& ple_share() {
	static PleHostShare share;
	return share;
}

} // namespace

void Qwen4ExpPipeline::ple_prepare( const pod::BatchRows& rows, int m, backend::stream_t stream ) {
	auto& table = ngram_table();
	ENGINE_ASSERT( table.is_open() && static_cast<int>( host_tokens_.size() ) >= m, "Qwen4ExpPipeline: n-gram table or token ids missing" );
	const int eos = 248044;   // text_config eos_token_id: n-grams never span it
	const int D = config_.ple_embed_dim;
	std::vector<backend::half_t> emb( tp_ > 1 ? 0 : static_cast<size_t>( m ) * D );
	for ( int r = 0; r < m; ++r ) {
		const int slot = rows.slot[r], pos = rows.pos[r];
		auto& H = hist_tok_[slot];
		auto& S = hist_seg_[slot];
		H[pos] = host_tokens_[r];
		S[pos] = ( pos > 0 ) ? ( H[pos - 1] == eos ? pos : S[pos - 1] ) : 0;
	}
	auto& share = ple_share();
	if ( tp_ > 1 && share.emb.size() < static_cast<size_t>( pod::MAX_ROWS ) * D ) {
		// Sized once, before any rank writes: the first barrier below also publishes the size
		std::lock_guard<std::mutex> lk( share.mu );
		if ( share.emb.size() < static_cast<size_t>( pod::MAX_ROWS ) * D ) share.emb.resize( static_cast<size_t>( pod::MAX_ROWS ) * D );
	}
	backend::half_t* shared_emb = tp_ > 1 ? share.emb.data() : emb.data();
	std::vector<float> row( engine::loader::NgramTable::ROW_DIM );
	std::vector<uint64_t> uids( table.num_heads() );
	for ( int r = ( tp_ > 1 ? rank_ : 0 ); r < m; r += ( tp_ > 1 ? tp_ : 1 ) ) {
		const int slot = rows.slot[r], pos = rows.pos[r];
		int64_t tokens[8];
		for ( int s = 0; s < table.ngram_size(); ++s ) {
			const bool valid = ( s == 0 ) || ( pos - s >= 0 && pos - hist_seg_[slot][pos] >= s );
			tokens[s] = valid ? hist_tok_[slot][pos - s] : eos;
		}
		table.row_ids( tokens, uids.data() );
		for ( int h = 0; h < table.num_heads(); ++h ) {
			table.decode_row( uids[h], h, row.data() );
			for ( int i = 0; i < engine::loader::NgramTable::ROW_DIM; ++i ) {
				shared_emb[static_cast<size_t>( r ) * D + static_cast<size_t>( h ) * engine::loader::NgramTable::ROW_DIM + i] = backend::from_float_half( row[i] );
			}
		}
	}
	if ( tp_ > 1 ) share.barrier( tp_ );   // every rank's rows are in
	backend::memcpy_async( ple_emb_.data, shared_emb, static_cast<size_t>( m ) * D * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );   // the host buffer is reused by the next call
	if ( tp_ > 1 ) share.barrier( tp_ );   // every rank has copied it
}

void Qwen4ExpPipeline::ple_forward( int m, backend::stream_t stream ) {
	flush_apply( m, stream );
	const int H = config_.hidden_size, hc = config_.hc_mult;
	if ( m > 8 ) {
		// Prefill rows: the dense fp16 projections as tensor-core GEMMs at the full row capacity (one cuBLAS algorithm, so a row does not depend on the chunking);
		// the plain GEMV behind Linear::forward takes 20 ms per 512-row chunk here
		const int R = gemm_rows( m ), H = config_.hidden_size, hc = config_.hc_mult;
		engine::modules::Vision::dense_gemm( ple_emb_.ptr<const backend::half_t>(), ple_.key_proj.w0.ptr<const backend::half_t>(), ple_key_.ptr<backend::half_t>(), R, hc * H, config_.ple_embed_dim, stream );
		engine::modules::Vision::dense_gemm( ple_emb_.ptr<const backend::half_t>(), ple_.value_proj.w0.ptr<const backend::half_t>(), ple_value_.ptr<backend::half_t>(), R, H, config_.ple_embed_dim, stream );
	}
	else {
		engine::modules::Linear::forward( ple_emb_, ple_.key_proj, ple_key_, stream );
		engine::modules::Linear::forward( ple_emb_, ple_.value_proj, ple_value_, stream );
	}
	engine::modules::PLE::gate_and_norm( ple_key_.ptr<const backend::half_t>(), ple_value_.ptr<const backend::half_t>(), streams_.ptr<const float>(),
		ple_.norm_key.ptr<const backend::half_t>(), ple_.norm_query.ptr<const backend::half_t>(), ple_.norm_conv.ptr<const backend::half_t>(),
		ple_gated_, ple_normed_, m, H, hc, config_.rms_norm_eps, 1.0f / std::sqrt( static_cast<float>( H ) ), stream );
	engine::modules::PLE::conv_apply( ple_normed_, ple_.conv_w.ptr<const backend::half_t>(), ple_ring_, d_rows_, ple_gated_, streams_.ptr<float>(), m, hc * H, 3, stream );
	engine::modules::PLE::ring_update( ple_normed_, ple_ring_, d_rows_, m, hc * H, stream );
}

void Qwen4ExpPipeline::moe_forward( const Qwen4ExpLayerWeights& lw, int m, backend::stream_t stream, bool defer_shared ) {
	const int H = config_.hidden_size;
	// Decode rows on CUDA: the shared expert (rank 0) runs on a side stream beside the router and the fused MoE kernels; neither side touches the
	// other's scratch (the fused kernels use no trellis workspace), and the gated add waits for both
	const bool overlap = rank_ == 0 && side_stream_ != nullptr && engine::modules::MoE::fused_decode( lw.experts, moe_scratch_, m );
	backend::stream_t sh_stream = overlap ? side_stream_ : stream;
	if ( overlap ) backend::stream_fork( stream, sh_stream );
	if ( rank_ == 0 ) {
		engine::modules::Linear::forward( mixed_, lw.shared_gate, sh_gate_buf_, sh_stream );
		engine::modules::Linear::forward( mixed_, lw.shared_up, sh_up_buf_, sh_stream );
		engine::modules::SwiGLU::forward( sh_gate_buf_, sh_up_buf_, sh_inter_buf_, sh_stream );
		hessian( hess_layer_, 3, sh_inter_buf_.data, config_.shared_intermediate, m, sh_stream );
		engine::modules::Linear::forward( sh_inter_buf_, lw.shared_down, sh_out_, sh_stream );
		// the fused all-reduce adds gate * shared output when it reduces y_, so only the gate is computed here (beside the experts)
#if ENGINE_BACKEND_CUDA
		if ( defer_shared ) engine::modules::GatedResidual::shared_gate( mixed_.ptr<const backend::half_t>(), lw.shared_expert_gate.ptr<const backend::half_t>(), sh_gate_.ptr<float>(), m, H, sh_stream );
#endif
	}
	engine::modules::MoERouter::route_softmax_topk( mixed_, lw.router, topk_idx_, topk_w_, H, config_.num_experts, config_.top_k, stream, m );
	if ( &lw == &layers_[0] ) trace_route( 700, m, stream );
	dump_router( m, stream );
	const int per_rank = config_.num_experts / tp_;
	// Plain SiLU gating (no swiglu limit): act_limit 0
	engine::modules::MoE::forward( mixed_, lw.experts, topk_idx_, topk_w_, y_, rank_ * per_rank, ( rank_ + 1 ) * per_rank, moe_scratch_, 0.0f, stream );
	if ( overlap ) backend::stream_join( stream, sh_stream );
	if ( rank_ == 0 && !defer_shared ) {
		engine::modules::GatedResidual::add_sigmoid_gated( y_.ptr<backend::half_t>(), sh_out_.ptr<const backend::half_t>(), mixed_.ptr<const backend::half_t>(),
			lw.shared_expert_gate.ptr<const backend::half_t>(), m, H, stream );
	}
}

// Debug (teacher_streams): replaces the residual streams entering layer L by the reference's
void Qwen4ExpPipeline::teacher_inject( int L, int m, backend::stream_t stream ) {
	static const std::string& path = runtime::options().teacher_streams;
	if ( path.empty() || m <= pod::MAX_BATCH ) return;
	static std::vector<float> data;
	static int rows = 0;
	static std::once_flag once;
	std::call_once( once, [&]() {
		if ( FILE* f = std::fopen( path.c_str(), "rb" ) ) {
			if ( std::fread( &rows, sizeof( int ), 1, f ) == 1 && rows > 0 ) {
				data.resize( static_cast<size_t>( 47 ) * rows * config_.hc_mult * config_.hidden_size );
				if ( std::fread( data.data(), sizeof( float ), data.size(), f ) != data.size() ) data.clear();
			}
			std::fclose( f );
		}
	} );
	if ( data.empty() || rows != m ) return;
	const size_t block = static_cast<size_t>( rows ) * config_.hc_mult * config_.hidden_size;
	flush_apply( m, stream );
	backend::memcpy_async( streams_.data, data.data() + static_cast<size_t>( L - 1 ) * block, block * sizeof( float ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );
}

bool Qwen4ExpPipeline::rebalance_experts( backend::stream_t stream ) {
	if ( !tier_.active() ) return false;
	backend::set_device( rank_ );
	tier_.reset_window();
	int moved = 0;
	for ( int l = 0; l < tier_layers_; ++l ) moved += tier_.rebalance( l, stream );
	if ( ++tier_.rebalances % 8 == 0 ) tier_.save_heat();
	if ( runtime::options().expert_log ) {
		LOG_INFO( "[experts rank %d] window hit rate %.3f over %llu routes, %d swaps (%llu total)\n", rank_, tier_.hit_rate(),
			static_cast<unsigned long long>( tier_.window_routes ), moved, static_cast<unsigned long long>( tier_.swaps ) );
	}
	return true;
}

void Qwen4ExpPipeline::hessian( int layer, int site, const void* x, int k, int m, backend::stream_t stream ) {
	if ( hess_.empty() || m <= pod::MAX_BATCH || layer < 0 ) return;
	HessianSite& hs = hess_[static_cast<size_t>( layer ) * 5 + site];
	if ( hs.h == nullptr ) return;
	ENGINE_ASSERT( hs.k == k, "Qwen4ExpPipeline::hessian: site width changed" );
	engine::modules::Hessian::accumulate( hs.h, static_cast<const backend::half_t*>( x ), m, k, stream );
	hs.rows += m;
}

// One file per site: int32 magic, int32 k, int64 rows, then k * k fp32 (x^T x summed over the rows)
void Qwen4ExpPipeline::hessian_dump() {
	if ( hess_.empty() ) return;
	const std::string& dir = runtime::options().hessian_dir;
	backend::stream_sync( nullptr );
	std::vector<float> host;
	for ( size_t i = 0; i < hess_.size(); ++i ) {
		const HessianSite& hs = hess_[i];
		if ( hs.h == nullptr || hs.rows == 0 ) continue;
		host.resize( static_cast<size_t>( hs.k ) * hs.k );
		backend::memcpy_sync( host.data(), hs.h, host.size() * sizeof( float ), backend::Copy::D2H );
		char path[512];
		std::snprintf( path, sizeof( path ), "%s/L%02d_s%d_r%d.f32", dir.c_str(), static_cast<int>( i / 5 ), static_cast<int>( i % 5 ), rank_ );
		if ( FILE* f = std::fopen( path, "wb" ) ) {
			const int32_t hdr[2] = { 0x48455353, hs.k };
			std::fwrite( hdr, sizeof( int32_t ), 2, f );
			std::fwrite( &hs.rows, sizeof( int64_t ), 1, f );
			std::fwrite( host.data(), sizeof( float ), host.size(), f );
			std::fclose( f );
		}
		else LOG_ERROR( "[RANK %d] hessian: cannot write %s\n", rank_, path );
	}
	LOG_INFO( "[RANK %d] hessians written to %s\n", rank_, dir.c_str() );
}

// The all-reduce of a sublayer's output y_ (after which the gated add is deferred into the next site's norm). With a ring, decode-sized rows and the next
// site's norm weight, one kernel reduces y_, adds it to the streams and writes the next site's normed streams; the next gr_mix then starts at the gate
// Whether site_reduce will run the fused all-reduce (so the shared expert's gated add can ride in it)
bool Qwen4ExpPipeline::fuses_site( int m, const backend::half_t* next_norm ) const {
#if ENGINE_BACKEND_CUDA
	static const bool fuse = runtime::options().gr_fuse && runtime::options().trace_layers.empty() && runtime::options().teacher_streams.empty() &&
		runtime::options().hessian_dir.empty();
	return ring_ != nullptr && fuse && next_norm != nullptr && m <= 8 && !apply_pending_;
#else
	return false;
#endif
}

void Qwen4ExpPipeline::site_reduce( int m, const backend::half_t* next_norm, backend::stream_t stream, bool shared_deferred ) {
	if ( !ring_ ) return;
#if ENGINE_BACKEND_CUDA
	if ( fuses_site( m, next_norm ) ) {
		ring_->all_reduce_gated_norm( y_, streams_.ptr<float>(), post_, next_norm, gr_scratch_, config_.hc_mult, config_.rms_norm_eps, stream,
			shared_deferred ? sh_out_.ptr<const backend::half_t>() : nullptr, sh_gate_.ptr<const float>() );
		normed_ready_ = true;
		return;
	}
#endif
	ring_->all_reduce( y_, stream );
}

void Qwen4ExpPipeline::attn_site( const Qwen4ExpLayerWeights& lw, int idx, int L, int m, backend::stream_t stream, const backend::half_t* next_norm ) {
	const int H = config_.hidden_size, hc = config_.hc_mult;
	gr_mix( lw.attn_hc, true, m, stream );
	hess_layer_ = L;
	hessian( L, 0, mixed_.data, H, m, stream );
	if ( L >= 0 ) trace_rows( 100 + L, mixed_.data, false, m, stream );
	if ( lw.is_full_attention ) attention_forward( lw, idx, m, stream );
	else gdn_forward( lw, idx, m, stream );
	site_reduce( m, next_norm, stream );
	if ( L >= 0 ) trace_rows( 200 + L, y_.data, false, m, stream );
	if ( !normed_ready_ ) defer_apply();
	if ( L >= 0 ) trace_rows( 300 + L, streams_.data, true, m, stream );
}

void Qwen4ExpPipeline::mlp_site( const Qwen4ExpLayerWeights& lw, int L, int m, backend::stream_t stream, const backend::half_t* next_norm ) {
	const int H = config_.hidden_size, hc = config_.hc_mult;
	gr_mix( lw.mlp_hc, true, m, stream );
	hess_layer_ = L;
	hessian( L, 1, mixed_.data, H, m, stream );
	if ( L >= 0 ) trace_rows( 400 + L, mixed_.data, false, m, stream );
	const bool defer_shared = fuses_site( m, next_norm );
	moe_forward( lw, m, stream, defer_shared );
	site_reduce( m, next_norm, stream, defer_shared && rank_ == 0 );
	if ( L >= 0 ) trace_rows( 500 + L, y_.data, false, m, stream );
	if ( !normed_ready_ ) defer_apply();
}

void Qwen4ExpPipeline::forward_body( int m, backend::stream_t stream ) {
	const int H = config_.hidden_size, hc = config_.hc_mult;
	const int halt = runtime::options().qwen4_halt;
	apply_pending_ = false;
	engine::modules::GatedResidual::init_streams( embed_rows_.ptr<const backend::half_t>(), streams_.ptr<float>(), m, H, hc, stream );
	trace_rows( 1000, streams_.data, true, m, stream );
	int linear_idx = 0, attn_idx = 0, site = 0;
	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		if ( L == config_.ple_layer ) {
			if ( ple_rows_ ) ple_prepare( *ple_rows_, m, stream );
			ple_forward( m, stream );
			trace_rows( 800, streams_.data, true, m, stream );
		}
		if ( L >= 1 ) teacher_inject( L, m, stream );
		attn_site( lw, lw.is_full_attention ? attn_idx++ : linear_idx++, L, m, stream, lw.mlp_hc.norm.ptr<const backend::half_t>() );
		if ( halt == site++ ) { flush_apply( m, stream ); return; }
		// The next site is the next layer's attention, unless the PLE layer comes first (it changes the streams) or this is the last layer (the head's mixer follows)
		const bool next_is_site = L + 1 < config_.num_layers && L + 1 != config_.ple_layer;
		mlp_site( lw, L, m, stream, next_is_site ? layers_[L + 1].attn_hc.norm.ptr<const backend::half_t>() : nullptr );
		if ( static const double eps_l = runtime::options().perturb_layers; eps_l != 0.0 && m > pod::MAX_BATCH ) {
			flush_apply( m, stream );
			const size_t n = static_cast<size_t>( m ) * hc * H;
			std::vector<float> st( n );
			backend::stream_sync( stream );
			backend::memcpy_async( st.data(), streams_.data, n * sizeof( float ), backend::Copy::D2H, stream );
			backend::stream_sync( stream );
			for ( size_t i = 0; i < n; ++i ) {
				uint32_t h = static_cast<uint32_t>( i ) * 2654435761u ^ static_cast<uint32_t>( L + 1 ) * 40503u;
				h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
				st[i] *= 1.0f + static_cast<float>( eps_l ) * ( ( static_cast<float>( h & 0xffff ) / 32767.5f ) - 1.0f );
			}
			backend::memcpy_async( streams_.data, st.data(), n * sizeof( float ), backend::Copy::H2D, stream );
			backend::stream_sync( stream );
		}
		trace_rows( 600 + L, streams_.data, true, m, stream );
		if ( halt == site++ ) { flush_apply( m, stream ); return; }
	}
	flush_apply( m, stream );

	// Final mixer (the gated residual without the post gate) collapses the streams into the head's input; every rank projects its vocabulary shard
	if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
	const int first = ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) ? m - 1 : 0;
	const int n = ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) ? 1 : m;
	engine::modules::GatedResidual::mix(
		streams_.ptr<const float>() + static_cast<size_t>( first ) * hc * H, final_mixer_.norm.ptr<const backend::half_t>(), final_mixer_.down.ptr<const backend::half_t>(),
		final_mixer_.up.ptr<const backend::half_t>(), nullptr, mixed_.ptr<backend::half_t>(), post_, gr_scratch_, n, H, hc, config_.hc_rank, config_.rms_norm_eps, stream );
	pod::TensorView hin = pod::make_tensor_2d( mixed_.ptr<backend::half_t>(), pod::DType::FP16, n, H );
	pod::TensorView lout = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, n, lm_shard_cols_ );
	hessian( 0, 4, hin.data, H, n, stream );
	engine::modules::Linear::forward( hin, lm_head_, lout, stream );
}

void Qwen4ExpPipeline::checkpoint_state( int slot_id, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen4ExpPipeline::checkpoint_state: slot out of range" );
	backend::set_device( rank_ );
	backend::memcpy_async( state_.gdn_checkpoint, state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, state_.slot_gdn_floats * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_checkpoint, state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, state_.slot_conv_floats * sizeof( float ), backend::Copy::D2D, stream );
}

void Qwen4ExpPipeline::rollback_state( int slot_id, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen4ExpPipeline::rollback_state: slot out of range" );
	backend::set_device( rank_ );
	backend::memcpy_async( state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, state_.gdn_checkpoint, state_.slot_gdn_floats * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, state_.conv_checkpoint, state_.slot_conv_floats * sizeof( float ), backend::Copy::D2D, stream );
}

void Qwen4ExpPipeline::set_spec_snapshot( bool on, backend::stream_t stream ) {
	backend::set_device( rank_ );
	const int v = on ? 1 : 0;
	backend::memcpy_async( state_.d_snap_on, &v, sizeof( int ), backend::Copy::H2D, stream );
}

void Qwen4ExpPipeline::commit_spec_row( int slot_id, int row, backend::stream_t stream ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots && row >= 0 && row < kSnapRows, "Qwen4ExpPipeline::commit_spec_row: out of range" );
	backend::set_device( rank_ );
	backend::memcpy_async( state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, state_.gdn_snap + static_cast<size_t>( row ) * state_.slot_gdn_floats,
		state_.slot_gdn_floats * sizeof( float ), backend::Copy::D2D, stream );
	backend::memcpy_async( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, state_.conv_snap + static_cast<size_t>( row ) * state_.slot_conv_floats,
		state_.slot_conv_floats * sizeof( float ), backend::Copy::D2D, stream );
}

// One MTP forward (see llama.cpp's qwen4exp MTP graph): per row, the target's pre-collapse stream stack h (4 x hidden) and the token that follows the row's
// position. streams[s] = fc_hidden( rmsnorm_s( h[s] ) ) + fc_embedding( rmsnorm( embed( token ) ) ); then the usual block (attention site, MLP site) on those
// streams with the head's own weights; the streams after the block are the next step's h and, through the head's mixer and the shared LM head, the draft logits.
// hidden_mode: 0 = the last forward's stacks, 1 = saved row hist_row0 then the last forward's rows 0 .. m - 2, 2 = saved rows from hist_row0, 3 = the previous step's output.
void Qwen4ExpPipeline::mtp_step( const pod::BatchRows& rows, const int* tokens, int hidden_mode, int hist_row0, bool logits_last, backend::stream_t stream ) {
	const int last = rows.count - 1;
	mtp_step_impl( rows, tokens, hidden_mode, hist_row0, &last, 1, logits_last, stream );
}

void Qwen4ExpPipeline::mtp_step_rows( const pod::BatchRows& rows, const int* tokens, int hidden_mode, int hist_row0, const int* out_rows, int n_out, backend::stream_t stream ) {
	mtp_step_impl( rows, tokens, hidden_mode, hist_row0, out_rows, n_out, true, stream );
}

// Stages saved stacks (rows of the history buffer, any slots) as the hidden input of the next joint step (hidden_mode 4)
void Qwen4ExpPipeline::mtp_gather_hist( const int* src_rows, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( mtp_on_ && count >= 1 && count <= pod::MAX_BATCH, "Qwen4ExpPipeline::mtp_gather_hist: rows out of range" );
	backend::set_device( rank_ );
	const size_t stack = static_cast<size_t>( config_.hc_mult ) * config_.hidden_size * sizeof( float );
	for ( int i = 0; i < count; ++i ) {
		ENGINE_ASSERT( src_rows[i] >= 0 && src_rows[i] < config_.max_slots * 2 * ( kDraftM + 1 ), "Qwen4ExpPipeline::mtp_gather_hist: history row out of range" );
		backend::memcpy_async( reinterpret_cast<char*>( mtp_src_ ) + static_cast<size_t>( i ) * stack, reinterpret_cast<const char*>( mtp_hist_ ) + static_cast<size_t>( src_rows[i] ) * stack, stack, backend::Copy::D2D, stream );
	}
}

void Qwen4ExpPipeline::mtp_step_impl( const pod::BatchRows& rows, const int* tokens, int hidden_mode, int hist_row0, const int* out_rows, int n_out, bool logits, backend::stream_t stream ) {
	ENGINE_ASSERT( mtp_on_ && weights_loaded_, "Qwen4ExpPipeline::mtp_step: no MTP head" );
	backend::set_device( rank_ );
	const int m = rows.count;
	const int H = config_.hidden_size, hc = config_.hc_mult;
	const size_t stack = static_cast<size_t>( hc ) * H * sizeof( float );
	ENGINE_ASSERT( m >= 1 && m <= pod::MAX_ROWS, "Qwen4ExpPipeline::mtp_step: rows out of range" );
	key_hint_ = rows.pos[m - 1] + 1;
	attn_tc_ = true;
	for ( int r = 1; r < m; ++r ) attn_tc_ = attn_tc_ && rows.slot[r] == rows.slot[0];
	set_rows( m );
	backend::memcpy_async( d_rows_, &rows, rows.upload_bytes(), backend::Copy::H2D, stream );

	// 1. embedding branch
	backend::memcpy_async( d_token_ids_, tokens, m * sizeof( int ), backend::Copy::H2D, stream );
	pod::TensorView emb = pod::make_tensor_2d( mtp_emb_.data, pod::DType::FP16, m, H );
	pod::TensorView xe = pod::make_tensor_2d( mtp_xe_.data, pod::DType::FP16, m, H );
	pod::TensorView e2 = pod::make_tensor_2d( mtp_e2_.data, pod::DType::FP16, m, H );
	engine::modules::Embedding::forward( embed_table_, d_token_ids_, m, emb, stream );
	engine::modules::RMSNorm::forward( emb, mtp_pre_norm_embedding_, xe, config_.rms_norm_eps, 1.0f, stream );
	engine::modules::Linear::forward( xe, mtp_fc_embedding_, e2, stream );

	// 2. hidden branch: per-stream norm, fc_hidden on every stream
	const float* src = nullptr;
	switch ( hidden_mode ) {
		case 0: src = streams_.ptr<const float>(); break;
		case 1: {
			backend::memcpy_async( mtp_src_, mtp_hist_ + static_cast<size_t>( hist_row0 ) * hc * H, stack, backend::Copy::D2D, stream );
			if ( m > 1 ) backend::memcpy_async( mtp_src_ + static_cast<size_t>( hc ) * H, streams_.data, static_cast<size_t>( m - 1 ) * stack, backend::Copy::D2D, stream );
			src = mtp_src_;
		} break;
		case 2: src = mtp_hist_ + static_cast<size_t>( hist_row0 ) * hc * H; break;
		case 4: src = mtp_src_; break;
		default: src = mtp_chain_; break;
	}
	for ( int r0 = 0; r0 < m; r0 += 128 ) {   // chunks of at most 512 stream rows
		const int n = MIN( 128, m - r0 );
		engine::modules::GatedResidual::stream_norm( src + static_cast<size_t>( r0 ) * hc * H, mtp_pre_norm_hidden_.ptr<const backend::half_t>(),
			mtp_hn_.ptr<backend::half_t>(), n, H, hc, config_.rms_norm_eps, stream );
		pod::TensorView hn = pod::make_tensor_2d( mtp_hn_.data, pod::DType::FP16, n * hc, H );
		pod::TensorView h2 = pod::make_tensor_2d( mtp_h2_.ptr<backend::half_t>() + static_cast<size_t>( r0 ) * hc * H, pod::DType::FP16, n * hc, H );
		engine::modules::Linear::forward( hn, mtp_fc_hidden_, h2, stream );
	}
	engine::modules::GatedResidual::add_broadcast( mtp_h2_.ptr<const backend::half_t>(), e2.ptr<const backend::half_t>(), streams_.ptr<float>(), m, H, hc, stream );

	// 3. the head's block (its attention is the last layer of the KV pool)
	attn_site( mtp_layer_, attn_layers_ - 1, -1, m, stream, mtp_layer_.mlp_hc.norm.ptr<const backend::half_t>() );
	mlp_site( mtp_layer_, -1, m, stream );
	flush_apply( m, stream );

	// 4. the output stacks of the requested rows feed the next step; the mixer and the shared head give their draft logits
	ENGINE_ASSERT( n_out >= 1 && n_out <= kSnapRows, "Qwen4ExpPipeline::mtp_step: output rows out of range" );
	for ( int j = 0; j < n_out; ++j ) {
		backend::memcpy_async( reinterpret_cast<char*>( mtp_chain_ ) + static_cast<size_t>( j ) * stack, streams_.ptr<const char>() + static_cast<size_t>( out_rows[j] ) * stack, stack, backend::Copy::D2D, stream );
	}
	if ( logits ) {
		engine::modules::GatedResidual::mix( mtp_chain_, mtp_mixer_.norm.ptr<const backend::half_t>(), mtp_mixer_.down.ptr<const backend::half_t>(),
			mtp_mixer_.up.ptr<const backend::half_t>(), nullptr, mixed_.ptr<backend::half_t>(), post_, gr_scratch_, n_out, H, hc, config_.hc_rank, config_.rms_norm_eps, stream );
		pod::TensorView hin = pod::make_tensor_2d( mixed_.ptr<backend::half_t>(), pod::DType::FP16, n_out, H );
		pod::TensorView lout = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, n_out, lm_shard_cols_ );
		engine::modules::Linear::forward( hin, lm_head_, lout, stream );
		logits_.shape[0] = n_out;
		last_rows_ = n_out;
	}
}

// Keeps the stacks of rows [src_row, src_row + rows) of the last forward for later steps (row dst_row onwards)
void Qwen4ExpPipeline::mtp_save_hist( int src_row, int rows, int dst_row, backend::stream_t stream ) {
	ENGINE_ASSERT( mtp_on_ && rows >= 1 && dst_row + rows <= config_.max_slots * 2 * ( kDraftM + 1 ), "Qwen4ExpPipeline::mtp_save_hist: rows out of range" );
	backend::set_device( rank_ );
	const size_t stack = static_cast<size_t>( config_.hc_mult ) * config_.hidden_size * sizeof( float );
	backend::memcpy_async( reinterpret_cast<char*>( mtp_hist_ ) + static_cast<size_t>( dst_row ) * stack, streams_.ptr<const char>() + static_cast<size_t>( src_row ) * stack,
		static_cast<size_t>( rows ) * stack, backend::Copy::D2D, stream );
}

void Qwen4ExpPipeline::sample_tokens( int* out_preds, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( false, "qwen4_exp: the vocabulary is sharded, use sample_local" );
	(void) out_preds; (void) count; (void) stream;
}

const pod::TensorView& Qwen4ExpPipeline::gather_logits( int rows, const backend::half_t* const* peer_shard_ptrs, int peer_shard_cols, backend::stream_t stream ) {
	ENGINE_ASSERT( rank_ == 0, "Qwen4ExpPipeline::gather_logits: only rank 0 materializes the full vocabulary" );
	ENGINE_ASSERT( rows >= 1 && rows <= last_rows_ && peer_shard_cols == lm_shard_cols_, "Qwen4ExpPipeline::gather_logits: rows or shard width" );
	backend::set_device( rank_ );
	const size_t V = config_.vocab_size;
	if ( rows > full_logits_rows_ ) {
		if ( full_logits_mem_ != nullptr ) backend::device_free( full_logits_mem_ );
		full_logits_mem_ = backend::device_malloc( static_cast<size_t>( rows ) * V * sizeof( backend::half_t ) );
		full_logits_rows_ = rows;
		const int shape[2] = { rows, config_.vocab_size };
		full_logits_ = pod::make_tensor_2d( full_logits_mem_, pod::DType::FP16, shape[0], shape[1] );
	}
	full_logits_.shape[0] = rows;
	for ( int d = 0; d < tp_; ++d ) {
		const backend::half_t* src_base = ( d == rank_ ) ? logits_.ptr<const backend::half_t>() : peer_shard_ptrs[d];
		ENGINE_ASSERT( src_base != nullptr, "Qwen4ExpPipeline::gather_logits: missing peer shard pointer" );
		// One copy per row and rank: rows are lm_shard_cols_ apart at the source, vocab_size apart at the destination. A peer copy, not a 2-D
		// copy by address: another rank's pointer is only meaningful on its own device (Vulkan devices share an address space layout)
		for ( int r = 0; r < rows; ++r ) {
			backend::peer_copy_async( full_logits_.ptr<backend::half_t>() + static_cast<size_t>( r ) * V + static_cast<size_t>( d ) * lm_shard_real_cols_, rank_,
				src_base + static_cast<size_t>( r ) * lm_shard_cols_, d, static_cast<size_t>( lm_shard_real_cols_ ) * sizeof( backend::half_t ), stream );
		}
	}
	return full_logits_;
}

bool Qwen4ExpPipeline::expert_heat( std::vector<uint32_t>& out, int& layers, int& experts, bool reset ) {
	backend::set_device( rank_ );
	backend::device_sync();
	layers = 0;
	experts = 0;
	out.clear();
	for ( auto& lw : layers_ ) {
		if ( lw.experts.heat == nullptr || lw.experts.num_local_experts <= 0 ) continue;
		experts = lw.experts.num_local_experts;
		const size_t at = out.size();
		out.resize( at + static_cast<size_t>( experts ) );
		backend::memcpy_sync( out.data() + at, lw.experts.heat, static_cast<size_t>( experts ) * sizeof( uint32_t ), backend::Copy::D2H );
		if ( reset ) backend::memset_sync( lw.experts.heat, 0, static_cast<size_t>( experts ) * sizeof( uint32_t ) );
		++layers;
	}
	return layers > 0;
}

bool Qwen4ExpPipeline::sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Qwen4ExpPipeline::sample_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax_range( logits_, d_argmax_idx_, d_argmax_val_, lm_shard_real_cols_, count, rank_ * lm_shard_real_cols_, stream );
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_tokens, h_argmax_idx_, count * sizeof( int ) );
	if ( out_vals != nullptr ) {
		std::vector<float> h_vals( count );
		backend::memcpy_async( h_vals.data(), d_argmax_val_, count * sizeof( float ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
		std::memcpy( out_vals, h_vals.data(), count * sizeof( float ) );
	}
	return true;
}

bool Qwen4ExpPipeline::topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Qwen4ExpPipeline::topk_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	engine::spec::dflash2_topk_shard( logits_.ptr<const backend::half_t>(), count, lm_shard_cols_, lm_shard_real_cols_, 16, d_samp_vals_, d_samp_idx_,
		rank_ * lm_shard_real_cols_, samp_part_v_, samp_part_i_, samp_counters_, stream );
	backend::memcpy_async( out_vals, d_samp_vals_, static_cast<size_t>( count ) * 16 * sizeof( float ), backend::Copy::D2H, stream );
	backend::memcpy_async( out_ids, d_samp_idx_, static_cast<size_t>( count ) * 16 * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

void Qwen4ExpPipeline::reset_state( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen4ExpPipeline::reset_state: slot out of range" );
	backend::set_device( rank_ );
	kv_pool_.reset_slot( slot_id );
	backend::memset_sync( qsa_ring_ + static_cast<size_t>( slot_id ) * qsa_ring_slot_, 0, qsa_ring_slot_ * sizeof( backend::half_t ) );
	backend::memset_sync( state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, 0, state_.slot_gdn_floats * sizeof( float ) );
	backend::memset_sync( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, 0, state_.slot_conv_floats * sizeof( float ) );
	const size_t ring = static_cast<size_t>( engine::modules::PLE::RING ) * config_.hc_mult * config_.hidden_size;
	backend::memset_sync( ple_ring_ + static_cast<size_t>( slot_id ) * ring, 0, ring * sizeof( backend::half_t ) );
	std::fill( hist_tok_[slot_id].begin(), hist_tok_[slot_id].end(), 0 );
	std::fill( hist_seg_[slot_id].begin(), hist_seg_[slot_id].end(), 0 );
}

namespace {

constexpr size_t kSeqAlign = 256;
constexpr size_t kCtxBytes = 16;   // two token ids and the eos segment start of the last position, padded

size_t kv_segment_bytes( pod::KVQ q, size_t row_elems, int rows ) {
	return pod::kv_data_bytes( q, row_elems * rows ) + pod::kv_scale_bytes( q, row_elems * rows );
}

} // namespace

size_t Qwen4ExpPipeline::seq_state_bytes( int from, int tokens, bool with_taps ) const {
	(void) with_taps;   // no drafter taps on this model
	if ( !kv_pool_.k_data || from < 0 || tokens < from || tokens > config_.max_seq ) return 0;
	const size_t recurrent = ( state_.slot_gdn_floats + state_.slot_conv_floats ) * sizeof( float );
	const size_t ring = static_cast<size_t>( engine::modules::PLE::RING ) * config_.hc_mult * config_.hidden_size * sizeof( backend::half_t );
	const size_t row_elems = static_cast<size_t>( kv_pool_.num_heads ) * kv_pool_.head_dim;
	const pod::KVQ kq = pod::kv_k_quant( kv_pool_.format ), vq = pod::kv_v_quant( kv_pool_.format );
	const size_t kv = static_cast<size_t>( kv_pool_.num_layers ) * ( kv_segment_bytes( kq, row_elems, tokens - from ) + kv_segment_bytes( vq, row_elems, tokens - from ) );
	const size_t pool_rows = static_cast<size_t>( tokens / engine::modules::QSA::POOL - from / engine::modules::QSA::POOL );
	const size_t pools = static_cast<size_t>( kv_pool_.num_layers ) * pool_rows * engine::modules::QSA::DIM * sizeof( backend::half_t );
	return recurrent + ring + kCtxBytes + kv + pools + kSeqAlign;
}

// Host <-> device copy of every KV range of the snapshot, in layout order
size_t Qwen4ExpPipeline::copy_kv_rows( int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream ) {
	char* const host0 = host;
	const size_t row_elems = static_cast<size_t>( kv_pool_.num_heads ) * kv_pool_.head_dim;
	const size_t count = static_cast<size_t>( tokens - from );
	if ( count == 0 ) return 0;
	const pod::KVQ kqs[2] = { pod::kv_k_quant( kv_pool_.format ), pod::kv_v_quant( kv_pool_.format ) };
	char* data_base[2] = { static_cast<char*>( kv_pool_.k_data ), static_cast<char*>( kv_pool_.v_data ) };
	char* scale_base[2] = { static_cast<char*>( kv_pool_.k_scale ), static_cast<char*>( kv_pool_.v_scale ) };
	const size_t slot_data[2] = { kv_pool_.k_slot_bytes, kv_pool_.v_slot_bytes };
	const size_t slot_scale[2] = { kv_pool_.k_scale_slot_bytes, kv_pool_.v_scale_slot_bytes };
	const backend::Copy dir = to_host ? backend::Copy::D2H : backend::Copy::H2D;
	for ( int layer = 0; layer < kv_pool_.num_layers; ++layer ) {
		const size_t elem0 = ( static_cast<size_t>( layer ) * kv_pool_.max_seq_len + from ) * row_elems;
		for ( int kv = 0; kv < 2; ++kv ) {
			const size_t db = pod::kv_data_bytes( kqs[kv], row_elems * count );
			char* dev = data_base[kv] + static_cast<size_t>( slot_id ) * slot_data[kv] + pod::kv_data_bytes( kqs[kv], elem0 );
			if ( to_host ) backend::memcpy_async( host, dev, db, dir, stream );
			else backend::memcpy_async( dev, host, db, dir, stream );
			host += db;
			if ( const size_t sb = pod::kv_scale_bytes( kqs[kv], row_elems * count ) ) {
				char* sdev = scale_base[kv] + static_cast<size_t>( slot_id ) * slot_scale[kv] + pod::kv_scale_bytes( kqs[kv], elem0 );
				if ( to_host ) backend::memcpy_async( host, sdev, sb, dir, stream );
				else backend::memcpy_async( sdev, host, sb, dir, stream );
				host += sb;
			}
		}
	}
	return static_cast<size_t>( host - host0 );
}

// Pooled index keys [from / 4, tokens / 4) of every attention layer (the snapshot positions are multiples of seq_align)
void Qwen4ExpPipeline::copy_pool_rows( int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream ) {
	const int P = engine::modules::QSA::POOL, D = engine::modules::QSA::DIM;
	const size_t rows = static_cast<size_t>( tokens / P - from / P );
	if ( rows == 0 ) return;
	const backend::Copy dir = to_host ? backend::Copy::D2H : backend::Copy::H2D;
	for ( int layer = 0; layer < kv_pool_.num_layers; ++layer ) {
		backend::half_t* dev = qsa_pool_ + static_cast<size_t>( slot_id ) * qsa_pool_slot_ + ( static_cast<size_t>( layer ) * qsa_max_pools_ + from / P ) * D;
		const size_t bytes = rows * D * sizeof( backend::half_t );
		if ( to_host ) backend::memcpy_async( host, dev, bytes, dir, stream );
		else backend::memcpy_async( dev, host, bytes, dir, stream );
		host += bytes;
	}
}

bool Qwen4ExpPipeline::save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* dst = static_cast<char*>( host_dst );
	const size_t gdn = state_.slot_gdn_floats * sizeof( float ), conv = state_.slot_conv_floats * sizeof( float );
	const size_t ring = static_cast<size_t>( engine::modules::PLE::RING ) * config_.hc_mult * config_.hidden_size * sizeof( backend::half_t );
	backend::memcpy_async( dst, state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, gdn, backend::Copy::D2H, stream );
	backend::memcpy_async( dst + gdn, state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, conv, backend::Copy::D2H, stream );
	backend::memcpy_async( dst + gdn + conv, ple_ring_ + static_cast<size_t>( slot_id ) * ( ring / sizeof( backend::half_t ) ), ring, backend::Copy::D2H, stream );
	dst += gdn + conv + ring;
	// Token context of the last position: what the n-gram hash of the next position reads
	int ctx[4] = { 0, 0, 0, 0 };
	if ( tokens >= 1 ) { ctx[0] = hist_tok_[slot_id][tokens - 1]; ctx[2] = hist_seg_[slot_id][tokens - 1]; }
	if ( tokens >= 2 ) ctx[1] = hist_tok_[slot_id][tokens - 2];
	std::memcpy( dst, ctx, kCtxBytes );
	dst += kCtxBytes;
	dst += copy_kv_rows( slot_id, from, tokens, dst, true, stream );
	copy_pool_rows( slot_id, from, tokens, dst, true, stream );
	return true;
}

bool Qwen4ExpPipeline::load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	const char* src = static_cast<const char*>( host_src );
	const size_t gdn = state_.slot_gdn_floats * sizeof( float ), conv = state_.slot_conv_floats * sizeof( float );
	const size_t ring = static_cast<size_t>( engine::modules::PLE::RING ) * config_.hc_mult * config_.hidden_size * sizeof( backend::half_t );
	if ( !kv_only ) {
		backend::memcpy_async( state_.gdn_states + static_cast<size_t>( slot_id ) * state_.slot_gdn_floats, src, gdn, backend::Copy::H2D, stream );
		backend::memcpy_async( state_.conv_history + static_cast<size_t>( slot_id ) * state_.slot_conv_floats, src + gdn, conv, backend::Copy::H2D, stream );
		backend::memcpy_async( ple_ring_ + static_cast<size_t>( slot_id ) * ( ring / sizeof( backend::half_t ) ), src + gdn + conv, ring, backend::Copy::H2D, stream );
		int ctx[4];
		std::memcpy( ctx, src + gdn + conv + ring, kCtxBytes );
		if ( tokens >= 1 ) { hist_tok_[slot_id][tokens - 1] = ctx[0]; hist_seg_[slot_id][tokens - 1] = ctx[2]; }
		if ( tokens >= 2 ) hist_tok_[slot_id][tokens - 2] = ctx[1];
	}
	char* kv_src = const_cast<char*>( src ) + gdn + conv + ring + kCtxBytes;
	kv_src += copy_kv_rows( slot_id, from, tokens, kv_src, false, stream );
	copy_pool_rows( slot_id, from, tokens, kv_src, false, stream );
	return true;
}

} // namespace engine::models
