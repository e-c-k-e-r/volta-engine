#include "engine/models/qwen4_exp.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/gdn.h"
#include "engine/modules/gated_residual.h"
#include "engine/modules/embedding.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/ple.h"
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
#include <fstream>
#include <sstream>

namespace engine::models {

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
	const auto* src = static_cast<const backend::half_t*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard_idx ), meta ) );
	if ( !src ) return pod::TensorView{};
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
Qwen4ExpPipeline::~Qwen4ExpPipeline() = default;

bool Qwen4ExpPipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	rank_ = device_id;
	arena_ = &arena;
	ring_ = ring;
	tp_ = ( ring != nullptr ) ? requested_tp_ : 1;
	if ( runtime::options().max_seq > 0 ) config_.max_seq = runtime::options().max_seq;
	if ( runtime::options().max_slots > 0 ) config_.max_slots = runtime::options().max_slots;
	if ( tp_ != ENGINE_NUM_GPUS ) {
		LOG_ERROR( "Qwen4ExpPipeline: tp=%d unsupported (experts are placed whole, 128 per rank: tp 4 only)\n", tp_ );
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
	kv_heads_ = 1;   // 2 KV heads over 4 ranks: ranks 2j and 2j + 1 each hold head j
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
	state_.gdn_snap = static_cast<float*>( engine::core::arena_alloc( arena, kDraftM * state_.slot_gdn_floats * sizeof( float ) ) );
	state_.conv_snap = static_cast<float*>( engine::core::arena_alloc( arena, kDraftM * state_.slot_conv_floats * sizeof( float ) ) );
	state_.d_snap_on = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
	backend::memset_sync( state_.d_snap_on, 0, sizeof( int ) );

	int full_layers = 0;
	for ( int L = 0; L < config_.num_layers; ++L ) full_layers += config_.is_full_attention( L ) ? 1 : 0;
	kv_pool_.init( rank_, arena, S, config_.max_seq, full_layers, kv_heads_, config_.head_dim );
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
	int qg_shape[2] = { R, q_heads_ * config_.head_dim * 2 };
	q_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qg_shape, 2 );
	int q_shape[2] = { R, q_heads_ * config_.head_dim };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	gate_attn_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	int kv_shape[2] = { R, kv_heads_ * config_.head_dim };
	k_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
	v_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
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
	moe_scratch_ = engine::modules::MoE::alloc_scratch( arena, config_.num_experts / tp_, config_.top_k, H, config_.moe_intermediate, R );
	int sh_shape[2] = { R, config_.shared_intermediate };
	sh_gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sh_shape, 2 );
	sh_up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sh_shape, 2 );
	sh_inter_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sh_shape, 2 );
	sh_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
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
	backend::set_device( rank_ );
	const int H = config_.hidden_size;
	const int tp = tp_;
	int missing = 0;
	char nm[256];

	auto dense = [&]( const char* name ) {
		const pod::TensorView t = repo.load( name, *arena_, stream, pod::DType::FP16 );
		if ( !t.data ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, name ); ++missing; }
		return t;
	};
	auto dense_slice = [&]( const char* name ) {
		const pod::TensorView t = repo.load_slice( name, *arena_, 0, rank_, tp, stream, pod::DType::FP16 );
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
		return g;
	};
	// Column-parallel (output split) and row-parallel (input split) trellis projections
	auto load_col = [&]( const char* name, int bits ) { return check( repo.load_linear_slice( name, *arena_, bits, 1, rank_, tp, stream ), name ); };
	auto load_row = [&]( const char* name, int bits ) { return check( repo.load_linear_slice( name, *arena_, bits, 0, rank_, tp, stream ), name ); };

	const char* P = "model.language_model";
	std::snprintf( nm, sizeof( nm ), "%s.embed_tokens.weight", P );
	embed_table_ = repo.load( nm, *arena_, stream, pod::DType::BF16 );
	if ( !embed_table_.data ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, nm ); ++missing; }
	std::snprintf( nm, sizeof( nm ), "%s.hyper_connection_mixer", P );
	final_mixer_ = load_gr( nm, false );
	lm_head_ = check( repo.load_linear_slice_padded( "lm_head", *arena_, 6, rank_, tp, lm_shard_cols_ - lm_shard_real_cols_, stream ), "lm_head" );

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
		std::call_once( once, [&]() { table_ok = ngram_table().open( std::string( model_dir ) + "/ngram_embedding.safetensors" ); } );
		if ( !table_ok ) { LOG_ERROR( "[RANK %d] the n-gram table failed to open\n", rank_ ); ++missing; }
	}
	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		char pre[160];
		std::snprintf( pre, sizeof( pre ), "%s.layers.%d", P, L );
		std::snprintf( nm, sizeof( nm ), "%s.attn_hyper_connection", pre ); lw.attn_hc = load_gr( nm, true );
		std::snprintf( nm, sizeof( nm ), "%s.mlp_hyper_connection", pre ); lw.mlp_hc = load_gr( nm, true );

		if ( lw.is_full_attention ) {
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.q_proj", pre ); lw.q_proj = load_col( nm, 8 );
			// 2 KV heads over 4 ranks: ranks 2j and 2j + 1 both take head j (half of the 512 columns)
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.k_proj", pre );
			lw.k_proj = check( repo.load_linear_slice( nm, *arena_, 8, 1, rank_ / 2, 2, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.v_proj", pre );
			lw.v_proj = check( repo.load_linear_slice( nm, *arena_, 8, 1, rank_ / 2, 2, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.o_proj", pre ); lw.o_proj = load_row( nm, 8 );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.q_norm.weight", pre ); lw.q_norm = dense( nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.k_norm.weight", pre ); lw.k_norm = dense( nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.indexer.index_qk_proj", pre );
			lw.idx_qk_proj = check( repo.load_linear( nm, *arena_, 8, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.indexer.q_layernorm.weight", pre ); lw.idx_q_norm = dense( nm );
			std::snprintf( nm, sizeof( nm ), "%s.self_attn.indexer.k_layernorm.weight", pre ); lw.idx_k_norm = dense( nm );
		}
		else {
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.in_proj_qkv", pre );
			lw.in_proj_qkv = check( repo.load_grouped_trellis_slice( nm, *arena_, QKV_BLOCKS, qkv_rank_blocks, 3, rank_, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.in_proj_z", pre ); lw.in_proj_z = load_col( nm, 8 );
			std::snprintf( nm, sizeof( nm ), "%s.linear_attn.out_proj", pre ); lw.out_proj = load_row( nm, 8 );
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
		lw.router = load_router_transposed( repo, nm, *arena_ );
		if ( !lw.router.data ) { LOG_ERROR( "[RANK %d] MISSING %s\n", rank_, nm ); ++missing; }
		auto& mw = lw.experts;
		mw.type = pod::QuantType::TRELLIS;
		const int per_rank = config_.num_experts / tp;
		mw.num_local_experts = per_rank;
		mw.top_k = config_.top_k;
		for ( int e = 0; e < per_rank; ++e ) {
			const int gid = rank_ * per_rank + e;
			std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.gate_proj", pre, gid ); mw.gate[e] = check( repo.load_linear( nm, *arena_, 4, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.up_proj", pre, gid ); mw.up[e] = check( repo.load_linear( nm, *arena_, 4, stream ), nm );
			std::snprintf( nm, sizeof( nm ), "%s.mlp.experts.%d.down_proj", pre, gid ); mw.down[e] = check( repo.load_linear( nm, *arena_, 4, stream ), nm );
		}
		engine::modules::MoE::init_device_ptrs( mw, *arena_, stream );
		mw.heat = static_cast<uint32_t*>( engine::core::arena_alloc( *arena_, static_cast<size_t>( per_rank ) * sizeof( uint32_t ), 128 ) );
		backend::memset_async( mw.heat, 0, static_cast<size_t>( per_rank ) * sizeof( uint32_t ), stream );
		std::snprintf( nm, sizeof( nm ), "%s.mlp.shared_expert.gate_proj", pre ); lw.shared_gate = check( repo.load_linear( nm, *arena_, 4, stream ), nm );
		std::snprintf( nm, sizeof( nm ), "%s.mlp.shared_expert.up_proj", pre ); lw.shared_up = check( repo.load_linear( nm, *arena_, 4, stream ), nm );
		std::snprintf( nm, sizeof( nm ), "%s.mlp.shared_expert.down_proj", pre ); lw.shared_down = check( repo.load_linear( nm, *arena_, 4, stream ), nm );
		std::snprintf( nm, sizeof( nm ), "%s.mlp.shared_expert_gate.weight", pre ); lw.shared_expert_gate = dense( nm );
	}
	(void) H;
	if ( stream ) backend::stream_sync( stream );
	repo.close();
	if ( missing > 0 ) {
		LOG_ERROR( "[RANK %d] %d tensors missing\n", rank_, missing );
		return false;
	}
	weights_loaded_ = true;
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
	ple_prepare( rows, m, stream );
	forward_body( m, stream );
}

void Qwen4ExpPipeline::trace_rows( int tag, const void* dev, bool streams, int m, backend::stream_t stream, int cols ) {
	static const std::string& path = runtime::options().trace_layers;
	if ( path.empty() || rank_ != 0 || m <= pod::MAX_BATCH ) return;
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
	engine::modules::GatedResidual::mix(
		streams_.ptr<const float>(), g.norm.ptr<const backend::half_t>(), g.down.ptr<const backend::half_t>(), g.up.ptr<const backend::half_t>(),
		with_post ? g.inject.ptr<const backend::half_t>() : nullptr, mixed_.ptr<backend::half_t>(), post_, gr_scratch_,
		m, config_.hidden_size, config_.hc_mult, config_.hc_rank, config_.rms_norm_eps, stream );
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
	engine::modules::Attention::forward_prefill( kv, q_buf_, gate_attn_buf_, attn_out_, kv_slot_stride, attn_idx, d_rows_, m, q_heads_, kv_heads_, hd,
		config_.max_seq, stream, m > pod::MAX_BATCH ? key_hint_ : 0, m <= pod::MAX_BATCH && !attn_tc_ );
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
	const int snap_rows = MIN( m, kDraftM );
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
	engine::modules::Linear::forward( gdn_out_, lw.out_proj, y_, stream );
}

// Hashes each row's n-gram (token, previous tokens within the eos segment) into the table rows and uploads the decoded embedding
void Qwen4ExpPipeline::ple_prepare( const pod::BatchRows& rows, int m, backend::stream_t stream ) {
	auto& table = ngram_table();
	ENGINE_ASSERT( table.is_open() && static_cast<int>( host_tokens_.size() ) >= m, "Qwen4ExpPipeline: n-gram table or token ids missing" );
	const int eos = 248044;   // text_config eos_token_id: n-grams never span it
	const int D = config_.ple_embed_dim;
	std::vector<backend::half_t> emb( static_cast<size_t>( m ) * D );
	for ( int r = 0; r < m; ++r ) {
		const int slot = rows.slot[r], pos = rows.pos[r];
		auto& H = hist_tok_[slot];
		auto& S = hist_seg_[slot];
		H[pos] = host_tokens_[r];
		S[pos] = ( pos > 0 ) ? ( H[pos - 1] == eos ? pos : S[pos - 1] ) : 0;
	}
	std::vector<float> row( engine::loader::NgramTable::ROW_DIM );
	std::vector<uint64_t> uids( table.num_heads() );
	for ( int r = 0; r < m; ++r ) {
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
				emb[static_cast<size_t>( r ) * D + static_cast<size_t>( h ) * engine::loader::NgramTable::ROW_DIM + i] = backend::from_float_half( row[i] );
			}
		}
	}
	backend::memcpy_async( ple_emb_.data, emb.data(), emb.size() * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );   // emb dies with this frame
}

void Qwen4ExpPipeline::ple_forward( int m, backend::stream_t stream ) {
	const int H = config_.hidden_size, hc = config_.hc_mult;
	engine::modules::Linear::forward( ple_emb_, ple_.key_proj, ple_key_, stream );
	engine::modules::Linear::forward( ple_emb_, ple_.value_proj, ple_value_, stream );
	engine::modules::PLE::gate_and_norm( ple_key_.ptr<const backend::half_t>(), ple_value_.ptr<const backend::half_t>(), streams_.ptr<const float>(),
		ple_.norm_key.ptr<const backend::half_t>(), ple_.norm_query.ptr<const backend::half_t>(), ple_.norm_conv.ptr<const backend::half_t>(),
		ple_gated_, ple_normed_, m, H, hc, config_.rms_norm_eps, 1.0f / std::sqrt( static_cast<float>( H ) ), stream );
	engine::modules::PLE::conv_apply( ple_normed_, ple_.conv_w.ptr<const backend::half_t>(), ple_ring_, d_rows_, ple_gated_, streams_.ptr<float>(), m, hc * H, 3, stream );
	engine::modules::PLE::ring_update( ple_normed_, ple_ring_, d_rows_, m, hc * H, stream );
}

void Qwen4ExpPipeline::moe_forward( const Qwen4ExpLayerWeights& lw, int m, backend::stream_t stream ) {
	const int H = config_.hidden_size;
	engine::modules::MoERouter::route_softmax_topk( mixed_, lw.router, topk_idx_, topk_w_, H, config_.num_experts, config_.top_k, stream, m );
	if ( &lw == &layers_[0] ) trace_route( 700, m, stream );
	const int per_rank = config_.num_experts / tp_;
	// Plain SiLU gating (no swiglu limit): act_limit 0
	engine::modules::MoE::forward( mixed_, lw.experts, topk_idx_, topk_w_, y_, rank_ * per_rank, ( rank_ + 1 ) * per_rank, moe_scratch_, 0.0f, stream );
	if ( rank_ == 0 ) {
		engine::modules::Linear::forward( mixed_, lw.shared_gate, sh_gate_buf_, stream );
		engine::modules::Linear::forward( mixed_, lw.shared_up, sh_up_buf_, stream );
		engine::modules::SwiGLU::forward( sh_gate_buf_, sh_up_buf_, sh_inter_buf_, stream );
		engine::modules::Linear::forward( sh_inter_buf_, lw.shared_down, sh_out_, stream );
		engine::modules::GatedResidual::add_sigmoid_gated( y_.ptr<backend::half_t>(), sh_out_.ptr<const backend::half_t>(), mixed_.ptr<const backend::half_t>(),
			lw.shared_expert_gate.ptr<const backend::half_t>(), m, H, stream );
	}
}

void Qwen4ExpPipeline::forward_body( int m, backend::stream_t stream ) {
	const int H = config_.hidden_size, hc = config_.hc_mult;
	const int halt = runtime::options().qwen4_halt;
	engine::modules::GatedResidual::init_streams( embed_rows_.ptr<const backend::half_t>(), streams_.ptr<float>(), m, H, hc, stream );
	trace_rows( 1000, streams_.data, true, m, stream );
	int linear_idx = 0, attn_idx = 0, site = 0;
	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		if ( L == config_.ple_layer ) {
			ple_forward( m, stream );
			trace_rows( 800, streams_.data, true, m, stream );
		}
		// Attention site
		gr_mix( lw.attn_hc, true, m, stream );
		trace_rows( 100 + L, mixed_.data, false, m, stream );
		if ( lw.is_full_attention ) {
			attention_forward( lw, attn_idx++, m, stream );
		}
		else {
			gdn_forward( lw, linear_idx++, m, stream );
		}
		if ( ring_ ) ring_->all_reduce( y_, stream );
		trace_rows( 200 + L, y_.data, false, m, stream );
		engine::modules::GatedResidual::apply( streams_.ptr<float>(), post_, y_.ptr<const backend::half_t>(), m, H, hc, stream );
		trace_rows( 300 + L, streams_.data, true, m, stream );
		if ( halt == site++ ) return;
		// MLP site
		gr_mix( lw.mlp_hc, true, m, stream );
		trace_rows( 400 + L, mixed_.data, false, m, stream );
		moe_forward( lw, m, stream );
		if ( ring_ ) ring_->all_reduce( y_, stream );
		trace_rows( 500 + L, y_.data, false, m, stream );
		engine::modules::GatedResidual::apply( streams_.ptr<float>(), post_, y_.ptr<const backend::half_t>(), m, H, hc, stream );
		trace_rows( 600 + L, streams_.data, true, m, stream );
		if ( halt == site++ ) return;
	}

	// Final mixer (the gated residual without the post gate) collapses the streams into the head's input; every rank projects its vocabulary shard
	if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
	const int first = ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) ? m - 1 : 0;
	const int n = ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) ? 1 : m;
	engine::modules::GatedResidual::mix(
		streams_.ptr<const float>() + static_cast<size_t>( first ) * hc * H, final_mixer_.norm.ptr<const backend::half_t>(), final_mixer_.down.ptr<const backend::half_t>(),
		final_mixer_.up.ptr<const backend::half_t>(), nullptr, mixed_.ptr<backend::half_t>(), post_, gr_scratch_, n, H, hc, config_.hc_rank, config_.rms_norm_eps, stream );
	pod::TensorView hin = pod::make_tensor_2d( mixed_.ptr<backend::half_t>(), pod::DType::FP16, n, H );
	pod::TensorView lout = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, n, lm_shard_cols_ );
	engine::modules::Linear::forward( hin, lm_head_, lout, stream );
}

void Qwen4ExpPipeline::sample_tokens( int* out_preds, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( false, "qwen4_exp: the vocabulary is sharded, use sample_local" );
	(void) out_preds; (void) count; (void) stream;
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
void Qwen4ExpPipeline::checkpoint_state( int, backend::stream_t ) {}
void Qwen4ExpPipeline::rollback_state( int, backend::stream_t ) {}

void Qwen4ExpPipeline::reset_state( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Qwen4ExpPipeline::reset_state: slot out of range" );
	backend::set_device( rank_ );
	kv_pool_.reset_slot( slot_id );
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
	return recurrent + ring + kCtxBytes + kv + kSeqAlign;
}

// Host <-> device copy of every KV range of the snapshot, in layout order
void Qwen4ExpPipeline::copy_kv_rows( int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream ) {
	const size_t row_elems = static_cast<size_t>( kv_pool_.num_heads ) * kv_pool_.head_dim;
	const size_t count = static_cast<size_t>( tokens - from );
	if ( count == 0 ) return;
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
	copy_kv_rows( slot_id, from, tokens, dst, true, stream );
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
	copy_kv_rows( slot_id, from, tokens, const_cast<char*>( src ) + gdn + conv + ring + kCtxBytes, false, stream );
	return true;
}

} // namespace engine::models
