#include "engine/models/qwen4_exp.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include "engine/loader/safetensors.h"
#include <nlohmann/json.hpp>
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
		std::snprintf( nm, sizeof( nm ), "%s.mlp.gate.weight", pre ); lw.router = dense( nm );
		auto& mw = lw.experts;
		mw.type = pod::QuantType::TRELLIS;
		const int per_rank = config_.num_experts / tp;
		mw.num_local_experts = per_rank;
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

void Qwen4ExpPipeline::embed_tokens( const int*, int, backend::stream_t ) { ENGINE_ASSERT( false, "qwen4_exp: embed_tokens is not implemented yet" ); }
void Qwen4ExpPipeline::forward( const pod::BatchRows&, backend::stream_t ) { ENGINE_ASSERT( false, "qwen4_exp: forward is not implemented yet" ); }
void Qwen4ExpPipeline::sample_tokens( int*, int, backend::stream_t ) { ENGINE_ASSERT( false, "qwen4_exp: sampling is not implemented yet" ); }
void Qwen4ExpPipeline::checkpoint_state( int, backend::stream_t ) {}
void Qwen4ExpPipeline::rollback_state( int, backend::stream_t ) {}
void Qwen4ExpPipeline::reset_state( int ) {}

} // namespace engine::models
