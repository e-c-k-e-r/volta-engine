#include "engine/models/gemma4.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/rope.h"
#include "engine/modules/attention.h"
#include "engine/modules/linear.h"
#include "engine/modules/embedding.h"
#include "engine/modules/gguf.h"
#include "engine/modules/vision.h"
#include "engine/spec/dflash.h"
#include "engine/spec/tap_ring.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace engine::models {

bool read_gemma4_config( const char* model_dir, Gemma4Config* c ) {
	std::ifstream f( std::string( model_dir ) + "/config.json" );
	if ( !f ) {
		LOG_ERROR( "[gemma4] no config.json in %s\n", model_dir );
		return false;
	}
	const nlohmann::json j = nlohmann::json::parse( f, nullptr, false );
	if ( j.is_discarded() || !j.is_object() ) {
		LOG_ERROR( "[gemma4] %s/config.json is not a JSON object\n", model_dir );
		return false;
	}
	const nlohmann::json& t = j.contains( "text_config" ) && j["text_config"].is_object() ? j["text_config"] : j;
	bool ok = true;
	auto fail = [&]( const char* what ) {
		LOG_ERROR( "[gemma4] config.json: %s\n", what );
		ok = false;
	};
	// A field that is there and not null (another kind than the one expected fails the read)
	auto has = [&]( const nlohmann::json& o, const char* key ) { return o.contains( key ) && !o[key].is_null(); };
	auto num = [&]( const nlohmann::json& o, const char* key, auto* out ) {
		if ( !has( o, key ) ) return false;
		if ( !o[key].is_number() ) {
			LOG_ERROR( "[gemma4] config.json: %s is not a number\n", key );
			ok = false;
			return false;
		}
		*out = o[key].get<std::remove_pointer_t<decltype( out )>>();
		return true;
	};
	auto flag = [&]( const char* key ) {
		if ( !has( t, key ) ) return false;
		if ( !t[key].is_boolean() ) {
			LOG_ERROR( "[gemma4] config.json: %s is not true or false\n", key );
			ok = false;
			return false;
		}
		return t[key].get<bool>();
	};
	auto text = [&]( const nlohmann::json& o, const char* key ) { return has( o, key ) && o[key].is_string() ? o[key].get<std::string>() : std::string(); };

	num( t, "num_hidden_layers", &c->num_layers );
	num( t, "hidden_size", &c->hidden_size );
	num( t, "intermediate_size", &c->intermediate_size );
	num( t, "num_attention_heads", &c->num_q_heads );
	num( t, "num_key_value_heads", &c->swa_kv_heads );
	num( t, "head_dim", &c->swa_head_dim );
	num( t, "sliding_window", &c->swa_window );
	num( t, "global_head_dim", &c->full_head_dim );
	num( t, "vocab_size", &c->vocab_size );
	num( t, "rms_norm_eps", &c->rms_norm_eps );
	if ( t.contains( "final_logit_softcapping" ) && t["final_logit_softcapping"].is_null() ) c->final_softcap = 0.0f;   // none
	else num( t, "final_logit_softcapping", &c->final_softcap );
	c->full_k_eq_v = flag( "attention_k_eq_v" );
	// the full layers' KV heads: num_global_key_value_heads where K = V, else those of the sliding layers
	c->full_kv_heads = c->swa_kv_heads;
	if ( c->full_k_eq_v ) num( t, "num_global_key_value_heads", &c->full_kv_heads );
	c->kv_shared_layers = 0;
	num( t, "num_kv_shared_layers", &c->kv_shared_layers );
	c->ple_dim = 0;
	num( t, "hidden_size_per_layer_input", &c->ple_dim );
	c->ple_vocab = c->vocab_size;
	num( t, "vocab_size_per_layer_input", &c->ple_vocab );
	c->double_wide_shared = flag( "use_double_wide_mlp" );

	if ( flag( "enable_moe_block" ) ) fail( "MoE blocks (enable_moe_block) are not implemented" );
	if ( flag( "attention_bias" ) ) fail( "attention biases are not implemented" );
	if ( has( t, "hidden_activation" ) && text( t, "hidden_activation" ) != "gelu_pytorch_tanh" ) fail( "hidden_activation must be gelu_pytorch_tanh" );
	const std::string bidir = text( t, "use_bidirectional_attention" );
	if ( bidir == "all" ) fail( "bidirectional attention over all tokens (an encoder) is not implemented" );
	c->bidirectional_images = bidir == "vision";

	c->full.clear();
	if ( has( t, "layer_types" ) ) {
		const nlohmann::json& lt = t["layer_types"];
		if ( !lt.is_array() || static_cast<int>( lt.size() ) != c->num_layers ) fail( "layer_types must list every layer" );
		else {
			for ( const auto& v : lt ) {
				const std::string s = v.is_string() ? v.get<std::string>() : std::string();
				if ( s != "full_attention" && s != "sliding_attention" ) fail( "layer_types holds a kind other than full_attention and sliding_attention" );
				c->full.push_back( s == "full_attention" ? 1 : 0 );
			}
		}
	}

	// RoPE: the sliding layers' default rotation over the whole head, the full layers' proportional one (frequencies over the head, its first
	// partial_rotary_factor rotating)
	float partial = 0.25f;
	if ( has( t, "rope_parameters" ) && t["rope_parameters"].is_object() ) {
		const nlohmann::json& rp = t["rope_parameters"];
		if ( has( rp, "sliding_attention" ) ) {
			const nlohmann::json& s = rp["sliding_attention"];
			if ( has( s, "rope_type" ) && text( s, "rope_type" ) != "default" ) fail( "the sliding layers' rope_type must be default" );
			float pf = 1.0f;
			if ( num( s, "partial_rotary_factor", &pf ) && pf != 1.0f ) fail( "the sliding layers rotate the whole head" );
			num( s, "rope_theta", &c->swa_rope_theta );
		}
		if ( has( rp, "full_attention" ) ) {
			const nlohmann::json& s = rp["full_attention"];
			if ( has( s, "rope_type" ) && text( s, "rope_type" ) != "proportional" ) fail( "the full layers' rope_type must be proportional" );
			num( s, "partial_rotary_factor", &partial );
			num( s, "rope_theta", &c->full_rope_theta );
		}
	}
	c->full_rot_pairs = static_cast<int>( std::lround( partial * c->full_head_dim / 2.0f ) );

	if ( !ok ) return false;
	if ( c->num_layers < 1 || c->hidden_size < 1 || c->intermediate_size < 1 || c->num_q_heads < 1 || c->swa_kv_heads < 1 || c->full_kv_heads < 1 ||
		c->num_q_heads % c->swa_kv_heads != 0 || c->num_q_heads % c->full_kv_heads != 0 ) {
		fail( "layer, width or head counts out of range" );
		return false;
	}
	if ( c->full_rot_pairs < 1 || c->full_rot_pairs > c->full_head_dim / 2 ) {
		fail( "the full layers' partial_rotary_factor is out of range" );
		return false;
	}
	if ( c->kv_shared_layers < 0 || c->kv_shared_layers >= c->num_layers ) {
		fail( "num_kv_shared_layers out of range" );
		return false;
	}
	// every kind of layer among the sharing ones needs an earlier layer of its kind with K / V of its own
	for ( int L = c->first_shared(); L < c->num_layers; ++L ) {
		bool source = false;
		for ( int s = 0; s < c->first_shared() && !source; ++s ) source = c->is_full( s ) == c->is_full( L );
		if ( !source ) {
			fail( "a KV-sharing layer has no earlier layer of its kind" );
			return false;
		}
	}
	if ( c->ple_dim < 0 || ( c->ple_dim > 0 && c->ple_vocab < 1 ) ) {
		fail( "per-layer input sizes out of range" );
		return false;
	}
	return true;
}

Gemma4Pipeline::Gemma4Pipeline() = default;

Gemma4Pipeline::~Gemma4Pipeline() {
	destroy();
}

bool Gemma4Pipeline::init( const Gemma4Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	config_ = config;
	rank_ = rank;
	arena_ = &arena;
	ring_ = ring;
	weights_loaded_ = false;

	backend::set_device( rank_ );

	const int tp_size = ( ring_ != nullptr ) ? requested_tp_ : 1;
	ENGINE_ASSERT( tp_size == 1 || tp_size == 2 || tp_size == 4, "Gemma4Pipeline: tensor parallelism of 1, 2 or 4 ranks" );
	if ( config_.num_q_heads % tp_size != 0 || config_.swa_kv_heads % tp_size != 0 || config_.full_kv_heads % tp_size != 0 || config_.intermediate_size % tp_size != 0 ) {
		LOG_ERROR( "Gemma4Pipeline: %d ranks don't divide the heads (%d query, %d and %d KV) and the MLP width\n", tp_size, config_.num_q_heads, config_.swa_kv_heads,
			config_.full_kv_heads );
		return false;
	}
	tp_size_ = tp_size;
	q_heads_ = config_.num_q_heads / tp_size;
	swa_kv_heads_ = config_.swa_kv_heads / tp_size;
	full_kv_heads_ = config_.full_kv_heads / tp_size;
	inter_size_ = config_.intermediate_size / tp_size;
	const int R = pod::MAX_ROWS;
	const int S = config_.max_slots;

	// Each layer's kind, its K / V (its own, or the last earlier layer of its kind's from the first KV-sharing layer on) and its MLP width
	layers_.assign( config_.num_layers, Gemma4LayerWeights{} );
	int n_swa = 0, n_full = 0, max_inter = inter_size_;
	for ( int L = 0; L < config_.num_layers; ++L ) {
		Gemma4LayerWeights& lw = layers_[L];
		lw.is_full = config_.is_full( L );
		lw.shared_kv = L >= config_.first_shared();
		lw.kv_source = L;
		if ( lw.shared_kv ) {
			for ( int s = config_.first_shared() - 1; s >= 0; --s ) {
				if ( layers_[s].is_full == lw.is_full ) {
					lw.kv_source = s;
					break;
				}
			}
			lw.kv_layer = layers_[lw.kv_source].kv_layer;
		}
		else lw.kv_layer = lw.is_full ? n_full++ : n_swa++;
		lw.inter = inter_size_ * ( lw.shared_kv && config_.double_wide_shared ? 2 : 1 );
		max_inter = std::max( max_inter, lw.inter );
	}
	if ( rank_ == 0 ) {
		LOG_INFO( "[gemma4] %d layers (%d full attention), hidden %d, MLP %d, %d query heads; sliding %d x %d (window %d), full %d x %d%s%s\n",
			config_.num_layers, static_cast<int>( std::count( config_.full.begin(), config_.full.end(), 1 ) ) + ( config_.full.empty() ? config_.num_layers / 6 : 0 ),
			config_.hidden_size, config_.intermediate_size, config_.num_q_heads, config_.swa_kv_heads, config_.swa_head_dim, config_.swa_window, config_.full_kv_heads,
			config_.full_head_dim, config_.full_k_eq_v ? " (K = V)" : "", config_.ple_dim > 0 ? ", per-layer inputs" : "" );
		if ( config_.kv_shared_layers > 0 ) LOG_INFO( "[gemma4] layers %d-%d attend with earlier layers' K / V\n", config_.first_shared(), config_.num_layers - 1 );
	}
	if ( speculation_ == Speculation::DFlash ) {
		for ( int tap = 0; tap < Gemma4Config::kTaps; ++tap ) {
			if ( config_.tap_layers[tap] + runtime::options().dflash_tap_shift >= config_.num_layers ) {
				LOG_ERROR( "Gemma4Pipeline: the DFlash drafter taps layer %d of a %d-layer model\n", config_.tap_layers[tap], config_.num_layers );
				return false;
			}
		}
	}
	// The sliding-window layers keep a ring of window + MAX_ROWS rows: a forward stores all its rows before attending, so a chunk's
	// first row still finds its window after the chunk's last row was written. A context no longer than that keeps every position.
	const int swa_ring = config_.max_seq > config_.swa_window + R ? config_.swa_window + R : 0;
	swa_pool_.init( rank_, arena, S, swa_ring > 0 ? swa_ring : config_.max_seq, n_swa, swa_kv_heads_, config_.swa_head_dim );
	swa_pool_.ring = swa_ring;
	full_pool_.init( rank_, arena, S, config_.max_seq, n_full, full_kv_heads_, config_.full_head_dim );

	const int H = config_.hidden_size;
	int h_shape[2] = { R, H };
	resid_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, h_shape, 2 );
	hidden_states_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	norm_hidden_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	mlp_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	backend::memset_sync( resid_.data, 0, resid_.size_bytes() );

	d_token_ids_ = static_cast<int*>( engine::core::arena_alloc( arena, R * sizeof( int ) ) );
	d_rows_ = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );
	if ( config_.kv_shared_layers > 0 ) d_tail_rows_ = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );

	// Attention buffers sized for the wider of the two layer kinds
	const int q_cols = q_heads_ * std::max( config_.swa_head_dim, config_.full_head_dim );
	const int kv_cols = std::max( swa_kv_heads_ * config_.swa_head_dim, full_kv_heads_ * config_.full_head_dim );
	int q_shape[2] = { R, q_cols };
	q_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	attn_out_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	int kv_shape[2] = { R, kv_cols };
	k_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );
	v_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, kv_shape, 2 );

	int mlp_shape[2] = { R, max_inter };
	gate_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	up_buf_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );
	mlp_inter_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, mlp_shape, 2 );

	// Per-layer inputs: the rows' table values, the projection of their embeddings, and both mixed, layer-major (gate_buf_ and up_buf_
	// hold the per-layer gate and its product with the slice)
	if ( config_.ple_dim > 0 ) {
		ENGINE_ASSERT( config_.ple_dim <= max_inter, "Gemma4Pipeline: the per-layer input is wider than the MLP" );
		int ple_shape[2] = { R, config_.num_layers * config_.ple_dim };
		ple_tok_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ple_shape, 2 );
		ple_proj_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ple_shape, 2 );
		ple_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ple_shape, 2 );
	}

	// Vocabulary-parallel head: every rank projects vocab / tp columns, padded to a multiple of 256 (the int8 GEMV's width)
	lm_shard_real_cols_ = config_.vocab_size / tp_size_;
	lm_shard_cols_ = lm_shard_real_cols_ + ( tp_size_ > 1 ? ( 256 - lm_shard_real_cols_ % 256 ) % 256 : 0 );
	{
		int logit_shape[2] = { pod::MAX_BATCH, lm_shard_cols_ };
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
		if ( rank_ == 0 && tp_size_ > 1 ) {
			int full_shape[2] = { pod::MAX_BATCH, config_.vocab_size };
			full_logits_ = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, full_shape, 2 );
		}
	}

	// x sqrt( hidden ) on the embedding rows, the multiplier rounded through BF16 as the reference does
	{
		const float s = std::sqrt( static_cast<float>( H ) );
		uint32_t u;
		std::memcpy( &u, &s, 4 );
		u = ( u + 0x7FFF + ( ( u >> 16 ) & 1 ) ) & 0xFFFF0000u;
		std::memcpy( &embed_scale_, &u, 4 );
	}

	// DFlash drafter: a tap ring per slot (the hidden after each tapped layer) and the head scratch
	if ( speculation_ == Speculation::DFlash ) {
		engine::models::alloc_draft_taps( arena, draft_hooks_, draft_, config_.max_seq, H, S, true, rank_, tp_size_, Gemma4Config::kTaps );
		engine::models::alloc_draft_head( arena, draft_hooks_, draft_, H, lm_shard_cols_, kTopkTile, rank_ * lm_shard_real_cols_, lm_shard_real_cols_ );
		draft_hooks_.embedding = &embed_tokens_;
		draft_hooks_.embed_scale = embed_scale_;
		tap_shift_ = runtime::options().dflash_tap_shift;
	}

#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );
#endif

	metadata_.model_name = "gemma4";
	metadata_.hidden_size = H;
	metadata_.vocab_size = config_.vocab_size;
	metadata_.num_layers = config_.num_layers;
	metadata_.num_heads = config_.num_q_heads;
	metadata_.num_kv_heads = config_.swa_kv_heads;
	metadata_.head_dim = config_.swa_head_dim;
	metadata_.max_seq_len = config_.max_seq;
	metadata_.tp_size = tp_size;
	metadata_.max_slots = S;
	metadata_.max_rows = R;
	metadata_.eos_ids[0] = 1;     // <eos>
	metadata_.eos_ids[1] = 106;   // <turn|>
	metadata_.eos_ids[2] = 50;    // <|tool_response>: after its calls the model hands the turn to the tool
	return true;
}

bool Gemma4Pipeline::check_config( bool log ) {
	(void) log;   // read_gemma4_config always says why it fails
	Gemma4Config c;
	return read_gemma4_config( checkpoint_dir_.c_str(), &c );
}

bool Gemma4Pipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	Gemma4Config config;
	if ( !checkpoint_dir_.empty() && !read_gemma4_config( checkpoint_dir_.c_str(), &config ) ) return false;
	if ( runtime::options().max_seq > 0 ) config.max_seq = runtime::options().max_seq;
	if ( runtime::options().max_slots > 0 ) config.max_slots = runtime::options().max_slots;
	return init( config, device_id, arena, ring );
}

void Gemma4Pipeline::reset_state( int slot_id ) {
	ENGINE_ASSERT( slot_id >= 0 && slot_id < config_.max_slots, "Gemma4Pipeline::reset_state: slot out of range" );
	backend::set_device( rank_ );
	swa_pool_.reset_slot( slot_id );
	full_pool_.reset_slot( slot_id );
}

// A one-element tensor (BF16, FP16 or FP32) read on the host
static float host_scalar( const engine::loader::SafeTensorsRepo& repo, const char* name ) {
	const pod::SafeTensorMeta* meta = repo.find( name );
	const void* p = repo.cpu_data( name );
	ENGINE_ASSERT( meta != nullptr && p != nullptr, "Gemma4Pipeline: missing scalar tensor" );
	if ( meta->dtype == pod::DType::FP32 ) return *static_cast<const float*>( p );
	if ( meta->dtype == pod::DType::BF16 ) return backend::to_float( *static_cast<const backend::bf16_t*>( p ) );
	ENGINE_ASSERT( meta->dtype == pod::DType::FP16, "Gemma4Pipeline: scalar tensor of an unexpected type" );
	return backend::to_float( *static_cast<const backend::half_t*>( p ) );
}

bool Gemma4Pipeline::load_weights( const char* model_dir, backend::stream_t stream ) {
	engine::loader::SafeTensorsRepo repo;
	if ( !repo.open( model_dir ) ) {
		LOG_ERROR( "[loader] FAILED to open model directory %s\n", model_dir );
		return false;
	}
	const char* P = "model.language_model";
	char name[256];

	std::snprintf( name, sizeof( name ), "%s.embed_tokens.weight", P );
	if ( runtime::options().embed_host ) embed_tokens_ = repo.shared_host_bf16( name, stream );
	if ( !embed_tokens_.data ) embed_tokens_ = repo.load( name, *arena_, stream, pod::DType::BF16 );

	std::snprintf( name, sizeof( name ), "%s.norm.weight", P );
	final_norm_ = repo.load( name, *arena_, stream, pod::DType::FP16 );

	// NVFP4 checkpoint (modelopt): the MLP in NVFP4, attention and the tied head dense BF16. nvfp4_dense groups (attn, head, or all)
	// quantize those to NVFP4 at load
	std::snprintf( name, sizeof( name ), "%s.layers.0.mlp.gate_proj.weight_scale", P );
	const bool fp4 = repo.find( name ) != nullptr;
	const std::string nv_groups = "," + runtime::options().nvfp4_dense + ",";
	auto nv_group = [&]( const char* g ) { return nv_groups == ",all," || nv_groups.find( std::string( "," ) + g + "," ) != std::string::npos; };
	if ( fp4 ) LOG_ERROR( "[RANK %d] NVFP4 checkpoint\n", rank_ );

	// Tied head: the EXL3 checkpoint quantizes it as lm_head, the NVFP4 one reads the embedding
	const int pad = lm_shard_cols_ - lm_shard_real_cols_;
	if ( fp4 && ( tp_size_ > 1 || rank_ == 0 ) ) {
		std::snprintf( name, sizeof( name ), "%s.embed_tokens", P );
		const int beg = rank_ * lm_shard_real_cols_, cnt = lm_shard_real_cols_;
		if ( nv_group( "head" ) ) lm_head_ = repo.quantize_nvfp4( name, *arena_, &beg, &cnt, 1, 0, config_.hidden_size, lm_shard_cols_, stream );
		else {
			std::snprintf( name, sizeof( name ), "%s.embed_tokens.weight", P );
			lm_head_ = repo.load_dense_rows_padded( name, *arena_, beg, cnt, pad, stream );
		}
	}
	else if ( tp_size_ > 1 ) lm_head_ = repo.load_linear_slice_padded( "lm_head", *arena_, 6, rank_, tp_size_, pad, stream );
	else if ( rank_ == 0 ) lm_head_ = repo.load_linear( "lm_head", *arena_, 6, stream );

	const int tp = tp_size_;
	// A dense projection: FP16 sliced along its output rows (column parallel) or input columns (row parallel), or NVFP4 at load
	auto dense_proj = [&]( const char* base, bool split_n ) {
		if ( nv_group( "attn" ) ) return repo.quantize_nvfp4_slice( base, *arena_, split_n, rank_, tp, 0, stream );
		char b[320];
		std::snprintf( b, sizeof( b ), "%s.weight", base );
		pod::TensorView w = tp > 1 ? repo.load_slice( b, *arena_, split_n ? 0 : 1, rank_, tp, stream, pod::DType::FP16 ) : repo.load( b, *arena_, stream, pod::DType::FP16 );
		ENGINE_ASSERT( w.data != nullptr && w.ndim == 2, "Gemma4Pipeline: missing dense projection" );
		std::swap( w.shape[0], w.shape[1] );   // [n, k] in memory, tagged [k, n] for make_fp16
		return pod::LinearWeight::make_fp16( w );
	};
	auto packed = [&]( const char* base ) {
		char b[320];
		std::snprintf( b, sizeof( b ), "%s.weight_scale", base );
		return repo.find( b ) != nullptr;
	};
	auto load_col = [&]( const char* n ) {
		if ( fp4 ) return packed( n ) ? repo.load_nvfp4_slice( n, *arena_, true, rank_, tp, stream ) : dense_proj( n, true );
		return tp > 1 ? repo.load_linear_slice( n, *arena_, 4, 1, rank_, tp, stream ) : repo.load_linear( n, *arena_, 4, stream );
	};
	auto load_row = [&]( const char* n ) {
		if ( fp4 ) return packed( n ) ? repo.load_nvfp4_slice( n, *arena_, false, rank_, tp, stream ) : dense_proj( n, false );
		return tp > 1 ? repo.load_linear_slice( n, *arena_, 4, 0, rank_, tp, stream ) : repo.load_linear( n, *arena_, 4, stream );
	};
	auto load_norm = [&]( int L, const char* n ) {
		std::snprintf( name, sizeof( name ), "%s.layers.%d.%s.weight", P, L, n );
		return repo.load( name, *arena_, stream, pod::DType::FP16 );
	};

	for ( int L = 0; L < config_.num_layers; ++L ) {
		auto& lw = layers_[L];
		lw.input_norm = load_norm( L, "input_layernorm" );
		lw.post_attn_norm = load_norm( L, "post_attention_layernorm" );
		lw.pre_ffn_norm = load_norm( L, "pre_feedforward_layernorm" );
		lw.post_ffn_norm = load_norm( L, "post_feedforward_layernorm" );
		lw.q_norm = load_norm( L, "self_attn.q_norm" );
		if ( !lw.shared_kv ) lw.k_norm = load_norm( L, "self_attn.k_norm" );
		std::snprintf( name, sizeof( name ), "%s.layers.%d.layer_scalar", P, L );
		lw.layer_scalar = host_scalar( repo, name );

		std::snprintf( name, sizeof( name ), "%s.layers.%d.self_attn.q_proj", P, L );
		lw.q_proj = load_col( name );
		if ( !lw.shared_kv ) {
			std::snprintf( name, sizeof( name ), "%s.layers.%d.self_attn.k_proj", P, L );
			lw.k_proj = load_col( name );
			if ( !lw.is_full || !config_.full_k_eq_v ) {
				std::snprintf( name, sizeof( name ), "%s.layers.%d.self_attn.v_proj", P, L );
				lw.v_proj = load_col( name );
			}
		}
		std::snprintf( name, sizeof( name ), "%s.layers.%d.self_attn.o_proj", P, L );
		lw.o_proj = load_row( name );

		std::snprintf( name, sizeof( name ), "%s.layers.%d.mlp.gate_proj", P, L );
		lw.gate_proj = load_col( name );
		std::snprintf( name, sizeof( name ), "%s.layers.%d.mlp.up_proj", P, L );
		lw.up_proj = load_col( name );
		std::snprintf( name, sizeof( name ), "%s.layers.%d.mlp.down_proj", P, L );
		lw.down_proj = load_row( name );

		// the per-layer input's projections are small: every rank holds them whole
		if ( config_.ple_dim > 0 ) {
			std::snprintf( name, sizeof( name ), "%s.layers.%d.per_layer_input_gate", P, L );
			lw.ple_gate = repo.load_linear( name, *arena_, 4, stream );
			std::snprintf( name, sizeof( name ), "%s.layers.%d.per_layer_projection", P, L );
			lw.ple_proj = repo.load_linear( name, *arena_, 4, stream );
			lw.post_ple_norm = load_norm( L, "post_per_layer_input_norm" );
		}
	}
	if ( config_.ple_dim > 0 && !load_ple_table( repo, model_dir, stream ) ) return false;
	if ( stream ) backend::stream_sync( stream );
	repo.close();
	weights_loaded_ = true;
	return true;
}

// The per-layer input table stays where the checkpoint maps it (a repo of its own, open as long as the pipeline): a forward copies its rows'
// bytes into a pinned stage and up, where block rows are decoded (LinearGguf::dequant) and float rows converted on the host first
bool Gemma4Pipeline::load_ple_table( engine::loader::SafeTensorsRepo& repo, const char* model_dir, backend::stream_t stream ) {
	const char* P = "model.language_model";
	char name[256];
	std::snprintf( name, sizeof( name ), "%s.per_layer_model_projection", P );
	ple_model_proj_ = repo.load_linear( name, *arena_, 4, stream );
	std::snprintf( name, sizeof( name ), "%s.per_layer_projection_norm.weight", P );
	ple_proj_norm_ = repo.load( name, *arena_, stream, pod::DType::FP16 );

	ple_repo_ = std::make_unique<engine::loader::SafeTensorsRepo>();
	std::snprintf( name, sizeof( name ), "%s.embed_tokens_per_layer.weight", P );
	const pod::SafeTensorMeta* meta = ple_repo_->open( model_dir ) ? ple_repo_->find( name ) : nullptr;
	const int width = config_.num_layers * config_.ple_dim;
	if ( meta == nullptr || meta->ndim != 2 || meta->shape[0] != config_.ple_vocab || meta->shape[1] != width ) {
		LOG_ERROR( "[RANK %d] Gemma4Pipeline: %s missing or not [%d, %d]\n", rank_, name, config_.ple_vocab, width );
		return false;
	}
	ple_table_.rows = static_cast<const uint8_t*>( ple_repo_->cpu_data( name ) );
	ple_table_.dtype = meta->dtype;
	ple_table_.gguf_type = meta->gguf_type;
	ple_table_.row_bytes = meta->size_bytes / static_cast<size_t>( meta->shape[0] );
	const bool blocks = meta->dtype == pod::DType::GGUF;
	if ( ple_table_.rows == nullptr || ( !blocks && meta->dtype != pod::DType::BF16 && meta->dtype != pod::DType::FP16 && meta->dtype != pod::DType::FP32 ) ) {
		LOG_ERROR( "[RANK %d] Gemma4Pipeline: %s is not readable on the host as block, BF16, FP16 or FP32 rows\n", rank_, name );
		return false;
	}
	const int R = pod::MAX_ROWS;
	const size_t stage_bytes = static_cast<size_t>( R ) * std::max( ple_table_.row_bytes, width * sizeof( backend::half_t ) );
	ple_stage_ = static_cast<uint8_t*>( engine::core::host_alloc_pinned( stage_bytes ) );
	if ( blocks ) ple_blocks_ = engine::core::arena_alloc( *arena_, static_cast<size_t>( R ) * ple_table_.row_bytes );
	if ( rank_ == 0 ) {
		const char* kind = blocks ? engine::modules::ggml_type_name( static_cast<engine::modules::GgmlType>( ple_table_.gguf_type ) )
			: meta->dtype == pod::DType::BF16 ? "BF16" : meta->dtype == pod::DType::FP16 ? "FP16" : "FP32";
		LOG_INFO( "[gemma4] per-layer inputs: a [%d, %d] table (%s, %.2f GiB) read on the host\n", config_.ple_vocab, width, kind,
			static_cast<double>( meta->size_bytes ) / ( 1u << 30 ) );
	}
	return true;
}

void Gemma4Pipeline::set_rows( int m ) {
	resid_.shape[0] = m;
	hidden_states_.shape[0] = m;
	norm_hidden_.shape[0] = m;
	q_buf_.shape[0] = m;
	k_buf_.shape[0] = m;
	v_buf_.shape[0] = m;
	attn_out_.shape[0] = m;
	gate_buf_.shape[0] = m;
	up_buf_.shape[0] = m;
	mlp_inter_.shape[0] = m;
	mlp_out_.shape[0] = m;
	logits_.shape[0] = std::min( m, pod::MAX_BATCH );
}

void Gemma4Pipeline::embed_tokens( const int* token_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_ROWS, "Gemma4Pipeline::embed_tokens: count out of range" );
	backend::set_device( rank_ );
	backend::memcpy_async( d_token_ids_, token_ids, count * sizeof( int ), backend::Copy::H2D, stream );
	set_rows( count );
	engine::modules::Embedding::forward( embed_tokens_, d_token_ids_, count, hidden_states_, stream );
	engine::modules::Elementwise::scale_to_f32( hidden_states_, resid_, embed_scale_, stream );
	if ( config_.ple_dim > 0 ) ple_rows( token_ids, count, stream );
}

// The per-layer table's rows of a forward's tokens (an id outside the table reads row 0, as the embedding does): their bytes into the stage,
// up to the device and to FP16 there (block rows), or converted on the host first (float rows)
void Gemma4Pipeline::ple_rows( const int* token_ids, int count, backend::stream_t stream ) {
	const int width = config_.num_layers * config_.ple_dim;
	const size_t rb = ple_table_.row_bytes;
	const bool blocks = ple_table_.dtype == pod::DType::GGUF;
	for ( int r = 0; r < count; ++r ) {
		const int id = token_ids[r] >= 0 && token_ids[r] < config_.ple_vocab ? token_ids[r] : 0;
		const uint8_t* src = ple_table_.rows + static_cast<size_t>( id ) * rb;
		if ( blocks || ple_table_.dtype == pod::DType::FP16 ) {
			std::memcpy( ple_stage_ + static_cast<size_t>( r ) * rb, src, rb );
			continue;
		}
		backend::half_t* dst = reinterpret_cast<backend::half_t*>( ple_stage_ ) + static_cast<size_t>( r ) * width;
		if ( ple_table_.dtype == pod::DType::BF16 ) {
			const backend::bf16_t* s = reinterpret_cast<const backend::bf16_t*>( src );
			for ( int i = 0; i < width; ++i ) dst[i] = backend::from_float_half( backend::to_float( s[i] ) );
		}
		else {
			const float* s = reinterpret_cast<const float*>( src );
			for ( int i = 0; i < width; ++i ) dst[i] = backend::from_float_half( s[i] );
		}
	}
	if ( blocks ) {
		backend::memcpy_async( ple_blocks_, ple_stage_, static_cast<size_t>( count ) * rb, backend::Copy::H2D, stream );
		engine::modules::LinearGguf::dequant( ple_blocks_, static_cast<engine::modules::GgmlType>( ple_table_.gguf_type ), static_cast<int64_t>( count ) * width,
			ple_tok_.ptr<backend::half_t>(), stream );
	}
	else backend::memcpy_async( ple_tok_.data, ple_stage_, static_cast<size_t>( count ) * width * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	backend::stream_sync( stream );   // the next forward reuses the stage
}

// The rows' per-layer inputs, layer-major: ( rms_norm( the projection of their embeddings ) * w + the table rows * sqrt( ple_dim ) ) / sqrt( 2 ). The
// reference projects the scaled embeddings times 1 / sqrt( hidden ): the raw rows projected differ by c = embed_scale / sqrt( hidden ), which the
// norm removes but for its eps (eps / c^2 here)
void Gemma4Pipeline::ple_mix( int m, backend::stream_t stream ) {
	const int L = config_.num_layers, D = config_.ple_dim;
	pod::TensorView proj = pod::make_tensor_2d( ple_proj_.data, pod::DType::FP16, m, L * D );
	pod::TensorView tok = pod::make_tensor_2d( ple_tok_.data, pod::DType::FP16, m, L * D );
	pod::TensorView out = pod::make_tensor_2d( ple_.data, pod::DType::FP16, m, L * D );
	engine::modules::Linear::forward( hidden_states_, ple_model_proj_, proj, stream );
	const float c = embed_scale_ / std::sqrt( static_cast<float>( config_.hidden_size ) );
	engine::modules::Elementwise::ple_mix( proj, tok, ple_proj_norm_, out, L, D, config_.rms_norm_eps / ( c * c ), std::sqrt( static_cast<float>( D ) ),
		1.0f / std::sqrt( 2.0f ), stream );
}

void Gemma4Pipeline::embed_tokens_device( const int* d_tokens, int src_dev, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_ROWS, "Gemma4Pipeline::embed_tokens_device: count out of range" );
	ENGINE_ASSERT( config_.ple_dim == 0, "Gemma4Pipeline::embed_tokens_device: the per-layer inputs need the ids on the host" );
	backend::set_device( rank_ );
	backend::peer_copy_async( d_token_ids_, rank_, d_tokens, src_dev, count * sizeof( int ), stream );
	set_rows( count );
	engine::modules::Embedding::forward( embed_tokens_, d_token_ids_, count, hidden_states_, stream );
	engine::modules::Elementwise::scale_to_f32( hidden_states_, resid_, embed_scale_, stream );
}

void Gemma4Pipeline::sample_tokens( int* out_preds, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( tp_size_ == 1, "Gemma4Pipeline::sample_tokens: vocabulary is sharded, use sample_local" );
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Gemma4Pipeline::sample_tokens: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax( logits_, d_argmax_idx_, lm_shard_cols_, count, stream );
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_preds, h_argmax_idx_, count * sizeof( int ) );
}

bool Gemma4Pipeline::sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream ) {
	if ( tp_size_ <= 1 ) return false;
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Gemma4Pipeline::sample_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax_range( logits_, d_argmax_idx_, d_argmax_val_, lm_shard_real_cols_, count, rank_ * lm_shard_real_cols_, stream );
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	if ( out_vals != nullptr ) backend::memcpy_async( out_vals, d_argmax_val_, count * sizeof( float ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_tokens, h_argmax_idx_, count * sizeof( int ) );
	return true;
}

bool Gemma4Pipeline::lse_local( float* out, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Gemma4Pipeline::lse_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	engine::spec::logits_lse_shard( logits_.ptr<const backend::half_t>(), count, lm_shard_cols_, lm_shard_real_cols_, d_samp_vals_, stream );
	backend::memcpy_async( out, d_samp_vals_, static_cast<size_t>( count ) * 2 * sizeof( float ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

bool Gemma4Pipeline::topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "Gemma4Pipeline::topk_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	engine::spec::dflash2_topk_shard( logits_.ptr<const backend::half_t>(), count, lm_shard_cols_, lm_shard_real_cols_, 16,
		d_samp_vals_, d_samp_idx_, rank_ * lm_shard_real_cols_, samp_part_v_, samp_part_i_, samp_counters_, stream );
	backend::memcpy_async( out_vals, d_samp_vals_, static_cast<size_t>( count ) * 16 * sizeof( float ), backend::Copy::D2H, stream );
	backend::memcpy_async( out_ids, d_samp_idx_, static_cast<size_t>( count ) * 16 * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

const pod::TensorView& Gemma4Pipeline::gather_logits( int rows, const backend::half_t* const* peer_shard_ptrs, int peer_shard_cols, backend::stream_t stream ) {
	ENGINE_ASSERT( tp_size_ == 1 || rank_ == 0, "Gemma4Pipeline::gather_logits: only rank 0 materializes the full vocabulary" );
	if ( tp_size_ == 1 ) return logits_;
	ENGINE_ASSERT( rows >= 1 && rows <= last_rows_ && rows <= pod::MAX_BATCH, "Gemma4Pipeline::gather_logits: rows out of range" );
	ENGINE_ASSERT( peer_shard_cols == lm_shard_cols_, "Gemma4Pipeline::gather_logits: shard width mismatch" );
	full_logits_.shape[0] = rows;
	const size_t shard_elems = static_cast<size_t>( lm_shard_real_cols_ );
	for ( int d = 0; d < tp_size_; ++d ) {
		const backend::half_t* src_base = ( d == rank_ ) ? logits_.ptr<const backend::half_t>() : peer_shard_ptrs[d];
		ENGINE_ASSERT( src_base != nullptr, "Gemma4Pipeline::gather_logits: missing peer shard pointer" );
		for ( int r = 0; r < rows; ++r ) {
			backend::half_t* dst = full_logits_.ptr<backend::half_t>() + ( static_cast<size_t>( r ) * tp_size_ + d ) * shard_elems;
			backend::peer_copy_async( dst, rank_, src_base + static_cast<size_t>( r ) * lm_shard_cols_, d, shard_elems * sizeof( backend::half_t ), stream );
		}
	}
	return full_logits_;
}

// The head shard over the drafter's hidden rows, then this rank's top-k (the softcap keeps the order, so it is skipped)
void Gemma4Pipeline::draft_head_topk( int rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_ && draft_.lm_logits.data != nullptr, "Gemma4Pipeline::draft_head_topk: no drafter head" );
	ENGINE_ASSERT( rows >= 1 && rows <= kDraftM, "Gemma4Pipeline::draft_head_topk: rows out of range" );
	backend::set_device( rank_ );
	pod::TensorView in = pod::make_tensor_2d( draft_hooks_.lm_in, pod::DType::FP16, kDraftM, config_.hidden_size );
	engine::modules::Linear::forward( in, lm_head_, draft_.lm_logits, stream );
	engine::spec::dflash2_topk_shard( draft_.lm_logits.ptr<const backend::half_t>(), kDraftM, lm_shard_cols_, lm_shard_real_cols_, kDraftTopK,
		draft_hooks_.cand_vals, draft_hooks_.cand_idx, draft_hooks_.shard_offset, draft_.cand_part_v, draft_.cand_part_i, draft_.cand_counters, stream );
}

bool Gemma4Pipeline::load_vision( const char* model_dir, backend::stream_t stream ) {
	if ( rank_ != 0 || runtime::options().max_image_tokens <= 0 ) return false;
	backend::set_device( rank_ );
	return vision_.load( model_dir, *arena_, runtime::options().max_image_tokens, stream );
}

bool Gemma4Pipeline::encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out, backend::stream_t stream ) {
	if ( rank_ != 0 || !vision_.loaded() ) return false;
	backend::set_device( rank_ );
	return vision_.encode( image, out, stream );
}

bool Gemma4Pipeline::inject_embeddings( const backend::half_t* host_rows, int dst_row, int rows, backend::stream_t stream ) {
	ENGINE_ASSERT( dst_row >= 0 && rows >= 1 && dst_row + rows <= pod::MAX_ROWS, "Gemma4Pipeline::inject_embeddings: rows out of range" );
	backend::set_device( rank_ );
	const int H = config_.hidden_size;
	backend::half_t* stage = hidden_states_.ptr<backend::half_t>() + static_cast<size_t>( dst_row ) * H;
	backend::memcpy_async( stage, host_rows, static_cast<size_t>( rows ) * H * sizeof( backend::half_t ), backend::Copy::H2D, stream );
	pod::TensorView in = pod::make_tensor_2d( stage, pod::DType::FP16, rows, H );
	pod::TensorView dst = pod::make_tensor_2d( resid_.ptr<float>() + static_cast<size_t>( dst_row ) * H, pod::DType::FP32, rows, H );
	engine::modules::Elementwise::scale_to_f32( in, dst, 1.0f, stream );
	return true;
}

bool Gemma4Pipeline::read_taps( int slot_id, int first_pos, int count, backend::half_t* host_dst, backend::stream_t stream ) {
	backend::set_device( rank_ );
	return read_draft_taps( draft_hooks_, slot_id, config_.max_slots, first_pos, count, host_dst, stream );
}

void Gemma4Pipeline::forward( const pod::BatchRows& rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "Gemma4Pipeline::forward: weights not loaded" );
	ENGINE_ASSERT( rows.count >= 1 && rows.count <= pod::MAX_ROWS, "Gemma4Pipeline::forward: row count out of range" );
	for ( int r = 0; r < rows.count; ++r ) {
		ENGINE_ASSERT( rows.slot[r] >= 0 && rows.slot[r] < config_.max_slots, "Gemma4Pipeline::forward: slot out of range" );
		ENGINE_ASSERT( rows.pos[r] >= 0 && rows.pos[r] < config_.max_seq, "Gemma4Pipeline::forward: position out of range" );
	}
	backend::set_device( rank_ );
	const int m = rows.count;
	lm_mode_ = m > pod::MAX_BATCH ? rows.logits_mode : pod::BatchRows::LOGITS_ALL;
	last_rows_ = ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) ? 0 : ( lm_mode_ == pod::BatchRows::LOGITS_LAST ? 1 : m );
	set_rows( m );
	key_hint_ = rows.pos[m - 1] + 1;
	backend::memcpy_async( d_rows_, &rows, rows.upload_bytes(), backend::Copy::H2D, stream );
	rows_ = d_rows_;
	ple_m_ = m;
	ple_row0_ = 0;
	// A prefill chunk reading one row's logits or none: the KV-sharing layers (which store nothing) run over its last kTailRows rows only, not at
	// all without logits. kTailRows is the smallest prefill-sized forward, so the last row takes the kernels it takes in the whole chunk and its
	// logits are the chunk's. Not with drafter taps, which read every row of every tapped layer; from twice that many rows (their move to the
	// front doesn't overlap)
	tail_ = config_.kv_shared_layers > 0 && m >= 2 * kTailRows && lm_mode_ != pod::BatchRows::LOGITS_ALL && draft_.taps.data == nullptr;
	if ( tail_ ) {
		pod::BatchRows& t = tail_host_;
		t.count = kTailRows;
		t.logits_mode = rows.logits_mode;
		t.mrope = rows.mrope;
		t.bidir = rows.bidir;
		for ( int i = 0; i < kTailRows; ++i ) {
			const int r = m - kTailRows + i;
			t.slot[i] = rows.slot[r];
			t.pos[i] = rows.pos[r];
			for ( int k = 0; k < 3; ++k ) t.rope_pos[k][i] = rows.rope_pos[k][r];
			t.key_end[i] = rows.key_end[r];
		}
		backend::memcpy_async( d_tail_rows_, &t, t.upload_bytes(), backend::Copy::H2D, stream );
	}
	// Tensor-core attention shares K/V tiles across the rows of one slot; rows of distinct slots get a block each (a separate graph)
	attn_tc_ = true;
	for ( int r = 1; r < m; ++r ) attn_tc_ = attn_tc_ && rows.slot[r] == rows.slot[0];
	graphs_.run( m, stream, [&]( backend::stream_t s ) { forward_body( m, s ); }, attn_tc_ ? 1 : 0 );
}

// One decoder layer over the m rows of rows_ (resid_ in and out):
//   x += post_attn_norm( attn( input_norm( x ) ) ); x += post_ffn_norm( mlp( pre_ffn_norm( x ) ) );
//   [ x += post_ple_norm( ple_proj( gelu( ple_gate( x ) ) * the layer's per-layer input ) ) ]; x *= layer_scalar
// With TP the reductions of the two sublayer outputs carry the sandwich-norm add and the next norm (all_reduce_normed_add): the attention one writes
// the pre-FFN norm, the FFN one the next layer's input norm (next_norm, without per-layer inputs), so that layer starts from norm_hidden_
// (input_normed). Returns whether the next input norm was written. A KV-sharing layer projects only its queries and attends with its source's K / V.
bool Gemma4Pipeline::layer_forward( const Gemma4LayerWeights& lw, int layer, int m, backend::stream_t stream, bool input_normed, const pod::TensorView* next_norm ) {
	using namespace engine::modules;
	const float eps = config_.rms_norm_eps;
	const engine::runtime::KVCachePool& pool = lw.is_full ? full_pool_ : swa_pool_;
	const pod::KVRef kv = pool.ref();
	const int hd = lw.is_full ? config_.full_head_dim : config_.swa_head_dim;
	const int kvh = lw.is_full ? full_kv_heads_ : swa_kv_heads_;
	const int kv_layer = lw.kv_layer;

	if ( !input_normed ) RMSNorm::forward( resid_, lw.input_norm, norm_hidden_, eps, 0.0f, stream );

	pod::TensorView q = pod::make_tensor_2d( q_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, q_heads_ * hd );
	pod::TensorView k = pod::make_tensor_2d( k_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, kvh * hd );
	pod::TensorView v = pod::make_tensor_2d( v_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, kvh * hd );
	pod::TensorView o = pod::make_tensor_2d( attn_out_.ptr<backend::half_t>(), pod::DType::FP16, m, q_heads_ * hd );
	Linear::forward( norm_hidden_, lw.q_proj, q, stream );
	RMSNorm::forward_heads( q, lw.q_norm, q_heads_, hd, eps, 0.0f, stream );
	const float theta = lw.is_full ? config_.full_rope_theta : config_.swa_rope_theta;
	const int rot_pairs = lw.is_full ? config_.full_rot_pairs : 0;
	if ( lw.shared_kv ) RoPE::forward( q, k, q_heads_, 0, hd, hd, rows_, m, theta, stream, false, rot_pairs );   // the queries only
	else {
		Linear::forward( norm_hidden_, lw.k_proj, k, stream );
		// Full layers with K = V take V from the raw K projection (attention_k_eq_v)
		if ( lw.is_full && config_.full_k_eq_v ) backend::memcpy_async( v.data, k.data, k.size_bytes(), backend::Copy::D2D, stream );
		else Linear::forward( norm_hidden_, lw.v_proj, v, stream );
		RMSNorm::forward_heads( k, lw.k_norm, kvh, hd, eps, 0.0f, stream );
		RMSNorm::forward_heads( v, pod::TensorView{}, kvh, hd, eps, 0.0f, stream );   // unweighted v_norm
		RoPE::forward( q, k, q_heads_, kvh, hd, hd, rows_, m, theta, stream, false, rot_pairs );
		Attention::store_kv_cache( kv, k, v, pool.slot_stride_elems, kv_layer, rows_, m, kvh, hd, pool.max_seq_len, stream );
	}
	// Tensor cores: K/V tiles shared across the rows of one slot, a block per row otherwise
	Attention::forward_prefill( kv, q, pod::TensorView{}, o, pool.slot_stride_elems, kv_layer, rows_, m, q_heads_, kvh, hd, pool.max_seq_len, stream,
		m > pod::MAX_BATCH ? key_hint_ : 0, m <= pod::MAX_BATCH && !attn_tc_, 1.0f, lw.is_full ? 0 : config_.swa_window );

	Linear::forward( o, lw.o_proj, mlp_out_, stream );
	const bool fuse = ring_ != nullptr && runtime::options().normed_add_fuse;
	if ( fuse ) ring_->all_reduce_normed_add( mlp_out_, resid_, lw.post_attn_norm, 1.0f, &lw.pre_ffn_norm, &norm_hidden_, eps, 0.0f, stream );
	else {
		if ( ring_ ) ring_->all_reduce( mlp_out_, stream );
		Elementwise::add_normed( resid_, mlp_out_, lw.post_attn_norm, eps, 0.0f, 1.0f, stream );
		RMSNorm::forward( resid_, lw.pre_ffn_norm, norm_hidden_, eps, 0.0f, stream );
	}
	pod::TensorView gate = pod::make_tensor_2d( gate_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, lw.inter );
	pod::TensorView up = pod::make_tensor_2d( up_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, lw.inter );
	pod::TensorView act = pod::make_tensor_2d( mlp_inter_.ptr<backend::half_t>(), pod::DType::FP16, m, lw.inter );
	Linear::forward( norm_hidden_, lw.gate_proj, gate, stream );
	Linear::forward( norm_hidden_, lw.up_proj, up, stream );
	SwiGLU::forward( gate, up, act, stream, 0.0f, true );   // gelu_tanh( gate ) * up
	Linear::forward( act, lw.down_proj, mlp_out_, stream );
	const bool ple = config_.ple_dim > 0;
	if ( fuse ) {
		ring_->all_reduce_normed_add( mlp_out_, resid_, lw.post_ffn_norm, ple ? 1.0f : lw.layer_scalar, ple ? nullptr : next_norm, !ple && next_norm != nullptr ? &norm_hidden_ : nullptr,
			eps, 0.0f, stream );
		if ( !ple ) return next_norm != nullptr;
	}
	else {
		if ( ring_ ) ring_->all_reduce( mlp_out_, stream );
		Elementwise::add_normed( resid_, mlp_out_, lw.post_ffn_norm, eps, 0.0f, ple ? 1.0f : lw.layer_scalar, stream );
	}
	if ( ple ) {
		// the gate reads the residual stream itself (FP16 for the projection), every rank the whole of it
		const int D = config_.ple_dim;
		Vision::f32_to_f16( resid_.ptr<const float>(), norm_hidden_.ptr<backend::half_t>(), static_cast<size_t>( m ) * config_.hidden_size, stream );
		pod::TensorView pg = pod::make_tensor_2d( gate_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, D );
		pod::TensorView pa = pod::make_tensor_2d( up_buf_.ptr<backend::half_t>(), pod::DType::FP16, m, D );
		const pod::TensorView slice = pod::make_tensor_2d( ple_.ptr<backend::half_t>() + ( static_cast<size_t>( layer ) * ple_m_ + ple_row0_ ) * D, pod::DType::FP16, m, D );
		Linear::forward( norm_hidden_, lw.ple_gate, pg, stream );
		SwiGLU::forward( pg, slice, pa, stream, 0.0f, true );   // gelu_tanh( gate ) * the slice
		Linear::forward( pa, lw.ple_proj, mlp_out_, stream );
		Elementwise::add_normed( resid_, mlp_out_, lw.post_ple_norm, eps, 0.0f, lw.layer_scalar, stream );
	}
	return false;
}

void Gemma4Pipeline::forward_body( int m, backend::stream_t stream ) {
	bool normed = false;
	if ( config_.ple_dim > 0 ) ple_mix( m, stream );
	for ( int L = 0; L < config_.num_layers; ++L ) {
		if ( tail_ && L == config_.first_shared() ) {
			if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
			// the last kTailRows rows on alone, as rows 0 .. kTailRows - 1 (their per-layer inputs stay where they are)
			const int H = config_.hidden_size, tn = kTailRows;
			backend::memcpy_async( resid_.ptr<float>(), resid_.ptr<float>() + static_cast<size_t>( m - tn ) * H, static_cast<size_t>( tn ) * H * sizeof( float ),
				backend::Copy::D2D, stream );
			if ( normed ) backend::memcpy_async( norm_hidden_.ptr<backend::half_t>(), norm_hidden_.ptr<backend::half_t>() + static_cast<size_t>( m - tn ) * H,
				static_cast<size_t>( tn ) * H * sizeof( backend::half_t ), backend::Copy::D2D, stream );
			ple_row0_ = m - tn;
			rows_ = d_tail_rows_;
			m = tn;
			set_rows( m );
		}
		const Gemma4LayerWeights& lw = layers_[L];
		normed = layer_forward( lw, L, m, stream, normed, L + 1 < config_.num_layers ? &layers_[L + 1].input_norm : nullptr );
		// Drafter taps: the hidden after the tapped layers, into each row's slot ring (slot and position from d_rows_: graph-replayable)
		for ( int tap = 0; tap < Gemma4Config::kTaps && draft_.taps.data != nullptr; ++tap ) {
			if ( L != config_.tap_layers[tap] + tap_shift_ ) continue;
			engine::spec::dflash_publish_taps_slots_f32( draft_.taps.ptr<backend::half_t>(), resid_.ptr<const float>(), d_rows_, m, tap, draft_hooks_.tap_slice,
				draft_hooks_.taps_cap, draft_hooks_.taps_slot_stride, stream );
		}
	}

	// Final norm, the head shard, the final logit softcap
	if ( tp_size_ == 1 && rank_ != 0 ) return;
	if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
	const int H = config_.hidden_size;
	if ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) {
		pod::TensorView r1 = pod::make_tensor_2d( resid_.ptr<float>() + static_cast<size_t>( m - 1 ) * H, pod::DType::FP32, 1, H );
		pod::TensorView n1 = pod::make_tensor_2d( norm_hidden_.ptr<backend::half_t>(), pod::DType::FP16, 1, H );
		pod::TensorView l1 = pod::make_tensor_2d( logits_.ptr<backend::half_t>(), pod::DType::FP16, 1, lm_shard_cols_ );
		engine::modules::RMSNorm::forward( r1, final_norm_, n1, config_.rms_norm_eps, 0.0f, stream );
		engine::modules::Linear::forward( n1, lm_head_, l1, stream );
		if ( config_.final_softcap > 0.0f ) engine::modules::Elementwise::softcap( l1, config_.final_softcap, stream );
		return;
	}
	ENGINE_ASSERT( m <= pod::MAX_BATCH, "Gemma4Pipeline: every row's logits are kept for up to MAX_BATCH rows" );
	engine::modules::RMSNorm::forward( resid_, final_norm_, norm_hidden_, config_.rms_norm_eps, 0.0f, stream );
	engine::modules::Linear::forward( norm_hidden_, lm_head_, logits_, stream );
	if ( config_.final_softcap > 0.0f ) engine::modules::Elementwise::softcap( logits_, config_.final_softcap, stream );
}

void Gemma4Pipeline::destroy() {
	if ( arena_ != nullptr ) {
		backend::set_device( rank_ );
		graphs_.reset();
	}
	if ( h_argmax_idx_ != nullptr ) {
		engine::core::host_free_pinned( h_argmax_idx_ );
		h_argmax_idx_ = nullptr;
	}
	if ( ple_stage_ != nullptr ) {
		engine::core::host_free_pinned( ple_stage_ );
		ple_stage_ = nullptr;
	}
	ple_repo_.reset();
}

// -- Sequence snapshots (prefix cache). Host layout: the drafter tap rows (with taps), the sliding pool's layers, then the full pool's; per
// layer K data, K scales, V data, V scales of positions [from, tokens). The sliding pool keeps only the last window of the range: a
// restore needs the window before its end, which this segment and its parents' last windows cover.
namespace {

constexpr size_t kSeqAlign = 256;

// First position of [from, tokens) a pool's segment keeps
int kept_from( const engine::runtime::KVCachePool& pool, int window, int from, int tokens ) {
	return pool.ring > 0 ? std::max( from, tokens - window ) : from;
}

size_t pool_rows_bytes( const engine::runtime::KVCachePool& pool, int rows ) {
	const size_t row_elems = static_cast<size_t>( pool.num_heads ) * pool.head_dim;
	const pod::KVQ kq = pod::kv_k_quant( pool.format ), vq = pod::kv_v_quant( pool.format );
	const size_t e = row_elems * rows;
	return static_cast<size_t>( pool.num_layers ) * ( pod::kv_data_bytes( kq, e ) + pod::kv_scale_bytes( kq, e ) + pod::kv_data_bytes( vq, e ) + pod::kv_scale_bytes( vq, e ) );
}

}

size_t Gemma4Pipeline::tap_snapshot_bytes( int tokens ) const {
	if ( !snap_taps_ || draft_.taps.data == nullptr ) return 0;
	return static_cast<size_t>( std::min( tokens, draft_hooks_.taps_cap ) ) * draft_hooks_.tap_slice.cols * sizeof( backend::half_t );
}

size_t Gemma4Pipeline::seq_state_bytes( int from, int tokens, bool with_taps ) const {
	if ( !swa_pool_.k_data || from < 0 || tokens < from || tokens > config_.max_seq ) return 0;
	const int swa_from = kept_from( swa_pool_, config_.swa_window, from, tokens );
	return ( with_taps ? tap_snapshot_bytes( tokens ) : 0 ) + pool_rows_bytes( swa_pool_, tokens - swa_from ) + pool_rows_bytes( full_pool_, tokens - from ) + kSeqAlign;
}

void Gemma4Pipeline::copy_kv_rows( const engine::runtime::KVCachePool& pool, int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream ) {
	from = kept_from( pool, config_.swa_window, from, tokens );
	const size_t row_elems = static_cast<size_t>( pool.num_heads ) * pool.head_dim;
	if ( tokens <= from ) return;
	const pod::KVQ kqs[2] = { pod::kv_k_quant( pool.format ), pod::kv_v_quant( pool.format ) };
	char* data_base[2] = { static_cast<char*>( pool.k_data ), static_cast<char*>( pool.v_data ) };
	char* scale_base[2] = { static_cast<char*>( pool.k_scale ), static_cast<char*>( pool.v_scale ) };
	const size_t slot_data[2] = { pool.k_slot_bytes, pool.v_slot_bytes };
	const size_t slot_scale[2] = { pool.k_scale_slot_bytes, pool.v_scale_slot_bytes };
	const backend::Copy dir = to_host ? backend::Copy::D2H : backend::Copy::H2D;
	for ( int layer = 0; layer < pool.num_layers; ++layer ) {
		for ( int kv = 0; kv < 2; ++kv ) {
			// positions [from, tokens) in runs of contiguous cache rows (a ring wraps at most once over a kept range)
			for ( int p = from; p < tokens; ) {
				const int row = pool.ring > 0 ? p % pool.ring : p;
				const int n = pool.ring > 0 ? std::min( tokens - p, pool.ring - row ) : tokens - p;
				const size_t elem0 = ( static_cast<size_t>( layer ) * pool.max_seq_len + row ) * row_elems;
				const size_t db = pod::kv_data_bytes( kqs[kv], row_elems * n );
				char* dev = data_base[kv] + static_cast<size_t>( slot_id ) * slot_data[kv] + pod::kv_data_bytes( kqs[kv], elem0 );
				if ( to_host ) backend::memcpy_async( host, dev, db, dir, stream );
				else backend::memcpy_async( dev, host, db, dir, stream );
				host += db;
				if ( const size_t sb = pod::kv_scale_bytes( kqs[kv], row_elems * n ) ) {
					char* sdev = scale_base[kv] + static_cast<size_t>( slot_id ) * slot_scale[kv] + pod::kv_scale_bytes( kqs[kv], elem0 );
					if ( to_host ) backend::memcpy_async( host, sdev, sb, dir, stream );
					else backend::memcpy_async( sdev, host, sb, dir, stream );
					host += sb;
				}
				p += n;
			}
		}
	}
}

bool Gemma4Pipeline::save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* dst = static_cast<char*>( host_dst );
	if ( const size_t tap_bytes = with_taps ? tap_snapshot_bytes( tokens ) : 0 ) {
		backend::memcpy_async( dst, draft_.taps.ptr<backend::half_t>() + static_cast<size_t>( slot_id ) * draft_hooks_.taps_slot_stride, tap_bytes, backend::Copy::D2H, stream );
		dst += tap_bytes;
	}
	copy_kv_rows( swa_pool_, slot_id, from, tokens, dst, true, stream );
	copy_kv_rows( full_pool_, slot_id, from, tokens, dst + pool_rows_bytes( swa_pool_, tokens - kept_from( swa_pool_, config_.swa_window, from, tokens ) ), true, stream );
	return true;
}

bool Gemma4Pipeline::load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* src = const_cast<char*>( static_cast<const char*>( host_src ) );
	if ( const size_t tap_bytes = with_taps ? tap_snapshot_bytes( tokens ) : 0 ) {
		if ( !kv_only ) backend::memcpy_async( draft_.taps.ptr<backend::half_t>() + static_cast<size_t>( slot_id ) * draft_hooks_.taps_slot_stride, src, tap_bytes, backend::Copy::H2D, stream );
		src += tap_bytes;
	}
	copy_kv_rows( swa_pool_, slot_id, from, tokens, src, false, stream );
	copy_kv_rows( full_pool_, slot_id, from, tokens, src + pool_rows_bytes( swa_pool_, tokens - kept_from( swa_pool_, config_.swa_window, from, tokens ) ), false, stream );
	return true;
}

bool Gemma4Pipeline::seq_rows_span( int from, int tokens, bool with_taps, size_t* offset, size_t* bytes ) const {
	const size_t all = seq_state_bytes( from, tokens, with_taps );
	if ( all == 0 ) return false;
	*offset = with_taps ? tap_snapshot_bytes( tokens ) : 0;
	*bytes = all - *offset;
	return true;
}

bool Gemma4Pipeline::load_seq_rows( int slot_id, int from, int tokens, const void* rows, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, false ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* src = const_cast<char*>( static_cast<const char*>( rows ) );
	copy_kv_rows( swa_pool_, slot_id, from, tokens, src, false, stream );
	copy_kv_rows( full_pool_, slot_id, from, tokens, src + pool_rows_bytes( swa_pool_, tokens - kept_from( swa_pool_, config_.swa_window, from, tokens ) ), false, stream );
	return true;
}

} // namespace engine::models
