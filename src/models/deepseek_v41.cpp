#include "engine/models/deepseek_v41.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/norm.h"
#include "engine/modules/mhc.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/embedding.h"
#include "engine/modules/dsv4.h"
#include "engine/spec/dflash.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <chrono>
#include <sys/mman.h>
#include <unistd.h>

namespace engine::models {

using engine::modules::DSV4;

namespace {

pod::TensorView g_embed_table;   // host BF16 [vocab, hidden], shared by the ranks

float bf16_to_float( uint16_t b ) {
	const uint32_t u = static_cast<uint32_t>( b ) << 16;
	float f;
	std::memcpy( &f, &u, 4 );
	return f;
}

float e4m3_to_float( uint8_t b ) {
	const int s = b >> 7, e = ( b >> 3 ) & 15, m = b & 7;
	float v;
	if ( e == 15 && m == 7 ) v = NAN;
	else if ( e == 0 ) v = std::ldexp( static_cast<float>( m ), -9 );
	else v = std::ldexp( 1.0f + m / 8.0f, e - 7 );
	return s ? -v : v;
}

// (cos, sin) of pos * freq_i (FP32 product, as the reference's outer product), DeepSeek's YaRN ramp when original_len > 0
std::vector<float> rope_table( int max_pos, float theta, int original_len, float factor, float beta_fast, float beta_slow ) {
	const int dim = DSV4::ROPE, half = dim / 2;
	std::vector<float> freqs( half );
	for ( int i = 0; i < half; ++i ) freqs[i] = 1.0f / std::pow( theta, static_cast<float>( 2 * i ) / dim );
	if ( original_len > 0 ) {
		auto corr = [&]( float rot ) { return dim * std::log( original_len / ( rot * 2.0 * M_PI ) ) / ( 2.0 * std::log( theta ) ); };
		const double low = std::max( std::floor( corr( beta_fast ) ), 0.0 ), high = std::min( std::ceil( corr( beta_slow ) ), static_cast<double>( dim - 1 ) );
		for ( int i = 0; i < half; ++i ) {
			const double ramp = std::clamp( ( i - low ) / std::max( high - low, 1e-3 ), 0.0, 1.0 );
			const float smooth = static_cast<float>( 1.0 - ramp );
			freqs[i] = freqs[i] / factor * ( 1.0f - smooth ) + freqs[i] * smooth;
		}
	}
	std::vector<float> t( static_cast<size_t>( max_pos ) * half * 2 );
	for ( int p = 0; p < max_pos; ++p ) {
		for ( int i = 0; i < half; ++i ) {
			const float a = static_cast<float>( p ) * freqs[i];
			t[( static_cast<size_t>( p ) * half + i ) * 2] = static_cast<float>( std::cos( static_cast<double>( a ) ) );
			t[( static_cast<size_t>( p ) * half + i ) * 2 + 1] = static_cast<float>( std::sin( static_cast<double>( a ) ) );
		}
	}
	return t;
}

template <typename T> T* dev_alloc( pod::MemoryArena& arena, size_t n, bool zero = true ) {
	T* p = static_cast<T*>( engine::core::arena_alloc( arena, n * sizeof( T ), 256 ) );
	ENGINE_ASSERT( p != nullptr, "DeepseekV41Pipeline: arena exhausted" );
	if ( zero ) backend::memset_sync( p, 0, n * sizeof( T ) );
	return p;
}

pod::TensorView dev_tensor( pod::MemoryArena& arena, pod::DType dt, int rows, int cols ) {
	int shape[2] = { rows, cols };
	pod::TensorView t = engine::core::arena_alloc_tensor( arena, dt, shape, 2 );
	ENGINE_ASSERT( t.data != nullptr, "DeepseekV41Pipeline: arena exhausted" );
	return t;
}

pod::TensorView view( const pod::TensorView& base, size_t elem_offset, int rows, int cols ) {
	const size_t es = base.dtype == pod::DType::FP32 ? 4 : 2;
	return pod::make_tensor_2d( static_cast<char*>( base.data ) + elem_offset * es, base.dtype, rows, cols );
}

void scale_device_half( backend::half_t* p, size_t n, float f, backend::stream_t stream ) {
	std::vector<backend::half_t> h( n );
	backend::stream_sync( stream );
	backend::memcpy_sync( h.data(), p, n * sizeof( backend::half_t ), backend::Copy::D2H );
	for ( auto& v : h ) v = backend::from_float_half( backend::to_float( v ) * f );
	backend::memcpy_sync( p, h.data(), n * sizeof( backend::half_t ), backend::Copy::H2D );
}

// A [N, K] tensor (BF16 / F32) as FP16 on the device, or FP8 E4M3 with [N/32, K/32] E8M0 block scales dequantized to FP16
pod::LinearWeight load_dense( engine::loader::SafeTensorsRepo& repo, const std::string& base, pod::MemoryArena& arena, backend::stream_t stream ) {
	int shard = -1;
	const auto* meta = repo.find( ( base + ".weight" ).c_str(), &shard );
	if ( meta == nullptr || meta->ndim != 2 ) return {};
	const int n = meta->shape[0], k = meta->shape[1];
	const uint8_t* src = static_cast<const uint8_t*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard ), meta ) );
	std::vector<backend::half_t> h( static_cast<size_t>( n ) * k );
	const auto* smeta = repo.find( ( base + ".scale" ).c_str(), &shard );
	if ( smeta != nullptr ) {
		const uint8_t* sc = static_cast<const uint8_t*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard ), smeta ) );
		const int sk = smeta->shape[1];
		ENGINE_ASSERT( meta->size_bytes == static_cast<size_t>( n ) * k && smeta->shape[0] == ( n + 31 ) / 32 && sk == ( k + 31 ) / 32, "load_dense: FP8 block layout" );
		float lut[256];
		for ( int i = 0; i < 256; ++i ) lut[i] = e4m3_to_float( static_cast<uint8_t>( i ) );
		for ( int r = 0; r < n; ++r ) {
			for ( int c = 0; c < k; ++c ) {
				const float s = std::ldexp( 1.0f, static_cast<int>( sc[static_cast<size_t>( r / 32 ) * sk + c / 32] ) - 127 );
				h[static_cast<size_t>( r ) * k + c] = backend::from_float_half( lut[src[static_cast<size_t>( r ) * k + c]] * s );
			}
		}
	}
	else if ( meta->dtype == pod::DType::BF16 ) {
		const uint16_t* b = reinterpret_cast<const uint16_t*>( src );
		for ( size_t i = 0; i < h.size(); ++i ) h[i] = backend::from_float_half( bf16_to_float( b[i] ) );
	}
	else {
		ENGINE_ASSERT( meta->dtype == pod::DType::FP32, "load_dense: unsupported dtype" );
		const float* f = reinterpret_cast<const float*>( src );
		for ( size_t i = 0; i < h.size(); ++i ) h[i] = backend::from_float_half( f[i] );
	}
	backend::half_t* d = dev_alloc<backend::half_t>( arena, h.size(), false );
	backend::memcpy_sync( d, h.data(), h.size() * sizeof( backend::half_t ), backend::Copy::H2D );
	(void) stream;
	return pod::LinearWeight::make_fp16( pod::make_tensor_2d( d, pod::DType::FP16, k, n ) );   // Linear takes k from shape[0], n from shape[1]
}

// A 1-D tensor (BF16 / F32) as FP16 or FP32 on the device, optionally elements [begin, begin + count)
pod::TensorView load_vec( engine::loader::SafeTensorsRepo& repo, const std::string& name, pod::MemoryArena& arena, pod::DType to, int begin = 0, int count = -1 ) {
	int shard = -1;
	const auto* meta = repo.find( name.c_str(), &shard );
	if ( meta == nullptr ) return {};
	size_t total = 1;
	for ( int i = 0; i < meta->ndim; ++i ) total *= static_cast<size_t>( meta->shape[i] );
	if ( count < 0 ) count = static_cast<int>( total ) - begin;
	const void* src = engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard ), meta );
	std::vector<float> f( count );
	for ( int i = 0; i < count; ++i ) {
		const size_t j = static_cast<size_t>( begin ) + i;
		f[i] = meta->dtype == pod::DType::BF16 ? bf16_to_float( static_cast<const uint16_t*>( src )[j] )
			: meta->dtype == pod::DType::FP16 ? backend::to_float( static_cast<const backend::half_t*>( src )[j] )
			: static_cast<const float*>( src )[j];
	}
	if ( to == pod::DType::FP32 ) {
		pod::TensorView t = dev_tensor( arena, pod::DType::FP32, 1, count );
		backend::memcpy_sync( t.data, f.data(), f.size() * sizeof( float ), backend::Copy::H2D );
		return t;
	}
	std::vector<backend::half_t> h( count );
	for ( int i = 0; i < count; ++i ) h[i] = backend::from_float_half( f[i] );
	pod::TensorView t = dev_tensor( arena, pod::DType::FP16, 1, count );
	backend::memcpy_sync( t.data, h.data(), h.size() * sizeof( backend::half_t ), backend::Copy::H2D );
	return t;
}

} // namespace

DeepseekV41Pipeline::DeepseekV41Pipeline() {
	config_.compress_ratios.assign( 40, 1 );
	config_.compress_ratios[0] = config_.compress_ratios[1] = 0;
	for ( int l = 2; l < 20; ++l ) config_.compress_ratios[l] = 2;
	config_.kv_sources = { 2, 8, 14, 20 };
	config_.index_sources = { 2, 8, 14, 20, 24, 28, 32, 36 };
	config_.engram_layers = { 1, 14 };
}

DeepseekV41Pipeline::~DeepseekV41Pipeline() {
	// route counts of this rank's experts since the start (checkpoint ids) to expert_heat_file.r<rank>, for the next run's placement
	const std::string& hf = runtime::options().expert_heat_file;
	if ( hf.empty() || !weights_loaded_ ) return;
	std::vector<uint32_t> tmp;
	int l = 0, e = 0;
	expert_heat( tmp, l, e, true );
	if ( FILE* f = std::fopen( ( hf + ".r" + std::to_string( rank_ ) ).c_str(), "wb" ) ) {
		const int32_t dims[2] = { config_.num_layers, config_.num_experts };
		std::fwrite( dims, sizeof( dims ), 1, f );
		std::fwrite( heat_total_.data(), sizeof( uint64_t ), heat_total_.size(), f );
		std::fclose( f );
	}
}

bool DeepseekV41Pipeline::expert_heat( std::vector<uint32_t>& out, int& layers, int& experts, bool reset ) {
	if ( !weights_loaded_ ) return false;
	backend::set_device( rank_ );
	backend::device_sync();
	layers = config_.num_layers;
	experts = config_.num_experts / config_.tp_size;
	out.assign( static_cast<size_t>( layers ) * experts, 0 );
	for ( int l = 0; l < layers; ++l ) {
		size_t at = static_cast<size_t>( l ) * experts;
		for ( auto& g : layers_[l].groups ) {
			backend::memcpy_sync( out.data() + at, g.mw->heat, static_cast<size_t>( g.count ) * sizeof( uint32_t ), backend::Copy::D2H );
			if ( reset ) backend::memset_sync( g.mw->heat, 0, static_cast<size_t>( g.count ) * sizeof( uint32_t ) );
			at += g.count;
		}
		if ( reset ) {
			for ( int i = 0; i < experts && i < static_cast<int>( mine_[l].size() ); ++i ) {
				const size_t id = static_cast<size_t>( l ) * config_.num_experts + mine_[l][i];
				heat_total_[id] += out[static_cast<size_t>( l ) * experts + i];
				pending_[id] += out[static_cast<size_t>( l ) * experts + i];
			}
		}
	}
	return true;
}

bool DeepseekV41Pipeline::read_config( const char* model_dir ) {
	std::ifstream f( std::string( model_dir ) + "/config.json" );
	if ( !f ) { LOG_ERROR( "[deepseek_v41] no config.json in %s\n", model_dir ); return false; }
	const nlohmann::json j = nlohmann::json::parse( f, nullptr, false );
	if ( j.is_discarded() ) return false;
	const nlohmann::json& t = j.contains( "text_config" ) ? j["text_config"] : j;
	bool ok = true;
	auto check = [&]( const char* key, auto expect ) {
		if ( !t.contains( key ) ) return;
		const auto v = t[key].get<decltype( expect )>();
		if ( v != expect ) { LOG_ERROR( "[deepseek_v41] config %s differs from the layout this engine implements\n", key ); ok = false; }
	};
	check( "num_hidden_layers", config_.num_layers );
	check( "hidden_size", config_.hidden_size );
	check( "num_attention_heads", config_.num_heads );
	check( "head_dim", DSV4::HD );
	check( "qk_rope_head_dim", DSV4::ROPE );
	check( "q_lora_rank", config_.q_lora_rank );
	check( "o_groups", config_.o_groups );
	check( "o_lora_rank", config_.o_lora_rank );
	check( "n_routed_experts", config_.num_experts );
	check( "num_experts_per_tok", config_.top_k );
	check( "moe_intermediate_size", config_.intermediate_size );
	check( "vocab_size", config_.vocab_size );
	check( "sliding_window", DSV4::WINDOW );
	check( "index_n_heads", DSV4::IDX_HEADS );
	check( "index_head_dim", DSV4::IDX_DIM );
	check( "index_topk", DSV4::TOPK );
	check( "hc_mult", 4 );
	check( "kv_source_layer_ids", config_.kv_sources );
	check( "index_source_layer_ids", config_.index_sources );
	check( "engram_layer_ids", config_.engram_layers );
	check( "candidate_source_layer_id", config_.candidate_source );
	check( "candidate_topk_blocks", config_.candidate_blocks );
	check( "candidate_block_size", config_.candidate_block );
	check( "engram_n_heads", config_.engram_heads );
	check( "engram_head_dim", config_.engram_head_dim );
	check( "engram_max_ngram_size", config_.engram_ngram );
	if ( t.contains( "compress_ratios" ) ) {
		std::vector<int> r = t["compress_ratios"].get<std::vector<int>>();
		r.resize( config_.num_layers );
		if ( r != config_.compress_ratios ) { LOG_ERROR( "[deepseek_v41] compress_ratios differ from the implemented layout\n" ); ok = false; }
	}
	if ( t.contains( "routed_scaling_factor" ) ) config_.route_scale = t["routed_scaling_factor"].get<float>();
	if ( t.contains( "swiglu_limit" ) ) config_.swiglu_limit = t["swiglu_limit"].get<float>();
	if ( t.contains( "rms_norm_eps" ) ) config_.norm_eps = t["rms_norm_eps"].get<float>();
	if ( t.contains( "eos_token_id" ) ) config_.eos_id = t["eos_token_id"].get<int>();
	else if ( j.contains( "eos_token_id" ) ) config_.eos_id = j["eos_token_id"].get<int>();
	if ( t.contains( "scoring_func" ) && t["scoring_func"] != "sqrtsoftplus" ) { LOG_ERROR( "[deepseek_v41] scoring_func must be sqrtsoftplus\n" ); ok = false; }
	return ok;
}

bool DeepseekV41Pipeline::init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) {
	rank_ = device_id;
	arena_ = &arena;
	ring_ = ring;
	config_.tp_size = ring != nullptr ? requested_tp_ : 1;
	ENGINE_ASSERT( config_.tp_size == 1 || config_.tp_size == 2 || config_.tp_size == 4, "DeepSeek-V4.1 runs on 1, 2 or 4 ranks" );
	if ( runtime::options().max_seq > 0 ) config_.max_seq = runtime::options().max_seq;
	if ( runtime::options().max_slots > 0 ) config_.max_slots = runtime::options().max_slots;
	ENGINE_ASSERT( config_.max_seq % 64 == 0, "DeepseekV41Pipeline: max_seq must be a multiple of 64" );
	lh_ = config_.num_heads / config_.tp_size;
	gl_ = config_.o_groups / config_.tp_size;
	backend::set_device( rank_ );
	const int R = pod::MAX_ROWS, H = config_.hidden_size, S = config_.max_slots, L = config_.num_layers;
	const size_t mark = arena.offset_bytes;

	// rotary tables
	{
		const auto plain = rope_table( config_.max_seq, config_.rope_theta, 0, 1.0f, 0.0f, 0.0f );
		const auto comp = rope_table( config_.max_seq, config_.compress_rope_theta, config_.rope_original_len, config_.rope_factor, config_.beta_fast, config_.beta_slow );
		rope_plain_ = dev_alloc<float>( arena, plain.size(), false );
		rope_comp_ = dev_alloc<float>( arena, comp.size(), false );
		backend::memcpy_sync( rope_plain_, plain.data(), plain.size() * sizeof( float ), backend::Copy::H2D );
		backend::memcpy_sync( rope_comp_, comp.data(), comp.size() * sizeof( float ), backend::Copy::H2D );
	}
	// caches
	win_slot_stride_ = static_cast<size_t>( L ) * DSV4::RING * DSV4::HD;
	win_ring_ = dev_alloc<backend::half_t>( arena, S * win_slot_stride_ );
	cring_slot_stride_ = static_cast<size_t>( DSV4::RING ) * DSV4::HD;
	int max_cap = 0;
	for ( int s : config_.kv_sources ) {
		const int ratio = config_.compress_ratios[s];
		const int cap = config_.max_seq / ratio;
		src_cap_.push_back( cap );
		max_cap = std::max( max_cap, cap );
		comp_slot_stride_.push_back( static_cast<size_t>( cap ) * DSV4::HD );
		comp_kv_.push_back( dev_alloc<backend::half_t>( arena, S * comp_slot_stride_.back() ) );
		idx_slot_stride_.push_back( static_cast<size_t>( cap ) * DSV4::IDX_DIM );
		idx_k_.push_back( dev_alloc<backend::half_t>( arena, S * idx_slot_stride_.back() ) );
		cring_kv_.push_back( ratio > 1 ? dev_alloc<float>( arena, S * cring_slot_stride_ ) : nullptr );
		cring_sc_.push_back( ratio > 1 ? dev_alloc<float>( arena, S * cring_slot_stride_ ) : nullptr );
	}
	if ( rank_ == 0 ) LOG_INFO( "[deepseek_v41] caches for %d slots of %d tokens: %.2f GiB per rank\n", S, config_.max_seq, ( arena.offset_bytes - mark ) / 1073741824.0 );

	// row buffers
	d_rows_ = dev_alloc<pod::BatchRows>( arena, 1 );
	d_token_ids_ = dev_alloc<int>( arena, R );
	embed_rows_ = dev_tensor( arena, pod::DType::FP16, R, H );
	streams_ = dev_tensor( arena, pod::DType::FP32, 4 * R, H );
	norm_x_ = dev_tensor( arena, pod::DType::FP16, R, H );
	collapsed_ = dev_tensor( arena, pod::DType::FP16, R, H );
	attn_out_ = dev_tensor( arena, pod::DType::FP16, R, H );
	moe_out_ = dev_tensor( arena, pod::DType::FP16, R, H );
	moe_tmp_ = dev_tensor( arena, pod::DType::FP16, R, H );
	sh_out_ = dev_tensor( arena, pod::DType::FP16, R, H );
	mhc_inv_rms_ = dev_tensor( arena, pod::DType::FP32, 1, R );
	mhc_mix_ = dev_tensor( arena, pod::DType::FP32, 1, 24 * R );
	mhc_post_ = dev_tensor( arena, pod::DType::FP32, 1, 4 * R );
	mhc_comb_ = dev_tensor( arena, pod::DType::FP32, 4 * R, 4 );
	pre_[0] = dev_alloc<float>( arena, 4 * R );
	pre_[1] = dev_alloc<float>( arena, 4 * R );
	qr_ = dev_tensor( arena, pod::DType::FP16, R, config_.q_lora_rank );
	qr_n_ = dev_tensor( arena, pod::DType::FP16, R, config_.q_lora_rank );
	q_ = dev_tensor( arena, pod::DType::FP16, R, lh_ * DSV4::HD );
	kv_ = dev_tensor( arena, pod::DType::FP16, R, DSV4::HD );
	kv_n_ = dev_tensor( arena, pod::DType::FP16, R, DSV4::HD );
	ckv_ = dev_tensor( arena, pod::DType::FP16, R, DSV4::HD );
	csc_ = dev_tensor( arena, pod::DType::FP16, R, DSV4::HD );
	latent_ = dev_tensor( arena, pod::DType::FP16, R, DSV4::HD );
	kraw_ = dev_tensor( arena, pod::DType::FP16, R, DSV4::IDX_DIM );
	qi_ = dev_tensor( arena, pod::DType::FP16, R, DSV4::IDX_HEADS * DSV4::IDX_DIM );
	wi_ = dev_tensor( arena, pod::DType::FP16, R, DSV4::IDX_HEADS );
	o_g_ = dev_tensor( arena, pod::DType::FP16, R, lh_ * DSV4::HD );
	woa_out_ = dev_tensor( arena, pod::DType::FP16, R, gl_ * config_.o_lora_rank );
	woa_cat_ = dev_tensor( arena, pod::DType::FP16, R, gl_ * config_.o_lora_rank );
	sh_g_ = dev_tensor( arena, pod::DType::FP16, R, config_.intermediate_size );
	sh_u_ = dev_tensor( arena, pod::DType::FP16, R, config_.intermediate_size );
	sh_i_ = dev_tensor( arena, pod::DType::FP16, R, config_.intermediate_size );
	idx_scores_ = dev_alloc<float>( arena, static_cast<size_t>( DSV4::SCORE_ROWS ) * max_cap, false );
	idx_cand_ = dev_alloc<uint8_t>( arena, static_cast<size_t>( R ) * ( max_cap / config_.candidate_block ) );
	idx_sel_ = dev_alloc<int>( arena, static_cast<size_t>( R ) * DSV4::TOPK );
	idx_nsel_ = dev_alloc<int>( arena, R );
	d_topk_idx_ = dev_alloc<int>( arena, static_cast<size_t>( R ) * config_.top_k );
	d_topk_w_ = dev_alloc<backend::half_t>( arena, static_cast<size_t>( R ) * config_.top_k );
	attend_ws_ = dev_alloc<float>( arena, DSV4::attend_ws_floats( lh_ ), false );
	route_scratch_ = dev_alloc<float>( arena, DSV4::route_scratch_floats( R, config_.num_experts ), false );
	const int ecols = config_.engram_heads * ( config_.engram_ngram - 1 ) * config_.engram_head_dim;
	e_rows_[0] = dev_tensor( arena, pod::DType::FP16, R, ecols );
	e_rows_[1] = dev_tensor( arena, pod::DType::FP16, R, ecols );
	e_kv_ = dev_tensor( arena, pod::DType::FP16, R, 5 * H );
	max_group_ = ( config_.num_experts + config_.tp_size - 1 ) / config_.tp_size;
	moe_scratch_ = engine::modules::MoE::alloc_scratch( arena, max_group_, config_.top_k, H, config_.intermediate_size, R, false );

	// logits: vocabulary-parallel head, shards padded to 256 columns
	lm_shard_real_cols_ = 128 * lm_shard_spans( config_.vocab_size, config_.tp_size, rank_ );
	lm_shard_offset_ = lm_shard_offset( config_.vocab_size, config_.tp_size, rank_ );
	{
		const int widest = 128 * lm_shard_spans( config_.vocab_size, config_.tp_size, 0 );
		lm_shard_cols_ = widest + ( 256 - widest % 256 ) % 256;
	}
	const int B = pod::MAX_BATCH;
	logit_rows_ = std::max( B, runtime::options().glm_logit_rows );
	logits_ = dev_tensor( arena, pod::DType::FP16, logit_rows_, lm_shard_cols_ );
	d_argmax_idx_ = dev_alloc<int>( arena, B );
	d_argmax_val_ = dev_alloc<float>( arena, B );
	h_argmax_idx_ = static_cast<int*>( engine::core::host_alloc_pinned( B * sizeof( int ) ) );
	{
		const int splits = ( lm_shard_cols_ + kTopkTile - 1 ) / kTopkTile;
		const size_t per = static_cast<size_t>( B ) * splits * 16;
		samp_part_v_ = dev_alloc<float>( arena, per );
		samp_part_i_ = dev_alloc<int>( arena, per );
		samp_counters_ = dev_alloc<int>( arena, B );
		d_samp_vals_ = dev_alloc<float>( arena, static_cast<size_t>( B ) * 16 );
		d_samp_idx_ = dev_alloc<int>( arena, static_cast<size_t>( B ) * 16 );
	}
	if ( rank_ == 0 ) full_logits_ = dev_tensor( arena, pod::DType::FP16, logit_rows_, config_.vocab_size );
#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_get_workspace( rank_, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( rank_ );
#endif
	metadata_.model_name = "deepseek_v41";
	metadata_.hidden_size = H;
	metadata_.vocab_size = config_.vocab_size;
	metadata_.num_layers = L;
	metadata_.num_heads = lh_;
	metadata_.num_kv_heads = 1;
	metadata_.head_dim = DSV4::HD;
	metadata_.max_seq_len = config_.max_seq;
	metadata_.tp_size = config_.tp_size;
	metadata_.max_slots = S;
	metadata_.max_rows = R;
	metadata_.eos_ids[0] = config_.eos_id;
	if ( rank_ == 0 ) LOG_INFO( "[deepseek_v41] rank buffers and caches: %.2f GiB\n", ( arena.offset_bytes - mark ) / 1073741824.0 );
	return true;
}

bool DeepseekV41Pipeline::load_engram( const char* model_dir ) {
	const std::string path = std::string( model_dir ) + "/engram_hash.bin";
	std::ifstream f( path, std::ios::binary );
	if ( !f ) {
		LOG_ERROR( "[deepseek_v41] %s missing: run tests/tools/dsv41/engram_hash_tables.py on the model directory\n", path.c_str() );
		return false;
	}
	char magic[4];
	int32_t hdr[6];
	f.read( magic, 4 );
	f.read( reinterpret_cast<char*>( hdr ), sizeof( hdr ) );
	if ( std::memcmp( magic, "DSEH", 4 ) != 0 || hdr[0] != 1 ) { LOG_ERROR( "[deepseek_v41] %s: unknown format\n", path.c_str() ); return false; }
	eg_.vocab = hdr[1]; eg_.layers = hdr[2]; eg_.ngram = hdr[3]; eg_.heads = hdr[4]; eg_.pad = hdr[5];
	ENGINE_ASSERT( eg_.layers == static_cast<int>( config_.engram_layers.size() ) && eg_.ngram == config_.engram_ngram && eg_.heads == config_.engram_heads, "engram_hash.bin layout" );
	std::vector<int32_t> ids( eg_.layers );
	f.read( reinterpret_cast<char*>( ids.data() ), ids.size() * 4 );
	eg_.token_map.resize( eg_.vocab );
	f.read( reinterpret_cast<char*>( eg_.token_map.data() ), eg_.token_map.size() * 4 );
	eg_.mult.resize( static_cast<size_t>( eg_.layers ) * eg_.ngram );
	eg_.primes.resize( static_cast<size_t>( eg_.layers ) * ( eg_.ngram - 1 ) * eg_.heads );
	eg_.offsets.resize( eg_.primes.size() );
	f.read( reinterpret_cast<char*>( eg_.mult.data() ), eg_.mult.size() * 8 );
	f.read( reinterpret_cast<char*>( eg_.primes.data() ), eg_.primes.size() * 8 );
	f.read( reinterpret_cast<char*>( eg_.offsets.data() ), eg_.offsets.size() * 8 );
	if ( !f ) { LOG_ERROR( "[deepseek_v41] %s truncated\n", path.c_str() ); return false; }
	for ( int i = 0; i < 256; ++i ) eg_.e4m3[i] = e4m3_to_float( static_cast<uint8_t>( i ) );
	for ( int e = 0; e < eg_.layers; ++e ) {
		const std::string pre = "layers." + std::to_string( config_.engram_layers[e] ) + ".engram.embed.";
		int sw = -1, ss = -1;
		const auto* mw = repo_->find( ( pre + "weight" ).c_str(), &sw );
		const auto* ms = repo_->find( ( pre + "scale" ).c_str(), &ss );
		if ( mw == nullptr || ms == nullptr ) {
			LOG_ERROR( "[deepseek_v41] Engram table %s not found: DeepSeek-V4.1-Flash shards 47 / 48 belong in the model directory\n", pre.c_str() );
			return false;
		}
		ENGINE_ASSERT( mw->shape[1] == eg_.dim && ms->shape[1] * eg_.block == eg_.dim, "Engram table shape" );
		eg_.table[e] = static_cast<const uint8_t*>( engine::loader::safetensors_get_data_cpu( *repo_->get_shard( sw ), mw ) );
		eg_.scale[e] = static_cast<const uint8_t*>( engine::loader::safetensors_get_data_cpu( *repo_->get_shard( ss ), ms ) );
		// random 4 KB reads: the default readahead of a fault (128 KB and up) would read mostly unrelated rows
		eg_.page = static_cast<size_t>( sysconf( _SC_PAGESIZE ) );
		for ( auto [ptr, bytes] : { std::pair<const uint8_t*, size_t>{ eg_.table[e], mw->size_bytes }, std::pair<const uint8_t*, size_t>{ eg_.scale[e], ms->size_bytes } } ) {
			const uintptr_t a = reinterpret_cast<uintptr_t>( ptr ) & ~( eg_.page - 1 );
			madvise( reinterpret_cast<void*>( a ), reinterpret_cast<uintptr_t>( ptr ) + bytes - a, MADV_RANDOM );
		}
	}
	eg_.hist.assign( config_.max_slots, std::vector<int>( config_.max_seq, eg_.pad ) );
	const size_t cols = static_cast<size_t>( eg_.heads ) * ( eg_.ngram - 1 ) * eg_.dim;
	eg_.host_rows = static_cast<backend::half_t*>( engine::core::host_alloc_pinned( 2 * static_cast<size_t>( pod::MAX_ROWS ) * cols * sizeof( backend::half_t ) ) );
	return true;
}

bool DeepseekV41Pipeline::load_weights( const char* model_dir, backend::stream_t stream ) {
	backend::set_device( rank_ );
	if ( !read_config( model_dir ) ) return false;
	repo_ = std::make_unique<engine::loader::SafeTensorsRepo>();
	if ( !repo_->open( model_dir ) ) { LOG_ERROR( "[deepseek_v41] cannot open %s\n", model_dir ); return false; }
	auto& repo = *repo_;
	auto& arena = *arena_;
	const int H = config_.hidden_size, L = config_.num_layers, tp = config_.tp_size;
	int missing = 0;
	auto need = [&]( bool ok, const std::string& what ) { if ( !ok ) { LOG_ERROR( "[RANK %d] MISSING: %s\n", rank_, what.c_str() ); ++missing; } };
	const size_t bytes0 = arena.offset_bytes;

	// embedding (host, rank 0), final norm, head shard
	if ( rank_ == 0 ) {
		int shard = -1;
		const auto* meta = repo.find( "embed.weight", &shard );
		need( meta != nullptr && meta->dtype == pod::DType::BF16, "embed.weight (BF16)" );
		if ( meta != nullptr ) {
			const size_t bytes = static_cast<size_t>( config_.vocab_size ) * H * sizeof( uint16_t );
			if ( g_embed_table.data == nullptr ) {
				void* host = backend::host_alloc_pinned( bytes, backend::Pin::MappedPortable );
				ENGINE_ASSERT( host != nullptr, "DeepseekV41Pipeline: pinned embedding allocation failed" );
				g_embed_table = pod::make_tensor_2d( host, pod::DType::BF16, config_.vocab_size, H );
			}
			std::memcpy( g_embed_table.data, engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard ), meta ), bytes );
		}
	}
	embed_table_ = g_embed_table;
	final_norm_ = load_vec( repo, "norm.weight", arena, pod::DType::FP16 );
	need( final_norm_.data != nullptr, "norm.weight" );
	lm_head_ = repo.load_linear_cols_padded( "head", arena, 4, lm_shard_offset_ / 16, lm_shard_real_cols_ / 16, lm_shard_cols_ - lm_shard_real_cols_, stream );
	need( lm_head_.is_valid(), "head" );

	// attention, norms, routers, hyper-connections, compressors, indexers, shared experts, Engram projections
	layers_.resize( L );
	auto src_index = [&]( int layer ) {   // the kv source a compressing layer reads: the last source at or before it
		int s = -1;
		for ( size_t i = 0; i < config_.kv_sources.size(); ++i ) if ( config_.kv_sources[i] <= layer ) s = static_cast<int>( i );
		return s;
	};
	for ( int l = 0; l < L; ++l ) {
		auto& lw = layers_[l];
		const std::string p = "layers." + std::to_string( l ) + ".";
		lw.ratio = config_.compress_ratios[l];
		lw.kv_src = lw.ratio > 0 ? src_index( l ) : -1;
		lw.is_kv_source = std::find( config_.kv_sources.begin(), config_.kv_sources.end(), l ) != config_.kv_sources.end();
		lw.is_index_source = std::find( config_.index_sources.begin(), config_.index_sources.end(), l ) != config_.index_sources.end();
		for ( size_t e = 0; e < config_.engram_layers.size(); ++e ) if ( config_.engram_layers[e] == l ) lw.engram = static_cast<int>( e );
		lw.attn_norm = load_vec( repo, p + "attn_norm.weight", arena, pod::DType::FP16 );
		lw.ffn_norm = load_vec( repo, p + "ffn_norm.weight", arena, pod::DType::FP16 );
		lw.q_norm = load_vec( repo, p + "attn.q_norm.weight", arena, pod::DType::FP16 );
		lw.kv_norm = load_vec( repo, p + "attn.kv_norm.weight", arena, pod::DType::FP16 );
		lw.sink = load_vec( repo, p + "attn.attn_sink", arena, pod::DType::FP32, rank_ * lh_, lh_ );
		need( lw.attn_norm.data && lw.ffn_norm.data && lw.q_norm.data && lw.kv_norm.data && lw.sink.data, p + "norms / sink" );
		lw.wq_a = repo.load_linear( ( p + "attn.wq_a" ).c_str(), arena, 4, stream );
		lw.wq_b = repo.load_linear_slice( ( p + "attn.wq_b" ).c_str(), arena, 4, 1, rank_, tp, stream );
		lw.wkv = repo.load_linear( ( p + "attn.wkv" ).c_str(), arena, 4, stream );
		lw.wo_b = repo.load_linear_slice( ( p + "attn.wo_b" ).c_str(), arena, 4, 0, rank_, tp, stream );
		need( lw.wq_a.is_valid() && lw.wq_b.is_valid() && lw.wkv.is_valid() && lw.wo_b.is_valid(), p + "attention linears" );
		if ( lw.wo_b.is_valid() ) scale_device_half( lw.wo_b.w2.ptr<backend::half_t>(), static_cast<size_t>( H ), 1.0f / kOutScale, stream );
		for ( int g = 0; g < gl_; ++g ) {
			lw.wo_a.push_back( repo.load_linear( ( p + "attn.wo_a.slice." + std::to_string( rank_ * gl_ + g ) ).c_str(), arena, 5, stream ) );
			need( lw.wo_a.back().is_valid(), p + "attn.wo_a slice" );
		}
		if ( lw.is_kv_source ) {
			lw.c_wkv = load_dense( repo, p + "attn.compressor.wkv", arena, stream );
			if ( lw.ratio > 1 ) lw.c_wgate = load_dense( repo, p + "attn.compressor.wgate", arena, stream );
			lw.c_norm = load_vec( repo, p + "attn.compressor.norm.weight", arena, pod::DType::FP16 );
			lw.idx_wk = load_dense( repo, p + "attn.indexer.wk", arena, stream );
			lw.idx_knorm = load_vec( repo, p + "attn.indexer.k_norm.weight", arena, pod::DType::FP16 );
			need( lw.c_wkv.is_valid() && ( lw.ratio == 1 || lw.c_wgate.is_valid() ) && lw.c_norm.data && lw.idx_wk.is_valid() && lw.idx_knorm.data, p + "compressor / index keys" );
		}
		if ( lw.is_index_source ) {
			lw.idx_wq_b = load_dense( repo, p + "attn.indexer.wq_b", arena, stream );
			lw.idx_wproj = load_dense( repo, p + "attn.indexer.weights_proj", arena, stream );
			need( lw.idx_wq_b.is_valid() && lw.idx_wproj.is_valid(), p + "indexer" );
		}
		// router [E, H] BF16 -> [H, E] FP16
		{
			int shard = -1;
			const auto* meta = repo.find( ( p + "ffn.gate.weight" ).c_str(), &shard );
			need( meta != nullptr && meta->shape[0] == config_.num_experts && meta->shape[1] == H, p + "ffn.gate.weight" );
			if ( meta != nullptr ) {
				const uint16_t* src = static_cast<const uint16_t*>( engine::loader::safetensors_get_data_cpu( *repo.get_shard( shard ), meta ) );
				std::vector<backend::half_t> t( static_cast<size_t>( H ) * config_.num_experts );
				for ( int e = 0; e < config_.num_experts; ++e )
					for ( int d = 0; d < H; ++d ) t[static_cast<size_t>( d ) * config_.num_experts + e] = backend::from_float_half( bf16_to_float( src[static_cast<size_t>( e ) * H + d] ) );
				lw.router_w_t = dev_tensor( arena, pod::DType::FP16, H, config_.num_experts );
				backend::memcpy_sync( lw.router_w_t.data, t.data(), t.size() * sizeof( backend::half_t ), backend::Copy::H2D );
			}
			lw.router_bias = load_vec( repo, p + "ffn.gate.bias", arena, pod::DType::FP32 );
			need( lw.router_bias.data != nullptr, p + "ffn.gate.bias" );
		}
		if ( rank_ == 0 ) {
			lw.sh_gate = repo.load_linear( ( p + "ffn.shared_experts.w1" ).c_str(), arena, 4, stream );
			lw.sh_up = repo.load_linear( ( p + "ffn.shared_experts.w3" ).c_str(), arena, 4, stream );
			lw.sh_down = repo.load_linear( ( p + "ffn.shared_experts.w2" ).c_str(), arena, 4, stream );
			need( lw.sh_gate.is_valid() && lw.sh_up.is_valid() && lw.sh_down.is_valid(), p + "shared expert" );
			if ( lw.sh_down.is_valid() ) scale_device_half( lw.sh_down.w2.ptr<backend::half_t>(), static_cast<size_t>( H ), 1.0f / kOutScale, stream );
		}
		if ( lw.engram >= 0 ) {
			lw.e_wkv = load_dense( repo, p + "engram.wkv", arena, stream );
			const pod::TensorView q = load_vec( repo, p + "engram.q_weight", arena, pod::DType::FP32 );
			const pod::TensorView k = load_vec( repo, p + "engram.k_weight", arena, pod::DType::FP32 );
			need( lw.e_wkv.is_valid() && q.data && k.data, p + "engram projection (shards 47 / 48)" );
			if ( q.data && k.data ) {
				std::vector<float> a( 4 * static_cast<size_t>( H ) ), b( a.size() );
				backend::memcpy_sync( a.data(), q.data, a.size() * 4, backend::Copy::D2H );
				backend::memcpy_sync( b.data(), k.data, b.size() * 4, backend::Copy::D2H );
				for ( size_t i = 0; i < a.size(); ++i ) a[i] *= b[i];   // only ever used as a product
				lw.e_w = q;
				backend::memcpy_sync( lw.e_w.data, a.data(), a.size() * 4, backend::Copy::H2D );
			}
		}
		lw.hc_attn_fn = load_vec( repo, p + "hc_attn_fn", arena, pod::DType::FP32 );
		lw.hc_attn_base = load_vec( repo, p + "hc_attn_base", arena, pod::DType::FP32 );
		lw.hc_attn_scale = load_vec( repo, p + "hc_attn_scale", arena, pod::DType::FP32 );
		lw.hc_ffn_fn = load_vec( repo, p + "hc_ffn_fn", arena, pod::DType::FP32 );
		lw.hc_ffn_base = load_vec( repo, p + "hc_ffn_base", arena, pod::DType::FP32 );
		lw.hc_ffn_scale = load_vec( repo, p + "hc_ffn_scale", arena, pod::DType::FP32 );
		need( lw.hc_attn_fn.data && lw.hc_attn_base.data && lw.hc_attn_scale.data && lw.hc_ffn_fn.data && lw.hc_ffn_base.data && lw.hc_ffn_scale.data, p + "hc" );
	}
	if ( !load_engram( model_dir ) ) return false;
	const size_t dense_bytes = arena.offset_bytes - bytes0;

	// routed experts. A rank's experts get consecutive engine ids grouped by bit width, so each width is one MoE call over an id range; perm
	// maps the router's ids to engine ids. Without route counts the experts are sorted by width and dealt round-robin; with the counts of
	// expert_heat_file (earlier runs) they are dealt hottest first to the rank with the least heat so far, which balances the work of a step
	const int E = config_.num_experts, per_rank = ( E + tp - 1 ) / tp;
	ENGINE_ASSERT( E % tp == 0, "DeepseekV41Pipeline: experts must split evenly over the ranks" );
	std::vector<std::vector<int>> bits( L, std::vector<int>( E ) );
	std::vector<uint64_t> heat;   // [layer][checkpoint expert] route counts, empty without a file
	if ( const std::string& hf = runtime::options().expert_heat_file; !hf.empty() ) {
		std::vector<uint64_t> sum( static_cast<size_t>( L ) * E, 0 );
		int found = 0;
		for ( int r = 0; r < tp; ++r ) {
			std::ifstream f( hf + ".r" + std::to_string( r ), std::ios::binary );
			int32_t dims[2] = {};
			if ( !f.read( reinterpret_cast<char*>( dims ), sizeof( dims ) ) || dims[0] != L || dims[1] != E ) continue;
			std::vector<uint64_t> part( sum.size() );
			if ( !f.read( reinterpret_cast<char*>( part.data() ), part.size() * sizeof( uint64_t ) ) ) continue;
			for ( size_t i = 0; i < sum.size(); ++i ) sum[i] += part[i];
			++found;
		}
		if ( found == tp ) heat = std::move( sum );
		if ( rank_ == 0 ) LOG_INFO( "[deepseek_v41] expert placement from %s: %s\n", hf.c_str(), heat.empty() ? "no complete set of route counts, default placement" : "hottest experts resident, heat balanced over the ranks" );
	}
	heat_total_.assign( static_cast<size_t>( L ) * E, 0 );
	pending_.assign( static_cast<size_t>( L ) * E, 0 );
	mine_.assign( L, {} );
	size_t code_bytes = 0;
	std::vector<std::vector<size_t>> expert_bytes( L, std::vector<size_t>( E, 0 ) );
	for ( int l = 0; l < L; ++l ) {
		const std::string p = "layers." + std::to_string( l ) + ".ffn.experts.";
		for ( int e = 0; e < E; ++e ) {
			const auto* meta = repo.find( ( p + std::to_string( e ) + ".w1.trellis" ).c_str() );
			need( meta != nullptr && meta->ndim == 3, p + std::to_string( e ) );
			bits[l][e] = meta ? meta->shape[2] / 16 : 0;
		}
		std::vector<std::vector<int>> per( tp );
		if ( heat.empty() ) {
			std::vector<int> order( E );
			std::iota( order.begin(), order.end(), 0 );
			std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return bits[l][a] < bits[l][b]; } );
			for ( int i = 0; i < E; ++i ) per[i % tp].push_back( order[i] );
		}
		else {
			const uint64_t* h = heat.data() + static_cast<size_t>( l ) * E;
			std::vector<int> order( E );
			std::iota( order.begin(), order.end(), 0 );
			std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return h[a] > h[b]; } );
			std::vector<uint64_t> load( tp, 0 );
			for ( int e : order ) {
				int best = -1;
				for ( int r = 0; r < tp; ++r ) if ( static_cast<int>( per[r].size() ) < per_rank && ( best < 0 || load[r] < load[best] ) ) best = r;
				per[best].push_back( e );
				load[best] += h[e];
			}
		}
		std::vector<int> perm( E );
		for ( int r = 0; r < tp; ++r ) {
			std::stable_sort( per[r].begin(), per[r].end(), [&]( int a, int b ) { return bits[l][a] < bits[l][b]; } );
			for ( int j = 0; j < static_cast<int>( per[r].size() ); ++j ) perm[per[r][j]] = r * per_rank + j;
		}
		mine_[l] = per[rank_];
		if ( !heat.empty() ) for ( int e : mine_[l] ) heat_total_[static_cast<size_t>( l ) * E + e] = heat[static_cast<size_t>( l ) * E + e];   // the file keeps the history
		layers_[l].perm = dev_alloc<int>( arena, E, false );
		backend::memcpy_sync( layers_[l].perm, perm.data(), E * sizeof( int ), backend::Copy::H2D );
		for ( int e : mine_[l] ) {
			for ( const char* w : { ".w1", ".w2", ".w3" } ) expert_bytes[l][e] += repo.trellis_bytes( ( p + std::to_string( e ) + w ).c_str() );
			code_bytes += expert_bytes[l][e];
		}
	}
	// placement: as many expert codes on the device as the arena holds (the same share of every layer's bytes; the hottest experts first when the
	// route counts are known), the rest in pinned host memory
	const size_t margin = static_cast<size_t>( std::max( runtime::options().vram_reserve_mib, 1024 ) ) * 1024 * 1024;
	const size_t meta_bytes = static_cast<size_t>( L ) * per_rank * 3 * ( 2 * H + 2 * config_.intermediate_size ) * 2 + ( 64ull << 20 );   // suh / svh and tables
	const size_t free_now = arena.total_bytes - arena.offset_bytes;
	const size_t budget = free_now > margin + meta_bytes ? free_now - margin - meta_bytes : 0;
	const double frac = std::min( 1.0, static_cast<double>( budget ) / std::max<size_t>( code_bytes, 1 ) );
	if ( frac < 1.0 ) {
		const size_t host_bytes = static_cast<size_t>( ( 1.0 - frac ) * code_bytes * 1.02 ) + ( 256ull << 20 );
		ENGINE_ASSERT( engine::core::arena_create_pinned( host_arena_, host_bytes ), "DeepseekV41Pipeline: pinned host arena for the cold experts failed" );
	}
	size_t on_host = 0;
	for ( int l = 0; l < L; ++l ) {
		auto& lw = layers_[l];
		const std::string p = "layers." + std::to_string( l ) + ".ffn.experts.";
		// the resident set: experts in order of heat (else engine order) while the layer's device share of bytes lasts
		std::vector<int> order = mine_[l];
		if ( !heat.empty() ) {
			const uint64_t* h = heat.data() + static_cast<size_t>( l ) * E;
			std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return h[a] > h[b]; } );
		}
		size_t layer_bytes = 0;
		for ( int e : order ) layer_bytes += expert_bytes[l][e];
		std::vector<char> resident( E, 0 );
		size_t used = 0;
		for ( int e : order ) {
			if ( static_cast<double>( used + expert_bytes[l][e] ) > frac * static_cast<double>( layer_bytes ) + 1.0 ) continue;
			used += expert_bytes[l][e];
			resident[e] = 1;
		}
		int j = 0;
		while ( j < static_cast<int>( mine_[l].size() ) ) {
			int k = j;
			while ( k < static_cast<int>( mine_[l].size() ) && bits[l][mine_[l][k]] == bits[l][mine_[l][j]] ) ++k;
			DeepseekV41ExpertGroup g;
			g.bits = bits[l][mine_[l][j]];
			g.first = rank_ * per_rank + j;
			g.count = k - j;
			g.mw = std::make_unique<pod::MoELayerWeights>();
			auto& mw = *g.mw;
			mw.type = pod::QuantType::TRELLIS;
			mw.num_local_experts = g.count;
			mw.top_k = config_.top_k;
			for ( int i = 0; i < g.count; ++i ) {
				const int e = mine_[l][j + i];
				const std::string b = p + std::to_string( e );
				pod::MemoryArena* codes = resident[e] ? nullptr : &host_arena_;
				mw.gate[i] = repo.load_linear( ( b + ".w1" ).c_str(), arena, g.bits, stream, codes );
				mw.up[i] = repo.load_linear( ( b + ".w3" ).c_str(), arena, g.bits, stream, codes );
				mw.down[i] = repo.load_linear( ( b + ".w2" ).c_str(), arena, g.bits, stream, codes );
				need( mw.gate[i].is_valid() && mw.up[i].is_valid() && mw.down[i].is_valid(), b );
				ENGINE_ASSERT( mw.up[i].bits == g.bits && mw.down[i].bits == g.bits, "DeepseekV41Pipeline: an expert's three projections differ in bit width" );
				scale_device_half( mw.down[i].w2.ptr<backend::half_t>(), static_cast<size_t>( H ), 1.0f / kOutScale, stream );
				if ( codes ) on_host += expert_bytes[l][e];
			}
			engine::modules::MoE::init_device_ptrs( mw, arena, stream );
			mw.heat = dev_alloc<uint32_t>( arena, g.count );
			g.resident.assign( g.count, 1 );
			g.score.assign( g.count, 0 );
			for ( int i = 0; i < g.count; ++i ) g.resident[i] = resident[mine_[l][j + i]];
			lw.groups.push_back( std::move( g ) );
			j = k;
		}
	}
	if ( rank_ == 0 || missing > 0 ) {
		constexpr double G = 1073741824.0;
		LOG_INFO( "[RANK %d] deepseek_v41 weights: dense %.2f GiB, expert codes %.2f GiB (%.2f GiB in pinned host memory, %.0f%% resident), arena %.2f / %.2f GiB, missing %d\n",
			rank_, dense_bytes / G, code_bytes / G, on_host / G, 100.0 * frac, arena.offset_bytes / G, arena.total_bytes / G, missing );
	}
	if ( frac < 1.0 ) {
		size_t max_part = 0;
		for ( auto& g0 : layers_[0].groups ) if ( g0.mw->gate[0].w0.size_bytes() > max_part ) max_part = g0.mw->gate[0].w0.size_bytes();
		for ( int b = 2; b <= 5; ++b ) swap_staging_[b] = static_cast<uint8_t*>( engine::core::arena_alloc( host_arena_, max_part, 256 ) );
	}
	backend::stream_sync( stream );
	weights_loaded_ = missing == 0;
	return weights_loaded_;
}

bool DeepseekV41Pipeline::rebalance_experts( backend::stream_t stream ) {
	if ( !weights_loaded_ || !host_arena_.base_ptr ) return false;
	backend::set_device( rank_ );
	backend::stream_sync( stream );
	const int L = config_.num_layers, E = config_.num_experts;
	const int cap_swaps = runtime::options().expert_swaps;
	const size_t budget = static_cast<size_t>( std::max( 1, runtime::options().expert_swap_mib ) ) << 20;
	const double cap_w = std::min( 100, std::max( 0, runtime::options().expert_history_pct ) ) / 100.0;
	const uint64_t margin = static_cast<uint64_t>( std::max( 0, runtime::options().expert_swap_margin ) );
	int moved = 0;
	for ( int l = 0; l < L; ++l ) {
		auto& groups = layers_[l].groups;
		if ( groups.empty() || groups[0].mw->heat == nullptr ) continue;
		const int first0 = groups[0].first;
		// drain the counters: what expert_heat took meanwhile (pending_) plus what the kernels counted since
		std::vector<uint32_t> cnt;
		double s_sum = 0.0, t_sum = 0.0;
		bool any_cold = false;
		for ( auto& g : groups ) {
			cnt.assign( g.count, 0 );
			backend::memcpy_sync( cnt.data(), g.mw->heat, static_cast<size_t>( g.count ) * sizeof( uint32_t ), backend::Copy::D2H );
			backend::memset_sync( g.mw->heat, 0, static_cast<size_t>( g.count ) * sizeof( uint32_t ) );
			for ( int i = 0; i < g.count; ++i ) {
				const size_t at = static_cast<size_t>( l ) * E + mine_[l][g.first - first0 + i];
				heat_total_[at] += cnt[i];
				const uint64_t n = static_cast<uint64_t>( cnt[i] ) + pending_[at];
				pending_[at] = 0;
				window_routes_ += n;
				if ( g.resident[i] ) window_hits_ += n; else any_cold = true;
				g.score[i] = g.score[i] / 2 + n;
				s_sum += static_cast<double>( g.score[i] );
				t_sum += static_cast<double>( heat_total_[at] );
			}
		}
		if ( !any_cold ) continue;
		// the decision blends the recent windows with the layer's long-run counts (as ExpertTier)
		const double w = ( s_sum > 0.0 && t_sum > 0.0 ) ? cap_w * t_sum / ( t_sum + 4.0 * s_sum ) : 0.0;
		size_t spent = 0;
		int done = 0;
		for ( auto& g : groups ) {
			std::vector<uint64_t> eff( g.count );
			for ( int i = 0; i < g.count; ++i ) {
				const double t = static_cast<double>( heat_total_[static_cast<size_t>( l ) * E + mine_[l][g.first - first0 + i]] );
				eff[i] = w > 0.0 ? static_cast<uint64_t>( ( 1.0 - w ) * static_cast<double>( g.score[i] ) + w * t * s_sum / t_sum + 0.5 ) : g.score[i];
			}
			const size_t part = g.mw->gate[0].w0.size_bytes();
			const size_t swap_bytes = 2 * 3 * part;
			while ( true ) {
				if ( cap_swaps >= 0 ? done >= cap_swaps : ( done > 0 && spent + swap_bytes > budget / static_cast<size_t>( L ) ) ) break;
				int hot = -1, cool = -1;   // the hottest cold expert, the coldest resident one
				for ( int i = 0; i < g.count; ++i ) {
					if ( !g.resident[i] ) { if ( hot < 0 || eff[i] > eff[hot] ) hot = i; }
					else if ( cool < 0 || eff[i] < eff[cool] ) cool = i;
				}
				if ( hot < 0 || cool < 0 || eff[hot] < eff[cool] + eff[cool] / 2 + margin ) break;
				pod::LinearWeight* set[3] = { g.mw->gate, g.mw->up, g.mw->down };
				void** table = static_cast<void**>( g.mw->device_ptrs );
				for ( int p = 0; p < 3; ++p ) {
					uint8_t* dev = const_cast<uint8_t*>( static_cast<const uint8_t*>( set[p][cool].w0.data ) );
					uint8_t* host = const_cast<uint8_t*>( static_cast<const uint8_t*>( set[p][hot].w0.data ) );
					ENGINE_ASSERT( set[p][cool].w0.size_bytes() == part && set[p][hot].w0.size_bytes() == part, "DeepseekV41Pipeline::rebalance_experts: code sizes differ within a bit width" );
					uint8_t*& stg = swap_staging_[g.bits];
					backend::memcpy_sync( stg, dev, part, backend::Copy::D2H );
					backend::memcpy_sync( dev, host, part, backend::Copy::H2D );
					set[p][cool].w0.data = stg;     // the resident expert's bytes now sit in the staging buffer, the hot expert's in its device buffer
					set[p][hot].w0.data = dev;
					stg = host;
					void* nh = dev; void* nc = set[p][cool].w0.data;
					backend::memcpy_sync( table + static_cast<size_t>( 3 * p ) * g.count + hot, &nh, sizeof( void* ), backend::Copy::H2D );
					backend::memcpy_sync( table + static_cast<size_t>( 3 * p ) * g.count + cool, &nc, sizeof( void* ), backend::Copy::H2D );
				}
				g.resident[hot] = 1;
				g.resident[cool] = 0;
				spent += swap_bytes;
				++done;
				++swaps_;
			}
		}
		moved += done;
	}
	if ( runtime::options().expert_log ) {
		LOG_INFO( "[experts rank %d] window hit rate %.3f over %llu routes, %d swaps (%llu total)\n", rank_, window_routes_ ? static_cast<double>( window_hits_ ) / window_routes_ : 1.0,
			static_cast<unsigned long long>( window_routes_ ), moved, static_cast<unsigned long long>( swaps_ ) );
	}
	window_routes_ = window_hits_ = 0;
	++rebalances_;
	return true;
}

void DeepseekV41Pipeline::embed_tokens( const int* token_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= pod::MAX_ROWS, "DeepseekV41Pipeline::embed_tokens: count out of range" );
	backend::set_device( rank_ );
	pending_tokens_.assign( token_ids, token_ids + count );
	embed_rows_.shape[0] = count;
	if ( rank_ == 0 ) {
		backend::memcpy_async( d_token_ids_, token_ids, count * sizeof( int ), backend::Copy::H2D, stream );
		engine::modules::Embedding::forward( embed_table_, d_token_ids_, count, embed_rows_, stream );
	}
	else {
		backend::memset_async( embed_rows_.data, 0, embed_rows_.size_bytes(), stream );
	}
	if ( ring_ ) ring_->all_reduce( embed_rows_, stream );
}

// Host side of the Engram lookups: each row's compressed id joins its slot's history, its n-gram hashes pick rows of the two tables
// (FP8 with power-of-two block scales, read through the checkpoint mapping), dequantized to FP16 and uploaded
void DeepseekV41Pipeline::engram_prepare( const pod::BatchRows& rows, backend::stream_t stream ) {
	const int m = rows.count;
	ENGINE_ASSERT( static_cast<int>( pending_tokens_.size() ) == m, "DeepseekV41Pipeline: embed_tokens and forward disagree on the row count" );
	const int N = eg_.ngram, NH = eg_.heads, D = eg_.dim, nb = D / eg_.block;
	const int per = ( N - 1 ) * NH;   // table rows per position and Engram layer
	const size_t cols = static_cast<size_t>( per ) * D;
	// 1. hashes (the history is updated row by row, so a chunk sees its own earlier rows)
	eg_.idx.resize( static_cast<size_t>( m ) * eg_.layers * per );
	for ( int r = 0; r < m; ++r ) {
		const int s = rows.slot[r], pos = rows.pos[r];
		const int tok = pending_tokens_[r];
		eg_.hist[s][pos] = ( tok >= 0 && tok < eg_.vocab ) ? eg_.token_map[tok] : eg_.pad;
		int64_t toks[8];
		for ( int k = 0; k < N; ++k ) toks[k] = pos - k >= 0 ? eg_.hist[s][pos - k] : eg_.pad;
		for ( int e = 0; e < eg_.layers; ++e ) {
			int64_t rolling = toks[0] * eg_.mult[static_cast<size_t>( e ) * N];
			for ( int i = 1; i < N; ++i ) {
				rolling ^= toks[i] * eg_.mult[static_cast<size_t>( e ) * N + i];
				for ( int h = 0; h < NH; ++h ) {
					const size_t c = static_cast<size_t>( i - 1 ) * NH + h;
					eg_.idx[( static_cast<size_t>( r ) * eg_.layers + e ) * per + c] =
						rolling % eg_.primes[static_cast<size_t>( e ) * per + c] + eg_.offsets[static_cast<size_t>( e ) * per + c];
				}
			}
		}
	}
	// 2. the rows' pages are requested all at once (the tables are far larger than the page cache: a miss is an NVMe read, and serial
	// faults cost ~1 ms each; asked together they overlap)
	const uintptr_t pg = static_cast<uintptr_t>( eg_.page );
	for ( size_t k = 0; k < eg_.idx.size(); ++k ) {
		const int e = static_cast<int>( ( k / per ) % eg_.layers );
		const uintptr_t w = reinterpret_cast<uintptr_t>( eg_.table[e] + static_cast<size_t>( eg_.idx[k] ) * D );
		const uintptr_t sc = reinterpret_cast<uintptr_t>( eg_.scale[e] + static_cast<size_t>( eg_.idx[k] ) * nb );
		madvise( reinterpret_cast<void*>( w & ~( pg - 1 ) ), ( ( w + D - 1 ) & ~( pg - 1 ) ) - ( w & ~( pg - 1 ) ) + pg, MADV_WILLNEED );
		madvise( reinterpret_cast<void*>( sc & ~( pg - 1 ) ), pg, MADV_WILLNEED );
	}
	// 3. gather and dequantize (FP8 with a power-of-two scale per block)
	for ( int r = 0; r < m; ++r ) {
		for ( int e = 0; e < eg_.layers; ++e ) {
			backend::half_t* dst = eg_.host_rows + ( static_cast<size_t>( e ) * pod::MAX_ROWS + r ) * cols;
			for ( int c = 0; c < per; ++c ) {
				const int64_t row = eg_.idx[( static_cast<size_t>( r ) * eg_.layers + e ) * per + c];
				const uint8_t* w = eg_.table[e] + static_cast<size_t>( row ) * D;
				const uint8_t* sc = eg_.scale[e] + static_cast<size_t>( row ) * nb;
				backend::half_t* o = dst + static_cast<size_t>( c ) * D;
				for ( int b = 0; b < nb; ++b ) {
					const float f = std::ldexp( 1.0f, static_cast<int>( sc[b] ) - 127 );
					for ( int d = b * eg_.block; d < ( b + 1 ) * eg_.block; ++d ) o[d] = backend::from_float_half( eg_.e4m3[w[d]] * f );
				}
			}
		}
	}
	for ( int e = 0; e < eg_.layers; ++e ) {
		backend::memcpy_async( e_rows_[e].data, eg_.host_rows + static_cast<size_t>( e ) * pod::MAX_ROWS * cols, static_cast<size_t>( m ) * cols * sizeof( backend::half_t ),
			backend::Copy::H2D, stream );
	}
}

void DeepseekV41Pipeline::set_rows( int m ) {
	for ( pod::TensorView* t : { &embed_rows_, &norm_x_, &collapsed_, &attn_out_, &moe_out_, &moe_tmp_, &sh_out_, &qr_, &qr_n_, &q_, &kv_, &kv_n_, &ckv_, &csc_,
			&latent_, &kraw_, &qi_, &wi_, &woa_cat_, &sh_g_, &sh_u_, &sh_i_, &e_kv_, &e_rows_[0], &e_rows_[1] } )
		t->shape[0] = m;
}

void DeepseekV41Pipeline::forward( const pod::BatchRows& rows, backend::stream_t stream ) {
	ENGINE_ASSERT( weights_loaded_, "DeepseekV41Pipeline::forward: weights not loaded" );
	ENGINE_ASSERT( rows.count >= 1 && rows.count <= pod::MAX_ROWS, "DeepseekV41Pipeline::forward: row count out of range" );
	backend::set_device( rank_ );
	for ( int r = 0; r < rows.count; ++r ) {
		ENGINE_ASSERT( rows.slot[r] >= 0 && rows.slot[r] < config_.max_slots, "DeepseekV41Pipeline::forward: slot out of range" );
		ENGINE_ASSERT( rows.pos[r] >= 0 && rows.pos[r] < config_.max_seq, "DeepseekV41Pipeline::forward: position out of range" );
	}
	const int m = rows.count;
	key_hint_ = rows.pos[m - 1] + 1;
	lm_mode_ = m > pod::MAX_BATCH ? rows.logits_mode : pod::BatchRows::LOGITS_ALL;
	last_rows_ = lm_mode_ == pod::BatchRows::LOGITS_NONE ? 0 : lm_mode_ == pod::BatchRows::LOGITS_LAST ? 1 : m;
	ENGINE_ASSERT( last_rows_ <= logit_rows_, "DeepseekV41Pipeline::forward: more logit rows than the buffer holds" );
	set_rows( m );
	engram_prepare( rows, stream );
	backend::memcpy_async( d_rows_, &rows, rows.upload_bytes(), backend::Copy::H2D, stream );
	graphs_.run( m, stream, [&]( backend::stream_t s ) { forward_body( m, s ); } );
}

void DeepseekV41Pipeline::trace( int tag, int m, backend::stream_t stream ) {
	static const std::string& path = runtime::options().trace_layers;
	if ( path.empty() || rank_ != 0 || m <= pod::MAX_BATCH ) return;
	std::vector<float> h( static_cast<size_t>( m ) * 4 * config_.hidden_size );
	backend::stream_sync( stream );
	backend::memcpy_sync( h.data(), streams_.data, h.size() * sizeof( float ), backend::Copy::D2H );
	if ( FILE* f = std::fopen( path.c_str(), "ab" ) ) {
		const int hdr[3] = { tag, m, key_hint_ - m };
		std::fwrite( hdr, sizeof( int ), 3, f );
		std::fwrite( h.data(), sizeof( float ), h.size(), f );
		std::fclose( f );
	}
}

void DeepseekV41Pipeline::trace_tensor( int tag, const pod::TensorView& t, int m, int cols, backend::stream_t stream ) {
	static const std::string& path = runtime::options().trace_layers;
	if ( path.empty() || rank_ != 0 || m <= pod::MAX_BATCH ) return;
	const size_t n = static_cast<size_t>( m ) * cols;
	std::vector<float> f( n );
	backend::stream_sync( stream );
	if ( t.dtype == pod::DType::FP32 ) backend::memcpy_sync( f.data(), t.data, n * sizeof( float ), backend::Copy::D2H );
	else {
		std::vector<backend::half_t> h( n );
		backend::memcpy_sync( h.data(), t.data, n * sizeof( backend::half_t ), backend::Copy::D2H );
		for ( size_t i = 0; i < n; ++i ) f[i] = backend::to_float( h[i] );
	}
	if ( FILE* fp = std::fopen( ( path + ".t" ).c_str(), "ab" ) ) {
		const int hdr[3] = { tag, m, cols };
		std::fwrite( hdr, sizeof( int ), 3, fp );
		std::fwrite( f.data(), sizeof( float ), n, fp );
		std::fclose( fp );
	}
}

void DeepseekV41Pipeline::forward_body( int m, backend::stream_t stream ) {
	const int H = config_.hidden_size;
	engine::modules::HyperConnection::init_streams( embed_rows_, streams_, H, stream, m );
	DSV4::hc_identity_pre( pre_[0], m, stream );
	pre_cur_ = 0;
	for ( int l = 0; l < config_.num_layers; ++l ) {
		auto& lw = layers_[l];
		if ( lw.engram >= 0 ) {
			engine::modules::Linear::forward( e_rows_[lw.engram], lw.e_wkv, e_kv_, stream );
			DSV4::engram_gate( streams_.ptr<float>(), e_kv_.ptr<const backend::half_t>(), lw.e_w.ptr<const float>(), H, config_.norm_eps, m, stream );
		}
		attention( l, lw, m, stream );
		trace( 100 + l, m, stream );
		ffn( l, lw, m, stream );
		trace( l, m, stream );
	}
	// head: the last pre-mix collapses the streams, then the final norm and this rank's vocabulary shard
	if ( lm_mode_ == pod::BatchRows::LOGITS_NONE ) return;
	DSV4::hc_collapse_norm( streams_.ptr<const float>(), pre_[pre_cur_], final_norm_.ptr<const backend::half_t>(), norm_x_.ptr<backend::half_t>(), nullptr, 1.0f, H,
		config_.norm_eps, m, stream );
	if ( lm_mode_ == pod::BatchRows::LOGITS_LAST ) {
		pod::TensorView n1 = view( norm_x_, static_cast<size_t>( m - 1 ) * H, 1, H );
		pod::TensorView l1 = view( logits_, 0, 1, lm_shard_cols_ );
		engine::modules::Linear::forward( n1, lm_head_, l1, stream );
		return;
	}
	logits_.shape[0] = m;
	engine::modules::Linear::forward( norm_x_, lm_head_, logits_, stream );
}

void DeepseekV41Pipeline::attention( int layer, DeepseekV41LayerWeights& lw, int m, backend::stream_t stream ) {
	using engine::modules::Linear;
	using engine::modules::RMSNorm;
	const int H = config_.hidden_size;
	const float eps = config_.norm_eps;
	// mHC: this sublayer's mixes from the streams; its input uses the pre-mix of the sublayer before (lagged)
	float* pre_new = pre_[pre_cur_ ^ 1];
	pod::TensorView pre_view = pod::make_tensor_2d( pre_new, pod::DType::FP32, 1, 4 * m );
	engine::modules::HyperConnection::collapse( streams_, lw.hc_attn_fn, lw.hc_attn_base, lw.hc_attn_scale, collapsed_, mhc_post_, mhc_comb_, mhc_inv_rms_, mhc_mix_,
		pre_view, H, config_.hc_sinkhorn_iters, config_.hc_eps, stream, m );
	DSV4::hc_collapse_norm( streams_.ptr<const float>(), pre_[pre_cur_], lw.attn_norm.ptr<const backend::half_t>(), norm_x_.ptr<backend::half_t>(), mhc_post_.ptr<float>(),
		kOutScale, H, eps, m, stream );
	pre_cur_ ^= 1;
	trace_tensor( 1000 + layer, norm_x_, m, H, stream );

	const float* rope = lw.ratio > 0 ? rope_comp_ : rope_plain_;
	Linear::forward( norm_x_, lw.wq_a, qr_, stream );
	RMSNorm::forward( qr_, lw.q_norm, qr_n_, eps, 0.0f, stream );
	Linear::forward( qr_n_, lw.wq_b, q_, stream );
	Linear::forward( norm_x_, lw.wkv, kv_, stream );
	RMSNorm::forward( kv_, lw.kv_norm, kv_n_, eps, 0.0f, stream );
	trace_tensor( 3000 + layer, kv_n_, m, DSV4::HD, stream );
	trace_tensor( 6000 + layer, q_, m, lh_ * DSV4::HD, stream );
	DSV4::window_store( kv_n_.ptr<const backend::half_t>(), win_ring_, rope, layer, m, d_rows_, win_slot_stride_, stream );
	const backend::half_t* comp = nullptr;
	if ( lw.ratio > 0 ) {
		const int s = lw.kv_src;
		if ( lw.is_kv_source ) {
			Linear::forward( norm_x_, lw.c_wkv, ckv_, stream );
			if ( lw.ratio > 1 ) Linear::forward( norm_x_, lw.c_wgate, csc_, stream );
			DSV4::compress( ckv_.ptr<const backend::half_t>(), csc_.ptr<const backend::half_t>(), lw.c_norm.ptr<const backend::half_t>(), cring_kv_[s], cring_sc_[s],
				latent_.ptr<backend::half_t>(), comp_kv_[s], rope, lw.ratio, m, d_rows_, cring_slot_stride_, comp_slot_stride_[s], eps, stream );
			Linear::forward( latent_, lw.idx_wk, kraw_, stream );
			DSV4::index_key_store( kraw_.ptr<const backend::half_t>(), lw.idx_knorm.ptr<const backend::half_t>(), idx_k_[s], rope, lw.ratio, m, d_rows_, idx_slot_stride_[s], eps, stream );
		}
		if ( lw.is_index_source ) {
			Linear::forward( qr_n_, lw.idx_wq_b, qi_, stream );
			Linear::forward( norm_x_, lw.idx_wproj, wi_, stream );
			const int cand_mode = layer == config_.candidate_source ? 1 : layer > config_.candidate_source ? 2 : 0;
			DSV4::indexer( qi_.ptr<const backend::half_t>(), wi_.ptr<const backend::half_t>(), idx_k_[s], idx_scores_, idx_cand_, idx_sel_, idx_nsel_, rope, lw.ratio, src_cap_[s],
				cand_mode, config_.candidate_blocks, config_.candidate_block, m, d_rows_, idx_slot_stride_[s], stream );
		}
		comp = comp_kv_[s];
	}
	DSV4::attend( q_.ptr<const backend::half_t>(), win_ring_, comp, idx_sel_, idx_nsel_, lw.sink.ptr<const float>(), rope, o_g_.ptr<backend::half_t>(), lh_, layer, m, d_rows_,
		win_slot_stride_, lw.ratio > 0 ? comp_slot_stride_[lw.kv_src] : 0, m <= pod::MAX_BATCH ? attend_ws_ : nullptr, stream );
	const int G = DSV4::GROUP_HEADS * DSV4::HD, O = config_.o_lora_rank;
	for ( int g = 0; g < gl_; ++g ) {
		pod::TensorView in = view( o_g_, static_cast<size_t>( g ) * m * G, m, G );
		pod::TensorView out = view( woa_out_, static_cast<size_t>( g ) * m * O, m, O );
		Linear::forward( in, lw.wo_a[g], out, stream );
	}
	DSV4::concat_groups( woa_out_.ptr<const backend::half_t>(), woa_cat_.ptr<backend::half_t>(), gl_, O, m, stream );
	Linear::forward( woa_cat_, lw.wo_b, attn_out_, stream );
	if ( ring_ ) ring_->all_reduce( attn_out_, stream );
	trace_tensor( 7000 + layer, o_g_, m, gl_ * DSV4::GROUP_HEADS * DSV4::HD, stream );
	trace_tensor( 2000 + layer, attn_out_, m, H, stream );
	engine::modules::HyperConnection::update( streams_, attn_out_, mhc_post_, mhc_comb_, H, stream, m );
}

void DeepseekV41Pipeline::ffn( int layer, DeepseekV41LayerWeights& lw, int m, backend::stream_t stream ) {
	(void) layer;
	using engine::modules::Linear;
	const int H = config_.hidden_size;
	float* pre_new = pre_[pre_cur_ ^ 1];
	pod::TensorView pre_view = pod::make_tensor_2d( pre_new, pod::DType::FP32, 1, 4 * m );
	engine::modules::HyperConnection::collapse( streams_, lw.hc_ffn_fn, lw.hc_ffn_base, lw.hc_ffn_scale, collapsed_, mhc_post_, mhc_comb_, mhc_inv_rms_, mhc_mix_,
		pre_view, H, config_.hc_sinkhorn_iters, config_.hc_eps, stream, m );
	DSV4::hc_collapse_norm( streams_.ptr<const float>(), pre_[pre_cur_], lw.ffn_norm.ptr<const backend::half_t>(), norm_x_.ptr<backend::half_t>(), mhc_post_.ptr<float>(),
		kOutScale, H, config_.norm_eps, m, stream );
	pre_cur_ ^= 1;
	trace_tensor( 4000 + layer, norm_x_, m, H, stream );

	DSV4::route( norm_x_.ptr<const backend::half_t>(), lw.router_w_t.ptr<const backend::half_t>(), lw.router_bias.ptr<const float>(), lw.perm, d_topk_idx_, d_topk_w_, route_scratch_,
		H, config_.num_experts, config_.top_k, config_.route_scale, m, stream );
	bool first = true;
	for ( auto& g : lw.groups ) {
		pod::TensorView& out = first ? moe_out_ : moe_tmp_;
		engine::modules::MoE::forward( norm_x_, *g.mw, d_topk_idx_, d_topk_w_, out, g.first, g.first + g.count, moe_scratch_, config_.swiglu_limit, stream );
		if ( !first ) engine::modules::MoERouter::accumulate( moe_out_.ptr<backend::half_t>(), moe_tmp_.ptr<const backend::half_t>(), backend::from_float_half( 1.0f ), m * H, stream );
		first = false;
	}
	if ( first ) backend::memset_async( moe_out_.data, 0, static_cast<size_t>( m ) * H * sizeof( backend::half_t ), stream );
	if ( lw.sh_gate.is_valid() ) {
		Linear::forward( norm_x_, lw.sh_gate, sh_g_, stream );
		Linear::forward( norm_x_, lw.sh_up, sh_u_, stream );
		engine::modules::SwiGLU::forward( sh_g_, sh_u_, sh_i_, stream, config_.swiglu_limit );
		Linear::forward( sh_i_, lw.sh_down, sh_out_, stream );
		engine::modules::MoERouter::accumulate( moe_out_.ptr<backend::half_t>(), sh_out_.ptr<const backend::half_t>(), backend::from_float_half( 1.0f ), m * H, stream );
	}
	if ( ring_ ) ring_->all_reduce( moe_out_, stream );
	trace_tensor( 5000 + layer, moe_out_, m, H, stream );
	engine::modules::HyperConnection::update( streams_, moe_out_, mhc_post_, mhc_comb_, H, stream, m );
}

// Snapshot layout: [window: layers x WINDOW x HD halves, the rows of positions tokens - WINDOW .. tokens - 1] then per kv source [compressed
// entries [from / r, tokens / r) x HD halves | index keys of the same entries x IDX_DIM halves], then the Engram ids of [from, tokens) (int32)
size_t DeepseekV41Pipeline::seq_state_bytes( int from, int tokens, bool with_taps ) const {
	(void) with_taps;
	if ( from < 0 || tokens <= from || tokens > config_.max_seq || from % 2 || tokens % 2 || !weights_loaded_ ) return 0;
	size_t b = static_cast<size_t>( config_.num_layers ) * DSV4::WINDOW * DSV4::HD * sizeof( backend::half_t );
	for ( int s : config_.kv_sources ) {
		const int r = config_.compress_ratios[s];
		b += static_cast<size_t>( tokens / r - from / r ) * ( DSV4::HD + DSV4::IDX_DIM ) * sizeof( backend::half_t );
	}
	return b + static_cast<size_t>( tokens - from ) * sizeof( int32_t ) + 256;
}

void DeepseekV41Pipeline::copy_seq_rows( int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream ) {
	auto move = [&]( void* dev, char* at, size_t bytes ) {
		if ( bytes == 0 ) return;
		if ( to_host ) backend::memcpy_async( at, dev, bytes, backend::Copy::D2H, stream );
		else backend::memcpy_async( dev, at, bytes, backend::Copy::H2D, stream );
	};
	for ( size_t i = 0; i < config_.kv_sources.size(); ++i ) {
		const int r = config_.compress_ratios[config_.kv_sources[i]];
		const size_t e0 = from / r, n = static_cast<size_t>( tokens / r ) - e0;
		const size_t kv_bytes = n * DSV4::HD * sizeof( backend::half_t ), idx_bytes = n * DSV4::IDX_DIM * sizeof( backend::half_t );
		move( comp_kv_[i] + static_cast<size_t>( slot_id ) * comp_slot_stride_[i] + e0 * DSV4::HD, host, kv_bytes );
		move( idx_k_[i] + static_cast<size_t>( slot_id ) * idx_slot_stride_[i] + e0 * DSV4::IDX_DIM, host + kv_bytes, idx_bytes );
		host += kv_bytes + idx_bytes;
	}
	// the Engram history is host memory
	int32_t* ids = reinterpret_cast<int32_t*>( host );
	std::vector<int>& hist = eg_.hist[slot_id];
	for ( int p = from; p < tokens; ++p ) {
		if ( to_host ) ids[p - from] = hist[p];
		else hist[p] = ids[p - from];
	}
}

bool DeepseekV41Pipeline::save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* dst = static_cast<char*>( host_dst );
	// the window: positions tokens - WINDOW .. tokens - 1 of every layer's ring (at most two contiguous runs)
	const size_t row = static_cast<size_t>( DSV4::HD ) * sizeof( backend::half_t );
	const int first = std::max( 0, tokens - DSV4::WINDOW );
	for ( int l = 0; l < config_.num_layers; ++l ) {
		const backend::half_t* ring = win_ring_ + static_cast<size_t>( slot_id ) * win_slot_stride_ + static_cast<size_t>( l ) * DSV4::RING * DSV4::HD;
		char* out = dst + static_cast<size_t>( l ) * DSV4::WINDOW * row;
		for ( int p = first; p < tokens; ) {
			const int r = p % DSV4::RING, n = std::min( tokens - p, DSV4::RING - r );
			backend::memcpy_async( out + static_cast<size_t>( p - first ) * row, ring + static_cast<size_t>( r ) * DSV4::HD, static_cast<size_t>( n ) * row, backend::Copy::D2H, stream );
			p += n;
		}
	}
	copy_seq_rows( slot_id, from, tokens, dst + static_cast<size_t>( config_.num_layers ) * DSV4::WINDOW * row, true, stream );
	backend::stream_sync( stream );
	return true;
}

bool DeepseekV41Pipeline::load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream ) {
	if ( seq_state_bytes( from, tokens, with_taps ) == 0 || slot_id < 0 || slot_id >= config_.max_slots ) return false;
	backend::set_device( rank_ );
	char* src = const_cast<char*>( static_cast<const char*>( host_src ) );
	const size_t row = static_cast<size_t>( DSV4::HD ) * sizeof( backend::half_t );
	if ( !kv_only ) {
		const int first = std::max( 0, tokens - DSV4::WINDOW );
		for ( int l = 0; l < config_.num_layers; ++l ) {
			backend::half_t* ring = win_ring_ + static_cast<size_t>( slot_id ) * win_slot_stride_ + static_cast<size_t>( l ) * DSV4::RING * DSV4::HD;
			const char* in = src + static_cast<size_t>( l ) * DSV4::WINDOW * row;
			for ( int p = first; p < tokens; ) {
				const int r = p % DSV4::RING, n = std::min( tokens - p, DSV4::RING - r );
				backend::memcpy_async( ring + static_cast<size_t>( r ) * DSV4::HD, in + static_cast<size_t>( p - first ) * row, static_cast<size_t>( n ) * row, backend::Copy::H2D, stream );
				p += n;
			}
		}
	}
	copy_seq_rows( slot_id, from, tokens, src + static_cast<size_t>( config_.num_layers ) * DSV4::WINDOW * row, false, stream );
	backend::stream_sync( stream );
	return true;
}

void DeepseekV41Pipeline::sample_tokens( int* out_preds, int count, backend::stream_t stream ) {
	(void) out_preds; (void) count; (void) stream;
	ENGINE_ASSERT( false, "DeepseekV41Pipeline::sample_tokens: the vocabulary is sharded, use sample_local" );
}

bool DeepseekV41Pipeline::sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "DeepseekV41Pipeline::sample_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	logits_.shape[0] = count;
	engine::modules::Elementwise::batched_argmax_range( logits_, d_argmax_idx_, d_argmax_val_, lm_shard_real_cols_, count, lm_shard_offset_, stream );
	backend::memcpy_async( h_argmax_idx_, d_argmax_idx_, count * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	std::memcpy( out_tokens, h_argmax_idx_, count * sizeof( int ) );
	if ( out_vals != nullptr ) {
		float h_vals[pod::MAX_BATCH];
		backend::memcpy_async( h_vals, d_argmax_val_, count * sizeof( float ), backend::Copy::D2H, stream );
		backend::stream_sync( stream );
		std::memcpy( out_vals, h_vals, count * sizeof( float ) );
	}
	return true;
}

bool DeepseekV41Pipeline::topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream ) {
	ENGINE_ASSERT( count >= 1 && count <= last_rows_, "DeepseekV41Pipeline::topk_local: count exceeds last forward rows" );
	backend::set_device( rank_ );
	engine::spec::dflash2_topk_shard( logits_.ptr<const backend::half_t>(), count, lm_shard_cols_, lm_shard_real_cols_, 16, d_samp_vals_, d_samp_idx_, lm_shard_offset_,
		samp_part_v_, samp_part_i_, samp_counters_, stream );
	backend::memcpy_async( out_vals, d_samp_vals_, static_cast<size_t>( count ) * 16 * sizeof( float ), backend::Copy::D2H, stream );
	backend::memcpy_async( out_ids, d_samp_idx_, static_cast<size_t>( count ) * 16 * sizeof( int ), backend::Copy::D2H, stream );
	backend::stream_sync( stream );
	return true;
}

const pod::TensorView& DeepseekV41Pipeline::gather_logits( int rows, const backend::half_t* const* peer_shard_ptrs, int peer_shard_cols, backend::stream_t stream ) {
	ENGINE_ASSERT( rank_ == 0 && rows >= 1 && rows <= last_rows_ && peer_shard_cols == lm_shard_cols_, "DeepseekV41Pipeline::gather_logits" );
	full_logits_.shape[0] = rows;
	for ( int d = 0; d < config_.tp_size; ++d ) {
		const size_t shard_elems = static_cast<size_t>( 128 * lm_shard_spans( config_.vocab_size, config_.tp_size, d ) );
		const size_t shard_off = static_cast<size_t>( lm_shard_offset( config_.vocab_size, config_.tp_size, d ) );
		const backend::half_t* src_base = d == rank_ ? logits_.ptr<const backend::half_t>() : peer_shard_ptrs[d];
		for ( int r = 0; r < rows; ++r ) {
			backend::peer_copy_async( full_logits_.ptr<backend::half_t>() + static_cast<size_t>( r ) * config_.vocab_size + shard_off, rank_,
				src_base + static_cast<size_t>( r ) * lm_shard_cols_, d, shard_elems * sizeof( backend::half_t ), stream );
		}
	}
	return full_logits_;
}

} // namespace engine::models
