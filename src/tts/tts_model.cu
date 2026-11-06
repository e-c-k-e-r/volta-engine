#include "engine/tts/tts_model.h"
#include "engine/core/log.h"
#include "engine/loader/device_upload.h"
#include <cublas_v2.h>
#include <nlohmann/json.hpp>

namespace engine::tts {

TtsModel::~TtsModel() {
	engine::loader::safetensors_close( st_ );
	if ( arena_.base_ptr ) { engine::core::arena_destroy( arena_ ); }
	if ( ws_arena_.base_ptr ) { engine::core::arena_destroy( ws_arena_ ); }
	if ( blas_ ) { cublasDestroy( static_cast<cublasHandle_t>( blas_ ) ); }
}

bool TtsModel::load( const std::string& path, int device ) {
	if ( !engine::loader::safetensors_open( st_, path.c_str() ) ) {
		LOG_ERROR( "tts: can't open %s", path.c_str() );
		return false;
	}
	const std::string* format = engine::loader::safetensors_metadata( st_, "format" );
	const std::string* config = engine::loader::safetensors_metadata( st_, "config" );
	const std::string* vocab = engine::loader::safetensors_metadata( st_, "vocab" );
	if ( !format || *format != "vall-e-v3" || !config || !vocab ) {
		LOG_ERROR( "tts: %s isn't a vall-e v3 export (no format / config / vocab metadata)", path.c_str() );
		return false;
	}
	try {
		auto c = nlohmann::json::parse( *config );
		cfg_.n_text = c.at( "n_text" );
		cfg_.d_model = c.at( "d_model" );
		cfg_.layers = c.at( "layers" );
		cfg_.heads = c.at( "heads" );
		cfg_.patch = c.value( "patch", 2 );
		cfg_.rot_frac = c.value( "rot_frac", 1.0f );
		cfg_.rope_theta = c.value( "rope_theta", 10000.0f );
		cfg_.gated = c.value( "gated", true );
		cfg_.qk_norm = c.value( "qk_norm", true );
		cfg_.flow_width = c.value( "flow_width", 512 );
		cfg_.flow_depth = c.value( "flow_depth", 4 );
		cfg_.n_modes = c.value( "n_modes", 3 );
		cfg_.length = c.value( "length", false );
		cfg_.text_prenet = c.value( "text_prenet", 0 );
		if ( c.value( "head", std::string( "classifier" ) ) != "flow" || c.value( "ar", false ) || c.value( "n_lang", 0 ) || c.value( "ctc", false ) ) {
			LOG_ERROR( "tts: only flow-head NAR models without accent / CTC heads are supported" );
			return false;
		}
		cfg_.vocab = nlohmann::json::parse( *vocab ).get<std::vector<std::string>>();
	} catch ( const std::exception& e ) {
		LOG_ERROR( "tts: bad config metadata: %s", e.what() );
		return false;
	}
	if ( cfg_.text_prenet ) {
		LOG_ERROR( "tts: the text pre-net isn't ported yet" );
		return false;
	}

	if ( !engine::core::arena_create( arena_, device, st_.file_size * 2 + ( 64ull << 20 ) ) ) {
		LOG_ERROR( "tts: can't allocate the weight arena" );
		return false;
	}
	engine::loader::DeviceUploader u( st_, arena_, "tts" );
	bool ok = true;
	auto F32 = [&]( const std::string& n ) { const float* p = u.f32( n ); ok &= p != nullptr; return p; };
	auto F16 = [&]( const std::string& n ) { const half* p = u.f16( n ); ok &= p != nullptr; return p; };
	w_.text = F16( "text.weight" );
	w_.audio_in = F16( "audio_in.weight" );
	w_.audio_in_b = F32( "audio_in.bias" );
	w_.mask_emb = F32( "mask_emb" );
	w_.segment = F32( "segment.weight" );
	w_.mode = F32( "mode.weight" );
	w_.norm = F32( "norm.weight" );
	for ( int l = 0; l < cfg_.layers; ++l ) {
		const std::string p = "blocks." + std::to_string( l ) + ".";
		TtsBlock b;
		b.n1 = F32( p + "n1.weight" );
		b.n2 = F32( p + "n2.weight" );
		b.qkv = F16( p + "attn.qkv.weight" );
		b.out = F16( p + "attn.out.weight" );
		if ( cfg_.gated ) { b.gate = F16( p + "attn.gate.weight" ); }
		if ( cfg_.qk_norm ) { b.q_norm = F32( p + "attn.q_norm.weight" ); b.k_norm = F32( p + "attn.k_norm.weight" ); }
		b.up = F16( p + "up.weight" );
		b.down = F16( p + "down.weight" );
		w_.blocks.push_back( b );
	}
	w_.f_inp = F16( "flow.inp.weight" ); w_.f_inp_b = F32( "flow.inp.bias" );
	w_.f_cond = F16( "flow.cond.weight" ); w_.f_cond_b = F32( "flow.cond.bias" );
	w_.f_time0 = F16( "flow.time.0.weight" ); w_.f_time0_b = F32( "flow.time.0.bias" );
	w_.f_time2 = F16( "flow.time.2.weight" ); w_.f_time2_b = F32( "flow.time.2.bias" );
	w_.f_final_mod = F16( "flow.final_mod.weight" ); w_.f_final_mod_b = F32( "flow.final_mod.bias" );
	w_.f_out = F16( "flow.out.weight" ); w_.f_out_b = F32( "flow.out.bias" );
	for ( int l = 0; l < cfg_.flow_depth; ++l ) {
		const std::string p = "flow.blocks." + std::to_string( l ) + ".";
		TtsFlowBlock b;
		b.mlp0 = F16( p + "mlp.0.weight" ); b.mlp0_b = F32( p + "mlp.0.bias" );
		b.mlp2 = F16( p + "mlp.2.weight" ); b.mlp2_b = F32( p + "mlp.2.bias" );
		b.mod = F16( p + "mod.weight" ); b.mod_b = F32( p + "mod.bias" );
		w_.flow_blocks.push_back( b );
	}
	if ( cfg_.length ) {
		w_.len_q = F32( "len_q" );
		w_.length_w = F32( "length.weight" );
		w_.length_b = F32( "length.bias" );
	}
	if ( !ok ) {
		LOG_ERROR( "tts: %s is missing tensors", path.c_str() );
		return false;
	}
	alloc_workspace();
	return true;
}

} // namespace engine::tts
