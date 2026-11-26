// The NeMo codec (vall_e/emb/nemo.py HiFiGANEncoder / GroupFiniteScalarQuantizer / HiFiGANDecoder), FP32. A convolution
// is GEMMs over chunks of time: the chunk's columns (each input channel's k taps, strided, dilated, reflection-padded)
// against the weight [co, ci * k]. A transposed convolution with stride s and kernel 2s gives output o = q s + r (phase
// r) from two input frames, so its weight is regrouped at load by phase and input shift: [s * co, 3 * ci] against columns
// of the input at q - 1, q, q + 1 (zeros past the ends), then the phases are interleaved into time. The operators are the
// backend's (engine/modules/tts.h CodecOps).
#include "engine/tts/codec.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/loader/device_upload.h"
#include "engine/modules/tts.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>

namespace engine::tts {

using engine::modules::CodecOps;

namespace {

constexpr size_t CHUNK = 16u << 20;  // floats of a GEMM chunk's columns, and of its phase outputs

}  // namespace

NemoCodec::~NemoCodec() {
	for ( float* p : { x_, cur_, a_, b_, acc_, cols_, ph_ } ) { if ( p ) { backend::device_free( p ); } }
	if ( digits_ ) { backend::device_free( digits_ ); }
	engine::loader::safetensors_close( st_ );
	if ( arena_.base_ptr ) { engine::core::arena_destroy( arena_ ); }
}

bool NemoCodec::load( const std::string& path, int device ) {
	if ( !engine::loader::safetensors_open( st_, path.c_str() ) ) {
		LOG_ERROR( "codec: can't open %s", path.c_str() );
		return false;
	}
	const std::string* format = engine::loader::safetensors_metadata( st_, "format" );
	const std::string* config = engine::loader::safetensors_metadata( st_, "config" );
	if ( !format || *format != "nemo-codec" || !config ) {
		LOG_ERROR( "codec: %s isn't a codec export (python -m vall_e.export codec)", path.c_str() );
		return false;
	}
	// the engine builds without exceptions: every key is checked before it's read (a missing or mistyped one would abort)
	const auto c = nlohmann::json::parse( *config, nullptr, false );
	auto nums = []( const nlohmann::json& o, std::initializer_list<const char*> keys ) {
		for ( const char* k : keys ) { if ( !o.contains( k ) || !o[k].is_number() ) { return false; } }
		return true;
	};
	auto int_lists = []( const nlohmann::json& o, std::initializer_list<const char*> keys ) {
		for ( const char* k : keys ) {
			if ( !o.contains( k ) || !o[k].is_array() ) { return false; }
			for ( const auto& v : o[k] ) { if ( !v.is_number_integer() ) { return false; } }
		}
		return true;
	};
	auto strings = []( const nlohmann::json& o, std::initializer_list<const char*> keys ) {
		for ( const char* k : keys ) { if ( !o.contains( k ) || !o[k].is_string() ) { return false; } }
		return true;
	};
	const bool good = c.is_object() && nums( c, { "sample_rate", "samples_per_frame", "groups", "fsq_eps" } ) && int_lists( c, { "levels" } ) &&
		c.contains( "encoder" ) && c["encoder"].is_object() && c.contains( "decoder" ) && c["decoder"].is_object() &&
		nums( c["encoder"], { "base_channels", "encoded_dim", "in_kernel_size", "out_kernel_size" } ) &&
		int_lists( c["encoder"], { "down_sample_rates", "resblock_kernel_sizes", "resblock_dilation_sizes" } ) && strings( c["encoder"], { "activation" } ) &&
		nums( c["decoder"], { "input_dim", "base_channels", "in_kernel_size", "out_kernel_size" } ) &&
		int_lists( c["decoder"], { "up_sample_rates", "resblock_kernel_sizes", "resblock_dilation_sizes" } ) &&
		strings( c["decoder"], { "activation", "output_activation" } );
	if ( !good ) {
		LOG_ERROR( "codec: bad config metadata in %s", path.c_str() );
		return false;
	}
	cfg_.sample_rate = c["sample_rate"];
	cfg_.samples_per_frame = c["samples_per_frame"];
	cfg_.levels = c["levels"].get<std::vector<int>>();
	cfg_.groups = c["groups"];
	cfg_.fsq_eps = c["fsq_eps"];
	const auto& e = c["encoder"];
	const auto& d = c["decoder"];
	cfg_.enc_channels = e["base_channels"];
	cfg_.enc_dim = e["encoded_dim"];
	cfg_.enc_in_kernel = e["in_kernel_size"];
	cfg_.enc_out_kernel = e["out_kernel_size"];
	cfg_.down_rates = e["down_sample_rates"].get<std::vector<int>>();
	cfg_.enc_res_kernels = e["resblock_kernel_sizes"].get<std::vector<int>>();
	cfg_.enc_res_dilations = e["resblock_dilation_sizes"].get<std::vector<int>>();
	cfg_.dec_dim = d["input_dim"];
	cfg_.dec_channels = d["base_channels"];
	cfg_.dec_in_kernel = d["in_kernel_size"];
	cfg_.dec_out_kernel = d["out_kernel_size"];
	cfg_.up_rates = d["up_sample_rates"].get<std::vector<int>>();
	cfg_.dec_res_kernels = d["resblock_kernel_sizes"].get<std::vector<int>>();
	cfg_.dec_res_dilations = d["resblock_dilation_sizes"].get<std::vector<int>>();
	if ( e["activation"] != "lrelu" || d["activation"] != "half_snake" || d["output_activation"] != "ClampActivation" ) {
		LOG_ERROR( "codec: only the leaky ReLU encoder and the HalfSnake / clamp decoder are supported" );
		return false;
	}
	if ( cfg_.enc_dim != 32 || cfg_.dec_dim != 32 || cfg_.levels.size() != 4 || cfg_.groups * 4 != 32 ) {
		LOG_ERROR( "codec: expected 8 FSQ groups of 4 dimensions" );
		return false;
	}
	// the regrouped transposed-conv weights are 3 / 2 the size of the stored ones, the rest as stored
	if ( !engine::core::arena_create( arena_, device, st_.file_size * 3 / 2 + ( 64ull << 20 ) ) ) {
		LOG_ERROR( "codec: can't allocate the weight arena" );
		return false;
	}
	engine::loader::DeviceUploader u( st_, arena_, "codec" );
	bool ok = true;
	auto F32 = [&]( const std::string& n ) { const float* p = u.f32( n ); ok &= p != nullptr; return p; };
	auto conv = [&]( const std::string& p, int co, int ci, int k, int dil, int stride = 1 ) {
		Conv c;
		c.w = F32( p + ".conv.weight" );
		c.b = F32( p + ".conv.bias" );
		c.co = co; c.ci = ci; c.k = k; c.dil = dil; c.stride = stride;
		c.pad = stride == 1 ? ( k - 1 ) / 2 * dil : ( k - stride + 1 ) / 2;
		return c;
	};
	auto res_layer = [&]( const std::string& p, int C, const std::vector<int>& kernels, const std::vector<int>& dilations, bool snake ) {
		ResLayer r;
		for ( size_t b = 0; b < kernels.size(); ++b ) {
			std::vector<ResBlock> chain;
			for ( size_t j = 0; j < dilations.size(); ++j ) {
				const std::string q = p + ".res_blocks." + std::to_string( b ) + ".res_blocks." + std::to_string( j );
				ResBlock rb;
				rb.in = conv( q + ".input_conv", C, C, kernels[b], dilations[j] );
				rb.skip = conv( q + ".skip_conv", C, C, kernels[b], 1 );
				if ( snake ) {
					rb.alpha_in = F32( q + ".input_activation.activation.snake_act.alpha" );
					rb.alpha_skip = F32( q + ".skip_activation.activation.snake_act.alpha" );
				}
				chain.push_back( rb );
			}
			r.push_back( chain );
		}
		return r;
	};

	// encoder
	int C = cfg_.enc_channels;
	enc_pre_ = conv( "audio_encoder.pre_conv", C, 1, cfg_.enc_in_kernel, 1 );
	for ( size_t i = 0; i < cfg_.down_rates.size(); ++i ) {
		const std::string p = std::to_string( i );
		const int s = cfg_.down_rates[i];
		EncStage st;
		st.res = res_layer( "audio_encoder.res_layers." + p, C, cfg_.enc_res_kernels, cfg_.enc_res_dilations, false );
		st.down = conv( "audio_encoder.down_sample_conv_layers." + p, 2 * C, C, 2 * s, 1, s );
		enc_.push_back( st );
		C *= 2;
	}
	enc_post_ = conv( "audio_encoder.post_conv", cfg_.enc_dim, C, cfg_.enc_out_kernel, 1 );

	// decoder
	C = cfg_.dec_channels;
	dec_pre_ = conv( "audio_decoder.pre_conv", C, cfg_.dec_dim, cfg_.dec_in_kernel, 1 );
	for ( size_t i = 0; i < cfg_.up_rates.size(); ++i ) {
		const std::string p = std::to_string( i );
		DecStage st;
		st.alpha = F32( "audio_decoder.activations." + p + ".activation.snake_act.alpha" );
		const int s = cfg_.up_rates[i], co = C / 2;
		// ConvTranspose1d [C, co, 2s], padding (s + 1) / 2: output q s + r takes tap jh = (r + pad) % s from input q + d
		// (d = (r + pad) / s) and tap jh + s from input q + d - 1
		const std::string wn = "audio_decoder.up_sample_conv_layers." + p + ".conv.weight";
		const pod::SafeTensorMeta* m = u.find( wn );
		if ( !m || m->dtype != pod::DType::FP32 ) { LOG_ERROR( "codec: %s missing or not FP32", wn.c_str() ); return false; }
		const float* w = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st_, m ) );
		const int k = 2 * s, pad = ( s + 1 ) / 2;
		std::vector<float> big( static_cast<size_t>( s ) * co * 3 * C, 0.0f );
		for ( int r = 0; r < s; ++r ) {
			const int d = ( r + pad ) / s, jh = ( r + pad ) % s;
			for ( int o = 0; o < co; ++o ) {
				float* row = big.data() + ( static_cast<size_t>( r ) * co + o ) * 3 * C;
				for ( int c = 0; c < C; ++c ) {
					row[( d + 1 ) * C + c] = w[( static_cast<size_t>( c ) * co + o ) * k + jh];   // input q + d
					row[d * C + c] = w[( static_cast<size_t>( c ) * co + o ) * k + jh + s];       // input q + d - 1
				}
			}
		}
		float* dev = static_cast<float*>( engine::core::arena_alloc( arena_, big.size() * 4 ) );
		backend::memcpy_sync( dev, big.data(), big.size() * 4, backend::Copy::H2D );
		st.up.w = dev;
		st.up.b = F32( "audio_decoder.up_sample_conv_layers." + p + ".conv.bias" );
		st.up.co = co; st.up.ci = C; st.up.s = s;
		C = co;
		st.res = res_layer( "audio_decoder.res_layers." + p, C, cfg_.dec_res_kernels, cfg_.dec_res_dilations, true );
		dec_.push_back( st );
	}
	dec_post_alpha_ = F32( "audio_decoder.post_activation.activation.snake_act.alpha" );
	dec_post_ = conv( "audio_decoder.post_conv", 1, C, cfg_.dec_out_kernel, 1 );
	if ( !ok ) {
		LOG_ERROR( "codec: %s is missing tensors", path.c_str() );
		return false;
	}
	cols_ = static_cast<float*>( backend::device_malloc( CHUNK * 4 ) );
	ph_ = static_cast<float*>( backend::device_malloc( CHUNK * 4 ) );
	return true;
}

bool NemoCodec::reserve( int n_frames ) {
	if ( n_frames <= frames_ ) { return true; }
	for ( float** p : { &x_, &cur_, &a_, &b_, &acc_ } ) { if ( *p ) { backend::device_free( *p ); *p = nullptr; } }
	if ( digits_ ) { backend::device_free( digits_ ); digits_ = nullptr; }
	frames_ = 0;
	// the largest [C, T] anywhere in either half
	size_t most = static_cast<size_t>( cfg_.dec_channels ) * n_frames;
	size_t T = n_frames;
	int C = cfg_.dec_channels;
	for ( int s : cfg_.up_rates ) { T *= s; C /= 2; most = std::max( most, static_cast<size_t>( C ) * T ); }
	T = static_cast<size_t>( n_frames ) * cfg_.samples_per_frame;
	C = cfg_.enc_channels;
	most = std::max( most, static_cast<size_t>( C ) * T );
	for ( int s : cfg_.down_rates ) { T /= s; C *= 2; most = std::max( most, static_cast<size_t>( C ) * T ); }
	bool ok = true;
	for ( float** p : { &x_, &cur_, &a_, &b_, &acc_ } ) { ok = ok && ( *p = static_cast<float*>( backend::device_malloc_try( most * 4 ) ) ) != nullptr; }
	ok = ok && ( digits_ = static_cast<uint8_t*>( backend::device_malloc_try( static_cast<size_t>( n_frames ) * 32 ) ) ) != nullptr;
	if ( !ok ) {
		backend::clear_last_error();   // an allocation failure isn't sticky: cleared so later checks don't report it
		for ( float** p : { &x_, &cur_, &a_, &b_, &acc_ } ) { if ( *p ) { backend::device_free( *p ); *p = nullptr; } }
		if ( digits_ ) { backend::device_free( digits_ ); digits_ = nullptr; }
		LOG_ERROR( "codec: no GPU memory for %d frames (%.2f GiB of activations)", n_frames, ( most * 20.0 + n_frames * 32.0 ) / 1073741824.0 );
		return false;
	}
	frames_ = n_frames;
	return true;
}

int NemoCodec::conv( const Conv& c, const float* x, float* y, int T ) {
	const int T_out = ( T + 2 * c.pad - c.dil * ( c.k - 1 ) - 1 ) / c.stride + 1;
	ENGINE_ASSERT( c.pad < T, "codec: input shorter than a convolution's reflection padding" );
	CodecOps::conv( x, c.w, c.b, y, c.ci, c.co, c.k, c.dil, c.stride, c.pad, T, T_out, cols_, CHUNK );
	return T_out;
}

void NemoCodec::up_conv( const UpConv& u, const float* x, float* y, int T_in ) {
	CodecOps::up_conv( x, u.w, u.b, y, u.ci, u.co, u.s, T_in, cols_, ph_, CHUNK );
}

void NemoCodec::act( const float* x, float* y, const float* alpha, int C, int T ) {
	CodecOps::act( x, y, alpha, C, T );
}

void NemoCodec::res_layer( const ResLayer& r, int C, int T ) {
	// the mean of the branches, each a chain of x + skip(act(in(act(x)))) from the layer's input
	const size_t n = static_cast<size_t>( C ) * T;
	for ( size_t b = 0; b < r.size(); ++b ) {
		backend::memcpy_async( cur_, x_, n * 4, backend::Copy::D2D, nullptr );
		for ( const ResBlock& rb : r[b] ) {
			act( cur_, a_, rb.alpha_in, C, T );
			conv( rb.in, a_, b_, T );
			act( b_, b_, rb.alpha_skip, C, T );
			conv( rb.skip, b_, a_, T );
			CodecOps::axpy( cur_, a_, 1.0f, 1.0f, n );
		}
		if ( b == 0 ) { backend::memcpy_async( acc_, cur_, n * 4, backend::Copy::D2D, nullptr ); }
		else { CodecOps::axpy( acc_, cur_, 1.0f, 1.0f, n ); }
	}
	CodecOps::axpy( x_, acc_, 0.0f, 1.0f / static_cast<float>( r.size() ), n );
}

std::vector<float> NemoCodec::decode( const std::vector<uint8_t>& digits, int n_frames ) {
	using backend::Copy;
	ENGINE_ASSERT( static_cast<int>( digits.size() ) >= n_frames * 32, "codec: fewer digits than frames" );
	ENGINE_ASSERT( n_frames >= 4, "codec: at least 4 frames" );
	if ( !reserve( n_frames ) ) { return {}; }
	backend::memcpy_sync( digits_, digits.data(), static_cast<size_t>( n_frames ) * 32, Copy::H2D );
	CodecOps::dequant( digits_, a_, n_frames, cfg_.levels.data() );
	conv( dec_pre_, a_, x_, n_frames );
	int C = cfg_.dec_channels, T = n_frames;
	for ( const DecStage& st : dec_ ) {
		act( x_, x_, st.alpha, C, T );
		up_conv( st.up, x_, acc_, T );
		std::swap( x_, acc_ );
		C /= 2;
		T *= st.up.s;
		res_layer( st.res, C, T );
	}
	act( x_, x_, dec_post_alpha_, C, T );
	conv( dec_post_, x_, a_, T );
	CodecOps::clamp( a_, static_cast<size_t>( T ) );
	backend::check_last_error();
	std::vector<float> out( T );
	backend::memcpy_sync( out.data(), a_, static_cast<size_t>( T ) * 4, Copy::D2H );
	return out;
}

std::vector<uint8_t> NemoCodec::encode( const std::vector<float>& audio ) {
	using backend::Copy;
	const int spf = cfg_.samples_per_frame;
	const int n_frames = static_cast<int>( ( audio.size() + spf - 1 ) / spf );
	ENGINE_ASSERT( n_frames >= 4, "codec: at least 4 frames of audio" );
	if ( !reserve( n_frames ) ) { return {}; }
	int T = n_frames * spf, C = cfg_.enc_channels;
	backend::memset_sync( a_, 0, static_cast<size_t>( T ) * 4 );   // zero-padded to whole frames
	backend::memcpy_sync( a_, audio.data(), audio.size() * 4, Copy::H2D );
	conv( enc_pre_, a_, x_, T );
	for ( const EncStage& st : enc_ ) {
		res_layer( st.res, C, T );
		act( x_, x_, nullptr, C, T );
		T = conv( st.down, x_, acc_, T );
		std::swap( x_, acc_ );
		C *= 2;
	}
	ENGINE_ASSERT( T == n_frames, "codec: the encoder's frame count" );
	act( x_, x_, nullptr, C, T );
	conv( enc_post_, x_, a_, T );
	CodecOps::quant( a_, digits_, T, cfg_.levels.data(), cfg_.fsq_eps );
	backend::check_last_error();
	std::vector<uint8_t> out( static_cast<size_t>( T ) * 32 );
	backend::memcpy_sync( out.data(), digits_, out.size(), Copy::D2H );
	return out;
}

}  // namespace engine::tts
