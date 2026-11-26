// The NeMo codec (vall_e/emb/nemo.py HiFiGANEncoder / GroupFiniteScalarQuantizer / HiFiGANDecoder), FP32. A convolution
// is GEMMs over chunks of time: the chunk's columns (each input channel's k taps, strided, dilated, reflection-padded)
// against the weight [co, ci * k]. A transposed convolution with stride s and kernel 2s gives output o = q s + r (phase
// r) from two input frames, so its weight is regrouped at load by phase and input shift: [s * co, 3 * ci] against columns
// of the input at q - 1, q, q + 1 (zeros past the ends), then the phases are interleaved into time.
#include "engine/tts/codec.h"
#include "engine/core/blas_check.h"
#include "engine/core/log.h"
#include "engine/loader/device_upload.h"
#include <cublas_v2.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>

namespace engine::tts {

namespace {

constexpr size_t CHUNK = 16u << 20;  // floats of a GEMM chunk's columns, and of its phase outputs

__global__ void k_dequant( const uint8_t* digits, float* x, int T, int4 hl ) {
	// [T, 8 * 4] level indices -> [32, T] values (index - levels / 2) / (levels / 2), as fsq.digits_to_values
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < 32 * T; i += gridDim.x * blockDim.x ) {
		const int c = i / T, t = i % T, j = c % 4;
		const int h = j == 0 ? hl.x : j == 1 ? hl.y : j == 2 ? hl.z : hl.w;
		x[i] = ( static_cast<float>( digits[static_cast<size_t>( t ) * 32 + c] ) - h ) / h;
	}
}

__global__ void k_quant( const float* x, uint8_t* digits, int T, int4 levels, float eps ) {
	// FiniteScalarQuantizer.encode per dimension: round(s tanh(x + tan(offset / s)) - offset) + levels / 2, with
	// s = (levels - 1) / 2 (1 - eps) and offset 0.5 for an even number of levels, rounding half to even (torch.round)
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < 32 * T; i += gridDim.x * blockDim.x ) {
		const int c = i / T, t = i % T, j = c % 4;
		const int L = j == 0 ? levels.x : j == 1 ? levels.y : j == 2 ? levels.z : levels.w;
		const float scale = ( L - 1 ) * 0.5f * ( 1.0f - eps ), offset = L % 2 == 0 ? 0.5f : 0.0f, shift = tanf( offset / scale );
		const float code = rintf( scale * tanhf( x[i] + shift ) - offset );
		digits[static_cast<size_t>( t ) * 32 + c] = static_cast<uint8_t>( static_cast<int>( code ) + L / 2 );
	}
}

__global__ void k_act( const float* x, float* y, const float* alpha, int C, int T ) {
	// HalfSnake: Snake x + sin(a x)^2 / (a + 1e-9) on the first C / 2 channels, leaky ReLU (0.01) on the rest; alpha null:
	// leaky ReLU everywhere
	const size_t n = static_cast<size_t>( C ) * T;
	const int sc = alpha ? C / 2 : 0;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int c = static_cast<int>( i / T );
		const float v = x[i];
		if ( c < sc ) {
			const float a = alpha[c], s = sinf( a * v );
			y[i] = v + s * s / ( a + 1e-9f );
		} else {
			y[i] = v > 0.0f ? v : 0.01f * v;
		}
	}
}

__global__ void k_cols( const float* x, float* cols, int ci, int k, int dil, int stride, int pad, int T, int t0, int n ) {
	// cols [ci * k, n]: x[c, reflect((t0 + t) stride + j dil - pad)]
	const size_t total = static_cast<size_t>( ci ) * k * n;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < total; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int t = static_cast<int>( i % n ), row = static_cast<int>( i / n ), c = row / k, j = row % k;
		int s = ( t0 + t ) * stride + j * dil - pad;
		s = s < 0 ? -s : s;
		s = s >= T ? 2 * ( T - 1 ) - s : s;
		cols[i] = x[static_cast<size_t>( c ) * T + s];
	}
}

__global__ void k_up_cols( const float* x, float* cols, int ci, int T, int q0, int n ) {
	// cols [3 * ci, n]: x[c, q0 + q + shift], shift -1 / 0 / +1, zero past the ends
	const size_t total = static_cast<size_t>( 3 ) * ci * n;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < total; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int q = static_cast<int>( i % n ), row = static_cast<int>( i / n ), sh = row / ci - 1, c = row % ci;
		const int s = q0 + q + sh;
		cols[i] = s >= 0 && s < T ? x[static_cast<size_t>( c ) * T + s] : 0.0f;
	}
}

__global__ void k_interleave( const float* ph, const float* bias, float* y, int co, int s, int T_out, int q0, int n ) {
	// ph [s * co, n] (phase r of channel c at row r co + c) -> y[c, (q0 + q) s + r] + bias
	const size_t total = static_cast<size_t>( s ) * co * n;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < total; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int q = static_cast<int>( i % n ), row = static_cast<int>( i / n ), r = row / co, c = row % co;
		y[static_cast<size_t>( c ) * T_out + static_cast<size_t>( q0 + q ) * s + r] = ph[i] + bias[c];
	}
}

__global__ void k_add_bias( float* y, const float* b, int C, int T ) {
	const size_t n = static_cast<size_t>( C ) * T;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] += b[i / T];
	}
}

__global__ void k_axpy( float* y, const float* x, float a, float b, size_t n ) {  // y = a y + b x
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] = a * y[i] + b * x[i];
	}
}

__global__ void k_clamp( float* y, size_t n ) {
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] = fminf( 1.0f, fmaxf( -1.0f, y[i] ) );
	}
}

int grid( size_t n ) { return static_cast<int>( std::min<size_t>( 4096, ( n + 255 ) / 256 ) ); }

}  // namespace

NemoCodec::~NemoCodec() {
	for ( float* p : { x_, cur_, a_, b_, acc_, cols_, ph_ } ) { if ( p ) { cudaFree( p ); } }
	if ( digits_ ) { cudaFree( digits_ ); }
	engine::loader::safetensors_close( st_ );
	if ( arena_.base_ptr ) { engine::core::arena_destroy( arena_ ); }
	if ( blas_ ) { cublasDestroy( static_cast<cublasHandle_t>( blas_ ) ); }
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
		CUDA_CHECK( cudaMemcpy( dev, big.data(), big.size() * 4, cudaMemcpyHostToDevice ) );
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
	cublasHandle_t h;
	CUBLAS_CHECK( cublasCreate( &h ) );
	blas_ = h;
	CUDA_CHECK( cudaMalloc( &cols_, CHUNK * 4 ) );
	CUDA_CHECK( cudaMalloc( &ph_, CHUNK * 4 ) );
	return true;
}

bool NemoCodec::reserve( int n_frames ) {
	if ( n_frames <= frames_ ) { return true; }
	for ( float** p : { &x_, &cur_, &a_, &b_, &acc_ } ) { if ( *p ) { cudaFree( *p ); *p = nullptr; } }
	if ( digits_ ) { cudaFree( digits_ ); digits_ = nullptr; }
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
	for ( float** p : { &x_, &cur_, &a_, &b_, &acc_ } ) { ok = ok && cudaMalloc( p, most * 4 ) == cudaSuccess; }
	ok = ok && cudaMalloc( &digits_, static_cast<size_t>( n_frames ) * 32 ) == cudaSuccess;
	if ( !ok ) {
		cudaGetLastError();   // an allocation failure isn't sticky: cleared so later checks don't report it
		for ( float** p : { &x_, &cur_, &a_, &b_, &acc_ } ) { if ( *p ) { cudaFree( *p ); *p = nullptr; } }
		if ( digits_ ) { cudaFree( digits_ ); digits_ = nullptr; }
		LOG_ERROR( "codec: no GPU memory for %d frames (%.2f GiB of activations)", n_frames, ( most * 20.0 + n_frames * 32.0 ) / 1073741824.0 );
		return false;
	}
	frames_ = n_frames;
	return true;
}

int NemoCodec::conv( const Conv& c, const float* x, float* y, int T ) {
	const int K = c.ci * c.k, T_out = ( T + 2 * c.pad - c.dil * ( c.k - 1 ) - 1 ) / c.stride + 1;
	ENGINE_ASSERT( c.pad < T, "codec: input shorter than a convolution's reflection padding" );
	const int n = static_cast<int>( std::max<size_t>( 1, std::min<size_t>( T_out, CHUNK / K ) ) );
	const float one = 1.0f, zero = 0.0f;
	for ( int t0 = 0; t0 < T_out; t0 += n ) {
		const int m = std::min( n, T_out - t0 );
		k_cols<<<grid( static_cast<size_t>( K ) * m ), 256>>>( x, cols_, c.ci, c.k, c.dil, c.stride, c.pad, T, t0, m );
		// row-major Y[co, T_out] at column t0 = column-major [m, co] with ld T_out: cols^T [m, K] x W^T [K, co]
		CUBLAS_CHECK( cublasSgemm( static_cast<cublasHandle_t>( blas_ ), CUBLAS_OP_N, CUBLAS_OP_N, m, c.co, K, &one, cols_, m, c.w, K, &zero, y + t0, T_out ) );
	}
	k_add_bias<<<grid( static_cast<size_t>( c.co ) * T_out ), 256>>>( y, c.b, c.co, T_out );
	return T_out;
}

void NemoCodec::up_conv( const UpConv& u, const float* x, float* y, int T_in ) {
	const int K = 3 * u.ci, rows = u.s * u.co;
	const int n = static_cast<int>( std::max<size_t>( 1, std::min<size_t>( { static_cast<size_t>( T_in ), CHUNK / K, CHUNK / rows } ) ) );
	const float one = 1.0f, zero = 0.0f;
	for ( int q0 = 0; q0 < T_in; q0 += n ) {
		const int m = std::min( n, T_in - q0 );
		k_up_cols<<<grid( static_cast<size_t>( K ) * m ), 256>>>( x, cols_, u.ci, T_in, q0, m );
		CUBLAS_CHECK( cublasSgemm( static_cast<cublasHandle_t>( blas_ ), CUBLAS_OP_N, CUBLAS_OP_N, m, rows, K, &one, cols_, m, u.w, K, &zero, ph_, m ) );
		k_interleave<<<grid( static_cast<size_t>( rows ) * m ), 256>>>( ph_, u.b, y, u.co, u.s, T_in * u.s, q0, m );
	}
}

void NemoCodec::act( const float* x, float* y, const float* alpha, int C, int T ) {
	k_act<<<grid( static_cast<size_t>( C ) * T ), 256>>>( x, y, alpha, C, T );
}

void NemoCodec::res_layer( const ResLayer& r, int C, int T ) {
	// the mean of the branches, each a chain of x + skip(act(in(act(x)))) from the layer's input
	const size_t n = static_cast<size_t>( C ) * T;
	for ( size_t b = 0; b < r.size(); ++b ) {
		CUDA_CHECK( cudaMemcpy( cur_, x_, n * 4, cudaMemcpyDeviceToDevice ) );
		for ( const ResBlock& rb : r[b] ) {
			act( cur_, a_, rb.alpha_in, C, T );
			conv( rb.in, a_, b_, T );
			act( b_, b_, rb.alpha_skip, C, T );
			conv( rb.skip, b_, a_, T );
			k_axpy<<<grid( n ), 256>>>( cur_, a_, 1.0f, 1.0f, n );
		}
		if ( b == 0 ) { CUDA_CHECK( cudaMemcpy( acc_, cur_, n * 4, cudaMemcpyDeviceToDevice ) ); }
		else { k_axpy<<<grid( n ), 256>>>( acc_, cur_, 1.0f, 1.0f, n ); }
	}
	k_axpy<<<grid( n ), 256>>>( x_, acc_, 0.0f, 1.0f / static_cast<float>( r.size() ), n );
}

std::vector<float> NemoCodec::decode( const std::vector<uint8_t>& digits, int n_frames ) {
	ENGINE_ASSERT( static_cast<int>( digits.size() ) >= n_frames * 32, "codec: fewer digits than frames" );
	ENGINE_ASSERT( n_frames >= 4, "codec: at least 4 frames" );
	if ( !reserve( n_frames ) ) { return {}; }
	CUDA_CHECK( cudaMemcpy( digits_, digits.data(), static_cast<size_t>( n_frames ) * 32, cudaMemcpyHostToDevice ) );
	const int4 hl = make_int4( cfg_.levels[0] / 2, cfg_.levels[1] / 2, cfg_.levels[2] / 2, cfg_.levels[3] / 2 );
	k_dequant<<<grid( static_cast<size_t>( 32 ) * n_frames ), 256>>>( digits_, a_, n_frames, hl );
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
	k_clamp<<<grid( static_cast<size_t>( T ) ), 256>>>( a_, static_cast<size_t>( T ) );
	CUDA_CHECK( cudaGetLastError() );
	std::vector<float> out( T );
	CUDA_CHECK( cudaMemcpy( out.data(), a_, static_cast<size_t>( T ) * 4, cudaMemcpyDeviceToHost ) );
	return out;
}

std::vector<uint8_t> NemoCodec::encode( const std::vector<float>& audio ) {
	const int spf = cfg_.samples_per_frame;
	const int n_frames = static_cast<int>( ( audio.size() + spf - 1 ) / spf );
	ENGINE_ASSERT( n_frames >= 4, "codec: at least 4 frames of audio" );
	if ( !reserve( n_frames ) ) { return {}; }
	int T = n_frames * spf, C = cfg_.enc_channels;
	CUDA_CHECK( cudaMemset( a_, 0, static_cast<size_t>( T ) * 4 ) );   // zero-padded to whole frames
	CUDA_CHECK( cudaMemcpy( a_, audio.data(), audio.size() * 4, cudaMemcpyHostToDevice ) );
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
	const int4 lv = make_int4( cfg_.levels[0], cfg_.levels[1], cfg_.levels[2], cfg_.levels[3] );
	k_quant<<<grid( static_cast<size_t>( 32 ) * T ), 256>>>( a_, digits_, T, lv, cfg_.fsq_eps );
	CUDA_CHECK( cudaGetLastError() );
	std::vector<uint8_t> out( static_cast<size_t>( T ) * 32 );
	CUDA_CHECK( cudaMemcpy( out.data(), digits_, out.size(), cudaMemcpyDeviceToHost ) );
	return out;
}

}  // namespace engine::tts
