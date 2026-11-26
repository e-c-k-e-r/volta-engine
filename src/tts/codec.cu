// The NeMo codec's HiFi-GAN decoder (vall_e/emb/nemo.py HiFiGANDecoder), FP32. A convolution is GEMMs over chunks of
// time: the chunk's columns (each input channel's k taps, reflection-padded and dilated) against the weight [co, ci * k].
// A transposed convolution with stride s and kernel 2s gives output o = q s + r (phase r) from two input frames, so its
// weight is regrouped at load by phase and input shift: [s * co, 3 * ci] against columns of the input at q - 1, q, q + 1
// (zeros past the ends), then the phases are interleaved into time.
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

__global__ void k_dequant( const uint8_t* digits, float* x, int T, int dims, int4 hl ) {
	// [T, groups * dims] level indices -> [groups * dims, T] values (index - levels / 2) / (levels / 2), as fsq.digits_to_values
	const int C = 32;
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < C * T; i += gridDim.x * blockDim.x ) {
		const int c = i / T, t = i % T, j = c % dims;
		const int h = j == 0 ? hl.x : j == 1 ? hl.y : j == 2 ? hl.z : hl.w;
		x[i] = ( static_cast<float>( digits[static_cast<size_t>( t ) * C + c] ) - h ) / h;
	}
}

__global__ void k_half_snake( const float* x, float* y, const float* alpha, int C, int T ) {
	// Snake x + sin(a x)^2 / (a + 1e-9) on the first C / 2 channels, leaky ReLU (0.01) on the rest
	const size_t n = static_cast<size_t>( C ) * T;
	const int sc = C / 2;
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

__global__ void k_cols( const float* x, float* cols, int ci, int k, int dil, int pad, int T, int t0, int n ) {
	// cols [ci * k, n]: x[c, reflect(t0 + t + j dil - pad)]
	const size_t total = static_cast<size_t>( ci ) * k * n;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < total; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int t = static_cast<int>( i % n ), row = static_cast<int>( i / n ), c = row / k, j = row % k;
		int s = t0 + t + j * dil - pad;
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

int CodecConfig::samples_per_frame() const {
	int s = 1;
	for ( int r : up_rates ) { s *= r; }
	return s;
}

CodecDecoder::~CodecDecoder() {
	for ( float* p : { x_, cur_, a_, b_, acc_, cols_, ph_ } ) { if ( p ) { cudaFree( p ); } }
	if ( digits_ ) { cudaFree( digits_ ); }
	engine::loader::safetensors_close( st_ );
	if ( arena_.base_ptr ) { engine::core::arena_destroy( arena_ ); }
	if ( blas_ ) { cublasDestroy( static_cast<cublasHandle_t>( blas_ ) ); }
}

bool CodecDecoder::load( const std::string& path, int device ) {
	if ( !engine::loader::safetensors_open( st_, path.c_str() ) ) {
		LOG_ERROR( "codec: can't open %s", path.c_str() );
		return false;
	}
	const std::string* format = engine::loader::safetensors_metadata( st_, "format" );
	const std::string* config = engine::loader::safetensors_metadata( st_, "config" );
	if ( !format || *format != "nemo-codec-decoder" || !config ) {
		LOG_ERROR( "codec: %s isn't a codec decoder export (python -m vall_e.export codec)", path.c_str() );
		return false;
	}
	try {
		auto c = nlohmann::json::parse( *config );
		cfg_.sample_rate = c.at( "sample_rate" );
		cfg_.input_dim = c.at( "input_dim" );
		cfg_.base_channels = c.at( "base_channels" );
		cfg_.in_kernel = c.at( "in_kernel_size" );
		cfg_.out_kernel = c.at( "out_kernel_size" );
		cfg_.up_rates = c.at( "up_sample_rates" ).get<std::vector<int>>();
		cfg_.res_kernels = c.at( "resblock_kernel_sizes" ).get<std::vector<int>>();
		cfg_.res_dilations = c.at( "resblock_dilation_sizes" ).get<std::vector<int>>();
		cfg_.levels = c.at( "levels" ).get<std::vector<int>>();
		cfg_.groups = c.at( "groups" );
		if ( c.at( "activation" ) != "half_snake" || c.at( "output_activation" ) != "ClampActivation" ) {
			LOG_ERROR( "codec: only the HalfSnake / clamp decoder is supported" );
			return false;
		}
	} catch ( const std::exception& e ) {
		LOG_ERROR( "codec: bad config metadata: %s", e.what() );
		return false;
	}
	if ( cfg_.input_dim != 32 || cfg_.levels.size() != 4 || cfg_.groups * 4 != 32 ) {
		LOG_ERROR( "codec: expected 8 FSQ groups of 4 dimensions" );
		return false;
	}
	// the regrouped transposed-conv weights are 3 / 2 the size of the stored ones
	if ( !engine::core::arena_create( arena_, device, st_.file_size * 2 + ( 64ull << 20 ) ) ) {
		LOG_ERROR( "codec: can't allocate the weight arena" );
		return false;
	}
	engine::loader::DeviceUploader u( st_, arena_, "codec" );
	bool ok = true;
	auto F32 = [&]( const std::string& n ) { const float* p = u.f32( n ); ok &= p != nullptr; return p; };
	auto conv = [&]( const std::string& p, int co, int ci, int k, int dil ) {
		Conv c;
		c.w = F32( p + ".conv.weight" );
		c.b = F32( p + ".conv.bias" );
		c.co = co; c.ci = ci; c.k = k; c.dil = dil;
		return c;
	};
	pre_ = conv( "pre_conv", cfg_.base_channels, cfg_.input_dim, cfg_.in_kernel, 1 );
	int C = cfg_.base_channels;
	for ( size_t i = 0; i < cfg_.up_rates.size(); ++i ) {
		const std::string p = std::to_string( i );
		Stage st;
		st.alpha = F32( "activations." + p + ".activation.snake_act.alpha" );
		const int s = cfg_.up_rates[i], co = C / 2;
		// ConvTranspose1d [C, co, 2s], padding (s + 1) / 2: output q s + r takes taps jh = (r + pad) % s from input q + d
		// (d = (r + pad) / s) and jh + s from input q + d - 1
		const std::string wn = "up_sample_conv_layers." + p + ".conv.weight";
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
		st.up.b = F32( "up_sample_conv_layers." + p + ".conv.bias" );
		st.up.co = co; st.up.ci = C; st.up.s = s;
		C = co;
		for ( size_t b = 0; b < cfg_.res_kernels.size(); ++b ) {
			std::vector<ResBlock> chain;
			for ( size_t j = 0; j < cfg_.res_dilations.size(); ++j ) {
				const std::string q = "res_layers." + p + ".res_blocks." + std::to_string( b ) + ".res_blocks." + std::to_string( j );
				ResBlock rb;
				rb.in = conv( q + ".input_conv", C, C, cfg_.res_kernels[b], cfg_.res_dilations[j] );
				rb.skip = conv( q + ".skip_conv", C, C, cfg_.res_kernels[b], 1 );
				rb.alpha_in = F32( q + ".input_activation.activation.snake_act.alpha" );
				rb.alpha_skip = F32( q + ".skip_activation.activation.snake_act.alpha" );
				chain.push_back( rb );
			}
			st.branches.push_back( chain );
		}
		stages_.push_back( st );
	}
	post_alpha_ = F32( "post_activation.activation.snake_act.alpha" );
	post_ = conv( "post_conv", 1, C, cfg_.out_kernel, 1 );
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

void CodecDecoder::reserve( int n_frames ) {
	if ( n_frames <= frames_ ) { return; }
	for ( float** p : { &x_, &cur_, &a_, &b_, &acc_ } ) { if ( *p ) { cudaFree( *p ); *p = nullptr; } }
	if ( digits_ ) { cudaFree( digits_ ); }
	// the largest [C, T] of any point of the decoder
	size_t most = static_cast<size_t>( cfg_.base_channels ) * n_frames;
	size_t T = n_frames;
	int C = cfg_.base_channels;
	for ( int s : cfg_.up_rates ) { T *= s; C /= 2; most = std::max( most, static_cast<size_t>( C ) * T ); }
	for ( float** p : { &x_, &cur_, &a_, &b_, &acc_ } ) { CUDA_CHECK( cudaMalloc( p, most * 4 ) ); }
	CUDA_CHECK( cudaMalloc( &digits_, static_cast<size_t>( n_frames ) * 32 ) );
	act_elems_ = most;
	frames_ = n_frames;
}

void CodecDecoder::conv( const Conv& c, const float* x, float* y, int T ) {
	const int K = c.ci * c.k, pad = ( c.k - 1 ) / 2 * c.dil;
	ENGINE_ASSERT( pad < T, "codec: input shorter than a convolution's reflection padding" );
	const int n = static_cast<int>( std::max<size_t>( 1, std::min<size_t>( T, CHUNK / K ) ) );
	const float one = 1.0f, zero = 0.0f;
	for ( int t0 = 0; t0 < T; t0 += n ) {
		const int m = std::min( n, T - t0 );
		k_cols<<<grid( static_cast<size_t>( K ) * m ), 256>>>( x, cols_, c.ci, c.k, c.dil, pad, T, t0, m );
		// row-major Y[co, T] at column t0 = column-major [m, co] with ld T: cols^T [m, K] x W^T [K, co]
		CUBLAS_CHECK( cublasSgemm( static_cast<cublasHandle_t>( blas_ ), CUBLAS_OP_N, CUBLAS_OP_N, m, c.co, K, &one, cols_, m, c.w, K, &zero, y + t0, T ) );
	}
	k_add_bias<<<grid( static_cast<size_t>( c.co ) * T ), 256>>>( y, c.b, c.co, T );
}

void CodecDecoder::up_conv( const UpConv& u, const float* x, float* y, int T_in ) {
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

void CodecDecoder::half_snake( float* x, const float* alpha, int C, int T ) {
	k_half_snake<<<grid( static_cast<size_t>( C ) * T ), 256>>>( x, x, alpha, C, T );
}

std::vector<float> CodecDecoder::decode( const std::vector<uint8_t>& digits, int n_frames ) {
	ENGINE_ASSERT( static_cast<int>( digits.size() ) >= n_frames * 32, "codec: fewer digits than frames" );
	ENGINE_ASSERT( n_frames >= 4, "codec: at least 4 frames" );
	reserve( n_frames );
	CUDA_CHECK( cudaMemcpy( digits_, digits.data(), static_cast<size_t>( n_frames ) * 32, cudaMemcpyHostToDevice ) );
	const int4 hl = make_int4( cfg_.levels[0] / 2, cfg_.levels[1] / 2, cfg_.levels[2] / 2, cfg_.levels[3] / 2 );
	k_dequant<<<grid( static_cast<size_t>( 32 ) * n_frames ), 256>>>( digits_, a_, n_frames, 4, hl );
	conv( pre_, a_, x_, n_frames );
	int C = cfg_.base_channels, T = n_frames;
	for ( const Stage& st : stages_ ) {
		half_snake( x_, st.alpha, C, T );
		up_conv( st.up, x_, acc_, T );
		std::swap( x_, acc_ );
		C /= 2;
		T *= st.up.s;
		const size_t n = static_cast<size_t>( C ) * T;
		// the residual layer: the mean of its branches, each a chain of x + skip(act(in(act(x)))) from the layer's input
		for ( size_t b = 0; b < st.branches.size(); ++b ) {
			CUDA_CHECK( cudaMemcpy( cur_, x_, n * 4, cudaMemcpyDeviceToDevice ) );
			for ( const ResBlock& rb : st.branches[b] ) {
				k_half_snake<<<grid( n ), 256>>>( cur_, a_, rb.alpha_in, C, T );
				conv( rb.in, a_, b_, T );
				half_snake( b_, rb.alpha_skip, C, T );
				conv( rb.skip, b_, a_, T );
				k_axpy<<<grid( n ), 256>>>( cur_, a_, 1.0f, 1.0f, n );
			}
			if ( b == 0 ) { CUDA_CHECK( cudaMemcpy( acc_, cur_, n * 4, cudaMemcpyDeviceToDevice ) ); }
			else { k_axpy<<<grid( n ), 256>>>( acc_, cur_, 1.0f, 1.0f, n ); }
		}
		k_axpy<<<grid( n ), 256>>>( x_, acc_, 0.0f, 1.0f / static_cast<float>( st.branches.size() ), n );
	}
	half_snake( x_, post_alpha_, C, T );
	conv( post_, x_, a_, T );
	k_clamp<<<grid( static_cast<size_t>( T ) ), 256>>>( a_, static_cast<size_t>( T ) );
	CUDA_CHECK( cudaGetLastError() );
	std::vector<float> out( T );
	CUDA_CHECK( cudaMemcpy( out.data(), a_, static_cast<size_t>( T ) * 4, cudaMemcpyDeviceToHost ) );
	return out;
}

}  // namespace engine::tts
