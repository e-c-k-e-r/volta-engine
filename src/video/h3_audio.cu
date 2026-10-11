// MiniMax-H3 audio VAE decoder: dec_in_proj (1x1) -> BigVGAN (7 transposed-conv upsamplers, 3 AMP resblocks each).
#include "engine/video/h3_audio.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

namespace engine::video {
namespace {

constexpr int kUp[7] = { 5, 5, 2, 2, 2, 2, 2 };
constexpr int kUpK[7] = { 9, 9, 4, 4, 4, 4, 4 };
constexpr int kResK[3] = { 3, 7, 11 };
constexpr int kDil[3] = { 1, 3, 5 };
constexpr int kFilter = 12;

// w = g * v / ||v|| per leading row
__global__ void k_wn_fold( const float* v, const float* g, float* w, int rowlen ) {
	__shared__ float red[32];
	const int r = blockIdx.x;
	float s = 0.0f;
	for ( int i = threadIdx.x; i < rowlen; i += blockDim.x ) { s += v[static_cast<size_t>( r ) * rowlen + i] * v[static_cast<size_t>( r ) * rowlen + i]; }
	for ( int d = 16; d; d >>= 1 ) { s += __shfl_xor_sync( 0xffffffffu, s, d ); }
	if ( ( threadIdx.x & 31 ) == 0 ) { red[threadIdx.x >> 5] = s; }
	__syncthreads();
	float t = 0.0f;
	for ( int i = 0; i < static_cast<int>( blockDim.x >> 5 ); ++i ) { t += red[i]; }
	const float scale = g[r] / sqrtf( t );
	for ( int i = threadIdx.x; i < rowlen; i += blockDim.x ) { w[static_cast<size_t>( r ) * rowlen + i] = v[static_cast<size_t>( r ) * rowlen + i] * scale; }
}

__global__ void k_exp( const float* a, float* o, int n ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) { o[i] = expf( a[i] ); }
}

// rows [2T, 32] (channel-major) -> z [2, 32, T] denormalized: z = rows * std + mean
__global__ void k_unpack_audio( const float* rows, const float* mean, const float* stdev, float* z, int T ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= static_cast<size_t>( 2 ) * 32 * T ) { return; }
	const int t = static_cast<int>( i % T ), c = static_cast<int>( ( i / T ) % 32 ), b = static_cast<int>( i / ( static_cast<size_t>( T ) * 32 ) );
	z[i] = rows[( static_cast<size_t>( b ) * T + t ) * 32 + c] * stdev[c] + mean[c];
}

// 'same' stride-1 conv: in [B, Ci, T], w [Co, Ci, K]
__global__ void k_conv1d( const float* in, const float* w, const float* bias, float* out, int Ci, int Co, int T, int K, int dil, int pad ) {
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	const int co = blockIdx.y, b = blockIdx.z;
	if ( t >= T ) { return; }
	float acc = bias ? bias[co] : 0.0f;
	const float* ib = in + static_cast<size_t>( b ) * Ci * T;
	const float* wc = w + static_cast<size_t>( co ) * Ci * K;
	for ( int ci = 0; ci < Ci; ++ci ) {
		const float* ic = ib + static_cast<size_t>( ci ) * T;
		const float* wk = wc + ci * K;
		#pragma unroll 4
		for ( int k = 0; k < K; ++k ) {
			const int ti = t + k * dil - pad;
			if ( ti >= 0 && ti < T ) { acc += ic[ti] * wk[k]; }
		}
	}
	out[( static_cast<size_t>( b ) * Co + co ) * T + t] = acc;
}

// ConvTranspose1d: in [B, Ci, T], w [Ci, Co, K], stride u, padding (K - u) / 2 -> out [B, Co, T * u]
__global__ void k_convt1d( const float* in, const float* w, const float* bias, float* out, int Ci, int Co, int T, int K, int u, int pad ) {
	const int Tout = T * u;
	const int to = blockIdx.x * blockDim.x + threadIdx.x;
	const int co = blockIdx.y, b = blockIdx.z;
	if ( to >= Tout ) { return; }
	float acc = bias ? bias[co] : 0.0f;
	const int k0 = ( to + pad ) % u;
	for ( int ci = 0; ci < Ci; ++ci ) {
		const float* ic = in + ( static_cast<size_t>( b ) * Ci + ci ) * T;
		const float* wk = w + ( static_cast<size_t>( ci ) * Co + co ) * K;
		for ( int k = k0; k < K; k += u ) {
			const int i = ( to + pad - k ) / u;
			if ( i >= 0 && i < T ) { acc += ic[i] * wk[k]; }
		}
	}
	out[( static_cast<size_t>( b ) * Co + co ) * Tout + to] = acc;
}

__device__ __forceinline__ float snakebeta( float x, float alpha, float beta ) {
	const float s = sinf( alpha * x );
	return x + s * s / ( beta + 1e-9f );
}

// Activation1d: 2x upsample (replicate pad, transposed conv with the kaiser-sinc filter) -> SnakeBeta -> 2x downsample (replicate pad, strided conv)
// alpha / beta already exponentiated. in/out [B, C, T].
__global__ void k_act1d( const float* in, float* out, const float* alpha, const float* beta, const float* fu, const float* fd, int C, int T ) {
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	const int c = blockIdx.y, b = blockIdx.z;
	if ( t >= T ) { return; }
	const float* x = in + ( static_cast<size_t>( b ) * C + c ) * T;
	const float a = alpha[c], bt = beta[c];
	float acc = 0.0f;
	for ( int k = 0; k < kFilter; ++k ) {
		int j = 2 * t + k - 5; // index into the replicate-padded upsampled signal
		j = min( max( j, 0 ), 2 * T - 1 );
		// up[j] = 2 * sum_i xp[i] * fu[j + 15 - 2 i], xp[i] = x[clamp(i - 5)]
		float up = 0.0f;
		const int imin = ( j + 5 ) / 2, imax = ( j + 15 ) / 2;
		for ( int i = imin; i <= imax; ++i ) {
			const int fi = j + 15 - 2 * i;
			if ( fi < 0 || fi >= kFilter ) { continue; }
			up += x[min( max( i - 5, 0 ), T - 1 )] * fu[fi];
		}
		acc += fd[k] * snakebeta( 2.0f * up, a, bt );
	}
	out[( static_cast<size_t>( b ) * C + c ) * T + t] = acc;
}

__global__ void k_add( float* x, const float* y, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) { x[i] += y[i]; }
}

__global__ void k_scale( float* x, float s, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) { x[i] *= s; }
}

__global__ void k_clamp( float* x, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) { x[i] = fminf( fmaxf( x[i], -1.0f ), 1.0f ); }
}

struct Act {
	const float *alpha = nullptr, *beta = nullptr, *fu = nullptr, *fd = nullptr;
};

struct ResBlock {
	int k = 0, ch = 0;
	const float* c1[3];
	const float* b1[3];
	const float* c2[3];
	const float* b2[3];
	Act act[6];
};

} // namespace

struct H3AudioVAE::Impl {
	int device = 0, max_frames = 0;
	pod::SafeTensorsFile st{};
	pod::MemoryArena warena{}, sarena{};
	std::vector<float> mean, stdev;
	float *d_mean = nullptr, *d_std = nullptr;
	const float *in_w = nullptr, *in_b = nullptr, *pre_w = nullptr, *pre_b = nullptr, *post_w = nullptr;
	const float* up_w[7];
	const float* up_b[7];
	ResBlock rb[21];
	Act post_act;
	float *bufA = nullptr, *bufB = nullptr, *bufC = nullptr, *bufD = nullptr, *acc = nullptr, *z = nullptr;
	cudaStream_t stream = nullptr;

	template <typename T>
	T* alloc( pod::MemoryArena& a, size_t n ) { return static_cast<T*>( engine::core::arena_alloc( a, n * sizeof( T ) ) ); }

	const float* raw( const std::string& name, size_t* n = nullptr ) {
		const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, name.c_str() );
		if ( !m ) { LOG_ERROR( "H3 audio: missing tensor %s", name.c_str() ); return nullptr; }
		ENGINE_ASSERT( m->dtype == pod::DType::FP32, "H3 audio: expected FP32" );
		float* d = static_cast<float*>( engine::core::arena_alloc( warena, m->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( d, engine::loader::safetensors_get_data_cpu( st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
		if ( n ) { *n = m->size_bytes / 4; }
		return d;
	}

	// weight-normalized conv: folds weight_g / weight_v into one weight tensor
	const float* wn( const std::string& base ) {
		size_t nv = 0, ng = 0;
		const float* v = raw( base + ".weight_v", &nv );
		const float* g = raw( base + ".weight_g", &ng );
		if ( !v || !g ) { return nullptr; }
		float* w = alloc<float>( warena, nv );
		k_wn_fold<<<static_cast<int>( ng ), 256>>>( v, g, w, static_cast<int>( nv / ng ) );
		CUDA_CHECK( cudaDeviceSynchronize() );
		return w;
	}

	Act load_act( const std::string& base, int ch ) {
		Act a;
		const float* al = raw( base + ".act.alpha" );
		const float* be = raw( base + ".act.beta" );
		float* ea = alloc<float>( warena, ch );
		float* eb = alloc<float>( warena, ch );
		k_exp<<<( ch + 255 ) / 256, 256>>>( al, ea, ch );
		k_exp<<<( ch + 255 ) / 256, 256>>>( be, eb, ch );
		a.alpha = ea;
		a.beta = eb;
		a.fu = raw( base + ".upsample.filter" );
		a.fd = raw( base + ".downsample.lowpass.filter" );
		return a;
	}

	void conv( const float* in, const float* w, const float* b, float* out, int B, int Ci, int Co, int T, int K, int dil ) {
		const int pad = ( K * dil - dil ) / 2;
		k_conv1d<<<dim3( ( T + 127 ) / 128, Co, B ), 128, 0, stream>>>( in, w, b, out, Ci, Co, T, K, dil, pad );
	}

	void act( const Act& a, const float* in, float* out, int B, int C, int T ) {
		k_act1d<<<dim3( ( T + 127 ) / 128, C, B ), 128, 0, stream>>>( in, out, a.alpha, a.beta, a.fu, a.fd, C, T );
	}
};

H3AudioVAE::H3AudioVAE() : p_( new Impl ) {}

H3AudioVAE::~H3AudioVAE() {
	if ( p_ ) {
		if ( p_->warena.base_ptr ) { engine::core::arena_destroy( p_->warena ); }
		if ( p_->sarena.base_ptr ) { engine::core::arena_destroy( p_->sarena ); }
		engine::loader::safetensors_close( p_->st );
		delete p_;
	}
}

bool H3AudioVAE::load( const char* dir, int device, int max_frames ) {
	Impl& s = *p_;
	s.device = device;
	s.max_frames = max_frames;
	{
		std::ifstream f( std::string( dir ) + "/config.json" );
		std::stringstream ss;
		ss << f.rdbuf();
		const auto j = nlohmann::json::parse( ss.str(), nullptr, false );
		if ( j.is_discarded() || !j.contains( "latents_mean" ) ) { LOG_ERROR( "H3 audio: bad config.json" ); return false; }
		s.mean = j["latents_mean"].get<std::vector<float>>();
		s.stdev = j["latents_std"].get<std::vector<float>>();
	}
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &s.stream, cudaStreamNonBlocking ) );
	if ( !engine::loader::safetensors_open( s.st, ( std::string( dir ) + "/model.safetensors" ).c_str() ) ) { return false; }
	if ( !engine::core::arena_create( s.warena, device, 1ULL << 30 ) ) { return false; }
	s.d_mean = s.alloc<float>( s.warena, 32 );
	s.d_std = s.alloc<float>( s.warena, 32 );
	CUDA_CHECK( cudaMemcpy( s.d_mean, s.mean.data(), 128, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( s.d_std, s.stdev.data(), 128, cudaMemcpyHostToDevice ) );
	s.in_w = s.raw( "dec_in_proj.weight" );
	s.in_b = s.raw( "dec_in_proj.bias" );
	s.pre_w = s.wn( "decoder.conv_pre" );
	s.pre_b = s.raw( "decoder.conv_pre.bias" );
	for ( int i = 0; i < 7; ++i ) {
		s.up_w[i] = s.wn( "decoder.ups." + std::to_string( i ) + ".0" );
		s.up_b[i] = s.raw( "decoder.ups." + std::to_string( i ) + ".0.bias" );
		const int ch = 1024 >> ( i + 1 );
		for ( int j = 0; j < 3; ++j ) {
			ResBlock& r = s.rb[i * 3 + j];
			const std::string p = "decoder.resblocks." + std::to_string( i * 3 + j );
			r.k = kResK[j];
			r.ch = ch;
			for ( int l = 0; l < 3; ++l ) {
				r.c1[l] = s.wn( p + ".convs1." + std::to_string( l ) );
				r.b1[l] = s.raw( p + ".convs1." + std::to_string( l ) + ".bias" );
				r.c2[l] = s.wn( p + ".convs2." + std::to_string( l ) );
				r.b2[l] = s.raw( p + ".convs2." + std::to_string( l ) + ".bias" );
			}
			for ( int a = 0; a < 6; ++a ) { r.act[a] = s.load_act( p + ".activations." + std::to_string( a ), ch ); }
		}
	}
	s.post_act = s.load_act( "decoder.activation_post", 8 );
	s.post_w = s.wn( "decoder.conv_post" );
	if ( !s.post_w ) { return false; }
	const size_t elems = static_cast<size_t>( 2 ) * max_frames * 6400 + 4096;
	if ( !engine::core::arena_create( s.sarena, device, elems * 4 * 6 + ( 64ULL << 20 ) ) ) { return false; }
	s.bufA = s.alloc<float>( s.sarena, elems );
	s.bufB = s.alloc<float>( s.sarena, elems );
	s.bufC = s.alloc<float>( s.sarena, elems );
	s.bufD = s.alloc<float>( s.sarena, elems );
	s.acc = s.alloc<float>( s.sarena, elems );
	s.z = s.alloc<float>( s.sarena, static_cast<size_t>( 2 ) * 2048 * max_frames );
	return true;
}

bool H3AudioVAE::decode( const float* d_rows, int T, std::vector<float>& pcm ) {
	Impl& s = *p_;
	if ( T > s.max_frames ) { return false; }
	const int B = 2;
	cudaStream_t st = s.stream;
	// latent -> denormalized [2, 32, T] in bufC, then dec_in_proj (1x1) -> [2, 2048, T] in s.z
	k_unpack_audio<<<( static_cast<size_t>( B ) * 32 * T + 255 ) / 256, 256, 0, st>>>( d_rows, s.d_mean, s.d_std, s.bufC, T );
	s.conv( s.bufC, s.in_w, s.in_b, s.z, B, 32, 2048, T, 1, 1 );
	float* x = s.bufA;
	s.conv( s.z, s.pre_w, s.pre_b, x, B, 2048, 1024, T, 7, 1 );
	int len = T;
	int ch = 1024;
	float* nxt = s.bufB;
	for ( int i = 0; i < 7; ++i ) {
		const int u = kUp[i], k = kUpK[i], co = ch / 2, pad = ( k - u ) / 2;
		k_convt1d<<<dim3( ( len * u + 127 ) / 128, co, B ), 128, 0, st>>>( x, s.up_w[i], s.up_b[i], nxt, ch, co, len, k, u, pad );
		len *= u;
		ch = co;
		std::swap( x, nxt ); // x: stage input [B, ch, len]; nxt is scratch
		const size_t n = static_cast<size_t>( B ) * ch * len;
		CUDA_CHECK( cudaMemsetAsync( s.acc, 0, n * sizeof( float ), st ) );
		for ( int j = 0; j < 3; ++j ) {
			const ResBlock& r = s.rb[i * 3 + j];
			float* cur = s.bufD; // working copy of x for this resblock
			CUDA_CHECK( cudaMemcpyAsync( cur, x, n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
			float* t1 = nxt;
			float* t2 = s.bufC;
			for ( int l = 0; l < 3; ++l ) {
				s.act( r.act[2 * l], cur, t1, B, ch, len );
				s.conv( t1, r.c1[l], r.b1[l], t2, B, ch, ch, len, r.k, kDil[l] );
				s.act( r.act[2 * l + 1], t2, t1, B, ch, len );
				s.conv( t1, r.c2[l], r.b2[l], t2, B, ch, ch, len, r.k, 1 );
				k_add<<<( n + 255 ) / 256, 256, 0, st>>>( cur, t2, n );
			}
			k_add<<<( n + 255 ) / 256, 256, 0, st>>>( s.acc, cur, n );
		}
		k_scale<<<( n + 255 ) / 256, 256, 0, st>>>( s.acc, 1.0f / 3.0f, n );
		CUDA_CHECK( cudaMemcpyAsync( x, s.acc, n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	}
	// post activation + conv_post (8 -> 1, k 7, no bias), clamp
	const size_t n = static_cast<size_t>( B ) * ch * len;
	s.act( s.post_act, x, nxt, B, ch, len );
	float* wave = s.bufC;
	s.conv( nxt, s.post_w, nullptr, wave, B, ch, 1, len, 7, 1 );
	k_clamp<<<( static_cast<size_t>( B ) * len + 255 ) / 256, 256, 0, st>>>( wave, static_cast<size_t>( B ) * len );
	(void)n;
	pcm.resize( static_cast<size_t>( B ) * len );
	CUDA_CHECK( cudaMemcpyAsync( pcm.data(), wave, pcm.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

} // namespace engine::video
