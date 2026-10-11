// MiniMax-H3 audio VAE decoder: dec_in_proj (1x1) -> BigVGAN (7 transposed-conv upsamplers, 3 AMP resblocks each).
#include "engine/video/h3_audio.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include <nlohmann/json.hpp>
#include <cublas_v2.h>
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


// ---- encoder kernels (reference-audio latents) ----

// strided conv: in [B, Ci, Tin], w [Co, Ci, K], zero padding `pad` on both sides -> out [B, Co, Tout]
__global__ void k_conv1d_s( const float* in, const float* w, const float* bias, float* out, int Ci, int Co, int Tin, int Tout, int K, int dil, int pad, int stride ) {
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	const int co = blockIdx.y, b = blockIdx.z;
	if ( t >= Tout ) { return; }
	float acc = bias ? bias[co] : 0.0f;
	const float* ib = in + static_cast<size_t>( b ) * Ci * Tin;
	const float* wc = w + static_cast<size_t>( co ) * Ci * K;
	for ( int ci = 0; ci < Ci; ++ci ) {
		const float* ic = ib + static_cast<size_t>( ci ) * Tin;
		const float* wk = wc + ci * K;
		for ( int k = 0; k < K; ++k ) {
			const int ti = t * stride + k * dil - pad;
			if ( ti >= 0 && ti < Tin ) { acc += ic[ti] * wk[k]; }
		}
	}
	out[( static_cast<size_t>( b ) * Co + co ) * Tout + t] = acc;
}

// Snake1d in place: x + sin^2(alpha x) / (alpha + 1e-9), x [B, C, T]
__global__ void k_snake( float* x, const float* alpha, int C, int T, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= n ) { return; }
	const float a = alpha[( i / T ) % C];
	const float s = sinf( a * x[i] );
	x[i] += s * s / ( a + 1e-9f );
}

// [B, C, T] -> [B, T, C]
__global__ void k_transpose_ct( const float* in, float* out, int C, int T, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= n ) { return; }
	const int c = static_cast<int>( i % C ), t = static_cast<int>( ( i / C ) % T );
	const size_t b = i / ( static_cast<size_t>( C ) * T );
	out[i] = in[( b * C + c ) * T + t];
}

// LayerNorm over the last dim C of [rows, C] (eps 1e-5), one block per row
__global__ void k_layernorm( const float* x, const float* w, const float* b, float* y, int C ) {
	__shared__ float red[64];
	const float* xr = x + static_cast<size_t>( blockIdx.x ) * C;
	float s = 0.0f, q = 0.0f;
	for ( int i = threadIdx.x; i < C; i += blockDim.x ) { s += xr[i]; q += xr[i] * xr[i]; }
	for ( int d = 16; d; d >>= 1 ) { s += __shfl_xor_sync( 0xffffffffu, s, d ); q += __shfl_xor_sync( 0xffffffffu, q, d ); }
	if ( ( threadIdx.x & 31 ) == 0 ) { red[threadIdx.x >> 5] = s; red[32 + ( threadIdx.x >> 5 )] = q; }
	__syncthreads();
	float ts = 0.0f, tq = 0.0f;
	for ( int i = 0; i < static_cast<int>( blockDim.x >> 5 ); ++i ) { ts += red[i]; tq += red[32 + i]; }
	const float mean = ts / C, var = tq / C - mean * mean;
	const float inv = rsqrtf( var + 1e-5f );
	for ( int i = threadIdx.x; i < C; i += blockDim.x ) { y[static_cast<size_t>( blockIdx.x ) * C + i] = ( xr[i] - mean ) * inv * w[i] + b[i]; }
}

// y[r, o] += bias[o]
__global__ void k_bias( float* y, const float* bias, int N, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) { y[i] += bias[i % N]; }
}

// causal attention, qkv [B, T, 3, H, hd] -> o [B, H, T, hd]; one block per (b, h, query)
__global__ void k_attn_causal( const float* qkv, float* o, int T, int H, int hd, float scale ) {
	extern __shared__ float sc[];
	const int q = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
	const size_t row = static_cast<size_t>( 3 ) * H * hd;
	const float* base = qkv + static_cast<size_t>( b ) * T * row;
	const float* qv = base + static_cast<size_t>( q ) * row + static_cast<size_t>( h ) * hd;
	for ( int j = threadIdx.x; j <= q; j += blockDim.x ) {
		const float* kv = base + static_cast<size_t>( j ) * row + static_cast<size_t>( H ) * hd + static_cast<size_t>( h ) * hd;
		float d = 0.0f;
		for ( int i = 0; i < hd; ++i ) { d += qv[i] * kv[i]; }
		sc[j] = d * scale;
	}
	__syncthreads();
	float m = -1e30f;
	for ( int j = 0; j <= q; ++j ) { m = fmaxf( m, sc[j] ); }
	float sum = 0.0f;
	for ( int j = 0; j <= q; ++j ) { sum += __expf( sc[j] - m ); }
	for ( int d = threadIdx.x; d < hd; d += blockDim.x ) {
		float acc = 0.0f;
		for ( int j = 0; j <= q; ++j ) { acc += __expf( sc[j] - m ) * base[static_cast<size_t>( j ) * row + 2 * static_cast<size_t>( H ) * hd + static_cast<size_t>( h ) * hd + d]; }
		o[( ( static_cast<size_t>( b ) * H + h ) * T + q ) * hd + d] = acc / sum;
	}
}

// mean over heads, then adaptive average pool hd -> P: pooled[b, t, g] = mean over h and d in group g
__global__ void k_head_pool( const float* o, float* pooled, int T, int H, int hd, int P, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= n ) { return; }
	const int g = static_cast<int>( i % P ), t = static_cast<int>( ( i / P ) % T );
	const size_t b = i / ( static_cast<size_t>( P ) * T );
	const int per = hd / P;
	float acc = 0.0f;
	for ( int h = 0; h < H; ++h ) {
		const float* r = o + ( ( b * H + h ) * T + t ) * hd + g * per;
		for ( int d = 0; d < per; ++d ) { acc += r[d]; }
	}
	pooled[i] = acc / ( H * per );
}

__global__ void k_add2( float* y, const float* a, const float* b, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) { y[i] = a[i] + b[i]; }
}

// GeGLU: g = gelu_tanh(h0) * h1; h0/h1 [rows, F]
__global__ void k_geglu( const float* h0, const float* h1, float* g, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= n ) { return; }
	const float x = h0[i];
	const float gelu = 0.5f * x * ( 1.0f + tanhf( 0.7978845608f * ( x + 0.044715f * x * x * x ) ) );
	g[i] = gelu * h1[i];
}

// rows[(b * T + t) * 32 + c] = (x[...] - mean[c]) / std[c]
__global__ void k_normalize_rows( float* x, const float* mean, const float* stdev, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) { x[i] = ( x[i] - mean[i % 32] ) / stdev[i % 32]; }
}

__global__ void k_pad_copy( const float* in, float* out, int L, int Lpad, int B ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= static_cast<size_t>( B ) * Lpad ) { return; }
	const int t = static_cast<int>( i % Lpad ), b = static_cast<int>( i / Lpad );
	out[i] = t < L ? in[static_cast<size_t>( b ) * L + t] : 0.0f;
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

	// reference-audio encoder (loaded on first use)
	struct EncRU {
		const float *a1, *w1, *b1, *a2, *w2, *b2;
	};
	struct EncBlock {
		EncRU ru[3];
		const float *sa, *cw, *cb;
	};
	pod::MemoryArena earena{};
	bool enc_ready = false;
	cublasHandle_t blas = nullptr;
	const float *e0w = nullptr, *e0b = nullptr, *e6a = nullptr, *e7w = nullptr, *e7b = nullptr;
	EncBlock eb[5];
	const float *n1w, *n1b, *qkvw, *qkvb, *pw, *pb, *n3w, *n3b, *p3w, *p3b, *n2w, *n2b, *mnw, *mnb, *w0w, *w0b, *w1w, *w1b, *w2w, *w2b, *mpw, *mpb;

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
		if ( p_->earena.base_ptr ) { engine::core::arena_destroy( p_->earena ); }
		if ( p_->blas ) { cublasDestroy( p_->blas ); }
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

bool H3AudioVAE::encode( const std::vector<float>& wave, int samples, std::vector<float>* rows, int* frames ) {
	Impl& s = *p_;
	if ( samples <= 0 || wave.size() < static_cast<size_t>( 2 ) * samples ) { return false; }
	CUDA_CHECK( cudaSetDevice( s.device ) );
	if ( !s.enc_ready ) {
		if ( !engine::core::arena_create( s.earena, s.device, 3ULL << 30 ) ) { return false; }
		pod::MemoryArena saved = s.warena;
		s.warena = s.earena; // raw() / wn() allocate from warena: point it at the encoder arena for the duration
		auto ld = [&]( const char* n ) { return s.raw( n ); };
		s.e0w = s.wn( "encoder.block.0" );
		s.e0b = ld( "encoder.block.0.bias" );
		for ( int i = 1; i <= 5; ++i ) {
			const std::string b = "encoder.block." + std::to_string( i ) + ".block.";
			for ( int j = 0; j < 3; ++j ) {
				const std::string r = b + std::to_string( j ) + ".block.";
				Impl::EncRU& u = s.eb[i - 1].ru[j];
				u.a1 = ld( ( r + "0.alpha" ).c_str() );
				u.w1 = s.wn( r + "1" );
				u.b1 = ld( ( r + "1.bias" ).c_str() );
				u.a2 = ld( ( r + "2.alpha" ).c_str() );
				u.w2 = s.wn( r + "3" );
				u.b2 = ld( ( r + "3.bias" ).c_str() );
			}
			s.eb[i - 1].sa = ld( ( b + "3.alpha" ).c_str() );
			s.eb[i - 1].cw = s.wn( b + "4" );
			s.eb[i - 1].cb = ld( ( b + "4.bias" ).c_str() );
		}
		s.e6a = ld( "encoder.block.6.alpha" );
		s.e7w = s.wn( "encoder.block.7" );
		s.e7b = ld( "encoder.block.7.bias" );
		s.n1w = ld( "pre_block.norm1.weight" ); s.n1b = ld( "pre_block.norm1.bias" );
		s.qkvw = ld( "pre_block.attn.qkv.weight" );
		{
			// qkv bias = [q_bias, 0, v_bias]
			float* bq = static_cast<float*>( engine::core::arena_alloc( s.warena, 3 * 2048 * sizeof( float ) ) );
			const float* qb = ld( "pre_block.attn.q_bias" );
			const float* vb = ld( "pre_block.attn.v_bias" );
			CUDA_CHECK( cudaMemset( bq, 0, 3 * 2048 * sizeof( float ) ) );
			CUDA_CHECK( cudaMemcpy( bq, qb, 2048 * sizeof( float ), cudaMemcpyDeviceToDevice ) );
			CUDA_CHECK( cudaMemcpy( bq + 4096, vb, 2048 * sizeof( float ), cudaMemcpyDeviceToDevice ) );
			s.qkvb = bq;
		}
		s.pw = ld( "pre_block.attn.proj.weight" ); s.pb = ld( "pre_block.attn.proj.bias" );
		s.n3w = ld( "pre_block.norm3.weight" ); s.n3b = ld( "pre_block.norm3.bias" );
		s.p3w = ld( "pre_block.proj.weight" ); s.p3b = ld( "pre_block.proj.bias" );
		s.n2w = ld( "pre_block.norm2.weight" ); s.n2b = ld( "pre_block.norm2.bias" );
		s.mnw = ld( "pre_block.mlp.norm.weight" ); s.mnb = ld( "pre_block.mlp.norm.bias" );
		s.w0w = ld( "pre_block.mlp.w0.weight" ); s.w0b = ld( "pre_block.mlp.w0.bias" );
		s.w1w = ld( "pre_block.mlp.w1.weight" ); s.w1b = ld( "pre_block.mlp.w1.bias" );
		s.w2w = ld( "pre_block.mlp.w2.weight" ); s.w2b = ld( "pre_block.mlp.w2.bias" );
		s.mpw = ld( "mean_proj.weight" ); s.mpb = ld( "mean_proj.bias" );
		s.earena = s.warena;
		s.warena = saved;
		if ( !s.mpb ) { return false; }
		cublasCreate( &s.blas );
		cublasSetMathMode( s.blas, CUBLAS_DEFAULT_MATH );
		s.enc_ready = true;
	}
	cudaStream_t st = s.stream;
	cublasSetStream( s.blas, st );
	const int B = 2, hop = 800;
	const int Lpad = ( samples + hop - 1 ) / hop * hop;
	const int T = Lpad / hop;
	// scratch: the widest activation is [B, 64, Lpad]; the strided convs write [B, 128, Lpad / 2], the same size
	const size_t big = static_cast<size_t>( B ) * 64 * Lpad;
	float *x = nullptr, *t1 = nullptr, *t2 = nullptr, *in = nullptr;
	CUDA_CHECK( cudaMalloc( &x, big * 4 ) );
	CUDA_CHECK( cudaMalloc( &t1, big * 4 ) );
	CUDA_CHECK( cudaMalloc( &t2, big * 4 ) );
	CUDA_CHECK( cudaMalloc( &in, static_cast<size_t>( 2 ) * samples * 4 ) );
	CUDA_CHECK( cudaMemcpyAsync( in, wave.data(), static_cast<size_t>( 2 ) * samples * 4, cudaMemcpyHostToDevice, st ) );
	k_pad_copy<<<( static_cast<size_t>( B ) * Lpad + 255 ) / 256, 256, 0, st>>>( in, t1, samples, Lpad, B );
	auto cv = [&]( const float* i, const float* w, const float* b, float* o, int Ci, int Co, int Tin, int Tout, int K, int dil, int pad, int stride ) {
		k_conv1d_s<<<dim3( ( Tout + 127 ) / 128, Co, B ), 128, 0, st>>>( i, w, b, o, Ci, Co, Tin, Tout, K, dil, pad, stride );
	};
	auto snake = [&]( float* v, const float* alpha, int C, int Tn ) {
		const size_t n = static_cast<size_t>( B ) * C * Tn;
		k_snake<<<( n + 255 ) / 256, 256, 0, st>>>( v, alpha, C, Tn, n );
	};
	cv( t1, s.e0w, s.e0b, x, 1, 64, Lpad, Lpad, 7, 1, 3, 1 );
	int len = Lpad, ch = 64;
	const int strides[5] = { 2, 4, 4, 5, 5 };
	const int dils[3] = { 1, 3, 9 };
	for ( int i = 0; i < 5; ++i ) {
		const Impl::EncBlock& b = s.eb[i];
		const size_t n = static_cast<size_t>( B ) * ch * len;
		for ( int j = 0; j < 3; ++j ) {
			const Impl::EncRU& u = b.ru[j];
			CUDA_CHECK( cudaMemcpyAsync( t1, x, n * 4, cudaMemcpyDeviceToDevice, st ) );
			snake( t1, u.a1, ch, len );
			cv( t1, u.w1, u.b1, t2, ch, ch, len, len, 7, dils[j], 3 * dils[j], 1 );
			snake( t2, u.a2, ch, len );
			cv( t2, u.w2, u.b2, t1, ch, ch, len, len, 1, 1, 0, 1 );
			k_add<<<( n + 255 ) / 256, 256, 0, st>>>( x, t1, n );
		}
		snake( x, b.sa, ch, len );
		const int stride = strides[i], k = 2 * stride, pad = ( stride + 1 ) / 2, out = ( len + 2 * pad - k ) / stride + 1;
		cv( x, b.cw, b.cb, t1, ch, ch * 2, len, out, k, 1, pad, stride );
		std::swap( x, t1 );
		ch *= 2;
		len = out;
	}
	// ch 2048, len T
	snake( x, s.e6a, ch, len );
	cv( x, s.e7w, s.e7b, t1, ch, ch, len, len, 3, 1, 1, 1 );
	const size_t R = static_cast<size_t>( B ) * T; // rows of the attention projection
	k_transpose_ct<<<( R * 2048 + 255 ) / 256, 256, 0, st>>>( t1, x, 2048, T, R * 2048 );
	// pre_block (all FP32 through cuBLAS): xo = proj(norm3(x)) + attn(norm1(x)); xo += mlp(norm2(xo))
	float* lat = nullptr; // [R, 32]
	float *n1 = nullptr, *qkv = nullptr, *ao = nullptr, *pooled = nullptr, *a2 = nullptr, *p3 = nullptr, *xo = nullptr, *h0 = nullptr, *h1 = nullptr, *g = nullptr, *mo = nullptr, *nn2 = nullptr;
	CUDA_CHECK( cudaMalloc( &n1, R * 2048 * 4 ) );
	CUDA_CHECK( cudaMalloc( &qkv, R * 6144 * 4 ) );
	CUDA_CHECK( cudaMalloc( &ao, static_cast<size_t>( B ) * 8 * T * 256 * 4 ) );
	CUDA_CHECK( cudaMalloc( &lat, R * 32 * 4 ) );
	CUDA_CHECK( cudaMalloc( &pooled, R * 32 * 4 ) );
	CUDA_CHECK( cudaMalloc( &a2, R * 32 * 4 ) );
	CUDA_CHECK( cudaMalloc( &p3, R * 32 * 4 ) );
	CUDA_CHECK( cudaMalloc( &xo, R * 32 * 4 ) );
	CUDA_CHECK( cudaMalloc( &nn2, R * 32 * 4 ) );
	CUDA_CHECK( cudaMalloc( &h0, R * 64 * 4 ) );
	CUDA_CHECK( cudaMalloc( &h1, R * 64 * 4 ) );
	CUDA_CHECK( cudaMalloc( &g, R * 64 * 4 ) );
	CUDA_CHECK( cudaMalloc( &mo, R * 32 * 4 ) );
	const float one = 1.0f, zero = 0.0f;
	// y[R, N] = x[R, K] W^T + bias
	auto linear = [&]( const float* xin, const float* W, const float* bias, float* y, int K, int N ) {
		cublasSgemm( s.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, static_cast<int>( R ), K, &one, W, K, xin, K, &zero, y, N );
		if ( bias ) { k_bias<<<( R * N + 255 ) / 256, 256, 0, st>>>( y, bias, N, R * N ); }
	};
	k_layernorm<<<static_cast<unsigned>( R ), 256, 0, st>>>( x, s.n1w, s.n1b, n1, 2048 );
	linear( n1, s.qkvw, s.qkvb, qkv, 2048, 6144 );
	k_attn_causal<<<dim3( T, 8, B ), 256, T * sizeof( float ), st>>>( qkv, ao, T, 8, 256, 1.0f / 16.0f );
	k_head_pool<<<( R * 32 + 255 ) / 256, 256, 0, st>>>( ao, pooled, T, 8, 256, 32, R * 32 );
	linear( pooled, s.pw, s.pb, a2, 32, 32 );
	k_layernorm<<<static_cast<unsigned>( R ), 256, 0, st>>>( x, s.n3w, s.n3b, n1, 2048 );
	linear( n1, s.p3w, s.p3b, p3, 2048, 32 );
	k_add2<<<( R * 32 + 255 ) / 256, 256, 0, st>>>( xo, p3, a2, R * 32 );
	k_layernorm<<<static_cast<unsigned>( R ), 64, 0, st>>>( xo, s.n2w, s.n2b, nn2, 32 );
	k_layernorm<<<static_cast<unsigned>( R ), 64, 0, st>>>( nn2, s.mnw, s.mnb, mo, 32 );
	linear( mo, s.w0w, s.w0b, h0, 32, 64 );
	linear( mo, s.w1w, s.w1b, h1, 32, 64 );
	k_geglu<<<( R * 64 + 255 ) / 256, 256, 0, st>>>( h0, h1, g, R * 64 );
	linear( g, s.w2w, s.w2b, mo, 64, 32 );
	k_add<<<( R * 32 + 255 ) / 256, 256, 0, st>>>( xo, mo, R * 32 );
	// mean_proj (1x1 conv == linear on the 32 channels), then (x - mean) / std
	linear( xo, s.mpw, s.mpb, lat, 32, 32 );
	k_normalize_rows<<<( R * 32 + 255 ) / 256, 256, 0, st>>>( lat, s.d_mean, s.d_std, R * 32 );
	rows->resize( R * 32 );
	CUDA_CHECK( cudaMemcpyAsync( rows->data(), lat, R * 32 * 4, cudaMemcpyDeviceToHost, st ) );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	CUDA_CHECK( cudaGetLastError() );
	for ( float* p : { x, t1, t2, in, n1, qkv, ao, lat, pooled, a2, p3, xo, nn2, h0, h1, g, mo } ) { cudaFree( p ); }
	*frames = T;
	return true;
}

} // namespace engine::video
