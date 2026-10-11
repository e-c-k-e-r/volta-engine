// MiniMax-H3 video VAE image encoder (see h3_vae_enc.h).
#include "engine/video/h3_vae_enc.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "h3_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <string>

#define CUBLAS_CHECK( call ) \
	do { \
		cublasStatus_t st_ = ( call ); \
		if ( ENGINE_UNLIKELY( st_ != CUBLAS_STATUS_SUCCESS ) ) { \
			std::fprintf( stderr, "[CUBLAS ERROR] %s (%s:%d): %d\n", #call, __FILE__, __LINE__, static_cast<int>( st_ ) ); \
			std::_Exit( 1 ); \
		} \
	} while ( 0 )

namespace engine::video {
namespace {

using namespace kern;

constexpr int kCh = 128;
constexpr int kMult[6] = { 1, 2, 2, 4, 4, 8 };
constexpr int kSpaceDown[6] = { 2, 2, 2, 2, 1, 1 };
constexpr int kTimeDown[6] = { 1, 2, 2, 1, 1, 1 };

__device__ __forceinline__ int reflect( int i, int n ) {
	if ( i < 0 ) { i = -i; }
	if ( i >= n ) { i = 2 * n - 2 - i; }
	return i;
}

// im2col for rows [row0, row0 + rows) of the output: A[k][p], k = (c * 3 + ky) * 3 + kx, p = row * Wout + col; reflect padding
static __global__ void k_im2col( const float* in, float* A, int C, int H, int W, int Wout, int stride, int pad_t, int pad_l, int row0, int P ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( C ) * 9 * P;
	if ( i >= total ) { return; }
	const int p = static_cast<int>( i % P );
	const int k = static_cast<int>( i / P );
	const int kx = k % 3, ky = ( k / 3 ) % 3, c = k / 9;
	const int orow = row0 + p / Wout, ocol = p % Wout;
	const int y = reflect( orow * stride + ky - pad_t, H ), x = reflect( ocol * stride + kx - pad_l, W );
	A[i] = in[( static_cast<size_t>( c ) * H + y ) * W + x];
}

// GroupNorm statistics for 32 groups over a [C, HW] tensor: mean and 1/std per group
static __global__ void k_gn_stats( const float* x, float* stats, int C, size_t HW, float eps ) {
	__shared__ float red[32];
	const int g = blockIdx.x;
	const int cg = C / 32;
	const float* base = x + static_cast<size_t>( g ) * cg * HW;
	const size_t n = static_cast<size_t>( cg ) * HW;
	double s = 0.0;
	for ( size_t i = threadIdx.x; i < n; i += blockDim.x ) { s += base[i]; }
	float fs = static_cast<float>( s );
	const float sum = block_sum( fs, red );
	const float mean = sum / static_cast<float>( n );
	double v = 0.0;
	for ( size_t i = threadIdx.x; i < n; i += blockDim.x ) {
		const double d = base[i] - mean;
		v += d * d;
	}
	const float var = block_sum( static_cast<float>( v ), red ) / static_cast<float>( n );
	if ( threadIdx.x == 0 ) {
		stats[g * 2] = mean;
		stats[g * 2 + 1] = rsqrtf( var + eps );
	}
}

static __global__ void k_gn_silu( const float* x, const float* stats, const float* w, const float* b, float* out, int C, size_t HW, bool silu ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= static_cast<size_t>( C ) * HW ) { return; }
	const int c = static_cast<int>( i / HW ), g = c / ( C / 32 );
	float v = ( x[i] - stats[g * 2] ) * stats[g * 2 + 1] * w[c] + b[c];
	if ( silu ) { v = v / ( 1.0f + expf( -v ) ); }
	out[i] = v;
}

static __global__ void k_add_bias_ch( float* y, const float* b, int C, size_t HW ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < static_cast<size_t>( C ) * HW ) { y[i] += b[i / HW]; }
}

static __global__ void k_add_eq( float* y, const float* x, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) { y[i] += x[i]; }
}

struct Conv {
	const float* w = nullptr; // [Co, Ci * 9] (3x3) or [Co, Ci] (1x1)
	const float* b = nullptr;
	int co = 0, ci = 0, k = 3;
};

struct Norm {
	const float *w = nullptr, *b = nullptr;
};

struct ResBlk {
	Norm n1, n2;
	Conv c1, c2, sc;
	bool has_sc = false;
};

} // namespace

struct H3ImageEncoder::Impl {
	pod::SafeTensorsFile st{};
	pod::MemoryArena warena{};
	cublasHandle_t blas = nullptr;
	cudaStream_t stream = nullptr;
	Conv conv_in, conv_out, quant;
	std::vector<ResBlk> blocks[6];
	Conv down[6];
	bool has_down[6] = {};
	Norm norm_out;
	float* a = nullptr; // activation ping-pong, im2col scratch, stats
	float *bufA = nullptr, *bufB = nullptr, *bufC = nullptr, *im2col = nullptr, *stats = nullptr;
	size_t buf_elems = 0, im2col_elems = 0;

	const float* f32( const std::string& name, size_t* n = nullptr ) {
		const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, name.c_str() );
		if ( !m ) { LOG_ERROR( "H3 encoder: missing tensor %s", name.c_str() ); return nullptr; }
		ENGINE_ASSERT( m->dtype == pod::DType::FP32, "H3 encoder: expected FP32" );
		float* d = static_cast<float*>( engine::core::arena_alloc( warena, m->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( d, engine::loader::safetensors_get_data_cpu( st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
		if ( n ) { *n = m->size_bytes / 4; }
		return d;
	}

	// 3-D conv weight [Co, Ci, 3, 3, 3] -> its last temporal tap [Co, Ci * 9]; [Co, Ci, 1, 1, 1] -> [Co, Ci]
	Conv load_conv( const std::string& base, int k ) {
		Conv c;
		c.k = k;
		const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, ( base + ".weight" ).c_str() );
		if ( !m ) { LOG_ERROR( "H3 encoder: missing %s.weight", base.c_str() ); return c; }
		c.co = m->shape[0];
		c.ci = m->shape[1];
				const float* src = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, m ) );
		const size_t taps = static_cast<size_t>( k ) * k;
		std::vector<float> w( static_cast<size_t>( c.co ) * c.ci * taps );
		// the loader keeps at most 4 dims, so the temporal kernel size comes from the element count
		const size_t kt = m->size_bytes / sizeof( float ) / ( static_cast<size_t>( c.co ) * c.ci * taps );
		ENGINE_ASSERT( kt >= 1 && kt <= 3, "H3 encoder: temporal kernel size" );
		for ( int o = 0; o < c.co; ++o ) {
			for ( int i = 0; i < c.ci; ++i ) {
				const float* s = src + ( static_cast<size_t>( o ) * c.ci + i ) * kt * taps + ( kt - 1 ) * taps;
				std::memcpy( &w[( static_cast<size_t>( o ) * c.ci + i ) * taps], s, taps * sizeof( float ) );
			}
		}
		float* d = static_cast<float*>( engine::core::arena_alloc( warena, w.size() * sizeof( float ) ) );
		CUDA_CHECK( cudaMemcpy( d, w.data(), w.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
		c.w = d;
		c.b = f32( base + ".bias" );
		return c;
	}

	Norm load_norm( const std::string& base ) {
		Norm n;
		n.w = f32( base + ".weight" );
		n.b = f32( base + ".bias" );
		return n;
	}

	// out [co, Ho * Wo] = conv(in [ci, H, W])  (3x3 reflect-padded with the given stride / asymmetric pad, or 1x1)
	void run_conv( const Conv& c, const float* in, float* out, int H, int W, int stride, int pad_t, int pad_b, int pad_l, int pad_r ) {
		const int Ho = ( H + pad_t + pad_b - c.k ) / stride + 1, Wo = ( W + pad_l + pad_r - c.k ) / stride + 1;
		const float alpha = 1.0f, beta = 0.0f;
		if ( c.k == 1 ) {
			const int P = H * W;
			CUBLAS_CHECK( cublasSgemm( blas, CUBLAS_OP_N, CUBLAS_OP_N, P, c.co, c.ci, &alpha, in, P, c.w, c.ci, &beta, out, P ) );
		} else {
			const int K = c.ci * 9;
			const int rows_per = std::max( 1, static_cast<int>( im2col_elems / ( static_cast<size_t>( K ) * Wo ) ) );
			for ( int r0 = 0; r0 < Ho; r0 += rows_per ) {
				const int rows = std::min( rows_per, Ho - r0 ), P = rows * Wo;
				const size_t total = static_cast<size_t>( K ) * P;
				k_im2col<<<( total + 255 ) / 256, 256, 0, stream>>>( in, im2col, c.ci, H, W, Wo, stride, pad_t, pad_l, r0, P );
				CUBLAS_CHECK( cublasSgemm( blas, CUBLAS_OP_N, CUBLAS_OP_N, P, c.co, K, &alpha, im2col, P, c.w, K, &beta, out + static_cast<size_t>( r0 ) * Wo, Ho * Wo ) );
			}
		}
		const size_t HW = static_cast<size_t>( Ho ) * Wo;
		if ( c.b ) { k_add_bias_ch<<<( c.co * HW + 255 ) / 256, 256, 0, stream>>>( out, c.b, c.co, HW ); }
	}

	void gn_silu( const Norm& n, const float* in, float* out, int C, size_t HW, bool silu ) {
		k_gn_stats<<<32, 1024, 0, stream>>>( in, stats, C, HW, 1e-6f );
		k_gn_silu<<<( C * HW + 255 ) / 256, 256, 0, stream>>>( in, stats, n.w, n.b, out, C, HW, silu );
	}
};

H3ImageEncoder::H3ImageEncoder() : p_( new Impl ) {}

H3ImageEncoder::~H3ImageEncoder() {
	if ( p_ ) {
		if ( p_->blas ) { cublasDestroy( p_->blas ); }
		if ( p_->warena.base_ptr ) { engine::core::arena_destroy( p_->warena ); }
		if ( p_->a ) { cudaFree( p_->a ); }
		engine::loader::safetensors_close( p_->st );
		delete p_;
	}
}

bool H3ImageEncoder::load( const char* dir, int device ) {
	Impl& s = *p_;
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &s.stream, cudaStreamNonBlocking ) );
	CUBLAS_CHECK( cublasCreate( &s.blas ) );
	CUBLAS_CHECK( cublasSetStream( s.blas, s.stream ) );
	CUBLAS_CHECK( cublasSetMathMode( s.blas, CUBLAS_DEFAULT_MATH ) );
	if ( !engine::loader::safetensors_open( s.st, ( std::string( dir ) + "/source/model.safetensors" ).c_str() ) ) { return false; }
	if ( !engine::core::arena_create( s.warena, device, 2ULL << 30 ) ) { return false; }
	s.conv_in = s.load_conv( "encoder.conv_in", 3 );
	int prev = kCh * kMult[0];
	for ( int l = 0; l < 6; ++l ) {
		const int mid = kCh * kMult[l];
		for ( int i = 0; i < 2; ++i ) {
			const std::string p = "encoder.down." + std::to_string( l ) + ".block." + std::to_string( i );
			ResBlk b;
			const int in_ch = i == 0 ? prev : mid;
			b.n1 = s.load_norm( p + ".norm1" );
			b.n2 = s.load_norm( p + ".norm2" );
			b.c1 = s.load_conv( p + ".conv1", 3 );
			b.c2 = s.load_conv( p + ".conv2", 3 );
			if ( in_ch != mid ) {
				b.sc = s.load_conv( p + ".nin_shortcut", 1 );
				b.has_sc = true;
			}
			s.blocks[l].push_back( b );
		}
		if ( kSpaceDown[l] * kTimeDown[l] > 1 ) {
			s.down[l] = s.load_conv( "encoder.down." + std::to_string( l ) + ".downsample.conv", 3 );
			s.has_down[l] = true;
		}
		prev = mid;
	}
	s.norm_out = s.load_norm( "encoder.norm_out" );
	s.conv_out = s.load_conv( "encoder.conv_out", 3 );
	s.quant = s.load_conv( "quant_conv", 1 );
	return s.conv_in.w && s.quant.w;
}

bool H3ImageEncoder::encode( const float* d_rgb, int H, int W, float* d_moments ) {
	Impl& s = *p_;
	if ( H % 16 || W % 16 ) { return false; }
	// activation buffers: the first level is the largest: 128 channels at H x W
	const size_t need = static_cast<size_t>( kCh ) * H * W;
	if ( s.buf_elems < need ) {
		if ( s.a ) { cudaFree( s.a ); }
		s.im2col_elems = 64ull << 20; // 256 MB
		CUDA_CHECK( cudaMalloc( &s.a, ( need * 3 + s.im2col_elems + 1024 ) * sizeof( float ) ) );
		s.bufA = s.a;
		s.bufB = s.a + need;
		s.bufC = s.a + 2 * need;
		s.im2col = s.a + 3 * need;
		s.stats = s.im2col + s.im2col_elems;
		s.buf_elems = need;
	}
	cudaStream_t st = s.stream;
	int h = H, w = W, ch = kCh * kMult[0];
	float* x = s.bufA;
	s.run_conv( s.conv_in, d_rgb, x, h, w, 1, 1, 1, 1, 1 );
	float* t1 = s.bufB;
	float* t2 = s.bufC;
	for ( int l = 0; l < 6; ++l ) {
		const int mid = kCh * kMult[l];
		for ( size_t i = 0; i < s.blocks[l].size(); ++i ) {
			const ResBlk& b = s.blocks[l][i];
			const size_t HW = static_cast<size_t>( h ) * w;
			s.gn_silu( b.n1, x, t1, ch, HW, true );
			s.run_conv( b.c1, t1, t2, h, w, 1, 1, 1, 1, 1 );
			s.gn_silu( b.n2, t2, t1, mid, HW, true );
			s.run_conv( b.c2, t1, t2, h, w, 1, 1, 1, 1, 1 );
			if ( b.has_sc ) {
				s.run_conv( b.sc, x, t1, h, w, 1, 0, 0, 0, 0 );
				k_add_eq<<<( mid * HW + 255 ) / 256, 256, 0, st>>>( t2, t1, mid * HW );
			} else {
				k_add_eq<<<( mid * HW + 255 ) / 256, 256, 0, st>>>( t2, x, mid * HW );
			}
			std::swap( x, t2 );
			ch = mid;
		}
		if ( s.has_down[l] ) {
			// reflect pad (0, 1, 0, 1) then a 3x3 stride-2 conv without spatial padding
			const int ds = kSpaceDown[l];
			s.run_conv( s.down[l], x, t2, h, w, ds, 0, 1, 0, 1 );
			h = ( h + 1 - 3 ) / ds + 1;
			w = ( w + 1 - 3 ) / ds + 1;
			std::swap( x, t2 );
		}
	}
	const size_t HW = static_cast<size_t>( h ) * w;
	s.gn_silu( s.norm_out, x, t1, ch, HW, true );
	s.run_conv( s.conv_out, t1, t2, h, w, 1, 1, 1, 1, 1 );
	s.run_conv( s.quant, t2, d_moments, h, w, 1, 0, 0, 0, 0 );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	CUDA_CHECK( cudaGetLastError() );
	return h == H / 16 && w == W / 16;
}

} // namespace engine::video
