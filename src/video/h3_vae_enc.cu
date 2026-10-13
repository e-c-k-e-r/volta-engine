// MiniMax-H3 video VAE image encoder (see h3_vae_enc.h).
#include "engine/video/h3_vae_enc.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_vae.h"
#include "h3_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <thread>

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

// im2col (FP16 operands for the tensor-core GEMM) for rows [row0, row0 + rows) of the output frame: A[k][p], k = (dt * C + c) * kk + tap over the `ntaps` valid temporal taps
// (consecutive input frames starting at in0), p = row * Wout + col; reflect padding. kk = 9 (3x3) or 1 (1x1).
static __global__ void k_im2col16( const float* in0, half* A, int C, int H, int W, int Wout, int stride, int pad_t, int pad_l, int row0, int P, int kk, size_t frame_stride ) {
	// one block row per k (the tap, channel and frame offset are uniform), threads along p
	const int p = blockIdx.x * blockDim.x + threadIdx.x;
	if ( p >= P ) { return; }
	const int k = blockIdx.y;
	const int tap = k % kk, c = ( k / kk ) % C, dt = k / ( kk * C );
	const int prow = p / Wout, pcol = p - prow * Wout;
	int y, x;
	if ( kk == 9 ) {
		y = reflect( ( row0 + prow ) * stride + tap / 3 - pad_t, H );
		x = reflect( pcol * stride + tap % 3 - pad_l, W );
	} else {
		y = ( row0 + prow ) * stride;
		x = pcol * stride;
	}
	A[static_cast<size_t>( k ) * P + p] = __float2half_rn( in0[dt * frame_stride + ( static_cast<size_t>( c ) * H + y ) * W + x] );
}

// The same with 8 outputs per thread (Wout % 8 == 0 keeps them in one row): one 16-byte store per thread
static __global__ void k_im2col16_v8( const float* in0, half* A, int C, int H, int W, int Wout, int stride, int pad_t, int pad_l, int row0, int P, int kk, size_t frame_stride ) {
	const int p0 = ( blockIdx.x * blockDim.x + threadIdx.x ) * 8;
	if ( p0 >= P ) { return; }
	const int k = blockIdx.y;
	const int tap = k % kk, c = ( k / kk ) % C, dt = k / ( kk * C );
	const int prow = p0 / Wout, pcol0 = p0 - prow * Wout;
	const float* base = in0 + dt * frame_stride + static_cast<size_t>( c ) * H * W;
	int y;
	if ( kk == 9 ) { y = reflect( ( row0 + prow ) * stride + tap / 3 - pad_t, H ); } else { y = ( row0 + prow ) * stride; }
	const float* line = base + static_cast<size_t>( y ) * W;
	__align__( 16 ) half v[8];
	#pragma unroll
	for ( int j = 0; j < 8; ++j ) {
		const int x = kk == 9 ? reflect( ( pcol0 + j ) * stride + tap % 3 - pad_l, W ) : ( pcol0 + j ) * stride;
		v[j] = __float2half_rn( line[x] );
	}
	*reinterpret_cast<uint4*>( &A[static_cast<size_t>( k ) * P + p0] ) = *reinterpret_cast<const uint4*>( v );
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
	const half* w16 = nullptr; // [Co, kt * Ci * kk]: k index (dt * Ci + c) * kk + tap; the causal 3x3x3 convs reach frames t - 2, t - 1, t (dt 0, 1, 2)
	const float* b = nullptr;
	int co = 0, ci = 0, k = 3, kt = 1, kfull = 0;
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
	int device = 0;
	Conv conv_in, conv_out, quant;
	std::vector<ResBlk> blocks[6];
	Conv down[6];
	bool has_down[6] = {};
	Norm norm_out;
	float* a = nullptr; // activation ping-pong, im2col scratch, stats
	float *bufA = nullptr, *bufB = nullptr, *bufC = nullptr, *stats = nullptr;
	half* im2col = nullptr; // FP16 GEMM operand scratch
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

	// 3-D conv weight [Co, Ci, kt, k, k] -> FP16 [Co, kt * Ci * k * k] (one row per output channel, taps stacked so that one GEMM covers all temporal taps)
	Conv load_conv( const std::string& base, int k ) {
		Conv c;
		c.k = k;
		const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, ( base + ".weight" ).c_str() );
		if ( !m ) { LOG_ERROR( "H3 encoder: missing %s.weight", base.c_str() ); return c; }
		c.co = m->shape[0];
		c.ci = m->shape[1];
		const float* src = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, m ) );
		const size_t taps = static_cast<size_t>( k ) * k;
		// the loader keeps at most 4 dims, so the temporal kernel size comes from the element count
		const size_t kt = m->size_bytes / sizeof( float ) / ( static_cast<size_t>( c.co ) * c.ci * taps );
		ENGINE_ASSERT( kt >= 1 && kt <= 3, "H3 encoder: temporal kernel size" );
		c.kt = static_cast<int>( kt );
		c.kfull = static_cast<int>( kt * c.ci * taps );
		std::vector<half> w( static_cast<size_t>( c.co ) * c.kfull );
		for ( int o = 0; o < c.co; ++o ) {
			for ( int i = 0; i < c.ci; ++i ) {
				for ( size_t t = 0; t < kt; ++t ) {
					const float* sp = src + ( static_cast<size_t>( o ) * c.ci + i ) * kt * taps + t * taps;
					for ( size_t q = 0; q < taps; ++q ) { w[static_cast<size_t>( o ) * c.kfull + ( t * c.ci + i ) * taps + q] = __float2half_rn( sp[q] ); }
				}
			}
		}
		half* d = static_cast<half*>( engine::core::arena_alloc( warena, w.size() * sizeof( half ) ) );
		CUDA_CHECK( cudaMemcpy( d, w.data(), w.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		c.w16 = d;
		c.b = f32( base + ".bias" );
		return c;
	}

	Norm load_norm( const std::string& base ) {
		Norm n;
		n.w = f32( base + ".weight" );
		n.b = f32( base + ".bias" );
		return n;
	}

	// Causal 3-D conv over T frames [T][ci][H][W] -> [To][co][Ho][Wo], To = (T - 1) / ts + 1 (kt = 1: To = T). Output frame j sums the taps over input frames
	// j * ts - 2 + dt; frames before the clip are zeros, so the first frames skip their missing taps (one tensor-core GEMM per frame over the valid taps, FP32 accumulate).
	int conv3d( const Conv& c, const float* in, float* out, int T, int H, int W, int ts, int stride, int pad_t, int pad_b, int pad_l, int pad_r ) {
		const int Ho = ( H + pad_t + pad_b - c.k ) / stride + 1, Wo = ( W + pad_l + pad_r - c.k ) / stride + 1;
		const size_t fin = static_cast<size_t>( c.ci ) * H * W, fout = static_cast<size_t>( c.co ) * Ho * Wo;
		const int To = c.kt == 1 ? T : ( T - 1 ) / ts + 1;
		const int kk = c.k * c.k;
		const float alpha = 1.0f, beta = 0.0f;
		for ( int j = 0; j < To; ++j ) {
			const int dt0 = c.kt == 1 ? 0 : std::max( 0, 2 - j * ts ); // taps before dt0 would read frames before the clip
			const int ntaps = c.kt - dt0, K = ntaps * c.ci * kk;
			const float* first = in + static_cast<size_t>( c.kt == 1 ? j : j * ts + dt0 - 2 ) * fin;
			const int rows_per = std::max( 1, static_cast<int>( im2col_elems / ( static_cast<size_t>( K ) * Wo ) ) );
			for ( int r0 = 0; r0 < Ho; r0 += rows_per ) {
				const int rows = std::min( rows_per, Ho - r0 ), P = rows * Wo;
				if ( Wo % 8 == 0 ) {
					k_im2col16_v8<<<dim3( ( P / 8 + 127 ) / 128, K ), 128, 0, stream>>>( first, im2col, c.ci, H, W, Wo, stride, pad_t, pad_l, r0, P, kk, fin );
				} else {
					k_im2col16<<<dim3( ( P + 255 ) / 256, K ), 256, 0, stream>>>( first, im2col, c.ci, H, W, Wo, stride, pad_t, pad_l, r0, P, kk, fin );
				}
				CUBLAS_CHECK( cublasGemmEx( blas, CUBLAS_OP_N, CUBLAS_OP_N, P, c.co, K, &alpha, im2col, CUDA_R_16F, P, c.w16 + static_cast<size_t>( dt0 ) * c.ci * kk, CUDA_R_16F, c.kfull, &beta,
					out + j * fout + static_cast<size_t>( r0 ) * Wo, CUDA_R_32F, Ho * Wo, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
			}
			const size_t HW = static_cast<size_t>( Ho ) * Wo;
			if ( c.b ) { k_add_bias_ch<<<( c.co * HW + 255 ) / 256, 256, 0, stream>>>( out + j * fout, c.b, c.co, HW ); }
		}
		return To;
	}

	// GroupNorm (+ SiLU) per frame: the checkpoint's norms are time-isolated
	void gn_frames( const Norm& n, const float* in, float* out, int T, int C, size_t HW, bool silu ) {
		for ( int t = 0; t < T; ++t ) { gn_silu( n, in + t * C * HW, out + t * C * HW, C, HW, silu ); }
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
		if ( p_->a ) { (void) cudaFree( p_->a ); }
		if ( p_->im2col ) { (void) cudaFree( p_->im2col ); }
		engine::loader::safetensors_close( p_->st );
		delete p_;
	}
}

bool H3ImageEncoder::load( const char* dir, int device ) {
	Impl& s = *p_;
	s.device = device;
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
	CUDA_CHECK( cudaDeviceSynchronize() ); // loading ran on the default stream; the models run on non-blocking streams, which it does not order against
	return s.conv_in.w16 && s.quant.w16;
}

// T frames [T][3][H][W] -> moments [T'][48][H/16][W/16] (T' = 1 + (T - 1) / 4 after the two time downsamplers)
bool H3ImageEncoder::encode_tile( const float* d_rgb, int T, int H, int W, float* d_moments ) {
	Impl& s = *p_;
	CUDA_CHECK( cudaSetDevice( s.device ) );
	if ( H % 16 || W % 16 ) { return false; }
	// activation buffers: the first level is the largest: 128 channels at H x W per frame
	const size_t need = static_cast<size_t>( T ) * kCh * H * W;
	if ( s.buf_elems < need ) {
		if ( s.a ) { (void) cudaFree( s.a ); }
		if ( s.im2col ) { (void) cudaFree( s.im2col ); }
		s.im2col_elems = 320ull << 20; // 640 MB of FP16: a whole 256 px frame of the widest 3x3 conv (128 channels x 27 taps) at once
		CUDA_CHECK( cudaMalloc( &s.im2col, s.im2col_elems * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &s.a, ( need * 3 + 1024 ) * sizeof( float ) ) );
		s.bufA = s.a;
		s.bufB = s.a + need;
		s.bufC = s.a + 2 * need;
		s.stats = s.a + 3 * need;
		s.buf_elems = need;
	}
	cudaStream_t st = s.stream;
	int h = H, w = W, t = T, ch = kCh * kMult[0];
	float* x = s.bufA;
	s.conv3d( s.conv_in, d_rgb, x, t, h, w, 1, 1, 1, 1, 1, 1 );
	float* t1 = s.bufB;
	float* t2 = s.bufC;
	for ( int l = 0; l < 6; ++l ) {
		const int mid = kCh * kMult[l];
		for ( size_t i = 0; i < s.blocks[l].size(); ++i ) {
			const ResBlk& b = s.blocks[l][i];
			const size_t HW = static_cast<size_t>( h ) * w;
			s.gn_frames( b.n1, x, t1, t, ch, HW, true );
			s.conv3d( b.c1, t1, t2, t, h, w, 1, 1, 1, 1, 1, 1 );
			s.gn_frames( b.n2, t2, t1, t, mid, HW, true );
			s.conv3d( b.c2, t1, t2, t, h, w, 1, 1, 1, 1, 1, 1 );
			const size_t n = static_cast<size_t>( t ) * mid * HW;
			if ( b.has_sc ) {
				s.conv3d( b.sc, x, t1, t, h, w, 1, 1, 0, 0, 0, 0 );
				k_add_eq<<<( n + 255 ) / 256, 256, 0, st>>>( t2, t1, n );
			} else {
				k_add_eq<<<( n + 255 ) / 256, 256, 0, st>>>( t2, x, n );
			}
			std::swap( x, t2 );
			ch = mid;
		}
		if ( s.has_down[l] ) {
			// reflect pad (0, 1, 0, 1) then a 3x3x3 conv (spatial stride ds without spatial padding, temporal stride 1 or 2)
			const int ds = kSpaceDown[l];
			t = s.conv3d( s.down[l], x, t2, t, h, w, kTimeDown[l], ds, 0, 1, 0, 1 );
			h = ( h + 1 - 3 ) / ds + 1;
			w = ( w + 1 - 3 ) / ds + 1;
			std::swap( x, t2 );
		}
	}
	const size_t HW = static_cast<size_t>( h ) * w;
	s.gn_frames( s.norm_out, x, t1, t, ch, HW, true );
	s.conv3d( s.conv_out, t1, t2, t, h, w, 1, 1, 1, 1, 1, 1 );
	s.conv3d( s.quant, t2, d_moments, t, h, w, 1, 1, 0, 0, 0, 0 );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	CUDA_CHECK( cudaGetLastError() );
	return h == H / 16 && w == W / 16 && t == 1 + ( T - 1 ) / 4;
}

namespace {

constexpr int kRatio = 16, kTile = 256, kOverlap = 64;

// Blends the raw tile moments raw[i * nj + j] ([nc, th/16, tw/16] each) into one [nc, H/16, W/16] block with the checkpoint's order: rows first, then columns against the
// unblended left tile, trailing overlaps cropped
void blend_tiles( const std::vector<std::vector<float>>& raw, const H3Tiles& ty, const H3Tiles& tx, const std::vector<int>& th, const std::vector<int>& tw, int NC, int H, int W, std::vector<float>* out_ ) {
	const int ni = static_cast<int>( ty.start.size() ), nj = static_cast<int>( tx.start.size() );
	const int LH = H / kRatio, LW = W / kRatio;
	std::vector<float>& out = *out_;
	out.assign( static_cast<size_t>( NC ) * LH * LW, 0.0f );
	for ( int i = 0; i < ni; ++i ) {
		for ( int j = 0; j < nj; ++j ) {
			const int lh = th[i] / kRatio, lw = tw[j] / kRatio;
			std::vector<float> tile = raw[static_cast<size_t>( i ) * nj + j];
			if ( i > 0 ) { // blend the top rows with the bottom rows of the raw tile above
				const std::vector<float>& a = raw[static_cast<size_t>( i - 1 ) * nj + j];
				const int ah = th[i - 1] / kRatio, ext = std::min( { ah, lh, ty.overlap[i - 1] / kRatio } );
				for ( int c = 0; c < NC; ++c ) {
					for ( int y = 0; y < ext; ++y ) {
						const float wb = static_cast<float>( y ) / ext, wa = 1.0f - wb;
						for ( int x = 0; x < lw; ++x ) {
							float& dst = tile[( static_cast<size_t>( c ) * lh + y ) * lw + x];
							dst = a[( static_cast<size_t>( c ) * ah + ( ah - ext + y ) ) * lw + x] * wa + dst * wb;
						}
					}
				}
			}
			if ( j > 0 ) { // then the left columns with the raw tile to the left
				const std::vector<float>& a = raw[static_cast<size_t>( i ) * nj + j - 1];
				const int aw = tw[j - 1] / kRatio, ext = std::min( { aw, lw, tx.overlap[j - 1] / kRatio } );
				for ( int c = 0; c < NC; ++c ) {
					for ( int y = 0; y < lh; ++y ) {
						for ( int x = 0; x < ext; ++x ) {
							const float wb = static_cast<float>( x ) / ext, wa = 1.0f - wb;
							float& dst = tile[( static_cast<size_t>( c ) * lh + y ) * lw + x];
							dst = a[( static_cast<size_t>( c ) * lh + y ) * aw + ( aw - ext + x )] * wa + dst * wb;
						}
					}
				}
			}
			const int keep_h = i < ni - 1 ? lh - ty.overlap[i] / kRatio : lh, keep_w = j < nj - 1 ? lw - tx.overlap[j] / kRatio : lw;
			const int y0 = ty.start[i] / kRatio, x0 = tx.start[j] / kRatio;
			for ( int c = 0; c < NC; ++c ) {
				for ( int y = 0; y < keep_h; ++y ) {
					std::memcpy( &out[( static_cast<size_t>( c ) * LH + y0 + y ) * LW + x0], &tile[( static_cast<size_t>( c ) * lh + y ) * lw], static_cast<size_t>( keep_w ) * sizeof( float ) );
				}
			}
		}
	}
}

// A clip is a list of frame pointers (host, [3][Hfull][Wfull] each); one job is one 256 px tile of one clip
struct EncodeJob {
	int clip, ti, tj;
};

} // namespace

// Host crop of tile (y0, x0, th, tw) of the clip's frames -> device -> encoder -> host moments [T'][48][th/16][tw/16]
bool H3ImageEncoder::encode_crop( const std::vector<const float*>& frames, int Hfull, int Wfull, int y0, int x0, int th, int tw, std::vector<float>* moments ) {
	Impl& s = *p_;
	const int T = static_cast<int>( frames.size() ), Tl = 1 + ( T - 1 ) / 4;
	std::vector<float> crop( static_cast<size_t>( T ) * 3 * th * tw );
	for ( int f = 0; f < T; ++f ) {
		for ( int c = 0; c < 3; ++c ) {
			for ( int y = 0; y < th; ++y ) {
				std::memcpy( &crop[( ( static_cast<size_t>( f ) * 3 + c ) * th + y ) * tw], frames[f] + ( static_cast<size_t>( c ) * Hfull + y0 + y ) * Wfull + x0, static_cast<size_t>( tw ) * sizeof( float ) );
			}
		}
	}
	CUDA_CHECK( cudaSetDevice( s.device ) );
	float *d_in = nullptr, *d_m = nullptr;
	CUDA_CHECK( cudaMalloc( &d_in, crop.size() * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_m, static_cast<size_t>( Tl ) * 48 * ( th / kRatio ) * ( tw / kRatio ) * sizeof( float ) ) );
	// on the encoder's own stream: a plain cudaMemcpy runs on the default stream, whose pageable-memory copy may still be in flight when kernels on this stream start
	CUDA_CHECK( cudaMemcpyAsync( d_in, crop.data(), crop.size() * sizeof( float ), cudaMemcpyHostToDevice, s.stream ) );
	const bool ok = encode_tile( d_in, T, th, tw, d_m );
	moments->resize( static_cast<size_t>( Tl ) * 48 * ( th / kRatio ) * ( tw / kRatio ) );
	CUDA_CHECK( cudaMemcpy( moments->data(), d_m, moments->size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
	(void) cudaFree( d_in );
	(void) cudaFree( d_m );
	return ok;
}

// Encodes every clip (all tiles of all clips are spread over the encoders, one thread per GPU) -> per clip moments [T'][48][H/16][W/16]
bool H3ImageEncoder::encode_clips( const std::vector<H3ImageEncoder*>& encs, const std::vector<std::vector<const float*>>& clips, int H, int W, std::vector<std::vector<float>>* out ) {
	if ( H % kRatio || W % kRatio || encs.empty() || clips.empty() ) { return false; }
	// the checkpoint tiles the encoder (256 px tiles, >= 64 px overlap) and blends the moments linearly
	const H3Tiles ty = h3_split_tiles( H, kTile, kOverlap, kRatio ), tx = h3_split_tiles( W, kTile, kOverlap, kRatio );
	const int ni = static_cast<int>( ty.start.size() ), nj = static_cast<int>( tx.start.size() );
	std::vector<int> th( ni ), tw( nj );
	for ( int i = 0; i < ni; ++i ) { th[i] = std::min( ty.len[i], H - ty.start[i] ); }
	for ( int j = 0; j < nj; ++j ) { tw[j] = std::min( tx.len[j], W - tx.start[j] ); }
	const int nclips = static_cast<int>( clips.size() );
	std::vector<std::vector<std::vector<float>>> raw( nclips, std::vector<std::vector<float>>( static_cast<size_t>( ni ) * nj ) );
	std::vector<EncodeJob> jobs;
	for ( int c = 0; c < nclips; ++c ) {
		for ( int i = 0; i < ni; ++i ) {
			for ( int j = 0; j < nj; ++j ) { jobs.push_back( { c, i, j } ); }
		}
	}
	const int nw = static_cast<int>( encs.size() );
	std::vector<char> ok( nw, 1 );
	std::vector<std::thread> threads;
	for ( int w = 0; w < nw; ++w ) {
		threads.emplace_back( [&, w] {
			for ( size_t k = static_cast<size_t>( w ); k < jobs.size(); k += static_cast<size_t>( nw ) ) {
				const EncodeJob& jb = jobs[k];
				if ( !encs[w]->encode_crop( clips[jb.clip], H, W, ty.start[jb.ti], tx.start[jb.tj], th[jb.ti], tw[jb.tj], &raw[jb.clip][static_cast<size_t>( jb.ti ) * nj + jb.tj] ) ) { ok[w] = 0; return; }
			}
		} );
	}
	for ( std::thread& t : threads ) { t.join(); }
	for ( char c : ok ) { if ( !c ) { return false; } }
	out->resize( nclips );
	for ( int c = 0; c < nclips; ++c ) {
		const int Tl = 1 + ( static_cast<int>( clips[c].size() ) - 1 ) / 4;
		blend_tiles( raw[c], ty, tx, th, tw, 48 * Tl, H, W, &( *out )[c] );
	}
	return true;
}

bool H3ImageEncoder::encode( const std::vector<H3ImageEncoder*>& encs, const float* h_rgb, int H, int W, std::vector<float>* moments ) {
	std::vector<std::vector<float>> out;
	if ( !encode_clips( encs, { { h_rgb } }, H, W, &out ) ) { return false; }
	*moments = std::move( out[0] );
	return true;
}

bool H3ImageEncoder::encode_video( const std::vector<H3ImageEncoder*>& encs, const float* h_rgb, int T, int H, int W, std::vector<float>* moments, int* latent_frames ) {
	constexpr int kClip = 17, kTail = 5, kTokens = 5, kDrop = 3;
	if ( T < kClip + kTail ) { LOG_ERROR( "H3 encoder: a reference video needs at least %d frames", kClip + kTail ); return false; }
	const int used = std::max( ( T - kTail ) / kClip, 1 ) * kClip + kTail; // trim to 17 n + 5 frames
	const int chunks = ( used + kClip - 1 ) / kClip;                      // the last chunk repeats the final frame
	const size_t frame = static_cast<size_t>( 3 ) * H * W;
	std::vector<std::vector<const float*>> clips( chunks );
	for ( int c = 0; c < chunks; ++c ) {
		for ( int f = 0; f < kClip; ++f ) { clips[c].push_back( h_rgb + static_cast<size_t>( std::min( c * kClip + f, used - 1 ) ) * frame ); }
	}
	std::vector<std::vector<float>> per_clip;
	if ( !encode_clips( encs, clips, H, W, &per_clip ) ) { return false; }
	moments->clear();
	for ( const std::vector<float>& m : per_clip ) { moments->insert( moments->end(), m.begin(), m.end() ); }
	*latent_frames = chunks * kTokens - kDrop;
	moments->resize( static_cast<size_t>( *latent_frames ) * 48 * ( H / 16 ) * ( W / 16 ) );
	return true;
}

} // namespace engine::video
