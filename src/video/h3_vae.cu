// MiniMax-H3 video VAE decoder (ViT3D) with the checkpoint's spatial tiling and temporal chunking.
#include "engine/video/h3_vae.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_fmha.h"
#include "h3_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
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

constexpr int kDim = 2048;
constexpr int kHeads = 32;
constexpr int kHd = 64;
constexpr int kFf = 8192;
constexpr int kLayers = 36;
constexpr int kSuffix = 5; // 4 register tokens + the zero cls token
constexpr int kPatchPix = 3 * 4 * 16 * 16;

__global__ void k_pack_latent( const float* z, float* rows, int T, int HW ) {
	const size_t n = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; // token (t, hw)
	if ( n >= static_cast<size_t>( T ) * HW ) { return; }
	const int t = static_cast<int>( n / HW ), hw = static_cast<int>( n % HW );
	for ( int c = 0; c < 24; ++c ) { rows[n * 24 + c] = z[( static_cast<size_t>( c ) * T + t ) * HW + hw]; }
}

__global__ void k_denorm_latent( float* z, const float* mean, const float* stdev, size_t per_channel ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= 24 * per_channel ) { return; }
	const int c = static_cast<int>( i / per_channel );
	z[i] = z[i] * stdev[c] + mean[c];
}

__global__ void k_set_suffix( float* x, const float* reg, int N ) {
	// rows N .. N+3 = register tokens, row N+4 = zeros
	const int r = blockIdx.x;
	for ( int c = threadIdx.x; c < kDim; c += blockDim.x ) { x[( static_cast<size_t>( N ) + r ) * kDim + c] = r < 4 ? reg[r * kDim + c] : 0.0f; }
}

// cos/sin for dims 0..23 of the 48 rotated dims (dims 24..47 repeat them): angle = 2 pi id inv_freq, id = 2 (i + 0.5) / n - 1 per axis
__global__ void k_vit_rope( float* cosb, float* sinb, int T, int H, int W, int S ) {
	const int s = blockIdx.x * blockDim.x + threadIdx.x;
	if ( s >= S ) { return; }
	const int N = T * H * W;
	float id[3] = { 0.0f, 0.0f, 0.0f };
	if ( s < N ) {
		const int w = s % W, h = ( s / W ) % H, t = s / ( W * H );
		id[0] = 2.0f * ( ( static_cast<float>( t ) + 0.5f ) / T ) - 1.0f;
		id[1] = 2.0f * ( ( static_cast<float>( h ) + 0.5f ) / H ) - 1.0f;
		id[2] = 2.0f * ( ( static_cast<float>( w ) + 0.5f ) / W ) - 1.0f;
	}
	for ( int j = 0; j < 24; ++j ) {
		const int axis = j / 8, f = j % 8;
		const float inv_freq = powf( 100.0f, -static_cast<float>( f ) * 0.125f );
		const float ang = 6.283185307179586f * id[axis] * inv_freq;
		cosb[static_cast<size_t>( s ) * 24 + j] = cosf( ang );
		sinb[static_cast<size_t>( s ) * 24 + j] = sinf( ang );
	}
}

// qkv [S, 32 heads, (q|k|v) x 64] + bias -> q/k RMSNorm (no weight) + rotary on the first 48 dims -> zero-padded [S, 32, 128] q/k/v
__global__ void k_vit_qkv( const half* qkv, const float* bias, const float* cosb, const float* sinb, half* qo, half* ko, half* vo, int S ) {
	__shared__ float sm[8][2][64];
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int s = blockIdx.x, head = blockIdx.y * 8 + warp;
	const half* base = qkv + static_cast<size_t>( s ) * 3 * kDim + head * 192;
	const float* bb = bias + head * 192;
	float q[2], k[2], v[2];
	float sq = 0.0f, sk = 0.0f;
	#pragma unroll
	for ( int i = 0; i < 2; ++i ) {
		const int d = lane + 32 * i;
		q[i] = __half2float( base[d] ) + bb[d];
		k[i] = __half2float( base[64 + d] ) + bb[64 + d];
		v[i] = __half2float( base[128 + d] ) + bb[128 + d];
		sq += q[i] * q[i];
		sk += k[i] * k[i];
	}
	for ( int o = 16; o; o >>= 1 ) {
		sq += __shfl_xor_sync( 0xffffffffu, sq, o );
		sk += __shfl_xor_sync( 0xffffffffu, sk, o );
	}
	const float iq = rsqrtf( sq / 64.0f + 1e-5f ), ik = rsqrtf( sk / 64.0f + 1e-5f );
	#pragma unroll
	for ( int i = 0; i < 2; ++i ) {
		sm[warp][0][lane + 32 * i] = q[i] * iq;
		sm[warp][1][lane + 32 * i] = k[i] * ik;
	}
	__syncwarp();
	const size_t ob = ( static_cast<size_t>( s ) * kHeads + head ) * 128;
	#pragma unroll
	for ( int i = 0; i < 2; ++i ) {
		const int d = lane + 32 * i;
		float oq = sm[warp][0][d], ok = sm[warp][1][d];
		if ( d < 48 ) {
			const int j = d % 24;
			const float c = cosb[static_cast<size_t>( s ) * 24 + j], sn = sinb[static_cast<size_t>( s ) * 24 + j];
			if ( d < 24 ) {
				oq = sm[warp][0][d] * c - sm[warp][0][d + 24] * sn;
				ok = sm[warp][1][d] * c - sm[warp][1][d + 24] * sn;
			} else {
				oq = sm[warp][0][d] * c + sm[warp][0][d - 24] * sn;
				ok = sm[warp][1][d] * c + sm[warp][1][d - 24] * sn;
			}
		}
		qo[ob + d] = __float2half_rn( oq );
		ko[ob + d] = __float2half_rn( ok );
		vo[ob + d] = __float2half_rn( v[i] );
	}
}

__global__ void k_unpad( const half* o, half* out, int S ) {
	const size_t n = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( n >= static_cast<size_t>( S ) * kDim ) { return; }
	const size_t s = n / kDim;
	const int c = static_cast<int>( n % kDim );
	out[n] = o[( s * kHeads + c / 64 ) * 128 + c % 64];
}

__global__ void k_layernorm( const float* x, const float* w, const float* b, float* out, int H, float eps ) {
	__shared__ float red[32];
	const int row = blockIdx.x;
	float s = 0.0f;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) { s += x[static_cast<size_t>( row ) * H + c]; }
	const float mean = block_sum( s, red ) / H;
	float v = 0.0f;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) {
		const float d = x[static_cast<size_t>( row ) * H + c] - mean;
		v += d * d;
	}
	const float inv = rsqrtf( block_sum( v, red ) / H + eps );
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) {
		out[static_cast<size_t>( row ) * H + c] = ( x[static_cast<size_t>( row ) * H + c] - mean ) * inv * w[c] + b[c];
	}
}

// proj [N, 3 x 4 x 16 x 16] (token = (t, h, w)) -> pixels [3, 4T, 16H, 16W]
__global__ void k_unpatch( const float* proj, float* pix, int T, int H, int W ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( T ) * H * W * kPatchPix;
	if ( i >= total ) { return; }
	const size_t n = i / kPatchPix;
	const int col = static_cast<int>( i % kPatchPix );
	const int pw = col % 16, ph = ( col / 16 ) % 16, pt = ( col / 256 ) % 4, c = col / 1024;
	const int w = static_cast<int>( n % W ), h = static_cast<int>( ( n / W ) % H ), t = static_cast<int>( n / ( static_cast<size_t>( W ) * H ) );
	const size_t F = static_cast<size_t>( T ) * 4, Hp = static_cast<size_t>( H ) * 16, Wp = static_cast<size_t>( W ) * 16;
	pix[( ( c * F + t * 4 + pt ) * Hp + h * 16 + ph ) * Wp + w * 16 + pw] = proj[i];
}

// crop a latent sub-tile z[:, :, y0:y0+h, x0:x0+w] of [24, T, H, W] into a contiguous [24, T, h, w]
__global__ void k_crop_latent( const float* z, float* out, int T, int H, int W, int y0, int x0, int h, int w ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( 24 ) * T * h * w;
	if ( i >= total ) { return; }
	const int xx = static_cast<int>( i % w ), yy = static_cast<int>( ( i / w ) % h );
	const size_t ct = i / ( static_cast<size_t>( w ) * h ); // c * T + t
	out[i] = z[( ct * H + y0 + yy ) * W + x0 + xx];
}

// Temporal slice of the latent: frames [t0, t0 + n) with frames past T repeating the last one; tile of a whole clip
__global__ void k_clip_latent( const float* z, float* out, int T, int t0, int n, size_t HW ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( 24 ) * n * HW;
	if ( i >= total ) { return; }
	const size_t hw = i % HW;
	const size_t ct = i / HW;
	const int c = static_cast<int>( ct / n ), t = static_cast<int>( ct % n );
	const int src_t = min( t0 + t, T - 1 );
	out[i] = z[( static_cast<size_t>( c ) * T + src_t ) * HW + hw];
}

// Blend a decoded tile with its up / left neighbours (the checkpoint's order: rows first, then columns against the unblended left tile),
// crop its trailing overlaps and write it into the clip buffer at (y0, x0).
__global__ void k_place_tile( const float* tile, const float* up, const float* left, float* dst, int C, int F, int th, int tw, int oy, int ox, int crop_y, int crop_x,
	int y0, int x0, int Hp, int Wp ) {
	const int ch = th - crop_y, cw = tw - crop_x;
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( C ) * F * ch * cw;
	if ( i >= total ) { return; }
	const int x = static_cast<int>( i % cw ), y = static_cast<int>( ( i / cw ) % ch );
	const size_t cf = i / ( static_cast<size_t>( cw ) * ch );
	float v = tile[( cf * th + y ) * tw + x];
	if ( up && y < oy ) {
		const float p = static_cast<float>( y ) / oy;
		v = up[( cf * th + ( th - oy + y ) ) * tw + x] * ( 1.0f - p ) + v * p;
	}
	if ( left && x < ox ) {
		const float p = static_cast<float>( x ) / ox;
		v = left[( cf * th + y ) * tw + ( tw - ox + x )] * ( 1.0f - p ) + v * p;
	}
	dst[( cf * Hp + y0 + y ) * Wp + x0 + x] = v;
}

// out[c, f, :, :] = clip[c, f0 + f, :, :] for f < n (frames are the second axis of [3, F, H, W])
__global__ void k_copy_frames( const float* src, int Fs, int f0, float* dst, int Fd, int d0, int n, size_t plane ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( 3 ) * n * plane;
	if ( i >= total ) { return; }
	const size_t p = i % plane;
	const size_t cf = i / plane;
	const int c = static_cast<int>( cf / n ), f = static_cast<int>( cf % n );
	dst[( static_cast<size_t>( c ) * Fd + d0 + f ) * plane + p] = src[( static_cast<size_t>( c ) * Fs + f0 + f ) * plane + p];
}

// temporal blend: dst frame (d0 + f) = a[f] * (1 - f/n) + src(f0 + f) * (f/n) for f < n; a is [3, n, H, W]
__global__ void k_blend_frames( const float* a, const float* src, int Fs, int f0, float* dst, int Fd, int d0, int n, size_t plane ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( 3 ) * n * plane;
	if ( i >= total ) { return; }
	const size_t p = i % plane;
	const size_t cf = i / plane;
	const int c = static_cast<int>( cf / n ), f = static_cast<int>( cf % n );
	const float w = static_cast<float>( f ) / n;
	const float b = src[( static_cast<size_t>( c ) * Fs + f0 + f ) * plane + p];
	dst[( static_cast<size_t>( c ) * Fd + d0 + f ) * plane + p] = a[i] * ( 1.0f - w ) + b * w;
}

// pixel space -> RGB8: clamp(x * std + mean, 0, 1), frames-last layout [F, H, W, 3]
__global__ void k_to_rgb8( const float* pix, uint8_t* out, int F, size_t plane ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( F ) * plane;
	if ( i >= total ) { return; }
	const size_t p = i % plane;
	const int f = static_cast<int>( i / plane );
	const float mean[3] = { 0.485f, 0.456f, 0.406f }, sd[3] = { 0.229f, 0.224f, 0.225f };
	#pragma unroll
	for ( int c = 0; c < 3; ++c ) {
		float v = pix[( static_cast<size_t>( c ) * F + f ) * plane + p] * sd[c] + mean[c];
		v = fminf( fmaxf( v, 0.0f ), 1.0f );
		out[i * 3 + c] = static_cast<uint8_t>( lrintf( v * 255.0f ) );
	}
}

struct VBlock {
	const half *qkv = nullptr, *out = nullptr, *w1 = nullptr, *w2 = nullptr;
	const float *qkv_b = nullptr, *out_b = nullptr, *w1_b = nullptr, *w2_b = nullptr, *n1 = nullptr, *n2 = nullptr, *s1 = nullptr, *s2 = nullptr;
};

struct Tiles {
	std::vector<int> start, len, overlap;
};

Tiles split_tiles( int input_len, int tile_size, int overlap_min, int ratio ) {
	Tiles t;
	if ( tile_size >= input_len ) {
		t.start = { 0 };
		t.len = { input_len };
		return t;
	}
	int N = ( input_len + tile_size - 1 ) / tile_size;
	std::vector<int> overlaps;
	int remaining = 0;
	for ( ;; ) {
		overlaps.assign( N - 1, overlap_min );
		int sum = 0;
		for ( int o : overlaps ) { sum += o; }
		remaining = tile_size * N - sum - input_len;
		if ( remaining < 0 ) { ++N; } else { break; }
	}
	const int units = remaining / ratio;
	for ( int i = 0; i < units; ++i ) { overlaps[i % ( N - 1 )] += ratio; }
	t.start.push_back( 0 );
	for ( int i = 0; i < N - 1; ++i ) { t.start.push_back( t.start.back() + tile_size - overlaps[i] ); }
	t.len.assign( N, tile_size );
	t.overlap = overlaps;
	return t;
}

} // namespace

struct H3VideoVAE::Impl {
	H3VaeParams par;
	int device = 0, max_tokens = 0;
	pod::SafeTensorsFile st{};
	pod::MemoryArena warena{}, sarena{};
	cublasHandle_t blas = nullptr;
	cudaStream_t stream = nullptr;
	std::vector<VBlock> blocks;
	const float *pq_w = nullptr, *pq_b = nullptr, *emb_w = nullptr, *emb_b = nullptr, *reg = nullptr, *no_w = nullptr, *no_b = nullptr, *po_w = nullptr, *po_b = nullptr;
	const float *d_mean = nullptr, *d_std = nullptr;
	float *x = nullptr, *br = nullptr, *rows = nullptr, *rows2 = nullptr, *cosb = nullptr, *sinb = nullptr, *rowscale = nullptr, *proj = nullptr, *xn = nullptr;
	half *h16 = nullptr, *qkv16 = nullptr, *qp = nullptr, *kp = nullptr, *vp = nullptr, *op = nullptr, *o16 = nullptr, *f1 = nullptr, *f2 = nullptr;

	template <typename T>
	T* alloc( pod::MemoryArena& a, size_t n ) { return static_cast<T*>( engine::core::arena_alloc( a, n * sizeof( T ) ) ); }

	size_t numel( const pod::SafeTensorMeta* m ) {
		size_t n = 1;
		for ( int i = 0; i < m->ndim; ++i ) { n *= static_cast<size_t>( m->shape[i] ); }
		return n;
	}

	const pod::SafeTensorMeta* find( const std::string& name ) {
		const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, name.c_str() );
		if ( !m ) { LOG_ERROR( "H3 VAE: missing tensor %s", name.c_str() ); }
		return m;
	}

	const float* f32( const std::string& name ) {
		const pod::SafeTensorMeta* m = find( name );
		if ( !m ) { return nullptr; }
		ENGINE_ASSERT( m->dtype == pod::DType::FP32, "H3 VAE: expected FP32 tensors" );
		void* d = engine::core::arena_alloc( warena, m->size_bytes );
		CUDA_CHECK( cudaMemcpy( d, engine::loader::safetensors_get_data_cpu( st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
		return static_cast<const float*>( d );
	}

	const half* f16( const std::string& name ) {
		const pod::SafeTensorMeta* m = find( name );
		if ( !m ) { return nullptr; }
		ENGINE_ASSERT( m->dtype == pod::DType::FP32, "H3 VAE: expected FP32 tensors" );
		const size_t n = numel( m );
		void* tmp = nullptr;
		CUDA_CHECK( cudaMalloc( &tmp, m->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( tmp, engine::loader::safetensors_get_data_cpu( st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
		half* d = alloc<half>( warena, n );
		k_convert<float, half><<<1024, 256>>>( static_cast<const float*>( tmp ), d, n );
		CUDA_CHECK( cudaDeviceSynchronize() );
		CUDA_CHECK( cudaFree( tmp ) );
		return d;
	}

	void gemm_h( const half* A, const half* W, void* C, int M, int N, int K, bool c32 ) {
		const float alpha = 1.0f, beta = 0.0f;
		CUBLAS_CHECK( cublasGemmEx( blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C, c32 ? CUDA_R_32F : CUDA_R_16F, N,
			CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
	}

	void gemm_f( const float* A, const float* W, float* C, int M, int N, int K ) {
		const float alpha = 1.0f, beta = 0.0f;
		CUBLAS_CHECK( cublasSgemm( blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, A, K, &beta, C, N ) );
	}

	void tile( const float* z, int T, int H, int W, float* pix ) {
		const int N = T * H * W, S = N + kSuffix;
		if ( S > max_tokens ) { std::fprintf( stderr, "H3 VAE: tile %dx%dx%d (%d tokens) exceeds scratch %d\n", T, H, W, S, max_tokens ); std::_Exit( 1 ); }
		cudaStream_t s = stream;
		k_pack_latent<<<( N + 255 ) / 256, 256, 0, s>>>( z, rows, T, H * W );
		gemm_f( rows, pq_w, rows2, N, 24, 24 );
		k_add_bias<<<256, 256, 0, s>>>( rows2, pq_b, N, 24 );
		gemm_f( rows2, emb_w, x, N, kDim, 24 );
		k_add_bias<<<1024, 256, 0, s>>>( x, emb_b, N, kDim );
		k_set_suffix<<<kSuffix, 256, 0, s>>>( x, reg, N );
		k_vit_rope<<<( S + 127 ) / 128, 128, 0, s>>>( cosb, sinb, T, H, W, S );
		for ( int i = 0; i < kLayers; ++i ) {
			const VBlock& b = blocks[i];
			k_norm_mod<half><<<S, 256, 0, s>>>( x, nullptr, nullptr, 0, b.n1, nullptr, nullptr, 0, nullptr, kDim, 1e-5f, h16 );
			gemm_h( h16, b.qkv, qkv16, S, 3 * kDim, kDim, false );
			k_vit_qkv<<<dim3( S, kHeads / 8 ), 256, 0, s>>>( qkv16, b.qkv_b, cosb, sinb, qp, kp, vp, S );
			h3_flash_attention( qp, kp, vp, op, S, kHeads, 0.125f, 0, S >= 1024 ? 128 : 64, s );
			k_unpad<<<( static_cast<size_t>( S ) * kDim + 255 ) / 256, 256, 0, s>>>( op, o16, S );
			gemm_h( o16, b.out, br, S, kDim, kDim, true );
			k_add_gated<<<S, 256, 0, s>>>( x, br, nullptr, b.s1, 0, nullptr, kDim, b.out_b );
			k_norm_mod<half><<<S, 256, 0, s>>>( x, nullptr, nullptr, 0, b.n2, nullptr, nullptr, 0, nullptr, kDim, 1e-5f, h16 );
			gemm_h( h16, b.w1, f1, S, 2 * kFf, kDim, false );
			k_silu_mul<<<S, 256, 0, s>>>( f1, f2, rowscale, kFf, b.w1_b );
			gemm_h( f2, b.w2, br, S, kDim, kFf, true );
			k_add_gated<<<S, 256, 0, s>>>( x, br, rowscale, b.s2, 0, nullptr, kDim, b.w2_b );
		}
		k_layernorm<<<N, 256, 0, s>>>( x, no_w, no_b, xn, kDim, 1e-5f );
		gemm_f( xn, po_w, proj, N, kPatchPix, kDim );
		k_add_bias<<<1024, 256, 0, s>>>( proj, po_b, N, kPatchPix );
		k_unpatch<<<( static_cast<size_t>( N ) * kPatchPix + 255 ) / 256, 256, 0, s>>>( proj, pix, T, H, W );
	}

	// One temporal clip (denormalized z_clip [24, Tc, H, W]) -> pixels [3, 4 Tc, Hp, Wp] with spatial tiling and blending
	void clip( const float* z_clip, int Tc, int H, int W, float* out ) {
		const int ratio = par.vae_ratio;
		const int Hp = H * ratio, Wp = W * ratio, F = Tc * par.vae_ratio_t;
		const Tiles ty = split_tiles( Hp, par.tile_size, par.tile_overlap_min, ratio ), tx = split_tiles( Wp, par.tile_size, par.tile_overlap_min, ratio );
		const int ni = static_cast<int>( ty.start.size() ), nj = static_cast<int>( tx.start.size() );
		std::vector<float*> tiles( static_cast<size_t>( ni ) * nj, nullptr );
		std::vector<int> th( ni ), tw( nj );
		for ( int i = 0; i < ni; ++i ) { th[i] = ty.len[i]; }
		for ( int j = 0; j < nj; ++j ) { tw[j] = tx.len[j]; }
		for ( int i = 0; i < ni; ++i ) {
			for ( int j = 0; j < nj; ++j ) {
				const int ih = th[i] / ratio, iw = tw[j] / ratio, y0 = ty.start[i] / ratio, x0 = tx.start[j] / ratio;
				float* sub = nullptr;
				const size_t sub_n = static_cast<size_t>( 24 ) * Tc * ih * iw;
				CUDA_CHECK( cudaMalloc( &sub, sub_n * sizeof( float ) ) );
				k_crop_latent<<<( sub_n + 255 ) / 256, 256, 0, stream>>>( z_clip, sub, Tc, H, W, y0, x0, ih, iw );
				float* pix = nullptr;
				CUDA_CHECK( cudaMalloc( &pix, static_cast<size_t>( 3 ) * F * th[i] * tw[j] * sizeof( float ) ) );
				tile( sub, Tc, ih, iw, pix );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				CUDA_CHECK( cudaFree( sub ) );
				tiles[static_cast<size_t>( i ) * nj + j] = pix;
			}
		}
		for ( int i = 0; i < ni; ++i ) {
			for ( int j = 0; j < nj; ++j ) {
				const float* t = tiles[static_cast<size_t>( i ) * nj + j];
				const float* up = i > 0 ? tiles[static_cast<size_t>( i - 1 ) * nj + j] : nullptr;
				const float* left = j > 0 ? tiles[static_cast<size_t>( i ) * nj + j - 1] : nullptr;
				const int oy = i > 0 ? ty.overlap[i - 1] : 0, ox = j > 0 ? tx.overlap[j - 1] : 0;
				const int crop_y = i < ni - 1 ? ty.overlap[i] : 0, crop_x = j < nj - 1 ? tx.overlap[j] : 0;
				const size_t total = static_cast<size_t>( 3 ) * F * ( th[i] - crop_y ) * ( tw[j] - crop_x );
				k_place_tile<<<( total + 255 ) / 256, 256, 0, stream>>>( t, up, left, out, 3, F, th[i], tw[j], oy, ox, crop_y, crop_x, ty.start[i], tx.start[j], Hp, Wp );
			}
		}
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		for ( float* p : tiles ) { CUDA_CHECK( cudaFree( p ) ); }
	}
};

H3VideoVAE::H3VideoVAE() : p_( new Impl ) {}

H3VideoVAE::~H3VideoVAE() {
	if ( p_ ) {
		if ( p_->blas ) { cublasDestroy( p_->blas ); }
		if ( p_->warena.base_ptr ) { engine::core::arena_destroy( p_->warena ); }
		if ( p_->sarena.base_ptr ) { engine::core::arena_destroy( p_->sarena ); }
		engine::loader::safetensors_close( p_->st );
		delete p_;
	}
}

const H3VaeParams& H3VideoVAE::params() const { return p_->par; }
size_t H3VideoVAE::weight_bytes() const { return p_->warena.offset_bytes; }

bool H3VideoVAE::load( const char* dir, int device, int max_tokens ) {
	Impl& s = *p_;
	s.device = device;
	s.max_tokens = max_tokens;
	if ( !h3_read_vae_params( dir, s.par ) ) { return false; }
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &s.stream, cudaStreamNonBlocking ) );
	CUBLAS_CHECK( cublasCreate( &s.blas ) );
	CUBLAS_CHECK( cublasSetStream( s.blas, s.stream ) );
	CUBLAS_CHECK( cublasSetMathMode( s.blas, CUBLAS_DEFAULT_MATH ) );
	if ( !engine::loader::safetensors_open( s.st, ( std::string( dir ) + "/source/model.safetensors" ).c_str() ) ) { return false; }
	// decoder weights: 36 x 67M parameters as FP16, small tensors as FP32
	if ( !engine::core::arena_create( s.warena, device, 6ULL << 30 ) ) { return false; }
	for ( int i = 0; i < kLayers; ++i ) {
		const std::string p = "decoder.transformer_blocks." + std::to_string( i );
		VBlock b;
		b.qkv = s.f16( p + ".attn.to_qkv.weight" );
		b.qkv_b = s.f32( p + ".attn.to_qkv.bias" );
		b.out = s.f16( p + ".attn.to_out.weight" );
		b.out_b = s.f32( p + ".attn.to_out.bias" );
		b.w1 = s.f16( p + ".ff.w1.weight" );
		b.w1_b = s.f32( p + ".ff.w1.bias" );
		b.w2 = s.f16( p + ".ff.w2.weight" );
		b.w2_b = s.f32( p + ".ff.w2.bias" );
		b.n1 = s.f32( p + ".norm1.weight" );
		b.n2 = s.f32( p + ".norm2.weight" );
		b.s1 = s.f32( p + ".scale1" );
		b.s2 = s.f32( p + ".scale2" );
		if ( !b.qkv || !b.s2 ) { return false; }
		s.blocks.push_back( b );
	}
	s.pq_w = s.f32( "post_quant_conv.weight" );
	s.pq_b = s.f32( "post_quant_conv.bias" );
	s.emb_w = s.f32( "decoder.x_embedder.weight" );
	s.emb_b = s.f32( "decoder.x_embedder.bias" );
	s.reg = s.f32( "decoder.register_tokens" );
	s.no_w = s.f32( "decoder.norm_out.weight" );
	s.no_b = s.f32( "decoder.norm_out.bias" );
	s.po_w = s.f32( "decoder.proj_out.weight" );
	s.po_b = s.f32( "decoder.proj_out.bias" );
	if ( !s.po_b ) { return false; }
	{
		float* dm = s.alloc<float>( s.warena, 24 );
		float* ds = s.alloc<float>( s.warena, 24 );
		CUDA_CHECK( cudaMemcpy( dm, s.par.mean.data(), 24 * 4, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( ds, s.par.stdev.data(), 24 * 4, cudaMemcpyHostToDevice ) );
		s.d_mean = dm;
		s.d_std = ds;
	}
	const size_t S = static_cast<size_t>( max_tokens );
	size_t need = S * kDim * 4 * 3 + S * 24 * 4 * 4 + S * 4 + S * kPatchPix * 4 + S * kDim * 2 * 2 + S * 3 * kDim * 2 + S * kHeads * 128 * 2 * 4 + S * 2 * kFf * 2 + S * kFf * 2 + ( 64ULL << 20 );
	if ( !engine::core::arena_create( s.sarena, device, need ) ) { return false; }
	s.x = s.alloc<float>( s.sarena, S * kDim );
	s.br = s.alloc<float>( s.sarena, S * kDim );
	s.xn = s.alloc<float>( s.sarena, S * kDim );
	s.rows = s.alloc<float>( s.sarena, S * 24 );
	s.rows2 = s.alloc<float>( s.sarena, S * 24 );
	s.cosb = s.alloc<float>( s.sarena, S * 24 );
	s.sinb = s.alloc<float>( s.sarena, S * 24 );
	s.rowscale = s.alloc<float>( s.sarena, S );
	s.proj = s.alloc<float>( s.sarena, S * kPatchPix );
	s.h16 = s.alloc<half>( s.sarena, S * kDim );
	s.o16 = s.alloc<half>( s.sarena, S * kDim );
	s.qkv16 = s.alloc<half>( s.sarena, S * 3 * kDim );
	s.qp = s.alloc<half>( s.sarena, S * kHeads * 128 );
	s.kp = s.alloc<half>( s.sarena, S * kHeads * 128 );
	s.vp = s.alloc<half>( s.sarena, S * kHeads * 128 );
	s.op = s.alloc<half>( s.sarena, S * kHeads * 128 );
	s.f1 = s.alloc<half>( s.sarena, S * 2 * kFf );
	s.f2 = s.alloc<half>( s.sarena, S * kFf );
	// padded head dims stay zero: the kernel writes only dims 0..63
	CUDA_CHECK( cudaMemset( s.qp, 0, S * kHeads * 128 * 2 ) );
	CUDA_CHECK( cudaMemset( s.kp, 0, S * kHeads * 128 * 2 ) );
	CUDA_CHECK( cudaMemset( s.vp, 0, S * kHeads * 128 * 2 ) );
	return h3_fmha_available();
}

void H3VideoVAE::decode_tile( const float* d_z, int T, int H, int W, float* d_pix ) {
	p_->tile( d_z, T, H, W, d_pix );
	CUDA_CHECK( cudaStreamSynchronize( p_->stream ) );
}

bool H3VideoVAE::decode_video( const float* d_z, int T, int H, int W, std::vector<uint8_t>& rgb, int& frames ) {
	Impl& s = *p_;
	const H3VaeParams& P = s.par;
	const int rt = P.vae_ratio_t, ratio = P.vae_ratio;
	const int tokens_chunk = ( P.clip_length + rt - 1 ) / rt;                          // 5
	const int frame_pre_padding = ( ( -P.clip_length ) % rt + rt ) % rt;               // 3
	const int token_overlap = ( ( -P.token_drop ) % tokens_chunk + tokens_chunk ) % tokens_chunk; // 2
	const int frame_overlap = std::max( token_overlap * rt - frame_pre_padding, 0 );   // 5
	const int chunk_dec = tokens_chunk * rt;                                           // 20
	const int split_count = P.token_drop > 0 ? 2 : 1;
	const int Hp = H * ratio, Wp = W * ratio;
	const size_t plane = static_cast<size_t>( Hp ) * Wp, HW = static_cast<size_t>( H ) * W;

	// denormalized copy of the latent
	float* zd = nullptr;
	CUDA_CHECK( cudaMalloc( &zd, static_cast<size_t>( 24 ) * T * HW * sizeof( float ) ) );
	CUDA_CHECK( cudaMemcpy( zd, d_z, static_cast<size_t>( 24 ) * T * HW * sizeof( float ), cudaMemcpyDeviceToDevice ) );
	k_denorm_latent<<<( 24 * static_cast<size_t>( T ) * HW + 255 ) / 256, 256, 0, s.stream>>>( zd, s.d_mean, s.d_std, static_cast<size_t>( T ) * HW );

	// temporal plan (decode_temporal, non-isolated config)
	const int pseudo = T + P.token_drop;
	const int rem = pseudo % tokens_chunk;
	const int pad_tokens = rem ? tokens_chunk - rem : 0;
	const int num_chunks = ( pseudo + pad_tokens ) / tokens_chunk - ( P.token_drop > 0 ? 1 : 0 );
	int pad_frames = 0;
	{
		const int intra_tail = P.clip_length % rt;
		if ( pad_tokens > 0 ) {
			if ( intra_tail == 0 ) {
				pad_frames = pad_tokens * rt;
			} else {
				for ( int k = 0; k < pad_tokens; ++k ) { pad_frames += ( T + k ) % tokens_chunk == 0 ? intra_tail : rt; }
			}
		}
	}
	// upper bound of the produced frames: each chunk yields <= chunk_dec - pre_padding frames per part
	const int max_frames = num_chunks * chunk_dec + 16;
	float* dec = nullptr;
	CUDA_CHECK( cudaMalloc( &dec, static_cast<size_t>( 3 ) * max_frames * plane * sizeof( float ) ) );
	float* overlap = nullptr; // [3, frame_overlap.., H, W]: the previous clip's second part
	int overlap_frames = 0;
	int write = 0;
	for ( int i = 0; i < num_chunks; ++i ) {
		const int t0 = i * tokens_chunk, n = tokens_chunk + token_overlap;
		float* zc = nullptr;
		CUDA_CHECK( cudaMalloc( &zc, static_cast<size_t>( 24 ) * n * HW * sizeof( float ) ) );
		k_clip_latent<<<( static_cast<size_t>( 24 ) * n * HW + 255 ) / 256, 256, 0, s.stream>>>( zd, zc, T, t0, n, HW );
		const int Fc = n * rt;
		float* cd = nullptr;
		CUDA_CHECK( cudaMalloc( &cd, static_cast<size_t>( 3 ) * Fc * plane * sizeof( float ) ) );
		s.clip( zc, n, H, W, cd );
		CUDA_CHECK( cudaFree( zc ) );
		for ( int j = 0; j < split_count; ++j ) {
			const int f_start = j * chunk_dec;
			const int f_end = std::min( f_start + chunk_dec, Fc );
			const int a = f_start + frame_pre_padding, nf = f_end - a;
			if ( nf <= 0 ) { continue; }
			if ( j == 0 ) {
				if ( overlap ) {
					const int nb = std::min( frame_overlap, std::min( overlap_frames, nf ) );
					ENGINE_ASSERT( nb == overlap_frames, "H3 VAE: temporal overlap must equal the blend extent" );
					k_blend_frames<<<( static_cast<size_t>( 3 ) * nb * plane + 255 ) / 256, 256, 0, s.stream>>>( overlap, cd, Fc, a, dec, max_frames, write, nb, plane );
					if ( nb < nf ) {
						k_copy_frames<<<( static_cast<size_t>( 3 ) * ( nf - nb ) * plane + 255 ) / 256, 256, 0, s.stream>>>( cd, Fc, a + nb, dec, max_frames, write + nb, nf - nb, plane );
					}
					CUDA_CHECK( cudaStreamSynchronize( s.stream ) );
					CUDA_CHECK( cudaFree( overlap ) );
					overlap = nullptr;
				} else {
					k_copy_frames<<<( static_cast<size_t>( 3 ) * nf * plane + 255 ) / 256, 256, 0, s.stream>>>( cd, Fc, a, dec, max_frames, write, nf, plane );
				}
				write += nf;
			} else {
				CUDA_CHECK( cudaMalloc( &overlap, static_cast<size_t>( 3 ) * nf * plane * sizeof( float ) ) );
				// compact copy of the part (frame axis nf)
				k_copy_frames<<<( static_cast<size_t>( 3 ) * nf * plane + 255 ) / 256, 256, 0, s.stream>>>( cd, Fc, a, overlap, nf, 0, nf, plane );
				overlap_frames = nf;
			}
		}
		CUDA_CHECK( cudaStreamSynchronize( s.stream ) );
		CUDA_CHECK( cudaFree( cd ) );
	}
	if ( overlap ) {
		k_copy_frames<<<( static_cast<size_t>( 3 ) * overlap_frames * plane + 255 ) / 256, 256, 0, s.stream>>>( overlap, overlap_frames, 0, dec, max_frames, write, overlap_frames, plane );
		write += overlap_frames;
		CUDA_CHECK( cudaStreamSynchronize( s.stream ) );
		CUDA_CHECK( cudaFree( overlap ) );
	}
	frames = write - pad_frames;
	if ( frames <= 0 ) { return false; }
	// dec is [3, max_frames, ...]; compact to [3, frames, ...] for the converter
	float* comp = nullptr;
	CUDA_CHECK( cudaMalloc( &comp, static_cast<size_t>( 3 ) * frames * plane * sizeof( float ) ) );
	k_copy_frames<<<( static_cast<size_t>( 3 ) * frames * plane + 255 ) / 256, 256, 0, s.stream>>>( dec, max_frames, 0, comp, frames, 0, frames, plane );
	uint8_t* d8 = nullptr;
	CUDA_CHECK( cudaMalloc( &d8, static_cast<size_t>( frames ) * plane * 3 ) );
	k_to_rgb8<<<( static_cast<size_t>( frames ) * plane + 255 ) / 256, 256, 0, s.stream>>>( comp, d8, frames, plane );
	rgb.resize( static_cast<size_t>( frames ) * plane * 3 );
	CUDA_CHECK( cudaMemcpyAsync( rgb.data(), d8, rgb.size(), cudaMemcpyDeviceToHost, s.stream ) );
	CUDA_CHECK( cudaStreamSynchronize( s.stream ) );
	CUDA_CHECK( cudaFree( d8 ) );
	CUDA_CHECK( cudaFree( comp ) );
	CUDA_CHECK( cudaFree( dec ) );
	CUDA_CHECK( cudaFree( zd ) );
	return true;
}

} // namespace engine::video
