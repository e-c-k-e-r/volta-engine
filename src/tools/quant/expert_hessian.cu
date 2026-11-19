#include "expert_hessian.h"
#include "engine/config.h"
#include "engine/core/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>

namespace engine::tools::quant {

namespace {

constexpr int CHUNK = 8192;   // rows per activation pass

#define EH_BLAS( x ) \
	do { \
		const cublasStatus_t s_ = ( x ); \
		ENGINE_ASSERT( s_ == CUBLAS_STATUS_SUCCESS, "expert_hessian: cuBLAS call failed" ); \
	} while ( 0 )

__global__ void k_to_f16( const float* in, half* out, size_t n ) {
	for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x ) out[i] = __float2half_rn( in[i] );
}

// g <- act( g ) * u, in place (limit 0: plain SiLU)
__global__ void k_swiglu( float* g, const float* u, size_t n, float limit ) {
	for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x ) {
		float a = g[i], b = u[i];
		if ( limit > 0.0f ) {
			a = fminf( a, limit );
			b = fminf( fmaxf( b, -limit ), limit );
		}
		g[i] = a / ( 1.0f + __expf( -a ) ) * b;
	}
}

unsigned blocks_for( size_t n ) { return static_cast<unsigned>( ( n + 255 ) / 256 < 65535u * 8u ? ( n + 255 ) / 256 : 65535u * 8u ); }

bool grow( void*& p, size_t& have, size_t want ) {
	if ( want <= have ) return true;
	if ( p ) CUDA_CHECK( cudaFree( p ) );
	p = nullptr;
	have = 0;
	if ( cudaMalloc( &p, want ) != cudaSuccess ) {
		(void) cudaGetLastError();
		p = nullptr;
		return false;
	}
	have = want;
	return true;
}

} // namespace

bool RouteRows::open( const std::string& path, int hidden ) {
	path_ = path;
	hidden_ = hidden;
	by_layer_.clear();
	FILE* f = std::fopen( path.c_str(), "rb" );
	if ( !f ) {
		LOG_ERROR( "[volta_quantize] cannot open the route dump %s", path.c_str() );
		return false;
	}
	int64_t pos = 0;
	int hdr[4];
	while ( std::fread( hdr, sizeof( int ), 4, f ) == 4 ) {
		const int layer = hdr[0], rows = hdr[1], K = hdr[3];
		if ( rows <= 0 || K <= 0 ) {
			LOG_ERROR( "[volta_quantize] %s: bad record at byte %lld", path.c_str(), static_cast<long long>( pos ) );
			std::fclose( f );
			return false;
		}
		const int64_t ids = static_cast<int64_t>( rows ) * K * 4, data = static_cast<int64_t>( rows ) * hidden * 2;
		by_layer_[layer].push_back( { pos + 16 + ids, rows } );
		pos += 16 + ids + data;
		if ( fseeko( f, pos, SEEK_SET ) != 0 ) break;
	}
	std::fclose( f );
	return true;
}

int64_t RouteRows::rows( int layer ) const {
	const auto it = by_layer_.find( layer );
	if ( it == by_layer_.end() ) return 0;
	int64_t n = 0;
	for ( const auto& r : it->second ) n += r.rows;
	return n;
}

bool RouteRows::read( int layer, int64_t max_rows, std::vector<uint16_t>& out ) const {
	const auto it = by_layer_.find( layer );
	if ( it == by_layer_.end() ) return false;
	int64_t total = rows( layer );
	if ( max_rows > 0 && total > max_rows ) total = max_rows;
	out.resize( static_cast<size_t>( total ) * hidden_ );
	FILE* f = std::fopen( path_.c_str(), "rb" );
	if ( !f ) return false;
	int64_t done = 0;
	for ( const auto& r : it->second ) {
		if ( done >= total ) break;
		const int64_t take = r.rows < total - done ? r.rows : total - done;
		if ( fseeko( f, r.offset, SEEK_SET ) != 0 ||
			std::fread( out.data() + static_cast<size_t>( done ) * hidden_, 2, static_cast<size_t>( take ) * hidden_, f ) !=
				static_cast<size_t>( take ) * hidden_ ) {
			std::fclose( f );
			return false;
		}
		done += take;
	}
	std::fclose( f );
	return true;
}

ExpertHessian::~ExpertHessian() {
	if ( blas_ ) cublasDestroy( static_cast<cublasHandle_t>( blas_ ) );
	if ( w16_ ) cudaFree( w16_ );
	if ( g_ ) cudaFree( g_ );
}

bool ExpertHessian::compute( const void* x, int64_t rows, int hidden, const float* gate, const float* up, int inter, float limit, float* H,
	backend::stream_t stream ) {
	auto st = static_cast<cudaStream_t>( stream );
	if ( !blas_ ) {
		cublasHandle_t h = nullptr;
		EH_BLAS( cublasCreate( &h ) );
		blas_ = h;
	}
	auto blas = static_cast<cublasHandle_t>( blas_ );
	EH_BLAS( cublasSetStream( blas, st ) );
	const size_t wn = static_cast<size_t>( hidden ) * inter;
	if ( !grow( w16_, w16_bytes_, 2 * wn * sizeof( half ) ) ) return false;
	void* gb = g_;
	if ( !grow( gb, g_bytes_, 2 * static_cast<size_t>( CHUNK ) * inter * sizeof( float ) ) ) return false;
	g_ = static_cast<float*>( gb );
	half* wg = static_cast<half*>( w16_ );
	half* wu = wg + wn;
	k_to_f16<<<blocks_for( wn ), 256, 0, st>>>( gate, wg, wn );
	k_to_f16<<<blocks_for( wn ), 256, 0, st>>>( up, wu, wn );
	CUDA_CHECK( cudaMemsetAsync( H, 0, static_cast<size_t>( inter ) * inter * sizeof( float ), st ) );
	const float one = 1.0f, zero = 0.0f;
	float* G = g_;
	float* U = g_ + static_cast<size_t>( CHUNK ) * inter;
	for ( int64_t r0 = 0; r0 < rows; r0 += CHUNK ) {
		const int c = static_cast<int>( rows - r0 < CHUNK ? rows - r0 : CHUNK );
		const half* xc = static_cast<const half*>( x ) + static_cast<size_t>( r0 ) * hidden;
		// Row-major G (c x inter) = x (c x hidden) W (hidden x inter): the column-major product W^T-view times x-view
		EH_BLAS( cublasGemmEx( blas, CUBLAS_OP_N, CUBLAS_OP_N, inter, c, hidden, &one, wg, CUDA_R_16F, inter, xc, CUDA_R_16F, hidden, &zero, G,
			CUDA_R_32F, inter, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
		EH_BLAS( cublasGemmEx( blas, CUBLAS_OP_N, CUBLAS_OP_N, inter, c, hidden, &one, wu, CUDA_R_16F, inter, xc, CUDA_R_16F, hidden, &zero, U,
			CUDA_R_32F, inter, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
		const size_t n = static_cast<size_t>( c ) * inter;
		k_swiglu<<<blocks_for( n ), 256, 0, st>>>( G, U, n, limit );
		// H (inter x inter) += A^T A, A = G (c x inter): column-major H = A-view (inter x c) times its transpose
		EH_BLAS( cublasSgemm( blas, CUBLAS_OP_N, CUBLAS_OP_T, inter, inter, c, &one, G, inter, G, inter, &one, H, inter ) );
	}
	CUDA_CHECK( cudaPeekAtLastError() );
	return true;
}

} // namespace engine::tools::quant
