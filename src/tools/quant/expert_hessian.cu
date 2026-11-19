#include "expert_hessian.h"
#include "engine/config.h"
#include "engine/core/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <sys/stat.h>

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

// out (c x n) = a (c x n) times w[r] per row r
__global__ void k_rows_times( const float* a, const float* w, int c, int n, float* out ) {
	for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < (size_t) c * n; i += (size_t) gridDim.x * blockDim.x ) out[i] = a[i] * w[i / n];
}

// out (c x n) = rows idx[0 .. c) of x (FP16, n wide) times sqrt( w )
__global__ void k_gather_scaled( const half* x, const int* idx, const float* w, int c, int n, float* out ) {
	for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < (size_t) c * n; i += (size_t) gridDim.x * blockDim.x ) {
		const size_t r = i / n;
		out[i] = __half2float( x[(size_t) idx[r] * n + i % n] ) * sqrtf( fmaxf( w[r], 0.0f ) );
	}
}

// y = a x
__global__ void k_scale_into( float* y, const float* x, float a, size_t n ) {
	for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x ) y[i] = a * x[i];
}

// y = a y + x
__global__ void k_axpy_scaled( float* y, const float* x, float a, size_t n ) {
	for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x ) y[i] = a * y[i] + x[i];
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
	int64_t pos = 0, wpos = 0;
	int hdr[4];
	while ( std::fread( hdr, sizeof( int ), 4, f ) == 4 ) {
		const int layer = hdr[0], rows = hdr[1], K = hdr[3];
		if ( rows <= 0 || K <= 0 ) {
			LOG_ERROR( "[volta_quantize] %s: bad record at byte %lld", path.c_str(), static_cast<long long>( pos ) );
			std::fclose( f );
			return false;
		}
		const int64_t ids = static_cast<int64_t>( rows ) * K * 4, data = static_cast<int64_t>( rows ) * hidden * 2;
		by_layer_[layer].push_back( { pos + 16 + ids, rows, K, wpos } );
		pos += 16 + ids + data;
		wpos += ids;
		if ( fseeko( f, pos, SEEK_SET ) != 0 ) break;
	}
	std::fclose( f );
	// The routing weights, when the dump has them for every record
	struct stat st {};
	has_w_ = stat( ( path + ".w" ).c_str(), &st ) == 0 && static_cast<int64_t>( st.st_size ) == wpos;
	if ( stat( ( path + ".w" ).c_str(), &st ) == 0 && !has_w_ ) LOG_WARN( "[volta_quantize] %s.w does not match the dump's records: no routing weights", path.c_str() );
	return true;
}

bool RouteRows::routes( int layer, int64_t max_rows, std::vector<int>& ids, std::vector<float>& w, int& K ) const {
	const auto it = by_layer_.find( layer );
	if ( !has_w_ || it == by_layer_.end() ) return false;
	K = it->second.front().k;
	int64_t total = rows( layer );
	if ( max_rows > 0 && total > max_rows ) total = max_rows;
	ids.resize( static_cast<size_t>( total ) * K );
	w.resize( ids.size() );
	FILE* f = std::fopen( path_.c_str(), "rb" );
	FILE* fw = std::fopen( ( path_ + ".w" ).c_str(), "rb" );
	bool ok = f && fw;
	int64_t done = 0;
	for ( const auto& r : it->second ) {
		if ( !ok || done >= total ) break;
		const int64_t take = r.rows < total - done ? r.rows : total - done;
		const size_t n = static_cast<size_t>( take ) * K;
		ok = r.k == K && fseeko( f, r.offset - static_cast<int64_t>( r.rows ) * r.k * 4, SEEK_SET ) == 0 &&
			std::fread( ids.data() + static_cast<size_t>( done ) * K, 4, n, f ) == n && fseeko( fw, r.w_offset, SEEK_SET ) == 0 &&
			std::fread( w.data() + static_cast<size_t>( done ) * K, 4, n, fw ) == n;
		done += take;
	}
	if ( f ) std::fclose( f );
	if ( fw ) std::fclose( fw );
	return ok;
}

int64_t RouteRows::rows( int layer ) const {
	const auto it = by_layer_.find( layer );
	if ( it == by_layer_.end() ) return 0;
	int64_t n = 0;
	for ( const auto& r : it->second ) n += r.rows;
	return n;
}

bool RouteRows::counts( int layer, int experts, std::vector<int64_t>& out ) const {
	out.assign( experts, 0 );
	const auto it = by_layer_.find( layer );
	if ( it == by_layer_.end() ) return false;
	FILE* f = std::fopen( path_.c_str(), "rb" );
	if ( !f ) return false;
	std::vector<int32_t> ids;
	bool ok = true;
	for ( const auto& r : it->second ) {
		ids.resize( static_cast<size_t>( r.rows ) * r.k );
		ok = fseeko( f, r.offset - static_cast<int64_t>( ids.size() ) * 4, SEEK_SET ) == 0 && std::fread( ids.data(), 4, ids.size(), f ) == ids.size();
		if ( !ok ) break;
		for ( int32_t e : ids ) {
			if ( e >= 0 && e < experts ) out[e]++;
		}
	}
	std::fclose( f );
	return ok;
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
	if ( aff_ ) cudaFree( aff_ );
}

bool ExpertHessian::compute( const void* x, int64_t rows, int hidden, const float* gate, const float* up, int inter, float limit, float* H,
	backend::stream_t stream, const float* row_w, double w_sum, float eps ) {
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
	const size_t hh = static_cast<size_t>( inter ) * inter;
	CUDA_CHECK( cudaMemsetAsync( H, 0, hh * sizeof( float ), st ) );
	float* Haff = nullptr;
	if ( row_w ) {
		if ( !grow( aff_, aff_bytes_, hh * sizeof( float ) ) ) return false;
		Haff = static_cast<float*>( aff_ );
		CUDA_CHECK( cudaMemsetAsync( Haff, 0, hh * sizeof( float ), st ) );
	}
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
		if ( Haff ) {
			// Haff += A^T diag( w ) A, the weighted rows in U (free after the activation)
			k_rows_times<<<blocks_for( n ), 256, 0, st>>>( G, row_w + r0, c, inter, U );
			EH_BLAS( cublasSgemm( blas, CUBLAS_OP_N, CUBLAS_OP_T, inter, inter, c, &one, G, inter, U, inter, &one, Haff, inter ) );
		}
	}
	// H = eps (S / N) H + Haff
	if ( Haff ) k_axpy_scaled<<<blocks_for( hh ), 256, 0, st>>>( H, Haff, static_cast<float>( eps * w_sum / static_cast<double>( rows ) ), hh );
	CUDA_CHECK( cudaPeekAtLastError() );
	return true;
}

bool ExpertHessian::affinity_input( const void* x, int64_t rows, int hidden, const int* routed, const float* routed_w, int n_routed, double w_sum,
	float eps, const float* H_in, float* H, backend::stream_t stream ) {
	auto st = static_cast<cudaStream_t>( stream );
	if ( !blas_ ) {
		cublasHandle_t h = nullptr;
		EH_BLAS( cublasCreate( &h ) );
		blas_ = h;
	}
	auto blas = static_cast<cublasHandle_t>( blas_ );
	EH_BLAS( cublasSetStream( blas, st ) );
	const size_t hh = static_cast<size_t>( hidden ) * hidden;
	if ( !grow( aff_, aff_bytes_, static_cast<size_t>( CHUNK ) * hidden * sizeof( float ) ) ) return false;
	float* Xs = static_cast<float*>( aff_ );
	// H = eps (S / N) H_in, then += (sqrt( w ) x)^T (sqrt( w ) x) / N over the routed rows
	k_scale_into<<<blocks_for( hh ), 256, 0, st>>>( H, H_in, static_cast<float>( eps * w_sum / static_cast<double>( rows ) ), hh );
	const float inv_n = static_cast<float>( 1.0 / static_cast<double>( rows ) ), one = 1.0f;
	for ( int r0 = 0; r0 < n_routed; r0 += CHUNK ) {
		const int c = n_routed - r0 < CHUNK ? n_routed - r0 : CHUNK;
		k_gather_scaled<<<blocks_for( static_cast<size_t>( c ) * hidden ), 256, 0, st>>>( static_cast<const half*>( x ), routed + r0, routed_w + r0, c,
			hidden, Xs );
		EH_BLAS( cublasSgemm( blas, CUBLAS_OP_N, CUBLAS_OP_T, hidden, hidden, c, &inv_n, Xs, hidden, Xs, hidden, &one, H, hidden ) );
	}
	CUDA_CHECK( cudaPeekAtLastError() );
	return true;
}

} // namespace engine::tools::quant
