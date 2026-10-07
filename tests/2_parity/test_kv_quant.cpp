// Quantized KV cache accuracy: every format against the FP16 cache on the same random K/V (K with outlier channels),
// through the store kernel and both the decode and prefill attention kernels, for the per-rank Qwen shape (GQA 6/1) and
// the GLM shape (16 heads, no GQA). Reports cosine and relative RMS error of the attention output vs the FP16 cache.
//   test_kv_quant [positions]
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/modules/attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

uint64_t g_state = 0x9E3779B97F4A7C15ull;
float uni() {  // splitmix64: no lattice structure between consecutive draws
	uint64_t z = ( g_state += 0x9E3779B97F4A7C15ull );
	z = ( z ^ ( z >> 30 ) ) * 0xBF58476D1CE4E5B9ull;
	z = ( z ^ ( z >> 27 ) ) * 0x94D049BB133111EBull;
	z ^= z >> 31;
	return ( z >> 40 ) / 16777216.0f;
}
float normal() { return uni() + uni() + uni() + uni() - 2.0f; }  // ~N(0, 0.33), scaled by the caller

struct Cache {
	pod::KVRef ref;
	size_t elems = 0;
	void* bufs[4] = {};
};

Cache make_cache( pod::KVFormat f, size_t elems ) {
	Cache c;
	c.elems = elems;
	c.ref.kq = pod::kv_k_quant( f );
	c.ref.vq = pod::kv_v_quant( f );
	CUDA_CHECK( cudaMalloc( &c.ref.k, pod::kv_data_bytes( c.ref.kq, elems ) ) );
	CUDA_CHECK( cudaMalloc( &c.ref.v, pod::kv_data_bytes( c.ref.vq, elems ) ) );
	CUDA_CHECK( cudaMemset( c.ref.k, 0, pod::kv_data_bytes( c.ref.kq, elems ) ) );
	CUDA_CHECK( cudaMemset( c.ref.v, 0, pod::kv_data_bytes( c.ref.vq, elems ) ) );
	if ( pod::kv_scale_bytes( c.ref.kq, elems ) ) {
		CUDA_CHECK( cudaMalloc( &c.ref.ks, pod::kv_scale_bytes( c.ref.kq, elems ) ) );
		CUDA_CHECK( cudaMemset( c.ref.ks, 0, pod::kv_scale_bytes( c.ref.kq, elems ) ) );
	}
	if ( pod::kv_scale_bytes( c.ref.vq, elems ) ) {
		CUDA_CHECK( cudaMalloc( &c.ref.vs, pod::kv_scale_bytes( c.ref.vq, elems ) ) );
		CUDA_CHECK( cudaMemset( c.ref.vs, 0, pod::kv_scale_bytes( c.ref.vq, elems ) ) );
	}
	return c;
}

void free_cache( Cache& c ) {
	cudaFree( c.ref.k ); cudaFree( c.ref.v ); cudaFree( c.ref.ks ); cudaFree( c.ref.vs );
}

pod::TensorView view( void* p, int rows, int cols ) {
	pod::TensorView t{};
	t.data = p; t.dtype = pod::DType::FP16; t.ndim = 2; t.shape[0] = rows; t.shape[1] = cols;
	return t;
}

}  // namespace

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const int N = argc > 1 ? std::atoi( argv[1] ) : 3000;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );
	const int hd = 256, max_seq = 4096;

	struct Shape { const char* name; int qh, kvh; };
	const Shape shapes[] = { { "qwen 6/1", 6, 1 }, { "glm 16/16", 16, 16 } };
	const pod::KVFormat formats[] = { pod::KVFormat::FP8, pod::KVFormat::K8V4, pod::KVFormat::NVFP4 };
	const char* fname[] = { "fp8", "k8v4", "nvfp4" };
	bool all_ok = true;

	for ( const Shape& sh : shapes ) {
		const int tot = sh.kvh * hd;
		const size_t elems = static_cast<size_t>( max_seq ) * tot;  // one layer, one slot
		// Random K/V for N positions, K with outlier channels
		std::vector<half> hk( static_cast<size_t>( N ) * tot ), hv( hk.size() );
		for ( int t = 0; t < N; ++t ) {
			for ( int c = 0; c < tot; ++c ) {
				float kvv = normal() * 1.5f;
				const int ch = c % hd;
				if ( ch == 3 || ch == 77 || ch == 150 || ch == 201 ) kvv *= 6.0f;
				hk[static_cast<size_t>( t ) * tot + c] = __float2half( kvv );
				hv[static_cast<size_t>( t ) * tot + c] = __float2half( normal() * 1.5f );
			}
		}
		half *d_k, *d_v;
		CUDA_CHECK( cudaMalloc( &d_k, hk.size() * 2 ) );
		CUDA_CHECK( cudaMalloc( &d_v, hv.size() * 2 ) );
		CUDA_CHECK( cudaMemcpy( d_k, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_v, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice ) );
		pod::BatchRows* d_rows;
		CUDA_CHECK( cudaMalloc( &d_rows, sizeof( pod::BatchRows ) ) );

		// Queries: 64 rows at the end of the context (the prefill kernel needs > 16 rows; Qwen shape only)
		const int Q = 64;
		std::vector<half> hq( static_cast<size_t>( Q ) * sh.qh * hd );
		for ( auto& v : hq ) v = __float2half( normal() * 1.0f );
		half *d_q, *d_o_ref, *d_o;
		CUDA_CHECK( cudaMalloc( &d_q, hq.size() * 2 ) );
		CUDA_CHECK( cudaMalloc( &d_o_ref, hq.size() * 2 ) );
		CUDA_CHECK( cudaMalloc( &d_o, hq.size() * 2 ) );
		CUDA_CHECK( cudaMemcpy( d_q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice ) );
		pod::TensorView q = view( d_q, Q, sh.qh * hd ), o_ref = view( d_o_ref, Q, sh.qh * hd ), o = view( d_o, Q, sh.qh * hd );
		pod::TensorView none{};

		auto fill = [&]( Cache& c ) {
			for ( int base = 0; base < N; base += pod::MAX_ROWS ) {
				const int cnt = std::min( static_cast<int>( pod::MAX_ROWS ), N - base );
				const auto rows = pod::BatchRows::causal( cnt, base, 0 );
				CUDA_CHECK( cudaMemcpy( d_rows, &rows, sizeof( rows ), cudaMemcpyHostToDevice ) );
				engine::modules::Attention::store_kv_cache( c.ref, view( d_k + static_cast<size_t>( base ) * tot, cnt, tot ),
					view( d_v + static_cast<size_t>( base ) * tot, cnt, tot ), 0, 0, d_rows, cnt, sh.kvh, hd, max_seq, s );
			}
		};
		auto attend = [&]( Cache& c, pod::TensorView& out, bool prefill ) {
			const auto rows = pod::BatchRows::causal( Q, N - Q, 0 );
			CUDA_CHECK( cudaMemcpy( d_rows, &rows, sizeof( rows ), cudaMemcpyHostToDevice ) );
			if ( prefill ) engine::modules::Attention::forward_prefill( c.ref, q, none, out, 0, 0, d_rows, Q, sh.qh, sh.kvh, hd, max_seq, s );
			else engine::modules::Attention::forward_decode( c.ref, q, none, out, 0, 0, d_rows, Q, sh.qh, sh.kvh, hd, max_seq, s );
			CUDA_CHECK( cudaStreamSynchronize( s ) );
		};
		auto compare = [&]( const char* what, const char* fmt, double min_cos ) {
			std::vector<half> a( hq.size() ), b( hq.size() );
			CUDA_CHECK( cudaMemcpy( a.data(), d_o_ref, a.size() * 2, cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( b.data(), d_o, b.size() * 2, cudaMemcpyDeviceToHost ) );
			double dot = 0, na = 0, nb = 0, err = 0;
			for ( size_t i = 0; i < a.size(); ++i ) {
				const double x = __half2float( a[i] ), y = __half2float( b[i] );
				dot += x * y; na += x * x; nb += y * y; err += ( x - y ) * ( x - y );
			}
			const double cosv = dot / std::sqrt( na * nb + 1e-30 ), rel = std::sqrt( err / ( na + 1e-30 ) );
			const bool ok = cosv >= min_cos;
			all_ok = all_ok && ok;
			std::printf( "  %-9s %-6s %-7s cos=%.6f  rel_rms=%.4f  %s\n", sh.name, fmt, what, cosv, rel, ok ? "ok" : "FAIL" );
		};

		Cache ref = make_cache( pod::KVFormat::FP16, elems );
		fill( ref );
		attend( ref, o_ref, false );
		for ( int fi = 0; fi < 3; ++fi ) {
			Cache c = make_cache( formats[fi], elems );
			fill( c );
			const double min_cos = formats[fi] == pod::KVFormat::FP8 ? 0.997 : ( formats[fi] == pod::KVFormat::K8V4 ? 0.99 : 0.98 );
			attend( c, o, false );
			compare( "decode", fname[fi], min_cos );
			{
				// Query at position 0 sees one key, so the output is that position's V as stored: V codec round trip
				const auto rows0 = pod::BatchRows::causal( Q, 0, 0 );
				CUDA_CHECK( cudaMemcpy( d_rows, &rows0, sizeof( rows0 ), cudaMemcpyHostToDevice ) );
				engine::modules::Attention::forward_decode( c.ref, q, none, o, 0, 0, d_rows, 1, sh.qh, sh.kvh, hd, max_seq, s );
				CUDA_CHECK( cudaStreamSynchronize( s ) );
				std::vector<half> b( static_cast<size_t>( sh.qh ) * hd );
				CUDA_CHECK( cudaMemcpy( b.data(), d_o, b.size() * 2, cudaMemcpyDeviceToHost ) );
				double err = 0, nrm = 0;
				for ( int h = 0; h < sh.qh; ++h ) {
					const int kvh = h / ( sh.qh / sh.kvh );
					for ( int d = 0; d < hd; ++d ) {
						const double x = __half2float( hv[static_cast<size_t>( kvh ) * hd + d] ), y = __half2float( b[static_cast<size_t>( h ) * hd + d] );
						err += ( x - y ) * ( x - y );
						nrm += x * x;
					}
				}
				const double rel = std::sqrt( err / nrm );
				const double max_rel = formats[fi] == pod::KVFormat::FP8 ? 0.08 : ( formats[fi] == pod::KVFormat::K8V4 ? 0.13 : 0.16 );
				all_ok = all_ok && rel <= max_rel;
				std::printf( "  %-9s %-6s V round trip rel_rms=%.4f (limit %.2f) %s\n", sh.name, fname[fi], rel, max_rel, rel <= max_rel ? "ok" : "FAIL" );
			}
			if ( sh.kvh == 1 ) {
				// prefill kernel on the same quantized cache vs the FP16 decode reference
				attend( c, o, true );
				compare( "prefill", fname[fi], min_cos );
			}
			free_cache( c );
		}
		free_cache( ref );
		cudaFree( d_k ); cudaFree( d_v ); cudaFree( d_q ); cudaFree( d_o_ref ); cudaFree( d_o ); cudaFree( d_rows );
	}
	std::printf( "%s KV quantization accuracy\n", all_ok ? "[+] PASS" : "[-] FAIL" );
	return all_ok ? 0 : 1;
}
