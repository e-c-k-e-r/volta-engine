// LinearGguf on the CPU backend over llama.cpp's own CPU kernels (src/kernels/cpu/gguf): each activation row quantized once to the block
// type the weight's dot product pairs with (q8_0, q8_1 or q8_K, ggml-cpu's type traits), in the weight's column order; the q8_K types go through
// ik_llama.cpp's iqk GEMM, the others one dot product per output row and activation row (split across the worker threads).
#include "cpu_kernels.h"
#include "engine/modules/gguf.h"
#include "engine/backend/numeric.h"
#include "../../kernels/cpu/gguf/gguf_cpu.h"
#include <algorithm>
#include <atomic>
#include <functional>
#include <vector>

namespace engine::modules {

namespace {

size_t row_bytes( GgmlType t, int k ) {
	const GgmlBlock b = ggml_block( t );
	return static_cast<size_t>( k / b.elems ) * b.bytes;
}

// The activation column a weight column reads (the tiled V-head order of llama.cpp's Qwen3.5 out_proj, as on CUDA)
int tiled_source( int i, GgufColTiling t, int k ) {
	if ( t.heads == 0 ) return i;
	const int per_v = t.heads * t.dim, v = i / per_v, head = i % per_v / t.dim, d = i % t.dim;
	return ( head * ( k / per_v ) + v ) * t.dim + d;
}

} // namespace

bool gguf_linear_supported( GgmlType type ) { return gguf_cpu::supported( type ); }

void LinearGguf::prepare() {}

void LinearGguf::dequant( const void* w, GgmlType type, int64_t elems, backend::half_t* out, backend::stream_t ) {
	ENGINE_ASSERT( gguf_cpu::supported( type ), "LinearGguf::dequant: no CPU decoder for this block type" );
	const GgmlBlock b = ggml_block( type );
	ENGINE_ASSERT( elems % b.elems == 0, "LinearGguf::dequant: not whole blocks" );
	// in pieces of 64 blocks
	const int64_t per = 64 * static_cast<int64_t>( b.elems ), pieces = ( elems + per - 1 ) / per;
	impl::cpu_parallel_for( pieces, 16, [&]( int64_t p0, int64_t p1 ) {
		std::vector<float> f( static_cast<size_t>( per ) );
		for ( int64_t p = p0; p < p1; ++p ) {
			const int64_t e0 = p * per, n = std::min( per, elems - e0 );
			gguf_cpu::to_float( type, static_cast<const uint8_t*>( w ) + static_cast<size_t>( e0 / b.elems ) * b.bytes, f.data(), n );
			for ( int64_t i = 0; i < n; ++i ) out[e0 + i] = backend::from_float_half( f[static_cast<size_t>( i )] );
		}
	} );
}

void LinearGguf::forward( const pod::TensorView& x, const void* w, GgmlType type, int n, int k, pod::TensorView& out, backend::stream_t, GgufColTiling tiling ) {
	ENGINE_ASSERT( gguf_cpu::supported( type ), "LinearGguf::forward: no CPU dot product for this block type" );
	ENGINE_ASSERT( x.dtype == pod::DType::FP16 && out.dtype == pod::DType::FP16 && x.shape[x.ndim - 1] == k, "LinearGguf: FP16 [m, k] activations and output" );
	int m = 1;
	for ( int d = 0; d < x.ndim - 1; ++d ) m *= x.shape[d];
	const backend::half_t* xs = x.ptr<const backend::half_t>();
	// the rows in the weight's column order, each quantized by q into bytes per row
	auto quantize_rows = [&]( size_t bytes, const std::function<void( const float*, void* )>& q ) {
		std::vector<uint8_t> xq( static_cast<size_t>( m ) * bytes );
		impl::cpu_parallel_for( m, 1, [&]( int64_t r0, int64_t r1 ) {
			std::vector<float> f( static_cast<size_t>( k ) );
			for ( int64_t r = r0; r < r1; ++r ) {
				for ( int i = 0; i < k; ++i ) f[static_cast<size_t>( i )] = backend::to_float( xs[static_cast<size_t>( r ) * k + tiled_source( i, tiling, k )] );
				q( f.data(), xq.data() + static_cast<size_t>( r ) * bytes );
			}
		} );
		return xq;
	};
	const size_t wb = row_bytes( type, k );
	backend::half_t* o = out.ptr<backend::half_t>();
	// a q8_K type: ik_llama.cpp's iqk GEMM (one decoded weight tile for all the rows; from 32 rows it repacks the weight per tile), on the
	// rows quantized to ik's own q8_K. A single row runs through it too: up to 8 rows each row then gets the bits the 1-row decode gives it,
	// so a speculative verify accepts exactly what plain decoding emits
	if ( gguf_cpu::iqk_rows( type ) ) {
		const std::vector<uint8_t> xi = quantize_rows( gguf_cpu::iqk_act_bytes( k ), [&]( const float* f, void* q ) { gguf_cpu::iqk_quantize_act( f, q, k ); } );
		const int nth = impl::cpu_threads();
		std::vector<float> f( static_cast<size_t>( m ) * n );
		std::atomic<bool> ok{ true };
		impl::cpu_parallel_for( nth, 1, [&]( int64_t t0, int64_t t1 ) {
			for ( int64_t t = t0; t < t1 && ok.load( std::memory_order_relaxed ); ++t ) {
				if ( !gguf_cpu::mul_rows( type, w, wb, n, k, xi.data(), m, f.data(), static_cast<int>( t ), nth ) ) ok.store( false );
			}
		} );
		if ( ok.load() ) {
			for ( size_t i = 0; i < f.size(); ++i ) o[i] = backend::from_float_half( f[i] );
			return;
		}
	}
	// otherwise llama.cpp's dot products, on the rows quantized to the activation type the weight's pairs with
	const GgmlType act = gguf_cpu::act_type( type );
	const size_t act_bytes = row_bytes( act, k );
	const std::vector<uint8_t> xq = quantize_rows( act_bytes, [&]( const float* f, void* q ) { gguf_cpu::quantize_act( act, f, q, k ); } );
	impl::cpu_parallel_for( n, 64, [&]( int64_t j0, int64_t j1 ) {
		for ( int64_t j = j0; j < j1; ++j ) {
			const void* wr = static_cast<const uint8_t*>( w ) + static_cast<size_t>( j ) * wb;
			for ( int r = 0; r < m; ++r ) o[static_cast<size_t>( r ) * n + j] = backend::from_float_half( gguf_cpu::dot( type, wr, xq.data() + static_cast<size_t>( r ) * act_bytes, k ) );
		}
	} );
}

} // namespace engine::modules
