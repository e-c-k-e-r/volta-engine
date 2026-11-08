// CPU backend: the per-shard top-k and log-sum-exp over the head's logits (sampling, log-probabilities, the speculators' candidates), as
// src/spec/dflash_kernels.cu defines them: the top 16 of each row's first `columns` logits in descending order (equal values: the lower
// id first), ids offset to global token ids; the row max and the sum of exp( x - max ). The CPU backend runs one rank, so no merge.
#include "cpu_kernels.h"
#include "engine/spec/dflash.h"
#include "engine/backend/numeric.h"
#include <cmath>

namespace engine::spec {

void dflash2_topk_shard(
	const backend::half_t* logits, int rows, int row_stride, int columns, int, float* values, int* indices, int index_offset,
	float*, int*, int*, backend::stream_t
) {
	if ( rows <= 0 || columns <= 0 ) return;
	constexpr int K = 16;
	impl::cpu_parallel_for( rows, 1, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t r = r0; r < r1; ++r ) {
			const backend::half_t* x = logits + static_cast<size_t>( r ) * row_stride;
			float v[K];
			int id[K];
			int n = 0;
			for ( int c = 0; c < columns; ++c ) {
				const float f = backend::to_float( x[c] );
				if ( n == K && !( f > v[K - 1] ) ) continue;
				int at = n < K ? n++ : K - 1;
				while ( at > 0 && f > v[at - 1] ) {
					v[at] = v[at - 1];
					id[at] = id[at - 1];
					--at;
				}
				v[at] = f;
				id[at] = c;
			}
			for ( int i = n; i < K; ++i ) {
				v[i] = -INFINITY;
				id[i] = 0;
			}
			for ( int i = 0; i < K; ++i ) {
				values[r * K + i] = v[i];
				indices[r * K + i] = id[i] + index_offset;
			}
		}
	} );
}

void logits_lse_shard( const backend::half_t* logits, int rows, int row_stride, int columns, float* out, backend::stream_t ) {
	if ( rows <= 0 || columns <= 0 ) return;
	impl::cpu_parallel_for( rows, 1, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t r = r0; r < r1; ++r ) {
			const backend::half_t* x = logits + static_cast<size_t>( r ) * row_stride;
			float m = -INFINITY;
			for ( int c = 0; c < columns; ++c ) m = std::fmax( m, backend::to_float( x[c] ) );
			double s = 0.0;
			for ( int c = 0; c < columns; ++c ) s += std::exp( backend::to_float( x[c] ) - m );
			out[2 * r] = m;
			out[2 * r + 1] = static_cast<float>( s );
		}
	} );
}

} // namespace engine::spec
