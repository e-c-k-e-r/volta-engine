// CPU backend: the QSA indexer (prep, pool update, scoring) and the sparse-list attention (DSA::select is in dsa_ref.cpp) as plain C++, the same arithmetic as
// src/backend/cuda/modules/{qsa,dsa,attention}.cu. Sparse attention is Attention::forward_sparse: row r attends to the nkeys[r] listed positions, or
// densely to 0..pos when nkeys[r] < 0
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "kv_codec.h"
#include "engine/modules/qsa.h"
#include "engine/modules/dsa.h"
#include "engine/modules/attention.h"
#include "engine/backend/tiers.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace engine::modules {

namespace {

constexpr int D = QSA::DIM;

using impl::h2f;
using impl::f2h;
inline float round_half( float v ) { return h2f( f2h( v ) ); }

// Rotate-half rope on the first rot_dim dims of a 128-wide vector, in place
void rope_vec( float* x, int pos, int rot_dim, float theta ) {
	const int half_rot = rot_dim / 2;
	float out[D];
	for ( int d = 0; d < D; ++d ) {
		if ( d >= rot_dim ) { out[d] = x[d]; continue; }
		const int i = d % half_rot;
		const float ang = static_cast<float>( pos ) * std::pow( theta, -( 2.0f * i / static_cast<float>( rot_dim ) ) );
		const float s = std::sin( ang ), c = std::cos( ang );
		out[d] = d < half_rot ? x[d] * c - x[d + half_rot] * s : x[d] * c + x[d - half_rot] * s;
	}
	for ( int d = 0; d < D; ++d ) x[d] = out[d];
}

} // namespace

void QSA::prep( const backend::half_t* qk, const backend::half_t* q_norm_w, backend::half_t* q_out, backend::half_t* ring, int layer, int rows,
	const pod::BatchRows* d_rows, size_t ring_slot_stride, int rot_dim, float theta, float eps, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		const int pos = d_rows->pos[r];
		const backend::half_t* src = qk + static_cast<size_t>( r ) * QSA::PROJ;
		backend::half_t* dst = ring + static_cast<size_t>( d_rows->slot[r] ) * ring_slot_stride + ( static_cast<size_t>( layer ) * QSA::RING + ( pos % QSA::RING ) ) * D;
		for ( int d = 0; d < D; ++d ) dst[d] = src[QSA::HEADS * D + d];
		for ( int h = 0; h < QSA::HEADS; ++h ) {
			float x[D];
			float ss = 0.0f;
			for ( int d = 0; d < D; ++d ) { x[d] = h2f( src[h * D + d] ); ss += x[d] * x[d]; }
			const float inv = 1.0f / std::sqrt( ss / D + eps );
			// rounded to the key dtype before the rope
			for ( int d = 0; d < D; ++d ) x[d] = round_half( x[d] * inv * ( 1.0f + h2f( q_norm_w[d] ) ) );
			rope_vec( x, pos, rot_dim, theta );
			for ( int d = 0; d < D; ++d ) q_out[( static_cast<size_t>( r ) * QSA::HEADS + h ) * D + d] = f2h( x[d] );
		}
	}
}

void QSA::pool_update( const backend::half_t* ring, backend::half_t* pool_plane, const backend::half_t* k_norm_w, int layer, int rows,
	const pod::BatchRows* d_rows, size_t ring_slot_stride, size_t pool_slot_stride, int max_pools, int rot_dim, float theta, float eps, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		const int pos = d_rows->pos[r];
		if ( pos % QSA::POOL != QSA::POOL - 1 ) continue;
		const int slot = d_rows->slot[r], pool = pos / QSA::POOL;
		const backend::half_t* rk = ring + static_cast<size_t>( slot ) * ring_slot_stride + static_cast<size_t>( layer ) * QSA::RING * D;
		float x[D];
		float ss = 0.0f;
		for ( int d = 0; d < D; ++d ) {
			float acc = 0.0f;
			for ( int j = 0; j < QSA::POOL; ++j ) acc += h2f( rk[static_cast<size_t>( ( pool * QSA::POOL + j ) % QSA::RING ) * D + d] );
			// mean in fp32, rounded to the key dtype before the norm
			x[d] = round_half( acc / QSA::POOL );
			ss += x[d] * x[d];
		}
		const float inv = 1.0f / std::sqrt( ss / D + eps );
		for ( int d = 0; d < D; ++d ) x[d] = round_half( x[d] * inv * ( 1.0f + h2f( k_norm_w[d] ) ) );
		rope_vec( x, pool * QSA::POOL, rot_dim, theta );
		backend::half_t* out = pool_plane + static_cast<size_t>( slot ) * pool_slot_stride + ( static_cast<size_t>( layer ) * max_pools + pool ) * D;
		for ( int d = 0; d < D; ++d ) out[d] = f2h( x[d] );
	}
}

void QSA::score( const backend::half_t* q, const backend::half_t* pool_plane, backend::half_t* scores, int score_stride, int layer, int max_pools, int rows,
	const pod::BatchRows* d_rows, size_t pool_slot_stride, backend::stream_t ) {
	const float scale = 1.0f / std::sqrt( static_cast<float>( D ) );
	for ( int r = 0; r < rows; ++r ) {
		const int bound = ( d_rows->pos[r] + 1 ) / QSA::POOL;
		if ( bound <= DSA::SEL_POOLS ) continue;
		const backend::half_t* qr = q + static_cast<size_t>( r ) * QSA::HEADS * D;
		const backend::half_t* plane = pool_plane + static_cast<size_t>( d_rows->slot[r] ) * pool_slot_stride + static_cast<size_t>( layer ) * max_pools * D;
		impl::cpu_parallel_for( bound, 256, [&]( int64_t p0, int64_t p1 ) {
			for ( int64_t p = p0; p < p1; ++p ) {
				const backend::half_t* key = plane + static_cast<size_t>( p ) * D;
				float total = 0.0f;
				for ( int h = 0; h < QSA::HEADS; ++h ) {
					float s = 0.0f;
					for ( int d = 0; d < D; ++d ) s += h2f( qr[h * D + d] ) * h2f( key[d] );
					total += std::fmax( s, 0.0f );
				}
				scores[static_cast<size_t>( r ) * score_stride + p] = f2h( total * scale );
			}
		} );
	}
}

void Attention::forward_sparse( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, const int* idx, const int* nkeys, int idx_stride, backend::stream_t ) {
	ENGINE_ASSERT( kv.vq != pod::KVQ::Q4, "Attention::forward_sparse: Q4 values are not supported" );
	const float scale = 1.0f / std::sqrt( static_cast<float>( head_dim ) );
	const int heads_per_kv = std::max( num_q_heads / num_kv_heads, 1 );
	const backend::half_t* qp = q.ptr<const backend::half_t>();
	const backend::half_t* gp = gate.data != nullptr ? gate.ptr<const backend::half_t>() : nullptr;
	backend::half_t* op = out.ptr<backend::half_t>();
	impl::cpu_parallel_for( static_cast<int64_t>( count ) * num_q_heads, 2, [&]( int64_t i0, int64_t i1 ) {
		std::vector<float> sc, acc( head_dim );
		for ( int64_t i = i0; i < i1; ++i ) {
			const int r = static_cast<int>( i / num_q_heads ), head = static_cast<int>( i % num_q_heads );
			const int kvh = head / heads_per_kv;
			const int nk = nkeys[r];
			const int total = nk >= 0 ? nk : std::min( d_rows->pos[r] + 1, max_seq );
			const size_t slot_off = static_cast<size_t>( d_rows->slot[r] ) * slot_stride;
			const backend::half_t* qh = qp + ( static_cast<size_t>( r ) * num_q_heads + head ) * head_dim;
			sc.assign( total, 0.0f );
			float mx = -1e30f;
			for ( int j = 0; j < total; ++j ) {
				const int t = nk >= 0 ? idx[static_cast<size_t>( r ) * idx_stride + j] : j;
				const size_t kr = slot_off + ( static_cast<size_t>( layer_idx ) * max_seq + t ) * ( num_kv_heads * head_dim ) + kvh * head_dim;
				float dot = 0.0f;
				for ( int d = 0; d < head_dim; ++d ) dot += h2f( qh[d] ) * kv_load( kv.k, kv.ks, kv.kq, kr + d );
				sc[j] = dot * scale;
				mx = std::fmax( mx, sc[j] );
			}
			std::fill( acc.begin(), acc.end(), 0.0f );
			float l = 0.0f;
			for ( int j = 0; j < total; ++j ) {
				const int t = nk >= 0 ? idx[static_cast<size_t>( r ) * idx_stride + j] : j;
				const size_t vr = slot_off + ( static_cast<size_t>( layer_idx ) * max_seq + t ) * ( num_kv_heads * head_dim ) + kvh * head_dim;
				const float e = std::exp( sc[j] - mx );
				l += e;
				for ( int d = 0; d < head_dim; ++d ) acc[d] += e * kv_load( kv.v, kv.vs, kv.vq, vr + d );
			}
			const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
			for ( int d = 0; d < head_dim; ++d ) {
				const size_t o = ( static_cast<size_t>( r ) * num_q_heads + head ) * head_dim + d;
				const float sig = gp != nullptr ? 1.0f / ( 1.0f + std::exp( -h2f( gp[o] ) ) ) : 1.0f;
				op[o] = f2h( acc[d] * inv_l * sig );
			}
		}
	} );
}

} // namespace engine::modules
