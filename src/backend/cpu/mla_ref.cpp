// CPU backend: MLA entry points the GLM model calls. Dense latent attention runs the portable reference kernel (ref_latent.slang, host
// target, bound weakly like the attention one); DSA-selected rows, the latent absorb / expand products, the latent store, the
// kv_b projection and the score diagnostic are plain C++ (FP32 accumulate, FP16 storage). The expanded-cache MLA entry points map
// onto the Attention ones (every head is its own KV head). The old MLADSA pair stays a stub (the model no longer calls it).
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/mla.h"
#include "engine/modules/attention.h"
#include "engine/backend/tiers.h"
#include <cmath>
#include <vector>

struct RefLatentParams {
	half* q_lat; half* cache; half* out_;
	int* slots; int* pos;
	int64_t slot_stride;
	int layer; int heads; int max_seq;
	float scale;
};

extern "C" void ref_latent_decode_cpu( ComputeVaryingInput*, void*, void* ) __attribute__(( weak ));

namespace engine::modules {

namespace {

constexpr int LORA = MLALatent::LORA;
constexpr int HD = MLALatent::HEAD;
constexpr int KPAD = 2080; // DSA::KPAD (the list stride attend() reads)

using h16 = backend::half_t;

}

void MLALatent::absorb_q( const pod::TensorView& q, const pod::TensorView& kv_b, pod::TensorView& q_lat, int heads, int rows, backend::stream_t ) {
	const h16* pq = q.ptr<const h16>();
	const h16* w = kv_b.ptr<const h16>();
	h16* out = q_lat.ptr<h16>();
	std::vector<float> acc( LORA );
	for ( int r = 0; r < rows; ++r ) {
		for ( int h = 0; h < heads; ++h ) {
			const h16* qh = pq + ( static_cast<size_t>( r ) * heads + h ) * HD;
			const h16* wh = w + static_cast<size_t>( h ) * 2 * HD * LORA;
			std::fill( acc.begin(), acc.end(), 0.0f );
			for ( int d = 0; d < HD; ++d ) {
				const float qv = static_cast<float>( qh[d] );
				const h16* wr = wh + static_cast<size_t>( d ) * LORA;
				for ( int c = 0; c < LORA; ++c ) acc[c] += qv * static_cast<float>( wr[c] );
			}
			h16* o = out + ( static_cast<size_t>( r ) * heads + h ) * LORA;
			for ( int c = 0; c < LORA; ++c ) o[c] = static_cast<h16>( acc[c] );
		}
	}
}

void MLALatent::expand_out( const pod::TensorView& o_lat, const pod::TensorView& kv_b, pod::TensorView& out, int heads, int rows, backend::stream_t ) {
	const h16* po = o_lat.ptr<const h16>();
	const h16* w = kv_b.ptr<const h16>();
	h16* dst = out.ptr<h16>();
	for ( int r = 0; r < rows; ++r ) {
		for ( int h = 0; h < heads; ++h ) {
			const h16* ol = po + ( static_cast<size_t>( r ) * heads + h ) * LORA;
			for ( int v = 0; v < HD; ++v ) {
				const h16* wr = w + ( static_cast<size_t>( h ) * 2 * HD + HD + v ) * LORA;
				float acc = 0.0f;
				for ( int c = 0; c < LORA; ++c ) acc += static_cast<float>( wr[c] ) * static_cast<float>( ol[c] );
				dst[( static_cast<size_t>( r ) * heads + h ) * HD + v] = static_cast<h16>( acc );
			}
		}
	}
}

void MLALatent::store( backend::half_t* cache, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, backend::stream_t ) {
	ENGINE_ASSERT( slot.rows != nullptr, "MLALatent::store: device-resolved positions only" );
	const h16* src = c.ptr<const h16>();
	for ( int r = 0; r < rows; ++r ) {
		const int pos = slot.rows->pos[slot.row + r];
		const size_t off = static_cast<size_t>( slot.rows->slot[slot.row + r] ) * slot.stride + ( static_cast<size_t>( layer ) * max_seq + pos ) * LORA;
		for ( int i = 0; i < LORA; ++i ) cache[off + i] = src[static_cast<size_t>( r ) * LORA + i];
	}
}

void MLALatent::attend( const backend::half_t* cache, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads,
	const pod::SlotRef& slot, float*, int, backend::stream_t, const int* sel_idx, const int* sel_n ) {
	ENGINE_ASSERT( slot.rows != nullptr && slot.row == 0, "MLALatent::attend (CPU): device-resolved positions starting at row 0" );
	const float scale = 1.0f / std::sqrt( static_cast<float>( HD ) );
	const h16* q = q_lat.ptr<const h16>();
	h16* out = o_lat.ptr<h16>();
	// Dense rows run the reference kernel one row at a time; a row with a selection list is attended in plain C++
	int dense_run = 0;
	auto flush_dense = [&]( int end ) {
		if ( dense_run == end ) return;
		if ( ref_latent_decode_cpu == nullptr ) {
			ENGINE_UNSUPPORTED( "MLALatent::attend" );
		}
		const int r0 = dense_run;
		RefLatentParams p{ const_cast<half*>( reinterpret_cast<const half*>( q ) ) + static_cast<size_t>( r0 ) * heads * LORA, const_cast<half*>( reinterpret_cast<const half*>( cache ) ),
			reinterpret_cast<half*>( out ) + static_cast<size_t>( r0 ) * heads * LORA, const_cast<int*>( slot.rows->slot ) + r0, const_cast<int*>( slot.rows->pos ) + r0,
			static_cast<int64_t>( slot.stride ), layer, heads, max_seq, scale };
		impl::cpu_dispatch( ref_latent_decode_cpu, &p, heads, end - r0 );
		dense_run = end;
	};
	for ( int r = 0; r < rows; ++r ) {
		const int nsel = ( sel_idx != nullptr && sel_n != nullptr ) ? sel_n[r] : -1;
		if ( nsel < 0 ) continue;
		flush_dense( r );
		dense_run = r + 1; // skip this row in the next dense run
		const h16* kv = cache + static_cast<size_t>( slot.rows->slot[r] ) * slot.stride + static_cast<size_t>( layer ) * max_seq * LORA;
		const int* lst = sel_idx + static_cast<size_t>( r ) * KPAD;
		std::vector<float> s( nsel ), o( LORA );
		for ( int h = 0; h < heads; ++h ) {
			const h16* qh = q + ( static_cast<size_t>( r ) * heads + h ) * LORA;
			float mx = -1e30f;
			for ( int i = 0; i < nsel; ++i ) {
				const h16* c = kv + static_cast<size_t>( lst[i] ) * LORA;
				float dot = 0.0f;
				for ( int d = 0; d < LORA; ++d ) dot += static_cast<float>( qh[d] ) * static_cast<float>( c[d] );
				s[i] = dot * scale;
				mx = std::fmax( mx, s[i] );
			}
			float l = 0.0f;
			std::fill( o.begin(), o.end(), 0.0f );
			for ( int i = 0; i < nsel; ++i ) {
				const float pk = std::exp( s[i] - mx );
				l += pk;
				const h16* c = kv + static_cast<size_t>( lst[i] ) * LORA;
				for ( int d = 0; d < LORA; ++d ) o[d] += pk * static_cast<float>( c[d] );
			}
			const float inv = l > 0.0f ? 1.0f / l : 0.0f;
			h16* oh = out + ( static_cast<size_t>( r ) * heads + h ) * LORA;
			for ( int d = 0; d < LORA; ++d ) oh[d] = static_cast<h16>( o[d] * inv );
		}
	}
	flush_dense( rows );
}

void MLALatent::diag_scores( const backend::half_t* q_lat_row, const backend::half_t* cache_slot_layer, int n, int heads, float* scores, backend::stream_t ) {
	const float scale = 1.0f / std::sqrt( static_cast<float>( HD ) );
	for ( int h = 0; h < heads; ++h ) {
		for ( int t = 0; t < n; ++t ) {
			float acc = 0.0f;
			for ( int d = 0; d < LORA; ++d ) acc += static_cast<float>( q_lat_row[static_cast<size_t>( h ) * LORA + d] ) * static_cast<float>( cache_slot_layer[static_cast<size_t>( t ) * LORA + d] );
			scores[static_cast<size_t>( h ) * n + t] = acc * scale;
		}
	}
}

// Expanded-cache path: kv_norm [rows, 512] x kv_b [heads * 2 * head_dim, 512]^T, rows alternate K block / V block per head
void MLA::project_kv_b( const pod::TensorView& kv_norm, const pod::TensorView& kv_b_proj, pod::TensorView& k_out, pod::TensorView& v_out, int num_heads, int head_dim, backend::stream_t, int rows ) {
	const h16* w = kv_b_proj.ptr<const h16>();
	const h16* x = kv_norm.ptr<const h16>();
	const int out_stride = num_heads * head_dim;
	for ( int m = 0; m < rows; ++m ) {
		for ( int row = 0; row < num_heads * head_dim * 2; ++row ) {
			float sum = 0.0f;
			for ( int i = 0; i < 512; ++i ) sum += static_cast<float>( w[static_cast<size_t>( row ) * 512 + i] ) * static_cast<float>( x[static_cast<size_t>( m ) * 512 + i] );
			const int h = row / ( head_dim * 2 ), rem = row % ( head_dim * 2 );
			if ( rem < head_dim ) k_out.ptr<h16>()[static_cast<size_t>( m ) * out_stride + h * head_dim + rem] = static_cast<h16>( sum );
			else v_out.ptr<h16>()[static_cast<size_t>( m ) * out_stride + h * head_dim + rem - head_dim] = static_cast<h16>( sum );
		}
	}
}

void MLA::store_kv_cache( const pod::KVRef& kv, const pod::TensorView& k, const pod::TensorView& v, int mla_layer_idx, int, int total_elems, int max_seq, backend::stream_t stream,
	const pod::SlotRef& slot, int rows ) {
	if ( !slot.rows ) {
		ENGINE_UNSUPPORTED( "MLA::store_kv_cache (host-resolved position)" );
	}
	ENGINE_ASSERT( slot.row == 0, "MLA::store_kv_cache: row offset unsupported" );
	Attention::store_kv_cache( kv, k, v, slot.stride, mla_layer_idx, slot.rows, rows, total_elems / 256, 256, max_seq, stream );
}

void MLA::forward_decode( const pod::TensorView& q, const pod::KVRef& kv, pod::TensorView& out, int mla_layer_idx, int, int num_heads, int head_dim, int max_seq, backend::stream_t stream,
	const pod::SlotRef& slot, int rows, int key_hint ) {
	if ( !slot.rows ) {
		ENGINE_UNSUPPORTED( "MLA::forward_decode (host-resolved position)" );
	}
	ENGINE_ASSERT( slot.row == 0, "MLA::forward_decode: row offset unsupported" );
	pod::TensorView no_gate{};
	Attention::forward_decode( kv, q, no_gate, out, slot.stride, mla_layer_idx, slot.rows, rows, num_heads, num_heads, head_dim, max_seq, stream );
	( void )key_hint;
}

} // namespace engine::modules
