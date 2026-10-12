// CPU backend: Attention entry points with a Slang kernel (FP16 KV store). Decode/prefill attention stay CUDA-only.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/attention.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

void Attention::store_kv_cache(
	const pod::KVRef& kv,
	const pod::TensorView& k,
	const pod::TensorView& v,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* d_rows,
	int count,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	backend::stream_t
) {
	const int total_elems = num_kv_heads * head_dim;
	if ( kv.kq != pod::KVQ::F16 || kv.vq != pod::KVQ::F16 ) {
		// Quantized formats: one invocation per quantization unit (kv_codec.slang), K and V each in their own format
		auto store = [&]( void* data, void* scales, const pod::TensorView& src, pod::KVQ q ) {
			const int unit = ( q == pod::KVQ::I8 || q == pod::KVQ::Q4 ) ? 64 : ( q == pod::KVQ::FP4 ? 16 : 1 );
			ENGINE_ASSERT( total_elems % unit == 0, "KV store: row width must be a multiple of the quantization group" );
			KvStoreQParams p{ static_cast<uint8_t*>( data ), static_cast<uint8_t*>( scales ), const_cast<half*>( src.ptr<const half>() ), const_cast<int*>( d_rows->slot ),
				const_cast<int*>( d_rows->pos ), static_cast<int64_t>( slot_stride ), layer_idx, count, total_elems, max_seq, static_cast<int>( q ) };
			impl::cpu_dispatch( ::kv_store_q, &p, ( count * ( total_elems / unit ) + 127 ) / 128 );
		};
		store( kv.k, kv.ks, k, kv.kq );
		store( kv.v, kv.vs, v, kv.vq );
		return;
	}
	const int groups = ( count * total_elems + 127 ) / 128;
	KvStoreParams pk{ static_cast<half*>( kv.k ), const_cast<half*>( k.ptr<const half>() ), const_cast<int*>( d_rows->slot ), const_cast<int*>( d_rows->pos ),
		static_cast<int64_t>( slot_stride ), layer_idx, count, total_elems, max_seq };
	impl::cpu_dispatch( ::kv_store_f16, &pk, groups );
	KvStoreParams pv = pk;
	pv.data = static_cast<half*>( kv.v );
	pv.src = const_cast<half*>( v.ptr<const half>() );
	impl::cpu_dispatch( ::kv_store_f16, &pv, groups );
}

} // namespace engine::modules
