// Vulkan backend: FP16 KV store through the SPIR-V kernel. Decode / prefill attention stay weak stubs (stubs_attention.cpp); the portable reference kernel ref_attn_decode is exercised by test_vulkan_ref
#include "mod_common.h"
#include "engine/modules/attention.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

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
	backend::stream_t stream
) {
	if ( kv.kq != pod::KVQ::F16 || kv.vq != pod::KVQ::F16 ) {
		ENGINE_UNSUPPORTED( "Attention::store_kv_cache (quantized KV cache)" );
	}
	const int total_elems = num_kv_heads * head_dim;
	const int groups = ( count * total_elems + 127 ) / 128;
	vki::KvStoreParams pk{ vki::dp( kv.k ), vki::dp( k.data ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
		static_cast<int64_t>( slot_stride ), layer_idx, count, total_elems, max_seq };
	vki::launch( "kv_store_f16", pk, stream, groups );
	vki::KvStoreParams pv = pk;
	pv.data = vki::dp( kv.v );
	pv.src = vki::dp( v.data );
	vki::launch( "kv_store_f16", pv, stream, groups );
}

} // namespace engine::modules
