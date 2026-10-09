// CPU backend: weak stubs for the attention entry points that have no CPU implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/attention.h.
#include "engine/modules/attention.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void Attention::forward_decode( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx, const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "Attention::forward_decode" );
	__builtin_unreachable();
}

__attribute__(( weak )) void Attention::forward_sparse( const pod::KVRef&, const pod::TensorView&, const pod::TensorView&, pod::TensorView&, size_t, int, const pod::BatchRows*, int, int, int, int, int,
	const int*, const int*, int, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "Attention::forward_sparse" );
	__builtin_unreachable();
}

__attribute__(( weak )) void Attention::forward_prefill( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx, const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream, int key_hint, bool per_row ) {
	ENGINE_UNSUPPORTED( "Attention::forward_prefill" );
	__builtin_unreachable();
}

__attribute__(( weak )) void Attention::unpack_q_gate( const pod::TensorView& q_gate, pod::TensorView& q, pod::TensorView& gate, int num_q_heads, int head_dim, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "Attention::unpack_q_gate" );
	__builtin_unreachable();
}

} // namespace engine::modules
