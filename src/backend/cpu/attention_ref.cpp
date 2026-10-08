// CPU backend: Attention decode / prefill over the FP16 KV cache through the portable reference kernel (ref_attention.slang, host
// target) and the Q/gate unpack as plain C++. Row m attends causally to positions 0..pos[m] of slot slot[m], so the prefill form
// is the same call over a run of ascending positions. Quantized KV formats stay unsupported.
// The reference kernel lives in the ref_attn_cpu library (cmake/reference_kernels.cmake); it is bound weakly so binaries that do not
// link that library still link (and abort with ENGINE_UNSUPPORTED if they reach it).
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/attention.h"
#include "engine/backend/tiers.h"
#include <cmath>

struct RefAttnParams {
	half* q; half* k; half* v; half* gate; half* out_;
	int* slots; int* pos;
	int64_t slot_stride;
	int layer_idx; int num_q_heads; int num_kv_heads; int head_dim; int max_seq; int has_gate;
	float scale;
};

extern "C" void ref_attn_decode_cpu( ComputeVaryingInput*, void*, void* ) __attribute__(( weak ));

namespace engine::modules {

namespace {

void ref_attention( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride,
	int layer_idx, const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq ) {
	if ( kv.kq != pod::KVQ::F16 || kv.vq != pod::KVQ::F16 ) {
		ENGINE_UNSUPPORTED( "Attention decode/prefill (quantized KV cache)" );
	}
	if ( ref_attn_decode_cpu == nullptr ) {
		ENGINE_UNSUPPORTED( "Attention decode/prefill (ref_attn_cpu library not linked)" );
	}
	ENGINE_ASSERT( head_dim <= 256 && num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, "Attention (CPU): head_dim <= 256 and GQA only" );
	RefAttnParams p{ const_cast<half*>( q.ptr<const half>() ), static_cast<half*>( kv.k ), static_cast<half*>( kv.v ), gate.data != nullptr ? const_cast<half*>( gate.ptr<const half>() ) : nullptr, out.ptr<half>(),
		const_cast<int*>( d_rows->slot ), const_cast<int*>( d_rows->pos ), static_cast<int64_t>( slot_stride ), layer_idx, num_q_heads, num_kv_heads,
		head_dim, max_seq, gate.data != nullptr ? 1 : 0, 1.0f / std::sqrt( static_cast<float>( head_dim ) ) };
	impl::cpu_dispatch( ref_attn_decode_cpu, &p, num_q_heads, count );
}

}

void Attention::forward_decode( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t ) {
	ref_attention( kv, q, gate, out, slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, head_dim, max_seq );
}

void Attention::forward_prefill( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t, int, bool ) {
	ref_attention( kv, q, gate, out, slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, head_dim, max_seq );
}

void Attention::unpack_q_gate( const pod::TensorView& q_gate, pod::TensorView& q, pod::TensorView& gate, int num_q_heads, int head_dim, backend::stream_t ) {
	const int rows = ( q_gate.ndim >= 2 && q_gate.shape[0] > 0 ) ? q_gate.shape[0] : 1;
	const uint16_t* src = q_gate.ptr<const uint16_t>();
	uint16_t* dq = q.ptr<uint16_t>();
	uint16_t* dg = gate.ptr<uint16_t>();
	for ( size_t i = 0; i < static_cast<size_t>( rows ) * num_q_heads; ++i ) {
		for ( int d = 0; d < head_dim; ++d ) {
			dq[i * head_dim + d] = src[i * 2 * head_dim + d];
			dg[i * head_dim + d] = src[i * 2 * head_dim + head_dim + d];
		}
	}
}

} // namespace engine::modules
