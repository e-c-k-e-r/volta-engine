// CPU backend: RoPE through the Slang host kernel
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/rope.h"

namespace engine::modules {

void RoPE::forward(
	pod::TensorView& q,
	pod::TensorView& k,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int rot_dim,
	const pod::BatchRows* d_rows,
	int count,
	float freq_base,
	backend::stream_t,
	bool interleaved
) {
	// d_rows is host-addressable on this backend; the kernel reads pos[row]
	RopeParams p{ q.ptr<half>(), k.ptr<half>(), const_cast<int*>( d_rows->pos ), num_q_heads, num_kv_heads, head_dim, rot_dim, interleaved ? 1 : 0, freq_base };
	impl::cpu_dispatch( ::rope, &p, count );
}

} // namespace engine::modules
