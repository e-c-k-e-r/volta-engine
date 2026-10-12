// Portable reference attention over the paged KV cache (FP16, or any quantized format of kv_codec.slang through ks / vs / kq / vq) (decode semantics of attn_chunked_paged_kernel; slow but simple).
// Row m of the call reads slot slots[m] at position pos[m] and attends causally to positions 0..pos[m], so the same kernel is
// the decode form and the causal prefill form (a chunk is a run of rows whose pos values ascend within one slot).
// Cache layout [slot][layer][max_seq][num_kv_heads * head_dim]; q, gate and out are [rows][num_q_heads][head_dim].
// One thread group per (head, row): x = head, y = row. head_dim <= 256. FP32 softmax with the running-max subtraction.
// The GPU entry streams keys in tiles of 256 (scores in shared memory, tree reductions); the host target has no group
// barriers, so it gets a serial two-pass entry under ENGINE_TARGET_CPU.
// kq / vq are the KVQ codes (0 = FP16, where k / v are read as halves and ks / vs are unused). Q4 values are stored Hadamard-rotated per
// 64 elements, so the attended output is rotated back per 64 (head_dim must be a multiple of 64 then).
#include "kv_codec.slang"

struct RefAttnParams {
	half* q;
	half* k;          // cache base (slot 0)
	half* v;
	half* gate;       // [rows][num_q_heads][head_dim] pre-sigmoid, may be null
	half* out_;
	int* slots;       // BatchRows::slot, indexed by row
	int* pos;         // BatchRows::pos, indexed by row
	int64_t slot_stride;
	int layer_idx;
	int num_q_heads;
	int num_kv_heads;
	int head_dim;
	int max_seq;
	int has_gate;
	float scale;
	uint8_t* ks;      // K / V scales for the quantized formats
	uint8_t* vs;
	int kq;
	int vq;
};

float ref_kv_k(RefAttnParams p, int64_t i) {
	return p.kq == 0 ? float(p.k[i]) : kv_load((uint8_t*)p.k, p.ks, p.kq, i);
}

float ref_kv_v(RefAttnParams p, int64_t i) {
	return p.vq == 0 ? float(p.v[i]) : kv_load((uint8_t*)p.v, p.vs, p.vq, i);
}

float ref_attn_gate(RefAttnParams p, int64_t idx) {
	if (p.has_gate == 0) return 1.0f;
	return 1.0f / (1.0f + exp(-float(p.gate[idx])));
}

#if !ENGINE_TARGET_CPU
groupshared float s_q[256];
groupshared float s_s[256];   // tile probabilities
groupshared float s_red[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void ref_attn_decode(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RefAttnParams p) {
	let t = int(tid.x);
	let head = int(gid.x);
	let m = int(gid.y);
	let per_kv = max(p.num_q_heads / p.num_kv_heads, 1);
	let kv_head = head / per_kv;
	let total = min(p.pos[m] + 1, p.max_seq);
	let row_elems = p.num_kv_heads * p.head_dim;
	let base = int64_t(p.slots[m]) * p.slot_stride + int64_t(p.layer_idx) * p.max_seq * row_elems + kv_head * p.head_dim;
	let qo = (int64_t(m) * p.num_q_heads + head) * p.head_dim;
	let active = t < p.head_dim;
	s_q[t] = active ? float(p.q[qo + t]) : 0.0f;
	GroupMemoryBarrierWithGroupSync();
	float m_run = -1e30f;
	float l_run = 0.0f;
	float o = 0.0f;
	for (int t0 = 0; t0 < total; t0 += 256) {
		let nk = min(256, total - t0);
		// thread t scores key t0 + t
		float score = -1e30f;
		if (t < nk) {
			let kp = base + int64_t(t0 + t) * row_elems;
			float dot = 0.0f;
			for (int d = 0; d < p.head_dim; ++d) dot = fma(s_q[d], ref_kv_k(p, kp + d), dot);
			score = dot * p.scale;
		}
		s_red[t] = score;
		GroupMemoryBarrierWithGroupSync();
		for (int s = 128; s > 0; s >>= 1) {
			if (t < s) s_red[t] = max(s_red[t], s_red[t + s]);
			GroupMemoryBarrierWithGroupSync();
		}
		let m_new = max(m_run, s_red[0]);
		GroupMemoryBarrierWithGroupSync();
		let pk = t < nk ? exp(score - m_new) : 0.0f;
		s_s[t] = pk;
		s_red[t] = pk;
		GroupMemoryBarrierWithGroupSync();
		for (int s = 128; s > 0; s >>= 1) {
			if (t < s) s_red[t] += s_red[t + s];
			GroupMemoryBarrierWithGroupSync();
		}
		let alpha = exp(m_run - m_new);
		l_run = l_run * alpha + s_red[0];
		m_run = m_new;
		o *= alpha;
		if (active) {
			for (int k = 0; k < nk; ++k) o = fma(s_s[k], ref_kv_v(p, base + int64_t(t0 + k) * row_elems + t), o);
		}
		GroupMemoryBarrierWithGroupSync();   // s_s / s_red are rewritten by the next tile
	}
	float val = l_run > 0.0f ? o / l_run : 0.0f;
	if (p.vq == 3) {
		// undo the per-64 Walsh-Hadamard rotation of the stored values: out[d] = 0.125 * sum_j (-1)^popcount(d & j) * o[j]
		s_red[t] = val;
		GroupMemoryBarrierWithGroupSync();
		let chunk = t & ~63;
		let dl = uint(t & 63);
		float acc = 0.0f;
		for (int j = 0; j < 64; ++j) acc += (countbits(dl & uint(j)) & 1u) != 0u ? -s_red[chunk + j] : s_red[chunk + j];
		val = acc * 0.125f;
	}
	if (active) p.out_[qo + t] = half(val * ref_attn_gate(p, qo + t));
}
#else
// Host target: one invocation per (head, row); pass 1 finds the max, pass 2 accumulates
[shader("compute")]
[numthreads(1, 1, 1)]
void ref_attn_decode_cpu(uint3 gid : SV_GroupID, uniform RefAttnParams p) {
	let head = int(gid.x);
	let m = int(gid.y);
	let per_kv = max(p.num_q_heads / p.num_kv_heads, 1);
	let kv_head = head / per_kv;
	let total = min(p.pos[m] + 1, p.max_seq);
	let row_elems = p.num_kv_heads * p.head_dim;
	let base = int64_t(p.slots[m]) * p.slot_stride + int64_t(p.layer_idx) * p.max_seq * row_elems + kv_head * p.head_dim;
	let qo = (int64_t(m) * p.num_q_heads + head) * p.head_dim;
	float qf[256];
	float o[256];
	for (int d = 0; d < p.head_dim; ++d) {
		qf[d] = float(p.q[qo + d]);
		o[d] = 0.0f;
	}
	float mx = -1e30f;
	for (int t = 0; t < total; ++t) {
		let kp = base + int64_t(t) * row_elems;
		float dot = 0.0f;
		for (int d = 0; d < p.head_dim; ++d) dot = fma(qf[d], ref_kv_k(p, kp + d), dot);
		mx = max(mx, dot * p.scale);
	}
	float l = 0.0f;
	for (int t = 0; t < total; ++t) {
		let kp = base + int64_t(t) * row_elems;
		float dot = 0.0f;
		for (int d = 0; d < p.head_dim; ++d) dot = fma(qf[d], ref_kv_k(p, kp + d), dot);
		let pk = exp(dot * p.scale - mx);
		l += pk;
		for (int d = 0; d < p.head_dim; ++d) o[d] = fma(pk, ref_kv_v(p, kp + d), o[d]);
	}
	for (int d = 0; d < p.head_dim; ++d) o[d] = l > 0.0f ? o[d] / l : 0.0f;
	if (p.vq == 3) {
		for (int c = 0; c < p.head_dim; c += 64) {
			float w[64];
			for (int j = 0; j < 64; ++j) w[j] = o[c + j];
			kv_fwht64(w);
			for (int j = 0; j < 64; ++j) o[c + j] = w[j];
		}
	}
	for (int d = 0; d < p.head_dim; ++d) p.out_[qo + d] = half(o[d] * ref_attn_gate(p, qo + d));
}
#endif

// Q/gate split of the fused [rows][num_q_heads][2 * head_dim] projection: q and gate are [rows][num_q_heads][head_dim].
// One invocation per output element (GPU: 128 per group; host: one group per element, entry carries the _cpu suffix).
struct RefUnpackParams {
	half* qg;
	half* q;
	half* gate;
	int total;        // rows * num_q_heads * head_dim
	int head_dim;
};

void ref_unpack_one(RefUnpackParams p, int64_t i) {
	if (i >= p.total) return;
	let hd = int64_t(p.head_dim);
	let src = (i / hd) * 2 * hd + (i % hd);
	p.q[i] = p.qg[src];
	p.gate[i] = p.qg[src + hd];
}

#if !ENGINE_TARGET_CPU
[shader("compute")]
[numthreads(128, 1, 1)]
void ref_unpack_q_gate(uint3 tid : SV_DispatchThreadID, uniform RefUnpackParams p) {
	ref_unpack_one(p, int64_t(tid.x));
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void ref_unpack_q_gate_cpu(uint3 gid : SV_GroupID, uniform RefUnpackParams p) {
	ref_unpack_one(p, int64_t(gid.x));
}
#endif
