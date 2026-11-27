// The step between the q / k / v projections and the attention in one kernel (Attention::norm_rope_store on Vulkan): a group per (row,
// head) over q's heads, then k's and v's. The head's RMS norm (q with q_norm, k with k_norm, v without a weight), the rotate-half rotary
// embedding of q and k (the first rot_pairs pairs when > 0), and k / v into the FP16 cache. The arithmetic and roundings are those of
// rms_norm_heads, rope and kv_store_f16 in turn (the normed head is rounded to half before it rotates), so the result is theirs bit for bit.

struct NormRopeStoreParams {
	half* q;
	half* k;
	half* v;
	half* q_norm;     // may be null (no weight)
	half* k_norm;
	half* k_cache;
	half* v_cache;
	int* slots;       // [rows] (BatchRows::slot)
	int* pos;         // [rows] (BatchRows::pos)
	int64_t slot_stride;
	int num_q_heads;
	int num_kv_heads; // 0: q alone
	int head_dim;     // up to 512
	int rot_dim;
	int rot_pairs;    // > 0: only the first rot_pairs pairs rotate (frequencies still from rot_dim)
	int layer_idx;
	int max_seq;
	int ring;         // > 0: position p goes to row p % ring
	float eps;
	float freq_base;
	float bias;       // added to the norm weights (rms_norm_heads' constant bias)
};

groupshared float s_red[256];
groupshared half s_h[512];

[shader("compute")]
[numthreads(256, 1, 1)]
void norm_rope_store(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform NormRopeStoreParams p) {
	let t = int(tid.x);
	let row = int(gid.x);
	let h = int(gid.y);
	let kind = h < p.num_q_heads ? 0 : (h < p.num_q_heads + p.num_kv_heads ? 1 : 2);   // q, k, v
	let hh = kind == 0 ? h : (kind == 1 ? h - p.num_q_heads : h - p.num_q_heads - p.num_kv_heads);
	let heads = kind == 0 ? p.num_q_heads : p.num_kv_heads;
	half* base = kind == 0 ? p.q : (kind == 1 ? p.k : p.v);
	half* ptr = base + (int64_t(row) * heads + hh) * p.head_dim;
	half* wptr = kind == 0 ? p.q_norm : p.k_norm;
	let has_w = kind != 2 && wptr != nullptr;
	// rms_norm_heads: thread t owns elements t and t + 256
	let active = t < p.head_dim;
	let active1 = t + 256 < p.head_dim;
	let val = active ? float(ptr[t]) : 0.0f;
	let val1 = active1 ? float(ptr[t + 256]) : 0.0f;
	s_red[t] = val * val + val1 * val1;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let inv = rsqrt(s_red[0] / float(p.head_dim) + p.eps);
	var n0 = half(0.0f);
	var n1 = half(0.0f);
	if (active) {
		let w = has_w ? float(wptr[t]) + p.bias : 1.0f;
		n0 = half(val * w * inv);
	}
	if (active1) {
		let w = has_w ? float(wptr[t + 256]) + p.bias : 1.0f;
		n1 = half(val1 * w * inv);
	}
	if (kind != 2) {
		// rope: pair i rotates elements i and i + rot_dim / 2 of the rounded head
		if (active) s_h[t] = n0;
		if (active1) s_h[t + 256] = n1;
		GroupMemoryBarrierWithGroupSync();
		let half_rot = p.rot_dim / 2;
		if (t < (p.rot_pairs > 0 ? p.rot_pairs : half_rot)) {
			let theta = float(p.pos[row]) * pow(p.freq_base, -(2.0f * float(t) / float(p.rot_dim)));
			let c = cos(theta);
			let s = sin(theta);
			let a = float(s_h[t]);
			let b = float(s_h[t + half_rot]);
			s_h[t] = half(a * c - b * s);
			s_h[t + half_rot] = half(a * s + b * c);
		}
		GroupMemoryBarrierWithGroupSync();
		if (active) n0 = s_h[t];
		if (active1) n1 = s_h[t + 256];
	}
	if (active) ptr[t] = n0;
	if (active1) ptr[t + 256] = n1;
	if (kind == 0) return;
	// kv_store_f16: the row's position in its slot's layer (a ring row for a sliding-window cache)
	half* cache = kind == 1 ? p.k_cache : p.v_cache;
	let ps = p.pos[row];
	let dst = int64_t(p.slots[row]) * p.slot_stride + (int64_t(p.layer_idx) * p.max_seq + (p.ring > 0 ? ps % p.ring : ps)) * (p.num_kv_heads * p.head_dim) +
		hh * p.head_dim;
	if (active) cache[dst + t] = n0;
	if (active1) cache[dst + t + 256] = n1;
}
