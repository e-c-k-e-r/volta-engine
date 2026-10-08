// Portable reference latent attention (GLM MLA, MLALatent::attend dense semantics). Every head scores the same single latent
// cache c (K and V are both c): score(h, t) = q_lat[h] . c_t * scale, o_lat[h] = sum_t softmax_t(score) c_t.
// Cache layout [slot][layer][max_seq][512]; q_lat and out are [rows][heads][512]. One thread group per (head, row):
// x = head, y = row. Row m attends to positions 0..pos[m] of slot slots[m]. FP32 softmax with the running-max subtraction.

struct RefLatentParams {
	half* q_lat;
	half* cache;
	half* out_;
	int* slots;
	int* pos;
	int64_t slot_stride;
	int layer;
	int heads;
	int max_seq;
	float scale;
};

#define REF_LORA 512

#if !ENGINE_TARGET_CPU
groupshared float s_q[REF_LORA];
groupshared float s_s[256];
groupshared float s_red[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void ref_latent_decode(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RefLatentParams p) {
	let t = int(tid.x);
	let head = int(gid.x);
	let m = int(gid.y);
	let total = min(p.pos[m] + 1, p.max_seq);
	let base = int64_t(p.slots[m]) * p.slot_stride + int64_t(p.layer) * p.max_seq * REF_LORA;
	let qo = (int64_t(m) * p.heads + head) * REF_LORA;
	s_q[t] = float(p.q_lat[qo + t]);
	s_q[t + 256] = float(p.q_lat[qo + t + 256]);
	GroupMemoryBarrierWithGroupSync();
	float m_run = -1e30f;
	float l_run = 0.0f;
	float o0 = 0.0f;
	float o1 = 0.0f;
	for (int t0 = 0; t0 < total; t0 += 256) {
		let nk = min(256, total - t0);
		float score = -1e30f;
		if (t < nk) {
			let kp = base + int64_t(t0 + t) * REF_LORA;
			float dot = 0.0f;
			for (int d = 0; d < REF_LORA; ++d) dot = fma(s_q[d], float(p.cache[kp + d]), dot);
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
		o0 *= alpha;
		o1 *= alpha;
		for (int k = 0; k < nk; ++k) {
			let kp = base + int64_t(t0 + k) * REF_LORA;
			o0 = fma(s_s[k], float(p.cache[kp + t]), o0);
			o1 = fma(s_s[k], float(p.cache[kp + t + 256]), o1);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	let inv = l_run > 0.0f ? 1.0f / l_run : 0.0f;
	p.out_[qo + t] = half(o0 * inv);
	p.out_[qo + t + 256] = half(o1 * inv);
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void ref_latent_decode_cpu(uint3 gid : SV_GroupID, uniform RefLatentParams p) {
	let head = int(gid.x);
	let m = int(gid.y);
	let total = min(p.pos[m] + 1, p.max_seq);
	let base = int64_t(p.slots[m]) * p.slot_stride + int64_t(p.layer) * p.max_seq * REF_LORA;
	let qo = (int64_t(m) * p.heads + head) * REF_LORA;
	float qf[512];
	float o[512];
	for (int d = 0; d < REF_LORA; ++d) {
		qf[d] = float(p.q_lat[qo + d]);
		o[d] = 0.0f;
	}
	float mx = -1e30f;
	for (int t = 0; t < total; ++t) {
		let kp = base + int64_t(t) * REF_LORA;
		float dot = 0.0f;
		for (int d = 0; d < REF_LORA; ++d) dot = fma(qf[d], float(p.cache[kp + d]), dot);
		mx = max(mx, dot * p.scale);
	}
	float l = 0.0f;
	for (int t = 0; t < total; ++t) {
		let kp = base + int64_t(t) * REF_LORA;
		float dot = 0.0f;
		for (int d = 0; d < REF_LORA; ++d) dot = fma(qf[d], float(p.cache[kp + d]), dot);
		let pk = exp(dot * p.scale - mx);
		l += pk;
		for (int d = 0; d < REF_LORA; ++d) o[d] = fma(pk, float(p.cache[kp + d]), o[d]);
	}
	for (int d = 0; d < REF_LORA; ++d) p.out_[qo + d] = half(l > 0.0f ? o[d] / l : 0.0f);
}
#endif
