// Selection kernels: batched argmax over FP16 logits and the MoE router's top-k. Ties go to the lowest index. One group per row;
// the GPU entry points reduce in shared memory, the host target (ENGINE_TARGET_CPU) has serial entry points.

struct ArgmaxParams {
	half* logits;
	int* out_idx;
	float* out_val;      // may be null
	int numel;
	int batch_size;
	int index_offset;
	int row_stride;
	int has_val;
};

#if !ENGINE_TARGET_CPU
groupshared float s_val[256];
groupshared int s_idx[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void batched_argmax(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform ArgmaxParams p) {
	let t = int(tid.x);
	let row = int(gid.x);
	half* rl = p.logits + int64_t(row) * p.row_stride;
	float best = -1e30f;
	int best_i = -1;
	for (int i = t; i < p.numel; i += 256) {
		let v = float(rl[i]);
		if (v > best) { best = v; best_i = i; }
	}
	s_val[t] = best;
	s_idx[t] = best_i;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) {
			let ov = s_val[t + s];
			let oi = s_idx[t + s];
			// strictly greater, or equal value with the lower index (an empty slot has index -1 and loses)
			if (ov > s_val[t] || (ov == s_val[t] && oi >= 0 && (s_idx[t] < 0 || oi < s_idx[t]))) {
				s_val[t] = ov;
				s_idx[t] = oi;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (t == 0) {
		p.out_idx[row] = s_idx[0] < 0 ? -1 : s_idx[0] + p.index_offset;
		if (p.has_val != 0) p.out_val[row] = s_val[0];
	}
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void batched_argmax(uint3 gid : SV_GroupID, uniform ArgmaxParams p) {
	let row = int(gid.x);
	half* rl = p.logits + int64_t(row) * p.row_stride;
	float best = -1e30f;
	int best_i = -1;
	for (int i = 0; i < p.numel; ++i) {
		let v = float(rl[i]);
		if (v > best) { best = v; best_i = i; }
	}
	p.out_idx[row] = best_i < 0 ? -1 : best_i + p.index_offset;
	if (p.has_val != 0) p.out_val[row] = best;
}
#endif

// Router: logit[e] = sum over k slices of partials[(row * max_slices + ks) * 288 + e] (ascending), raw = sigmoid(logit),
// choose top_k experts by raw + bias[e], weight = raw / sum(raw of chosen) * scale (FP16 out)
struct RouterParams {
	float* partials;
	half* bias;           // may be null
	int* topk_idx;
	half* topk_w;
	int num_experts;      // <= 288
	int top_k;            // <= 16
	int k_slices;
	int max_slices;
	float scale;
	int has_bias;
};

float sigmoid_f(float x) { return 1.0f / (1.0f + exp(-x)); }

#if !ENGINE_TARGET_CPU
groupshared float s_raw[288];
groupshared float s_sel[288];

[shader("compute")]
[numthreads(288, 1, 1)]
void router_select(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RouterParams p) {
	let e = int(tid.x);
	let row = int(gid.x);
	if (e < p.num_experts) {
		float logit = 0.0f;
		for (int ks = 0; ks < p.k_slices; ++ks)
			logit += p.partials[(int64_t(row) * p.max_slices + ks) * 288 + e];
		let raw = sigmoid_f(logit);
		s_raw[e] = raw;
		s_sel[e] = raw + (p.has_bias != 0 ? float(p.bias[e]) : 0.0f);
	}
	GroupMemoryBarrierWithGroupSync();
	if (e == 0) {
		int chosen[16];
		for (int j = 0; j < 16; ++j) chosen[j] = -1;
		float total = 0.0f;
		for (int k = 0; k < p.top_k; ++k) {
			float best = -1e30f;
			int best_i = 0;
			for (int i = 0; i < p.num_experts; ++i) {
				bool taken = false;
				for (int j = 0; j < k; ++j) taken = taken || chosen[j] == i;
				if (!taken && s_sel[i] > best) { best = s_sel[i]; best_i = i; }
			}
			chosen[k] = best_i;
			total += s_raw[best_i];
		}
		let inv = p.scale / (total + 1e-9f);
		for (int k = 0; k < p.top_k; ++k) {
			p.topk_idx[row * p.top_k + k] = chosen[k];
			p.topk_w[row * p.top_k + k] = half(s_raw[chosen[k]] * inv);
		}
	}
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void router_select(uint3 gid : SV_GroupID, uniform RouterParams p) {
	let row = int(gid.x);
	float raw[288];
	float sel[288];
	for (int e = 0; e < p.num_experts; ++e) {
		float logit = 0.0f;
		for (int ks = 0; ks < p.k_slices; ++ks)
			logit += p.partials[(int64_t(row) * p.max_slices + ks) * 288 + e];
		raw[e] = sigmoid_f(logit);
		sel[e] = raw[e] + (p.has_bias != 0 ? float(p.bias[e]) : 0.0f);
	}
	int chosen[16];
	for (int j = 0; j < 16; ++j) chosen[j] = -1;
	float total = 0.0f;
	for (int k = 0; k < p.top_k; ++k) {
		float best = -1e30f;
		int best_i = 0;
		for (int i = 0; i < p.num_experts; ++i) {
			bool taken = false;
			for (int j = 0; j < k; ++j) taken = taken || chosen[j] == i;
			if (!taken && sel[i] > best) { best = sel[i]; best_i = i; }
		}
		chosen[k] = best_i;
		total += raw[best_i];
	}
	let inv = p.scale / (total + 1e-9f);
	for (int k = 0; k < p.top_k; ++k) {
		p.topk_idx[row * p.top_k + k] = chosen[k];
		p.topk_w[row * p.top_k + k] = half(raw[chosen[k]] * inv);
	}
}
#endif
