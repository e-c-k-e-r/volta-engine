// RMSNorm over rows and over per-head slices (Tier B portable kernels; the CUDA build keeps the hand-tuned versions).
// One thread group per row (rms_norm) or per (head, row) (rms_norm_heads), 256 threads, shared-memory tree reduction so the
// same source compiles for CUDA, SPIR-V and the host. ENGINE_TARGET_CPU selects the serial entry points for the host target,
// which has no group barriers.

struct NormParams {
	half* src;
	half* dst;
	half* weight;   // may be null: unit weight
	int hidden_size;
	int has_weight;
	float eps;
	float bias;
};

groupshared float s_red[256];

float inv_rms_from(float total, int n, float eps) {
	return rsqrt(total / float(n) + eps);
}

#if !ENGINE_TARGET_CPU
[shader("compute")]
[numthreads(256, 1, 1)]
void rms_norm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform NormParams p) {
	let t = int(tid.x);
	half* row_src = p.src + int64_t(gid.x) * p.hidden_size;
	half* row_dst = p.dst + int64_t(gid.x) * p.hidden_size;
	float sq = 0.0f;
	for (int i = t; i < p.hidden_size; i += 256) {
		let v = float(row_src[i]);
		sq = fma(v, v, sq);
	}
	s_red[t] = sq;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let inv = inv_rms_from(s_red[0], p.hidden_size, p.eps);
	for (int i = t; i < p.hidden_size; i += 256) {
		let w = p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
		row_dst[i] = half(float(row_src[i]) * w * inv);
	}
}

// qk viewed as [rows, num_heads, head_dim]; head_dim <= 256; in place
struct HeadNormParams {
	half* qk;
	half* weight;   // [head_dim], may be null
	int num_heads;
	int head_dim;
	int has_weight;
	float eps;
	float bias;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void rms_norm_heads(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform HeadNormParams p) {
	let t = int(tid.x);
	half* ptr = p.qk + (int64_t(gid.y) * p.num_heads + int64_t(gid.x)) * p.head_dim;
	let active = t < p.head_dim;
	let val = active ? float(ptr[t]) : 0.0f;
	s_red[t] = val * val;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let inv = inv_rms_from(s_red[0], p.head_dim, p.eps);
	if (active) {
		let w = p.has_weight != 0 ? float(p.weight[t]) + p.bias : 1.0f;
		ptr[t] = half(val * w * inv);
	}
}
#else
// Host target: one invocation per row / per (head, row), serial sums
struct HeadNormParams {
	half* qk;
	half* weight;
	int num_heads;
	int head_dim;
	int has_weight;
	float eps;
	float bias;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void rms_norm(uint3 gid : SV_GroupID, uniform NormParams p) {
	half* row_src = p.src + int64_t(gid.x) * p.hidden_size;
	half* row_dst = p.dst + int64_t(gid.x) * p.hidden_size;
	float sq = 0.0f;
	for (int i = 0; i < p.hidden_size; ++i) {
		let v = float(row_src[i]);
		sq = fma(v, v, sq);
	}
	let inv = inv_rms_from(sq, p.hidden_size, p.eps);
	for (int i = 0; i < p.hidden_size; ++i) {
		let w = p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
		row_dst[i] = half(float(row_src[i]) * w * inv);
	}
}

[shader("compute")]
[numthreads(1, 1, 1)]
void rms_norm_heads(uint3 gid : SV_GroupID, uniform HeadNormParams p) {
	half* ptr = p.qk + (int64_t(gid.y) * p.num_heads + int64_t(gid.x)) * p.head_dim;
	float sq = 0.0f;
	for (int i = 0; i < p.head_dim; ++i) {
		let v = float(ptr[i]);
		sq = fma(v, v, sq);
	}
	let inv = inv_rms_from(sq, p.head_dim, p.eps);
	for (int i = 0; i < p.head_dim; ++i) {
		let w = p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
		ptr[i] = half(float(ptr[i]) * w * inv);
	}
}
#endif
