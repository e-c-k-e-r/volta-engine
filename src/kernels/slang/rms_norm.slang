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

// An FP32 residual stream (Gemma 4): rms_norm_f32 normalizes FP32 rows into FP16; add_normed adds a normalized FP16 sublayer output to it and
// scales the sum: resid = ( resid + rms_norm( y ) * w ) * scale
struct NormF32Params {
	float* src;
	half* dst;
	half* weight;
	int hidden_size;
	int has_weight;
	float eps;
	float bias;
};

struct AddNormedParams {
	float* resid;
	half* y;
	half* weight;
	int hidden_size;
	int has_weight;
	float eps;
	float bias;
	float scale;
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

// qk viewed as [rows, num_heads, head_dim]; head_dim <= 512; in place
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
	// thread t owns elements t and t + 256 (head_dim up to 512)
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
	let inv = inv_rms_from(s_red[0], p.head_dim, p.eps);
	if (active) {
		let w = p.has_weight != 0 ? float(p.weight[t]) + p.bias : 1.0f;
		ptr[t] = half(val * w * inv);
	}
	if (active1) {
		let w = p.has_weight != 0 ? float(p.weight[t + 256]) + p.bias : 1.0f;
		ptr[t + 256] = half(val1 * w * inv);
	}
}
[shader("compute")]
[numthreads(256, 1, 1)]
void rms_norm_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform NormF32Params p) {
	let t = int(tid.x);
	float* row_src = p.src + int64_t(gid.x) * p.hidden_size;
	half* row_dst = p.dst + int64_t(gid.x) * p.hidden_size;
	float sq = 0.0f;
	for (int i = t; i < p.hidden_size; i += 256) sq = fma(row_src[i], row_src[i], sq);
	s_red[t] = sq;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let inv = inv_rms_from(s_red[0], p.hidden_size, p.eps);
	for (int i = t; i < p.hidden_size; i += 256) {
		let w = p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
		row_dst[i] = half(row_src[i] * w * inv);
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_normed(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform AddNormedParams p) {
	let t = int(tid.x);
	half* row_y = p.y + int64_t(gid.x) * p.hidden_size;
	float* row_r = p.resid + int64_t(gid.x) * p.hidden_size;
	float sq = 0.0f;
	for (int i = t; i < p.hidden_size; i += 256) {
		let v = float(row_y[i]);
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
		row_r[i] = (row_r[i] + float(row_y[i]) * inv * w) * p.scale;
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
[shader("compute")]
[numthreads(1, 1, 1)]
void rms_norm_f32(uint3 gid : SV_GroupID, uniform NormF32Params p) {
	float* row_src = p.src + int64_t(gid.x) * p.hidden_size;
	half* row_dst = p.dst + int64_t(gid.x) * p.hidden_size;
	float sq = 0.0f;
	for (int i = 0; i < p.hidden_size; ++i) sq = fma(row_src[i], row_src[i], sq);
	let inv = inv_rms_from(sq, p.hidden_size, p.eps);
	for (int i = 0; i < p.hidden_size; ++i) {
		let w = p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
		row_dst[i] = half(row_src[i] * w * inv);
	}
}

[shader("compute")]
[numthreads(1, 1, 1)]
void add_normed(uint3 gid : SV_GroupID, uniform AddNormedParams p) {
	half* row_y = p.y + int64_t(gid.x) * p.hidden_size;
	float* row_r = p.resid + int64_t(gid.x) * p.hidden_size;
	float sq = 0.0f;
	for (int i = 0; i < p.hidden_size; ++i) {
		let v = float(row_y[i]);
		sq = fma(v, v, sq);
	}
	let inv = inv_rms_from(sq, p.hidden_size, p.eps);
	for (int i = 0; i < p.hidden_size; ++i) {
		let w = p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
		row_r[i] = (row_r[i] + float(row_y[i]) * inv * w) * p.scale;
	}
}
#endif
