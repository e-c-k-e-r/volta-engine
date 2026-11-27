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
// scales the sum: resid = ( resid + rms_norm( y ) * w ) * scale, and with next_mode the new rows into next as FP16 too (1: as they are, 2:
// normalized as rms_norm_f32 does, with next_weight)
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
	half* next;
	half* next_weight;
	int next_mode;
	int has_next_weight;
};

// Gemma 4's per-layer inputs: proj and tok [rows, layers * dim] hold each row's layer slices; out [layers, rows, dim] =
// ( rms_norm( the proj slice ) * weight + the tok slice * tok_scale ) * out_scale. A group per (row, layer)
struct PleMixParams {
	half* proj;
	half* tok;
	half* weight;
	half* out;
	int rows;
	int layers;
	int dim;
	float eps;
	float tok_scale;
	float out_scale;
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

// The sum over a 256-thread group, a tree in shared memory: the order the norms have always summed in (Gemma 4 31B's large residuals make
// its logits move with the order, so a subgroup reduction is not a free change)
float group_sum256(int t, float v) {
	s_red[t] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let total = s_red[0];
	GroupMemoryBarrierWithGroupSync();   // every thread has read s_red[0]
	return total;
}

// Columns a thread keeps in registers between the passes (rows up to 16 x 256 = 4096 wide); wider rows read them again
static const int ADD_NORMED_REG = 16;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_normed(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform AddNormedParams p) {
	let t = int(tid.x);
	let H = p.hidden_size;
	half* row_y = p.y + int64_t(gid.x) * H;
	float* row_r = p.resid + int64_t(gid.x) * H;
	half* row_n = p.next + int64_t(gid.x) * H;
	let held = H <= ADD_NORMED_REG * 256;
	float yv[ADD_NORMED_REG];
	float rv[ADD_NORMED_REG];
	float wv[ADD_NORMED_REG];    // both norms' weights read with the row: every global read in one round
	float nwv[ADD_NORMED_REG];
	float sq = 0.0f;
	if (held) {
		[ForceUnroll]
		for (int k = 0; k < ADD_NORMED_REG; ++k) {
			let i = t + k * 256;
			yv[k] = i < H ? float(row_y[i]) : 0.0f;
			rv[k] = i < H ? row_r[i] : 0.0f;
			wv[k] = i < H && p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
			nwv[k] = i < H && p.next_mode == 2 && p.has_next_weight != 0 ? float(p.next_weight[i]) + p.bias : 1.0f;
			sq = fma(yv[k], yv[k], sq);
		}
	} else {
		for (int i = t; i < H; i += 256) {
			let v = float(row_y[i]);
			sq = fma(v, v, sq);
		}
	}
	let inv = inv_rms_from(group_sum256(t, sq), H, p.eps);
	float nsq = 0.0f;
	if (held) {
		[ForceUnroll]
		for (int k = 0; k < ADD_NORMED_REG; ++k) {
			let i = t + k * 256;
			if (i < H) {
				let r = (rv[k] + yv[k] * inv * wv[k]) * p.scale;
				rv[k] = r;
				row_r[i] = r;
				if (p.next_mode == 1) row_n[i] = half(r);
				nsq = fma(r, r, nsq);
			}
		}
	} else {
		for (int i = t; i < H; i += 256) {
			let w = p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
			let r = (row_r[i] + float(row_y[i]) * inv * w) * p.scale;
			row_r[i] = r;
			if (p.next_mode == 1) row_n[i] = half(r);
			nsq = fma(r, r, nsq);
		}
	}
	if (p.next_mode != 2) return;
	let ninv = inv_rms_from(group_sum256(t, nsq), H, p.eps);
	if (held) {
		[ForceUnroll]
		for (int k = 0; k < ADD_NORMED_REG; ++k) {
			let i = t + k * 256;
			if (i < H) row_n[i] = half(rv[k] * nwv[k] * ninv);
		}
	} else {
		for (int i = t; i < H; i += 256) {
			let w = p.has_next_weight != 0 ? float(p.next_weight[i]) + p.bias : 1.0f;
			row_n[i] = half(row_r[i] * w * ninv);
		}
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ple_mix(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PleMixParams p) {
	let t = int(tid.x);
	let in_off = (int64_t(gid.x) * p.layers + int64_t(gid.y)) * p.dim;
	half* pr = p.proj + in_off;
	half* tk = p.tok + in_off;
	half* o = p.out + (int64_t(gid.y) * p.rows + int64_t(gid.x)) * p.dim;
	float sq = 0.0f;
	for (int i = t; i < p.dim; i += 256) {
		let v = float(pr[i]);
		sq = fma(v, v, sq);
	}
	s_red[t] = sq;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let inv = inv_rms_from(s_red[0], p.dim, p.eps);
	for (int i = t; i < p.dim; i += 256) o[i] = half((float(pr[i]) * inv * float(p.weight[i]) + float(tk[i]) * p.tok_scale) * p.out_scale);
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
	half* row_n = p.next + int64_t(gid.x) * p.hidden_size;
	float nsq = 0.0f;
	for (int i = 0; i < p.hidden_size; ++i) {
		let w = p.has_weight != 0 ? float(p.weight[i]) + p.bias : 1.0f;
		let r = (row_r[i] + float(row_y[i]) * inv * w) * p.scale;
		row_r[i] = r;
		if (p.next_mode == 1) row_n[i] = half(r);
		nsq = fma(r, r, nsq);
	}
	if (p.next_mode != 2) return;
	let ninv = inv_rms_from(nsq, p.hidden_size, p.eps);
	for (int i = 0; i < p.hidden_size; ++i) {
		let w = p.has_next_weight != 0 ? float(p.next_weight[i]) + p.bias : 1.0f;
		row_n[i] = half(row_r[i] * w * ninv);
	}
}

[shader("compute")]
[numthreads(1, 1, 1)]
void ple_mix(uint3 gid : SV_GroupID, uniform PleMixParams p) {
	let in_off = (int64_t(gid.x) * p.layers + int64_t(gid.y)) * p.dim;
	half* pr = p.proj + in_off;
	half* tk = p.tok + in_off;
	half* o = p.out + (int64_t(gid.y) * p.rows + int64_t(gid.x)) * p.dim;
	float sq = 0.0f;
	for (int i = 0; i < p.dim; ++i) {
		let v = float(pr[i]);
		sq = fma(v, v, sq);
	}
	let inv = inv_rms_from(sq, p.dim, p.eps);
	for (int i = 0; i < p.dim; ++i) o[i] = half((float(pr[i]) * inv * float(p.weight[i]) + float(tk[i]) * p.tok_scale) * p.out_scale);
}
#endif
