// Multi-head hyper-connection kernels (Tier B), 4 streams of FP32 [rows, 4 * D]: stream init, the collapse mix (partial sums
// + finalize with softmax and Sinkhorn), the stream update and the hyper-head mean. Reductions are shared-memory trees on
// the GPU targets and serial entry points on the host target (ENGINE_TARGET_CPU). M = 24 mix logits: 4 pre, 4 post, 16 comb.

struct InitStreamsParams {
	half* in_;        // [rows, D]
	float* streams;   // [rows, 4 * D]
	int hidden_size;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void init_streams(uint3 tid : SV_DispatchThreadID, uniform InitStreamsParams p) {
	let d = int(tid.x);
	if (d >= p.hidden_size) return;
	let r = int64_t(tid.y);
	let v = float(p.in_[r * p.hidden_size + d]);
	float* o = p.streams + r * 4 * p.hidden_size;
	for (int h = 0; h < 4; ++h) o[h * p.hidden_size + d] = v;
}

// x_out[r][h] = post[h] * y + sum_g comb[g][h] * x_in[r][g] (or + x_in[h] without comb). x_out may alias x_in.
struct ApplyParams {
	float* x_in;
	float* x_out;
	half* y;          // [rows, D]
	float* post;      // [rows, 4]
	float* comb;      // [rows, 4, 4]
	int hidden_size;
	int has_comb;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void hc_apply(uint3 tid : SV_DispatchThreadID, uniform ApplyParams p) {
	let d = int(tid.x);
	if (d >= p.hidden_size) return;
	let r = int64_t(tid.y);
	float* xi = p.x_in + r * 4 * p.hidden_size;
	float* xo = p.x_out + r * 4 * p.hidden_size;
	float xv[4];
	for (int h = 0; h < 4; ++h) xv[h] = xi[h * p.hidden_size + d];
	let yv = float(p.y[r * p.hidden_size + d]);
	for (int h = 0; h < 4; ++h) {
		var o = p.post[r * 4 + h] * yv;
		if (p.has_comb != 0) {
			for (int g = 0; g < 4; ++g) o = fma(p.comb[(r * 4 + g) * 4 + h], xv[g], o);
		}
		else {
			o += xv[h];
		}
		xo[h * p.hidden_size + d] = o;
	}
}

// Partial sums over a chunk of the 4 * D row: acc[j] = sum s * fn[j][col], acc[24] = sum s^2.
// partials layout: ((row * gridDim.x) + chunk) * 25 + k.
struct PartialsParams {
	float* streams;   // [rows, row_len]
	float* fn;        // [24, row_len]
	float* partials;
	int row_len;
	int chunk_cols;
	int num_chunks;
};

#if !ENGINE_TARGET_CPU
groupshared float s_part[25 * 64];

[shader("compute")]
[numthreads(64, 1, 1)]
void hc_mix_partials(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PartialsParams p) {
	let t = int(tid.x);
	let r = int64_t(gid.y);
	let c0 = int(gid.x) * p.chunk_cols;
	let c1 = min(c0 + p.chunk_cols, p.row_len);
	float* s = p.streams + r * p.row_len;
	float acc[25];
	for (int k = 0; k < 25; ++k) acc[k] = 0.0f;
	for (int c = c0 + t; c < c1; c += 64) {
		let sv = s[c];
		acc[24] = fma(sv, sv, acc[24]);
		for (int j = 0; j < 24; ++j) acc[j] = fma(sv, p.fn[int64_t(j) * p.row_len + c], acc[j]);
	}
	for (int k = 0; k < 25; ++k) s_part[k * 64 + t] = acc[k];
	GroupMemoryBarrierWithGroupSync();
	for (int st = 32; st > 0; st >>= 1) {
		if (t < st) {
			for (int k = 0; k < 25; ++k) s_part[k * 64 + t] += s_part[k * 64 + t + st];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (t < 25) p.partials[(r * p.num_chunks + int(gid.x)) * 25 + t] = s_part[t * 64];
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void hc_mix_partials(uint3 gid : SV_GroupID, uniform PartialsParams p) {
	let r = int64_t(gid.y);
	let c0 = int(gid.x) * p.chunk_cols;
	let c1 = min(c0 + p.chunk_cols, p.row_len);
	float* s = p.streams + r * p.row_len;
	float acc[25];
	for (int k = 0; k < 25; ++k) acc[k] = 0.0f;
	for (int c = c0; c < c1; ++c) {
		let sv = s[c];
		acc[24] += sv * sv;
		for (int j = 0; j < 24; ++j) acc[j] += sv * p.fn[int64_t(j) * p.row_len + c];
	}
	for (int k = 0; k < 25; ++k) p.partials[(r * p.num_chunks + int(gid.x)) * 25 + k] = acc[k];
}
#endif

float sigmoid_m(float x) {
	return 1.0f / (1.0f + exp(-x));
}

struct FinalizeParams {
	float* streams;
	float* partials;
	float* base;      // [24]
	float* scale;     // [3]
	float* post;      // [rows, 4]
	float* comb;      // [rows, 4, 4]
	half* collapsed;  // [rows, D]
	float* inv_rms_buf;     // [rows], may be null
	float* mix_logits_buf;  // [rows, 24], may be null
	float* pre_buf;         // [rows, 4], may be null
	int hidden_size;
	int num_chunks;   // partial chunks per row
	int chunk_cols_c; // output columns per group
	int sinkhorn_iters;
	float rms_eps;
	float hc_eps;
};

// Group (bx == 0) also publishes post, comb (softmax over each row of 4, then Sinkhorn: column, then (iters - 1) x (row, column))
void hc_post_comb(FinalizeParams p, int r, float mix[25], float rmr) {
	for (int h = 0; h < 4; ++h)
		p.post[r * 4 + h] = 2.0f * sigmoid_m(fma(mix[4 + h] * rmr, p.scale[1], p.base[4 + h]));
	float v[16];
	for (int i = 0; i < 16; ++i) v[i] = fma(mix[8 + i] * rmr, p.scale[2], p.base[8 + i]);
	for (int a = 0; a < 4; ++a) {
		var m = v[a * 4];
		for (int b = 1; b < 4; ++b) m = max(m, v[a * 4 + b]);
		var s = 0.0f;
		for (int b = 0; b < 4; ++b) {
			v[a * 4 + b] = exp(v[a * 4 + b] - m);
			s += v[a * 4 + b];
		}
		for (int b = 0; b < 4; ++b) v[a * 4 + b] = v[a * 4 + b] / s + p.hc_eps;
	}
	for (int b = 0; b < 4; ++b) {
		var cs = 0.0f;
		for (int a = 0; a < 4; ++a) cs += v[a * 4 + b];
		for (int a = 0; a < 4; ++a) v[a * 4 + b] = v[a * 4 + b] / (cs + p.hc_eps);
	}
	for (int it = 0; it < p.sinkhorn_iters - 1; ++it) {
		for (int a = 0; a < 4; ++a) {
			var rs = 0.0f;
			for (int b = 0; b < 4; ++b) rs += v[a * 4 + b];
			for (int b = 0; b < 4; ++b) v[a * 4 + b] = v[a * 4 + b] / (rs + p.hc_eps);
		}
		for (int b = 0; b < 4; ++b) {
			var cs = 0.0f;
			for (int a = 0; a < 4; ++a) cs += v[a * 4 + b];
			for (int a = 0; a < 4; ++a) v[a * 4 + b] = v[a * 4 + b] / (cs + p.hc_eps);
		}
	}
	for (int i = 0; i < 16; ++i) p.comb[r * 16 + i] = v[i];
}

void hc_publish_debug(FinalizeParams p, int r, float mix[25], float rmr, float pre[4]) {
	if (p.inv_rms_buf != nullptr) p.inv_rms_buf[r] = rmr;
	if (p.mix_logits_buf != nullptr) {
		for (int k = 0; k < 24; ++k) p.mix_logits_buf[r * 24 + k] = mix[k];
	}
	if (p.pre_buf != nullptr) {
		for (int h = 0; h < 4; ++h) p.pre_buf[r * 4 + h] = pre[h];
	}
}

#if !ENGINE_TARGET_CPU
groupshared float s_mix[25];

[shader("compute")]
[numthreads(256, 1, 1)]
void hc_mix_finalize(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform FinalizeParams p) {
	let t = int(tid.x);
	let r = int(gid.y);
	let D = p.hidden_size;
	if (t < 25) {
		float v = 0.0f;
		for (int i = 0; i < p.num_chunks; ++i) v += p.partials[(int64_t(r) * p.num_chunks + i) * 25 + t];
		s_mix[t] = v;
	}
	GroupMemoryBarrierWithGroupSync();
	float mix[25];
	for (int k = 0; k < 25; ++k) mix[k] = s_mix[k];
	let rmr = rsqrt(mix[24] / float(4 * D) + p.rms_eps);
	float pre[4];
	for (int h = 0; h < 4; ++h) pre[h] = sigmoid_m(fma(mix[h] * rmr, p.scale[0], p.base[h])) + p.hc_eps;
	if (gid.x == 0 && t == 0) {
		hc_post_comb(p, r, mix, rmr);
		hc_publish_debug(p, r, mix, rmr, pre);
	}
	let c0 = int(gid.x) * p.chunk_cols_c;
	let c1 = min(c0 + p.chunk_cols_c, D);
	float* s = p.streams + int64_t(r) * 4 * D;
	for (int d = c0 + t; d < c1; d += 256) {
		var o = 0.0f;
		for (int h = 0; h < 4; ++h) o = fma(pre[h], s[h * D + d], o);
		p.collapsed[int64_t(r) * D + d] = half(o);
	}
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void hc_mix_finalize(uint3 gid : SV_GroupID, uniform FinalizeParams p) {
	let r = int(gid.y);
	let D = p.hidden_size;
	float mix[25];
	for (int k = 0; k < 25; ++k) {
		float v = 0.0f;
		for (int i = 0; i < p.num_chunks; ++i) v += p.partials[(int64_t(r) * p.num_chunks + i) * 25 + k];
		mix[k] = v;
	}
	let rmr = rsqrt(mix[24] / float(4 * D) + p.rms_eps);
	float pre[4];
	for (int h = 0; h < 4; ++h) pre[h] = sigmoid_m(fma(mix[h] * rmr, p.scale[0], p.base[h])) + p.hc_eps;
	if (gid.x == 0) {
		hc_post_comb(p, r, mix, rmr);
		hc_publish_debug(p, r, mix, rmr, pre);
	}
	let c0 = int(gid.x) * p.chunk_cols_c;
	let c1 = min(c0 + p.chunk_cols_c, D);
	float* s = p.streams + int64_t(r) * 4 * D;
	for (int d = c0; d < c1; ++d) {
		float o = 0.0f;
		for (int h = 0; h < 4; ++h) o += pre[h] * s[h * D + d];
		p.collapsed[int64_t(r) * D + d] = half(o);
	}
}
#endif

struct HeadMeanParams {
	float* streams;   // [rows, 4 * D]
	half* final_hidden;
	int hidden_size;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void hyperhead_mean(uint3 tid : SV_DispatchThreadID, uniform HeadMeanParams p) {
	let d = int(tid.x);
	if (d >= p.hidden_size) return;
	let r = int64_t(tid.y);
	float* s = p.streams + r * 4 * p.hidden_size;
	float o = 0.0f;
	for (int h = 0; h < 4; ++h) o += s[h * p.hidden_size + d];
	p.final_hidden[r * p.hidden_size + d] = half(o * 0.25f);
}
