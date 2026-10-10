// Vision-tower operators for the Vulkan backend (GPU targets only; the CPU backend has plain C++ in src/backend/cpu/vision.cpp): norms, patch embedding bias,
// interpolated position table, rotary with biases, bidirectional attention, bias / activation / residual helpers and the merger's gather. Same arithmetic as
// src/backend/cuda/modules/vision.cu; slow but simple. The GEMMs go through ref_gemm_nt_f16.

groupshared float v_red[256];

float vsum256(int t, float v) {
	v_red[t] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) v_red[t] += v_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let r = v_red[0];
	GroupMemoryBarrierWithGroupSync();
	return r;
}

struct F32ToF16Params { float* in_; half* out_; int64_t n; };

[shader("compute")]
[numthreads(256, 1, 1)]
void vision_f32_to_f16(uint3 tid : SV_DispatchThreadID, uniform F32ToF16Params p) {
	let i = int64_t(tid.x) + int64_t(tid.y) * 65535 * 256;
	if (i < p.n) p.out_[i] = half(p.in_[i]);
}

struct PosInterpParams { half* table; half* out_; int gh; int gw; int merge; int side; int dim; };

// Group per output row (merge-block order)
[shader("compute")]
[numthreads(128, 1, 1)]
void vision_pos_interp(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PosInterpParams p) {
	let row = int(gid.x);
	let blocks_w = p.gw / p.merge;
	let mm = p.merge * p.merge;
	let blk = row / mm;
	let within = row % mm;
	let by = blk / blocks_w;
	let bx = blk % blocks_w;
	let y = by * p.merge + within / p.merge;
	let x = bx * p.merge + within % p.merge;
	let hy = p.gh == 1 ? 0.0f : float(y) * float(p.side - 1) / float(p.gh - 1);
	let wx = p.gw == 1 ? 0.0f : float(x) * float(p.side - 1) / float(p.gw - 1);
	let hf = int(hy);
	let wf = int(wx);
	let hc = min(hf + 1, p.side - 1);
	let wc = min(wf + 1, p.side - 1);
	let dh = hy - float(hf);
	let dw = wx - float(wf);
	half* t00 = p.table + (int64_t(hf) * p.side + wf) * p.dim;
	half* t01 = p.table + (int64_t(hf) * p.side + wc) * p.dim;
	half* t10 = p.table + (int64_t(hc) * p.side + wf) * p.dim;
	half* t11 = p.table + (int64_t(hc) * p.side + wc) * p.dim;
	for (int d = int(tid.x); d < p.dim; d += 128) {
		let val = (1.0f - dh) * (1.0f - dw) * float(t00[d]) + (1.0f - dh) * dw * float(t01[d]) + dh * (1.0f - dw) * float(t10[d]) + dh * dw * float(t11[d]);
		p.out_[int64_t(row) * p.dim + d] = half(val);
	}
}

struct PatchEmbedParams { half* patches; half* w; half* bias; half* add; float* out_; int m; int n; int k; int has_add; };

// Thread (column, row): fp32 GEMM, then bias and the optional position add
[shader("compute")]
[numthreads(64, 1, 1)]
void vision_patch_embed(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PatchEmbedParams p) {
	let c = int(gid.x) * 64 + int(tid.x);
	let r = int(gid.y);
	if (c >= p.n) return;
	half* pr = p.patches + int64_t(r) * p.k;
	half* wr = p.w + int64_t(c) * p.k;
	float acc = 0.0f;
	for (int i = 0; i < p.k; ++i) acc = fma(float(pr[i]), float(wr[i]), acc);
	p.out_[int64_t(r) * p.n + c] = acc + float(p.bias[c]) + (p.has_add != 0 ? float(p.add[int64_t(r) * p.n + c]) : 0.0f);
}

struct VNormParams { float* x32; half* x16; half* w; half* b; half* out_; int n; int group; int kind; float eps; };

// Group per row. kind 0: LayerNorm of an FP32 row (output grouped as in layer_norm_merge), 1: LayerNorm of an FP16 row, 2: RMS norm of an FP32 row (gain w, no bias)
[shader("compute")]
[numthreads(256, 1, 1)]
void vision_norm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform VNormParams p) {
	let t = int(tid.x);
	let row = int(gid.x);
	float s = 0.0f;
	for (int i = t; i < p.n; i += 256) s += p.kind == 1 ? float(p.x16[int64_t(row) * p.n + i]) : p.x32[int64_t(row) * p.n + i];
	let mean = p.kind == 2 ? 0.0f : vsum256(t, s) / p.n;
	float q = 0.0f;
	for (int i = t; i < p.n; i += 256) {
		let v = p.kind == 1 ? float(p.x16[int64_t(row) * p.n + i]) : p.x32[int64_t(row) * p.n + i];
		q += (v - mean) * (v - mean);
	}
	let inv = rsqrt(vsum256(t, q) / p.n + p.eps);
	half* o = p.out_ + int64_t(row / p.group) * p.group * p.n + int64_t(row % p.group) * p.n;
	for (int i = t; i < p.n; i += 256) {
		let v = p.kind == 1 ? float(p.x16[int64_t(row) * p.n + i]) : p.x32[int64_t(row) * p.n + i];
		if (p.kind == 2) o[i] = half(v * inv * float(p.w[i]));
		else o[i] = half((v - mean) * inv * float(p.w[i]) + float(p.b[i]));
	}
}

struct QkvRopeParams { half* q; half* k; half* v; half* bq; half* bk; half* bv; half* qn; half* kn; float* cs; float* sn; int heads; int head_dim; float eps; int norm; };

// Group per row, a thread per (head, rotary pair), up to 4 pairs each. norm = 1: head_dim 64 with the per-head RMS norm of q and k before the rotary
// (qkv_bias_norm_rope); norm = 0: bias and rotary only (qkv_bias_rope). A pair's new values are computed from the unmodified row, then written after a barrier
// (other pairs of the same head read the same inputs for the norm)
[shader("compute")]
[numthreads(256, 1, 1)]
void vision_qkv_rope(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform QkvRopeParams p) {
	let row = int(gid.x);
	let half_dim = p.head_dim / 2;
	let width = p.heads * p.head_dim;
	let base = int64_t(row) * width;
	let pairs = p.heads * half_dim;
	float na[12];
	float nb[12];
	for (int it = 0; it < 4; ++it) {
		let idx = int(tid.x) + it * 256;
		for (int which = 0; which < 3; ++which) {
			na[it * 3 + which] = 0.0f;
			nb[it * 3 + which] = 0.0f;
		}
		if (idx < pairs) {
			let h = idx / half_dim;
			let d = idx % half_dim;
			let i1 = h * p.head_dim + d;
			let i2 = i1 + half_dim;
			let c1 = p.cs[int64_t(row) * p.head_dim + d];
			let s1 = p.sn[int64_t(row) * p.head_dim + d];
			let c2 = p.cs[int64_t(row) * p.head_dim + d + half_dim];
			let s2 = p.sn[int64_t(row) * p.head_dim + d + half_dim];
			for (int which = 0; which < 3; ++which) {
				half* t = which == 0 ? p.q : (which == 1 ? p.k : p.v);
				half* bias = which == 0 ? p.bq : (which == 1 ? p.bk : p.bv);
				float a = float(t[base + i1]) + float(bias[i1]);
				float b = float(t[base + i2]) + float(bias[i2]);
				if (which < 2) {
					if (p.norm != 0) {
						float ss = 0.0f;
						for (int e = 0; e < half_dim; ++e) {
							let ea = float(t[base + h * p.head_dim + e]) + float(bias[h * p.head_dim + e]);
							let eb = float(t[base + h * p.head_dim + e + half_dim]) + float(bias[h * p.head_dim + e + half_dim]);
							ss += ea * ea + eb * eb;
						}
						let inv = rsqrt(ss / float(p.head_dim) + p.eps);
						half* gain = which == 0 ? p.qn : p.kn;
						a = a * inv * float(gain[d]);
						b = b * inv * float(gain[d + half_dim]);
					}
					let ra = a * c1 - b * s1;
					let rb = b * c2 + a * s2;
					a = ra;
					b = rb;
				}
				na[it * 3 + which] = a;
				nb[it * 3 + which] = b;
			}
		}
	}
	GroupMemoryBarrierWithGroupSync();
	for (int it = 0; it < 4; ++it) {
		let idx = int(tid.x) + it * 256;
		if (idx < pairs) {
			let h = idx / half_dim;
			let d = idx % half_dim;
			let i1 = h * p.head_dim + d;
			let i2 = i1 + half_dim;
			for (int which = 0; which < 3; ++which) {
				half* t = which == 0 ? p.q : (which == 1 ? p.k : p.v);
				t[base + i1] = half(na[it * 3 + which]);
				t[base + i2] = half(nb[it * 3 + which]);
			}
		}
	}
}

struct MergeGatherParams { half* h; half* out_; int n; int group; };

[shader("compute")]
[numthreads(256, 1, 1)]
void vision_merge_gather(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform MergeGatherParams p) {
	let t = int(gid.x);
	for (int idx = int(tid.x); idx < p.n * p.group; idx += 256) {
		let c = idx / p.group;
		let j = idx % p.group;
		p.out_[int64_t(t) * p.n * p.group + idx] = p.h[(int64_t(t) * p.group + j) * p.n + c];
	}
}

struct VAttnParams { half* q; half* k; half* v; half* out_; int m; int heads; int head_dim; float scale; };

groupshared float va_q[256];
groupshared float va_s[256];

// Group (head, query): online softmax over all m keys in tiles of 256
[shader("compute")]
[numthreads(256, 1, 1)]
void vision_attention(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform VAttnParams p) {
	let t = int(tid.x);
	let head = int(gid.x);
	let qi = int(gid.y);
	let width = p.heads * p.head_dim;
	let active = t < p.head_dim;
	va_q[t] = active ? float(p.q[int64_t(qi) * width + head * p.head_dim + t]) * p.scale : 0.0f;
	GroupMemoryBarrierWithGroupSync();
	float m_run = -1e30f;
	float l_run = 0.0f;
	float o = 0.0f;
	for (int t0 = 0; t0 < p.m; t0 += 256) {
		let n = min(256, p.m - t0);
		float score = -1e30f;
		if (t < n) {
			half* kr = p.k + int64_t(t0 + t) * width + head * p.head_dim;
			float dot = 0.0f;
			for (int d = 0; d < p.head_dim; ++d) dot = fma(va_q[d], float(kr[d]), dot);
			score = dot;
		}
		v_red[t] = score;
		GroupMemoryBarrierWithGroupSync();
		for (int s = 128; s > 0; s >>= 1) {
			if (t < s) v_red[t] = max(v_red[t], v_red[t + s]);
			GroupMemoryBarrierWithGroupSync();
		}
		let m_new = max(m_run, v_red[0]);
		GroupMemoryBarrierWithGroupSync();
		let pk = t < n ? exp(score - m_new) : 0.0f;
		va_s[t] = pk;
		v_red[t] = pk;
		GroupMemoryBarrierWithGroupSync();
		for (int s = 128; s > 0; s >>= 1) {
			if (t < s) v_red[t] += v_red[t + s];
			GroupMemoryBarrierWithGroupSync();
		}
		let alpha = exp(m_run - m_new);
		l_run = l_run * alpha + v_red[0];
		m_run = m_new;
		o *= alpha;
		if (active) {
			for (int kk = 0; kk < n; ++kk) o = fma(va_s[kk], float(p.v[int64_t(t0 + kk) * width + head * p.head_dim + t]), o);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (active) p.out_[int64_t(qi) * width + head * p.head_dim + t] = half(o / l_run);
}

struct BiasActParams { half* x; half* bias; int64_t total; int n; int act; int has_bias; };

[shader("compute")]
[numthreads(256, 1, 1)]
void vision_bias_act(uint3 tid : SV_DispatchThreadID, uniform BiasActParams p) {
	let i = int64_t(tid.x) + int64_t(tid.y) * 65535 * 256;
	if (i >= p.total) return;
	float v = float(p.x[i]) + (p.has_bias != 0 ? float(p.bias[i % p.n]) : 0.0f);
	if (p.act == 1) {
		let u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
		v = 0.5f * v * (1.0f + tanh(u));
	}
	else if (p.act == 2) {
		// erf by the Abramowitz-Stegun 7.1.26 form refined once: |error| < 1.5e-7
		let z = v * 0.7071067811865476f;
		let a = abs(z);
		let tt = 1.0f / (1.0f + 0.3275911f * a);
		let poly = tt * (0.254829592f + tt * (-0.284496736f + tt * (1.421413741f + tt * (-1.453152027f + tt * 1.061405429f))));
		let e = 1.0f - poly * exp(-a * a);
		v = 0.5f * v * (1.0f + (z < 0.0f ? -e : e));
	}
	p.x[i] = half(v);
}

struct BiasResidualParams { float* hidden; half* y; half* bias; int64_t total; int n; };

[shader("compute")]
[numthreads(256, 1, 1)]
void vision_bias_residual(uint3 tid : SV_DispatchThreadID, uniform BiasResidualParams p) {
	let i = int64_t(tid.x) + int64_t(tid.y) * 65535 * 256;
	if (i >= p.total) return;
	p.hidden[i] += float(p.y[i]) + float(p.bias[i % p.n]);
}
