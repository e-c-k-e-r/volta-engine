// Qwen4Exp (Flash-Next) kernels for the Vulkan backend (GPU targets only; the CPU backend has plain C++ in src/backend/cpu): gated residual over
// fp32 streams, the n-gram embedding layer (PLE), the QSA indexer, DSA selection, the sparse-list attention, the softmax top-k router and the
// MoE gather / combine. Same arithmetic as src/backend/cuda/modules/{gated_residual,ple,qsa,dsa,attention,moe}.cu; slow but simple: one
// thread per output where the reduction is short, one group per row / stream where it is long (shared-memory trees, no wave ops).

static const int QSA_DIM = 128;
static const int QSA_HEADS = 4;
static const int QSA_POOL = 4;
static const int QSA_RING = 1024;
static const int PLE_RING = 16;
static const int SEL_POOLS = 512;

groupshared float s_red[256];

// Sum over the first n (power of two) threads of the group; every thread gets the total
float gsum(int t, int n, float v) {
	s_red[t] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = n >> 1; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let r = s_red[0];
	GroupMemoryBarrierWithGroupSync();
	return r;
}

float sigm(float v) { return 1.0f / (1.0f + exp(-v)); }

// ---------------------------------------------------------------- gated residual

struct GrInitParams { half* in_; float* x; int hidden; int hc; };

[shader("compute")]
[numthreads(256, 1, 1)]
void gr_init(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrInitParams p) {
	let d = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	if (d >= p.hidden) return;
	let v = float(p.in_[int64_t(r) * p.hidden + d]);
	for (int s = 0; s < p.hc; ++s) p.x[(int64_t(r) * p.hc + s) * p.hidden + d] = v;
}

struct GrNormParams { float* x; half* w; float* normed; float* prev_post; half* prev_y; half* hi; half* lo; int hidden; int hc; float eps; int has_prev; int has_hi; };

// One group per (row, stream): optional folded deferred apply, then x * rsqrt(mean(x^2) + eps) * (1 + w)
[shader("compute")]
[numthreads(256, 1, 1)]
void gr_norm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrNormParams p) {
	let t = int(tid.x);
	let r = int(gid.x);
	let s = int(gid.y);
	let base = (int64_t(r) * p.hc + s) * p.hidden;
	let post = p.has_prev != 0 ? p.prev_post[int64_t(r) * p.hc + s] : 0.0f;
	float sq = 0.0f;
	for (int d = t; d < p.hidden; d += 256) {
		float v = p.x[base + d];
		if (p.has_prev != 0) {
			v += post * float(p.prev_y[int64_t(r) * p.hidden + d]);
			p.x[base + d] = v;
		}
		sq += v * v;
	}
	let inv = rsqrt(gsum(t, 256, sq) / p.hidden + p.eps);
	for (int d = t; d < p.hidden; d += 256) {
		let nv = p.x[base + d] * inv * (1.0f + float(p.w[int64_t(s) * p.hidden + d]));
		p.normed[base + d] = nv;
		if (p.has_hi != 0) {
			let hv = half(nv);
			p.hi[base + d] = hv;
			p.lo[base + d] = half(nv - float(hv));
		}
	}
}

struct GrGateParams { float* normed; half* down; half* inject; float* t_out; float* post; int flat; int rank; int hc; int has_inject; };

// Group (j, row): t[r][j] = silu(normed . down[j] / hc) for j < rank, post[r][j - rank] = 2 sigmoid(normed . inject / hc)
[shader("compute")]
[numthreads(256, 1, 1)]
void gr_gate(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrGateParams p) {
	let t = int(tid.x);
	let j = int(gid.x);
	let r = int(gid.y);
	half* w = j < p.rank ? p.down + int64_t(j) * p.flat : p.inject + int64_t(j - p.rank) * p.flat;
	float* nr = p.normed + int64_t(r) * p.flat;
	float acc = 0.0f;
	for (int k = t; k < p.flat; k += 256) acc = fma(nr[k], float(w[k]), acc);
	let z = gsum(t, 256, acc) / p.hc;
	if (t == 0) {
		if (j < p.rank) p.t_out[int64_t(r) * p.rank + j] = z / (1.0f + exp(-z));
		else p.post[int64_t(r) * p.hc + (j - p.rank)] = 2.0f * sigm(z);
	}
}

struct GrUpMixParams { float* normed; half* up; float* t_in; half* mixed; int hidden; int hc; int rank; int rows; };

// Thread (d, row): mixed = mean_s( sigmoid(up[s * hidden + d] . t) * normed[s][d] )
[shader("compute")]
[numthreads(64, 1, 1)]
void gr_up_mix(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrUpMixParams p) {
	let d = int(gid.x) * 64 + int(tid.x);
	let r = int(gid.y);
	if (d >= p.hidden) return;
	float* tr = p.t_in + int64_t(r) * p.rank;
	float outv = 0.0f;
	for (int s = 0; s < p.hc; ++s) {
		half* u = p.up + (int64_t(s) * p.hidden + d) * p.rank;
		float acc = 0.0f;
		for (int k = 0; k < p.rank; ++k) acc = fma(tr[k], float(u[k]), acc);
		outv += sigm(acc) * p.normed[(int64_t(r) * p.hc + s) * p.hidden + d];
	}
	p.mixed[int64_t(r) * p.hidden + d] = half(outv / p.hc);
}

struct GrApplyParams { float* x; float* post; half* y; int hidden; int hc; };

[shader("compute")]
[numthreads(256, 1, 1)]
void gr_apply(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrApplyParams p) {
	let d = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	let s = int(gid.z);
	if (d >= p.hidden) return;
	p.x[(int64_t(r) * p.hc + s) * p.hidden + d] += p.post[int64_t(r) * p.hc + s] * float(p.y[int64_t(r) * p.hidden + d]);
}

struct GrSigGatedParams { half* y; half* sh; half* x; half* w; int hidden; };

// Group per row: g = sigmoid(x . w), y += g * s
[shader("compute")]
[numthreads(256, 1, 1)]
void gr_sigmoid_gated(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrSigGatedParams p) {
	let t = int(tid.x);
	let r = int(gid.x);
	float acc = 0.0f;
	for (int i = t; i < p.hidden; i += 256) acc = fma(float(p.x[int64_t(r) * p.hidden + i]), float(p.w[i]), acc);
	let g = sigm(gsum(t, 256, acc));
	for (int i = t; i < p.hidden; i += 256) {
		let o = int64_t(r) * p.hidden + i;
		p.y[o] = half(float(p.y[o]) + g * float(p.sh[o]));
	}
}

struct GrStreamNormParams { float* x; half* w; half* out_; int hidden; int hc; float eps; };

[shader("compute")]
[numthreads(256, 1, 1)]
void gr_stream_norm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrStreamNormParams p) {
	let t = int(tid.x);
	let r = int(gid.x);
	let s = int(gid.y);
	let base = (int64_t(r) * p.hc + s) * p.hidden;
	float sq = 0.0f;
	for (int d = t; d < p.hidden; d += 256) sq += p.x[base + d] * p.x[base + d];
	let inv = rsqrt(gsum(t, 256, sq) / p.hidden + p.eps);
	for (int d = t; d < p.hidden; d += 256) p.out_[base + d] = half(p.x[base + d] * inv * (1.0f + float(p.w[int64_t(s) * p.hidden + d])));
}

struct GrBroadcastParams { half* h; half* e; float* streams; int hidden; int hc; };

[shader("compute")]
[numthreads(256, 1, 1)]
void gr_add_broadcast(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrBroadcastParams p) {
	let d = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	let s = int(gid.z);
	if (d >= p.hidden) return;
	let o = (int64_t(r) * p.hc + s) * p.hidden + d;
	p.streams[o] = float(p.h[o]) + float(p.e[int64_t(r) * p.hidden + d]);
}

struct GrGemmParams { half* a; half* w; float* out_; int m; int n; int k; int accumulate; };

// out[m][n] (+)= a[m][k] . w[n][k], fp32 accumulation. Thread (n, m)
[shader("compute")]
[numthreads(64, 1, 1)]
void gr_gemm_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrGemmParams p) {
	let c = int(gid.x) * 64 + int(tid.x);
	let r = int(gid.y);
	if (c >= p.n) return;
	half* ar = p.a + int64_t(r) * p.k;
	half* wr = p.w + int64_t(c) * p.k;
	float acc = 0.0f;
	for (int i = 0; i < p.k; ++i) acc = fma(float(ar[i]), float(wr[i]), acc);
	let o = int64_t(r) * p.n + c;
	p.out_[o] = p.accumulate != 0 ? p.out_[o] + acc : acc;
}

struct GrGateActParams { float* z; half* t_hi; half* t_lo; float* post; int ld; int rank; int hc; int has_inject; };

[shader("compute")]
[numthreads(128, 1, 1)]
void gr_gate_act(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrGateActParams p) {
	let r = int(gid.x);
	for (int j = int(tid.x); j < p.rank + (p.has_inject != 0 ? p.hc : 0); j += 128) {
		let v = p.z[int64_t(r) * p.ld + j] / p.hc;
		if (j < p.rank) {
			let tv = v / (1.0f + exp(-v));
			let hv = half(tv);
			p.t_hi[int64_t(r) * p.rank + j] = hv;
			p.t_lo[int64_t(r) * p.rank + j] = half(tv - float(hv));
		}
		else p.post[int64_t(r) * p.hc + (j - p.rank)] = 2.0f * sigm(v);
	}
}

struct GrUpEpilogueParams { float* u; float* normed; half* mixed; int hidden; int hc; };

[shader("compute")]
[numthreads(256, 1, 1)]
void gr_up_epilogue(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GrUpEpilogueParams p) {
	let d = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	if (d >= p.hidden) return;
	float outv = 0.0f;
	for (int s = 0; s < p.hc; ++s)
		outv += sigm(p.u[int64_t(r) * p.hc * p.hidden + int64_t(s) * p.hidden + d]) * p.normed[(int64_t(r) * p.hc + s) * p.hidden + d];
	p.mixed[int64_t(r) * p.hidden + d] = half(outv / p.hc);
}

// ---------------------------------------------------------------- PLE

struct PleGateParams { half* key; half* value; float* streams; half* wk; half* wq; half* wc; float* gated; half* normed; int hidden; int hc; float eps; float gate_scale; };

// Group per (row, stream)
[shader("compute")]
[numthreads(256, 1, 1)]
void ple_gate(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PleGateParams p) {
	let t = int(tid.x);
	let r = int(gid.x);
	let s = int(gid.y);
	let base = (int64_t(r) * p.hc + s) * p.hidden;
	half* k = p.key + base;
	float* x = p.streams + base;
	half* v = p.value + int64_t(r) * p.hidden;
	half* wks = p.wk + int64_t(s) * p.hidden;
	half* wqs = p.wq + int64_t(s) * p.hidden;
	half* wcs = p.wc + int64_t(s) * p.hidden;
	float sk = 0.0f;
	float sq = 0.0f;
	float sv = 0.0f;
	for (int d = t; d < p.hidden; d += 256) {
		let kv = float(k[d]);
		let vv = float(v[d]);
		sk += kv * kv;
		sq += x[d] * x[d];
		sv += vv * vv;
	}
	sk = gsum(t, 256, sk);
	sq = gsum(t, 256, sq);
	sv = gsum(t, 256, sv);
	let ik = rsqrt(sk / p.hidden + p.eps);
	let iq = rsqrt(sq / p.hidden + p.eps);
	float dot = 0.0f;
	for (int d = t; d < p.hidden; d += 256) dot += (float(k[d]) * ik * (1.0f + float(wks[d]))) * (x[d] * iq * (1.0f + float(wqs[d])));
	dot = gsum(t, 256, dot);
	let g = dot * p.gate_scale;
	let sign = g > 0.0f ? 1.0f : (g < 0.0f ? -1.0f : 0.0f);
	let sg = sigm(sign * sqrt(max(abs(g), 1e-6f)));
	let ig = rsqrt(sg * sg * sv / p.hidden + p.eps);
	for (int d = t; d < p.hidden; d += 256) {
		let gv = sg * float(v[d]);
		p.gated[base + d] = gv;
		p.normed[base + d] = half(gv * ig * (1.0f + float(wcs[d])));
	}
}

struct PleConvParams { half* normed; half* conv_w; half* ring; int* slots; int* pos; float* gated; float* streams; int channels; int dilation; };

[shader("compute")]
[numthreads(256, 1, 1)]
void ple_conv(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PleConvParams p) {
	let c = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	if (c >= p.channels) return;
	let slot = p.slots[r];
	let pos = p.pos[r];
	float acc = 0.0f;
	for (int j = 0; j < 4; ++j) {
		let k = (3 - j) * p.dilation;
		let sp = pos - k;
		float xv = 0.0f;
		if (sp >= 0) {
			let rr = r - k;
			if (k == 0 || (rr >= 0 && p.slots[rr] == slot && p.pos[rr] == sp))
				xv = float(p.normed[int64_t(k == 0 ? r : rr) * p.channels + c]);
			else
				xv = float(p.ring[(int64_t(slot) * PLE_RING + (sp % PLE_RING)) * p.channels + c]);
		}
		acc = fma(float(p.conv_w[int64_t(c) * 4 + j]), xv, acc);
	}
	let o = int64_t(r) * p.channels + c;
	p.streams[o] += p.gated[o] + acc / (1.0f + exp(-acc));
}

struct PleRingParams { half* normed; half* ring; int* slots; int* pos; int channels; };

[shader("compute")]
[numthreads(256, 1, 1)]
void ple_ring(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PleRingParams p) {
	let c = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	if (c >= p.channels) return;
	p.ring[(int64_t(p.slots[r]) * PLE_RING + (p.pos[r] % PLE_RING)) * p.channels + c] = p.normed[int64_t(r) * p.channels + c];
}

// ---------------------------------------------------------------- QSA

groupshared float s_x[QSA_DIM];

// Rotate-half rope of element d of the group's 128-vector held in s_x (the first rot_dim dims)
float rope_elem(int d, int pos, int rot_dim, float theta) {
	let x = s_x[d];
	if (d >= rot_dim) return x;
	let half_rot = rot_dim / 2;
	let i = d % half_rot;
	let ang = float(pos) * pow(theta, -(2.0f * float(i) / float(rot_dim)));
	let c = cos(ang);
	let sn = sin(ang);
	return d < half_rot ? x * c - s_x[d + half_rot] * sn : x * c + s_x[d - half_rot] * sn;
}

struct QsaPrepParams { half* qk; half* qw; half* q_out; half* ring; int* slots; int* pos; int layer; int64_t ring_slot_stride; int rot_dim; float theta; float eps; };

// Group per (row, head); head QSA_HEADS is the raw key, stored as is
[shader("compute")]
[numthreads(128, 1, 1)]
void qsa_prep(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform QsaPrepParams p) {
	let d = int(tid.x);
	let r = int(gid.x);
	let h = int(gid.y);
	half* src = p.qk + int64_t(r) * ((QSA_HEADS + 1) * QSA_DIM) + h * QSA_DIM;
	let pos = p.pos[r];
	if (h == QSA_HEADS) {
		p.ring[int64_t(p.slots[r]) * p.ring_slot_stride + (int64_t(p.layer) * QSA_RING + (pos % QSA_RING)) * QSA_DIM + d] = src[d];
		return;
	}
	let x = float(src[d]);
	let ms = gsum(d, 128, x * x) / QSA_DIM;
	// rounded to the key dtype before the rope
	s_x[d] = float(half(x * rsqrt(ms + p.eps) * (1.0f + float(p.qw[d]))));
	GroupMemoryBarrierWithGroupSync();
	p.q_out[(int64_t(r) * QSA_HEADS + h) * QSA_DIM + d] = half(rope_elem(d, pos, p.rot_dim, p.theta));
}

struct QsaPoolParams { half* ring; half* pool_plane; half* kw; int* slots; int* pos; int layer; int64_t ring_slot_stride; int64_t pool_slot_stride; int max_pools; int rot_dim; float theta; float eps; };

[shader("compute")]
[numthreads(128, 1, 1)]
void qsa_pool_update(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform QsaPoolParams p) {
	let d = int(tid.x);
	let r = int(gid.x);
	let pos = p.pos[r];
	if ((pos % QSA_POOL) != QSA_POOL - 1) return;   // uniform per group: the barriers below are not reached by any thread of it
	let slot = p.slots[r];
	let pool = pos / QSA_POOL;
	half* rk = p.ring + int64_t(slot) * p.ring_slot_stride + int64_t(p.layer) * QSA_RING * QSA_DIM;
	float acc = 0.0f;
	for (int j = 0; j < QSA_POOL; ++j) acc += float(rk[int64_t((pool * QSA_POOL + j) % QSA_RING) * QSA_DIM + d]);
	// mean in fp32, rounded to the key dtype before the norm
	let mean = float(half(acc / QSA_POOL));
	let ms = gsum(d, 128, mean * mean) / QSA_DIM;
	s_x[d] = float(half(mean * rsqrt(ms + p.eps) * (1.0f + float(p.kw[d]))));
	GroupMemoryBarrierWithGroupSync();
	p.pool_plane[int64_t(slot) * p.pool_slot_stride + (int64_t(p.layer) * p.max_pools + pool) * QSA_DIM + d] = half(rope_elem(d, pool * QSA_POOL, p.rot_dim, p.theta));
}

struct QsaScoreParams { half* q; half* pool_plane; half* scores; int* slots; int* pos; int score_stride; int layer; int max_pools; int64_t pool_slot_stride; float scale; };

groupshared float s_q4[QSA_HEADS * QSA_DIM];

// Group (64 pools, row): a thread scores one pool against the row's HEADS queries
[shader("compute")]
[numthreads(64, 1, 1)]
void qsa_score(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform QsaScoreParams p) {
	let t = int(tid.x);
	let r = int(gid.y);
	let bound = (p.pos[r] + 1) / QSA_POOL;
	if (bound <= SEL_POOLS) return;
	for (int i = t; i < QSA_HEADS * QSA_DIM; i += 64) s_q4[i] = float(p.q[int64_t(r) * QSA_HEADS * QSA_DIM + i]);
	GroupMemoryBarrierWithGroupSync();
	let pl = int(gid.x) * 64 + t;
	if (pl >= bound) return;
	half* key = p.pool_plane + int64_t(p.slots[r]) * p.pool_slot_stride + (int64_t(p.layer) * p.max_pools + pl) * QSA_DIM;
	float total = 0.0f;
	for (int h = 0; h < QSA_HEADS; ++h) {
		float s = 0.0f;
		for (int d = 0; d < QSA_DIM; ++d) s = fma(s_q4[h * QSA_DIM + d], float(key[d]), s);
		total += max(s, 0.0f);
	}
	p.scores[int64_t(r) * p.score_stride + pl] = half(total * p.scale);
}

// ---------------------------------------------------------------- DSA selection

struct SelectParams { uint16_t* scores; int* idx; int* nkeys; int* pos; int score_stride; int kpad; };

groupshared int s_hist[256];
groupshared int s_sel[SEL_POOLS];
groupshared int s_misc[4];   // b1, above1, thr, need_eq

int score_key(uint16_t b) {
    let v = int(b);
    return (v & 0x8000) != 0 ? ((~v) & 0xFFFF) : (v | 0x8000);
}

// Group per row: the SEL_POOLS best pools by fp16 score (two 8-bit radix passes), ties at the threshold taken in pool order, expanded to
// token indices followed by the raw tail. Rows with at most SEL_POOLS pools are dense (nkeys = -1)
[shader("compute")]
[numthreads(256, 1, 1)]
void dsa_select(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform SelectParams p) {
	let t = int(tid.x);
	let r = int(gid.x);
	let vis = p.pos[r] + 1;
	let bound = vis / QSA_POOL;
	if (bound <= SEL_POOLS) {
		if (t == 0) p.nkeys[r] = -1;
		return;
	}
	uint16_t* sc = p.scores + int64_t(r) * p.score_stride;
	s_hist[t] = 0;
	GroupMemoryBarrierWithGroupSync();
	for (int i = t; i < bound; i += 256) {
		int old;
		InterlockedAdd(s_hist[score_key(sc[i]) >> 8], 1, old);
	}
	GroupMemoryBarrierWithGroupSync();
	if (t == 0) {
		int cum = 0;
		int b = 255;
		for (; b > 0; --b) {
			if (cum + s_hist[b] >= SEL_POOLS) break;
			cum += s_hist[b];
		}
		s_misc[0] = b;
		s_misc[1] = cum;
	}
	GroupMemoryBarrierWithGroupSync();
	let b1 = s_misc[0];
	s_hist[t] = 0;
	GroupMemoryBarrierWithGroupSync();
	for (int i = t; i < bound; i += 256) {
		let key = score_key(sc[i]);
		if ((key >> 8) == b1) {
			int old;
			InterlockedAdd(s_hist[key & 255], 1, old);
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if (t == 0) {
		let rem = SEL_POOLS - s_misc[1];
		int cum = 0;
		int b = 255;
		for (; b > 0; --b) {
			if (cum + s_hist[b] >= rem) break;
			cum += s_hist[b];
		}
		let thr = (b1 << 8) | b;
		let need_eq = rem - cum;
		let total_gt = SEL_POOLS - need_eq;
		// compaction in pool order: strictly better pools into [0, total_gt), ties at the threshold into [total_gt, SEL_POOLS)
		int ng = 0;
		int ne = 0;
		for (int i = 0; i < bound; ++i) {
			let key = score_key(sc[i]);
			if (key > thr) { if (ng < total_gt) s_sel[ng] = i; ng++; }
			else if (key == thr) { if (ne < need_eq) s_sel[total_gt + ne] = i; ne++; }
		}
	}
	GroupMemoryBarrierWithGroupSync();
	int* o = p.idx + int64_t(r) * p.kpad;
	for (int i = t; i < SEL_POOLS * QSA_POOL; i += 256) o[i] = s_sel[i / QSA_POOL] * QSA_POOL + (i % QSA_POOL);
	let tail = vis % QSA_POOL;
	if (t < tail) o[SEL_POOLS * QSA_POOL + t] = vis - tail + t;
	if (t == 0) p.nkeys[r] = SEL_POOLS * QSA_POOL + tail;
}

// ---------------------------------------------------------------- sparse-list attention

struct SparseAttnParams {
	half* q; half* k; half* v; half* gate; half* out_;
	int* slots; int* pos; int* idx; int* nkeys;
	int64_t slot_stride;
	int layer_idx; int num_q_heads; int num_kv_heads; int head_dim; int max_seq; int idx_stride; int has_gate;
	float scale;
};

groupshared float s_aq[256];
groupshared float s_as[256];

// Group (head, row): row r attends to the nkeys[r] listed positions of its slot, or densely to 0..pos when nkeys[r] < 0. Online softmax over tiles of 256 keys
[shader("compute")]
[numthreads(256, 1, 1)]
void sparse_attn(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform SparseAttnParams p) {
	let t = int(tid.x);
	let head = int(gid.x);
	let m = int(gid.y);
	let per_kv = max(p.num_q_heads / p.num_kv_heads, 1);
	let kv_head = head / per_kv;
	let nk = p.nkeys[m];
	let total = nk >= 0 ? nk : min(p.pos[m] + 1, p.max_seq);
	let row_elems = p.num_kv_heads * p.head_dim;
	let base = int64_t(p.slots[m]) * p.slot_stride + int64_t(p.layer_idx) * p.max_seq * row_elems + kv_head * p.head_dim;
	let qo = (int64_t(m) * p.num_q_heads + head) * p.head_dim;
	let active = t < p.head_dim;
	s_aq[t] = active ? float(p.q[qo + t]) : 0.0f;
	GroupMemoryBarrierWithGroupSync();
	float m_run = -1e30f;
	float l_run = 0.0f;
	float o = 0.0f;
	for (int t0 = 0; t0 < total; t0 += 256) {
		let n = min(256, total - t0);
		float score = -1e30f;
		if (t < n) {
			let tok = nk >= 0 ? p.idx[int64_t(m) * p.idx_stride + t0 + t] : t0 + t;
			let kp = base + int64_t(tok) * row_elems;
			float dot = 0.0f;
			for (int d = 0; d < p.head_dim; ++d) dot = fma(s_aq[d], float(p.k[kp + d]), dot);
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
		let pk = t < n ? exp(score - m_new) : 0.0f;
		s_as[t] = pk;
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
			for (int kk = 0; kk < n; ++kk) {
				let tok = nk >= 0 ? p.idx[int64_t(m) * p.idx_stride + t0 + kk] : t0 + kk;
				o = fma(s_as[kk], float(p.v[base + int64_t(tok) * row_elems + t]), o);
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (active) {
		let val = l_run > 0.0f ? o / l_run : 0.0f;
		let g = p.has_gate != 0 ? sigm(float(p.gate[qo + t])) : 1.0f;
		p.out_[qo + t] = half(val * g);
	}
}

// ---------------------------------------------------------------- softmax top-k router, MoE gather / combine

struct RouterParams { half* x; half* w; int* topk_idx; half* topk_w; int hidden; int experts; int top_k; int round_fp16; };

groupshared float s_logit[512];

// Group per row: logits (an expert per thread, strided), then the top_k largest (ties to the lowest index) and the softmax over just those
[shader("compute")]
[numthreads(256, 1, 1)]
void router_softmax_topk(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RouterParams p) {
	let t = int(tid.x);
	let r = int(gid.x);
	half* xr = p.x + int64_t(r) * p.hidden;
	for (int e = t; e < p.experts; e += 256) {
		float acc = 0.0f;
		for (int d = 0; d < p.hidden; ++d) acc = fma(float(xr[d]), float(p.w[int64_t(d) * p.experts + e]), acc);
		s_logit[e] = p.round_fp16 != 0 ? float(half(acc)) : acc;
	}
	GroupMemoryBarrierWithGroupSync();
	if (t == 0) {
		float val[16];
		for (int i = 0; i < p.top_k; ++i) {
			float best = -1e30f;
			int bi = -1;
			for (int e = 0; e < p.experts; ++e) {
				if (s_logit[e] > best) { best = s_logit[e]; bi = e; }
			}
			s_logit[bi] = -3.0e38f;
			val[i] = best;
			p.topk_idx[int64_t(r) * p.top_k + i] = bi;
		}
		float sum = 0.0f;
		for (int i = 0; i < p.top_k; ++i) sum += exp(val[i] - val[0]);
		for (int i = 0; i < p.top_k; ++i) p.topk_w[int64_t(r) * p.top_k + i] = half(exp(val[i] - val[0]) / sum);
	}
}

struct MoeCombineParams { half* route_out; half* topk_w; int* topk_idx; half* out_; int hidden; int top_k; int start_expert; int end_expert; };

// Thread (c, row): out = sum over the routes into [start, end) in route order, fp32, rounded to fp16 once
[shader("compute")]
[numthreads(256, 1, 1)]
void moe_combine_routes(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform MoeCombineParams p) {
	let c = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	if (c >= p.hidden) return;
	float acc = 0.0f;
	for (int j = 0; j < p.top_k; ++j) {
		let e = p.topk_idx[int64_t(r) * p.top_k + j];
		if (e < p.start_expert || e >= p.end_expert) continue;
		acc += float(p.topk_w[int64_t(r) * p.top_k + j]) * float(p.route_out[(int64_t(r) * p.top_k + j) * p.hidden + c]);
	}
	p.out_[int64_t(r) * p.hidden + c] = half(acc);
}
