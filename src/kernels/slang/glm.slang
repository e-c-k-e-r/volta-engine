// GLM-5.3 kernels for the Vulkan backend (GPU targets only; the CPU backend has plain C++ in src/backend/cpu): MLA latent products, store and attention (dense
// rows and DSA-selected rows in one kernel), the kv_b projection, the DSA indexer scoring and the KDA gate projections and recurrence. Same arithmetic as
// src/backend/cuda/modules/{mla_latent,mla,dsa,gdn}.cu and the CPU references (mla_ref.cpp, dsa_ref.cpp, gdn_ref.cpp); slow but simple.

static const int LORA = 512;
static const int LHEAD = 256;
static const int DSA_D = 128;
static const int DSA_H = 32;
static const int DSA_P = 4;
static const int DSA_SEL = 512;

groupshared float g_red[256];

float gsum256(int t, float v) {
	g_red[t] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) g_red[t] += g_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let r = g_red[0];
	GroupMemoryBarrierWithGroupSync();
	return r;
}

float gsum128(int t, float v) {
	g_red[t] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 64; s > 0; s >>= 1) {
		if (t < s) g_red[t] += g_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let r = g_red[0];
	GroupMemoryBarrierWithGroupSync();
	return r;
}

// ---------------------------------------------------------------- MLA latent

struct AbsorbParams { half* q; half* kv_b; half* q_lat; int heads; };

// q_lat[r][h][c] = sum_d q[r][h][d] * kv_b[h][d][c]. Thread (c, h, r)
[shader("compute")]
[numthreads(64, 1, 1)]
void latent_absorb_q(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform AbsorbParams p) {
	let c = int(gid.x) * 64 + int(tid.x);
	let h = int(gid.y);
	let r = int(gid.z);
	if (c >= LORA) return;
	half* qh = p.q + (int64_t(r) * p.heads + h) * LHEAD;
	half* wh = p.kv_b + int64_t(h) * 2 * LHEAD * LORA;
	float acc = 0.0f;
	for (int d = 0; d < LHEAD; ++d) acc = fma(float(qh[d]), float(wh[int64_t(d) * LORA + c]), acc);
	p.q_lat[(int64_t(r) * p.heads + h) * LORA + c] = half(acc);
}

struct ExpandParams { half* o_lat; half* kv_b; half* out_; int heads; };

// out[r][h][v] = sum_c kv_b[h][HEAD + v][c] * o_lat[r][h][c]. Thread (v, h, r)
[shader("compute")]
[numthreads(64, 1, 1)]
void latent_expand_out(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform ExpandParams p) {
	let v = int(gid.x) * 64 + int(tid.x);
	let h = int(gid.y);
	let r = int(gid.z);
	if (v >= LHEAD) return;
	half* ol = p.o_lat + (int64_t(r) * p.heads + h) * LORA;
	half* wr = p.kv_b + (int64_t(h) * 2 * LHEAD + LHEAD + v) * LORA;
	float acc = 0.0f;
	for (int c = 0; c < LORA; ++c) acc = fma(float(wr[c]), float(ol[c]), acc);
	p.out_[(int64_t(r) * p.heads + h) * LHEAD + v] = half(acc);
}

struct StoreParams { half* c; half* cache; int* slots; int* pos; int64_t slot_stride; int layer; int max_seq; };

[shader("compute")]
[numthreads(256, 1, 1)]
void latent_store(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform StoreParams p) {
	let i = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	if (i >= LORA) return;
	let off = int64_t(p.slots[r]) * p.slot_stride + (int64_t(p.layer) * p.max_seq + p.pos[r]) * LORA;
	p.cache[off + i] = p.c[int64_t(r) * LORA + i];
}

struct LatentAttnParams { half* q_lat; half* cache; half* out_; int* slots; int* pos; int* sel_idx; int* sel_n; int64_t slot_stride; int layer; int heads; int max_seq; int idx_stride; int has_sel; float scale; };

groupshared float l_q[LORA];
groupshared float l_s[256];

// Group (head, row): every head scores the one latent cache (K and V are both c). A row with a selection list attends to the listed tokens, otherwise to 0..pos
[shader("compute")]
[numthreads(256, 1, 1)]
void latent_attend(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform LatentAttnParams p) {
	let t = int(tid.x);
	let h = int(gid.x);
	let r = int(gid.y);
	let nsel = p.has_sel != 0 ? p.sel_n[r] : -1;
	let total = nsel >= 0 ? nsel : p.pos[r] + 1;
	half* kv = p.cache + int64_t(p.slots[r]) * p.slot_stride + int64_t(p.layer) * p.max_seq * LORA;
	half* qh = p.q_lat + (int64_t(r) * p.heads + h) * LORA;
	l_q[t] = float(qh[t]);
	l_q[t + 256] = float(qh[t + 256]);
	GroupMemoryBarrierWithGroupSync();
	float m_run = -1e30f;
	float l_run = 0.0f;
	float o0 = 0.0f;
	float o1 = 0.0f;
	for (int t0 = 0; t0 < total; t0 += 256) {
		let n = min(256, total - t0);
		float score = -1e30f;
		if (t < n) {
			let tok = nsel >= 0 ? p.sel_idx[int64_t(r) * p.idx_stride + t0 + t] : t0 + t;
			half* c = kv + int64_t(tok) * LORA;
			float dot = 0.0f;
			for (int d = 0; d < LORA; ++d) dot = fma(l_q[d], float(c[d]), dot);
			score = dot * p.scale;
		}
		g_red[t] = score;
		GroupMemoryBarrierWithGroupSync();
		for (int s = 128; s > 0; s >>= 1) {
			if (t < s) g_red[t] = max(g_red[t], g_red[t + s]);
			GroupMemoryBarrierWithGroupSync();
		}
		let m_new = max(m_run, g_red[0]);
		GroupMemoryBarrierWithGroupSync();
		let pk = t < n ? exp(score - m_new) : 0.0f;
		l_s[t] = pk;
		g_red[t] = pk;
		GroupMemoryBarrierWithGroupSync();
		for (int s = 128; s > 0; s >>= 1) {
			if (t < s) g_red[t] += g_red[t + s];
			GroupMemoryBarrierWithGroupSync();
		}
		let alpha = exp(m_run - m_new);
		l_run = l_run * alpha + g_red[0];
		m_run = m_new;
		o0 *= alpha;
		o1 *= alpha;
		for (int k = 0; k < n; ++k) {
			let tok = nsel >= 0 ? p.sel_idx[int64_t(r) * p.idx_stride + t0 + k] : t0 + k;
			half* c = kv + int64_t(tok) * LORA;
			o0 = fma(l_s[k], float(c[t]), o0);
			o1 = fma(l_s[k], float(c[t + 256]), o1);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	let inv = l_run > 0.0f ? 1.0f / l_run : 0.0f;
	half* oh = p.out_ + (int64_t(r) * p.heads + h) * LORA;
	oh[t] = half(o0 * inv);
	oh[t + 256] = half(o1 * inv);
}

struct DiagParams { half* q_row; half* cache; float* scores; int n; int heads; float scale; };

// scores[h][t] = q[h] . c[t] * scale. Thread (t, h)
[shader("compute")]
[numthreads(64, 1, 1)]
void latent_diag_scores(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform DiagParams p) {
	let t = int(gid.x) * 64 + int(tid.x);
	let h = int(gid.y);
	if (t >= p.n) return;
	float acc = 0.0f;
	for (int d = 0; d < LORA; ++d) acc += float(p.q_row[int64_t(h) * LORA + d]) * float(p.cache[int64_t(t) * LORA + d]);
	p.scores[int64_t(h) * p.n + t] = acc * p.scale;
}

struct KvBParams { half* x; half* w; half* k_out; half* v_out; int num_heads; int head_dim; };

// Expanded-cache path: rows of kv_b alternate K block / V block per head. Thread (row of kv_b, m)
[shader("compute")]
[numthreads(64, 1, 1)]
void mla_project_kv_b(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform KvBParams p) {
	let row = int(gid.x) * 64 + int(tid.x);
	let m = int(gid.y);
	if (row >= p.num_heads * p.head_dim * 2) return;
	float sum = 0.0f;
	for (int i = 0; i < 512; ++i) sum += float(p.w[int64_t(row) * 512 + i]) * float(p.x[int64_t(m) * 512 + i]);
	let h = row / (p.head_dim * 2);
	let rem = row % (p.head_dim * 2);
	let stride = p.num_heads * p.head_dim;
	if (rem < p.head_dim) p.k_out[int64_t(m) * stride + h * p.head_dim + rem] = half(sum);
	else p.v_out[int64_t(m) * stride + h * p.head_dim + rem - p.head_dim] = half(sum);
}

// ---------------------------------------------------------------- DSA indexer scoring

struct DsaScoreParams { half* q; half* w; half* pool_plane; half* scores; int* slots; int* pos; int score_stride; int layer; int max_seq; int64_t pool_slot_stride; float norm; };

groupshared float d_q[DSA_H * DSA_D];
groupshared float d_w[DSA_H];

// Group (64 pools, row): a thread scores one pool: relu(q[h] . key) weighted per head, two 16-head halves added
[shader("compute")]
[numthreads(64, 1, 1)]
void dsa_score(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform DsaScoreParams p) {
	let t = int(tid.x);
	let r = int(gid.y);
	let bound = (p.pos[r] + 1) / DSA_P;
	if (bound <= DSA_SEL) return;
	for (int i = t; i < DSA_H * DSA_D; i += 64) d_q[i] = float(p.q[int64_t(r) * DSA_H * DSA_D + i]);
	if (t < DSA_H) d_w[t] = float(p.w[int64_t(r) * DSA_H + t]);
	GroupMemoryBarrierWithGroupSync();
	let pl = int(gid.x) * 64 + t;
	if (pl >= bound) return;
	half* key = p.pool_plane + int64_t(p.slots[r]) * p.pool_slot_stride + int64_t(p.layer) * (p.max_seq / DSA_P) * DSA_D + int64_t(pl) * DSA_D;
	float half_acc[2] = { 0.0f, 0.0f };
	for (int h = 0; h < DSA_H; ++h) {
		float dot = 0.0f;
		for (int d = 0; d < DSA_D; ++d) dot += d_q[h * DSA_D + d] * float(key[d]);
		half_acc[h / 16] = fma(max(dot, 0.0f), d_w[h], half_acc[h / 16]);
	}
	p.scores[int64_t(r) * p.score_stride + pl] = half((half_acc[0] + half_acc[1]) * p.norm);
}

// ---------------------------------------------------------------- KDA

struct KdaSmallParams { half* x; half* b_proj; half* f_a_proj; half* g_a_proj; float* beta; half* f_a_buf; half* g_a_buf; int hidden; int has_b; int has_fa; int has_ga; };

// Group (output, row): the 16 beta gates (sigmoid of the b projection) and the 128 + 128 low-rank intermediates, dot products over the hidden size
[shader("compute")]
[numthreads(128, 1, 1)]
void kda_gates_small(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform KdaSmallParams p) {
	let t = int(tid.x);
	let o = int(gid.x);
	let r = int(gid.y);
	half* xr = p.x + int64_t(r) * p.hidden;
	half* w;
	if (o < 16) { if (p.has_b == 0) return; w = p.b_proj + int64_t(o) * p.hidden; }
	else if (o < 16 + 128) { if (p.has_fa == 0) return; w = p.f_a_proj + int64_t(o - 16) * p.hidden; }
	else { if (p.has_ga == 0) return; w = p.g_a_proj + int64_t(o - 16 - 128) * p.hidden; }
	float acc = 0.0f;
	for (int k = t; k < p.hidden; k += 128) acc = fma(float(xr[k]), float(w[k]), acc);
	let v = gsum128(t, acc);
	if (t == 0) {
		if (o < 16) p.beta[int64_t(r) * 16 + o] = 1.0f / (1.0f + exp(-v));
		else if (o < 16 + 128) p.f_a_buf[int64_t(r) * 128 + (o - 16)] = half(v);
		else p.g_a_buf[int64_t(r) * 128 + (o - 16 - 128)] = half(v);
	}
}

struct KdaBigParams { half* f_a_buf; half* g_a_buf; half* f_b_proj; half* g_b_proj; half* A_log; half* dt_bias; float* g_decay; half* z_out; int channels; int has_fb; int has_gb; int has_a; int has_dt; };

// Thread (channel, row): the decay gate from f_b (with A_log and dt_bias) and z from g_b
[shader("compute")]
[numthreads(64, 1, 1)]
void kda_gates_big(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform KdaBigParams p) {
	let j = int(gid.x) * 64 + int(tid.x);
	let r = int(gid.y);
	if (j >= p.channels) return;
	if (p.has_fb != 0) {
		half* fa = p.f_a_buf + int64_t(r) * 128;
		half* wr = p.f_b_proj + int64_t(j) * 128;
		float acc = 0.0f;
		for (int k = 0; k < 128; ++k) acc = fma(float(fa[k]), float(wr[k]), acc);
		let dt = p.has_dt != 0 ? float(p.dt_bias[j]) : 0.0f;
		let a_val = p.has_a != 0 ? float(p.A_log[j / 128]) : 0.0f;
		p.g_decay[int64_t(r) * p.channels + j] = -5.0f / (1.0f + exp(-exp(a_val) * (acc + dt)));
	}
	if (p.has_gb != 0) {
		half* ga = p.g_a_buf + int64_t(r) * 128;
		half* wr = p.g_b_proj + int64_t(j) * 128;
		float acc = 0.0f;
		for (int k = 0; k < 128; ++k) acc = fma(float(ga[k]), float(wr[k]), acc);
		p.z_out[int64_t(r) * p.channels + j] = half(acc);
	}
}

struct KdaStepParams {
	float* state; half* qkv; float* g_decay; float* beta; half* z; half* o_norm; half* out_;
	int* slots; float* snap; int* snap_on;
	int64_t slot_stride; int64_t snap_stride;
	int num_heads; int has_norm; int has_z; int has_o_norm; int decay_first; int snap_rows; int rows; float eps;
};

groupshared float k_k[128];
groupshared float k_q[128];
groupshared float k_g[128];

// Group per head, a thread per state column c; the rows run in order inside the group (rows of one slot are a causal block: each reads the state the previous
// one left). State layout [row][col] of 128 x 128. decay_first = 1 is the fused / rows form (the state is decayed, then read); 0 is the unfused form (kv_mem is
// read from the undecayed column, as gdn.cu does). With has_norm the 128 outputs are rms-normed, scaled by o_norm and gated by sigmoid(z)
[shader("compute")]
[numthreads(128, 1, 1)]
void kda_step(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform KdaStepParams p) {
	let c = int(tid.x);
	let h = int(gid.x);
	let channels = p.num_heads * 128;
	for (int m = 0; m < p.rows; ++m) {
		half* base = p.qkv + int64_t(m) * 3 * channels;
		k_q[c] = float(base[h * 128 + c]);
		k_k[c] = float(base[channels + h * 128 + c]);
		k_g[c] = exp(p.g_decay[int64_t(m) * channels + h * 128 + c]);
		GroupMemoryBarrierWithGroupSync();
		let slot = p.slots != nullptr ? p.slots[m] : 0;
		float* st = p.state + int64_t(slot) * p.slot_stride + int64_t(h) * 128 * 128;
		let b = p.beta[m * p.num_heads + h];
		float kv_mem = 0.0f;
		if (p.decay_first != 0) {
			for (int r = 0; r < 128; ++r) {
				let s = st[r * 128 + c] * k_g[r];
				st[r * 128 + c] = s;
				kv_mem = fma(s, k_k[r], kv_mem);
			}
		}
		else {
			for (int r = 0; r < 128; ++r) {
				let sp = st[r * 128 + c];
				kv_mem = fma(sp, k_k[r], kv_mem);
				st[r * 128 + c] = sp * k_g[r];
			}
		}
		let delta = (float(base[2 * channels + h * 128 + c]) - kv_mem) * b;
		float y = 0.0f;
		for (int r = 0; r < 128; ++r) {
			let s = fma(k_k[r], delta, st[r * 128 + c]);
			st[r * 128 + c] = s;
			y = fma(s, k_q[r], y);
		}
		if (p.snap != nullptr && p.snap_on[0] != 0 && m < p.snap_rows) {
			float* sp2 = p.snap + int64_t(m) * p.snap_stride + int64_t(h) * 128 * 128;
			for (int r = 0; r < 128; ++r) sp2[r * 128 + c] = st[r * 128 + c];
		}
		float outv = y;
		if (p.has_norm != 0) {
			let ss = gsum128(c, y * y);
			let inv_rms = 1.0f / sqrt(ss / 128.0f + p.eps);
			let gamma = p.has_o_norm != 0 ? float(p.o_norm[c]) : 1.0f;
			let zv = p.has_z != 0 ? float(p.z[int64_t(m) * channels + h * 128 + c]) : 0.0f;
			outv = y * inv_rms * gamma * (1.0f / (1.0f + exp(-zv)));
		}
		p.out_[int64_t(m) * channels + h * 128 + c] = half(outv);
		GroupMemoryBarrierWithGroupSync();
	}
}
