// The chunked delta rule (GDN::recurrent_step_chunked on Vulkan), FP32 on the shader cores. Per v-head and 64-row chunk, with gamma_t the
// chunk's inclusive cumulative log decay, G_t = exp( gamma_t ), S0 the state entering the chunk ([v][k]) and T = ( I + A )^-1,
// A_tj = beta_t exp( gamma_t - gamma_j ) k_t.k_j (j < t):
//   Delta = U - W S0^T, W = T diag( beta G ) K, U = T diag( beta ) V      (the rows' deltas of the recurrence)
//   S = G_last S0 + ( D Delta )^T K, D_j = exp( gamma_last - gamma_j )
//   O = G Q S0^T + P Delta, P_tj = exp( gamma_t - gamma_j ) q_t.k_j (j <= t)
// gdn_chunk_prep computes P, W, U and gamma of every chunk at once (T in 16 x 16 blocks); gdn_chunk_scan walks the chunks with 16 rows of
// the state in registers and keeps each chunk's S0 and Delta; gdn_chunk_out then computes O for every chunk at once. A partial last chunk
// reads zero-padded copies of its K and Q rows (its rows past the end count for nothing: beta, gate, k, q and v are 0 there).

struct ChunkParams {
	half* qkv;       // [rows, (2 h_k + h_v) * 128]
	float* gate;     // [rows, h_v]
	float* beta;
	float* state;    // layer base; slot s at + s * slot_stride
	int* slots;      // BatchRows::slot (one slot: slots[0])
	half* output;    // [rows, h_v * 128]
	float* wu;       // per (chunk, head): W [64][128], then U [64][128]
	float* pm;       // per (chunk, head): P [64][64]
	float* gam;      // per (chunk, head): gamma [64]
	float* s0;       // per (chunk, head): S0 [128][128]
	float* dd;       // per (chunk, head): Delta [64][128]
	half* pad;       // per K head: K [64][128], then Q [64][128], of a partial last chunk
	int h_k;
	int h_v;
	int rows;
	float scale;
	int64_t slot_stride;
};

int tri_index(int i, int j) { return i * (i + 1) / 2 + j; }

// The K / Q rows of chunk c (the padded copies for a partial chunk) and their pitch in halves
half* chunk_rows(ChunkParams p, int c, int kh, bool q) {
	let n = min(64, p.rows - c * 64);
	if (n < 64) return p.pad + int64_t(kh) * 2 * 64 * 128 + (q ? 64 * 128 : 0);
	return p.qkv + int64_t(c) * 64 * (2 * p.h_k + p.h_v) * 128 + (q ? 0 : p.h_k * 128) + kh * 128;
}
int64_t chunk_pitch(ChunkParams p, int c) { return min(64, p.rows - c * 64) < 64 ? 128 : int64_t(2 * p.h_k + p.h_v) * 128; }

float2 h2f(uint w) { return float2(bit_cast<half2>(w)); }

// ---- prep: one group per (v-head, chunk) ----
groupshared uint g_r1[4096];   // K^T [k][t] (two rows a word), then V [t][d]
groupshared uint g_r2[4096];   // Q^T [k][t], then A (FP32, ten blocks) and the solve's partial products, then K [t][d]
groupshared float g_t[2560];   // T (ten 16 x 16 blocks)
groupshared float g_g[64];
groupshared float g_b[64];
groupshared float g_bg[64];

// W or U: the thread's 4 x 8 tile of T diag( scale ) X over the rows below its diagonal, X [t][d] in g_r1 / g_r2 (32 words a row)
void prep_product(int tid, bool from_r2, bool with_g, float* out) {
	let t0 = (tid >> 4) * 4;
	let d0 = (tid & 15) * 8;
	float acc[4][8];
	for (int r = 0; r < 4; ++r) for (int e = 0; e < 8; ++e) acc[r][e] = 0.0f;
	for (int j = 0; j <= t0 + 3; ++j) {
		let sc = with_g ? g_bg[j] : g_b[j];
		float tv[4];
		for (int r = 0; r < 4; ++r) {
			let t = t0 + r;
			tv[r] = j <= t ? g_t[tri_index(t >> 4, j >> 4) * 256 + (t & 15) * 16 + (j & 15)] * sc : 0.0f;
		}
		let w = j * 64 + (d0 >> 1);
		uint4 x;
		if (from_r2) x = uint4(g_r2[w], g_r2[w + 1], g_r2[w + 2], g_r2[w + 3]);
		else x = uint4(g_r1[w], g_r1[w + 1], g_r1[w + 2], g_r1[w + 3]);
		float xv[8];
		for (int u = 0; u < 4; ++u) {
			let f = h2f(x[u]);
			xv[2 * u] = f.x;
			xv[2 * u + 1] = f.y;
		}
		for (int r = 0; r < 4; ++r) for (int e = 0; e < 8; ++e) acc[r][e] = fma(tv[r], xv[e], acc[r][e]);
	}
	for (int r = 0; r < 4; ++r) {
		float* o = out + (t0 + r) * 128 + d0;
		storeAligned<16>((float4*)o, float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]));
		storeAligned<16>((float4*)(o + 4), float4(acc[r][4], acc[r][5], acc[r][6], acc[r][7]));
	}
}

// Rows t of the chunk into [t][d] words (zero past n)
void stage_rows(int tid, half* src, int64_t pitch, int n, bool to_r2) {
	for (int i = 0; i < 4; ++i) {
		let x = tid + i * 256;
		let t = x >> 4;
		let v = (x & 15) * 8;
		uint4 w = uint4(0u, 0u, 0u, 0u);
		if (t < n) w = loadAligned<16>((uint4*)(src + int64_t(t) * pitch + v));
		let o = t * 64 + (v >> 1);
		for (int u = 0; u < 4; ++u) {
			if (to_r2) g_r2[o + u] = w[u];
			else g_r1[o + u] = w[u];
		}
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gdn_chunk_prep(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uniform ChunkParams p) {
	let tid = int(gtid.x);
	let h = int(gid.x);
	let c = int(gid.y);
	let ratio = p.h_v / p.h_k;
	let kh = h / ratio;
	let r0 = c * 64;
	let n = min(64, p.rows - r0);
	let stride = int64_t(2 * p.h_k + p.h_v) * 128;
	let hc = int64_t(c) * p.h_v + h;

	// K and Q transposed, two rows a word: word ( k, tp ) holds rows 2 tp, 2 tp + 1 of dimension k
	for (int i = 0; i < 2; ++i) {
		let x = tid + i * 256;
		let tp = x >> 4;
		let v = (x & 15) * 8;
		uint4 qa = uint4(0u, 0u, 0u, 0u), qb = qa, ka = qa, kb = qa;
		let ta = 2 * tp;
		let tb = 2 * tp + 1;
		half* base = p.qkv + int64_t(r0) * stride + kh * 128 + v;
		if (ta < n) {
			qa = loadAligned<16>((uint4*)(base + int64_t(ta) * stride));
			ka = loadAligned<16>((uint4*)(base + int64_t(ta) * stride + p.h_k * 128));
		}
		if (tb < n) {
			qb = loadAligned<16>((uint4*)(base + int64_t(tb) * stride));
			kb = loadAligned<16>((uint4*)(base + int64_t(tb) * stride + p.h_k * 128));
		}
		for (int u = 0; u < 4; ++u) {
			let qa2 = bit_cast<half2>(qa[u]);
			let qb2 = bit_cast<half2>(qb[u]);
			let ka2 = bit_cast<half2>(ka[u]);
			let kb2 = bit_cast<half2>(kb[u]);
			g_r2[(v + 2 * u) * 32 + tp] = bit_cast<uint>(half2(qa2.x, qb2.x));
			g_r2[(v + 2 * u + 1) * 32 + tp] = bit_cast<uint>(half2(qa2.y, qb2.y));
			g_r1[(v + 2 * u) * 32 + tp] = bit_cast<uint>(half2(ka2.x, kb2.x));
			g_r1[(v + 2 * u + 1) * 32 + tp] = bit_cast<uint>(half2(ka2.y, kb2.y));
		}
	}
	if (tid < 64) {
		let in_ = tid < n;
		g_g[tid] = in_ ? p.gate[int64_t(r0 + tid) * p.h_v + h] : 0.0f;
		g_b[tid] = in_ ? p.beta[int64_t(r0 + tid) * p.h_v + h] : 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();
	if (tid == 0) {
		float a = 0.0f;
		for (int t = 0; t < 64; ++t) {
			a += g_g[t];
			g_g[t] = a;
		}
	}
	// A partial last chunk: its zero-padded K and Q, once per K head
	if (n < 64 && h % ratio == 0) {
		half* pk = p.pad + int64_t(kh) * 2 * 64 * 128;
		for (int i = 0; i < 4; ++i) {
			let x = tid + i * 256;
			let t = x >> 4;
			let v = (x & 15) * 8;
			uint4 kq = uint4(0u, 0u, 0u, 0u), qq = kq;
			if (t < n) {
				half* row = p.qkv + int64_t(r0 + t) * stride + kh * 128 + v;
				qq = loadAligned<16>((uint4*)row);
				kq = loadAligned<16>((uint4*)(row + p.h_k * 128));
			}
			storeAligned<16>((uint4*)(pk + t * 128 + v), kq);
			storeAligned<16>((uint4*)(pk + 64 * 128 + t * 128 + v), qq);
		}
	}
	GroupMemoryBarrierWithGroupSync();

	// Q K^T and K K^T, a 4 x 4 tile a thread (tiles on or below the diagonal)
	let ti = tid >> 4;
	let tj = tid & 15;
	float qk[4][4], kk[4][4];
	for (int r = 0; r < 4; ++r) for (int e = 0; e < 4; ++e) { qk[r][e] = 0.0f; kk[r][e] = 0.0f; }
	if (tj <= ti) {
		for (int k = 0; k < 128; ++k) {
			let wi = k * 32 + 2 * ti;
			let wj = k * 32 + 2 * tj;
			let q01 = h2f(g_r2[wi]);
			let q23 = h2f(g_r2[wi + 1]);
			let k01 = h2f(g_r1[wi]);
			let k23 = h2f(g_r1[wi + 1]);
			let j01 = h2f(g_r1[wj]);
			let j23 = h2f(g_r1[wj + 1]);
			let qv = float4(q01.x, q01.y, q23.x, q23.y);
			let kv = float4(k01.x, k01.y, k23.x, k23.y);
			let jv = float4(j01.x, j01.y, j23.x, j23.y);
			for (int r = 0; r < 4; ++r) for (int e = 0; e < 4; ++e) {
				qk[r][e] = fma(qv[r], jv[e], qk[r][e]);
				kk[r][e] = fma(kv[r], jv[e], kk[r][e]);
			}
		}
	}
	GroupMemoryBarrierWithGroupSync();
	{
		float* pm = p.pm + hc * 4096;
		for (int r = 0; r < 4; ++r) {
			let t = ti * 4 + r;
			float4 pv;
			for (int e = 0; e < 4; ++e) {
				let j = tj * 4 + e;
				pv[e] = j <= t ? exp(g_g[t] - g_g[j]) * qk[r][e] : 0.0f;
			}
			storeAligned<16>((float4*)(pm + t * 64 + tj * 4), pv);
		}
		if (tj <= ti) {
			// A into its block (ti >> 2, tj >> 2) of g_r2 (FP32 words)
			for (int r = 0; r < 4; ++r) {
				let t = ti * 4 + r;
				for (int e = 0; e < 4; ++e) {
					let j = tj * 4 + e;
					let a = j < t ? g_b[t] * exp(g_g[t] - g_g[j]) * kk[r][e] : 0.0f;
					g_r2[tri_index(t >> 4, j >> 4) * 256 + (t & 15) * 16 + (j & 15)] = bit_cast<uint>(a);
				}
			}
		}
		if (tid < 64) {
			p.gam[hc * 64 + tid] = g_g[tid];
			g_bg[tid] = g_b[tid] * exp(g_g[tid]);
		}
	}
	GroupMemoryBarrierWithGroupSync();

	// T = ( I + A )^-1 in 16 x 16 blocks: the diagonal blocks by forward substitution (a column a thread), then the blocks below them a
	// diagonal at a time, T_ij = -T_ii sum_{j <= k < i} A_ik T_kj
	if (tid < 64) {
		let bb = tid >> 4;
		let col = tid & 15;
		let ab = tri_index(bb, bb) * 256;
		float x[16];
		for (int i = 0; i < 16; ++i) x[i] = 0.0f;
		for (int i = 0; i < 16; ++i) {
			float s0 = 0.0f, s1 = 0.0f;
			for (int j = 0; j < i; ++j) {
				let a = bit_cast<float>(g_r2[ab + i * 16 + j]);
				if ((j & 1) != 0) s1 = fma(a, x[j], s1);
				else s0 = fma(a, x[j], s0);
			}
			x[i] = i < col ? 0.0f : (i == col ? 1.0f : -(s0 + s1));
		}
		for (int i = 0; i < 16; ++i) g_t[ab + i * 16 + col] = x[i];
	}
	GroupMemoryBarrierWithGroupSync();
	for (int d = 1; d < 4; ++d) {
		let nb = 4 - d;
		if (tid < nb * 256 / 4) {
			// each thread four elements of the products M_ij = sum_k A_ik T_kj
			for (int q = 0; q < 4; ++q) {
				let e = tid * 4 + q;
				let j = e >> 8;
				let i = j + d;
				let r = (e >> 4) & 15;
				let cc = e & 15;
				float s0 = 0.0f, s1 = 0.0f;
				for (int k = j; k < i; ++k) {
					let ar = tri_index(i, k) * 256 + r * 16;
					let tc = tri_index(k, j) * 256 + cc;
					for (int l = 0; l < 16; l += 2) {
						s0 = fma(bit_cast<float>(g_r2[ar + l]), g_t[tc + l * 16], s0);
						s1 = fma(bit_cast<float>(g_r2[ar + l + 1]), g_t[tc + (l + 1) * 16], s1);
					}
				}
				g_r2[2560 + e] = bit_cast<uint>(s0 + s1);
			}
		}
		GroupMemoryBarrierWithGroupSync();
		if (tid < nb * 256 / 4) {
			for (int q = 0; q < 4; ++q) {
				let e = tid * 4 + q;
				let j = e >> 8;
				let i = j + d;
				let r = (e >> 4) & 15;
				let cc = e & 15;
				let tr = tri_index(i, i) * 256 + r * 16;
				float s0 = 0.0f, s1 = 0.0f;
				for (int l = 0; l < 16; l += 2) {
					s0 = fma(g_t[tr + l], bit_cast<float>(g_r2[2560 + j * 256 + l * 16 + cc]), s0);
					s1 = fma(g_t[tr + l + 1], bit_cast<float>(g_r2[2560 + j * 256 + (l + 1) * 16 + cc]), s1);
				}
				g_t[tri_index(i, j) * 256 + r * 16 + cc] = -(s0 + s1);
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// W = T diag( beta G ) K, U = T diag( beta ) V
	stage_rows(tid, p.qkv + int64_t(r0) * stride + (p.h_k + kh) * 128, stride, n, true);
	stage_rows(tid, p.qkv + int64_t(r0) * stride + (2 * p.h_k + h) * 128, stride, n, false);
	GroupMemoryBarrierWithGroupSync();
	float* wu = p.wu + hc * 2 * 8192;
	prep_product(tid, true, true, wu);
	prep_product(tid, false, false, wu + 8192);
}

// ---- scan: one group per (v-head, 16 state rows), the chunks in order ----
groupshared float s_s[16 * 129];   // S0 [v][k]
groupshared float s_e[64 * 17];    // D Delta [t][v]
groupshared uint s_k[4096];        // K [t][k], two halves a word
groupshared float s_dt[64];

[shader("compute")]
[numthreads(256, 1, 1)]
void gdn_chunk_scan(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uniform ChunkParams p) {
	let tid = int(gtid.x);
	let h = int(gid.x);
	let d0 = int(gid.y) * 16;
	let ratio = p.h_v / p.h_k;
	let kh = h / ratio;
	let chunks = (p.rows + 63) / 64;
	float* st = p.state + int64_t(p.slots[0]) * p.slot_stride + (int64_t(h) * 128 + d0) * 128;
	// The thread's 8 state values: row sd, columns sk .. sk + 7; its Delta values: row dt, columns dq .. dq + 3
	let sd = tid >> 4;
	let sk = (tid & 15) * 8;
	let dt = tid >> 2;
	let dq = (tid & 3) * 4;
	float s[8];
	{
		let a = loadAligned<16>((float4*)(st + sd * 128 + sk));
		let b = loadAligned<16>((float4*)(st + sd * 128 + sk + 4));
		s[0] = a.x; s[1] = a.y; s[2] = a.z; s[3] = a.w; s[4] = b.x; s[5] = b.y; s[6] = b.z; s[7] = b.w;
	}
	uint4 rk[4];
	for (int i = 0; i < 4; ++i) {
		let x = tid + i * 256;
		rk[i] = loadAligned<16>((uint4*)(chunk_rows(p, 0, kh, false) + (x >> 4) * chunk_pitch(p, 0) + (x & 15) * 8));
	}
	for (int c = 0; c < chunks; ++c) {
		let n = min(64, p.rows - c * 64);
		let hc = int64_t(c) * p.h_v + h;
		// The state entering the chunk: shared memory (Delta's product) and the scratch (the output kernel's); K of the chunk
		for (int e = 0; e < 8; ++e) s_s[sd * 129 + sk + e] = s[e];
		{
			float* so = p.s0 + hc * 16384 + (d0 + sd) * 128 + sk;
			storeAligned<16>((float4*)so, float4(s[0], s[1], s[2], s[3]));
			storeAligned<16>((float4*)(so + 4), float4(s[4], s[5], s[6], s[7]));
		}
		for (int i = 0; i < 4; ++i) {
			let x = tid + i * 256;
			let o = (x >> 4) * 64 + (x & 15) * 4;
			s_k[o] = rk[i].x; s_k[o + 1] = rk[i].y; s_k[o + 2] = rk[i].z; s_k[o + 3] = rk[i].w;
		}
		if (c + 1 < chunks) {
			for (int i = 0; i < 4; ++i) {
				let x = tid + i * 256;
				rk[i] = loadAligned<16>((uint4*)(chunk_rows(p, c + 1, kh, false) + (x >> 4) * chunk_pitch(p, c + 1) + (x & 15) * 8));
			}
		}
		if (tid < 64) {
			float* g = p.gam + hc * 64;
			s_dt[tid] = exp(g[63] - g[tid]);
		}
		GroupMemoryBarrierWithGroupSync();

		// Delta = U - W S0^T for the thread's 4 values
		{
			float* w = p.wu + hc * 2 * 8192 + dt * 128;
			float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for (int k = 0; k < 128; k += 4) {
				let wv = loadAligned<16>((float4*)(w + k));
				for (int i = 0; i < 4; ++i) {
					let b = (dq + i) * 129 + k;
					acc[i] = fma(wv.x, s_s[b], acc[i]);
					acc[i] = fma(wv.y, s_s[b + 1], acc[i]);
					acc[i] = fma(wv.z, s_s[b + 2], acc[i]);
					acc[i] = fma(wv.w, s_s[b + 3], acc[i]);
				}
			}
			let u = loadAligned<16>((float4*)(w + 8192 + d0 + dq));
			float4 dv;
			for (int i = 0; i < 4; ++i) dv[i] = dt < n ? u[i] - acc[i] : 0.0f;
			storeAligned<16>((float4*)(p.dd + hc * 8192 + dt * 128 + d0 + dq), dv);
			for (int i = 0; i < 4; ++i) s_e[dt * 17 + dq + i] = dv[i] * s_dt[dt];
		}
		GroupMemoryBarrierWithGroupSync();

		// S = G_last S0 + ( D Delta )^T K
		{
			let gl = exp(p.gam[hc * 64 + 63]);
			for (int e = 0; e < 8; ++e) s[e] *= gl;
			for (int t = 0; t < 64; ++t) {
				let ev = s_e[t * 17 + sd];
				let o = t * 64 + (sk >> 1);
				for (int u = 0; u < 4; ++u) {
					let kf = h2f(s_k[o + u]);
					s[2 * u] = fma(ev, kf.x, s[2 * u]);
					s[2 * u + 1] = fma(ev, kf.y, s[2 * u + 1]);
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
	storeAligned<16>((float4*)(st + sd * 128 + sk), float4(s[0], s[1], s[2], s[3]));
	storeAligned<16>((float4*)(st + sd * 128 + sk + 4), float4(s[4], s[5], s[6], s[7]));
}

// ---- output: one group per (v-head, chunk, 32 v columns): O = G Q S0^T + P Delta ----
groupshared float o_s[32 * 129];   // S0 [v][k]
groupshared float o_d[64 * 33];    // Delta [t][v]
groupshared uint o_q[64 * 65];     // Q [t][k], two halves a word (65 words a row)

[shader("compute")]
[numthreads(256, 1, 1)]
void gdn_chunk_out(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uniform ChunkParams p) {
	let tid = int(gtid.x);
	let h = int(gid.x);
	let c = int(gid.y);
	let v0 = int(gid.z) * 32;
	let ratio = p.h_v / p.h_k;
	let kh = h / ratio;
	let r0 = c * 64;
	let n = min(64, p.rows - r0);
	let hc = int64_t(c) * p.h_v + h;
	{
		half* qp = chunk_rows(p, c, kh, true);
		let ld = chunk_pitch(p, c);
		for (int i = 0; i < 4; ++i) {
			let x = tid + i * 256;
			let t = x >> 4;
			let v = (x & 15) * 8;
			let w = loadAligned<16>((uint4*)(qp + t * ld + v));
			let o = t * 65 + (v >> 1);
			o_q[o] = w.x; o_q[o + 1] = w.y; o_q[o + 2] = w.z; o_q[o + 3] = w.w;
		}
		float* sp = p.s0 + hc * 16384 + int64_t(v0) * 128;
		for (int i = 0; i < 4; ++i) {
			let x = tid + i * 256;
			let r = x >> 5;
			let k = (x & 31) * 4;
			let w = loadAligned<16>((float4*)(sp + r * 128 + k));
			for (int e = 0; e < 4; ++e) o_s[r * 129 + k + e] = w[e];
		}
		float* dp = p.dd + hc * 8192 + v0;
		for (int i = 0; i < 2; ++i) {
			let x = tid + i * 256;
			let t = x >> 3;
			let d = (x & 7) * 4;
			let w = loadAligned<16>((float4*)(dp + t * 128 + d));
			for (int e = 0; e < 4; ++e) o_d[t * 33 + d + e] = w[e];
		}
	}
	GroupMemoryBarrierWithGroupSync();
	let t = tid >> 2;
	let dq = (tid & 3) * 8;
	float aq[8], ap[8];
	for (int e = 0; e < 8; ++e) { aq[e] = 0.0f; ap[e] = 0.0f; }
	for (int k = 0; k < 128; k += 2) {
		let qv = h2f(o_q[t * 65 + (k >> 1)]);
		for (int e = 0; e < 8; ++e) {
			aq[e] = fma(qv.x, o_s[(dq + e) * 129 + k], aq[e]);
			aq[e] = fma(qv.y, o_s[(dq + e) * 129 + k + 1], aq[e]);
		}
	}
	float* pr = p.pm + hc * 4096 + t * 64;
	for (int j = 0; j <= t; j += 4) {
		let pv = loadAligned<16>((float4*)(pr + j));
		for (int u = 0; u < 4; ++u) {
			for (int e = 0; e < 8; ++e) ap[e] = fma(pv[u], o_d[(j + u) * 33 + dq + e], ap[e]);
		}
	}
	if (t < n) {
		let gt = exp(p.gam[hc * 64 + t]);
		half o[8];
		for (int e = 0; e < 8; ++e) o[e] = half((gt * aq[e] + ap[e]) * p.scale);
		let w = uint4(bit_cast<uint>(half2(o[0], o[1])), bit_cast<uint>(half2(o[2], o[3])), bit_cast<uint>(half2(o[4], o[5])), bit_cast<uint>(half2(o[6], o[7])));
		storeAligned<16>((uint4*)(p.output + int64_t(r0 + t) * p.h_v * 128 + h * 128 + v0 + dq), w);
	}
}
