// The chunked delta rule of the KDA prefill (GDN::recurrent_step_kda_chunked on Vulkan), FP32 on the shader cores. Per head and 64-row chunk, with gamma_t
// the chunk's inclusive cumulative log decay of each key channel and S0 the state entering the chunk ([k][v]):
//   A_ts = beta_t sum_r k_tr k_sr exp( gamma_tr - gamma_sr ) (s < t), T = ( I + A )^-1
//   Delta = U - W S0, W = T diag( beta ) ( exp( gamma ) o K ), U = T diag( beta ) V
//   S = diag( exp( gamma_last ) ) S0 + Kbar^T Delta, Kbar_s = exp( gamma_last - gamma_s ) o k_s
//   Y = Qt S0 + P Delta, Qt_t = exp( gamma_t ) o q_t, P_ts = sum_r q_tr k_sr exp( gamma_tr - gamma_sr ) (s <= t)
// then each row's gated RMS norm. Every decay factor is the exponential of a difference that is at most 0. kda_chunk_prep computes A and P (each
// pair's decays exactly), T, W, U, Kbar and Qt of every chunk at once; kda_chunk_scan walks the chunks with 16 columns of the state in registers
// and keeps each chunk's S0 and Delta; kda_chunk_out computes Y of every chunk at once and kda_chunk_norm the gated norm. Rows past the end of a
// partial chunk count for nothing (k, q, v, beta and the decay are 0 there).

struct KdaChunkParams {
	half* qkv;       // [rows, 3 heads 128]: q, k, v
	float* g_decay;  // [rows, heads 128]
	float* beta;     // [rows, heads]
	half* z;         // [rows, heads 128]
	half* o_norm;    // [128]
	float* state;    // layer base; slot s at + s * slot_stride, [k][v] per head
	int* slots;      // BatchRows::slot (one slot: slots[0])
	half* output;    // [rows, heads 128]
	float* pm;       // per (chunk, head): P [64][64]
	half* qt;        // Qt [64][128]
	float* kb;       // Kbar [64][128]
	float* s0;       // Ktilde [64][128] (the prep's), then S0 [v][k] (the scan's)
	float* wy;       // W [64][128], then Y [64][128] (the output's)
	float* uu;       // U [64][128]
	float* dd;       // Delta [64][128]
	float* dl;       // exp( gamma_last ) [128]
	int heads;
	int rows;
	int has_z;
	int has_norm;
	float eps;
	int pad0;
	int64_t slot_stride;
};

static const float LOG2E = 1.4426950408889634f;
static const int QP = 36;     // pitch of the prep's 64 x 32 quarters
static const int KF = 0;      // their offsets in p_x: K, Q and log2 gamma
static const int QF = 64 * QP;
static const int GM = 2 * 64 * QP;

int tri_index(int i, int j) { return i * (i + 1) / 2 + j; }
float2 h2f(uint w) { return float2(bit_cast<half2>(w)); }

// ---- prep: one group per (head, chunk) ----
groupshared float p_x[8192];    // a quarter of K, Q and log2 gamma, then A (ten blocks) and the solve's partial products, then Ktilde or V [t][128]
groupshared float p_t[2560];    // T (ten 16 x 16 blocks)
groupshared float p_b[64];
groupshared float p_tot[256];

// W or U: the thread's 4 x 8 tile of T diag( beta ) X over the rows below its diagonal, X [t][128] in p_x
void prep_product(int tid, float* out) {
	let t0 = (tid >> 4) * 4;
	let d0 = (tid & 15) * 8;
	float acc[4][8];
	for (int r = 0; r < 4; ++r) for (int e = 0; e < 8; ++e) acc[r][e] = 0.0f;
	for (int j = 0; j <= t0 + 3; ++j) {
		let sc = p_b[j];
		float tv[4];
		for (int r = 0; r < 4; ++r) {
			let t = t0 + r;
			tv[r] = j <= t ? p_t[tri_index(t >> 4, j >> 4) * 256 + (t & 15) * 16 + (j & 15)] * sc : 0.0f;
		}
		float xv[8];
		for (int e = 0; e < 8; ++e) xv[e] = p_x[j * 128 + d0 + e];
		for (int r = 0; r < 4; ++r) for (int e = 0; e < 8; ++e) acc[r][e] = fma(tv[r], xv[e], acc[r][e]);
	}
	for (int r = 0; r < 4; ++r) {
		float* o = out + (t0 + r) * 128 + d0;
		storeAligned<16>((float4*)o, float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]));
		storeAligned<16>((float4*)(o + 4), float4(acc[r][4], acc[r][5], acc[r][6], acc[r][7]));
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void kda_chunk_prep(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uniform KdaChunkParams p) {
	let tid = int(gtid.x);
	let h = int(gid.x);
	let c = int(gid.y);
	let channels = p.heads * 128;
	let r0 = c * 64;
	let n = min(64, p.rows - r0);
	let stride = int64_t(3) * channels;
	let hc = int64_t(c) * p.heads + h;
	let warp = tid >> 5;
	let lane = tid & 31;
	let lt = tid >> 2;             // the thread's 8 elements of a quarter: row lt, columns lj .. lj + 7
	let lj = (tid & 3) * 8;
	if (tid < 64) p_b[tid] = tid < n ? p.beta[int64_t(r0 + tid) * p.heads + h] : 0.0f;

	// A and P summed over the four quarters of the key channels: item ii of the thread is rows 4 tg .. 4 tg + 3 against key s = 32 sh + lane
	// (pairs above the diagonal are computed and dropped)
	float acc_a[3][4], acc_p[3][4];
	[ForceUnroll]
	for (int ii = 0; ii < 3; ++ii) {
		[ForceUnroll]
		for (int i = 0; i < 4; ++i) {
			acc_a[ii][i] = 0.0f;
			acc_p[ii][i] = 0.0f;
		}
	}
	for (int qq = 0; qq < 4; ++qq) {
		let cq = qq * 32;
		uint4 k8 = uint4(0u, 0u, 0u, 0u);
		uint4 q8 = k8;
		float4 g0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 g1 = g0;
		if (lt < n) {
			half* row = p.qkv + int64_t(r0 + lt) * stride + h * 128 + cq + lj;
			q8 = loadAligned<16>((uint4*)row);
			k8 = loadAligned<16>((uint4*)(row + channels));
			float* gr = p.g_decay + int64_t(r0 + lt) * channels + h * 128 + cq + lj;
			g0 = loadAligned<16>((float4*)gr);
			g1 = loadAligned<16>((float4*)(gr + 4));
		}
		GroupMemoryBarrierWithGroupSync();   // the previous quarter's readers are done
		for (int u = 0; u < 4; ++u) {
			let kv = h2f(k8[u]);
			let qv = h2f(q8[u]);
			p_x[KF + lt * QP + lj + 2 * u] = kv.x;
			p_x[KF + lt * QP + lj + 2 * u + 1] = kv.y;
			p_x[QF + lt * QP + lj + 2 * u] = qv.x;
			p_x[QF + lt * QP + lj + 2 * u + 1] = qv.y;
			p_x[GM + lt * QP + lj + u] = g0[u];
			p_x[GM + lt * QP + lj + 4 + u] = g1[u];
		}
		GroupMemoryBarrierWithGroupSync();
		// gamma: thread group w sums rows 8 w .. 8 w + 7 of channel `lane`, then adds the totals of the segments before it
		{
			float a = 0.0f;
			float x[8];
			for (int i = 0; i < 8; ++i) {
				a += p_x[GM + (warp * 8 + i) * QP + lane];
				x[i] = a;
			}
			p_tot[warp * 32 + lane] = a;
			GroupMemoryBarrierWithGroupSync();
			float pre = 0.0f;
			for (int w = 0; w < warp; ++w) pre += p_tot[w * 32 + lane];
			for (int i = 0; i < 8; ++i) p_x[GM + (warp * 8 + i) * QP + lane] = (x[i] + pre) * LOG2E;
		}
		GroupMemoryBarrierWithGroupSync();
		// This quarter of Kbar, Ktilde, Qt and exp( gamma_last )
		{
			float xb[8], xt[8];
			half xq[8];
			for (int i = 0; i < 8; ++i) {
				let g = p_x[GM + lt * QP + lj + i];
				let gl = p_x[GM + 63 * QP + lj + i];
				let kv = p_x[KF + lt * QP + lj + i];
				let e = exp2(g);
				xb[i] = kv * exp2(min(gl - g, 0.0f));
				xt[i] = kv * e;
				xq[i] = half(p_x[QF + lt * QP + lj + i] * e);
			}
			let o = int64_t(lt) * 128 + cq + lj;
			float* kb = p.kb + hc * 8192 + o;
			storeAligned<16>((float4*)kb, float4(xb[0], xb[1], xb[2], xb[3]));
			storeAligned<16>((float4*)(kb + 4), float4(xb[4], xb[5], xb[6], xb[7]));
			float* kt = p.s0 + hc * 16384 + o;
			storeAligned<16>((float4*)kt, float4(xt[0], xt[1], xt[2], xt[3]));
			storeAligned<16>((float4*)(kt + 4), float4(xt[4], xt[5], xt[6], xt[7]));
			storeAligned<16>((uint4*)(p.qt + hc * 8192 + o), uint4(bit_cast<uint>(half2(xq[0], xq[1])), bit_cast<uint>(half2(xq[2], xq[3])),
				bit_cast<uint>(half2(xq[4], xq[5])), bit_cast<uint>(half2(xq[6], xq[7]))));
			if (tid < 32) p.dl[hc * 128 + cq + tid] = exp2(p_x[GM + 63 * QP + tid]);
		}
		[ForceUnroll]
		for (int ii = 0; ii < 3; ++ii) {
			let it = warp + ii * 8;
			let tg = it < 16 ? it : it - 8;
			let s = (it < 16 ? 0 : 32) + lane;
			for (int r = 0; r < 32; ++r) {
				let gs = p_x[GM + s * QP + r];
				let ks = p_x[KF + s * QP + r];
				[ForceUnroll]
				for (int i = 0; i < 4; ++i) {
					let t = tg * 4 + i;
					let e = exp2(min(p_x[GM + t * QP + r] - gs, 0.0f)) * ks;
					acc_a[ii][i] = fma(p_x[KF + t * QP + r], e, acc_a[ii][i]);
					acc_p[ii][i] = fma(p_x[QF + t * QP + r], e, acc_p[ii][i]);
				}
			}
		}
	}
	GroupMemoryBarrierWithGroupSync();

	// A (beta_t times the sum, s < t) into its blocks at p_x[0], P (s <= t) to the scratch; P's entries t < 32 <= s are zero
	{
		float* pm = p.pm + hc * 4096;
		[ForceUnroll]
		for (int ii = 0; ii < 3; ++ii) {
			let it = warp + ii * 8;
			let tg = it < 16 ? it : it - 8;
			let s = (it < 16 ? 0 : 32) + lane;
			[ForceUnroll]
			for (int i = 0; i < 4; ++i) {
				let t = tg * 4 + i;
				if ((t >> 4) >= (s >> 4)) p_x[tri_index(t >> 4, s >> 4) * 256 + (t & 15) * 16 + (s & 15)] = s < t ? p_b[t] * acc_a[ii][i] : 0.0f;
				pm[t * 64 + s] = s <= t ? acc_p[ii][i] : 0.0f;
			}
		}
		for (int x = tid; x < 32 * 32; x += 256) pm[(x >> 5) * 64 + 32 + (x & 31)] = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();

	// T = ( I + A )^-1 in 16 x 16 blocks: the diagonal blocks by forward substitution (a column a thread), then the blocks below them a
	// diagonal at a time, T_ij = -T_ii sum_{j <= k < i} A_ik T_kj (the partial products at p_x[2560])
	if (tid < 64) {
		let bb = tid >> 4;
		let col = tid & 15;
		let ab = tri_index(bb, bb) * 256;
		float x[16];
		for (int i = 0; i < 16; ++i) x[i] = 0.0f;
		for (int i = 0; i < 16; ++i) {
			float s0 = 0.0f, s1 = 0.0f;
			for (int j = 0; j < i; ++j) {
				let a = p_x[ab + i * 16 + j];
				if ((j & 1) != 0) s1 = fma(a, x[j], s1);
				else s0 = fma(a, x[j], s0);
			}
			x[i] = i < col ? 0.0f : (i == col ? 1.0f : -(s0 + s1));
		}
		for (int i = 0; i < 16; ++i) p_t[ab + i * 16 + col] = x[i];
	}
	GroupMemoryBarrierWithGroupSync();
	for (int d = 1; d < 4; ++d) {
		let nb = 4 - d;
		if (tid < nb * 256 / 4) {
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
						s0 = fma(p_x[ar + l], p_t[tc + l * 16], s0);
						s1 = fma(p_x[ar + l + 1], p_t[tc + (l + 1) * 16], s1);
					}
				}
				p_x[2560 + e] = s0 + s1;
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
					s0 = fma(p_t[tr + l], p_x[2560 + j * 256 + l * 16 + cc], s0);
					s1 = fma(p_t[tr + l + 1], p_x[2560 + j * 256 + (l + 1) * 16 + cc], s1);
				}
				p_t[tri_index(i, j) * 256 + r * 16 + cc] = -(s0 + s1);
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// W = T diag( beta ) Ktilde, then U = T diag( beta ) V. Each thread reads back the Ktilde it wrote (the group barrier orders shared memory only)
	for (int qq = 0; qq < 4; ++qq) {
		let o = lt * 128 + qq * 32 + lj;
		let a = loadAligned<16>((float4*)(p.s0 + hc * 16384 + o));
		let b = loadAligned<16>((float4*)(p.s0 + hc * 16384 + o + 4));
		for (int e = 0; e < 4; ++e) {
			p_x[o + e] = a[e];
			p_x[o + 4 + e] = b[e];
		}
	}
	GroupMemoryBarrierWithGroupSync();
	prep_product(tid, p.wy + hc * 8192);
	GroupMemoryBarrierWithGroupSync();
	for (int i = 0; i < 4; ++i) {
		let x = tid + i * 256;
		let t = x >> 4;
		let v = (x & 15) * 8;
		uint4 w = uint4(0u, 0u, 0u, 0u);
		if (t < n) w = loadAligned<16>((uint4*)(p.qkv + int64_t(r0 + t) * stride + 2 * channels + h * 128 + v));
		for (int u = 0; u < 4; ++u) {
			let f = h2f(w[u]);
			p_x[t * 128 + v + 2 * u] = f.x;
			p_x[t * 128 + v + 2 * u + 1] = f.y;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	prep_product(tid, p.uu + hc * 8192);
}

// ---- scan: one group per (head, 16 state columns), the chunks in order ----
groupshared float c_s[128 * 17];   // S0 [k][v]
groupshared float c_e[64 * 17];    // Delta [t][v]
groupshared float c_k[8192];       // Kbar [t][k]
groupshared float c_dl[128];

[shader("compute")]
[numthreads(256, 1, 1)]
void kda_chunk_scan(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uniform KdaChunkParams p) {
	let tid = int(gtid.x);
	let h = int(gid.x);
	let d0 = int(gid.y) * 16;
	let chunks = (p.rows + 63) / 64;
	float* st = p.state + int64_t(p.slots[0]) * p.slot_stride + int64_t(h) * 16384 + d0;
	// The thread's 8 state values: row sk, columns sv .. sv + 7 of the group's 16; its Delta values: row dt, columns dq .. dq + 3
	let sk = tid >> 1;
	let sv = (tid & 1) * 8;
	let dt = tid >> 2;
	let dq = (tid & 3) * 4;
	float s[8];
	{
		let a = loadAligned<16>((float4*)(st + sk * 128 + sv));
		let b = loadAligned<16>((float4*)(st + sk * 128 + sv + 4));
		s[0] = a.x; s[1] = a.y; s[2] = a.z; s[3] = a.w; s[4] = b.x; s[5] = b.y; s[6] = b.z; s[7] = b.w;
	}
	float4 rk[8];
	for (int i = 0; i < 8; ++i) rk[i] = loadAligned<16>((float4*)(p.kb + int64_t(h) * 8192 + (tid + i * 256) * 4));
	for (int c = 0; c < chunks; ++c) {
		let n = min(64, p.rows - c * 64);
		let hc = int64_t(c) * p.heads + h;
		// The state entering the chunk: shared memory (Delta's product) and the scratch ([v][k], the output kernel's); Kbar and the decay of the chunk
		for (int e = 0; e < 8; ++e) {
			c_s[sk * 17 + sv + e] = s[e];
			p.s0[hc * 16384 + int64_t(d0 + sv + e) * 128 + sk] = s[e];
		}
		for (int i = 0; i < 8; ++i) {
			let x = (tid + i * 256) * 4;
			c_k[x] = rk[i].x; c_k[x + 1] = rk[i].y; c_k[x + 2] = rk[i].z; c_k[x + 3] = rk[i].w;
		}
		if (c + 1 < chunks) {
			for (int i = 0; i < 8; ++i) rk[i] = loadAligned<16>((float4*)(p.kb + (hc + p.heads) * 8192 + (tid + i * 256) * 4));
		}
		if (tid < 128) c_dl[tid] = p.dl[hc * 128 + tid];
		GroupMemoryBarrierWithGroupSync();

		// Delta = U - W S0 for the thread's 4 values
		{
			float* w = p.wy + hc * 8192 + dt * 128;
			float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for (int k = 0; k < 128; k += 4) {
				let wv = loadAligned<16>((float4*)(w + k));
				for (int i = 0; i < 4; ++i) {
					let b = k * 17 + dq + i;
					acc[i] = fma(wv.x, c_s[b], acc[i]);
					acc[i] = fma(wv.y, c_s[b + 17], acc[i]);
					acc[i] = fma(wv.z, c_s[b + 34], acc[i]);
					acc[i] = fma(wv.w, c_s[b + 51], acc[i]);
				}
			}
			let u = loadAligned<16>((float4*)(p.uu + hc * 8192 + dt * 128 + d0 + dq));
			float4 dv;
			for (int i = 0; i < 4; ++i) dv[i] = dt < n ? u[i] - acc[i] : 0.0f;
			storeAligned<16>((float4*)(p.dd + hc * 8192 + dt * 128 + d0 + dq), dv);
			for (int i = 0; i < 4; ++i) c_e[dt * 17 + dq + i] = dv[i];
		}
		GroupMemoryBarrierWithGroupSync();

		// S = diag( exp( gamma_last ) ) S0 + Kbar^T Delta
		{
			let g = c_dl[sk];
			for (int e = 0; e < 8; ++e) s[e] *= g;
			for (int t = 0; t < 64; ++t) {
				let kv = c_k[t * 128 + sk];
				for (int e = 0; e < 8; ++e) s[e] = fma(kv, c_e[t * 17 + sv + e], s[e]);
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
	storeAligned<16>((float4*)(st + sk * 128 + sv), float4(s[0], s[1], s[2], s[3]));
	storeAligned<16>((float4*)(st + sk * 128 + sv + 4), float4(s[4], s[5], s[6], s[7]));
}

// ---- output: one group per (head, chunk, 32 v columns): Y = Qt S0 + P Delta (FP32, over W in the scratch) ----
groupshared float o_s[32 * 129];   // S0 [v][k]
groupshared float o_d[64 * 33];    // Delta [t][v]
groupshared uint o_q[64 * 65];     // Qt [t][k], two halves a word (65 words a row)

[shader("compute")]
[numthreads(256, 1, 1)]
void kda_chunk_out(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uniform KdaChunkParams p) {
	let tid = int(gtid.x);
	let h = int(gid.x);
	let c = int(gid.y);
	let v0 = int(gid.z) * 32;
	let hc = int64_t(c) * p.heads + h;
	{
		half* qp = p.qt + hc * 8192;
		for (int i = 0; i < 4; ++i) {
			let x = tid + i * 256;
			let t = x >> 4;
			let v = (x & 15) * 8;
			let w = loadAligned<16>((uint4*)(qp + t * 128 + v));
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
	float acc[8];
	for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
	for (int k = 0; k < 128; k += 2) {
		let qv = h2f(o_q[t * 65 + (k >> 1)]);
		for (int e = 0; e < 8; ++e) {
			acc[e] = fma(qv.x, o_s[(dq + e) * 129 + k], acc[e]);
			acc[e] = fma(qv.y, o_s[(dq + e) * 129 + k + 1], acc[e]);
		}
	}
	float* pr = p.pm + hc * 4096 + t * 64;
	for (int j = 0; j <= t; j += 4) {
		let pv = loadAligned<16>((float4*)(pr + j));
		for (int u = 0; u < 4; ++u) {
			for (int e = 0; e < 8; ++e) acc[e] = fma(pv[u], o_d[(j + u) * 33 + dq + e], acc[e]);
		}
	}
	float* y = p.wy + hc * 8192 + t * 128 + v0 + dq;
	storeAligned<16>((float4*)y, float4(acc[0], acc[1], acc[2], acc[3]));
	storeAligned<16>((float4*)(y + 4), float4(acc[4], acc[5], acc[6], acc[7]));
}

// ---- the gated RMS norm: one group per (head, chunk), four threads a row (thread q of row t takes columns 4 q + 16 i .. + 3) ----
groupshared float n_ss[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void kda_chunk_norm(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uniform KdaChunkParams p) {
	let tid = int(gtid.x);
	let h = int(gid.x);
	let c = int(gid.y);
	let channels = p.heads * 128;
	let r0 = c * 64;
	let n = min(64, p.rows - r0);
	let hc = int64_t(c) * p.heads + h;
	let t = tid >> 2;
	let q = tid & 3;
	float* yr = p.wy + hc * 8192 + t * 128;
	float y[32];
	float ss = 0.0f;
	for (int i = 0; i < 8; ++i) {
		let v = loadAligned<16>((float4*)(yr + q * 4 + i * 16));
		for (int e = 0; e < 4; ++e) {
			y[i * 4 + e] = v[e];
			ss = fma(v[e], v[e], ss);
		}
	}
	n_ss[tid] = ss;
	GroupMemoryBarrierWithGroupSync();
	let tot = (n_ss[t * 4] + n_ss[t * 4 + 1]) + (n_ss[t * 4 + 2] + n_ss[t * 4 + 3]);
	let inv_rms = 1.0f / sqrt(tot / 128.0f + p.eps);
	if (t < n) {
		let base = int64_t(r0 + t) * channels + h * 128;
		for (int i = 0; i < 8; ++i) {
			for (int e = 0; e < 4; ++e) {
				let col = q * 4 + i * 16 + e;
				let gamma = p.has_norm != 0 ? float(p.o_norm[col]) : 1.0f;
				let zv = p.has_z != 0 ? float(p.z[base + col]) : 0.0f;
				p.output[base + col] = half(y[i * 4 + e] * inv_rms * gamma * (1.0f / (1.0f + exp(-zv))));
			}
		}
	}
}
