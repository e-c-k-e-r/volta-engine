// Tiled attention over the paged KV cache for the Vulkan backend (the semantics of ref_attention.slang: row m reads slot slots[m] and attends
// to positions first..pos[m] of it, or below key_end[m] for an image row; a window, a ring cache, any KV format of kv_codec.slang, a sigmoid
// gate). Cache layout [slot][layer][max_seq][num_kv_heads * head_dim]; q, gate and out are [rows][num_q_heads][head_dim]; head_dim a multiple of
// 16, at most 512. Without slots (null) the rows are dense: K / V are [max_seq][num_kv_heads * head_dim] and every row attends to all of them
// (bidirectional attention outside a cache: the TTS backbone, vision towers).
//
// fa_attn: a workgroup of 128 per (group of group_rows rows of one slot, KV head, split of the key range). Its query vectors are the group's
// rows times the query heads of the KV head (at most 16), held in shared memory (scaled); the keys go by in tiles of bc, each tile's K staged in
// shared memory once (rows padded by one word: the lanes of a warp read distinct keys from distinct banks) and scored a query vector per warp, a
// key per lane (half2 products accumulated 16 at a time in FP16, then in FP32); the softmax is the running one per query vector; V is read from the cache a pair of dimensions per thread (coalesced) into
// the accumulators of every query vector. One split writes the output; several write their unnormalized output with its running max and sum
// to part, and fa_reduce combines them (the Q4 values' inverse Hadamard rotation and the gate there too).
#include "kv_codec.slang"

struct FaParams {
	half* q;
	half* k;          // cache base (slot 0)
	half* v;
	half* gate;       // [rows][num_q_heads][head_dim] pre-sigmoid, may be null
	half* out_;
	int* slots;
	int* pos;
	int64_t slot_stride;
	int layer_idx;
	int num_q_heads;
	int num_kv_heads;
	int head_dim;
	int max_seq;
	int has_gate;
	float scale;
	uint8_t* ks;
	uint8_t* vs;
	int kq;
	int vq;
	int window;
	int ring;
	int* key_end;
	int* bidir;
	int rows;         // rows of the call
	int group_rows;   // rows per workgroup (all in one slot); group_rows * (num_q_heads / num_kv_heads) <= 16
	int splits;       // workgroups over the key range of a group (grid z), each a share of whole tiles
	int bc;           // keys per tile: 16 or 32
	int direct;       // one split: the output itself (normalized, gated when has_gate); else the split's share into part
	float* part;      // [split][rows][num_q_heads][head_dim] (the output unnormalized), then [split][rows][num_q_heads][2] (the running max and sum)
	int parts;        // fa_decode: workgroups per row group (grid x = groups * parts), each a run of at most its RM query vectors
};

static const int FA_THREADS = 128;
static const int FA_QV = 16;
static const int FA_POOL_WORDS = 10496;   // half pairs: q [R][head_dim / 2], then a K tile [bc][head_dim / 2 + 1] (head_dim 512: R 16 with bc 16, R 8 with bc 32)

groupshared half2 s_pool[FA_POOL_WORDS];
groupshared float s_p[FA_QV * 32];
groupshared float s_m[FA_QV];
groupshared float s_l[FA_QV];
groupshared float s_alpha[FA_QV];
groupshared int s_first[FA_QV];
groupshared int s_end[FA_QV];
groupshared int s_range[2];

float fa_k(FaParams p, int64_t i) {
	return p.kq == 0 ? float(p.k[i]) : kv_load((uint8_t*)p.k, p.ks, p.kq, i);
}

float fa_v(FaParams p, int64_t i) {
	return p.vq == 0 ? float(p.v[i]) : kv_load((uint8_t*)p.v, p.vs, p.vq, i);
}

half2 fa_pack(float a, float b) {
	return half2(half(a), half(b));
}

// The 8 halves at element e (a multiple of 8) as one 16-byte load: a plain uint4 pointer load is only 4-byte aligned in the SPIR-V
uint4 fa_ld16(half* base, int64_t e) {
	return loadAligned<16>((uint4*)(base + e));
}

// Keys kt .. kt + nk - 1 of the K (is_k) or V cache into the pool at `at`, a row of head_dim / 2 words per key, rows head_dim / 2 + 1 apart,
// zeros past nk. FP16: 16-byte loads, four in flight per thread before their stores; a quantized format through kv_load
void fa_stage(FaParams p, bool is_k, int at, int kt, int nk, int64_t base, int row_elems, int t) {
	let hd = p.head_dim;
	let kstride = hd / 2 + 1;
	let fmt = is_k ? p.kq : p.vq;
	if (fmt == 0) {
		let vpr = hd / 8;   // 16-byte vectors per row
		let nvec = p.bc * vpr;
		half* src = is_k ? p.k : p.v;
		for (int i0 = t; i0 < nvec; i0 += FA_THREADS * 4) {
			uint4 buf[4];
			[ForceUnroll]
			for (int u = 0; u < 4; ++u) {
				let i = i0 + u * FA_THREADS;
				let key = i / vpr;
				buf[u] = uint4(0u, 0u, 0u, 0u);
				if (i < nvec && key < nk) {
					let ps = kt + key;
					let off = base + int64_t(p.ring > 0 ? ps % p.ring : ps) * row_elems + (i % vpr) * 8;
					buf[u] = fa_ld16(src, off);
				}
			}
			[ForceUnroll]
			for (int u = 0; u < 4; ++u) {
				let i = i0 + u * FA_THREADS;
				if (i < nvec) {
					let w = at + (i / vpr) * kstride + (i % vpr) * 4;
					s_pool[w + 0] = bit_cast<half2>(buf[u].x);
					s_pool[w + 1] = bit_cast<half2>(buf[u].y);
					s_pool[w + 2] = bit_cast<half2>(buf[u].z);
					s_pool[w + 3] = bit_cast<half2>(buf[u].w);
				}
			}
		}
		return;
	}
	let hw = hd / 2;
	for (int i = t; i < p.bc * hw; i += FA_THREADS) {
		let key = i / hw;
		let w = i % hw;
		half2 packed = half2(half(0.0f), half(0.0f));
		if (key < nk) {
			let ps = kt + key;
			let kp = base + int64_t(p.ring > 0 ? ps % p.ring : ps) * row_elems + 2 * w;
			packed = is_k ? fa_pack(fa_k(p, kp), fa_k(p, kp + 1)) : fa_pack(fa_v(p, kp), fa_v(p, kp + 1));
		}
		s_pool[at + key * kstride + w] = packed;
	}
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fa_attn(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform FaParams p) {
	let t = int(tid.x);
	let kv_head = int(gid.y);
	let split = int(gid.z);
	let G = p.num_q_heads / p.num_kv_heads;
	let r0 = int(gid.x) * p.group_rows;
	let nr = min(p.group_rows, p.rows - r0);
	let R = nr * G;
	let hd = p.head_dim;
	let hw = hd / 2;           // words per q / K row
	let kstride = hw + 1;      // words per staged K row
	let row_elems = p.num_kv_heads * hd;

	// each row's key range, and the group's (no slots: dense rows, every one against all max_seq keys)
	if (t < nr) {
		if (p.slots == nullptr) {
			s_end[t] = p.max_seq;
			s_first[t] = 0;
		}
		else {
			let r = r0 + t;
			let ps = p.pos[r];
			let end = (p.window > 0 && p.bidir != nullptr && p.bidir[0] != 0) ? p.key_end[r] : ps + 1;
			s_end[t] = p.ring > 0 ? end : min(end, p.max_seq);
			s_first[t] = p.window > 0 ? max(0, ps + 1 - p.window) : 0;
		}
	}
	if (t < FA_QV) {
		s_m[t] = -1e30f;
		s_l[t] = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();
	if (t == 0) {
		int lo = s_first[0];
		int hi = s_end[0];
		for (int i = 1; i < nr; ++i) {
			lo = min(lo, s_first[i]);
			hi = max(hi, s_end[i]);
		}
		// the group's keys shared out over the splits, whole tiles each (the range is the rows', known here and not at launch)
		let share = ((hi - lo + p.splits - 1) / p.splits + p.bc - 1) / p.bc * p.bc;
		let k_lo = lo + split * share;
		s_range[0] = k_lo;
		s_range[1] = min(hi, k_lo + share);
	}
	GroupMemoryBarrierWithGroupSync();
	let k_lo = s_range[0];
	let k_hi = s_range[1];
	let base = (p.slots != nullptr ? int64_t(p.slots[r0]) * p.slot_stride : 0) + int64_t(p.layer_idx) * p.max_seq * row_elems + kv_head * hd;

	// the query vectors, scaled: vector qv = row qv / G, query head kv_head * G + qv % G
	for (int i = t; i < R * hw; i += FA_THREADS) {
		let qv = i / hw;
		let w = i % hw;
		let qo = (int64_t(r0 + qv / G) * p.num_q_heads + kv_head * G + qv % G) * hd + 2 * w;
		s_pool[qv * hw + w] = fa_pack(float(p.q[qo]) * p.scale, float(p.q[qo + 1]) * p.scale);
	}
	let kbase = R * hw;   // the K tile after the query vectors (the host keeps R * hw + bc * kstride within the pool)

	// output pairs (2 d, 2 d + 1) of the thread: d = t + 128 j
	float o[FA_QV][4];
	[ForceUnroll]
	for (int a = 0; a < FA_QV; ++a) {
		[ForceUnroll]
		for (int c = 0; c < 4; ++c) o[a][c] = 0.0f;
	}

	for (int kt = k_lo; kt < k_hi; kt += p.bc) {
		let nk = min(p.bc, k_hi - kt);
		fa_stage(p, true, kbase, kt, nk, base, row_elems, t);
		GroupMemoryBarrierWithGroupSync();
		// scores: pair i is query vector i / bc and key i % bc (a warp: one query vector over 32 keys, or two over 16)
		for (int i = t; i < R * p.bc; i += FA_THREADS) {
			let qv = i / p.bc;
			let key = i % p.bc;
			let rr = qv / G;
			let ps = kt + key;
			float s = -1e30f;
			if (key < nk && ps >= s_first[rr] && ps < s_end[rr]) {
				// half2 products, 16 at a time in FP16 (one conversion per 8 pairs), summed in FP32
				float acc = 0.0f;
				let qa = qv * hw;
				let ka = kbase + key * kstride;
				for (int w0 = 0; w0 < hw; w0 += 8) {
					half2 a2 = half2(half(0.0f), half(0.0f));
					[ForceUnroll]
					for (int u = 0; u < 8; ++u) a2 += s_pool[qa + w0 + u] * s_pool[ka + w0 + u];
					acc += float(a2.x) + float(a2.y);
				}
				s = acc;
			}
			s_p[qv * p.bc + key] = s;
		}
		GroupMemoryBarrierWithGroupSync();
		// the running softmax of each query vector
		if (t < R) {
			let m_old = s_m[t];
			float mx = m_old;
			for (int key = 0; key < nk; ++key) mx = max(mx, s_p[t * p.bc + key]);
			float sum = 0.0f;
			for (int key = 0; key < p.bc; ++key) {
				let s = s_p[t * p.bc + key];
				let e = (key < nk && s > -1e29f) ? exp(s - mx) : 0.0f;
				s_p[t * p.bc + key] = e;
				sum += e;
			}
			let alpha = exp(m_old - mx);
			s_alpha[t] = alpha;
			s_l[t] = s_l[t] * alpha + sum;
			s_m[t] = mx;
		}
		// the V tile where the K tile was (the scores are in s_p)
		fa_stage(p, false, kbase, kt, nk, base, row_elems, t);
		GroupMemoryBarrierWithGroupSync();
		// V into the accumulators
		[ForceUnroll]
		for (int j = 0; j < 2; ++j) {
			let w = t + FA_THREADS * j;
			if (2 * w < hd) {
				[ForceUnroll]
				for (int a = 0; a < FA_QV; ++a) {
					if (a < R) {
						o[a][2 * j] *= s_alpha[a];
						o[a][2 * j + 1] *= s_alpha[a];
					}
				}
				for (int key = 0; key < nk; ++key) {
					let vw = s_pool[kbase + key * kstride + w];
					let v0 = float(vw.x);
					let v1 = float(vw.y);
					[ForceUnroll]
					for (int a = 0; a < FA_QV; ++a) {
						if (a < R) {
							let pk = s_p[a * p.bc + key];
							o[a][2 * j] = fma(pk, v0, o[a][2 * j]);
							o[a][2 * j + 1] = fma(pk, v1, o[a][2 * j + 1]);
						}
					}
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();   // the next tile rewrites the K tile and the probabilities
	}

	// the output (one split), or this split's share
	[ForceUnroll]
	for (int j = 0; j < 2; ++j) {
		let d = 2 * (t + FA_THREADS * j);
		if (d < hd) {
			[ForceUnroll]
			for (int a = 0; a < FA_QV; ++a) {
				if (a < R) {
					let oo = (int64_t(r0 + a / G) * p.num_q_heads + kv_head * G + a % G);
					if (p.direct != 0) {
						let l = s_l[a];
						let inv = l > 0.0f ? 1.0f / l : 0.0f;
						let gi = oo * hd + d;
						let g0 = p.has_gate != 0 ? 1.0f / (1.0f + exp(-float(p.gate[gi]))) : 1.0f;
						let g1 = p.has_gate != 0 ? 1.0f / (1.0f + exp(-float(p.gate[gi + 1]))) : 1.0f;
						p.out_[gi] = half(o[a][2 * j] * inv * g0);
						p.out_[gi + 1] = half(o[a][2 * j + 1] * inv * g1);
					}
					else {
						let po = ((int64_t(split) * p.rows * p.num_q_heads) + oo) * hd + d;
						p.part[po] = o[a][2 * j];
						p.part[po + 1] = o[a][2 * j + 1];
					}
				}
			}
		}
	}
	if (p.direct == 0 && t < R) {
		let oo = (int64_t(r0 + t / G) * p.num_q_heads + kv_head * G + t % G);
		let q_all = int64_t(p.rows) * p.num_q_heads;
		let po = int64_t(p.splits) * q_all * hd + ((int64_t(split) * q_all) + oo) * 2;
		p.part[po] = s_m[t];
		p.part[po + 1] = s_l[t];
	}
}

// fa_decode: fa_attn's grid, partials and output for an FP16 cache and few query vectors (a decode or a short verify), with no tiles: a lane
// holds 8 * NV dimensions of each of the workgroup's query vectors (vectors v0 .. v0 + R - 1 of its row group, R <= RM; head_dim 256 * NV) and
// of their accumulators; each warp streams its share of the split's keys from the cache, 4 / NV at a time (their K and V rows loaded together,
// 16 bytes a lane per 256 dimensions), every key's dot products summed across the warp (half2 products in FP16 per lane, the lanes in FP32), the
// softmax the running one per vector; the four warps' maxima, sums and accumulators are merged in shared memory at the end. A split without
// keys writes only its (empty) sum.
groupshared float s_wm[4 * 8];
groupshared float s_wl[4 * 8];

void fa_decode_body<let NV : int, let RM : int>(uint3 gid, int t, FaParams p) {
	static const int DC = 4 / NV;   // keys per warp step
	let lane = t & 31;
	let warp = t >> 5;
	let kv_head = int(gid.y);
	let split = int(gid.z);
	let G = p.num_q_heads / p.num_kv_heads;
	let grp = int(gid.x) / p.parts;
	let v0 = (int(gid.x) % p.parts) * RM;
	let r0 = grp * p.group_rows;
	let nr = min(p.group_rows, p.rows - r0);
	let R = min(RM, nr * G - v0);
	let hd = p.head_dim;
	let row_elems = p.num_kv_heads * hd;
	let q_all = int64_t(p.rows) * p.num_q_heads;

	// each row's key range, and the group's (no slots: dense rows, every one against all max_seq keys)
	if (t < nr) {
		if (p.slots == nullptr) {
			s_end[t] = p.max_seq;
			s_first[t] = 0;
		}
		else {
			let r = r0 + t;
			let ps = p.pos[r];
			let end = (p.window > 0 && p.bidir != nullptr && p.bidir[0] != 0) ? p.key_end[r] : ps + 1;
			s_end[t] = p.ring > 0 ? end : min(end, p.max_seq);
			s_first[t] = p.window > 0 ? max(0, ps + 1 - p.window) : 0;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if (t == 0) {
		int lo = s_first[0];
		int hi = s_end[0];
		for (int i = 1; i < nr; ++i) {
			lo = min(lo, s_first[i]);
			hi = max(hi, s_end[i]);
		}
		let share = ((hi - lo + p.splits - 1) / p.splits + 4 * DC - 1) / (4 * DC) * (4 * DC);
		let k_lo = lo + split * share;
		s_range[0] = k_lo;
		s_range[1] = min(hi, k_lo + share);
	}
	GroupMemoryBarrierWithGroupSync();
	let k_lo = s_range[0];
	let k_hi = s_range[1];
	if (k_lo >= k_hi) {
		// no keys: an empty sum (fa_reduce skips the split)
		if (p.direct == 0 && t < R) {
			let qv = v0 + t;
			let oo = int64_t(r0 + qv / G) * p.num_q_heads + kv_head * G + qv % G;
			let po = int64_t(p.splits) * q_all * hd + (int64_t(split) * q_all + oo) * 2;
			p.part[po] = -1e30f;
			p.part[po + 1] = 0.0f;
		}
		return;
	}
	let base = (p.slots != nullptr ? int64_t(p.slots[r0]) * p.slot_stride : 0) + int64_t(p.layer_idx) * p.max_seq * row_elems + kv_head * hd;

	// the lane's dimensions of the query vectors, scaled: d = 256 u + 8 lane + j
	half2 q[RM][4 * NV];
	int first[RM], end[RM];
	[ForceUnroll]
	for (int a = 0; a < RM; ++a) {
		let qv = v0 + min(a, R - 1);
		first[a] = s_first[qv / G];
		end[a] = s_end[qv / G];
		let qo = (int64_t(r0 + qv / G) * p.num_q_heads + kv_head * G + qv % G) * hd;
		[ForceUnroll]
		for (int u = 0; u < NV; ++u) {
			let w = fa_ld16(p.q, qo + u * 256 + lane * 8);
			let x = uint4(w.x, w.y, w.z, w.w);
			[ForceUnroll]
			for (int c = 0; c < 4; ++c) {
				let h = bit_cast<half2>(x[c]);
				q[a][u * 4 + c] = fa_pack(float(h.x) * p.scale, float(h.y) * p.scale);
			}
		}
	}
	float m[RM], l[RM], o[RM][8 * NV];
	[ForceUnroll]
	for (int a = 0; a < RM; ++a) {
		m[a] = -1e30f;
		l[a] = 0.0f;
		[ForceUnroll]
		for (int d = 0; d < 8 * NV; ++d) o[a][d] = 0.0f;
	}

	// the warp's keys: steps of DC keys, the warps taking turns
	for (int kc = k_lo + warp * DC; kc < k_hi; kc += 4 * DC) {
		uint4 kr[DC][NV], vr[DC][NV];
		[ForceUnroll]
		for (int c = 0; c < DC; ++c) {
			let ps = kc + c;
			let off = base + int64_t(p.ring > 0 ? ps % p.ring : ps) * row_elems + lane * 8;
			[ForceUnroll]
			for (int u = 0; u < NV; ++u) {
				kr[c][u] = ps < k_hi ? fa_ld16(p.k, off + u * 256) : uint4(0u, 0u, 0u, 0u);
				vr[c][u] = ps < k_hi ? fa_ld16(p.v, off + u * 256) : uint4(0u, 0u, 0u, 0u);
			}
		}
		[ForceUnroll]
		for (int a = 0; a < RM; ++a) {
			if (a < R) {
				float s[DC];
				float mx = m[a];
				[ForceUnroll]
				for (int c = 0; c < DC; ++c) {
					half2 acc = half2(half(0.0f), half(0.0f));
					[ForceUnroll]
					for (int u = 0; u < NV; ++u) {
						let x = uint4(kr[c][u].x, kr[c][u].y, kr[c][u].z, kr[c][u].w);
						[ForceUnroll]
						for (int w = 0; w < 4; ++w) acc += q[a][u * 4 + w] * bit_cast<half2>(x[w]);
					}
					let ps = kc + c;
					let dot = WaveActiveSum(float(acc.x) + float(acc.y));
					s[c] = (ps < k_hi && ps >= first[a] && ps < end[a]) ? dot : -1e30f;
					mx = max(mx, s[c]);
				}
				let alpha = exp(m[a] - mx);
				float pc[DC];
				float sum = 0.0f;
				[ForceUnroll]
				for (int c = 0; c < DC; ++c) {
					pc[c] = s[c] > -1e29f ? exp(s[c] - mx) : 0.0f;
					sum += pc[c];
				}
				l[a] = l[a] * alpha + sum;
				m[a] = mx;
				[ForceUnroll]
				for (int u = 0; u < NV; ++u) {
					[ForceUnroll]
					for (int w = 0; w < 4; ++w) {
						float a0 = o[a][u * 8 + 2 * w] * alpha;
						float a1 = o[a][u * 8 + 2 * w + 1] * alpha;
						[ForceUnroll]
						for (int c = 0; c < DC; ++c) {
							let x = uint4(vr[c][u].x, vr[c][u].y, vr[c][u].z, vr[c][u].w);
							let h = bit_cast<half2>(x[w]);
							a0 = fma(pc[c], float(h.x), a0);
							a1 = fma(pc[c], float(h.y), a1);
						}
						o[a][u * 8 + 2 * w] = a0;
						o[a][u * 8 + 2 * w + 1] = a1;
					}
				}
			}
		}
	}

	// the warps merged: their maxima and sums, then each warp's accumulators scaled to the common maximum (as float bits in the pool)
	if (lane == 0) {
		[ForceUnroll]
		for (int a = 0; a < RM; ++a) {
			if (a < R) {
				s_wm[warp * 8 + a] = m[a];
				s_wl[warp * 8 + a] = l[a];
			}
		}
	}
	GroupMemoryBarrierWithGroupSync();
	[ForceUnroll]
	for (int a = 0; a < RM; ++a) {
		if (a < R) {
			float M = s_wm[a];
			[ForceUnroll]
			for (int w = 1; w < 4; ++w) M = max(M, s_wm[w * 8 + a]);
			let f = m[a] > -1e29f ? exp(m[a] - M) : 0.0f;
			[ForceUnroll]
			for (int u = 0; u < NV; ++u) {
				[ForceUnroll]
				for (int j = 0; j < 8; ++j) s_pool[(warp * RM + a) * hd + u * 256 + lane * 8 + j] = bit_cast<half2>(o[a][u * 8 + j] * f);
			}
		}
	}
	GroupMemoryBarrierWithGroupSync();
	for (int i = t; i < R * hd; i += FA_THREADS) {
		let a = i / hd;
		let d = i % hd;
		float M = s_wm[a];
		[ForceUnroll]
		for (int w = 1; w < 4; ++w) M = max(M, s_wm[w * 8 + a]);
		float L = 0.0f;
		float val = 0.0f;
		[ForceUnroll]
		for (int w = 0; w < 4; ++w) {
			let mw = s_wm[w * 8 + a];
			L += s_wl[w * 8 + a] > 0.0f ? s_wl[w * 8 + a] * exp(mw - M) : 0.0f;
			val += bit_cast<float>(s_pool[(w * RM + a) * hd + d]);
		}
		let qv = v0 + a;
		let oo = int64_t(r0 + qv / G) * p.num_q_heads + kv_head * G + qv % G;
		if (p.direct != 0) {
			let gi = oo * hd + d;
			let g = p.has_gate != 0 ? 1.0f / (1.0f + exp(-float(p.gate[gi]))) : 1.0f;
			p.out_[gi] = half((L > 0.0f ? val / L : 0.0f) * g);
		}
		else {
			p.part[(int64_t(split) * q_all + oo) * hd + d] = val;
			if (d == 0) {
				let po = int64_t(p.splits) * q_all * hd + (int64_t(split) * q_all + oo) * 2;
				p.part[po] = M;
				p.part[po + 1] = L;
			}
		}
	}
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fa_decode_256(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform FaParams p) {
	fa_decode_body<1, 8>(gid, int(tid.x), p);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fa_decode_512(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform FaParams p) {
	fa_decode_body<2, 4>(gid, int(tid.x), p);
}

// The splits of fa_attn combined: a workgroup of 128 per (row, query head), out = sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) l_s, then the Q4
// values' inverse Walsh-Hadamard rotation per 64 and the gate
struct FaReduceParams {
	float* part;
	half* gate;
	half* out_;
	int rows;
	int num_q_heads;
	int head_dim;
	int splits;
	int has_gate;
	int vq;
};

groupshared float s_o[512];
groupshared float s_w[64];
groupshared float s_sg[8];   // the four subgroups' maxima, then their weighted sums
groupshared float s_inv;

[shader("compute")]
[numthreads(128, 1, 1)]
void fa_reduce(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform FaReduceParams p) {
	let t = int(tid.x);
	let oo = int64_t(gid.y) * p.num_q_heads + int64_t(gid.x);
	let hd = p.head_dim;
	let q_all = int64_t(p.rows) * p.num_q_heads;
	let stride = q_all * hd;
	// every split's running max and sum at once (a thread each, at most 64 splits), their maximum and the weights' sum over the subgroups
	float ms = -1e30f, ls = 0.0f;
	if (t < p.splits) {
		let at = int64_t(p.splits) * stride + ( int64_t(t) * q_all + oo ) * 2;
		ms = p.part[at];
		ls = p.part[at + 1];
	}
	let wmax = WaveActiveMax(ls > 0.0f ? ms : -1e30f);
	if ((t & 31) == 0) s_sg[t >> 5] = wmax;
	GroupMemoryBarrierWithGroupSync();
	let mx = max(max(s_sg[0], s_sg[1]), max(s_sg[2], s_sg[3]));
	let w = ls > 0.0f ? exp(ms - mx) : 0.0f;
	if (t < 64) s_w[t] = w;
	let wsum = WaveActiveSum(w * ls);
	if ((t & 31) == 0) s_sg[4 + (t >> 5)] = wsum;
	GroupMemoryBarrierWithGroupSync();
	if (t == 0) {
		let l = (s_sg[4] + s_sg[5]) + (s_sg[6] + s_sg[7]);
		s_inv = l > 0.0f ? 1.0f / l : 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();
	// four dimensions a thread (16-byte loads), the splits four at a time so their loads are in flight together
	let d4 = 4 * t;
	if (d4 < hd) {
		float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);
		for (int s0 = 0; s0 < p.splits; s0 += 4) {
			float4 v[4];
			[ForceUnroll]
			for (int u = 0; u < 4; ++u) {
				let s = s0 + u;
				v[u] = (s < p.splits && s_w[s] > 0.0f) ? loadAligned<16>((float4*)(p.part + s * stride + oo * hd + d4)) : float4(0.0f, 0.0f, 0.0f, 0.0f);
			}
			[ForceUnroll]
			for (int u = 0; u < 4; ++u) {
				let s = s0 + u;
				let w = s < p.splits ? s_w[s] : 0.0f;
				acc += w * v[u];
			}
		}
		s_o[d4] = acc.x * s_inv;
		s_o[d4 + 1] = acc.y * s_inv;
		s_o[d4 + 2] = acc.z * s_inv;
		s_o[d4 + 3] = acc.w * s_inv;
	}
	GroupMemoryBarrierWithGroupSync();
	for (int d = t; d < hd; d += 128) {
		float val = s_o[d];
		if (p.vq == 3) {
			// out[d] = 0.125 * sum_j (-1)^popcount(d & j) * o[j] within the 64-chunk of d
			let dl = uint(d & 63);
			let c0 = d & ~63;
			float acc = 0.0f;
			for (int j = 0; j < 64; ++j) acc += (countbits(dl & uint(j)) & 1u) != 0u ? -s_o[c0 + j] : s_o[c0 + j];
			val = acc * 0.125f;
		}
		let gi = oo * hd + d;
		if (p.has_gate != 0) val *= 1.0f / (1.0f + exp(-float(p.gate[gi])));
		p.out_[gi] = half(val);
	}
}

// fa_dense_64: bidirectional attention outside a cache at head_dim 64 (the TTS backbone, vision towers), q / out [rows][heads][64] against every
// key of k / v [keys][kv_heads][64]. A workgroup of 128 per 32 query rows of a head: a thread per query, holding it (scaled) and its
// accumulators in registers; each warp takes a quarter of every tile of 64 keys, whose K and V rows are staged in shared memory as FP32 and read
// by the whole warp at once. Scores in FP32, the softmax the running one of each thread over blocks of 8 keys, the four quarters' states merged
// at the end.
struct FaDenseParams { half* q; half* k; half* v; half* out_; int rows; int keys; int heads; int kv_heads; float scale; };

static const int FD_KT = 64;     // keys per tile
static const int FD_ROW = 17;    // float4 per staged row (64 floats + 4: the quarters' rows in distinct banks)
groupshared float4 s_fd[2 * FD_KT * FD_ROW];   // the K tile, then the V tile; at the end the quarters' accumulators [4][32] rows of FD_ROW
groupshared float s_fdm[4 * 32];
groupshared float s_fdl[4 * 32];

float4 fd_lo(uint4 w) {
	let a = float2(bit_cast<half2>(w.x));
	let b = float2(bit_cast<half2>(w.y));
	return float4(a.x, a.y, b.x, b.y);
}

float4 fd_hi(uint4 w) {
	let a = float2(bit_cast<half2>(w.z));
	let b = float2(bit_cast<half2>(w.w));
	return float4(a.x, a.y, b.x, b.y);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fa_dense_64(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform FaDenseParams p) {
	let t = int(tid.x);
	let lane = t & 31;
	let quarter = t >> 5;
	let head = int(gid.y);
	let kvh = head / (p.heads / p.kv_heads);
	let row = int(gid.x) * 32 + lane;
	let qo = int64_t(row < p.rows ? row : 0) * p.heads * 64 + head * 64;
	float4 q[16];
	float4 o[16];
	[ForceUnroll]
	for (int e = 0; e < 8; ++e) {
		let w = fa_ld16(p.q, qo + e * 8);
		q[2 * e] = fd_lo(w) * p.scale;
		q[2 * e + 1] = fd_hi(w) * p.scale;
	}
	[ForceUnroll]
	for (int c = 0; c < 16; ++c) o[c] = float4(0.0f);
	float m = -1e30f;
	float l = 0.0f;
	let kstride = int64_t(p.kv_heads) * 64;

	for (int kt = 0; kt < p.keys; kt += FD_KT) {
		let nk = min(FD_KT, p.keys - kt);
		// the tile's K and V rows, 16-byte loads (64 keys x 8 vectors each)
		for (int i = t; i < FD_KT * 8; i += 128) {
			let key = i >> 3;
			let vec = i & 7;
			uint4 kw = uint4(0u, 0u, 0u, 0u);
			uint4 vw = uint4(0u, 0u, 0u, 0u);
			if (key < nk) {
				let off = int64_t(kt + key) * kstride + kvh * 64 + vec * 8;
				kw = fa_ld16(p.k, off);
				vw = fa_ld16(p.v, off);
			}
			s_fd[key * FD_ROW + 2 * vec] = fd_lo(kw);
			s_fd[key * FD_ROW + 2 * vec + 1] = fd_hi(kw);
			s_fd[(FD_KT + key) * FD_ROW + 2 * vec] = fd_lo(vw);
			s_fd[(FD_KT + key) * FD_ROW + 2 * vec + 1] = fd_hi(vw);
		}
		GroupMemoryBarrierWithGroupSync();
		// this warp's 16 keys, in two blocks of 8
		[ForceUnroll]
		for (int b = 0; b < 2; ++b) {
			let j0 = quarter * 16 + b * 8;
			float s[8];
			[ForceUnroll]
			for (int jj = 0; jj < 8; ++jj) s[jj] = 0.0f;
			[ForceUnroll]
			for (int c = 0; c < 16; ++c) {
				[ForceUnroll]
				for (int jj = 0; jj < 8; ++jj) {
					let kv = s_fd[(j0 + jj) * FD_ROW + c];
					s[jj] = fma(q[c].x, kv.x, fma(q[c].y, kv.y, fma(q[c].z, kv.z, fma(q[c].w, kv.w, s[jj]))));
				}
			}
			float mx = m;
			[ForceUnroll]
			for (int jj = 0; jj < 8; ++jj) {
				if (j0 + jj >= nk) s[jj] = -1e30f;
				mx = max(mx, s[jj]);
			}
			let alpha = exp(m - mx);
			float sum = 0.0f;
			[ForceUnroll]
			for (int jj = 0; jj < 8; ++jj) {
				s[jj] = j0 + jj < nk ? exp(s[jj] - mx) : 0.0f;
				sum += s[jj];
			}
			l = l * alpha + sum;
			m = mx;
			[ForceUnroll]
			for (int c = 0; c < 16; ++c) o[c] *= alpha;
			[ForceUnroll]
			for (int jj = 0; jj < 8; ++jj) {
				[ForceUnroll]
				for (int c = 0; c < 16; ++c) o[c] = fma(float4(s[jj]), s_fd[(FD_KT + j0 + jj) * FD_ROW + c], o[c]);
			}
		}
		GroupMemoryBarrierWithGroupSync();   // the next tile rewrites K and V
	}

	// the quarters' states: a thread per (query, 16 dimensions) combines them
	[ForceUnroll]
	for (int c = 0; c < 16; ++c) s_fd[(quarter * 32 + lane) * FD_ROW + c] = o[c];
	s_fdm[quarter * 32 + lane] = m;
	s_fdl[quarter * 32 + lane] = l;
	GroupMemoryBarrierWithGroupSync();
	let qq = t >> 2;
	let c0 = (t & 3) * 4;
	float mm = -1e30f;
	[ForceUnroll]
	for (int k4 = 0; k4 < 4; ++k4) mm = max(mm, s_fdm[k4 * 32 + qq]);
	float w[4];
	float ll = 0.0f;
	[ForceUnroll]
	for (int k4 = 0; k4 < 4; ++k4) {
		w[k4] = exp(s_fdm[k4 * 32 + qq] - mm);
		ll += s_fdl[k4 * 32 + qq] * w[k4];
	}
	let inv = ll > 0.0f ? 1.0f / ll : 0.0f;
	let r = int(gid.x) * 32 + qq;
	if (r < p.rows) {
		let oo = (int64_t(r) * p.heads + head) * 64;
		[ForceUnroll]
		for (int c = 0; c < 4; ++c) {
			float4 acc = float4(0.0f);
			[ForceUnroll]
			for (int k4 = 0; k4 < 4; ++k4) acc = fma(float4(w[k4]), s_fd[(k4 * 32 + qq) * FD_ROW + c0 + c], acc);
			acc *= inv;
			let d = (c0 + c) * 4;
			p.out_[oo + d] = half(acc.x);
			p.out_[oo + d + 1] = half(acc.y);
			p.out_[oo + d + 2] = half(acc.z);
			p.out_[oo + d + 3] = half(acc.w);
		}
	}
}

// One split's direct output of Q4 values, in place: their inverse rotation per 64 and the gate (a workgroup of 128 per (row, query head))
[shader("compute")]
[numthreads(128, 1, 1)]
void fa_finish(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform FaReduceParams p) {
	let t = int(tid.x);
	let oo = int64_t(gid.y) * p.num_q_heads + int64_t(gid.x);
	let hd = p.head_dim;
	for (int d = t; d < hd; d += 128) s_o[d] = float(p.out_[oo * hd + d]);
	GroupMemoryBarrierWithGroupSync();
	for (int d = t; d < hd; d += 128) {
		let dl = uint(d & 63);
		let c0 = d & ~63;
		float acc = 0.0f;
		for (int j = 0; j < 64; ++j) acc += (countbits(dl & uint(j)) & 1u) != 0u ? -s_o[c0 + j] : s_o[c0 + j];
		float val = acc * 0.125f;
		let gi = oo * hd + d;
		if (p.has_gate != 0) val *= 1.0f / (1.0f + exp(-float(p.gate[gi])));
		p.out_[gi] = half(val);
	}
}
