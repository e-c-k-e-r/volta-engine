// The head's per-row top-k and log-sum-exp over a vocabulary shard (sampling, log-probabilities, the speculators' candidates) and the merge of
// the ranks' top-k, as src/spec/dflash_kernels.cu defines them: the top 16 of a row's first `columns` logits in descending order (equal
// values: the lower id first), ids offset to global token ids; the row max and the sum of exp(x - max). GPU targets (shared-memory trees).
static const int TK = 16;
static const int TK_THREADS = 256;

// a before b: the larger value, or the same value and the lower id
bool tk_before(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

struct TopkShardParams { half* logits; float* values; int* indices; int rows; int row_stride; int columns; int index_offset; };
groupshared float s_tv[TK_THREADS][TK];
groupshared int s_ti[TK_THREADS][TK];

// A workgroup per row: each thread keeps the top 16 of its strided columns (ascending ids, so a tie keeps the earlier one), then the
// threads' lists merge pairwise down a tree
[shader("compute")]
[numthreads(TK_THREADS, 1, 1)]
void topk_shard(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform TopkShardParams p) {
	let row = int(gid.x);
	let t = int(tid.x);
	float v[TK];
	int id[TK];
	for (int i = 0; i < TK; ++i) {
		v[i] = -1.0f / 0.0f;
		id[i] = 0x7fffffff;
	}
	for (int c = t; c < p.columns; c += TK_THREADS) {
		let x = float(p.logits[int64_t(row) * p.row_stride + c]);
		if (!(x > v[TK - 1])) continue;
		int at = TK - 1;
		while (at > 0 && x > v[at - 1]) {
			v[at] = v[at - 1];
			id[at] = id[at - 1];
			--at;
		}
		v[at] = x;
		id[at] = c;
	}
	for (int i = 0; i < TK; ++i) {
		s_tv[t][i] = v[i];
		s_ti[t][i] = id[i];
	}
	GroupMemoryBarrierWithGroupSync();
	for (int s = TK_THREADS / 2; s > 0; s >>= 1) {
		if (t < s) {
			int a = 0;
			int b = 0;
			for (int i = 0; i < TK; ++i) {
				if (tk_before(s_tv[t][a], s_ti[t][a], s_tv[t + s][b], s_ti[t + s][b])) {
					v[i] = s_tv[t][a];
					id[i] = s_ti[t][a];
					++a;
				} else {
					v[i] = s_tv[t + s][b];
					id[i] = s_ti[t + s][b];
					++b;
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();
		if (t < s) {
			for (int i = 0; i < TK; ++i) {
				s_tv[t][i] = v[i];
				s_ti[t][i] = id[i];
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (t < TK) {
		p.values[row * TK + t] = s_tv[0][t];
		p.indices[row * TK + t] = (s_ti[0][t] == 0x7fffffff ? 0 : s_ti[0][t]) + p.index_offset;
	}
}

struct LseShardParams { half* logits; float* out_; int rows; int row_stride; int columns; };
groupshared float s_red[TK_THREADS];

[shader("compute")]
[numthreads(TK_THREADS, 1, 1)]
void lse_shard(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform LseShardParams p) {
	let row = int(gid.x);
	let t = int(tid.x);
	float m = -1.0f / 0.0f;
	for (int c = t; c < p.columns; c += TK_THREADS) m = max(m, float(p.logits[int64_t(row) * p.row_stride + c]));
	s_red[t] = m;
	GroupMemoryBarrierWithGroupSync();
	for (int s = TK_THREADS / 2; s > 0; s >>= 1) {
		if (t < s) s_red[t] = max(s_red[t], s_red[t + s]);
		GroupMemoryBarrierWithGroupSync();
	}
	let mx = s_red[0];
	GroupMemoryBarrierWithGroupSync();
	float sum = 0.0f;
	for (int c = t; c < p.columns; c += TK_THREADS) sum += exp(float(p.logits[int64_t(row) * p.row_stride + c]) - mx);
	s_red[t] = sum;
	GroupMemoryBarrierWithGroupSync();
	for (int s = TK_THREADS / 2; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	if (t == 0) {
		p.out_[2 * row] = mx;
		p.out_[2 * row + 1] = s_red[0];
	}
}

// A thread per row: the global top k of the ranks' lists (rank d's slot pos of the row at d * rows * 16 + row * 16 + pos; equal values:
// the earlier rank and slot first)
struct TopkMergeParams { float* part_vals; int* part_ix; float* out_vals; int* out_ix; int tp_size; int rows; int k; };

[shader("compute")]
[numthreads(64, 1, 1)]
void topk_merge(uint3 gid : SV_DispatchThreadID, uniform TopkMergeParams p) {
	let row = int(gid.x);
	if (row >= p.rows) return;
	let n = p.tp_size * p.k;
	uint taken[4] = { 0u, 0u, 0u, 0u };   // at most 128 candidates
	for (int i = 0; i < p.k; ++i) {
		int best = -1;
		float bv = 0.0f;
		for (int j = 0; j < n; ++j) {
			if ((taken[j >> 5] & (1u << uint(j & 31))) != 0u) continue;
			let x = p.part_vals[(j / p.k) * p.rows * TK + row * TK + j % p.k];
			if (best < 0 || x > bv) {
				best = j;
				bv = x;
			}
		}
		taken[best >> 5] |= 1u << uint(best & 31);
		p.out_vals[row * p.k + i] = bv;
		p.out_ix[row * p.k + i] = p.part_ix[(best / p.k) * p.rows * TK + row * TK + best % p.k];
	}
}
