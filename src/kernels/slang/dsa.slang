// DSA indexer prep and pool kernels (Tier B): LayerNorm of the indexer key into the per-token plane ring, head-weight
// extraction, and the softmax-gated pooling that completes a pool of POOL tokens. The score kernel (wmma) is CUDA-only.
// One group per batch row, 128 threads (one per key dimension); reductions as shared-memory trees on the GPU targets and a
// serial entry point on the host target (ENGINE_TARGET_CPU).

static const int DSA_DIM = 128;
static const int DSA_HEADS = 32;
static const int DSA_POOL = 4;
static const int DSA_PROJ = DSA_DIM + DSA_HEADS + DSA_DIM;
static const int DSA_PLANE = 2 * DSA_DIM;
static const int DSA_PLANE_CAP = 1024;

struct PrepParams {
	half* proj;       // [rows, PROJ] = [k | w | gate]
	half* ln_w;       // [DIM]
	half* ln_b;
	half* plane;      // slot s at + s * plane_slot_stride; [layer][PLANE_CAP][PLANE]
	half* w_out;      // [rows, HEADS]
	int* slots;
	int* positions;
	int layer;
	int64_t plane_slot_stride;
};

void prep_store(PrepParams p, int r, int d, float kn) {
	half* pr = p.proj + int64_t(r) * DSA_PROJ;
	let pos = p.positions[r];
	half* dst = p.plane + int64_t(p.slots[r]) * p.plane_slot_stride + (int64_t(p.layer) * DSA_PLANE_CAP + (pos & (DSA_PLANE_CAP - 1))) * DSA_PLANE;
	dst[d] = half(kn);
	dst[DSA_DIM + d] = pr[DSA_DIM + DSA_HEADS + d];
	if (d < DSA_HEADS) p.w_out[int64_t(r) * DSA_HEADS + d] = pr[DSA_DIM + d];
}

#if !ENGINE_TARGET_CPU
groupshared float s_red[128];

float group_sum(int t, float v) {
	s_red[t] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 64; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let r = s_red[0];
	GroupMemoryBarrierWithGroupSync();
	return r;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void dsa_prep(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PrepParams p) {
	let d = int(tid.x);
	let r = int(gid.x);
	let kv = float(p.proj[int64_t(r) * DSA_PROJ + d]);
	let mean = group_sum(d, kv) * (1.0f / DSA_DIM);
	let dv = kv - mean;
	let variance = group_sum(d, dv * dv) * (1.0f / DSA_DIM);
	prep_store(p, r, d, dv * rsqrt(variance + 1e-6f) * float(p.ln_w[d]) + float(p.ln_b[d]));
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void dsa_prep(uint3 gid : SV_GroupID, uniform PrepParams p) {
	let r = int(gid.x);
	half* pr = p.proj + int64_t(r) * DSA_PROJ;
	float s = 0.0f;
	for (int d = 0; d < DSA_DIM; ++d) s += float(pr[d]);
	let mean = s * (1.0f / DSA_DIM);
	float v = 0.0f;
	for (int d = 0; d < DSA_DIM; ++d) v += (float(pr[d]) - mean) * (float(pr[d]) - mean);
	let inv = rsqrt(v * (1.0f / DSA_DIM) + 1e-6f);
	for (int d = 0; d < DSA_DIM; ++d)
		prep_store(p, r, d, (float(pr[d]) - mean) * inv * float(p.ln_w[d]) + float(p.ln_b[d]));
}
#endif

struct PoolParams {
	half* plane;
	half* pool_plane; // slot s at + s * pool_slot_stride; [layer][max_seq / POOL][DIM]
	half* ape;        // [POOL, DIM]
	int* slots;
	int* positions;
	int layer;
	int max_seq;
	int64_t plane_slot_stride;
	int64_t pool_slot_stride;
};

// Only the row that completes a pool does work. No cross-thread exchange, so one source serves every target.
[shader("compute")]
[numthreads(128, 1, 1)]
void dsa_pool_update(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform PoolParams p) {
	let d = int(tid.x);
	let r = int(gid.x);
	let pos = p.positions[r];
	if ((pos & (DSA_POOL - 1)) != DSA_POOL - 1) return;
	let slot = p.slots[r];
	half* base = p.plane + int64_t(slot) * p.plane_slot_stride + int64_t(p.layer) * DSA_PLANE_CAP * DSA_PLANE;
	float k[4];
	float g[4];
	float mx = -1e30f;
	for (int j = 0; j < DSA_POOL; ++j) {
		half* src = base + int64_t((pos - (DSA_POOL - 1) + j) & (DSA_PLANE_CAP - 1)) * DSA_PLANE;
		k[j] = float(src[d]);
		g[j] = float(src[DSA_DIM + d]) + float(p.ape[j * DSA_DIM + d]);
		mx = max(mx, g[j]);
	}
	float sum = 0.0f;
	float acc = 0.0f;
	for (int j = 0; j < DSA_POOL; ++j) {
		let e = exp(g[j] - mx);
		sum += e;
		acc += e * k[j];
	}
	let pool = pos / DSA_POOL;
	p.pool_plane[int64_t(slot) * p.pool_slot_stride + (int64_t(p.layer) * (p.max_seq / DSA_POOL) + pool) * DSA_DIM + d] = half(acc / sum);
}
