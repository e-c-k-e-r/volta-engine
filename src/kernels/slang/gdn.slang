// Gated delta net head normalizations (Tier B): L2 normalization of the q/k head slices of a packed qkv row, and the gated RMSNorm
// applied to the recurrence output. Heads are <= 256 wide; one group per head (and row), shared-memory reduction on the GPU
// targets, serial entry points for the host target (ENGINE_TARGET_CPU).

struct L2Params {
	half* qkv;
	int num_k_heads;   // groups 0 .. 2 * num_k_heads - 1 are the q then k slices
	int head_dim;
	int scale_q;
	int64_t row_stride;
};

struct GatedNormParams {
	half* attn_out;    // [rows, num_v_heads * head_dim], in place
	half* z;           // same layout, may be null
	half* norm_weight; // [head_dim], may be null
	int num_v_heads;
	int head_dim;
	int use_sigmoid;
	int has_z;
	int has_weight;
};

float gate_value(float z, bool use_sigmoid) {
	return use_sigmoid ? 1.0f / (1.0f + exp(-z)) : z / (1.0f + exp(-z));
}

#if !ENGINE_TARGET_CPU
groupshared float s_red[256];

float group_sum(int t, float v) {
	s_red[t] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) s_red[t] += s_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	return s_red[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void l2_norm_heads(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform L2Params p) {
	let t = int(tid.x);
	let block = int(gid.x);
	half* ptr = p.qkv + int64_t(gid.y) * p.row_stride + int64_t(block) * p.head_dim;
	let active = t < p.head_dim;
	let val = active ? float(ptr[t]) : 0.0f;
	var inv = rsqrt(group_sum(t, val * val) + 1e-6f);
	if (p.scale_q != 0 && block < p.num_k_heads)
		inv *= 1.0f / sqrt(float(p.head_dim));
	if (active) ptr[t] = half(val * inv);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gated_rmsnorm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GatedNormParams p) {
	let t = int(tid.x);
	let row_off = int64_t(gid.y) * p.num_v_heads * p.head_dim;
	half* head_out = p.attn_out + row_off + int64_t(gid.x) * p.head_dim;
	let active = t < p.head_dim;
	let val = active ? float(head_out[t]) : 0.0f;
	let inv = rsqrt(group_sum(t, val * val) / float(p.head_dim) + 1e-6f);
	if (active) {
		let gamma = p.has_weight != 0 ? float(p.norm_weight[t]) : 1.0f;
		let zv = p.has_z != 0 ? float(p.z[row_off + int64_t(gid.x) * p.head_dim + t]) : 0.0f;
		head_out[t] = half(val * inv * gamma * gate_value(zv, p.use_sigmoid != 0));
	}
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void l2_norm_heads(uint3 gid : SV_GroupID, uniform L2Params p) {
	let block = int(gid.x);
	half* ptr = p.qkv + int64_t(gid.y) * p.row_stride + int64_t(block) * p.head_dim;
	float sq = 0.0f;
	for (int i = 0; i < p.head_dim; ++i) sq += float(ptr[i]) * float(ptr[i]);
	var inv = rsqrt(sq + 1e-6f);
	if (p.scale_q != 0 && block < p.num_k_heads)
		inv *= 1.0f / sqrt(float(p.head_dim));
	for (int i = 0; i < p.head_dim; ++i) ptr[i] = half(float(ptr[i]) * inv);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void gated_rmsnorm(uint3 gid : SV_GroupID, uniform GatedNormParams p) {
	let row_off = int64_t(gid.y) * p.num_v_heads * p.head_dim;
	half* head_out = p.attn_out + row_off + int64_t(gid.x) * p.head_dim;
	float sq = 0.0f;
	for (int i = 0; i < p.head_dim; ++i) sq += float(head_out[i]) * float(head_out[i]);
	let inv = rsqrt(sq / float(p.head_dim) + 1e-6f);
	for (int i = 0; i < p.head_dim; ++i) {
		let gamma = p.has_weight != 0 ? float(p.norm_weight[i]) : 1.0f;
		let zv = p.has_z != 0 ? float(p.z[row_off + int64_t(gid.x) * p.head_dim + i]) : 0.0f;
		head_out[i] = half(float(head_out[i]) * inv * gamma * gate_value(zv, p.use_sigmoid != 0));
	}
}
#endif
