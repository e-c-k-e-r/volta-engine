// Rotary embedding on q and k in place, [rows, heads, head_dim]. interleaved = 0: pairs (i, i + rot/2) (Llama / Qwen);
// 1: pairs (2i, 2i + 1) (GLM4V / MRoPE). pos is the per-row position array.
struct RopeParams {
	half* q;
	half* k;
	int* pos;
	int num_q_heads;
	int num_kv_heads;
	int head_dim;
	int rot_dim;
	int interleaved;
	float freq_base;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void rope(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RopeParams p) {
	// group x = row, thread = rotary pair index
	let pair = int(tid.x);
	let half_rot = p.rot_dim / 2;
	if (pair >= half_rot)
		return;
	let row = int(gid.x);
	let theta = float(p.pos[row]) * pow(p.freq_base, -(2.0f * float(pair) / float(p.rot_dim)));
	let c = cos(theta);
	let s = sin(theta);
	let i0 = p.interleaved != 0 ? 2 * pair : pair;
	let i1 = p.interleaved != 0 ? 2 * pair + 1 : pair + half_rot;
	half* qm = p.q + int64_t(row) * p.num_q_heads * p.head_dim;
	for (int h = 0; h < p.num_q_heads; ++h) {
		half* qh = qm + h * p.head_dim;
		let a = float(qh[i0]);
		let b = float(qh[i1]);
		qh[i0] = half(a * c - b * s);
		qh[i1] = half(a * s + b * c);
	}
	half* km = p.k + int64_t(row) * p.num_kv_heads * p.head_dim;
	for (int h = 0; h < p.num_kv_heads; ++h) {
		half* kh = km + h * p.head_dim;
		let a = float(kh[i0]);
		let b = float(kh[i1]);
		kh[i0] = half(a * c - b * s);
		kh[i1] = half(a * s + b * c);
	}
}
