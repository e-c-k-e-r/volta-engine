// Gated delta net row kernels (Tier B): gate projection, causal conv1d + SiLU (multi-row, tiled prefill and history update)
// and the multi-row delta-rule recurrence. Slot indirection arrives as a device int array (BatchRows::slot). The reducing
// kernels use shared-memory trees on the GPU targets and serial entry points on the host target (ENGINE_TARGET_CPU).

struct GatesParams {
	half* norm_hidden; // [rows, hidden]
	half* in_proj_a;   // [num_v_heads, hidden]
	half* in_proj_b;
	half* A_log;       // [num_v_heads], may be null
	half* dt_bias;     // [num_v_heads], may be null
	float* gate_out;   // [rows, num_v_heads]
	float* beta_out;
	int num_v_heads;
	int hidden_size;
	int has_a_log;
	int has_dt_bias;
};

void gates_finish(GatesParams p, int head, int row, float total_a, float total_b) {
	let beta_val = 1.0f / (1.0f + exp(-total_b));
	let dt_val = p.has_dt_bias != 0 ? float(p.dt_bias[head]) : 0.0f;
	let a_val = p.has_a_log != 0 ? float(p.A_log[head]) : -2.0f;
	let v = total_a + dt_val;
	let sp = v > 20.0f ? v : log(1.0f + exp(v));
	p.gate_out[row * p.num_v_heads + head] = -exp(a_val) * sp;
	p.beta_out[row * p.num_v_heads + head] = beta_val;
}

#if !ENGINE_TARGET_CPU
groupshared float s_a[256];
groupshared float s_b[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void project_gates(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GatesParams p) {
	let t = int(tid.x);
	let head = int(gid.x);
	let row = int(gid.y);
	half* x = p.norm_hidden + row * p.hidden_size;
	half* a_row = p.in_proj_a + head * p.hidden_size;
	half* b_row = p.in_proj_b + head * p.hidden_size;
	float da = 0.0f;
	float db = 0.0f;
	for (int i = t; i < p.hidden_size; i += 256) {
		let xv = float(x[i]);
		da = fma(float(a_row[i]), xv, da);
		db = fma(float(b_row[i]), xv, db);
	}
	s_a[t] = da;
	s_b[t] = db;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) {
			s_a[t] += s_a[t + s];
			s_b[t] += s_b[t + s];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (t == 0) gates_finish(p, head, row, s_a[0], s_b[0]);
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void project_gates(uint3 gid : SV_GroupID, uniform GatesParams p) {
	let head = int(gid.x);
	let row = int(gid.y);
	half* x = p.norm_hidden + row * p.hidden_size;
	half* a_row = p.in_proj_a + head * p.hidden_size;
	half* b_row = p.in_proj_b + head * p.hidden_size;
	float da = 0.0f;
	float db = 0.0f;
	for (int i = 0; i < p.hidden_size; ++i) {
		let xv = float(x[i]);
		da += float(a_row[i]) * xv;
		db += float(b_row[i]) * xv;
	}
	gates_finish(p, head, row, da, db);
}
#endif

// Conv taps are [channel][4] (oldest first); the history holds the previous three inputs per channel as float.
struct ConvRowsParams {
	half* qkv;        // [rows, channels], in place
	float* history;   // slot s at + s * slot_stride
	half* weights;    // [channels, 4], may be null (identity on the newest tap)
	int* slots;       // [rows] slot of each row
	float* snap;      // post-row history per row at + row * snap_stride, may be null
	int* snap_on;
	int channels;
	int rows;
	int snap_rows;
	int has_weights;
	int64_t slot_stride;
	int64_t snap_stride;
};

float silu_f(float x) {
	return x / (1.0f + exp(-x));
}

// One thread per channel walks the rows in order, keeping the history in registers across rows of the same slot.
[shader("compute")]
[numthreads(256, 1, 1)]
void causal_conv1d_silu_rows(uint3 tid : SV_DispatchThreadID, uniform ConvRowsParams p) {
	let c = int(tid.x);
	if (c >= p.channels) return;
	let hw = p.has_weights != 0;
	let w0 = hw ? float(p.weights[c * 4 + 0]) : 0.0f;
	let w1 = hw ? float(p.weights[c * 4 + 1]) : 0.0f;
	let w2 = hw ? float(p.weights[c * 4 + 2]) : 0.0f;
	let w3 = hw ? float(p.weights[c * 4 + 3]) : 1.0f;
	let do_snap = p.snap != nullptr && p.snap_on[0] != 0;
	int cur = -1;
	float v0 = 0.0f;
	float v1 = 0.0f;
	float v2 = 0.0f;
	for (int m = 0; m < p.rows; ++m) {
		let slot = p.slots[m];
		if (slot != cur) {
			if (cur >= 0) {
				float* h = p.history + int64_t(cur) * p.slot_stride;
				h[c * 3 + 0] = v0;
				h[c * 3 + 1] = v1;
				h[c * 3 + 2] = v2;
			}
			float* h = p.history + int64_t(slot) * p.slot_stride;
			v0 = h[c * 3 + 0];
			v1 = h[c * 3 + 1];
			v2 = h[c * 3 + 2];
			cur = slot;
		}
		let idx = int64_t(m) * p.channels + c;
		let in_val = float(p.qkv[idx]);
		p.qkv[idx] = half(silu_f(v0 * w0 + v1 * w1 + v2 * w2 + in_val * w3));
		v0 = v1;
		v1 = v2;
		v2 = in_val;
		if (do_snap && m < p.snap_rows) {
			float* sp = p.snap + int64_t(m) * p.snap_stride;
			sp[c * 3 + 0] = v0;
			sp[c * 3 + 1] = v1;
			sp[c * 3 + 2] = v2;
		}
	}
	if (cur >= 0) {
		float* h = p.history + int64_t(cur) * p.slot_stride;
		h[c * 3 + 0] = v0;
		h[c * 3 + 1] = v1;
		h[c * 3 + 2] = v2;
	}
}

// Prefill: the FIR is over inputs, so tiles of rows are independent given the three previous inputs. src and dst are
// distinct buffers; one slot per chunk (slots[0]). Group y selects the tile.
struct ConvTiledParams {
	half* src;
	half* dst;
	float* history;
	half* weights;
	int* slots;
	int channels;
	int rows;
	int tile_rows;
	int has_weights;
	int64_t slot_stride;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void causal_conv1d_silu_tiled(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform ConvTiledParams p) {
	let c = int(gid.x) * 128 + int(tid.x);
	if (c >= p.channels) return;
	let r0 = int(gid.y) * p.tile_rows;
	let r1 = min(p.rows, r0 + p.tile_rows);
	float* h = p.history + int64_t(p.slots[0]) * p.slot_stride + c * 3;
	let hw = p.has_weights != 0;
	let w0 = hw ? float(p.weights[c * 4 + 0]) : 0.0f;
	let w1 = hw ? float(p.weights[c * 4 + 1]) : 0.0f;
	let w2 = hw ? float(p.weights[c * 4 + 2]) : 0.0f;
	let w3 = hw ? float(p.weights[c * 4 + 3]) : 1.0f;
	float v[3];
	for (int k = 0; k < 3; ++k) {
		let idx = r0 - 3 + k;
		v[k] = idx >= 0 ? float(p.src[int64_t(idx) * p.channels + c]) : h[3 + idx];
	}
	var v0 = v[0];
	var v1 = v[1];
	var v2 = v[2];
	for (int t = r0; t < r1; ++t) {
		let in_val = float(p.src[int64_t(t) * p.channels + c]);
		p.dst[int64_t(t) * p.channels + c] = half(silu_f(v0 * w0 + v1 * w1 + v2 * w2 + in_val * w3));
		v0 = v1;
		v1 = v2;
		v2 = in_val;
	}
}

// After the tiles: the next chunk's history is the last three inputs.
[shader("compute")]
[numthreads(128, 1, 1)]
void causal_conv1d_history(uint3 tid : SV_DispatchThreadID, uniform ConvTiledParams p) {
	let c = int(tid.x);
	if (c >= p.channels) return;
	float* h = p.history + int64_t(p.slots[0]) * p.slot_stride + c * 3;
	float nv[3];
	for (int k = 0; k < 3; ++k) {
		let idx = p.rows - 3 + k;
		nv[k] = idx >= 0 ? float(p.src[int64_t(idx) * p.channels + c]) : h[3 + idx];
	}
	h[0] = nv[0];
	h[1] = nv[1];
	h[2] = nv[2];
}

// Delta-rule recurrence, head_dim 128. State layout [v_head][col][row] (row fastest) per slot. A group handles
// (head, 4 columns) on the GPU targets: 32 lanes along x each own 4 rows of the state column, y selects the column.
struct RecurrentParams {
	float* state;      // slot s at + s * slot_stride
	half* qkv;         // [rows, (2 * h_k + h_v) * 128]
	float* gate;       // [rows, h_v]
	float* beta;
	half* output;      // [rows, h_v * 128]
	int* slots;
	float* snap;       // may be null
	int* snap_on;
	int h_k;
	int h_v;
	int rows;
	int snap_rows;
	float scale;
	int64_t slot_stride;
	int64_t snap_stride;
};

#if !ENGINE_TARGET_CPU
groupshared float s_red[128];

// Sum across the 32 lanes of row y of the group; result broadcast to the row.
float lane_sum(int lane, int base, float v) {
	s_red[base + lane] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 16; s > 0; s >>= 1) {
		if (lane < s) s_red[base + lane] += s_red[base + lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let r = s_red[base];
	GroupMemoryBarrierWithGroupSync();
	return r;
}

[shader("compute")]
[numthreads(32, 4, 1)]
void recurrent_step_rows(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RecurrentParams p) {
	let lane = int(tid.x);
	let base = int(tid.y) * 32;
	let head = int(gid.x);
	let col = int(gid.z) * 4 + int(tid.y);
	let q_head = head / (p.h_v / p.h_k);
	let qkv_stride = int64_t(2 * p.h_k + p.h_v) * 128;
	let do_snap = p.snap != nullptr && p.snap_on[0] != 0;
	let sidx = (int64_t(head) * 128 + col) * 128;
	float s_shard[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	int cur = -1;
	for (int m = 0; m < p.rows; ++m) {
		let slot = p.slots[m];
		if (slot != cur) {
			if (cur >= 0) {
				float* st = p.state + int64_t(cur) * p.slot_stride;
				for (int r = 0; r < 4; ++r) st[sidx + r * 32 + lane] = s_shard[r];
			}
			float* st = p.state + int64_t(slot) * p.slot_stride;
			for (int r = 0; r < 4; ++r) s_shard[r] = st[sidx + r * 32 + lane];
			cur = slot;
		}
		half* q_ptr = p.qkv + m * qkv_stride + q_head * 128;
		half* k_ptr = p.qkv + m * qkv_stride + p.h_k * 128 + q_head * 128;
		half* v_ptr = p.qkv + m * qkv_stride + 2 * p.h_k * 128 + head * 128;
		float k_reg[4];
		float q_reg[4];
		for (int r = 0; r < 4; ++r) {
			k_reg[r] = float(k_ptr[r * 32 + lane]);
			q_reg[r] = float(q_ptr[r * 32 + lane]);
		}
		let g_val = exp(p.gate[m * p.h_v + head]);
		float kv_shard = 0.0f;
		for (int r = 0; r < 4; ++r) kv_shard += s_shard[r] * k_reg[r];
		let kv_col = lane_sum(lane, base, kv_shard);
		let delta_col = (float(v_ptr[col]) - g_val * kv_col) * p.beta[m * p.h_v + head];
		float attn_partial = 0.0f;
		for (int r = 0; r < 4; ++r) {
			s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
			attn_partial += s_shard[r] * q_reg[r];
		}
		let attn_col = lane_sum(lane, base, attn_partial);
		if (lane == 0) p.output[m * int64_t(p.h_v) * 128 + head * 128 + col] = half(attn_col * p.scale);
		if (do_snap && m < p.snap_rows) {
			float* sp = p.snap + int64_t(m) * p.snap_stride;
			for (int r = 0; r < 4; ++r) sp[sidx + r * 32 + lane] = s_shard[r];
		}
	}
	if (cur >= 0) {
		float* st = p.state + int64_t(cur) * p.slot_stride;
		for (int r = 0; r < 4; ++r) st[sidx + r * 32 + lane] = s_shard[r];
	}
}
#else
// Host target: one group per (head, column), serial over the 128 rows
[shader("compute")]
[numthreads(1, 1, 1)]
void recurrent_step_rows(uint3 gid : SV_GroupID, uniform RecurrentParams p) {
	let head = int(gid.x);
	let col = int(gid.z);
	let q_head = head / (p.h_v / p.h_k);
	let qkv_stride = int64_t(2 * p.h_k + p.h_v) * 128;
	let do_snap = p.snap != nullptr && p.snap_on[0] != 0;
	let sidx = (int64_t(head) * 128 + col) * 128;
	float s[128];
	for (int i = 0; i < 128; ++i) s[i] = 0.0f;
	int cur = -1;
	for (int m = 0; m < p.rows; ++m) {
		let slot = p.slots[m];
		if (slot != cur) {
			if (cur >= 0) {
				float* st = p.state + int64_t(cur) * p.slot_stride;
				for (int i = 0; i < 128; ++i) st[sidx + i] = s[i];
			}
			float* st = p.state + int64_t(slot) * p.slot_stride;
			for (int i = 0; i < 128; ++i) s[i] = st[sidx + i];
			cur = slot;
		}
		half* q_ptr = p.qkv + m * qkv_stride + q_head * 128;
		half* k_ptr = p.qkv + m * qkv_stride + p.h_k * 128 + q_head * 128;
		half* v_ptr = p.qkv + m * qkv_stride + 2 * p.h_k * 128 + head * 128;
		let g_val = exp(p.gate[m * p.h_v + head]);
		float kv = 0.0f;
		for (int i = 0; i < 128; ++i) kv += s[i] * float(k_ptr[i]);
		let delta_col = (float(v_ptr[col]) - g_val * kv) * p.beta[m * p.h_v + head];
		float attn = 0.0f;
		for (int i = 0; i < 128; ++i) {
			s[i] = g_val * s[i] + float(k_ptr[i]) * delta_col;
			attn += s[i] * float(q_ptr[i]);
		}
		p.output[m * int64_t(p.h_v) * 128 + head * 128 + col] = half(attn * p.scale);
		if (do_snap && m < p.snap_rows) {
			float* sp = p.snap + int64_t(m) * p.snap_stride;
			for (int i = 0; i < 128; ++i) sp[sidx + i] = s[i];
		}
	}
	if (cur >= 0) {
		float* st = p.state + int64_t(cur) * p.slot_stride;
		for (int i = 0; i < 128; ++i) st[sidx + i] = s[i];
	}
}
#endif
