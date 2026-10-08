// Elementwise Tier B kernels: residual add, SwiGLU (separate and fused gate|up layouts), token embedding from a BF16 table.
// Plain per-element kernels: compile unchanged for CUDA, SPIR-V and the host target.

struct AddParams { half* hidden; half* residual; int n; };

[shader("compute")]
[numthreads(256, 1, 1)]
void add_residual(uint3 tid : SV_DispatchThreadID, uniform AddParams p) {
	let i = int(tid.x);
	if (i < p.n)
		p.hidden[i] = half(float(p.hidden[i]) + float(p.residual[i]));
}

struct SwigluParams { half* gate; half* up; half* out_; int n; float limit; };

float swiglu_value(float g, float u, float limit) {
	if (limit > 0.0f) {
		g = min(g, limit);
		u = max(-limit, min(u, limit));
	}
	return g / (1.0f + exp(-g)) * u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void swiglu(uint3 tid : SV_DispatchThreadID, uniform SwigluParams p) {
	let i = int(tid.x);
	if (i < p.n)
		p.out_[i] = half(swiglu_value(float(p.gate[i]), float(p.up[i]), p.limit));
}

// gate_up rows are [gate (inter_dim) | up (inter_dim)]
struct SwigluFusedParams { half* gate_up; half* out_; int inter_dim; int total; float limit; };

[shader("compute")]
[numthreads(256, 1, 1)]
void swiglu_fused(uint3 tid : SV_DispatchThreadID, uniform SwigluFusedParams p) {
	let i = int(tid.x);
	if (i < p.total) {
		let m = i / p.inter_dim;
		let c = i % p.inter_dim;
		let base = m * (2 * p.inter_dim);
		p.out_[i] = half(swiglu_value(float(p.gate_up[base + c]), float(p.gate_up[base + p.inter_dim + c]), p.limit));
	}
}

// table: BF16 bits; grid x covers the hidden dimension, y is the row
struct EmbedParams { uint16_t* table; int* token_ids; half* out_; int hidden_size; };

[shader("compute")]
[numthreads(256, 1, 1)]
void embed_tokens(uint3 tid : SV_DispatchThreadID, uniform EmbedParams p) {
	let col = int(tid.x);
	let row = int(tid.y);
	if (col < p.hidden_size) {
		let src = int64_t(p.token_ids[row]) * p.hidden_size + col;
		p.out_[int64_t(row) * p.hidden_size + col] = half(asfloat(uint(p.table[src]) << 16));
	}
}
