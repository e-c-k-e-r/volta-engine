// The vall-e TTS model's and the NeMo codec's operators for the Vulkan backend (engine/modules/tts.h; the CUDA forms are in
// src/backend/cuda/modules/tts.cu, the same arithmetic in FP32). Reductions go through shared memory, so no kernel depends on the subgroup
// size. Elementwise kernels take a grid of up to 65535 x n workgroups of 256 (element tid.x + tid.y * 65535 * 256). The GEMMs are llama.cpp's
// matmul (mod_gguf.cpp), the attention fa.slang's fa_attn without slots, and the rotary table and the timestep embedding are made on the host.

groupshared float t_red[256];

float tsum256(int t, float v) {
	t_red[t] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int s = 128; s > 0; s >>= 1) {
		if (t < s) t_red[t] += t_red[t + s];
		GroupMemoryBarrierWithGroupSync();
	}
	let r = t_red[0];
	GroupMemoryBarrierWithGroupSync();
	return r;
}

int64_t elem(uint3 tid) { return int64_t(tid.x) + int64_t(tid.y) * 65535 * 256; }

float silu(float x) { return x / (1.0f + exp(-x)); }

// Round half to even (torch.round)
float rne(float x) {
	let f = floor(x);
	let d = x - f;
	if (d > 0.5f) return f + 1.0f;
	if (d < 0.5f) return f;
	return fmod(f, 2.0f) == 0.0f ? f : f + 1.0f;
}

// ---- the backbone

struct DigitsParams { uint8_t* digits; half* vals; int64_t n; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_digits_values(uint3 tid : SV_DispatchThreadID, uniform DigitsParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	let h = (i % 4 == 0) ? 4.0f : 2.0f;
	p.vals[i] = half((float(p.digits[i]) - h) / h);
}

struct EmbedMixParams { float* x; float* a; half* text_emb; int* ids; uint8_t* kind; uint8_t* masked; float* mask_emb; float* len_q; float* segment; float* mode_vec; int d; };

// A workgroup per position
[shader("compute")]
[numthreads(256, 1, 1)]
void tts_embed_mix(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform EmbedMixParams p) {
	let pp = int(gid.x);
	let k = int(p.kind[pp]);
	let seg = min(max(k - 1, 0), 2);
	for (int c = int(tid.x); c < p.d; c += 256) {
		float v = k == 1 ? float(p.text_emb[int64_t(p.ids[pp]) * p.d + c]) : p.a[int64_t(pp) * p.d + c];
		if (p.masked[pp] != 0) v = p.mask_emb[c];
		if (k == 5 && p.len_q != nullptr) v = p.len_q[c];
		p.x[int64_t(pp) * p.d + c] = v + p.segment[seg * p.d + c] + p.mode_vec[c];
	}
}

struct AddNormParams { float* x; float* branch; float* w; half* out16; float* out32; int d; float eps; };

// A workgroup per row, d <= 256 * 24: x += branch, then RMSNorm( x ) * w
static const int TTS_NORM_PER = 24;
[shader("compute")]
[numthreads(256, 1, 1)]
void tts_add_norm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform AddNormParams p) {
	let t = int(tid.x);
	let row = int64_t(gid.x) * p.d;
	float v[TTS_NORM_PER];
	float ss = 0.0f;
	[ForceUnroll]
	for (int e = 0; e < TTS_NORM_PER; ++e) {
		let c = t + e * 256;
		float a = 0.0f;
		if (c < p.d) {
			a = p.x[row + c];
			if (p.branch != nullptr) {
				a = a + p.branch[row + c];
				p.x[row + c] = a;
			}
		}
		v[e] = a;
		ss += a * a;
	}
	ss = tsum256(t, ss);
	let inv = rsqrt(ss / float(p.d) + p.eps);
	[ForceUnroll]
	for (int e = 0; e < TTS_NORM_PER; ++e) {
		let c = t + e * 256;
		if (c < p.d) {
			let o = v[e] * inv * p.w[c];
			if (p.out16 != nullptr) p.out16[row + c] = half(o);
			else p.out32[row + c] = o;
		}
	}
}

struct RowBiasParams { float* y; float* bias; int64_t n; int cols; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_add_bias(uint3 tid : SV_DispatchThreadID, uniform RowBiasParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	p.y[i] += p.bias[int(i % p.cols)];
}

// A workgroup per position, every head of q and k in shared memory (H * hd <= 4096): the per-head sums of squares by one thread each, then the
// gains and the rotary (rotate-half, cos / sin from the table at the position's place in its row)
static const int TTS_QK_MAX = 4096;
groupshared float s_qk[2 * TTS_QK_MAX];
groupshared float s_rs[2 * 64];

struct QkNormRopeParams { half* qkv; float* qn; float* kn; half* q; half* k; half* v; int* pos; float* cs; float* sn; int L; int H; int hd; int rot; float eps; int head_major; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_qk_norm_rope(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform QkNormRopeParams p) {
	let t = int(tid.x);
	let pp = int(gid.x);
	let d = p.H * p.hd;
	let src = int64_t(pp) * 3 * d;
	for (int c = t; c < 2 * d; c += 256) s_qk[c] = float(p.qkv[src + c]);
	GroupMemoryBarrierWithGroupSync();
	if (t < 2 * p.H) {
		float ss = 0.0f;
		let base = t * p.hd;   // q heads, then k heads
		for (int e = 0; e < p.hd; ++e) ss += s_qk[base + e] * s_qk[base + e];
		s_rs[t] = rsqrt(ss / float(p.hd) + p.eps);
	}
	GroupMemoryBarrierWithGroupSync();
	for (int c = t; c < 2 * d; c += 256) {
		let which = c / d;
		let cc = c % p.hd;
		float* wt = which == 0 ? p.qn : p.kn;
		if (wt != nullptr) s_qk[c] = s_qk[c] * s_rs[c / p.hd] * wt[cc];
	}
	GroupMemoryBarrierWithGroupSync();
	let half_rot = p.rot / 2;
	let ps = p.pos[pp];
	for (int c = t; c < d; c += 256) {
		let h = c / p.hd;
		let cc = c % p.hd;
		let dst = p.head_major != 0 ? (int64_t(h) * p.L + pp) * p.hd + cc : int64_t(pp) * d + c;
		float co = 1.0f;
		float sn = 0.0f;
		if (cc < p.rot) {
			let ti = int64_t(ps) * half_rot + cc % half_rot;
			co = p.cs[ti];
			sn = p.sn[ti];
		}
		[ForceUnroll]
		for (int which = 0; which < 2; ++which) {
			let b = which * d + h * p.hd;
			float o = s_qk[b + cc];
			if (cc < p.rot) {
				let i = cc % half_rot;
				let x1 = s_qk[b + i];
				let x2 = s_qk[b + i + half_rot];
				o = cc < half_rot ? x1 * co - x2 * sn : x2 * co + x1 * sn;
			}
			if (which == 0) p.q[dst] = half(o);
			else p.k[dst] = half(o);
		}
		p.v[dst] = p.qkv[src + 2 * d + c];
	}
}

struct HeadsGateParams { half* o; half* g; half* y; int L; int H; int hd; int head_major; };

// A workgroup per position
[shader("compute")]
[numthreads(256, 1, 1)]
void tts_heads_gate(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform HeadsGateParams p) {
	let pp = int(gid.x);
	let d = p.H * p.hd;
	for (int c = int(tid.x); c < d; c += 256) {
		let h = c / p.hd;
		let e = c % p.hd;
		float v = float(p.o[p.head_major != 0 ? (int64_t(h) * p.L + pp) * p.hd + e : int64_t(pp) * d + c]);
		if (p.g != nullptr) v *= 1.0f / (1.0f + exp(-float(p.g[int64_t(pp) * d + c])));
		p.y[int64_t(pp) * d + c] = half(v);
	}
}

struct TtsSwigluParams { half* up; half* act; int F; };

// A workgroup per row: up [rows, 2F] (a | b) -> silu( a ) * b
[shader("compute")]
[numthreads(256, 1, 1)]
void tts_swiglu(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform TtsSwigluParams p) {
	let r = int64_t(gid.x);
	for (int c = int(tid.x); c < p.F; c += 256) {
		let a = float(p.up[r * 2 * p.F + c]);
		let b = float(p.up[r * 2 * p.F + p.F + c]);
		p.act[r * p.F + c] = half(silu(a) * b);
	}
}

// ---- the flow head

struct GatherParams { float* src; int* pos; half* dst; int offset; int d; };

// A workgroup per gathered row
[shader("compute")]
[numthreads(256, 1, 1)]
void tts_gather_rows(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GatherParams p) {
	let i = int(gid.x);
	let s = int64_t(p.offset + p.pos[i]) * p.d;
	for (int c = int(tid.x); c < p.d; c += 256) p.dst[int64_t(i) * p.d + c] = half(p.src[s + c]);
}

struct BiasActHParams { float* x; float* bias; half* y; int64_t n; int cols; int act; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_bias_act(uint3 tid : SV_DispatchThreadID, uniform BiasActHParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	float v = p.x[i] + (p.bias != nullptr ? p.bias[int(i % p.cols)] : 0.0f);
	if (p.act != 0) v = silu(v);
	p.y[i] = half(v);
}

struct CondSiluParams { float* cond; float* temb; half* sc; int64_t n; int W; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_cond_silu(uint3 tid : SV_DispatchThreadID, uniform CondSiluParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	p.sc[i] = half(silu(p.cond[i] + p.temb[int(i % p.W)]));
}

struct LnModParams { float* x; float* mod; half* out_; int mstride; int W; };

// A workgroup per row: LayerNorm (no affine, eps 1e-6), then the modulation
[shader("compute")]
[numthreads(256, 1, 1)]
void tts_ln_mod(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform LnModParams p) {
	let t = int(tid.x);
	let r = int64_t(gid.x);
	float* xr = p.x + r * p.W;
	float s = 0.0f;
	for (int c = t; c < p.W; c += 256) s += xr[c];
	let mean = tsum256(t, s) / float(p.W);
	float q = 0.0f;
	for (int c = t; c < p.W; c += 256) {
		let dl = xr[c] - mean;
		q += dl * dl;
	}
	let inv = rsqrt(tsum256(t, q) / float(p.W) + 1e-6f);
	float* m = p.mod + r * p.mstride;
	for (int c = t; c < p.W; c += 256) p.out_[r * p.W + c] = half((xr[c] - mean) * inv * (1.0f + m[p.W + c]) + m[c]);
}

struct GateAddParams { float* x; float* branch; float* bias; float* mod; int64_t n; int mstride; int W; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_gate_add(uint3 tid : SV_DispatchThreadID, uniform GateAddParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	let r = i / p.W;
	let c = int(i % p.W);
	p.x[i] += p.mod[r * p.mstride + 2 * p.W + c] * (p.branch[i] + p.bias[c]);
}

struct ReplicateParams { float* x; half* y; int64_t total; int n; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_replicate(uint3 tid : SV_DispatchThreadID, uniform ReplicateParams p) {
	let i = elem(tid);
	if (i >= p.total) return;
	p.y[i] = half(p.x[int(i % p.n)]);
}

struct GuideParams { float* v; float* bias; float* x; float* vt; int n; int V; int R; float cfg; float cfg_prompt; float dt; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_guide_step(uint3 tid : SV_DispatchThreadID, uniform GuideParams p) {
	let i = int(elem(tid));
	if (i >= p.n) return;
	let b = p.bias[i % p.V];
	let vc = p.v[i] + b;
	float g = vc;
	if (p.R >= 2) {
		let vn = p.v[p.n + i] + b;
		g = vn + (vc - vn) * p.cfg;
		if (p.R == 3) g += (vc - (p.v[2 * p.n + i] + b)) * p.cfg_prompt;
	}
	if (p.vt != nullptr) p.vt[i] = g;
	p.x[i] += g * p.dt;
}

struct SnapParams { float* x; uint8_t* out_; int n; };

[shader("compute")]
[numthreads(256, 1, 1)]
void tts_snap(uint3 tid : SV_DispatchThreadID, uniform SnapParams p) {
	let i = int(elem(tid));
	if (i >= p.n) return;
	let L = (i % 4 == 0) ? 8 : 5;
	let h = float(L / 2);
	let r = max(rne(p.x[i] * h + h), 0.0f);
	p.out_[i] = uint8_t(int(min(r, float(L - 1))));
}

// ---- the codec

struct DequantParams { uint8_t* digits; float* x; int T; int h0; int h1; int h2; int h3; };

// digits [T, 32] -> x [32, T]
[shader("compute")]
[numthreads(256, 1, 1)]
void codec_dequant(uint3 tid : SV_DispatchThreadID, uniform DequantParams p) {
	let i = int(elem(tid));
	if (i >= 32 * p.T) return;
	let c = i / p.T;
	let t = i % p.T;
	let j = c % 4;
	let h = j == 0 ? p.h0 : j == 1 ? p.h1 : j == 2 ? p.h2 : p.h3;
	p.x[i] = (float(p.digits[int64_t(t) * 32 + c]) - float(h)) / float(h);
}

// Per place j of a group of 4: scale ( levels - 1 ) / 2 ( 1 - eps ), offset 0.5 for an even level count, shift tan( offset / scale ) (the
// host's), half the level count
struct QuantParams { float* x; uint8_t* digits; int T; float sc0; float sc1; float sc2; float sc3; float of0; float of1; float of2; float of3; float sh0; float sh1;
	float sh2; float sh3; int hl0; int hl1; int hl2; int hl3; };

[shader("compute")]
[numthreads(256, 1, 1)]
void codec_quant(uint3 tid : SV_DispatchThreadID, uniform QuantParams p) {
	let i = int(elem(tid));
	if (i >= 32 * p.T) return;
	let c = i / p.T;
	let t = i % p.T;
	let j = c % 4;
	let sc = j == 0 ? p.sc0 : j == 1 ? p.sc1 : j == 2 ? p.sc2 : p.sc3;
	let of = j == 0 ? p.of0 : j == 1 ? p.of1 : j == 2 ? p.of2 : p.of3;
	let sh = j == 0 ? p.sh0 : j == 1 ? p.sh1 : j == 2 ? p.sh2 : p.sh3;
	let hl = j == 0 ? p.hl0 : j == 1 ? p.hl1 : j == 2 ? p.hl2 : p.hl3;
	let code = rne(sc * tanh(p.x[i] + sh) - of);
	p.digits[int64_t(t) * 32 + c] = uint8_t(int(code) + hl);
}

struct CodecActParams { float* x; float* y; float* alpha; int64_t n; int T; int sc; };

// HalfSnake on the first sc channels (alpha), leaky ReLU on the rest
[shader("compute")]
[numthreads(256, 1, 1)]
void codec_act(uint3 tid : SV_DispatchThreadID, uniform CodecActParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	let c = int(i / p.T);
	let v = p.x[i];
	if (c < p.sc) {
		let a = p.alpha[c];
		let s = sin(a * v);
		p.y[i] = v + s * s / (a + 1e-9f);
	}
	else p.y[i] = v > 0.0f ? v : 0.01f * v;
}

struct ColsParams { float* x; float* cols; int ci; int k; int dil; int stride; int pad; int T; int t0; int n; };

// cols [n, ci * k] (each column a GEMM row): x[c, reflect((t0 + t) stride + j dil - pad)] at [t, c k + j]
[shader("compute")]
[numthreads(256, 1, 1)]
void codec_cols(uint3 tid : SV_DispatchThreadID, uniform ColsParams p) {
	let K = p.ci * p.k;
	let i = elem(tid);
	if (i >= int64_t(p.n) * K) return;
	let t = int(i / K);
	let row = int(i % K);
	let c = row / p.k;
	let j = row % p.k;
	int s = (p.t0 + t) * p.stride + j * p.dil - p.pad;
	s = s < 0 ? -s : s;
	s = s >= p.T ? 2 * (p.T - 1) - s : s;
	p.cols[i] = p.x[int64_t(c) * p.T + s];
}

struct UpColsParams { float* x; float* cols; int ci; int T; int q0; int n; };

// cols [n, 3 ci]: x[c, q0 + q + shift] at [q, (shift + 1) ci + c], shift -1 / 0 / +1, zero past the ends
[shader("compute")]
[numthreads(256, 1, 1)]
void codec_up_cols(uint3 tid : SV_DispatchThreadID, uniform UpColsParams p) {
	let K = 3 * p.ci;
	let i = elem(tid);
	if (i >= int64_t(p.n) * K) return;
	let q = int(i / K);
	let row = int(i % K);
	let sh = row / p.ci - 1;
	let c = row % p.ci;
	let s = p.q0 + q + sh;
	p.cols[i] = s >= 0 && s < p.T ? p.x[int64_t(c) * p.T + s] : 0.0f;
}

struct InterleaveParams { float* ph; float* bias; float* y; int co; int s; int T_out; int q0; int n; };

// ph [s * co, n] (phase r of channel c at row r co + c) -> y[c, (q0 + q) s + r] + bias
[shader("compute")]
[numthreads(256, 1, 1)]
void codec_interleave(uint3 tid : SV_DispatchThreadID, uniform InterleaveParams p) {
	let i = elem(tid);
	if (i >= int64_t(p.s) * p.co * p.n) return;
	let q = int(i % p.n);
	let row = int(i / p.n);
	let r = row / p.co;
	let c = row % p.co;
	p.y[int64_t(c) * p.T_out + int64_t(p.q0 + q) * p.s + r] = p.ph[i] + p.bias[c];
}

struct ChannelBiasParams { float* y; float* b; int64_t n; int T; };

[shader("compute")]
[numthreads(256, 1, 1)]
void codec_channel_bias(uint3 tid : SV_DispatchThreadID, uniform ChannelBiasParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	p.y[i] += p.b[int(i / p.T)];
}

struct AxpyParams { float* y; float* x; int64_t n; float a; float b; };

[shader("compute")]
[numthreads(256, 1, 1)]
void codec_axpy(uint3 tid : SV_DispatchThreadID, uniform AxpyParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	p.y[i] = p.a * p.y[i] + p.b * p.x[i];
}

struct ClampParams { float* y; int64_t n; };

[shader("compute")]
[numthreads(256, 1, 1)]
void codec_clamp(uint3 tid : SV_DispatchThreadID, uniform ClampParams p) {
	let i = elem(tid);
	if (i >= p.n) return;
	p.y[i] = min(1.0f, max(-1.0f, p.y[i]));
}
