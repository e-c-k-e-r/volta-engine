// GGUF block weights for the Vulkan backend (GPU targets only; the CPU backend runs llama.cpp's own CPU kernels, src/kernels/cpu/gguf):
// element-wise decoders that follow ggml's dequantize_row_* (ggml-quants.c) value for value, a GEMV / GEMM that decodes the weight row in
// the shader and accumulates against the FP16 activations in fp32 (no activation quantization), and a plain dequantization. Blocks are
// read through 32-bit words (block sizes are not word multiples); the IQ grids, the sign table and the IQ4 values are one word buffer the
// host fills from ggml-common.h (offsets below, in words).

// Type ids (ggml_type)
static const int T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8, T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14;
static const int T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23, T_IQ1_M = 29;

// Word offsets of the tables (mod_gguf.cpp writes them in this order)
static const int W_IQ2XXS = 0, W_IQ2XS = 512, W_IQ2S = 1536, W_IQ3XXS = 3584, W_IQ3S = 3840, W_IQ1S = 4352, W_KSIGNS = 8448, W_KVAL = 8480;

uint rd8(uint* p, int64_t off) { return (p[off >> 2] >> (8u * uint(off & 3))) & 255u; }
uint rd16(uint* p, int64_t off) { return rd8(p, off) | (rd8(p, off + 1) << 8u); }
uint rd32(uint* p, int64_t off) { return rd16(p, off) | (rd16(p, off + 2) << 16u); }
float rdh(uint* p, int64_t off) { return f16tof32(rd16(p, off)); }
int s8(uint b) { return int(b) - ((b & 128u) != 0u ? 256 : 0); }

// byte j (0..7) of 64-bit grid entry e of the table at word offset base
uint grid8(uint* t, int base, uint e, int j) { return (t[base + int(e) * 2 + (j >> 2)] >> (8u * uint(j & 3))) & 255u; }
// byte j (0..3) of 32-bit grid entry e
uint grid4(uint* t, int base, uint e, int j) { return (t[base + int(e)] >> (8u * uint(j))) & 255u; }
uint ksigns(uint* t, uint i) { return (t[W_KSIGNS + int(i >> 2)] >> (8u * (i & 3u))) & 255u; }
int kvalue(uint* t, uint i) { return s8((t[W_KVAL + int(i >> 2)] >> (8u * (i & 3u))) & 255u); }

int block_elems(int type) { return (type == T_Q4_0 || type == T_Q4_1 || type == T_Q5_0 || type == T_Q5_1 || type == T_Q8_0 || type == T_IQ4_NL) ? 32 : 256; }
int block_bytes(int type) {
	switch (type) {
	case T_Q4_0: return 18; case T_Q4_1: return 20; case T_Q5_0: return 22; case T_Q5_1: return 24; case T_Q8_0: return 34; case T_IQ4_NL: return 18;
	case T_Q2_K: return 84; case T_Q3_K: return 110; case T_Q4_K: return 144; case T_Q5_K: return 176; case T_Q6_K: return 210;
	case T_IQ2_XXS: return 66; case T_IQ2_XS: return 74; case T_IQ2_S: return 82; case T_IQ3_XXS: return 98; case T_IQ3_S: return 110;
	case T_IQ1_S: return 50; case T_IQ1_M: return 56; case T_IQ4_XS: return 136;
	default: return 0;
	}
}

// get_scale_min_k4 of the Q4_K / Q5_K 12-byte scales at byte offset s
void scale_min_k4(uint* p, int64_t s, int j, out float sc, out float mn) {
	if (j < 4) {
		sc = float(rd8(p, s + j) & 63u);
		mn = float(rd8(p, s + j + 4) & 63u);
	} else {
		sc = float((rd8(p, s + j + 4) & 15u) | ((rd8(p, s + j - 4) >> 6) << 4));
		mn = float((rd8(p, s + j + 4) >> 4) | ((rd8(p, s + j) >> 6) << 4));
	}
}

// Value i of the block at byte offset b
float dq(int type, uint* p, int64_t b, int i, uint* t) {
	switch (type) {
	case T_Q8_0: return rdh(p, b) * float(s8(rd8(p, b + 2 + i)));
	case T_Q4_0: {
		let q = (rd8(p, b + 2 + (i & 15)) >> (i >= 16 ? 4u : 0u)) & 15u;
		return rdh(p, b) * (float(q) - 8.0f);
	}
	case T_Q4_1: {
		let q = (rd8(p, b + 4 + (i & 15)) >> (i >= 16 ? 4u : 0u)) & 15u;
		return rdh(p, b) * float(q) + rdh(p, b + 2);
	}
	case T_Q5_0: {
		let qh = rd32(p, b + 2);
		let lo = (rd8(p, b + 6 + (i & 15)) >> (i >= 16 ? 4u : 0u)) & 15u;
		let hi = (qh >> uint(i)) & 1u;
		return rdh(p, b) * (float(lo | (hi << 4)) - 16.0f);
	}
	case T_Q5_1: {
		let qh = rd32(p, b + 4);
		let lo = (rd8(p, b + 8 + (i & 15)) >> (i >= 16 ? 4u : 0u)) & 15u;
		let hi = (qh >> uint(i)) & 1u;
		return rdh(p, b) * float(lo | (hi << 4)) + rdh(p, b + 2);
	}
	case T_IQ4_NL: {
		let q = (rd8(p, b + 2 + (i & 15)) >> (i >= 16 ? 4u : 0u)) & 15u;
		return rdh(p, b) * float(kvalue(t, q));
	}
	case T_Q2_K: {
		// scales[16] @0, qs[64] @16, d @80, dmin @82
		let h = i / 128;
		let w = i % 128;
		let j = w / 32;
		let l = w % 32;
		let q = (rd8(p, b + 16 + 32 * h + l) >> uint(2 * j)) & 3u;
		let sc = rd8(p, b + 8 * h + 2 * j + (l >= 16 ? 1 : 0));
		return rdh(p, b + 80) * float(sc & 15u) * float(q) - rdh(p, b + 82) * float(sc >> 4);
	}
	case T_Q3_K: {
		// hmask[32] @0, qs[64] @32, scales[12] @96, d @108 (dequantize_row_q3_K: the 16 6-bit scales unpacked from 12 bytes)
		let h = i / 128;
		let w = i % 128;
		let j = w / 32;
		let l = w % 32;
		let is = 8 * h + 2 * j + (l >= 16 ? 1 : 0);
		// scales: low 4 bits from bytes 0..7 (two per byte), high 2 bits from bytes 8..11
		let lo = (rd8(p, b + 96 + (is & 7)) >> (is >= 8 ? 4u : 0u)) & 15u;
		let hi = (rd8(p, b + 96 + 8 + (is & 3)) >> uint(2 * (is >> 2))) & 3u;
		let sc = float(int(lo | (hi << 4)) - 32);
		let q = (rd8(p, b + 32 + 32 * h + l) >> uint(2 * j)) & 3u;
		let m = (rd8(p, b + l) >> uint(4 * h + j)) & 1u;
		return rdh(p, b + 108) * sc * (float(q) - (m != 0u ? 0.0f : 4.0f));
	}
	case T_Q4_K: {
		// d @0, dmin @2, scales[12] @4, qs[128] @16
		let j = i / 64;
		let w = i % 64;
		float sc, mn;
		scale_min_k4(p, b + 4, 2 * j + (w >= 32 ? 1 : 0), sc, mn);
		let q = (rd8(p, b + 16 + 32 * j + (w & 31)) >> (w >= 32 ? 4u : 0u)) & 15u;
		return rdh(p, b) * sc * float(q) - rdh(p, b + 2) * mn;
	}
	case T_Q5_K: {
		// d @0, dmin @2, scales[12] @4, qh[32] @16, qs[128] @48
		let j = i / 64;
		let w = i % 64;
		let l = w & 31;
		float sc, mn;
		scale_min_k4(p, b + 4, 2 * j + (w >= 32 ? 1 : 0), sc, mn);
		let lo = (rd8(p, b + 48 + 32 * j + l) >> (w >= 32 ? 4u : 0u)) & 15u;
		let hi = (rd8(p, b + 16 + l) >> uint(2 * j + (w >= 32 ? 1 : 0))) & 1u;
		return rdh(p, b) * sc * float(lo | (hi << 4)) - rdh(p, b + 2) * mn;
	}
	case T_Q6_K: {
		// ql[128] @0, qh[64] @128, scales[16] (int8) @192, d @208
		let h = i / 128;
		let w = i % 128;
		let quarter = w / 32;
		let l = w % 32;
		let ql = (rd8(p, b + 64 * h + l + 32 * (quarter & 1)) >> (quarter >= 2 ? 4u : 0u)) & 15u;
		let qh = (rd8(p, b + 128 + 32 * h + l) >> uint(2 * quarter)) & 3u;
		let sc = s8(rd8(p, b + 192 + 8 * h + l / 16 + 2 * quarter));
		return rdh(p, b + 208) * float(sc) * (float(ql | (qh << 4)) - 32.0f);
	}
	case T_IQ4_XS: {
		// d @0, scales_h (u16) @2, scales_l[4] @4, qs[128] @8
		let ib = i / 32;
		let w = i % 32;
		let ls = ((rd8(p, b + 4 + ib / 2) >> uint(4 * (ib % 2))) & 15u) | (((rd16(p, b + 2) >> uint(2 * ib)) & 3u) << 4);
		let q = (rd8(p, b + 8 + 16 * ib + (w & 15)) >> (w >= 16 ? 4u : 0u)) & 15u;
		return rdh(p, b) * (float(ls) - 32.0f) * float(kvalue(t, q));
	}
	case T_IQ3_S: {
		// d @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106
		let ib = i / 32;
		let w = i % 32;
		let l = w / 8;
		let j = w % 8;
		let qh = rd8(p, b + 66 + ib);
		let gi = rd8(p, b + 2 + 8 * ib + 2 * l + (j >= 4 ? 1 : 0)) | ((qh << uint(j >= 4 ? 7 - 2 * l : 8 - 2 * l)) & 256u);
		let g = float(grid4(t, W_IQ3S, gi, j & 3));
		let sign = (rd8(p, b + 74 + 4 * ib + l) & (1u << uint(j))) != 0u ? -1.0f : 1.0f;
		let db = rdh(p, b) * float(1 + 2 * int((rd8(p, b + 106 + ib / 2) >> uint(4 * (ib % 2))) & 15u));
		return db * g * sign;
	}
	case T_IQ3_XXS: {
		// d @0, qs[64] @2, scales and signs (8 x u32) @66
		let ib = i / 32;
		let w = i % 32;
		let l = w / 8;
		let j = w % 8;
		let aux = rd32(p, b + 66 + 4 * ib);
		let g = float(grid4(t, W_IQ3XXS, rd8(p, b + 2 + 8 * ib + 2 * l + (j >= 4 ? 1 : 0)), j & 3));
		let sign = (ksigns(t, (aux >> uint(7 * l)) & 127u) & (1u << uint(j))) != 0u ? -1.0f : 1.0f;
		let db = rdh(p, b) * (0.5f + float(aux >> 28)) * 0.5f;
		return db * g * sign;
	}
	case T_IQ2_S: {
		// d @0, qs[32] @2, signs[32] @34, qh[8] @66, scales[8] @74
		let ib = i / 32;
		let w = i % 32;
		let l = w / 8;
		let j = w % 8;
		let gi = rd8(p, b + 2 + 4 * ib + l) | ((rd8(p, b + 66 + ib) << uint(8 - 2 * l)) & 768u);
		let g = float(grid8(t, W_IQ2S, gi, j));
		let sign = (rd8(p, b + 34 + 4 * ib + l) & (1u << uint(j))) != 0u ? -1.0f : 1.0f;
		let db = rdh(p, b) * (0.5f + float((rd8(p, b + 74 + ib) >> uint(4 * (l / 2))) & 15u)) * 0.25f;
		return db * g * sign;
	}
	case T_IQ2_XS: {
		// d @0, qs (u16)[32] @2, scales[8] @66
		let ib = i / 32;
		let w = i % 32;
		let l = w / 8;
		let j = w % 8;
		let q = rd16(p, b + 2 + 2 * (4 * ib + l));
		let g = float(grid8(t, W_IQ2XS, q & 511u, j));
		let sign = (ksigns(t, q >> 9) & (1u << uint(j))) != 0u ? -1.0f : 1.0f;
		let db = rdh(p, b) * (0.5f + float((rd8(p, b + 66 + ib) >> uint(4 * (l / 2))) & 15u)) * 0.25f;
		return db * g * sign;
	}
	case T_IQ2_XXS: {
		// d @0, qs (u16)[32] @2: per 32 values two words, the first four grid indices, the second the scale and four 7-bit sign indices
		let ib = i / 32;
		let w = i % 32;
		let l = w / 8;
		let j = w % 8;
		let a1 = rd32(p, b + 2 + 8 * ib + 4);
		let g = float(grid8(t, W_IQ2XXS, rd8(p, b + 2 + 8 * ib + l), j));
		let sign = (ksigns(t, (a1 >> uint(7 * l)) & 127u) & (1u << uint(j))) != 0u ? -1.0f : 1.0f;
		let db = rdh(p, b) * (0.5f + float(a1 >> 28)) * 0.25f;
		return db * g * sign;
	}
	case T_IQ1_S: {
		// d @0, qs[32] @2, qh (u16)[8] @34
		let ib = i / 32;
		let w = i % 32;
		let l = w / 8;
		let j = w % 8;
		let qh = rd16(p, b + 34 + 2 * ib);
		let dl = rdh(p, b) * float(2 * int((qh >> 12) & 7u) + 1);
		let delta = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
		let gi = rd8(p, b + 2 + 4 * ib + l) | (((qh >> uint(3 * l)) & 7u) << 8);
		return dl * (float(s8(grid8(t, W_IQ1S, gi, j))) + delta);
	}
	case T_IQ1_M: {
		// qs[32] @0, qh[16] @32, scales (u16)[4] @48; the block scale is the top nibbles of the four scale words (an fp16)
		let ib = i / 32;
		let w = i % 32;
		let l = w / 8;
		let j = w % 8;
		let s0 = rd16(p, b + 48);
		let s1 = rd16(p, b + 50);
		let s2 = rd16(p, b + 52);
		let s3 = rd16(p, b + 54);
		let d = f16tof32((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u));
		let sc = rd16(p, b + 48 + 2 * (ib / 2));
		let dl = d * float(2 * int((sc >> uint(6 * (ib % 2) + 3 * (l / 2))) & 7u) + 1);
		let qh = rd8(p, b + 32 + 2 * ib + l / 2);
		let gi = rd8(p, b + 4 * ib + l) | (((l % 2) == 0 ? (qh << 8) : (qh << 4)) & 0x700u);
		let delta = (qh & ((l % 2) == 0 ? 0x08u : 0x80u)) != 0u ? -0.125f : 0.125f;
		return dl * (float(s8(grid8(t, W_IQ1S, gi, j))) + delta);
	}
	default: return 0.0f;
	}
}

// out[r][c] = sum_i x[r][src(i)] * w[c][i]: a thread per output column and row, the weight row decoded in the shader. tiled: the
// columns are in llama.cpp's tiled V-head order (heads x dim per v), the activation in grouped order (the Qwen3.5 linear-attention out_proj)
struct GgufParams { uint* w; uint* tables; half* x; half* out_; int64_t row_bytes; int type; int m; int k; int n; int heads; int dim; };

int tiled_source(int i, int heads, int dim, int k) {
	if (heads == 0) return i;
	let per_v = heads * dim;
	let v = i / per_v;
	let head = (i % per_v) / dim;
	let d = i % dim;
	return (head * (k / per_v) + v) * dim + d;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gguf_gemv(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GgufParams p) {
	let c = int(gid.x) * 64 + int(tid.x);
	let r = int(gid.y);
	if (c >= p.n || r >= p.m) return;
	let be = block_elems(p.type);
	let bb = block_bytes(p.type);
	half* xr = p.x + int64_t(r) * p.k;
	let row = int64_t(c) * p.row_bytes;
	float acc = 0.0f;
	for (int kb = 0; kb < p.k / be; ++kb) {
		let b = row + int64_t(kb) * bb;
		for (int i = 0; i < be; ++i) acc = fma(float(xr[tiled_source(kb * be + i, p.heads, p.dim, p.k)]), dq(p.type, p.w, b, i, p.tables), acc);
	}
	p.out_[int64_t(r) * p.n + c] = half(acc);
}

// out[e] = value e of the blocks, a thread per value
struct GgufDequantParams { uint* w; uint* tables; half* out_; int64_t elems; int type; };

[shader("compute")]
[numthreads(256, 1, 1)]
void gguf_dequant(uint3 gid : SV_DispatchThreadID, uniform GgufDequantParams p) {
	let e = int64_t(gid.x) + int64_t(gid.y) * 65535 * 256;
	if (e >= p.elems) return;
	let be = block_elems(p.type);
	p.out_[e] = half(dq(p.type, p.w, (e / be) * block_bytes(p.type), int(e % be), p.tables));
}
