// GGUF block weights for the Vulkan backend (GPU targets only; the CPU backend runs llama.cpp's own CPU kernels, src/kernels/cpu/gguf):
// element-wise decoders that follow ggml's dequantize_row_* (ggml-quants.c) value for value (and 8-value run decoders with the same
// arithmetic), a GEMV / tiled GEMM that decode the weight in the shader and accumulates against the FP16 activations in fp32 (no activation quantization), and a plain dequantization. Blocks are
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

// Bytes off .. off + 3 as one word (little-endian), from the one or two words they straddle
uint rdw(uint* p, int64_t off) {
	let s = uint(off & 3) * 8u;
	let lo = p[off >> 2];
	return s == 0u ? lo : (lo >> s) | (p[(off >> 2) + 1] << (32u - s));
}
// byte j (0..7) of the 8 bytes held in a (0..3) and c (4..7)
uint byte8(uint a, uint c, int j) { return ((j < 4 ? a : c) >> (8u * uint(j & 3))) & 255u; }

// Values off .. off + 7 of the block at byte offset b (off a multiple of 8): the run shares its block's scales, signs and grid entries, read
// once; the same arithmetic as dq value for value. Types without a run decoder take dq per value
void dq8(int type, uint* p, int64_t b, int off, uint* t, out float v[8]) {
	switch (type) {
	case T_Q8_0: {
		let d = rdh(p, b);
		let a = rdw(p, b + 2 + off);
		let c = rdw(p, b + 6 + off);
		for (int j = 0; j < 8; ++j) v[j] = d * float(s8(byte8(a, c, j)));
		return;
	}
	case T_Q2_K: {
		let h = off / 128;
		let w = off % 128;
		let j = w / 32;
		let l = w % 32;
		let a = rdw(p, b + 16 + 32 * h + l);
		let c = rdw(p, b + 20 + 32 * h + l);
		let sc = rd8(p, b + 8 * h + 2 * j + (l >= 16 ? 1 : 0));
		let dl = rdh(p, b + 80) * float(sc & 15u);
		let ml = rdh(p, b + 82) * float(sc >> 4);
		for (int k = 0; k < 8; ++k) v[k] = dl * float((byte8(a, c, k) >> uint(2 * j)) & 3u) - ml;
		return;
	}
	case T_Q3_K: {
		let h = off / 128;
		let w = off % 128;
		let j = w / 32;
		let l = w % 32;
		let is = 8 * h + 2 * j + (l >= 16 ? 1 : 0);
		let lo = (rd8(p, b + 96 + (is & 7)) >> (is >= 8 ? 4u : 0u)) & 15u;
		let hi = (rd8(p, b + 96 + 8 + (is & 3)) >> uint(2 * (is >> 2))) & 3u;
		let dl = rdh(p, b + 108) * float(int(lo | (hi << 4)) - 32);
		let a = rdw(p, b + 32 + 32 * h + l);
		let c = rdw(p, b + 36 + 32 * h + l);
		let ma = rdw(p, b + l);
		let mc = rdw(p, b + 4 + l);
		for (int k = 0; k < 8; ++k) {
			let q = (byte8(a, c, k) >> uint(2 * j)) & 3u;
			let m = (byte8(ma, mc, k) >> uint(4 * h + j)) & 1u;
			v[k] = dl * (float(q) - (m != 0u ? 0.0f : 4.0f));
		}
		return;
	}
	case T_Q4_K: {
		let j = off / 64;
		let w = off % 64;
		float sc, mn;
		scale_min_k4(p, b + 4, 2 * j + (w >= 32 ? 1 : 0), sc, mn);
		let dl = rdh(p, b) * sc;
		let ml = rdh(p, b + 2) * mn;
		let a = rdw(p, b + 16 + 32 * j + (w & 31));
		let c = rdw(p, b + 20 + 32 * j + (w & 31));
		let sh = w >= 32 ? 4u : 0u;
		for (int k = 0; k < 8; ++k) v[k] = dl * float((byte8(a, c, k) >> sh) & 15u) - ml;
		return;
	}
	case T_Q6_K: {
		let h = off / 128;
		let w = off % 128;
		let quarter = w / 32;
		let l = w % 32;
		let a = rdw(p, b + 64 * h + l + 32 * (quarter & 1));
		let c = rdw(p, b + 64 * h + l + 32 * (quarter & 1) + 4);
		let ha = rdw(p, b + 128 + 32 * h + l);
		let hc = rdw(p, b + 132 + 32 * h + l);
		let dl = rdh(p, b + 208) * float(s8(rd8(p, b + 192 + 8 * h + l / 16 + 2 * quarter)));
		let sh = quarter >= 2 ? 4u : 0u;
		for (int k = 0; k < 8; ++k) {
			let ql = (byte8(a, c, k) >> sh) & 15u;
			let qh = (byte8(ha, hc, k) >> uint(2 * quarter)) & 3u;
			v[k] = dl * (float(ql | (qh << 4)) - 32.0f);
		}
		return;
	}
	case T_IQ4_XS: {
		let ib = off / 32;
		let w = off % 32;
		let ls = ((rd8(p, b + 4 + ib / 2) >> uint(4 * (ib % 2))) & 15u) | (((rd16(p, b + 2) >> uint(2 * ib)) & 3u) << 4);
		let dl = rdh(p, b) * (float(ls) - 32.0f);
		let a = rdw(p, b + 8 + 16 * ib + (w & 15));
		let c = rdw(p, b + 12 + 16 * ib + (w & 15));
		let sh = w >= 16 ? 4u : 0u;
		for (int k = 0; k < 8; ++k) v[k] = dl * float(kvalue(t, (byte8(a, c, k) >> sh) & 15u));
		return;
	}
	case T_IQ3_S: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let qh = rd8(p, b + 66 + ib);
		let g0 = t[W_IQ3S + int(rd8(p, b + 2 + 8 * ib + 2 * l) | ((qh << uint(8 - 2 * l)) & 256u))];
		let g1 = t[W_IQ3S + int(rd8(p, b + 3 + 8 * ib + 2 * l) | ((qh << uint(7 - 2 * l)) & 256u))];
		let sg = rd8(p, b + 74 + 4 * ib + l);
		let db = rdh(p, b) * float(1 + 2 * int((rd8(p, b + 106 + ib / 2) >> uint(4 * (ib % 2))) & 15u));
		for (int k = 0; k < 8; ++k) v[k] = db * float(byte8(g0, g1, k)) * ((sg & (1u << uint(k))) != 0u ? -1.0f : 1.0f);
		return;
	}
	case T_IQ3_XXS: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let aux = rd32(p, b + 66 + 4 * ib);
		let g0 = t[W_IQ3XXS + int(rd8(p, b + 2 + 8 * ib + 2 * l))];
		let g1 = t[W_IQ3XXS + int(rd8(p, b + 3 + 8 * ib + 2 * l))];
		let sg = ksigns(t, (aux >> uint(7 * l)) & 127u);
		let db = rdh(p, b) * (0.5f + float(aux >> 28)) * 0.5f;
		for (int k = 0; k < 8; ++k) v[k] = db * float(byte8(g0, g1, k)) * ((sg & (1u << uint(k))) != 0u ? -1.0f : 1.0f);
		return;
	}
	case T_IQ2_S: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let gi = int(rd8(p, b + 2 + 4 * ib + l) | ((rd8(p, b + 66 + ib) << uint(8 - 2 * l)) & 768u));
		let g0 = t[W_IQ2S + gi * 2];
		let g1 = t[W_IQ2S + gi * 2 + 1];
		let sg = rd8(p, b + 34 + 4 * ib + l);
		let db = rdh(p, b) * (0.5f + float((rd8(p, b + 74 + ib) >> uint(4 * (l / 2))) & 15u)) * 0.25f;
		for (int k = 0; k < 8; ++k) v[k] = db * float(byte8(g0, g1, k)) * ((sg & (1u << uint(k))) != 0u ? -1.0f : 1.0f);
		return;
	}
	case T_IQ2_XS: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let q = rd16(p, b + 2 + 2 * (4 * ib + l));
		let gi = int(q & 511u);
		let g0 = t[W_IQ2XS + gi * 2];
		let g1 = t[W_IQ2XS + gi * 2 + 1];
		let sg = ksigns(t, q >> 9);
		let db = rdh(p, b) * (0.5f + float((rd8(p, b + 66 + ib) >> uint(4 * (l / 2))) & 15u)) * 0.25f;
		for (int k = 0; k < 8; ++k) v[k] = db * float(byte8(g0, g1, k)) * ((sg & (1u << uint(k))) != 0u ? -1.0f : 1.0f);
		return;
	}
	case T_IQ2_XXS: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let a1 = rd32(p, b + 2 + 8 * ib + 4);
		let gi = int(rd8(p, b + 2 + 8 * ib + l));
		let g0 = t[W_IQ2XXS + gi * 2];
		let g1 = t[W_IQ2XXS + gi * 2 + 1];
		let sg = ksigns(t, (a1 >> uint(7 * l)) & 127u);
		let db = rdh(p, b) * (0.5f + float(a1 >> 28)) * 0.25f;
		for (int k = 0; k < 8; ++k) v[k] = db * float(byte8(g0, g1, k)) * ((sg & (1u << uint(k))) != 0u ? -1.0f : 1.0f);
		return;
	}
	default:
		for (int k = 0; k < 8; ++k) v[k] = dq(type, p, b, off + k, t);
		return;
	}
}

// out[r][c] = sum_i x[r][src(i)] * w[c][i], the weight row decoded in the shader. tiled: the
// columns are in llama.cpp's tiled V-head order (heads x dim per v), the activation in grouped order (the Qwen3.5 linear-attention out_proj). The
// activations are read as 16-byte vectors (runs of 8 halves)
struct GgufParams { uint* w; uint* tables; uint4* xv; half* out_; int64_t row_bytes; int type; int m; int k; int n; int heads; int dim; };

int tiled_source(int i, int heads, int dim, int k) {
	if (heads == 0) return i;
	let per_v = heads * dim;
	let v = i / per_v;
	let head = (i % per_v) / dim;
	let d = i % dim;
	return (head * (k / per_v) + v) * dim + d;
}

static const int GEMV_ROWS = 8;

// A subgroup per output column (GEMV_SG_COLS of them per workgroup) and group of up to 8 activation rows: each lane decodes runs of 8
// consecutive values (one 256-value block per pass of the subgroup; a run shares its block's scale and grid bytes), each weight once for all
// the group's rows, the partial sums reduced across the subgroup. The lanes of a column are 32 consecutive invocations: one subgroup on a
// 32-wide device
static const int GEMV_SG_COLS = 4;

[shader("compute")]
[numthreads(32 * GEMV_SG_COLS, 1, 1)]
void gguf_gemv_sg(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GgufParams p) {
	let c = int(gid.x) * GEMV_SG_COLS + int(tid.x) / 32;
	if (c >= p.n) return;   // the whole subgroup leaves together
	let lane = int(tid.x) % 32;
	let r0 = int(gid.y) * GEMV_ROWS;
	let rows = min(GEMV_ROWS, p.m - r0);
	let be = block_elems(p.type);
	let bb = block_bytes(p.type);
	let row = int64_t(c) * p.row_bytes;
	float acc[GEMV_ROWS];
	for (int r = 0; r < GEMV_ROWS; ++r) acc[r] = 0.0f;
	for (int i0 = lane * 8; i0 < p.k; i0 += 256) {
		let blk = row + int64_t(i0 / be) * bb;
		let off = i0 % be;
		float wv[8];
		dq8(p.type, p.w, blk, off, p.tables, wv);
		// the run's 8 activations of a row are contiguous (the tiled order moves runs of dim, a multiple of 8): one 16-byte load
		let src = tiled_source(i0, p.heads, p.dim, p.k);
		for (int r = 0; r < GEMV_ROWS; ++r) {
			if (r < rows) {
				let xv = p.xv[(int64_t(r0 + r) * p.k + src) / 8];
				float a = acc[r];
				a = fma(f16tof32(xv.x & 0xffffu), wv[0], a);
				a = fma(f16tof32(xv.x >> 16), wv[1], a);
				a = fma(f16tof32(xv.y & 0xffffu), wv[2], a);
				a = fma(f16tof32(xv.y >> 16), wv[3], a);
				a = fma(f16tof32(xv.z & 0xffffu), wv[4], a);
				a = fma(f16tof32(xv.z >> 16), wv[5], a);
				a = fma(f16tof32(xv.w & 0xffffu), wv[6], a);
				a = fma(f16tof32(xv.w >> 16), wv[7], a);
				acc[r] = a;
			}
		}
	}
	for (int r = 0; r < GEMV_ROWS; ++r) {
		let s = WaveActiveSum(acc[r]);
		if (lane == 0 && r < rows) p.out_[int64_t(r0 + r) * p.n + c] = half(s);
	}
}

// Integer path (devices with packed int8 dot products). The activation rows quantized in blocks of 32 as llama.cpp's q8_1 does (d = amax / 127,
// q = round(x / d), half away from zero), in the weight's column order: per block d (a float) and the 32 q packed in 8 words, 10 words
static const int Q8_WORDS = 10;
struct GgufQuantParams { uint4* xv; uint* xq; int m; int k; int heads; int dim; };

[shader("compute")]
[numthreads(128, 1, 1)]
void gguf_quantize_q8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GgufQuantParams p) {
	let blk = int(gid.x) * 128 + int(tid.x);
	let r = int(gid.y);
	if (blk * 32 >= p.k) return;
	float v[32];
	for (int q = 0; q < 4; ++q) {
		let xv = p.xv[(int64_t(r) * p.k + tiled_source(blk * 32 + q * 8, p.heads, p.dim, p.k)) / 8];
		v[q * 8 + 0] = f16tof32(xv.x & 0xffffu);
		v[q * 8 + 1] = f16tof32(xv.x >> 16);
		v[q * 8 + 2] = f16tof32(xv.y & 0xffffu);
		v[q * 8 + 3] = f16tof32(xv.y >> 16);
		v[q * 8 + 4] = f16tof32(xv.z & 0xffffu);
		v[q * 8 + 5] = f16tof32(xv.z >> 16);
		v[q * 8 + 6] = f16tof32(xv.w & 0xffffu);
		v[q * 8 + 7] = f16tof32(xv.w >> 16);
	}
	float amax = 0.0f;
	for (int i = 0; i < 32; ++i) amax = max(amax, abs(v[i]));
	let d = amax / 127.0f;
	let id = d != 0.0f ? 1.0f / d : 0.0f;
	let o = (int64_t(r) * (p.k / 32) + blk) * Q8_WORDS;
	p.xq[o] = asuint(d);
	for (int w = 0; w < 8; ++w) {
		uint packed = 0u;
		for (int j = 0; j < 4; ++j) {
			let x = v[w * 4 + j] * id;
			let q = int(sign(x) * floor(abs(x) + 0.5f));
			packed |= (uint(q) & 255u) << (8u * uint(j));
		}
		p.xq[o + 2 + w] = packed;
	}
}

// Word sources for the integer decoders (the blocks, the tables)
interface IWords { uint at(int i); }
struct GWords : IWords { uint* p; uint at(int i) { return p[i]; } }
static const int GEMV_SG_COLS_DOT = 4;

uint wrd8<W : IWords>(W p, int off) { return (p.at(off >> 2) >> (8u * uint(off & 3))) & 255u; }
uint wrdw<W : IWords>(W p, int off) {
	let s = uint(off & 3) * 8u;
	let lo = p.at(off >> 2);
	return s == 0u ? lo : (lo >> s) | (p.at((off >> 2) + 1) << (32u - s));
}
uint wrd16<W : IWords>(W p, int off) { return wrdw(p, off) & 0xffffu; }
uint wrd32<W : IWords>(W p, int off) { return wrdw(p, off); }
float wrdh<W : IWords>(W p, int off) { return f16tof32(wrd16(p, off)); }
void wscale_min_k4<W : IWords>(W p, int s, int j, out float sc, out float mn) {
	if (j < 4) {
		sc = float(wrd8(p, s + j) & 63u);
		mn = float(wrd8(p, s + j + 4) & 63u);
	} else {
		sc = float((wrd8(p, s + j + 4) & 15u) | ((wrd8(p, s + j - 4) >> 6) << 4));
		mn = float((wrd8(p, s + j + 4) >> 4) | ((wrd8(p, s + j) >> 6) << 4));
	}
}

struct GTab : IWords { uint* t; uint at(int i) { return t[i]; } }
uint wksigns<T : IWords>(T t, uint i) { return (t.at(W_KSIGNS + int(i >> 2)) >> (8u * (i & 3u))) & 255u; }
int wkvalue<T : IWords>(T t, uint i) { return s8((t.at(W_KVAL + int(i >> 2)) >> (8u * (i & 3u))) & 255u); }

// Bytes 0xff where bit j (0..3) of bits is set
uint sign_mask4(uint bits) { return (((bits & 15u) * 0x00204081u) & 0x01010101u) * 255u; }
// The bytes of g (each 1..127) negated where the mask has 0xff (no byte is 0, so ~g + 1 never carries into the next byte)
uint neg_bytes(uint g, uint m) { return (g ^ m) + (m & 0x01010101u); }

// Values off .. off + 7 of the block at byte offset b as two words of signed bytes (a, c) and a scale and offset: value j = sc * q_j - mn.
// False for types without an integer run decoder
bool dq8i<W : IWords, T : IWords>(int type, W p, int b, int off, T t, out uint a, out uint c, out float sc, out float mn) {
	a = 0u;
	c = 0u;
	sc = 0.0f;
	mn = 0.0f;
	switch (type) {
	case T_Q8_0: {
		a = wrdw(p, b + 2 + off);
		c = wrdw(p, b + 6 + off);
		sc = wrdh(p, b);
		return true;
	}
	case T_Q2_K: {
		let h = off / 128;
		let w = off % 128;
		let j = w / 32;
		let l = w % 32;
		a = (wrdw(p, b + 16 + 32 * h + l) >> uint(2 * j)) & 0x03030303u;
		c = (wrdw(p, b + 20 + 32 * h + l) >> uint(2 * j)) & 0x03030303u;
		let s = wrd8(p, b + 8 * h + 2 * j + (l >= 16 ? 1 : 0));
		sc = wrdh(p, b + 80) * float(s & 15u);
		mn = wrdh(p, b + 82) * float(s >> 4);
		return true;
	}
	case T_Q3_K: {
		let h = off / 128;
		let w = off % 128;
		let j = w / 32;
		let l = w % 32;
		let is = 8 * h + 2 * j + (l >= 16 ? 1 : 0);
		let lo = (wrd8(p, b + 96 + (is & 7)) >> (is >= 8 ? 4u : 0u)) & 15u;
		let hi = (wrd8(p, b + 96 + 8 + (is & 3)) >> uint(2 * (is >> 2))) & 3u;
		sc = wrdh(p, b + 108) * float(int(lo | (hi << 4)) - 32);
		mn = 4.0f * sc;   // q - (hmask ? 0 : 4) = (q | hmask << 2) - 4
		let sh = uint(4 * h + j);
		a = ((wrdw(p, b + 32 + 32 * h + l) >> uint(2 * j)) & 0x03030303u) | (((wrdw(p, b + l) >> sh) & 0x01010101u) << 2);
		c = ((wrdw(p, b + 36 + 32 * h + l) >> uint(2 * j)) & 0x03030303u) | (((wrdw(p, b + 4 + l) >> sh) & 0x01010101u) << 2);
		return true;
	}
	case T_Q4_K: {
		let j = off / 64;
		let w = off % 64;
		float s;
		float m;
		wscale_min_k4(p, b + 4, 2 * j + (w >= 32 ? 1 : 0), s, m);
		sc = wrdh(p, b) * s;
		mn = wrdh(p, b + 2) * m;
		let sh = w >= 32 ? 4u : 0u;
		a = (wrdw(p, b + 16 + 32 * j + (w & 31)) >> sh) & 0x0f0f0f0fu;
		c = (wrdw(p, b + 20 + 32 * j + (w & 31)) >> sh) & 0x0f0f0f0fu;
		return true;
	}
	case T_Q6_K: {
		let h = off / 128;
		let w = off % 128;
		let quarter = w / 32;
		let l = w % 32;
		let sh = quarter >= 2 ? 4u : 0u;
		let lb = b + 64 * h + l + 32 * (quarter & 1);
		a = ((wrdw(p, lb) >> sh) & 0x0f0f0f0fu) | (((wrdw(p, b + 128 + 32 * h + l) >> uint(2 * quarter)) & 0x03030303u) << 4);
		c = ((wrdw(p, lb + 4) >> sh) & 0x0f0f0f0fu) | (((wrdw(p, b + 132 + 32 * h + l) >> uint(2 * quarter)) & 0x03030303u) << 4);
		sc = wrdh(p, b + 208) * float(s8(wrd8(p, b + 192 + 8 * h + l / 16 + 2 * quarter)));
		mn = 32.0f * sc;
		return true;
	}
	case T_IQ4_XS: {
		let ib = off / 32;
		let w = off % 32;
		let ls = ((wrd8(p, b + 4 + ib / 2) >> uint(4 * (ib % 2))) & 15u) | (((wrd16(p, b + 2) >> uint(2 * ib)) & 3u) << 4);
		sc = wrdh(p, b) * (float(ls) - 32.0f);
		let sh = w >= 16 ? 4u : 0u;
		let qa = wrdw(p, b + 8 + 16 * ib + (w & 15)) >> sh;
		let qc = wrdw(p, b + 12 + 16 * ib + (w & 15)) >> sh;
		for (int j = 0; j < 4; ++j) {
			a |= (uint(wkvalue(t, (qa >> uint(8 * j)) & 15u)) & 255u) << uint(8 * j);
			c |= (uint(wkvalue(t, (qc >> uint(8 * j)) & 15u)) & 255u) << uint(8 * j);
		}
		return true;
	}
	case T_IQ3_S: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let qh = wrd8(p, b + 66 + ib);
		let g0 = t.at(W_IQ3S + int(wrd8(p, b + 2 + 8 * ib + 2 * l) | ((qh << uint(8 - 2 * l)) & 256u)));
		let g1 = t.at(W_IQ3S + int(wrd8(p, b + 3 + 8 * ib + 2 * l) | ((qh << uint(7 - 2 * l)) & 256u)));
		let sg = wrd8(p, b + 74 + 4 * ib + l);
		a = neg_bytes(g0, sign_mask4(sg));
		c = neg_bytes(g1, sign_mask4(sg >> 4));
		sc = wrdh(p, b) * float(1 + 2 * int((wrd8(p, b + 106 + ib / 2) >> uint(4 * (ib % 2))) & 15u));
		return true;
	}
	case T_IQ3_XXS: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let aux = wrd32(p, b + 66 + 4 * ib);
		let g0 = t.at(W_IQ3XXS + int(wrd8(p, b + 2 + 8 * ib + 2 * l)));
		let g1 = t.at(W_IQ3XXS + int(wrd8(p, b + 3 + 8 * ib + 2 * l)));
		let sg = wksigns(t, (aux >> uint(7 * l)) & 127u);
		a = neg_bytes(g0, sign_mask4(sg));
		c = neg_bytes(g1, sign_mask4(sg >> 4));
		sc = wrdh(p, b) * (0.5f + float(aux >> 28)) * 0.5f;
		return true;
	}
	case T_IQ2_S: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let gi = int(wrd8(p, b + 2 + 4 * ib + l) | ((wrd8(p, b + 66 + ib) << uint(8 - 2 * l)) & 768u));
		let sg = wrd8(p, b + 34 + 4 * ib + l);
		a = neg_bytes(t.at(W_IQ2S + gi * 2), sign_mask4(sg));
		c = neg_bytes(t.at(W_IQ2S + gi * 2 + 1), sign_mask4(sg >> 4));
		sc = wrdh(p, b) * (0.5f + float((wrd8(p, b + 74 + ib) >> uint(4 * (l / 2))) & 15u)) * 0.25f;
		return true;
	}
	case T_IQ2_XS: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let q = wrd16(p, b + 2 + 2 * (4 * ib + l));
		let gi = int(q & 511u);
		let sg = wksigns(t, q >> 9);
		a = neg_bytes(t.at(W_IQ2XS + gi * 2), sign_mask4(sg));
		c = neg_bytes(t.at(W_IQ2XS + gi * 2 + 1), sign_mask4(sg >> 4));
		sc = wrdh(p, b) * (0.5f + float((wrd8(p, b + 66 + ib) >> uint(4 * (l / 2))) & 15u)) * 0.25f;
		return true;
	}
	case T_IQ2_XXS: {
		let ib = off / 32;
		let l = (off % 32) / 8;
		let a1 = wrd32(p, b + 2 + 8 * ib + 4);
		let gi = int(wrd8(p, b + 2 + 8 * ib + l));
		let sg = wksigns(t, (a1 >> uint(7 * l)) & 127u);
		a = neg_bytes(t.at(W_IQ2XXS + gi * 2), sign_mask4(sg));
		c = neg_bytes(t.at(W_IQ2XXS + gi * 2 + 1), sign_mask4(sg >> 4));
		sc = wrdh(p, b) * (0.5f + float(a1 >> 28)) * 0.25f;
		return true;
	}
	default: return false;
	}
}

// gguf_gemv_sg on the q8 activation rows: each run of 8 is two packed int8 dot products against the run's integer weights, scaled by
// the run's scale and the activation block's d, less the run's offset times the activations' sum (one more dot product each). Per pass
// the subgroup of a column copies the column's next 1024 values of blocks into shared memory (coalesced word loads), then each lane
// decodes its 32 values (four runs) from there. The columns of a workgroup run the same passes (the barriers), a missing one idles
// One SPIR-V per block type (-DGGUF_T=<ggml type>): the type a compile-time constant, its decoder alone in the kernel
#ifdef GGUF_T
#define GGUF_TYPE(p) GGUF_T
#else
#define GGUF_TYPE(p) (p).type
#endif

// gguf_gemv_sg on the q8 activation rows: each run of 8 is two packed int8 dot products against the run's integer weights, scaled by
// the run's scale and the activation block's d, less the run's offset times the activations' sum (one more dot product each). A lane
// takes 32 values (four runs) per pass, the blocks read in place
struct GgufDotParams { uint* w; uint* tables; uint* xq; half* out_; int64_t row_bytes; int type; int m; int k; int n; };

[shader("compute")]
[numthreads(32 * GEMV_SG_COLS_DOT, 1, 1)]
void gguf_gemv_dot(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GgufDotParams p) {
	GTab st;
	st.t = p.tables;
	let lane = int(tid.x) % 32;
	let r0 = int(gid.y) * GEMV_ROWS;
	let rows = min(GEMV_ROWS, p.m - r0);
	let type = GGUF_TYPE(p);
	let be = block_elems(type);
	let bb = block_bytes(type);
	let kb = p.k / 32;
	{
		let c = int(gid.x) * GEMV_SG_COLS_DOT + int(tid.x) / 32;
		if (c >= p.n) return;   // the whole subgroup leaves together
		let row = int64_t(c) * p.row_bytes;
		float acc[GEMV_ROWS];
		for (int r = 0; r < GEMV_ROWS; ++r) acc[r] = 0.0f;
		for (int i32 = lane * 32; i32 < p.k; i32 += 1024) {
			// the words from the 32 values' first block on, read in place through 32-bit offsets
			let b0 = i32 / be;
			let start = row + int64_t(b0) * bb;
			GWords gw;
			gw.p = p.w + (start >> 2);
			for (int q4 = 0; q4 < 4; ++q4) {
				let i0 = i32 + q4 * 8;
				let lb = (i0 / be - b0) * bb + int(start & 3);
				let off = i0 % be;
				let qo = (i0 / 32) * Q8_WORDS + 2 + (i0 % 32) / 4;
				uint wa;
				uint wc;
				float sc;
				float mn;
				if (dq8i(type, gw, lb, off, st, wa, wc, sc, mn)) {
					for (int r = 0; r < GEMV_ROWS; ++r) {
						if (r < rows) {
							let xo = (r0 + r) * kb * Q8_WORDS;
							let xa = p.xq[xo + qo];
							let xc = p.xq[xo + qo + 1];
							let d = asfloat(p.xq[xo + (i0 / 32) * Q8_WORDS]);
							let dotw = dot4add_i8packed(wa, xa, dot4add_i8packed(wc, xc, 0));
							let dots = dot4add_i8packed(0x01010101u, xa, dot4add_i8packed(0x01010101u, xc, 0));
							acc[r] = fma(d, sc * float(dotw) - mn * float(dots), acc[r]);
						}
					}
				} else {
					float wv[8];
					dq8(type, p.w, row + int64_t(i0 / be) * bb, off, p.tables, wv);
					for (int r = 0; r < GEMV_ROWS; ++r) {
						if (r < rows) {
							let xo = (r0 + r) * kb * Q8_WORDS;
							let xa = p.xq[xo + qo];
							let xc = p.xq[xo + qo + 1];
							let d = asfloat(p.xq[xo + (i0 / 32) * Q8_WORDS]);
							float sum = 0.0f;
							for (int j = 0; j < 8; ++j) sum = fma(wv[j], float(s8(byte8(xa, xc, j))), sum);
							acc[r] = fma(d, sum, acc[r]);
						}
					}
				}
			}
		}
		for (int r = 0; r < GEMV_ROWS; ++r) {
			let s = WaveActiveSum(acc[r]);
			if (lane == 0 && r < rows) p.out_[int64_t(r0 + r) * p.n + c] = half(s);
		}
	}
}

// More than 8 activation rows: a workgroup per tile of 64 columns x 64 rows, k in steps of 32. Each step decodes the 64 x 32 weight tile
// into shared memory once (two runs of 8 per thread) and loads the 64 x 32 activation tile (one 16-byte load per run), then each thread
// accumulates 4 columns x 8 rows from shared memory
static const int GT_N = 64;
static const int GT_M = 64;
static const int GT_K = 32;
groupshared float s_w[GT_K][GT_N + 4];
groupshared float s_x[GT_K][GT_M + 4];

[shader("compute")]
[numthreads(128, 1, 1)]
void gguf_gemm_tile(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GgufParams p) {
	let t = int(tid.x);
	let c0 = int(gid.x) * GT_N;
	let r0 = int(gid.y) * GT_M;
	let be = block_elems(p.type);
	let bb = block_bytes(p.type);
	let tc = (t % 16) * 4;   // this thread's 4 columns
	let tr = (t / 16) * 8;   // and 8 rows
	float acc[8][4];
	for (int r = 0; r < 8; ++r) {
		for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;
	}
	for (int k0 = 0; k0 < p.k; k0 += GT_K) {
		// the tiles: 64 columns (rows) x 4 runs of 8 values, two runs per thread
		for (int q = 0; q < 2; ++q) {
			let ri = t * 2 + q;
			let lc = ri / 4;
			let kr = (ri % 4) * 8;
			let kk = k0 + kr;
			float wv[8];
			if (c0 + lc < p.n) dq8(p.type, p.w, int64_t(c0 + lc) * p.row_bytes + int64_t(kk / be) * bb, kk % be, p.tables, wv);
			else {
				for (int j = 0; j < 8; ++j) wv[j] = 0.0f;
			}
			for (int j = 0; j < 8; ++j) s_w[kr + j][lc] = wv[j];
			uint4 xv = uint4(0u, 0u, 0u, 0u);
			if (r0 + lc < p.m) xv = p.xv[(int64_t(r0 + lc) * p.k + tiled_source(kk, p.heads, p.dim, p.k)) / 8];
			s_x[kr + 0][lc] = f16tof32(xv.x & 0xffffu);
			s_x[kr + 1][lc] = f16tof32(xv.x >> 16);
			s_x[kr + 2][lc] = f16tof32(xv.y & 0xffffu);
			s_x[kr + 3][lc] = f16tof32(xv.y >> 16);
			s_x[kr + 4][lc] = f16tof32(xv.z & 0xffffu);
			s_x[kr + 5][lc] = f16tof32(xv.z >> 16);
			s_x[kr + 6][lc] = f16tof32(xv.w & 0xffffu);
			s_x[kr + 7][lc] = f16tof32(xv.w >> 16);
		}
		GroupMemoryBarrierWithGroupSync();
		for (int kk = 0; kk < GT_K; ++kk) {
			float w[4];
			float x[8];
			for (int c = 0; c < 4; ++c) w[c] = s_w[kk][tc + c];
			for (int r = 0; r < 8; ++r) x[r] = s_x[kk][tr + r];
			for (int r = 0; r < 8; ++r) {
				for (int c = 0; c < 4; ++c) acc[r][c] = fma(x[r], w[c], acc[r][c]);
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
	for (int r = 0; r < 8; ++r) {
		if (r0 + tr + r >= p.m) break;
		for (int c = 0; c < 4; ++c) {
			if (c0 + tc + c < p.n) p.out_[int64_t(r0 + tr + r) * p.n + c0 + tc + c] = half(acc[r][c]);
		}
	}
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

// out[r][i] = x[r][src(i)]: the activation rows in the weight's tiled column order, for kernels that read them as they are
struct GgufPermuteParams { half* x; half* out_; int m; int k; int heads; int dim; };

[shader("compute")]
[numthreads(256, 1, 1)]
void gguf_permute_rows(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform GgufPermuteParams p) {
	let i = int(gid.x) * 256 + int(tid.x);
	let r = int(gid.y);
	if (i >= p.k) return;
	p.out_[int64_t(r) * p.k + i] = p.x[int64_t(r) * p.k + tiled_source(i, p.heads, p.dim, p.k)];
}
