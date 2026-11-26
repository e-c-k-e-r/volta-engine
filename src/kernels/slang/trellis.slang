// EXL3 trellis linear for the Vulkan backend (the packed layout, the codebooks and the rotations of ref_trellis.slang): the input rotated
// by had128 (x * suh), a GEMV over the packed tiles for a few rows (trellis_gemv) or the weight decoded to FP16 for llama.cpp's matmul
// (trellis_dequant, transposed: [n, k]), the output rotated back (had128, then * svh). The GEMV and the dequantization are compiled once per
// width code (TR_CODE, engine/modules/trellis_bits.h: whole rates 1 to 8 bits, the half-integer rates 9 to 11); both tile orders.
//
// A warp decodes a 16 x 16 tile, a lane its 8 positions p = 8 lane .. 8 lane + 7: the window of p ends at bit end(p) of the tile's cyclic,
// MSB-first stream, and the 8 windows of a lane end at fixed distances before its last one, so they are funnel shifts of the three words
// ending there. The codebook works on two windows at a time (FP16 pairs, as the CUDA kernels' decode_3inst_2). Stock order: lane L holds
// rows 2 (L & 3) + {0, 1, 8, 9} of columns c0 = 2 (L >> 3) + ((L >> 2) & 1) (values 0 to 3) and c0 + 8 (4 to 7); Volta order: lane L holds
// rows 8 ((L >> 3) & 1) + 0..7 of column 8 ((L >> 2) & 1) + 4 (L >> 4) + (L & 3).

#ifndef TR_CODE
#define TR_CODE 4
#endif
static const int CODE = TR_CODE;
static const bool HALF_RATE = CODE > 8;
static const int KB = HALF_RATE ? CODE - 8 : CODE;
static const int TILE_BITS = HALF_RATE ? 128 * (2 * KB + 1) : 256 * KB;
static const int TILE_WORDS = TILE_BITS / 32;

// The end bit of position p's window
int tr_end(int p) {
	if (HALF_RATE) return (p >> 1) * (2 * KB + 1) + ((p & 1) != 0 ? 2 * KB + 1 : KB);
	return (p + 1) * KB;
}

// The codebooks on two 16-bit windows: MUL1 (cb 2) the byte sum of the product as the FP16 1024 + s, mapped by an FP16 multiply-add;
// MCG (cb 1) and 3INST (cb 0) the sum of the two halves of the masked product
uint tr_mul1_bits(uint w) {
	let x = w * 0x83DCD12Du;
#if TR_DOT
	return dot4add_u8packed(x, 0x01010101u, 0x6400u);   // the byte sum as one packed dot product (shaderIntegerDotProduct)
#else
	let t = (x & 0x00ff00ffu) + ((x >> 8) & 0x00ff00ffu);
	return ((t & 0xffffu) + (t >> 16) + 0x6400u) & 0xffffu;
#endif
}
half2 tr_pair(uint w0, uint w1, int cb) {
	if (cb == 2) {
		let h = bit_cast<half2>(tr_mul1_bits(w0) | (tr_mul1_bits(w1) << 16));
		return fma(h, bit_cast<half2>(0x1eee1eeeu), bit_cast<half2>(0xc931c931u));
	}
	var x0 = w0;
	var x1 = w1;
	if (cb == 0) {
		x0 = x0 * 89226354u + 64248484u;
		x1 = x1 * 89226354u + 64248484u;
	}
	else {
		x0 = x0 * 0xCBAC1FEDu;
		x1 = x1 * 0xCBAC1FEDu;
	}
	let y0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
	let y1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
	return bit_cast<half2>((y0 & 0xffffu) | (y1 << 16)) + bit_cast<half2>((y0 >> 16) | (y1 & 0xffff0000u));
}

// A lane's 96 stream bits ending at the end of its last window, r0 the lowest word (the last window's end at bit 0 of r0)
struct TrWords { uint r0; uint r1; uint r2; };

uint tr_shr64(uint hi, uint lo, int s) { return uint((((uint64_t)hi << 32) | (uint64_t)lo) >> uint(s)); }

TrWords tr_words(uint* codes, int64_t tile, int lane) {
	let b = tr_end(lane * 8 + 7) + TILE_BITS;   // unwrapped by a tile: the words before it never go negative
	let w = (b - 1) >> 5;
	let wc = codes[tile + (w % TILE_WORDS)];
	let wb = codes[tile + ((w - 1) % TILE_WORDS)];
	let wa = codes[tile + ((w - 2) % TILE_WORDS)];
	let s = (w + 1) * 32 - b;   // the last window ends s bits above the bottom of word w (0..31)
	TrWords r;
	r.r0 = tr_shr64(wb, wc, s);
	r.r1 = tr_shr64(wa, wb, s);
	r.r2 = wa >> uint(s);   // the windows reach at most 72 bits above the end (8 bits: s <= 24 there), all within wa
	return r;
}

// Bits [c, c + 16) of (r2 : r1 : r0) for a constant c
uint tr_bits16(TrWords t, int c) {
	if (c + 16 <= 32) return (t.r0 >> uint(c)) & 0xffffu;
	if (c < 32) return tr_shr64(t.r1, t.r0, c) & 0xffffu;
	if (c + 16 <= 64) return (t.r1 >> uint(c - 32)) & 0xffffu;
	return tr_shr64(t.r2, t.r1, c - 32) & 0xffffu;
}

// The window of value i ends tr_end(7) - tr_end(i) bits before the lane's last one (the same for every lane, a constant)
void tr_decode(TrWords t, int cb, out half2 v[4]) {
	[ForceUnroll]
	for (int j = 0; j < 4; ++j) v[j] = tr_pair(tr_bits16(t, tr_end(7) - tr_end(2 * j)), tr_bits16(t, tr_end(7) - tr_end(2 * j + 1)), cb);
}

// ---- the rotations

struct TrHadParams { half* src; half* dst; half* pre; half* post; int cols; float r_scale; };

groupshared float s_had[128];

// had128 over blocks of 128 of [rows, cols]: dst = post * half( r_scale * H( half( pre * src ) ) ), the butterfly in FP32. A workgroup of 128
// per (block, row)
[shader("compute")]
[numthreads(128, 1, 1)]
void trellis_had128(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform TrHadParams p) {
	let j = int(tid.x);
	let base = int64_t(gid.y) * p.cols + int64_t(gid.x) * 128;
	let sbase = int(gid.x) * 128;
	float v = float(p.src[base + j]);
	if (p.pre != nullptr) v = float(half(v * float(p.pre[sbase + j])));
	s_had[j] = v;
	GroupMemoryBarrierWithGroupSync();
	for (int h = 1; h < 128; h <<= 1) {
		let mine = s_had[j];
		let other = s_had[j ^ h];
		GroupMemoryBarrierWithGroupSync();
		s_had[j] = (j & h) != 0 ? other - mine : mine + other;
		GroupMemoryBarrierWithGroupSync();
	}
	var o = float(half(s_had[j] * p.r_scale));
	if (p.post != nullptr) o = float(half(o * float(p.post[sbase + j])));
	p.dst[base + j] = half(o);
}

// ---- the GEMV: a workgroup of 128 per 16-column tile, its four warps taking four of every sixteen K tiles at a time; FP32 sums, FP16 out

struct TrGemvParams { uint* codes; half* x; half* y; int m; int k; int n; int cb; int volta; };

#ifndef TR_GEMV_ROWS
#define TR_GEMV_ROWS 8
#endif
static const int TR_ROWS = TR_GEMV_ROWS;   // the rows a GEMV takes at most (compiled for 1 and for 8)
groupshared float s_tr[8 * 32 * TR_ROWS * 2];

void tr_accumulate(half* x, int k, int m, int kt, int lane, int volta, half2 v[4], inout float acc[TR_ROWS][2]) {
	if (volta != 0) {
		let k0 = kt * 16 + 8 * ((lane >> 3) & 1);
		float2 f[4];
		[ForceUnroll]
		for (int j = 0; j < 4; ++j) f[j] = float2(v[j]);
		[ForceUnroll]
		for (int r = 0; r < TR_ROWS; ++r) {
			if (r < m) {
				let xw = loadAligned<16>((uint4*)(x + int64_t(r) * k + k0));
				let a = float2(bit_cast<half2>(xw.x));
				let b = float2(bit_cast<half2>(xw.y));
				let c = float2(bit_cast<half2>(xw.z));
				let d = float2(bit_cast<half2>(xw.w));
				acc[r][0] += dot(a, f[0]) + dot(b, f[1]) + dot(c, f[2]) + dot(d, f[3]);
			}
		}
	}
	else {
		let k0 = kt * 16 + 2 * (lane & 3);
		let f0 = float2(v[0]);
		let f1 = float2(v[1]);
		let f2 = float2(v[2]);
		let f3 = float2(v[3]);
		[ForceUnroll]
		for (int r = 0; r < TR_ROWS; ++r) {
			if (r < m) {
				half* xr = x + int64_t(r) * k + k0;
				let x01 = float2(*(half2*)xr);
				let x89 = float2(*(half2*)(xr + 8));
				acc[r][0] += dot(x01, f0) + dot(x89, f1);
				acc[r][1] += dot(x01, f2) + dot(x89, f3);
			}
		}
	}
}

#ifndef TR_UNROLL
#define TR_UNROLL 4
#endif
#ifndef TR_THREADS
#define TR_THREADS 128
#endif
static const int TR_WARPS = TR_THREADS / 32;

[shader("compute")]
[numthreads(TR_THREADS, 1, 1)]
void trellis_gemv(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform TrGemvParams p) {
	let t = int(tid.x);
	let lane = t & 31;
	let warp = t >> 5;
	let nt = int(gid.x);
	let n_tiles = p.n / 16;
	let k_tiles = p.k / 16;
	float acc[TR_ROWS][2];
	[ForceUnroll]
	for (int r = 0; r < TR_ROWS; ++r) { acc[r][0] = 0.0f; acc[r][1] = 0.0f; }
	// TR_UNROLL tiles a step (their loads in flight together), the warps' steps interleaved
	for (int kt0 = TR_UNROLL * warp; kt0 < k_tiles; kt0 += TR_UNROLL * TR_WARPS) {
		TrWords tw[TR_UNROLL];
		[ForceUnroll]
		for (int u = 0; u < TR_UNROLL; ++u) tw[u] = tr_words(p.codes, (int64_t(min(kt0 + u, k_tiles - 1)) * n_tiles + nt) * TILE_WORDS, lane);
		[ForceUnroll]
		for (int u = 0; u < TR_UNROLL; ++u) {
			if (kt0 + u < k_tiles) {
				half2 v[4];
				tr_decode(tw[u], p.cb, v);
				tr_accumulate(p.x, p.k, p.m, kt0 + u, lane, p.volta, v, acc);
			}
		}
	}
	[ForceUnroll]
	for (int r = 0; r < TR_ROWS; ++r) {
		s_tr[((warp * 32 + lane) * TR_ROWS + r) * 2] = acc[r][0];
		s_tr[((warp * 32 + lane) * TR_ROWS + r) * 2 + 1] = acc[r][1];
	}
	GroupMemoryBarrierWithGroupSync();
	// a thread per (row, column): the lanes holding the column, over the warps
	if (t < 16 * p.m) {
		let r = t >> 4;
		let col = t & 15;
		float sum = 0.0f;
		[ForceUnroll]
		for (int w = 0; w < TR_WARPS; ++w) {
			if (p.volta != 0) {
				// lanes 16 ((col >> 2) & 1) + 4 (col >> 3) + (col & 3) and that + 8
				let lb = 16 * ((col >> 2) & 1) + 4 * (col >> 3) + (col & 3);
				sum += s_tr[((w * 32 + lb) * TR_ROWS + r) * 2] + s_tr[((w * 32 + lb + 8) * TR_ROWS + r) * 2];
			}
			else {
				// lanes 8 ((col & 7) >> 1) + 4 (col & 1) + 0..3, slot col >> 3
				let lb = 8 * ((col & 7) >> 1) + 4 * (col & 1);
				let slot = col >> 3;
				[ForceUnroll]
				for (int u = 0; u < 4; ++u) sum += s_tr[((w * 32 + lb + u) * TR_ROWS + r) * 2 + slot];
			}
		}
		p.y[int64_t(r) * p.n + nt * 16 + col] = half(sum);
	}
}

// ---- the matmul's split-K partials (FP32 slabs of n values) summed in order into the FP16 product

struct TrSplitSumParams { half* out_; float* parts; int64_t n; int splits; };

[shader("compute")]
[numthreads(256, 1, 1)]
void trellis_split_sum(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform TrSplitSumParams p) {
	let i = (int64_t(gid.y) * 65535 + int64_t(gid.x)) * 256 + int64_t(tid.x);
	if (i >= p.n) return;
	float acc = 0.0f;
	for (int s = 0; s < p.splits; ++s) acc += p.parts[int64_t(s) * p.n + i];
	p.out_[i] = half(acc);
}

// ---- the dequantization for the matmul: W^T [n, k] FP16. A workgroup of 128 per four consecutive K tiles of a column tile, a warp per
// tile: the 16 x 64 block staged in shared memory (half pairs, rows padded), then written as 16 rows of 128 contiguous bytes

struct TrDequantParams { uint* codes; half* out_; int k; int n; int cb; int volta; };

static const int TR_DQ_STRIDE = 36;   // uint (half pairs) per staged row: 32 + 4
groupshared uint s_dq[16 * TR_DQ_STRIDE];

[shader("compute")]
[numthreads(128, 1, 1)]
void trellis_dequant(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform TrDequantParams p) {
	let t = int(tid.x);
	let lane = t & 31;
	let w = t >> 5;
	let n_tiles = p.n / 16;
	let nt = int(gid.x % uint(n_tiles));
	let kt0 = int(gid.x / uint(n_tiles)) * 4;
	let kt = kt0 + w;
	half2 v[4];
	tr_decode(tr_words(p.codes, (int64_t(kt) * n_tiles + nt) * TILE_WORDS, lane), p.cb, v);
	if (p.volta != 0) {
		// column 8 ((lane >> 2) & 1) + 4 (lane >> 4) + (lane & 3), rows 8 ((lane >> 3) & 1) + 0..7: four pairs
		let col = 8 * ((lane >> 2) & 1) + 4 * (lane >> 4) + (lane & 3);
		let at = col * TR_DQ_STRIDE + (w * 16 + 8 * ((lane >> 3) & 1)) / 2;
		[ForceUnroll]
		for (int j = 0; j < 4; ++j) s_dq[at + j] = bit_cast<uint>(v[j]);
	}
	else {
		// columns c0 and c0 + 8, rows r0, r0 + 1 and r0 + 8, r0 + 9: a pair each
		let c0 = 2 * (lane >> 3) + ((lane >> 2) & 1);
		let r = (w * 16 + 2 * (lane & 3)) / 2;
		s_dq[c0 * TR_DQ_STRIDE + r] = bit_cast<uint>(v[0]);
		s_dq[c0 * TR_DQ_STRIDE + r + 4] = bit_cast<uint>(v[1]);
		s_dq[(c0 + 8) * TR_DQ_STRIDE + r] = bit_cast<uint>(v[2]);
		s_dq[(c0 + 8) * TR_DQ_STRIDE + r + 4] = bit_cast<uint>(v[3]);
	}
	GroupMemoryBarrierWithGroupSync();
	// thread t: row t >> 3, 8 halves at (t & 7) * 8
	let row = t >> 3;
	let vec = t & 7;
	let at = row * TR_DQ_STRIDE + vec * 4;
	let o = uint4(s_dq[at], s_dq[at + 1], s_dq[at + 2], s_dq[at + 3]);
	storeAligned<16>((uint4*)(p.out_ + int64_t(nt * 16 + row) * p.k + kt0 * 16 + vec * 8), o);
}
