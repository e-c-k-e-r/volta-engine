// Reference EXL3 trellis dequantization and rotations (the engine's reconstruct_slice + had_r_128 math).
//
// Packed layout (codes: uint16 [K/16, N/16, 16*bits]): one 16x16 weight tile per (kt, nt) holds 256 codes of `bits` bits.
// The tile's uint16 words are read as little-endian uint32 words; within a word the stream is MSB-first, and the stream is
// cyclic over 256*bits bits. Position p in 0..255 owns the 16-bit window that starts at stream bit
// (p*bits + bits - 16) mod 256*bits, MSB first (dq() in exl3_dq.cuh). The window is decoded by a multiply-and-mask codebook
// (decode_3inst in codebook.cuh) to one FP16 value. Position p maps to tile element (row, col) as the reconstruct kernel
// scatters it: with L = p / 8 (lane), i = p % 8,
//   row = (L & 3) * 2 + (i & 1) + 8 * ((i >> 1) & 1),  col = 2 * (L / 8 + 4 * (i >> 2)) + ((L >> 2) & 1)
// The result is the weight in the Hadamard-rotated basis, [K, N] row-major, without suh/svh.
//
// Linear(x) = had128( ( had128( x * suh ) @ W ) , then * svh ), with had128 the block-diagonal normalized Walsh-Hadamard over
// blocks of 128 (natural order, scale 1/sqrt(128)); intermediates are FP16 as in LinearTrellis::forward_recon.
// Codebooks: cb 0 = default 3INST (unused by shipped models), cb 1 = MCG (DFlash drafters), cb 2 = MUL1 (everything else).

struct RefReconParams {
	uint16_t* codes;
	half* out_;
	int k_dim;
	int n_dim;
	int bits;
	int cb;
};

// IEEE binary16 bit pattern to float (the codebook only produces finite values)
float half_bits_to_float(uint h) {
	let sign = (h & 0x8000u) != 0u ? -1.0f : 1.0f;
	let e = int((h >> 10) & 31u);
	let m = int(h & 1023u);
	if (e == 0)
		return sign * float(m) * (1.0f / 16777216.0f);
	return sign * asfloat((uint(e + 112) << 23) | (uint(m) << 13));
}

// 16-bit window -> decoded FP16 value (as float)
float decode_3inst_ref(uint w, int cb) {
	if (cb == 2) {
		// MUL1: byte sum of the product, then an FP16 affine map (0x6400 = 1024.0 plus the integer sum)
		let x = w * 0x83DCD12Du;
		let s = (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
		let v = fma(float(1024u + s), half_bits_to_float(0x1eeeu), half_bits_to_float(0xc931u));
		return float(half(v));
	}
	var x = w;
	if (cb == 0)
		x = w * 89226354u + 64248484u;
	else
		x = w * 0xCBAC1FEDu;
	let y = (x & 0x8fff8fffu) ^ 0x3b603b60u;
	let s = half_bits_to_float(y & 0xffffu) + half_bits_to_float(y >> 16);
	return float(half(s));
}

float trellis_value(uint16_t* codes, int tile_base, int bits, int cb, int p) {
	let total = 256 * bits;
	let q0 = (p * bits + bits - 16 + total) % total;
	uint w = 0u;
	for (int j = 0; j < 16; ++j) {
		let q = (q0 + j) % total;
		let bitpos = 31 - (q & 31);
		let word16 = uint(codes[tile_base + (q >> 5) * 2 + (bitpos >> 4)]);
		w = (w << 1) | ((word16 >> uint(bitpos & 15)) & 1u);
	}
	return decode_3inst_ref(w, cb);
}

// One invocation per output element, dispatch ceil(k*n / 256) groups
[shader("compute")]
[numthreads(256, 1, 1)]
void ref_trellis_reconstruct(uint3 tid : SV_DispatchThreadID, uniform RefReconParams p) {
	let idx = int64_t(tid.x);
	if (idx >= int64_t(p.k_dim) * p.n_dim)
		return;
	let r = int(idx / p.n_dim);
	let c = int(idx % p.n_dim);
	let kt = r / 16;
	let nt = c / 16;
	let rr = r % 16;
	let cl = c % 16;
	let c8 = cl >> 1;
	let lane = (c8 & 3) * 8 + (cl & 1) * 4 + ((rr >> 1) & 3);
	let i = (c8 >> 2) * 4 + (rr >> 3) * 2 + (rr & 1);
	let tile_base = (kt * (p.n_dim / 16) + nt) * 16 * p.bits;
	p.out_[idx] = half(trellis_value(p.codes, tile_base, p.bits, p.cb, lane * 8 + i));
}

// Blockwise 128-point Hadamard along the last dim of [rows, cols]: dst = post * (r_scale * H(pre * src)), pre/post optional.
// Group = (block, row), 128 threads, each output element is an explicit 128-term signed sum (no butterfly).
struct RefHadParams {
	half* src;
	half* dst;
	half* pre;
	half* post;
	int cols;
	float r_scale;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void ref_had128(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform RefHadParams p) {
	let j = int(tid.x);
	let base = int64_t(gid.y) * p.cols + int64_t(gid.x) * 128;
	let sbase = int(gid.x) * 128;
	float acc = 0.0f;
	for (int i = 0; i < 128; ++i) {
		var v = float(p.src[base + i]);
		if (p.pre != nullptr)
			v = float(half(v * float(p.pre[sbase + i])));
		let sgn = (countbits(uint(i & j)) & 1u) != 0u ? -1.0f : 1.0f;
		acc += sgn * v;
	}
	var o = float(half(acc * p.r_scale));
	if (p.post != nullptr)
		o = float(half(o * float(p.post[sbase + j])));
	p.dst[base + j] = half(o);
}
