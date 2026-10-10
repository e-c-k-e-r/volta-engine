// NVFP4 W4A16 linear for the Vulkan backend (GPU targets only; the CPU backend has plain C++ in src/backend/cpu/linear_nvfp4.cpp): the prepacked fragment
// layout of src/loader/nvfp4.cpp (streams [tile][group][lane], 32 columns per tile, 16-wide K groups). A thread owns an output column and a row, decodes the
// column's weights (E2M1 * E4M3 block scale * global scale, rounded to fp16 at the points the CUDA kernels round) and accumulates in fp32.

struct Nvfp4Params { uint* codes; uint* scales; half* x; half* out_; int m; int k; int n; float gm; };

float e4m3_value(uint b) {
	let e = int((b >> 3) & 15u);
	let mant = float(b & 7u);
	let v = e == 0 ? exp2(-6.0f) * (mant / 8.0f) : exp2(float(e - 7)) * (1.0f + mant / 8.0f);
	return (b & 0x80u) != 0u ? -v : v;
}

float e2m1_value(uint c) {
	float v;
	switch (c & 7u) {
	case 0: v = 0.0f; break;
	case 1: v = 0.5f; break;
	case 2: v = 1.0f; break;
	case 3: v = 1.5f; break;
	case 4: v = 2.0f; break;
	case 5: v = 3.0f; break;
	case 6: v = 4.0f; break;
	default: v = 6.0f; break;
	}
	return (c & 8u) != 0u ? -v : v;
}

float r16(float f) { return float(half(f)); }

[shader("compute")]
[numthreads(64, 1, 1)]
void nvfp4_gemm(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Nvfp4Params p) {
	let c = int(gid.x) * 64 + int(tid.x);
	let r = int(gid.y);
	if (c >= p.n) return;
	let G = p.k / 16;
	let t = c / 32;
	let j = c % 32;
	let qp = j >> 3;
	let rr = j & 7;
	let lane = qp * 4 + (rr & 3) + (rr >= 4 ? 16 : 0);
	half* xr = p.x + int64_t(r) * p.k;
	float acc = 0.0f;
	for (int g = 0; g < G; ++g) {
		let at = (int64_t(t) * G + g) * 32 + lane;
		let sb = (p.scales[at >> 2] >> (8u * uint(at & 3))) & 255u;
		let s = r16(e4m3_value(sb) * p.gm);
		for (int h = 0; h < 2; ++h) {
			let cw = p.codes[at * 2 + h];
			for (int i = 0; i < 4; ++i) {
				let w0 = r16(e2m1_value((cw >> uint(4 * i)) & 15u) * s) * (1.0f / 16384.0f);
				let w1 = r16(e2m1_value((cw >> uint(16 + 4 * i)) & 15u) * s) * (1.0f / 16384.0f);
				let kk = g * 16 + h * 8 + 2 * i;
				acc = fma(float(xr[kk]), w0, acc);
				acc = fma(float(xr[kk + 1]), w1, acc);
			}
		}
	}
	p.out_[int64_t(r) * p.n + c] = half(acc);
}
