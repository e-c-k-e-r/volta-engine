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

// NVFP4 experts for a decode or a short verify (MoE::forward on Vulkan; the CUDA kernels' one-route-per-block form): a workgroup per (32-column
// tile, route), its eight warps over eighths of K, a lane per column of the tile (the prepacked layout puts a column's 16 values of a K group in
// a lane's 8 code bytes, its block scale in the lane's scale byte). tbl holds the local experts' pointers, num_local each: gate codes, gate scales,
// up codes, up scales, down codes, down scales; gs their global scales (gate, up, down). A route whose expert is not local is skipped (its
// output stays unwritten; moe_combine_routes skips it too). The weights are the CUDA kernels' FP16 ones: a nibble pair's E2M1 values times
// 2^-14 are its bits shifted into two halves (normal and subnormal alike), times the block scale fp16( e4m3 * fp16( global * 2^14 ) ) in one FP16
// multiply; the products with the FP16 activations sum in FP32, four K groups at a time into their own sums, the warps' added in order;
// gate_up writes h = silu( fp16( gate ) ) * fp16( up ), down each route's output as FP16
struct Nvfp4MoeParams {
	uint8_t** tbl;
	float* gs;
	half* x;          // gate_up: the rows [rows, hidden]; down: h [routes, inter]
	int* route_idx;   // [rows * topk] global expert ids
	half* out_;       // gate_up: h [routes, inter]; down: the routes' outputs [routes, hidden]
	int* order;       // may be null: workgroup row y takes route order[y] (a prefill chunk's local routes, expert by expert, so an expert's weights
	                  // stay in L2 from one route to the next), else route y
	int topk;
	int num_local;
	int min_expert;
	int hidden;
	int inter;
	float act_limit;  // > 0: up clamped both ways, the activated gate from above
};

static const int NV_MOE_MAX_K = 8192;
static const int NV_WARPS = 8;
groupshared float4 s_x4[NV_MOE_MAX_K / 4];
groupshared float s_wp[NV_WARPS * 64];

// The tile column of a lane (the inverse of nvfp4_gemm's lane of a column)
int nv_lane_col(int lane) {
	return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane >> 4) & 1) * 4;
}

// K values [0, K) of an FP16 row into s_x4 as floats (K a multiple of 4)
void nv_stage_row(half* row, int K, int t) {
	for (int i = t; i < K / 4; i += NV_WARPS * 32) {
		let w = loadAligned<8>((uint2*)row + i);
		let a = bit_cast<half2>(w.x);
		let b = bit_cast<half2>(w.y);
		s_x4[i] = float4(float(a.x), float(a.y), float(b.x), float(b.y));
	}
}

// One K group of the lane's column against s_x4[k0 / 4 ..], into acc: q its 16 nibbles (value 2 i + j of word h at bits 4 i + 16 j), s2 the
// block scale (FP16, at 2^14) in both halves
void nv_group(uint2 q, half2 s2, int k4, inout float acc) {
	[ForceUnroll]
	for (int h = 0; h < 2; ++h) {
		let cw = h == 0 ? q.x : q.y;
		[ForceUnroll]
		for (int i = 0; i < 4; i += 2) {
			let n0 = (cw >> uint(4 * i)) & 0x000F000Fu;
			let n1 = (cw >> uint(4 * i + 4)) & 0x000F000Fu;
			let w0 = bit_cast<half2>(((n0 & 0x00070007u) << 9) | ((n0 & 0x00080008u) << 12)) * s2;
			let w1 = bit_cast<half2>(((n1 & 0x00070007u) << 9) | ((n1 & 0x00080008u) << 12)) * s2;
			let xv = s_x4[k4 + h * 2 + (i >> 1)];
			acc = fma(xv.x, float(w0.x), acc);
			acc = fma(xv.y, float(w0.y), acc);
			acc = fma(xv.z, float(w1.x), acc);
			acc = fma(xv.w, float(w1.y), acc);
		}
	}
}

// The lane's dot product over K groups [g0, g1) of matrix (codes, scales) at element offset off; gm = fp16( global * 2^14 ). Four groups at a time,
// each into its own sum
float nv_dot(uint8_t* codes, uint8_t* scales, int64_t off, int g0, int g1, half gm) {
	float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	for (int g = g0; g < g1; g += 4) {
		uint2 q[4];
		uint sb[4];
		[ForceUnroll]
		for (int u = 0; u < 4; ++u) {
			let gg = min(g + u, g1 - 1);
			q[u] = loadAligned<8>((uint2*)codes + off + int64_t(gg) * 32);
			sb[u] = uint(scales[off + int64_t(gg) * 32]);
		}
		[ForceUnroll]
		for (int u = 0; u < 4; ++u) {
			if (g + u < g1) {
				let s = half(e4m3_value(sb[u])) * gm;
				nv_group(q[u], half2(s, s), (g + u) * 4, acc[u]);
			}
		}
	}
	return (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void nvfp4_moe_gate_up(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Nvfp4MoeParams p) {
	let t = int(tid.x);
	let lane = t & 31;
	let warp = t >> 5;
	let route = p.order != nullptr ? p.order[gid.y] : int(gid.y);
	let e = p.route_idx[route] - p.min_expert;
	if (e < 0 || e >= p.num_local) return;
	let K = p.hidden;
	nv_stage_row(p.x + int64_t(route / p.topk) * K, K, t);
	GroupMemoryBarrierWithGroupSync();
	let G = K / 16;
	let Gq = (G + NV_WARPS - 1) / NV_WARPS;
	let g0 = min(G, warp * Gq);
	let g1 = min(G, g0 + Gq);
	let tile = int(gid.x);
	let off = int64_t(tile) * G * 32 + lane;
	let E = p.num_local;
	let ag = nv_dot(p.tbl[0 * E + e], p.tbl[1 * E + e], off, g0, g1, half(p.gs[e] * 16384.0f));
	let au = nv_dot(p.tbl[2 * E + e], p.tbl[3 * E + e], off, g0, g1, half(p.gs[E + e] * 16384.0f));
	s_wp[warp * 64 + lane] = ag;
	s_wp[warp * 64 + 32 + lane] = au;
	GroupMemoryBarrierWithGroupSync();
	if (warp == 0) {
		float g = 0.0f, u = 0.0f;
		[ForceUnroll]
		for (int w = 0; w < NV_WARPS; ++w) {
			g += s_wp[w * 64 + lane];
			u += s_wp[w * 64 + 32 + lane];
		}
		let gh = r16(g);
		float up = r16(u);
		float act = gh / (1.0f + exp(-gh));
		if (p.act_limit > 0.0f) {
			up = min(max(up, -p.act_limit), p.act_limit);
			act = min(act, p.act_limit);
		}
		p.out_[int64_t(route) * p.inter + tile * 32 + nv_lane_col(lane)] = half(act * up);
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void nvfp4_moe_down(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Nvfp4MoeParams p) {
	let t = int(tid.x);
	let lane = t & 31;
	let warp = t >> 5;
	let route = p.order != nullptr ? p.order[gid.y] : int(gid.y);
	let e = p.route_idx[route] - p.min_expert;
	if (e < 0 || e >= p.num_local) return;
	let K = p.inter;
	nv_stage_row(p.x + int64_t(route) * K, K, t);
	GroupMemoryBarrierWithGroupSync();
	let G = K / 16;
	let Gq = (G + NV_WARPS - 1) / NV_WARPS;
	let g0 = min(G, warp * Gq);
	let g1 = min(G, g0 + Gq);
	let tile = int(gid.x);
	let off = int64_t(tile) * G * 32 + lane;
	let E = p.num_local;
	s_wp[warp * 64 + lane] = nv_dot(p.tbl[4 * E + e], p.tbl[5 * E + e], off, g0, g1, half(p.gs[2 * E + e] * 16384.0f));
	GroupMemoryBarrierWithGroupSync();
	if (warp == 0) {
		float v = 0.0f;
		[ForceUnroll]
		for (int w = 0; w < NV_WARPS; ++w) v += s_wp[w * 64 + lane];
		p.out_[int64_t(route) * p.hidden + tile * 32 + nv_lane_col(lane)] = half(v);
	}
}
