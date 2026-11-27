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

// ---- the experts of a prefill chunk, a group of up to NVR routes of one expert a workgroup (the chunk's local routes in expert order, `order`;
// groups[3 g] the local expert of group g, groups[3 g + 1] its first entry of order, groups[3 g + 2] its routes): the routes' rows staged in
// chunks of NVR_CK values (FP32 in shared memory), each lane's weights of a K group decoded once for every route of the group, the eight warps
// over the K groups of a chunk and their sums added in order. The same FP16 weights and rounding points as the one-route kernels

struct Nvfp4MoeGrpParams { uint8_t** tbl; float* gs; half* x; half* out_; int* order; int* groups; int topk; int num_local; int hidden; int inter; float act_limit; };

static const int NVR = 8;
static const int NVR_CK = 1024;
groupshared float4 s_xr[NVR * NVR_CK / 4];   // the routes' chunk; at the end the warps' partials

// The lane's 16 weights of K group g (16 FP32 values in four float4, K order)
void nv_weights(uint8_t* codes, uint8_t* scales, int64_t at, half gm, out float4 wv[4]) {
	let q = loadAligned<8>((uint2*)codes + at);
	let s = half(e4m3_value(uint(scales[at]))) * gm;
	let s2 = half2(s, s);
	[ForceUnroll]
	for (int h = 0; h < 2; ++h) {
		let cw = h == 0 ? q.x : q.y;
		[ForceUnroll]
		for (int i = 0; i < 4; i += 2) {
			let n0 = (cw >> uint(4 * i)) & 0x000F000Fu;
			let n1 = (cw >> uint(4 * i + 4)) & 0x000F000Fu;
			let w0 = float2(bit_cast<half2>(((n0 & 0x00070007u) << 9) | ((n0 & 0x00080008u) << 12)) * s2);
			let w1 = float2(bit_cast<half2>(((n1 & 0x00070007u) << 9) | ((n1 & 0x00080008u) << 12)) * s2);
			wv[h * 2 + (i >> 1)] = float4(w0.x, w0.y, w1.x, w1.y);
		}
	}
}

// The group's rows [k0, k0 + ck) into s_xr: route r of the group reads row row_of(order[first + r]) of x (gate_up: the token, down: the route)
void nv_stage_routes(half* x, int K, int* order, int first, int cnt, int k0, int ck, int div, int t) {
	for (int i = t; i < cnt * (ck / 4); i += 256) {
		let r = i / (ck / 4);
		let c4 = i % (ck / 4);
		let row = order[first + r] / div;
		let w = loadAligned<8>((uint2*)(x + int64_t(row) * K + k0) + c4);
		let a = float2(bit_cast<half2>(w.x));
		let b = float2(bit_cast<half2>(w.y));
		s_xr[r * (NVR_CK / 4) + c4] = float4(a.x, a.y, b.x, b.y);
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void nvfp4_moe_gate_up_grp(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Nvfp4MoeGrpParams p) {
	let t = int(tid.x);
	let lane = t & 31;
	let warp = t >> 5;
	let tile = int(gid.x);
	let e = p.groups[3 * gid.y];
	let first = p.groups[3 * gid.y + 1];
	let cnt = p.groups[3 * gid.y + 2];
	let K = p.hidden;
	let G = K / 16;
	let off = int64_t(tile) * G * 32 + lane;
	let E = p.num_local;
	let gm_g = half(p.gs[e] * 16384.0f);
	let gm_u = half(p.gs[E + e] * 16384.0f);
	float ag[NVR];
	float au[NVR];
	[ForceUnroll]
	for (int r = 0; r < NVR; ++r) { ag[r] = 0.0f; au[r] = 0.0f; }
	for (int k0 = 0; k0 < K; k0 += NVR_CK) {
		let ck = min(NVR_CK, K - k0);
		nv_stage_routes(p.x, K, p.order, first, cnt, k0, ck, p.topk, t);
		GroupMemoryBarrierWithGroupSync();
		for (int g = k0 / 16 + warp; g < (k0 + ck) / 16; g += NV_WARPS) {
			float4 wg[4];
			float4 wu[4];
			nv_weights(p.tbl[0 * E + e], p.tbl[1 * E + e], off + int64_t(g) * 32, gm_g, wg);
			nv_weights(p.tbl[2 * E + e], p.tbl[3 * E + e], off + int64_t(g) * 32, gm_u, wu);
			let c4 = (g * 16 - k0) / 4;
			[ForceUnroll]
			for (int r = 0; r < NVR; ++r) {
				if (r < cnt) {
					float a = ag[r];
					float b = au[r];
					[ForceUnroll]
					for (int u = 0; u < 4; ++u) {
						let xv = s_xr[r * (NVR_CK / 4) + c4 + u];
						a = fma(xv.x, wg[u].x, fma(xv.y, wg[u].y, fma(xv.z, wg[u].z, fma(xv.w, wg[u].w, a))));
						b = fma(xv.x, wu[u].x, fma(xv.y, wu[u].y, fma(xv.z, wu[u].z, fma(xv.w, wu[u].w, b))));
					}
					ag[r] = a;
					au[r] = b;
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();   // the next chunk (or the partials) rewrite s_xr
	}
	// the warps' partials, (warp, lane) a float4 per pair of routes: [gate r, up r, gate r + 1, up r + 1]
	[ForceUnroll]
	for (int r = 0; r < NVR; r += 2) s_xr[(warp * 32 + lane) * (NVR / 2) + r / 2] = float4(ag[r], au[r], ag[r + 1], au[r + 1]);
	GroupMemoryBarrierWithGroupSync();
	if (t < 32 * cnt) {
		let r = t >> 5;
		let l = t & 31;
		float g = 0.0f, u = 0.0f;
		[ForceUnroll]
		for (int w = 0; w < NV_WARPS; ++w) {
			let v = s_xr[(w * 32 + l) * (NVR / 2) + r / 2];
			g += (r & 1) != 0 ? v.z : v.x;
			u += (r & 1) != 0 ? v.w : v.y;
		}
		let gh = r16(g);
		float up = r16(u);
		float act = gh / (1.0f + exp(-gh));
		if (p.act_limit > 0.0f) {
			up = min(max(up, -p.act_limit), p.act_limit);
			act = min(act, p.act_limit);
		}
		p.out_[int64_t(p.order[first + r]) * p.inter + tile * 32 + nv_lane_col(l)] = half(act * up);
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void nvfp4_moe_down_grp(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Nvfp4MoeGrpParams p) {
	let t = int(tid.x);
	let lane = t & 31;
	let warp = t >> 5;
	let tile = int(gid.x);
	let e = p.groups[3 * gid.y];
	let first = p.groups[3 * gid.y + 1];
	let cnt = p.groups[3 * gid.y + 2];
	let K = p.inter;
	let G = K / 16;
	let off = int64_t(tile) * G * 32 + lane;
	let E = p.num_local;
	let gm = half(p.gs[2 * E + e] * 16384.0f);
	float ad[NVR];
	[ForceUnroll]
	for (int r = 0; r < NVR; ++r) ad[r] = 0.0f;
	for (int k0 = 0; k0 < K; k0 += NVR_CK) {
		let ck = min(NVR_CK, K - k0);
		nv_stage_routes(p.x, K, p.order, first, cnt, k0, ck, 1, t);
		GroupMemoryBarrierWithGroupSync();
		for (int g = k0 / 16 + warp; g < (k0 + ck) / 16; g += NV_WARPS) {
			float4 wd[4];
			nv_weights(p.tbl[4 * E + e], p.tbl[5 * E + e], off + int64_t(g) * 32, gm, wd);
			let c4 = (g * 16 - k0) / 4;
			[ForceUnroll]
			for (int r = 0; r < NVR; ++r) {
				if (r < cnt) {
					float a = ad[r];
					[ForceUnroll]
					for (int u = 0; u < 4; ++u) {
						let xv = s_xr[r * (NVR_CK / 4) + c4 + u];
						a = fma(xv.x, wd[u].x, fma(xv.y, wd[u].y, fma(xv.z, wd[u].z, fma(xv.w, wd[u].w, a))));
					}
					ad[r] = a;
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
	[ForceUnroll]
	for (int r = 0; r < NVR; r += 4) s_xr[(warp * 32 + lane) * (NVR / 4) + r / 4] = float4(ad[r], ad[r + 1], ad[r + 2], ad[r + 3]);
	GroupMemoryBarrierWithGroupSync();
	if (t < 32 * cnt) {
		let r = t >> 5;
		let l = t & 31;
		float v = 0.0f;
		[ForceUnroll]
		for (int w = 0; w < NV_WARPS; ++w) {
			let q = s_xr[(w * 32 + l) * (NVR / 4) + r / 4];
			let c = r & 3;
			v += c == 0 ? q.x : c == 1 ? q.y : c == 2 ? q.z : q.w;
		}
		p.out_[int64_t(p.order[first + r]) * p.hidden + tile * 32 + nv_lane_col(l)] = half(v);
	}
}

// ---- the dense linear for a few rows (LinearNVFP4::forward up to 8 rows): a workgroup of 256 per 32-column tile, a lane per column, the eight
// warps over the K groups of each chunk of 1024 values staged for every row (FP32 in shared memory); a lane decodes its 16 weights of a group once
// (the experts' FP16 products) and multiplies them into every row, the warps' sums added in order

struct Nvfp4GemvParams { uint8_t* codes; uint8_t* scales; half* x; half* out_; int m; int k; int n; float gm; };

// ---- the GEMV of a decode or a verify block (up to NVG_ROWS rows, compiled for 1, 2, 4 and 8): a workgroup per 32-column tile, a lane per
// column, warp w over a contiguous NVG_WARPS-th of the K groups, NVG_BATCH groups at a time: their codes and scales loaded together before
// their math (the loads in flight are what reaches the memory's bandwidth), the warp's lanes converting the batch's activations of every row to
// FP32 once into the warp's slice of shared memory, which every lane then reads as broadcasts. Rows past m re-read the last row, so the loop has
// no branch on the row count; their sums are dropped. The same FP16 weights as nvfp4_gemm, FP32 sums, the warps' added in order. Shared memory
// is NVG_WARPS * NVG_ROWS * 640 bytes (40 KB at 8 x 8): keep it within the 48 KB a workgroup has here
#ifndef NVG_ROWS
#define NVG_ROWS 8
#endif
#ifndef NVG_WARPS
#define NVG_WARPS 8
#endif
static const int NVG_BATCH = 8;
groupshared float s_pg[NVG_WARPS * NVG_ROWS * 32];
groupshared float4 s_xw[NVG_WARPS * NVG_ROWS * 32];   // per warp and row: its batch's 128 K values

// A lane's 16 FP16 weights of a K group (FP16, at 2^14), as eight pairs in K order: q its 16 nibbles (value 2 i + j of word h at bits 4 i + 16 j)
void nvg_weights(uint2 q, uint sb, half gm, out float2 w[8]) {
	let s = half(e4m3_value(sb)) * gm;
	let s2 = half2(s, s);
	[ForceUnroll]
	for (int h = 0; h < 2; ++h) {
		let cw = h == 0 ? q.x : q.y;
		[ForceUnroll]
		for (int i = 0; i < 4; ++i) {
			let n = (cw >> uint(4 * i)) & 0x000F000Fu;
			w[h * 4 + i] = float2(bit_cast<half2>(((n & 0x00070007u) << 9) | ((n & 0x00080008u) << 12)) * s2);
		}
	}
}

// One K group (group u of the warp's staged batch) into the rows' sums
void nvg_step_staged(uint2 q, uint sb, half gm, int base, int u, inout float acc[NVG_ROWS]) {
	float2 w[8];
	nvg_weights(q, sb, gm, w);
	[ForceUnroll]
	for (int r = 0; r < NVG_ROWS; ++r) {
		float a = acc[r];
		[ForceUnroll]
		for (int j = 0; j < 4; ++j) {
			let xv = s_xw[base + r * 32 + u * 4 + j];
			a = fma(xv.x, w[2 * j].x, fma(xv.y, w[2 * j].y, a));
			a = fma(xv.z, w[2 * j + 1].x, fma(xv.w, w[2 * j + 1].y, a));
		}
		acc[r] = a;
	}
}

[shader("compute")]
[numthreads(NVG_WARPS * 32, 1, 1)]
void nvfp4_gemv(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Nvfp4GemvParams p) {
	let t = int(tid.x);
	let lane = t & 31;
	let warp = t >> 5;
	let tile = int(gid.x);
	let G = p.k / 16;
	let per = (G + NVG_WARPS - 1) / NVG_WARPS;
	let g0 = min(G, warp * per);
	let g1 = min(G, g0 + per);
	let off = int64_t(tile) * G * 32 + lane;
	let gm = half(p.gm);
	uint2* codes = (uint2*)p.codes;
	half* xr[NVG_ROWS];
	float acc[NVG_ROWS];
	[ForceUnroll]
	for (int r = 0; r < NVG_ROWS; ++r) {
		xr[r] = p.x + int64_t(min(r, p.m - 1)) * p.k;
		acc[r] = 0.0f;
	}
	let base = warp * NVG_ROWS * 32;
	for (int g = g0; g < g1; g += NVG_BATCH) {
		let nb = min(NVG_BATCH, g1 - g);
		uint2 q[NVG_BATCH];
		uint sb[NVG_BATCH];
		[ForceUnroll]
		for (int u = 0; u < NVG_BATCH; ++u) {
			let gg = int64_t(min(g + u, g1 - 1));
			q[u] = loadAligned<8>(codes + off + gg * 32);
			sb[u] = uint(p.scales[off + gg * 32]);
		}
		// the batch's K values 4 lane .. 4 lane + 3 of every row (zero past the batch's groups)
		[ForceUnroll]
		for (int r = 0; r < NVG_ROWS; ++r) {
			float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
			if (lane * 4 < nb * 16) {
				let w = loadAligned<8>((uint2*)(xr[r] + int64_t(g) * 16 + lane * 4));
				let a = float2(bit_cast<half2>(w.x));
				let b = float2(bit_cast<half2>(w.y));
				v = float4(a.x, a.y, b.x, b.y);
			}
			s_xw[base + r * 32 + lane] = v;
		}
		GroupMemoryBarrierWithWaveSync();
		[ForceUnroll]
		for (int u = 0; u < NVG_BATCH; ++u) {
			if (u < nb) nvg_step_staged(q[u], sb[u], gm, base, u, acc);
		}
		GroupMemoryBarrierWithWaveSync();   // the next batch rewrites the slice
	}
	[ForceUnroll]
	for (int r = 0; r < NVG_ROWS; ++r) s_pg[(warp * NVG_ROWS + r) * 32 + lane] = acc[r];
	GroupMemoryBarrierWithGroupSync();
	for (int e = t; e < NVG_ROWS * 32; e += NVG_WARPS * 32) {
		let r = e >> 5;
		let l = e & 31;
		if (r < p.m) {
			float v = 0.0f;
			[ForceUnroll]
			for (int w = 0; w < NVG_WARPS; ++w) v += s_pg[(w * NVG_ROWS + r) * 32 + l];
			p.out_[int64_t(r) * p.n + tile * 32 + nv_lane_col(l)] = half(v);
		}
	}
}

// ---- the weight decoded to FP16 W^T [n, k] for the FP16 matmul (more rows): a workgroup per (32-column tile, NVD_GROUPS groups), a thread
// per (group, lane) decoding into shared memory, then each row's 16 NVD_GROUPS weights stored as one 128-byte run (a lane's 32-byte piece per
// row scatters the stores over 32 rows, at about half the write bandwidth). K is a multiple of 64: whole runs.

static const int NVD_GROUPS = 4;
groupshared uint4 s_dq[32][NVD_GROUPS * 2 + 1];

[shader("compute")]
[numthreads(128, 1, 1)]
void nvfp4_dequant(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Nvfp4GemvParams p) {
	let G = p.k / 16;
	let runs = G / NVD_GROUPS;
	let b = int(gid.x + gid.y * 65535u);
	if (b >= (p.n / 32) * runs) return;
	let tile = b / runs;
	let g0 = (b - tile * runs) * NVD_GROUPS;
	let t = int(tid.x);
	let lane = t & 31;
	let i = (int64_t(tile) * G + g0 + (t >> 5)) * 32 + lane;
	let gm = half(p.gm);
	let q = loadAligned<8>((uint2*)p.codes + i);
	let s = half(e4m3_value(uint(p.scales[i]))) * gm;
	let s2 = half2(s, s);
	uint wv[8];
	[ForceUnroll]
	for (int h = 0; h < 2; ++h) {
		let cw = h == 0 ? q.x : q.y;
		[ForceUnroll]
		for (int j = 0; j < 4; ++j) {
			// values 2 j and 2 j + 1 of word h: nibbles at bits 4 j and 16 + 4 j
			let n = (cw >> uint(4 * j)) & 0x000F000Fu;
			wv[h * 4 + j] = bit_cast<uint>(bit_cast<half2>(((n & 0x00070007u) << 9) | ((n & 0x00080008u) << 12)) * s2);
		}
	}
	let r = nv_lane_col(lane);
	s_dq[r][(t >> 5) * 2] = uint4(wv[0], wv[1], wv[2], wv[3]);
	s_dq[r][(t >> 5) * 2 + 1] = uint4(wv[4], wv[5], wv[6], wv[7]);
	GroupMemoryBarrierWithGroupSync();
	[ForceUnroll]
	for (int u = 0; u < 2; ++u) {
		let e = t + u * NVD_GROUPS * 32;
		let row = e / (NVD_GROUPS * 2);
		let c = e - row * (NVD_GROUPS * 2);
		storeAligned<16>((uint4*)(p.out_ + int64_t(tile * 32 + row) * p.k + g0 * 16) + c, s_dq[row][c]);
	}
}
